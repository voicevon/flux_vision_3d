"""
相机实时取流器 (Camera Streamer)
================================
提供非阻塞、轻量级相机取流与抓拍辅助：
- 自动检测并优先连接 Intel RealSense D435 物理硬件 (1080P/720P)
- 未插相机或硬件异常时自动优雅切入高保真 AprilTag 仿真视频流 (--mock 模式)
- 保持低延迟、稳定帧率，适配 GUI 实时主循环与连拍归档
"""

import os
import sys
import time
import math
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False


class CameraStreamer:
    """统一相机取流服务，支持硬件真实流与数学仿真流无缝切换"""

    def __init__(self, force_mock: bool = False, req_width: int = 1280, req_height: int = 720, req_fps: int = 30):
        self.force_mock = force_mock
        self.req_width = req_width
        self.req_height = req_height
        self.req_fps = req_fps

        self.pipeline = None
        self.is_running = False
        self.is_mock = force_mock
        self.stream_desc = "未初始化"
        self.last_valid_frame = None
        self.frame_count = 0
        
        # FPS 统计
        self.fps = 0.0
        self._fps_last_time = time.time()
        self._fps_frame_count = 0

        # AprilTag 字典用于 mock 生成
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

    def start(self) -> bool:
        """启动相机流"""
        if self.is_running:
            return True

        if not self.force_mock and HAVE_REALSENSE:
            try:
                ctx = rs.context()
                devices = list(ctx.query_devices())
                if len(devices) > 0:
                    dev = devices[0]
                    dev_name = dev.get_info(rs.camera_info.name)
                    self.pipeline = rs.pipeline()
                    config = rs.config()

                    # 尝试 1280x720 30fps
                    started = False
                    try:
                        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                        self.pipeline.start(config)
                        self.stream_desc = f"{dev_name} (1280x720 @ 30fps)"
                        started = True
                    except Exception:
                        pass

                    if not started:
                        try:
                            config = rs.config()
                            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                            self.pipeline.start(config)
                            self.stream_desc = f"{dev_name} (640x480 @ 30fps)"
                            started = True
                        except Exception:
                            pass

                    if started:
                        # 丢弃前 3 帧等待曝光稳定
                        for _ in range(3):
                            self.pipeline.wait_for_frames(timeout_ms=1000)
                        self.is_running = True
                        self.is_mock = False
                        return True
            except Exception as e:
                print(f"[CameraStreamer] 连接物理相机失败: {e}，切至仿真模式")

        # 优雅回退至 Mock 模式
        self.is_mock = True
        self.is_running = True
        self.stream_desc = "仿真模拟相机 (--mock 模式)"
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        """非阻塞或微等待读取一帧 (BGR)"""
        if not self.is_running:
            return False, None

        self.frame_count += 1
        now = time.time()
        self._fps_frame_count += 1
        if now - self._fps_last_time >= 0.5:
            self.fps = self._fps_frame_count / (now - self._fps_last_time)
            self._fps_frame_count = 0
            self._fps_last_time = now

        if not self.is_mock and self.pipeline is not None:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                color_frame = frames.get_color_frame()
                if color_frame:
                    bgr = np.asanyarray(color_frame.get_data())
                    self.last_valid_frame = bgr
                    return True, bgr
            except Exception:
                if self.last_valid_frame is not None:
                    return True, self.last_valid_frame

        # 仿真帧
        mock_frame = self._generate_mock_frame(self.frame_count)
        self.last_valid_frame = mock_frame
        return True, mock_frame

    def _generate_mock_frame(self, frame_idx: int) -> np.ndarray:
        """生成带动态运动与 AprilTag 的仿真视频帧"""
        h, w = 720, 1280
        frame = np.full((h, w, 3), 32, dtype=np.uint8)

        # 柔和网格工作台
        for x in range(0, w, 80):
            cv2.line(frame, (x, 0), (x, h), (44, 44, 44), 1)
        for y in range(0, h, 80):
            cv2.line(frame, (0, y), (w, y), (44, 44, 44), 1)

        # 动态小扰动模拟手持相机微动
        dx = int(12 * math.sin(frame_idx * 0.08))
        dy = int(8 * math.cos(frame_idx * 0.06))

        tag_configs = [
            (0, 320 + dx, 220 + dy, 110),
            (1, 640 + dx, 200 - dy, 110),
            (2, 960 - dx, 220 + dy, 110),
            (18, 400 - dx, 480 + dy, 100),
            (28, 880 + dx, 480 - dy, 100),
            (4, 640 + dx, 500 + dy, 90),
        ]

        for tag_id, cx, cy, sz in tag_configs:
            hs = sz // 2
            x1, y1 = max(0, cx - hs), max(0, cy - hs)
            x2, y2 = min(w, cx + hs), min(h, cy + hs)
            tag_img = cv2.aruco.generateImageMarker(self.dictionary, tag_id, sz)
            tag_bgr = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
            h_sub, w_sub = y2 - y1, x2 - x1
            if h_sub > 0 and w_sub > 0:
                frame[y1:y2, x1:x2] = tag_bgr[:h_sub, :w_sub]

        return frame

    def stop(self):
        """停止取流并释放相机资源"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.is_running = False
