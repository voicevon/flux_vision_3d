#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot 在线跟踪 - 相机硬件控制器:
类型/分辨率状态机 + RealSense/USB 生命周期 + 内参按分辨率刷新 + 帧采集。
任务互斥、Toast 提示、世界系联动等业务编排仍在主控制器 RobotOnlineTracker。
"""

import os

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


class CameraController:
    """相机硬件管理: 只负责取流启停 / 内参刷新 / 帧读取, 不含业务编排"""

    def __init__(self, engine):
        self.engine = engine            # 引用主控制器的几何引擎, 内参按实际分辨率刷新
        self.camera_type = "realsense"
        self.camera_options = [
            ("realsense", "RealSense D435"),
            ("usb",       "USB 普通摄像头"),
        ]
        self.resolution = "1280x720"
        self.resolution_options = [
            ("1280x720",  "1280 × 720  (推荐)"),
            ("1920x1080", "1920 × 1080"),
            ("848x480",   "848 × 480"),
            ("640x480",   "640 × 480"),
        ]
        self.pipeline = None
        self.usb_capture = None
        self.pipeline_running = False
        _w, _h = self.resolution.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)

    # ------------------------------ 状态切换 ------------------------------
    def set_resolution_key(self, res_key):
        """更新分辨率选择 (仅状态与帧尺寸, 不启停硬件)"""
        self.resolution = res_key
        _w, _h = res_key.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)

    # ------------------------------ 取流启停 ------------------------------
    def start(self):
        """按当前类型/分辨率启动取流并刷新引擎内参, 失败抛异常 (pipeline_running 不变)"""
        w, h = self.frame_w, self.frame_h
        if self.camera_type == "realsense":
            self._start_realsense(w, h)
        else:
            self._start_usb(w, h)
        self.pipeline_running = True

    def stop(self):
        """幂等关闭取流"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        if self.usb_capture is not None:
            try:
                self.usb_capture.release()
            except Exception:
                pass
            self.usb_capture = None
        self.pipeline_running = False

    def read_frame(self):
        """从当前后端读取一帧彩色图, 失败返回 None"""
        try:
            if self.camera_type == "realsense" and self.pipeline is not None:
                frames = self.pipeline.wait_for_frames(timeout_ms=4000)
                color = frames.get_color_frame()
                if not color:
                    return None
                return np.asanyarray(color.get_data())
            if self.usb_capture is not None:
                ok, frame = self.usb_capture.read()
                return frame if ok else None
        except Exception:
            return None
        return None

    # ------------------------------ 硬件后端 ------------------------------
    def _start_realsense(self, w, h):
        """启动 RealSense 彩色流 (逐级尝试 30/15/8 fps)，并按实际分辨率刷新引擎内参"""
        if rs is None:
            raise RuntimeError("pyrealsense2 未安装, 请先安装 RealSense SDK")
        ctx = rs.context()
        if not list(ctx.query_devices()):
            raise RuntimeError("未检测到 RealSense 设备, 请检查 USB 连接")
        fps = 30 if w <= 1280 else 8
        pipeline = rs.pipeline()
        cfg = rs.config()
        last_err = None
        for f in (fps, 15, 8):
            try:
                cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                pipeline.start(cfg)
                print(f"[OK] RealSense 彩色流: {w}x{h} @ {f}fps")
                break
            except Exception as e:
                last_err = e
                cfg = rs.config()
        else:
            raise RuntimeError(f"RealSense {w}x{h} 启动失败: {last_err}")
        self.pipeline = pipeline
        for _ in range(5):  # 预热
            pipeline.wait_for_frames(timeout_ms=2000)
        self._apply_intrinsics_realsense(w, h)

    def _apply_intrinsics_realsense(self, w, h):
        """RealSense: 使用 config.yaml 标定内参 (与建图一致) 并按实际分辨率自适应缩放"""
        if resolve_camera_intrinsics is not None:
            K, dist, meta = resolve_camera_intrinsics(
                CONFIG_PATH, actual_image_shape=(h, w))
            print(f"[OK] 相机内参: {meta.get('source')} | {w}x{h}")
        else:
            K = np.array([[1363.68, 0, 971.19], [0, 1361.19, 566.26], [0, 0, 1]], dtype=np.float64)
            dist = np.zeros((5, 1), dtype=np.float64)
        self.engine.camera_matrix = np.array(K, dtype=np.float64)
        self.engine.dist_coeffs = np.array(dist, dtype=np.float64)

    def _start_usb(self, w, h):
        """启动普通 USB 摄像头 (cv2.VideoCapture)，使用近似内参 (未标定)"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开 USB 摄像头 (index=0)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.usb_capture = cap
        # USB 相机无标定内参: 使用近似针孔模型 (世界坐标解算精度受限)
        K = np.array([
            [0.8 * max(w, h), 0.0, w / 2.0],
            [0.0, 0.8 * max(w, h), h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.engine.camera_matrix = K
        self.engine.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        print(f"[WARN] USB 摄像头已开启 {w}x{h} (未标定内参, 世界坐标仅供流程验证)")
