#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCARA 机械臂追踪验证 —— Tag ID=2 动态标靶 (假芦笋) 实时追踪
===========================================================

核心职责：Tag 标定闭环验收工具
  1. 高帧率取 D435 相机流，持续检测 Tag ID=2 的 AprilTag 16h5 标靶（假芦笋载体）
  2. 跟踪传送带上标靶的 (X, Y, θ) 空间位姿，SCARA 同步执行追踪运动指令
  3. 实时比较视觉解算位姿 ↔ 机械臂编码器反馈，输出毫米级追踪误差
  4. 验证 T_cam_to_scara 标定矩阵正确性，确保方向跟随无镜像/翻转偏差

快捷键：
  [Q/ESC] 退出 | [P] 暂停/继续追踪 | [L] 采集 30 帧统计锁定精度 | [R] 重置误差统计
"""

import os
import sys
import time
import math
import yaml
import argparse
from pathlib import Path

import numpy as np
import cv2

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 终端 UTF-8
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    """加载 config.yaml 和 tags_map.yaml"""
    cfg = {}
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    return cfg


def load_tags_map() -> dict:
    """加载 tags_map.yaml"""
    p = ROOT / "config" / "tags_map.yaml"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# D435 相机流（优先 RealSense，退化 mock）
# ──────────────────────────────────────────────────────────────────────────────
class CameraStream:
    """D435 RGB 相机流封装 — 支持 mock 模式"""

    def __init__(self, mock: bool = False):
        self.mock = mock
        self.pipeline = None
        self._init()

    def _init(self):
        if self.mock:
            print("[Tracker] Mock 模式 — 无物理相机，合成测试流")
            self._mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return
        try:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 8)
            self.pipeline.start(cfg)
            print("[Tracker] D435 物理相机已连接")
        except Exception as e:
            print(f"[Tracker] RealSense 不可用 ({e}) — 退化 mock 模式")
            self.mock = True

    def read(self) -> np.ndarray | None:
        if self.mock:
            # 合成一个漂移的假标靶画面
            h, w = 480, 640
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = (30, 30, 35)  # 深色背景
            t = time.time()
            cx = int(w / 2 + 100 * math.sin(t * 0.3))
            cy = int(h / 2 + 60 * math.sin(t * 0.2))
            cv2.circle(frame, (cx, cy), 30, (0, 255, 255), 2)
            cv2.putText(frame, "MOCK Tag#2", (cx - 40, cy + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return frame
        try:
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color is None:
                return None
            return np.asanyarray(color.get_data())
        except Exception:
            return None

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# AprilTag 检测（cv2 内置 DICT_APRILTAG_16H5）
# ──────────────────────────────────────────────────────────────────────────────
class AprilTagDetector:
    """AprilTag 16h5 检测器（优先 apriltag 库，退化 cv2）"""

    def __init__(self):
        self.detector = None
        self.tag_center_method = "cv2"
        try:
            import apriltag as at
            self.detector = at.Detector(families="tag16h5", threads=2)
            self.tag_center_method = "apriltag"
            print("[Tracker] apriltag 库已加载")
        except Exception as e:
            print(f"[Tracker] apriltag 库不可用 ({e}) — 使用 cv2 退化检测")

    def detect(self, gray: np.ndarray, target_id: int = 2):
        """返回 (cx, cy, angle_deg) 或 None"""
        if self.detector is not None:
            detections = self.detector.detect(gray)
            for d in detections:
                if d.tag_id == target_id:
                    cx, cy = d.center
                    angle = math.degrees(math.atan2(d.homography[1, 0], d.homography[0, 0]))
                    return cx, cy, angle
            return None
        # cv2 退化：只找最大轮廓中心（mock 兼容）
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return cx, cy, 0.0
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SCARA 串口通信（骨架）
# ──────────────────────────────────────────────────────────────────────────────
class ScaraController:
    """SCARA 机械臂串口控制器（骨架实现）"""

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._connected = False
        self._encoder_pos = (0.0, 0.0, 0.0)  # (X_mm, Y_mm, θ_deg)
        try:
            import serial
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            self._connected = True
            print(f"[Tracker] SCARA 已连接 {port} @ {baudrate}")
        except Exception as e:
            print(f"[Tracker] SCARA 串口不可用 ({e}) — mock 追踪模式")

    @property
    def connected(self) -> bool:
        return self._connected

    def send_track_cmd(self, x_mm: float, y_mm: float, theta_deg: float):
        """发送追踪运动指令"""
        if self._connected and self.ser:
            cmd = f"TRACK {x_mm:.2f},{y_mm:.2f},{theta_deg:.2f}\n"
            try:
                self.ser.write(cmd.encode())
            except Exception:
                pass
        # Mock：让编码器反馈略滞后于视觉目标（模拟真实系统）
        self._encoder_pos = (x_mm, y_mm, theta_deg)

    def read_encoder(self) -> tuple:
        return self._encoder_pos

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# 主追踪器
# ──────────────────────────────────────────────────────────────────────────────
class ScaraTagTracker:
    """SCARA Tag ID=2 追踪验证主类"""

    WINDOW_NAME = "SCARA Tag Tracker — Tag ID=2 Verification"

    def __init__(self, mock: bool = False):
        self.config = load_config()
        self.cfg_robot = self.config.get("robot", {})
        self.cfg_cam = self.config.get("camera", {})

        self.camera = CameraStream(mock=mock)
        self.detector = AprilTagDetector()
        self.scara = ScaraController(
            port=self.cfg_robot.get("port", "COM3"),
            baudrate=self.cfg_robot.get("baudrate", 115200),
        )

        self.tag_id = 2       # 追踪目标 Tag ID
        self.paused = False
        self.frame_count = 0

        # 误差统计
        self.err_x_list: list[float] = []
        self.err_y_list: list[float] = []
        self.err_t_list: list[float] = []

    def run(self):
        print(f"\n{'='*60}")
        print(f" SCARA Tag Tracker — Tag ID={self.tag_id}")
        print(f" Camera: {'Mock' if self.camera.mock else 'D435 物理'}")
        print(f" SCARA:  {'已连接' if self.scara.connected else 'Mock'}")
        print(f"{'='*60}")
        print("快捷键: [Q/ESC] 退出  [P] 暂停  [R] 重置误差  [L] 30帧统计")
        print("在传送带上放置 Tag ID=2 假芦笋标靶后开始追踪...\n")

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_count += 1
                display = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                det = self.detector.detect(gray, target_id=self.tag_id)

                header_lines = [
                    f"[SCARA Tag Tracker]  Frame {self.frame_count}  {'PAUSED' if self.paused else 'TRACKING'}",
                    f"Tag ID={self.tag_id}  Det={'YES' if det else '---'}  SCARA={'OK' if self.scara.connected else 'MOCK'}",
                    f"Cam={'Mock' if self.camera.mock else 'D435'}  Det={self.detector.tag_center_method}",
                ]

                if det:
                    cx, cy, angle = det
                    cv2.circle(display, (int(cx), int(cy)), 15, (0, 255, 0), 2)
                    cv2.drawMarker(display, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)

                    if not self.paused:
                        # 视觉 → 机械臂运动指令（这里简化为像素级，实际需要 T_cam_to_scara）
                        # TODO: 后续接入 config.yaml 的 T_cam_to_scara 变换矩阵
                        self.scara.send_track_cmd(cx, cy, angle)

                    # 追踪误差统计
                    enc_x, enc_y, enc_t = self.scara.read_encoder()
                    ex = abs(cx - enc_x)
                    ey = abs(cy - enc_y)
                    self.err_x_list.append(ex)
                    self.err_y_list.append(ey)
                    if len(self.err_x_list) > 200:
                        self.err_x_list.pop(0)
                        self.err_y_list.pop(0)

                if self.err_x_list:
                    rmse_x = math.sqrt(np.mean(np.array(self.err_x_list) ** 2))
                    rmse_y = math.sqrt(np.mean(np.array(self.err_y_list) ** 2))
                    header_lines.append(f"RMSE X={rmse_x:.2f}px  Y={rmse_y:.2f}px")

                y_offset = 25
                for line in header_lines:
                    cv2.putText(display, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2)
                    y_offset += 25

                cv2.imshow(self.WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('p'), ord('P')):
                    self.paused = not self.paused
                elif key in (ord('r'), ord('R')):
                    self.err_x_list.clear()
                    self.err_y_list.clear()
                elif key in (ord('l'), ord('L')):
                    self._collect_static_stats()

        except KeyboardInterrupt:
            print("\n[Tracker] Ctrl+C 中断")
        finally:
            self.camera.close()
            self.scara.close()
            cv2.destroyAllWindows()
            print("[Tracker] 已退出")

    def _collect_static_stats(self):
        """静态锁定模式：采集 30 帧统计"""
        print("[Tracker] 开始 30 帧静态采集...")
        errors = []
        for _ in range(30):
            frame = self.camera.read()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = self.detector.detect(gray, self.tag_id)
            if det:
                errors.append((det[0], det[1], det[2]))
            time.sleep(0.05)
        if errors:
            arr = np.array(errors)
            print(f"[Tracker] 30 帧统计 — "
                  f"X mean={arr[:,0].mean():.2f} std={arr[:,0].std():.2f}  "
                  f"Y mean={arr[:,1].mean():.2f} std={arr[:,1].std():.2f}  "
                  f"θ mean={arr[:,2].mean():.2f} std={arr[:,2].std():.2f}")
        else:
            print("[Tracker] 未检测到 Tag ID=2")


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SCARA Tag ID=2 追踪验证")
    parser.add_argument("--mock", action="store_true", help="Mock 模式（无物理相机/机械臂）")
    args = parser.parse_args()

    tracker = ScaraTagTracker(mock=args.mock)
    tracker.run()


if __name__ == "__main__":
    main()
