#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot 在线跟踪 (Robot Online Tracker) — FR-12
==============================================
Dashboard 第 5 张卡片「Robot 在线跟踪」的主工具：
  - GUI 先行启动 (不自动开相机)：顶部工具栏选相机类型 (RealSense D435 / USB 摄像头) →
    分辨率 → [开启] 乒乓开关 (布局与风格借鉴 d435_viewer 深度相机诊断工具)；
  - 真实相机实时取流，基于世界坐标地图 (FR-9.6, 地图世界系=机械臂坐标系) 实时检测标靶；
  - 用视野内非目标标靶的世界角点 PnP 解算相机世界系位姿，进而解出目标 Tag (默认 2 号) 的世界坐标实时显示；
  - 按 [T] 经机械臂串口 (FR-7.1) 以"抬起→平移→下探"安全路径驱动末端跟踪目标 Tag 世界坐标；
  - 到位后 M114 回读末端实际坐标，与视觉解算世界坐标同屏对比偏差 (FR-12.4 相机位置校准)。

工具栏: [相机类型 ▼] [分辨率 ▼] [开启/关闭] [连接机械臂] [跟踪] ... [退出 X]
快捷键: [C] 连接/断开机械臂  [T] 触发跟踪  [X]/[ESC] 退出
"""

import os
import sys
import json
import time
import argparse
import threading

import numpy as np
import cv2
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.calibration.offline_engine import OfflineVerificationEngine
from src.control.robot_serial import RobotSerial

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "robot_online_tracker"

# ============================ 视觉样式常量 (BGR, 与 d435_viewer 同源工业深色主题) ============================
COLOR_BG = (15, 17, 21)         # 工具栏 / 占位背景
COLOR_CARD_BG = (22, 26, 33)    # 按钮常态底色
COLOR_CARD_SEL = (28, 44, 58)   # 乒乓开关激活底色
COLOR_BORDER = (38, 46, 58)     # 常态描边
COLOR_BORDER_SEL = (0, 240, 200)  # 激活描边
COLOR_ACCENT = (0, 210, 180)    # 主题强调色
COLOR_TEXT_SUB = (155, 170, 185)
COL_GREEN = (110, 220, 90)      # 实测 / 目标标靶高亮
COL_YELLOW = (90, 200, 245)     # 理论值
COL_CYAN = (235, 205, 80)       # 偏差 / 支撑标靶
COL_RED = (85, 85, 245)         # 错误提示
COL_WHITE = (240, 240, 240)
COL_GRAY = (165, 165, 170)
COL_PANEL_BG = (26, 26, 30)
COL_PANEL_EDGE = (95, 95, 105)

TOOLBAR_H = 44  # 顶部工具栏高度

_FONT_CACHE = {}


def draw_text(img, text, pos, font_size=16, color=COL_WHITE, bold=False):
    """在 OpenCV BGR 图像上绘制中文/西文 (局部轻量 Patch 贴图)"""
    if not text:
        return
    if any(ord(c) > 127 for c in text):
        key = (font_size, bold)
        if key not in _FONT_CACHE:
            try:
                font_path = "C:/Windows/Fonts/msyh.ttc"
                if not os.path.exists(font_path):
                    font_path = "C:/Windows/Fonts/simhei.ttf"
                _FONT_CACHE[key] = ImageFont.truetype(font_path, font_size)
            except Exception:
                _FONT_CACHE[key] = ImageFont.load_default()
        font = _FONT_CACHE[key]
        x, y = pos
        ih, iw = img.shape[:2]
        if x >= iw or y >= ih or x < 0 or y < 0:
            return
        text_w = int(len(text) * font_size * 1.15) + 12
        text_h = int(font_size * 1.5) + 6
        x2 = min(iw, x + text_w)
        y2 = min(ih, y + text_h)
        if x2 <= x or y2 <= y:
            return
        patch_bgr = img[y:y2, x:x2]
        pil_img = Image.fromarray(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((0, 0), text, font=font, fill=(color[2], color[1], color[0]))
        img[y:y2, x:x2] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        scale = font_size / 28.0
        cv2.putText(img, text, (pos[0], pos[1] + int(font_size * 0.85)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)


def fmt_point(p, signed=False):
    """三维坐标一行式格式化 (一位小数, 右对齐, 无 X/Y/Z 前缀)"""
    if p is None:
        return "    --      --      --"
    if signed:
        return f"{p[0]:+8.1f} {p[1]:+8.1f} {p[2]:+8.1f}"
    return f"{p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}"


class RobotOnlineTracker:
    """Robot 在线跟踪控制器: GUI 先行 + Tag 世界坐标实时解算 + 机械臂联动跟踪 + 到位偏差对比"""

    def __init__(self, map_path=None, target_tag_id=2, port=None, baudrate=0):
        self.target_tag_id = int(target_tag_id)
        self.map_path = map_path or self._default_map_path()

        # 1. 世界坐标地图与计算引擎 (内参待相机开启后按实际分辨率刷新)
        self.engine = None
        self._load_engine()

        # 2. 机械臂串口控制器
        self.robot = RobotSerial(port=port or "", baudrate=baudrate)

        # 3. 工具栏状态 (借鉴 d435_viewer: 相机类型 → 分辨率 → 开关)
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
        self.active_dropdown = None     # "CAMERA_TYPE_DROPDOWN" | "RES_DROPDOWN" | None
        self.gui_buttons = []           # [(btn_id, (x1,y1,x2,y2), payload), ...] 每帧重建
        self.mouse_pos = (-1, -1)
        self._camera_type_rect = None
        self._resolution_rect = None

        # 4. 相机运行状态 (GUI 先行, 不自动开相机)
        self.pipeline = None
        self.usb_capture = None
        self.pipeline_running = False
        self.frame_w, self.frame_h = 1280, 720

        # 5. 跟踪运行状态
        self.measured = None        # 目标 Tag 世界坐标实测 (EMA 平滑)
        self.rmse = None            # 世界位姿 PnP 重投影 RMSE (px)
        self.support_ids = []       # 支撑世界位姿解算的标靶 ID
        self.theoretical = None     # 地图中目标 Tag 的理论世界坐标
        self.tracking = False       # 跟踪任务执行中
        self.track_stage = ""
        self.track_thread = None
        self.last_dev = None        # 最近一次到位偏差 (dx, dy, dz)
        self.robot_pos = None       # 最近一次 M114 末端坐标
        self.toast = "选择相机类型与分辨率后点击 [开启]"
        self.toast_err = False
        self.toast_time = time.time()
        self._quit_requested = False

        # 6. 持久化恢复工具栏状态
        self._load_viewer_state()

    # ------------------------------ 初始化 ------------------------------
    @staticmethod
    def _default_map_path() -> str:
        """默认地图: 当前活动场景地图, 回退全局 config/tags_map.yaml"""
        try:
            from src.calibration.scene_manager import CalibrationSceneManager
            return CalibrationSceneManager().get_active_scene().map_path
        except Exception:
            return os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")

    def _load_engine(self):
        """加载世界坐标地图并构建纯几何计算引擎 (相机内参以 config.yaml 默认值初始化)"""
        with open(self.map_path, "r", encoding="utf-8") as f:
            tags_map = yaml.safe_load(f) or {}
        marker_size = float(tags_map.get("marker_size_mm") or 40.0)

        if resolve_camera_intrinsics is not None:
            camera_matrix, dist_coeffs, _ = resolve_camera_intrinsics(CONFIG_PATH)
        else:
            camera_matrix = np.array([
                [1363.68, 0.0, 971.19],
                [0.0, 1361.19, 566.26],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.engine = OfflineVerificationEngine(
            tags_map=tags_map,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            marker_size_mm=marker_size
        )
        n_tags = len(tags_map.get("tags", {}))
        print(f"[OK] 世界坐标地图已加载: {self.map_path} | 标靶 {n_tags} 枚 | 边长 {marker_size:.1f}mm")
        T = self.engine.get_tag_world_transform(self.target_tag_id)
        self.theoretical = T[:3, 3].copy() if T is not None else None

    # ------------------------------ 工具栏状态持久化 ------------------------------
    def _load_viewer_state(self):
        """从 config/gui_settings.json 恢复相机类型与分辨率选择"""
        try:
            if not os.path.exists(GUI_SETTINGS_FILE):
                return
            with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            state = (root.get(APP_ID) or {}).get("viewer_state") or {}
            if state.get("camera_type") in ("realsense", "usb"):
                self.camera_type = state["camera_type"]
            if any(k == state.get("resolution") for k, _ in self.resolution_options):
                self.resolution = state["resolution"]
        except Exception:
            pass

    def _save_viewer_state(self):
        """保存相机类型与分辨率选择到 config/gui_settings.json"""
        try:
            root = {}
            if os.path.exists(GUI_SETTINGS_FILE):
                try:
                    with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                        root = json.load(f)
                    if not isinstance(root, dict):
                        root = {}
                except Exception:
                    root = {}
            node = root.setdefault(APP_ID, {})
            node["viewer_state"] = {
                "camera_type": self.camera_type,
                "resolution": self.resolution,
            }
            node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(GUI_SETTINGS_FILE), exist_ok=True)
            with open(GUI_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ------------------------------ 相机开关 (借鉴 d435_viewer) ------------------------------
    def _select_camera_type(self, cam_key):
        """切换相机类型：如果 pipeline 已运行则先停再切换"""
        if cam_key == self.camera_type:
            return
        if self.pipeline_running:
            self._toggle_camera(force_off=True)
        self.camera_type = cam_key
        self._save_viewer_state()
        print(f"[INFO] 相机类型已切换为: {dict(self.camera_options).get(cam_key, cam_key)}")

    def _change_resolution(self, res_key):
        """切换分辨率：运行中则先停再按新分辨率重启"""
        if res_key == self.resolution:
            return
        was_running = self.pipeline_running
        if was_running:
            self._toggle_camera(force_off=True)
        self.resolution = res_key
        self._save_viewer_state()
        if was_running:
            self._toggle_camera()
        print(f"[INFO] 分辨率已切换: {res_key}")

    def _toggle_camera(self, force_off=False):
        """开启或关闭相机取流"""
        if force_off or self.pipeline_running:
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
            self.set_toast("相机已关闭")
            print("[INFO] 相机已关闭")
        else:
            try:
                w_str, h_str = self.resolution.split("x")
                self.frame_w, self.frame_h = int(w_str), int(h_str)
                if self.camera_type == "realsense":
                    self._start_realsense(self.frame_w, self.frame_h)
                else:
                    self._start_usb(self.frame_w, self.frame_h)
                self.pipeline_running = True
                self.set_toast(f"相机已开启: {dict(self.camera_options).get(self.camera_type)} @ {self.resolution}")
            except Exception as e:
                print(f"[ERROR] 相机开启失败: {e}")
                self.set_toast(f"相机开启失败: {e}", True)
                self.pipeline_running = False

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

    def get_frame(self):
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

    # ------------------------------ 几何解算 ------------------------------
    def solve_frame(self, frame):
        """单帧解算: 检测标靶 -> 世界位姿 PnP -> 目标 Tag 世界坐标 (EMA 平滑)"""
        det = self.engine.detect_tags(frame)
        sol = {"support": [], "rmse": None, "target_world": None}

        obj_list, img_list, ids = [], [], []
        for tid, corners in det.items():
            if tid == self.target_tag_id:
                continue
            wc = self.engine.get_tag_world_corners(tid)
            if wc is None:
                continue
            obj_list.append(wc)
            img_list.append(corners.reshape((4, 2)))
            ids.append(tid)

        if obj_list:
            obj = np.vstack(obj_list).astype(np.float64)
            img = np.vstack(img_list).astype(np.float64)
            rvec, tvec, ok = self.engine.solve_pnp(obj, img)
            if ok:
                proj, _ = cv2.projectPoints(obj, rvec, tvec,
                                            self.engine.camera_matrix, self.engine.dist_coeffs)
                sol["rmse"] = float(np.mean(np.linalg.norm(
                    proj.reshape(-1, 2) - img, axis=1)))
                sol["support"] = ids

                c2 = det.get(self.target_tag_id)
                if c2 is not None:
                    ok2, _, t2 = self.engine.solve_single_tag_pnp(c2)
                    if ok2:
                        R, _ = cv2.Rodrigues(rvec)
                        p_cam = t2.reshape(3)                      # 目标 Tag 中心 (相机系)
                        sol["target_world"] = R.T @ (p_cam - tvec.reshape(3))  # -> 世界系

        # 更新显示状态 (实测坐标 EMA 平滑抑制抖动)
        self.support_ids = sol["support"]
        self.rmse = sol["rmse"]
        if sol["target_world"] is not None:
            p = sol["target_world"]
            self.measured = p if self.measured is None else 0.5 * self.measured + 0.5 * p
        return det

    # ------------------------------ 机械臂联动 ------------------------------
    def toggle_robot(self):
        """连接 / 断开机械臂串口"""
        if self.tracking:
            self.set_toast("跟踪任务执行中, 禁止断开", True)
            return
        if self.robot.is_connected:
            self.robot.close()
            self.robot_pos = None
            self.set_toast("机械臂串口已断开")
        else:
            self.set_toast(f"连接中: {self.robot.port} @ {self.robot.baudrate} ...")
            try:
                self.robot.connect()
                self.set_toast(f"机械臂已连接: {self.robot.port}")
            except Exception as e:
                self.set_toast(f"连接失败: {e}", True)

    def trigger_tracking(self):
        """触发一次"抬起→平移→下探"跟踪任务 (后台线程执行)"""
        if self.tracking:
            self.set_toast("跟踪任务执行中, 请稍候")
            return
        if not self.robot.is_connected:
            self.set_toast("机械臂未连接, 请先连接", True)
            return
        if self.measured is None:
            self.set_toast("尚无有效的目标世界坐标解算结果", True)
            return
        target = self.measured.copy()
        self.tracking = True
        self.track_thread = threading.Thread(
            target=self._track_worker, args=(target,), daemon=True)
        self.track_thread.start()

    def _track_worker(self, target):
        """跟踪线程: 三段式安全移动 -> M114 回读 -> 偏差计算"""
        try:
            self.track_stage = "移动中: 抬起 → 平移 → 下探"
            ok = self.robot.move_to(target[0], target[1], target[2])
            if not ok:
                self.set_toast("机械臂移动失败, 详见终端日志", True)
                return
            pos = self.robot.get_position()
            self.robot_pos = pos
            if pos is not None:
                dev = np.array(pos, dtype=np.float64) - target
                self.last_dev = dev
                self.set_toast(
                    f"到位完成 | 偏差 {dev[0]:+.1f} {dev[1]:+.1f} {dev[2]:+.1f} mm"
                    f" | 总 {np.linalg.norm(dev):.2f} mm")
            else:
                self.set_toast("移动完成, 但 M114 回读失败")
        finally:
            self.track_stage = ""
            self.tracking = False

    def set_toast(self, msg: str, is_err: bool = False):
        self.toast = msg
        self.toast_err = is_err
        self.toast_time = time.time()

    # ------------------------------ 界面渲染 ------------------------------
    def _draw_dropdown_button(self, canvas, rect, label, is_open):
        """扁平化下拉按钮 (借鉴 d435_viewer / studio_renderer)"""
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)
        if is_open:
            bg_col, border_col, text_col = (48, 56, 72), COLOR_BORDER_SEL, COL_WHITE
            arrow = "▲"
        elif is_hover:
            bg_col, border_col, text_col = (36, 40, 52), (0, 180, 220), COL_WHITE
            arrow = "▼"
        else:
            bg_col, border_col, text_col = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB
            arrow = "▼"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)
        draw_text(canvas, f"{label} {arrow}", (x1 + 8, y1 + (y2 - y1) // 2 - 8), 14, text_col)

    def _render_dropdown_popup(self, canvas, rect, options, active_key, btn_prefix):
        """置顶悬浮下拉列表浮层 (借鉴 d435_viewer)"""
        rx1, ry1, rx2, ry2 = rect
        item_h = 30
        pop_w = max(rx2 - rx1, 210)
        pop_x1, pop_y1 = rx1, ry2 + 2
        pop_x2, pop_y2 = pop_x1 + pop_w, pop_y1 + len(options) * item_h + 6
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (30, 34, 42), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), COLOR_BORDER_SEL, 1)

        for i, (key, label) in enumerate(options):
            iy1 = pop_y1 + 3 + i * item_h
            iy2 = iy1 + item_h
            is_active = (key == active_key)
            if is_active:
                cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), COLOR_CARD_SEL, -1)
            draw_text(canvas, label, (pop_x1 + 10, iy1 + (item_h - 16) // 2 - 2), 15,
                      COLOR_ACCENT if is_active else COL_WHITE, bold=is_active)
            self.gui_buttons.append((f"{btn_prefix}{i}", (pop_x1 + 2, iy1, pop_x2 - 2, iy2), key))

    def _draw_toolbar(self, canvas):
        """顶部工具栏: 相机类型 ▼ | 分辨率 ▼ | 开启/关闭 | 连接机械臂 | 跟踪 | ... | 退出 X"""
        tw = canvas.shape[1]
        self.gui_buttons = []
        cv2.rectangle(canvas, (0, 0), (tw, TOOLBAR_H), COLOR_BG, -1)
        cv2.line(canvas, (0, TOOLBAR_H - 1), (tw, TOOLBAR_H - 1), COLOR_BORDER, 1)

        y1, btn_h = 6, TOOLBAR_H - 12
        y2 = y1 + btn_h
        gap = 6

        # 1. 相机类型下拉 (最左)
        cam_x1, cam_x2 = 8, 8 + 150
        cam_label = dict(self.camera_options).get(self.camera_type, self.camera_type)
        self._draw_dropdown_button(canvas, (cam_x1, y1, cam_x2, y2), cam_label,
                                   is_open=(self.active_dropdown == "CAMERA_TYPE_DROPDOWN"))
        self.gui_buttons.append(("TOGGLE_CAM_DD", (cam_x1, y1, cam_x2, y2), "CAMERA_TYPE_DROPDOWN"))
        self._camera_type_rect = (cam_x1, y1, cam_x2, y2)

        # 2. 分辨率下拉
        res_x1 = cam_x2 + gap
        res_x2 = res_x1 + 110
        self._draw_dropdown_button(canvas, (res_x1, y1, res_x2, y2), self.resolution,
                                   is_open=(self.active_dropdown == "RES_DROPDOWN"))
        self.gui_buttons.append(("TOGGLE_RES_DD", (res_x1, y1, res_x2, y2), "RES_DROPDOWN"))
        self._resolution_rect = (res_x1, y1, res_x2, y2)

        # 3. 开启/关闭乒乓按钮
        sw_x1 = res_x2 + gap
        sw_x2 = sw_x1 + 70
        if self.pipeline_running:
            sw_bg, sw_border, sw_txt, sw_label = (55, 45, 30), (255, 160, 40), (255, 200, 80), "关闭"
        else:
            sw_bg, sw_border, sw_txt, sw_label = (30, 50, 40), (0, 200, 120), (80, 230, 160), "开启"
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_bg, -1)
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_border, 1)
        draw_text(canvas, sw_label, (sw_x1 + 22, y1 + (btn_h - 16) // 2 - 1), 15, sw_txt, True)
        self.gui_buttons.append(("TOGGLE_CAMERA", (sw_x1, y1, sw_x2, y2), None))

        # 4. 机械臂按钮
        rb_x1 = sw_x2 + gap
        rb_x2 = rb_x1 + 110
        connected = self.robot.is_connected
        rb_bg = COLOR_CARD_SEL if connected else COLOR_CARD_BG
        rb_border = COLOR_BORDER_SEL if connected else COLOR_BORDER
        rb_txt = COLOR_ACCENT if connected else COLOR_TEXT_SUB
        cv2.rectangle(canvas, (rb_x1, y1), (rb_x2, y2), rb_bg, -1)
        cv2.rectangle(canvas, (rb_x1, y1), (rb_x2, y2), rb_border, 1)
        draw_text(canvas, "断开机械臂" if connected else "连接机械臂",
                  (rb_x1 + 8, y1 + (btn_h - 16) // 2 - 1), 14, rb_txt, connected)
        self.gui_buttons.append(("TOGGLE_ROBOT", (rb_x1, y1, rb_x2, y2), None))

        # 5. 跟踪按钮
        tk_x1 = rb_x2 + gap
        tk_x2 = tk_x1 + 90
        busy = self.tracking
        cv2.rectangle(canvas, (tk_x1, y1), (tk_x2, y2), COLOR_CARD_SEL if busy else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (tk_x1, y1), (tk_x2, y2), (255, 160, 40) if busy else COLOR_BORDER, 1)
        draw_text(canvas, "跟踪中..." if busy else f"跟踪 Tag {self.target_tag_id}",
                  (tk_x1 + 8, y1 + (btn_h - 16) // 2 - 1), 14,
                  (255, 200, 80) if busy else COL_GREEN, True)
        self.gui_buttons.append(("TRIGGER_TRACK", (tk_x1, y1, tk_x2, y2), None))

        # 6. 退出按钮 (最右)
        exit_x1, exit_x2 = tw - 90, tw - 8
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2), (60, 60, 80), 1)
        draw_text(canvas, "退出 X", (exit_x1 + 14, y1 + (btn_h - 16) // 2 - 1), 14,
                  (190, 190, 200), True)
        self.gui_buttons.append(("QUIT", (exit_x1, y1, exit_x2, y2), None))

        # 展开的下拉浮层
        if self.active_dropdown == "CAMERA_TYPE_DROPDOWN" and self._camera_type_rect:
            self._render_dropdown_popup(canvas, self._camera_type_rect,
                                        self.camera_options, self.camera_type, "DD_CAM_")
        elif self.active_dropdown == "RES_DROPDOWN" and self._resolution_rect:
            self._render_dropdown_popup(canvas, self._resolution_rect,
                                        self.resolution_options, self.resolution, "DD_RES_")

    def _draw_overlay(self, canvas, det):
        """标靶多边形叠加: 支撑标靶青色 / 目标 Tag 绿色高亮"""
        for tid, corners in det.items():
            pts = corners.reshape((-1, 2)).astype(np.int32)
            if tid == self.target_tag_id:
                cv2.polylines(canvas, [pts], True, COL_GREEN, 3, cv2.LINE_AA)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.drawMarker(canvas, (cx, cy), COL_GREEN, cv2.MARKER_CROSS, 18, 2)
                draw_text(canvas, f"Tag {tid}", (cx + 12, cy - 24), 17, COL_GREEN, True)
            else:
                cv2.polylines(canvas, [pts], True, COL_CYAN, 1, cv2.LINE_AA)

    def _draw_info_panel(self, canvas, y_off=0):
        """左上信息面板 (实测 / 理论 / 偏差 / 机械臂状态)"""
        x1, y1 = 14, y_off + 14
        x2, y2 = 478, y_off + 268
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_PANEL_EDGE, 1)

        y = y1 + 26
        draw_text(canvas, f"Robot 在线跟踪 | 目标 Tag {self.target_tag_id}",
                  (x1 + 14, y), 19, COL_WHITE, True)
        y += 28
        draw_text(canvas, f"世界系地图: {os.path.basename(self.map_path)}",
                  (x1 + 14, y), 15, COL_GRAY)
        y += 24
        if self.support_ids and self.rmse is not None:
            draw_text(canvas, f"支撑: {len(self.support_ids)} 靶 {self.support_ids}"
                              f" | RMSE {self.rmse:.2f}px", (x1 + 14, y), 15, COL_GRAY)
        else:
            draw_text(canvas, "支撑: 无已知标靶入镜, 世界位姿失效",
                      (x1 + 14, y), 15, COL_RED)
        y += 26
        draw_text(canvas, f"实测 {fmt_point(self.measured)}",
                  (x1 + 14, y), 17, COL_GREEN if self.measured is not None else COL_GRAY, True)
        y += 26
        theo_txt = f"理论 {fmt_point(self.theoretical)}" if self.theoretical is not None \
            else "理论      (地图中无该 Tag)"
        draw_text(canvas, theo_txt, (x1 + 14, y), 17, COL_YELLOW)
        y += 26
        if self.measured is not None and self.theoretical is not None:
            dev = self.measured - self.theoretical
            draw_text(canvas, f"偏差 {fmt_point(dev, signed=True)}",
                      (x1 + 14, y), 17, COL_CYAN)
        else:
            draw_text(canvas, "偏差          --", (x1 + 14, y), 17, COL_GRAY)
        y += 26
        if self.robot.is_connected:
            pos_txt = f"末端 {fmt_point(self.robot_pos)}" if self.robot_pos is not None \
                else "末端 --"
            draw_text(canvas, f"机械臂: {self.robot.port} | {pos_txt}",
                      (x1 + 14, y), 15, COL_GREEN)
        else:
            draw_text(canvas, "机械臂: 未连接 [C] 连接", (x1 + 14, y), 15, COL_GRAY)

    def _draw_toast(self, canvas):
        """右下角浮动通知"""
        h, w = canvas.shape[:2]
        if time.time() - self.toast_time < 6.0:
            col = COL_RED if self.toast_err else COL_WHITE
            tw = int(len(self.toast) * 16 * 1.05) + 16
            tx = max(w - tw - 20, 20)
            ty = h - 40
            overlay = canvas.copy()
            cv2.rectangle(overlay, (tx - 10, ty - 8), (w - 10, ty + 24), COL_PANEL_BG, -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
            draw_text(canvas, self.toast, (tx, ty), 16, col)

    def _compose_canvas(self, frame):
        """视频帧 + 顶部工具栏拼合 (工具栏独立于画面, 不遮挡视频内容)"""
        fh, fw = frame.shape[:2]
        tool_area = np.full((TOOLBAR_H, fw, 3), COLOR_BG, dtype=np.uint8)
        full = np.vstack((tool_area, frame))
        return full

    # ------------------------------ 鼠标交互 ------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            for btn_id, rect, payload in self.gui_buttons:
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._handle_action(btn_id, payload)
                    return
            # 点击空白处收起下拉
            if self.active_dropdown is not None:
                self.active_dropdown = None

    def _handle_action(self, btn_id, payload):
        """工具栏按钮动作分发"""
        if btn_id == "TOGGLE_CAM_DD":
            self.active_dropdown = None if self.active_dropdown == "CAMERA_TYPE_DROPDOWN" \
                else "CAMERA_TYPE_DROPDOWN"
        elif btn_id == "TOGGLE_RES_DD":
            self.active_dropdown = None if self.active_dropdown == "RES_DROPDOWN" \
                else "RES_DROPDOWN"
        elif btn_id.startswith("DD_CAM_"):
            self.active_dropdown = None
            self._select_camera_type(payload)
        elif btn_id.startswith("DD_RES_"):
            self.active_dropdown = None
            self._change_resolution(payload)
        elif btn_id == "TOGGLE_CAMERA":
            self.active_dropdown = None
            self._toggle_camera()
        elif btn_id == "TOGGLE_ROBOT":
            self.toggle_robot()
        elif btn_id == "TRIGGER_TRACK":
            self.trigger_tracking()
        elif btn_id == "QUIT":
            self._quit_requested = True

    # ------------------------------ 主循环 ------------------------------
    def run(self):
        """主事件循环 — GUI 先行启动, 相机等用户点击 [开启]"""
        win_name = "Robot 在线跟踪 | flux_vision_3d"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setWindowTitle(win_name, win_name)
        cv2.setMouseCallback(win_name, self._on_mouse)
        print("\n" + "=" * 68)
        print(" Robot 在线跟踪 (GUI 已启动, 相机未开启)")
        print("   顶部工具栏: 相机类型 → 分辨率 → [开启] → [连接机械臂] → [跟踪]")
        print("   快捷键: [C] 连接机械臂 | [T] 触发跟踪 | [X]/[ESC] 退出")
        print("=" * 68 + "\n")

        try:
            while not self._quit_requested:
                if self.pipeline_running:
                    frame = self.get_frame()
                    if frame is None:
                        canvas = np.full((self.frame_h + TOOLBAR_H, self.frame_w, 3),
                                         COLOR_BG, dtype=np.uint8)
                        draw_text(canvas, "取流中...", (self.frame_w // 2 - 60,
                                                       self.frame_h // 2), 22, COL_YELLOW, True)
                    else:
                        canvas = self._compose_canvas(frame)
                        det = self.solve_frame(frame)
                        self._draw_overlay(canvas, det)
                        self._draw_info_panel(canvas, y_off=TOOLBAR_H)
                else:
                    # 相机未开启: 占位画面
                    cw, ch = 1280, 676
                    canvas = np.full((ch, cw, 3), COLOR_BG, dtype=np.uint8)
                    draw_text(canvas, "相机未开启",
                              (cw // 2 - 120, ch // 2 - 50), 32, COLOR_ACCENT, True)
                    draw_text(canvas, "请先选择相机类型和分辨率，然后点击 [开启] 按钮",
                              (cw // 2 - 250, ch // 2 + 10), 18, COLOR_TEXT_SUB)

                self._draw_toolbar(canvas)
                self._draw_toast(canvas)
                cv2.imshow(win_name, canvas)

                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                key = cv2.waitKeyEx(30)
                if key == -1:
                    continue
                k = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""
                if k in ("x", "q") or key == 27:
                    break
                elif k == "c":
                    self.toggle_robot()
                elif k == "t":
                    self.trigger_tracking()
        finally:
            self.robot.close()
            self._toggle_camera(force_off=True)
            cv2.destroyAllWindows()
            print("[OK] Robot 在线跟踪已退出")


def main():
    ap = argparse.ArgumentParser(
        description="Robot 在线跟踪 — Tag 世界坐标实时解算与机械臂联动校准 (FR-12)")
    ap.add_argument("--map", default=None, help="世界坐标地图 yaml 路径 (默认当前活动场景地图)")
    ap.add_argument("--tag", type=int, default=2, help="跟踪目标 Tag ID (默认 2)")
    ap.add_argument("--port", default=None, help="机械臂串口 (默认读 config.yaml robot.port)")
    ap.add_argument("--baudrate", type=int, default=0, help="波特率 (默认读 config.yaml robot.baudrate)")
    args = ap.parse_args()

    try:
        app = RobotOnlineTracker(map_path=args.map, target_tag_id=args.tag,
                                 port=args.port, baudrate=args.baudrate)
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
