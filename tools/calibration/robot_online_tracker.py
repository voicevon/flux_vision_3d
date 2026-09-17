#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot 在线跟踪 (Robot Online Tracker) — FR-12
==============================================
Dashboard 第 5 张卡片「Robot 在线跟踪」的主工具：
  - GUI 先行启动 (不自动开相机)：顶部工具栏选相机类型 (RealSense D435 / USB 摄像头) →
    分辨率 → [开启] 乒乓开关 (布局与风格借鉴 d435_viewer 深度相机诊断工具)；
  - [识别] 一键单帧闭环 (与 Offline Studio 单帧流程一致)：开相机 → 拍一张 → 关相机 →
    识别视野内已知 Tag (蓝色实测棱柱) → 单帧 PnP 确定世界坐标系零点 →
    地图白名单全部 Tag 以绿色理论棱柱叠加在静态照片上；
  - 真实相机实时取流，基于世界坐标地图 (FR-9.6, 地图世界系=机械臂坐标系) 实时检测标靶；
  - 用视野内非目标标靶的世界角点 PnP 解算相机世界系位姿，进而解出目标 Tag (默认 2 号) 的世界坐标实时显示；
  - 按 [T] 经机械臂串口 (FR-7.1) 以"抬起→平移→下探"安全路径驱动末端跟踪目标 Tag 世界坐标；
  - 到位后 M114 回读末端实际坐标，与视觉解算世界坐标同屏对比偏差 (FR-12.4 相机位置校准)。

工具栏: [相机类型 ▼] [分辨率 ▼] [开启/关闭] [识别] [确定世界坐标系] [XY平面 ▼] [显示已知Tag] [识别 Tag 2] [连接机械臂] [跟踪] ... [退出 X]
快捷键: [S] 一键识别  [P] XY平面下拉  [L] 确定/解除世界坐标系  [A] 显示已知Tag  [R] 识别Tag2  [C] 连接/断开机械臂  [T] 触发跟踪  [X]/[ESC] 退出
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
COL_BLUE = (255, 170, 0)        # 单帧实测棱柱 (蓝, 与已知Tag叠加的实测色一致)
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


def _tag_local_frame(wc):
    """由标靶世界角点构造局部坐标系 (X右/Y上/Z=面法向朝镜头, 右手系) 与中心点。
    角点顺序: 左上, 右上, 右下, 左下; Z = X×Y 朝外 (与单靶 PnP 位姿同向)。"""
    x = wc[1] - wc[0]
    x = x / (np.linalg.norm(x) + 1e-9)
    y0 = wc[0] - wc[3]
    y0 = y0 / (np.linalg.norm(y0) + 1e-9)
    z = np.cross(x, y0)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    return np.column_stack([x, y, z]), wc.mean(axis=0)


PRISM_HW_MM = 15.0        # 棱柱截面半宽 mm (截面 30x30, 与 Offline Studio 一致)
PRISM_HEIGHT_MM = 75.0    # 棱柱生长高度 mm (与 Offline Studio 一致)
_PRISM_PTS = np.array([
    # 底面 4 点 (Z=0) / 顶面 4 点 (Z=生长高) / 顶面中心
    [-PRISM_HW_MM, -PRISM_HW_MM, 0.0], [PRISM_HW_MM, -PRISM_HW_MM, 0.0],
    [PRISM_HW_MM, PRISM_HW_MM, 0.0], [-PRISM_HW_MM, PRISM_HW_MM, 0.0],
    [-PRISM_HW_MM, -PRISM_HW_MM, PRISM_HEIGHT_MM], [PRISM_HW_MM, -PRISM_HW_MM, PRISM_HEIGHT_MM],
    [PRISM_HW_MM, PRISM_HW_MM, PRISM_HEIGHT_MM], [-PRISM_HW_MM, PRISM_HW_MM, PRISM_HEIGHT_MM],
    [0.0, 0.0, PRISM_HEIGHT_MM],
], dtype=np.float64)


class RobotOnlineTracker:
    """Robot 在线跟踪控制器: GUI 先行 + Tag 世界坐标实时解算 + 机械臂联动跟踪 + 到位偏差对比"""

    LOCK_MIN_SAMPLES = 5     # 确定世界坐标系最少采样帧数 (保证滤波统计意义)
    LOCK_MAX_SAMPLES = 30    # 采样帧数上限
    LOCK_CONVERGE_MM = 2.0   # 提前收敛阈值: 帧位姿与累计均值的平移偏差 (mm)
    LOCK_CONVERGE_DEG = 0.5  # 提前收敛阈值: 旋转偏差 (deg)
    LOCK_CONVERGE_STREAK = 2  # 连续满足收敛条件的次数
    PLANE_EXTENT_MM = 600     # XY 平面网格半宽 (mm)
    PLANE_STEP_MM = 100       # XY 平面网格间距 (mm)
    PLANE_Z_MM = 600          # Z 轴长度 (mm)
    PLANE_Z_CHOICES = (405, 196, 350, 300, 250, 200, 150, 100, 50, 0)  # 平面高度选项 (mm, 350 为新增档)
    PLANE_Z_LABELS = {405: " (Tag 0)", 196: " (Tag 1)", 0: " (地面)"}  # 特殊高度标注
    HP_TARGET_SIDE_PX = 240   # 高精度模式: ROI 放大后目标 Tag 边长 (px)

    def __init__(self, map_path=None, target_tag_id=2, port=None, baudrate=0):
        self.target_tag_id = int(target_tag_id)
        self.map_path = map_path or self._default_map_path()

        # 1. 世界坐标地图与计算引擎 (内参待相机开启后按实际分辨率刷新)
        self.engine = None
        self.theoretical = None     # 地图中目标 Tag 的理论世界坐标 (_load_engine 填充)
        self.anchor_positions = {}  # 锚定标靶 BA 理论世界中心 {tag_id: np.array(3)} (不含 Tag 2)
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
        self.active_dropdown = None     # "CAMERA_TYPE_DROPDOWN" | "RES_DROPDOWN" | "PLANE_DROPDOWN" | None
        self.gui_buttons = []           # [(btn_id, (x1,y1,x2,y2), payload), ...] 每帧重建
        self.mouse_pos = (-1, -1)
        self._camera_type_rect = None
        self._resolution_rect = None
        self._plane_z_rect = None
        self.plane_z = 0                # XY 平面绘制高度 (mm, 下拉框选择)
        self.plane_options = [
            (None, "不绘制 XY 平面")
        ] + [
            (z, f"Z {z} mm" + self.PLANE_Z_LABELS.get(z, ""))
            for z in self.PLANE_Z_CHOICES
        ]

        # 4. 相机运行状态 (GUI 先行, 不自动开相机)
        self.pipeline = None
        self.usb_capture = None
        self.pipeline_running = False
        _w, _h = self.resolution.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)
        self._last_canvas_size = None

        # 5. 识别与世界系锁定状态 (相机开启默认纯预览, FR-12.5/12.6)
        self.recog_tag2_on = False     # "识别 Tag 2" 乒乓开关 (默认关, 仅识别目标 Tag)
        self.show_anchors_on = False   # "显示已知 Tag" 乒乓开关 (绿=BA理论 / 蓝=实测)
        self.show_xy_plane_on = False  # "XY平面" 下拉选择状态 (False=不绘制, True=绘制 plane_z 高度平面)
        self.world_locked = False      # 世界坐标系锁定状态
        self.locked_rvec = None        # 锁定的相机世界位姿 (rvec)
        self.locked_tvec = None        # 锁定的相机世界位姿 (tvec)
        self.lock_info = ""            # 锁定信息 "30帧均值 | RMSE 0.31px"
        self.sampling = False          # 确定世界坐标系流程执行中
        self.sample_stage = ""
        self.recognizing = False       # "识别" 一键单帧闭环流程执行中
        self.recog_stage = ""          # 识别流程阶段提示 (按钮上显示)
        self.static_frame = None       # 识别保留的单帧照片 (相机已关闭, 静态显示)
        self.static_det = None         # 单帧照片的检测结果 {tag_id: corners(4,2)}

        # 6. 跟踪运行状态
        self.measured = None        # 目标 Tag 世界坐标实测 (EMA 平滑)
        self.rmse = None            # 世界位姿 PnP 重投影 RMSE (px)
        self.support_ids = []       # 支撑世界位姿解算的标靶 ID
        self.tracking = False       # 跟踪任务执行中
        self.track_stage = ""
        self.track_thread = None
        self.last_dev = None        # 最近一次到位偏差 (dx, dy, dz)
        self.robot_pos = None       # 最近一次 M114 末端坐标
        self.toast = "选择相机类型与分辨率后点击 [开启]"
        self.toast_err = False
        self.toast_time = time.time()
        self._quit_requested = False

        # 7. 持久化恢复工具栏状态
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

        # 锚定标靶 (地图白名单内, 排除移动的 Tag 2) 的 BA 理论世界中心
        self.anchor_positions = {}
        for tid_s in tags_map.get("tags", {}):
            tid_i = int(tid_s)
            if tid_i == self.target_tag_id:
                continue
            T_i = self.engine.get_tag_world_transform(tid_i)
            if T_i is not None:
                self.anchor_positions[tid_i] = T_i[:3, 3].copy()

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
        """切换分辨率：运行中则先停再按新分辨率重启 (任务执行中禁止)"""
        if res_key == self.resolution:
            return
        if self.sampling or self.tracking or self.recognizing:
            self.set_toast("任务执行中, 禁止切换分辨率", True)
            return
        was_running = self.pipeline_running
        if was_running:
            self._toggle_camera(force_off=True)
        self.resolution = res_key
        self._save_viewer_state()
        _w, _h = res_key.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)
        if was_running:
            self._toggle_camera()
        print(f"[INFO] 分辨率已切换: {res_key}")

    def _toggle_camera(self, force_off=False, _internal=False):
        """开启或关闭相机取流 (采样/跟踪/识别任务执行中禁止手动开关; 内部调用与强制关闭除外)"""
        if not force_off and not _internal and (self.sampling or self.tracking or self.recognizing):
            self.set_toast("任务执行中, 禁止切换相机状态", True)
            return
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
            self._release_world_lock(silent=True)
            self.recog_tag2_on = False
            self.set_toast("相机已关闭")
            print("[INFO] 相机已关闭")
        else:
            # 重新开启相机: 单帧识别静态结果让位于实时预览
            self.static_frame = None
            self.static_det = None
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

    def _release_world_lock(self, silent=False):
        """解除世界系锁定 (相机关闭/切换时位姿失效)"""
        if self.world_locked or self.lock_info:
            self.world_locked = False
            self.locked_rvec = None
            self.locked_tvec = None
            self.lock_info = ""
            if not silent:
                self.set_toast("世界坐标系锁定已解除, 可重新确定")

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
        """单帧解算 (相机开启默认纯预览, 仅在下列情况执行检测):
        - "识别 Tag 2" 开关打开: 每帧仅识别 Tag 2, 已锁定时解算其世界坐标 (FR-12.6)
        - "确定世界坐标系" 流程采样在后台线程独立执行, 不走此函数
        """
        if not (self.recog_tag2_on or self.show_anchors_on or self.show_xy_plane_on):
            self.support_ids = []
            self.rmse = None
            return None

        det = self.engine.detect_tags(frame)
        det = self._detect_high_precision(frame, det)  # 目标 Tag ROI 放大重检 (高精度)
        target_world = None
        c2 = det.get(self.target_tag_id)

        if c2 is not None:
            ok2, _, t2 = self.engine.solve_single_tag_pnp(c2)
            if ok2:
                if self.world_locked and self.locked_rvec is not None:
                    R_lock, _ = cv2.Rodrigues(self.locked_rvec)
                    p_cam = t2.reshape(3)
                    target_world = R_lock.T @ (p_cam - self.locked_tvec.reshape(3))
                    self.support_ids = ["锁定"]
                else:
                    self.support_ids = []

        # 更新显示状态 (实测坐标 EMA 平滑抑制抖动)
        if target_world is not None:
            p = target_world
            self.measured = p if self.measured is None else 0.5 * self.measured + 0.5 * p
        return det

    def _solve_per_frame(self, det):
        """确定世界坐标系流程用: 识别锚定标靶 (排除 Tag 2) -> 相机世界位姿 PnP"""
        sol = {"support": [], "rmse": None, "target_world": None,
               "rvec": None, "tvec": None}

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
                sol["rvec"] = rvec
                sol["tvec"] = tvec.reshape(3)
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
        return sol

    def _detect_high_precision(self, frame, det):
        """目标 Tag 高精度二次识别: 全景粗检定位 -> ROI 裁剪双三次放大 ->
        双路参数重检 -> 亚像素精修 -> 角点映射回原图坐标。
        放大后标靶边缘像素数倍增, 亚像素角点精度显著提升 (目标边长 HP_TARGET_SIDE_PX)。
        """
        c2 = det.get(self.target_tag_id)
        if c2 is None:
            return det
        det_bright = getattr(self.engine, "detector_bright", None)
        det_dark = getattr(self.engine, "detector_dark", None)
        if det_bright is None:
            return det
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h_f, w_f = gray.shape[:2]
        c = np.asarray(c2, dtype=np.float64).reshape(4, 2)
        side = float((np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[1] - c[2])) / 2.0)
        pad = max(side * 0.8, 30.0)                        # ROI 外扩 (含倾斜余量)
        x1 = int(max(0, c[:, 0].min() - pad))
        y1 = int(max(0, c[:, 1].min() - pad))
        x2 = int(min(w_f, c[:, 0].max() + pad))
        y2 = int(min(h_f, c[:, 1].max() + pad))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return det
        roi = gray[y1:y2, x1:x2]
        scale = float(np.clip(self.HP_TARGET_SIDE_PX / max(side, 1.0), 2.0, 8.0))
        scale = min(scale, 800.0 / max(roi.shape[0], roi.shape[1], 1))  # 限制放大图尺寸
        if scale < 1.01:
            return det
        roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        refined = None
        for d in (det_bright, det_dark):                   # 双路重检 (与全景检测同参数)
            corners, ids, _ = d.detectMarkers(roi_up)
            if ids is not None and len(ids) > 0:
                for i, tid in enumerate(ids.flatten()):
                    if int(tid) == self.target_tag_id:
                        refined = corners[i].reshape(4, 2)
                        break
            if refined is not None:
                break
        if refined is None:
            return det
        refine = getattr(self.engine, "refine_corners_subpix", None)
        if refine is not None:
            refined = refine(roi_up, refined)
        refined_orig = np.asarray(refined, dtype=np.float64) / scale + np.array([x1, y1])
        det = dict(det)
        det[self.target_tag_id] = refined_orig.reshape(4, 2)
        return det

    def _get_world_pose(self, det):
        """当前相机世界位姿 (世界->相机): 优先锁定位姿, 未锁定用当帧锚定 PnP"""
        if self.world_locked and self.locked_rvec is not None:
            return self.locked_rvec, self.locked_tvec
        if det is None:
            return None, None
        sol = self._solve_per_frame(det)
        if sol["rvec"] is not None:
            return sol["rvec"], np.asarray(sol["tvec"]).reshape(3, 1)
        return None, None

    def toggle_world_lock(self):
        """触发 / 解除"确定世界坐标系"一键流程 (FR-12.5)"""
        if self.world_locked:
            self._release_world_lock()
            return
        if self.sampling:
            self.set_toast("世界坐标系确定流程进行中, 请稍候")
            return
        if self.recognizing:
            self.set_toast("识别流程进行中, 请稍候")
            return
        if not self.pipeline_running:
            self.set_toast("相机未开启, 无法确定世界坐标系", True)
            return
        if self.tracking:
            self.set_toast("跟踪任务执行中, 禁止采样", True)
            return
        self.sampling = True
        threading.Thread(target=self._lock_worker, daemon=True).start()

    def _lock_worker(self):
        """确定世界坐标系线程 (动态采样): 识别锚定标靶 (排除 Tag 2) ->
        最少 5 帧 / 上限 30 帧, 当帧位姿与累计均值偏差 < 2mm 且旋转 < 0.5deg
        连续 2 次时提前收敛, 否则采满上限; 最终均值滤波锁定相机世界位姿
        """
        try:
            rot_mats, tvecs, rmses = [], [], []
            n = self.LOCK_MAX_SAMPLES
            stable_streak = 0
            converged_at = None
            for i in range(1, n + 1):
                self.sample_stage = f"采样中 {i}/{n}"
                frame = self.get_frame()
                if frame is None:
                    continue
                det = self.engine.detect_tags(frame)
                sol = self._solve_per_frame(det)
                if sol["rvec"] is None or sol["rmse"] is None:
                    continue
                R_i = cv2.Rodrigues(sol["rvec"])[0]
                t_i = np.asarray(sol["tvec"], dtype=np.float64).reshape(3)
                rot_mats.append(R_i)
                tvecs.append(t_i)
                rmses.append(sol["rmse"])

                # 动态收敛判定: 新帧与此前累计均值的偏差足够小且连续稳定
                if len(tvecs) >= self.LOCK_MIN_SAMPLES:
                    t_mean = np.mean(tvecs, axis=0)
                    R_mean = self._average_rotation(rot_mats)
                    d_mm = float(np.linalg.norm(t_i - t_mean))
                    d_deg = self._rotation_angle_deg(R_i, R_mean)
                    if d_mm < self.LOCK_CONVERGE_MM and d_deg < self.LOCK_CONVERGE_DEG:
                        stable_streak += 1
                        if stable_streak >= self.LOCK_CONVERGE_STREAK:
                            converged_at = len(tvecs)
                            break
                    else:
                        stable_streak = 0

            if len(rmses) < self.LOCK_MIN_SAMPLES:
                self.set_toast(f"有效采样不足 ({len(rmses)} < 最少{self.LOCK_MIN_SAMPLES}帧), 世界坐标系确定失败", True)
                return
            # 均值滤波: 平移取均值, 旋转 SVD 投影到 SO(3)
            self.locked_rvec = cv2.Rodrigues(self._average_rotation(rot_mats))[0]
            self.locked_tvec = np.mean(tvecs, axis=0).reshape(3, 1)
            self.world_locked = True
            tip = (f" | 第{converged_at}帧提前收敛(<{self.LOCK_CONVERGE_MM:g}mm)"
                   if converged_at else "")
            self.lock_info = f"{len(rmses)}帧均值 | RMSE {np.mean(rmses):.2f}px{tip}"
            self.recog_tag2_on = True  # 锁定后自动开启"识别 Tag 2"
            self.set_toast(f"世界坐标系零点已确定并锁定 ({self.lock_info}), "
                           f"已自动开启识别 Tag {self.target_tag_id}")
        finally:
            self.sampling = False
            self.sample_stage = ""

    # ------------------------------ 一键识别 (单帧闭环) ------------------------------
    def trigger_recognize(self):
        """一键识别: 开相机 → 拍一张 → 关相机 → 识别已知 Tag (蓝棱柱) →
        单帧 PnP 确定世界坐标系零点 → 地图白名单全部 Tag 绿色理论棱柱"""
        if self.recognizing:
            self.set_toast("识别流程进行中, 请稍候")
            return
        if self.sampling or self.tracking:
            self.set_toast("任务执行中, 请稍候", True)
            return
        self.recognizing = True
        threading.Thread(target=self._recognize_worker, daemon=True).start()

    def _recognize_worker(self):
        """识别线程: 完整"开→拍→关"闭环, 结束后相机保持关闭 (与 Offline Studio 单帧流程一致)"""
        try:
            # 1. 开相机 (若未开启; 内部调用不受任务互斥限制)
            if not self.pipeline_running:
                self.recog_stage = "开启相机..."
                self._toggle_camera(_internal=True)
                if not self.pipeline_running:
                    return
            # 2. 采集一帧 (连读数帧待曝光稳定, 取最后一帧)
            self.recog_stage = "采集照片..."
            frame = None
            for _ in range(8):
                f = self.get_frame()
                if f is not None:
                    frame = f
                time.sleep(0.05)
            # 3. 关相机 (拍完即关, 完整闭环)
            self.recog_stage = "关闭相机..."
            if self.pipeline_running:
                self._toggle_camera(force_off=True)
            if frame is None:
                self.set_toast("采集照片失败, 识别中止", True)
                return
            # 4. 识别视野内已知的 Tag (目标 Tag 追加 ROI 放大重检, 高精度)
            self.recog_stage = "检测 Tag... (高精度)"
            det = self._detect_high_precision(frame, self.engine.detect_tags(frame))
            self.static_frame = frame.copy()
            self.static_det = dict(det) if det else {}
            if not det:
                self.set_toast("识别完成: 未识别到任何 Tag", True)
                return
            # 5. 单帧 PnP 确定世界坐标系零点 (锚定标靶, 排除目标 Tag)
            sol = self._solve_per_frame(det)
            if sol["rvec"] is None:
                self.set_toast(f"识别到 {len(det)} 枚 Tag, 但无已知标靶入镜, 无法确定世界坐标系", True)
                return
            self.locked_rvec = sol["rvec"]
            self.locked_tvec = np.asarray(sol["tvec"], dtype=np.float64).reshape(3, 1)
            self.world_locked = True
            self.lock_info = f"单帧识别 | RMSE {sol['rmse']:.2f}px | 支撑 {sol['support']}"
            self.support_ids = sol["support"]
            self.rmse = sol["rmse"]
            if sol["target_world"] is not None:
                self.measured = np.asarray(sol["target_world"], dtype=np.float64)
            n_map = len(self.anchor_positions) + (1 if self.theoretical is not None else 0)
            self.set_toast(f"识别完成: {len(det)} 枚 Tag(蓝) | 世界坐标系已确定 | "
                           f"地图 {n_map} 枚理论 Tag(绿) | RMSE {sol['rmse']:.2f}px")
        finally:
            self.recognizing = False
            self.recog_stage = ""

    @staticmethod
    def _average_rotation(rot_mats):
        """旋转均值滤波: 矩阵均值后 SVD 投影到 SO(3)"""
        U, _, Vt = np.linalg.svd(np.mean(rot_mats, axis=0))
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        return R

    @staticmethod
    def _rotation_angle_deg(R_a, R_b):
        """两个旋转矩阵之间的夹角 (deg)"""
        R_d = R_a @ R_b.T
        return float(np.degrees(np.arccos(np.clip((np.trace(R_d) - 1.0) / 2.0, -1.0, 1.0))))

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
        """顶部工具栏: 相机类型 ▼ | 分辨率 ▼ | 开启/关闭 | 识别 | 确定世界坐标系 | XY平面 ▼ | 显示已知Tag | 识别 Tag 2 | 连接机械臂 | 跟踪 | 退出 X"""
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

        # 4. "识别" 一键单帧闭环按钮: 开相机→拍一张→关相机→识别Tag→定世界系→蓝/绿棱柱
        rc_x1 = sw_x2 + gap
        rc_x2 = rc_x1 + 76
        if self.recognizing:
            rc_label = self.recog_stage or "识别中..."
            rc_bg, rc_border, rc_txt = COLOR_CARD_BG, (255, 160, 40), (255, 200, 80)
        elif self.static_frame is not None:
            rc_label = "识别"
            rc_bg, rc_border, rc_txt = COLOR_CARD_SEL, COLOR_BORDER_SEL, COLOR_ACCENT
        else:
            rc_label = "识别"
            rc_bg, rc_border, rc_txt = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB
        cv2.rectangle(canvas, (rc_x1, y1), (rc_x2, y2), rc_bg, -1)
        cv2.rectangle(canvas, (rc_x1, y1), (rc_x2, y2), rc_border, 1)
        draw_text(canvas, rc_label,
                  (rc_x1 + (8 if self.recognizing else 20), y1 + (btn_h - 16) // 2 - 1),
                  14, rc_txt, True)
        self.gui_buttons.append(("TRIGGER_RECOG", (rc_x1, y1, rc_x2, y2), None))

        # 5. "确定世界坐标系" 一键流程按钮 (FR-12.5)
        lk_x1 = rc_x2 + gap
        lk_x2 = lk_x1 + 130
        if self.sampling:
            lk_label, lk_border, lk_txt = (self.sample_stage or "采样中..."), (255, 160, 40), (255, 200, 80)
        elif self.world_locked:
            lk_label, lk_border, lk_txt = "解除锁定", COLOR_BORDER_SEL, COLOR_ACCENT
        else:
            lk_label, lk_border, lk_txt = "确定世界坐标系", COLOR_BORDER, COLOR_TEXT_SUB
        cv2.rectangle(canvas, (lk_x1, y1), (lk_x2, y2), COLOR_CARD_SEL if self.world_locked else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (lk_x1, y1), (lk_x2, y2), lk_border, 1)
        draw_text(canvas, lk_label, (lk_x1 + 8, y1 + (btn_h - 16) // 2 - 1), 14, lk_txt, True)
        self.gui_buttons.append(("TOGGLE_LOCK", (lk_x1, y1, lk_x2, y2), None))

        # 6. "XY平面" 下拉框: 合并原[绘制XY平面]开关与[Z高度]下拉, 最高项"不绘制", 其余为绘制高度
        pl_x1 = lk_x2 + gap
        pl_x2 = pl_x1 + 110
        pl_label = f"Z={self.plane_z}" if self.show_xy_plane_on else "XY平面"
        self._draw_dropdown_button(canvas, (pl_x1, y1, pl_x2, y2), pl_label,
                                   is_open=(self.active_dropdown == "PLANE_DROPDOWN"))
        self.gui_buttons.append(("TOGGLE_PLANE_DD", (pl_x1, y1, pl_x2, y2), "PLANE_DROPDOWN"))
        self._plane_z_rect = (pl_x1, y1, pl_x2, y2)

        # 7. "显示已知 Tag" 乒乓开关 (绿色=BA理论位置 / 蓝色=当帧实测位置)
        an_x1 = pl_x2 + gap
        an_x2 = an_x1 + 120
        cv2.rectangle(canvas, (an_x1, y1), (an_x2, y2),
                      (30, 50, 40) if self.show_anchors_on else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (an_x1, y1), (an_x2, y2),
                      (0, 200, 120) if self.show_anchors_on else COLOR_BORDER, 1)
        draw_text(canvas, "显示已知Tag", (an_x1 + 12, y1 + (btn_h - 16) // 2 - 1), 14,
                  (80, 230, 160) if self.show_anchors_on else COLOR_TEXT_SUB, self.show_anchors_on)
        self.gui_buttons.append(("TOGGLE_ANCHORS", (an_x1, y1, an_x2, y2), None))

        # 8. "识别 Tag 2" 乒乓开关 (默认关, FR-12.6)
        rg_x1 = an_x2 + gap
        rg_x2 = rg_x1 + 110
        cv2.rectangle(canvas, (rg_x1, y1), (rg_x2, y2),
                      (30, 50, 40) if self.recog_tag2_on else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (rg_x1, y1), (rg_x2, y2),
                      (0, 200, 120) if self.recog_tag2_on else COLOR_BORDER, 1)
        draw_text(canvas, "识别 Tag 2", (rg_x1 + 14, y1 + (btn_h - 16) // 2 - 1), 14,
                  (80, 230, 160) if self.recog_tag2_on else COLOR_TEXT_SUB, self.recog_tag2_on)
        self.gui_buttons.append(("TOGGLE_RECOG", (rg_x1, y1, rg_x2, y2), None))

        # 9. 机械臂按钮
        rb_x1 = rg_x2 + gap
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

        # 10. 跟踪按钮
        tk_x1 = rb_x2 + gap
        tk_x2 = tk_x1 + 90
        busy = self.tracking
        cv2.rectangle(canvas, (tk_x1, y1), (tk_x2, y2), COLOR_CARD_SEL if busy else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (tk_x1, y1), (tk_x2, y2), (255, 160, 40) if busy else COLOR_BORDER, 1)
        draw_text(canvas, "跟踪中..." if busy else f"跟踪 Tag {self.target_tag_id}",
                  (tk_x1 + 8, y1 + (btn_h - 16) // 2 - 1), 14,
                  (255, 200, 80) if busy else COL_GREEN, True)
        self.gui_buttons.append(("TRIGGER_TRACK", (tk_x1, y1, tk_x2, y2), None))

        # 11. 退出按钮 (最右)
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
        elif self.active_dropdown == "PLANE_DROPDOWN" and self._plane_z_rect:
            self._render_dropdown_popup(canvas, self._plane_z_rect,
                                        self.plane_options,
                                        self.plane_z if self.show_xy_plane_on else None, "DD_PLANE_")

    def _draw_studio_prism(self, canvas, rvec, tvec, is_theory, is_target=False):
        """Offline Studio 同款四棱柱 (VerificationVisualizer 参数一致):
        截面 30x30mm (半宽15) x 生长高度 75mm, 半透明填充 + 棱线描边 + 顶面中心点;
        is_theory=True 翡翠绿(BA理论) / False 科技天蓝(实测);
        目标 Tag 额外绘制底面中心点与中心生长轴 (至固定高顶面中心)。
        """
        K, dist = self.engine.camera_matrix, self.engine.dist_coeffs
        proj = cv2.projectPoints(_PRISM_PTS, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(int)
        b, t, tc = proj[0:4], proj[4:8], tuple(proj[8])
        if is_theory:
            side_c, cap_c, edge_c, top_c, dot_c = \
                (0, 185, 60), (50, 240, 100), (0, 255, 100), (120, 255, 160), (0, 255, 120)
        else:
            side_c, cap_c, edge_c, top_c, dot_c = \
                (235, 125, 20), (255, 175, 50), (255, 195, 70), (255, 255, 255), (255, 200, 60)
        overlay = canvas.copy()
        for i in range(4):
            j = (i + 1) % 4
            cv2.fillPoly(overlay, [np.array([b[i], b[j], t[j], t[i]], dtype=np.int32)], side_c)
        cv2.fillPoly(overlay, [t], cap_c)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.polylines(canvas, [b], True, edge_c, 2, cv2.LINE_AA)
        cv2.polylines(canvas, [t], True, top_c, 2, cv2.LINE_AA)
        for i in range(4):
            cv2.line(canvas, tuple(b[i]), tuple(t[i]), edge_c, 2, cv2.LINE_AA)
        cv2.circle(canvas, tc, 4, dot_c, -1, cv2.LINE_AA)      # 生长固定高后的顶面中心点
        if is_target:                                          # 目标 Tag: 底面中心点 + 中心生长轴
            pb = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec, K, dist)[0]
            pb = tuple(pb.reshape(2).astype(int))
            cv2.circle(canvas, pb, 4, dot_c, -1, cv2.LINE_AA)
            cv2.line(canvas, pb, tc, dot_c, 1, cv2.LINE_AA)

    def _draw_overlay(self, canvas, det):
        """叠加层: 目标 Tag 绿色高亮框 + Studio 同款蓝色实测棱柱 (含底面/顶面中心点)"""
        corners = det.get(self.target_tag_id)
        if corners is None:
            return
        pts = corners.reshape((-1, 2)).astype(np.int32)
        cv2.polylines(canvas, [pts], True, COL_GREEN, 3, cv2.LINE_AA)
        if self.recog_tag2_on or self.show_anchors_on:
            ok_b, rvec_b, tvec_b = self.engine.solve_single_tag_pnp(corners)
            if ok_b:
                self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False,
                                        is_target=True)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.drawMarker(canvas, (cx, cy), COL_GREEN, cv2.MARKER_CROSS, 18, 2)
        draw_text(canvas, f"Tag {self.target_tag_id}", (cx + 12, cy - 24), 17, COL_GREEN, True)

    def _draw_anchor_overlay(self, canvas, det):
        """已知标靶棱柱叠加 (与 Offline Studio 同款棱柱参数, 数据源=相机实时帧):
        绿色棱柱=BA 理论位姿棱柱; 蓝色棱柱=当帧实测单靶 PnP 位姿棱柱。
        位姿: 锁定后用锁定位姿, 未锁定用当帧锚定 PnP。
        """
        if not self.show_anchors_on:
            return
        # 位姿来源 (世界->相机): 优先锁定位姿, 未锁定用当帧锚定 PnP
        rvec, tvec = self._get_world_pose(det)
        R, _ = cv2.Rodrigues(rvec) if rvec is not None else (None, None)
        t_flat = np.asarray(tvec).reshape(3) if tvec is not None else None

        def _world_tag_pose(wc):
            """世界角点 -> 标靶相机系位姿 (rvec, tvec)"""
            R_t, c_w = _tag_local_frame(wc)
            return cv2.Rodrigues(R @ R_t)[0], R @ c_w + t_flat

        for tid in self.anchor_positions:
            wc = self.engine.get_tag_world_corners(tid)
            if wc is None or R is None:
                continue
            rvec_t, tvec_t = _world_tag_pose(wc)
            if tvec_t[2] <= 1e-6:
                continue
            # 绿色理论棱柱 (Offline Studio 同款: 30x30 截面 x 75mm 生长高)
            self._draw_studio_prism(canvas, rvec_t, tvec_t, True)
            if tid in det:
                # 蓝色实测棱柱 (单靶 PnP 位姿)
                ok_b, rvec_b, tvec_b = self.engine.solve_single_tag_pnp(det[tid])
                if ok_b:
                    self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False)
                    c = det[tid].reshape(4, 2).mean(axis=0).astype(int)
                    draw_text(canvas, str(tid), (int(c[0]) + 8, int(c[1]) - 22), 14, COL_BLUE, True)
            else:
                # 未入镜的理论 Tag: 在标靶中心标注编号
                pc = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec_t, tvec_t,
                                       self.engine.camera_matrix, self.engine.dist_coeffs)[0]
                pc = pc.reshape(2).astype(int)
                draw_text(canvas, str(tid), (int(pc[0]) + 8, int(pc[1]) - 22), 14, COL_GREEN, True)

    def _draw_recognition_overlay(self, canvas):
        """单帧识别结果叠加 (静态照片, 与 Offline Studio 同款棱柱参数):
        绿色棱柱 = 地图白名单全部 Tag 的 BA 理论位姿棱柱 (需已确定世界坐标系);
        蓝色棱柱 = 当帧识别 Tag 的单靶 PnP 位姿棱柱; 目标 Tag 附带底面/顶面中心点。
        """
        R = t_flat = None
        if self.world_locked and self.locked_rvec is not None:
            R, _ = cv2.Rodrigues(self.locked_rvec)
            t_flat = np.asarray(self.locked_tvec, dtype=np.float64).reshape(3)
        K = self.engine.camera_matrix

        def _world_tag_pose(wc):
            """世界角点 -> 标靶相机系位姿 (rvec, tvec)"""
            R_t, c_w = _tag_local_frame(wc)
            return cv2.Rodrigues(R @ R_t)[0], R @ c_w + t_flat

        # 1. 绿色理论棱柱: 地图白名单全部 Tag (含目标 Tag, 若在地图中)
        map_ids = sorted(set(self.anchor_positions) |
                         ({self.target_tag_id} if self.theoretical is not None else set()))
        for tid in map_ids:
            wc = self.engine.get_tag_world_corners(tid)
            if wc is None or R is None:
                continue
            rvec_t, tvec_t = _world_tag_pose(wc)
            if tvec_t[2] <= 1e-6:
                continue
            self._draw_studio_prism(canvas, rvec_t, tvec_t, True,
                                    is_target=(tid == self.target_tag_id))
            if tid not in (self.static_det or {}):
                # 未入镜的理论 Tag: 在标靶中心标注编号
                pc = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec_t, tvec_t, K,
                                       self.engine.dist_coeffs)[0].reshape(2).astype(int)
                draw_text(canvas, str(tid), (int(pc[0]) + 8, int(pc[1]) - 22), 14, COL_GREEN, True)

        # 2. 蓝色实测棱柱: 当帧识别到的全部 Tag (单靶 PnP)
        for tid, corners in (self.static_det or {}).items():
            ok_b, rvec_b, tvec_b = self.engine.solve_single_tag_pnp(corners)
            if ok_b:
                self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False,
                                        is_target=(tid == self.target_tag_id))
                c = corners.reshape(4, 2).mean(axis=0).astype(int)
                draw_text(canvas, str(tid), (int(c[0]) + 8, int(c[1]) - 22), 14, COL_BLUE, True)
            else:
                pts = corners.reshape((-1, 2)).astype(np.int32)
                cv2.polylines(canvas, [pts], True, COL_BLUE, 2, cv2.LINE_AA)

    def _draw_xy_plane_overlay(self, canvas, det):
        """世界 XY 平面透视网格叠加 (高度由下拉框选择, 默认 Z=0 地面):
        两组互相垂直的平行线网格 + 三轴加粗高亮 (X红/Y绿/Z蓝) + 原点标记, 直观透视世界系。
        位姿: 优先锁定位姿, 未锁定用当帧锚定 PnP。
        """
        if not self.show_xy_plane_on:
            return
        rvec, tvec = self._get_world_pose(det)
        if rvec is None:
            return
        R, _ = cv2.Rodrigues(rvec)
        t_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
        K = self.engine.camera_matrix
        h_c, w_c = canvas.shape[:2]
        ext, step, z0 = self.PLANE_EXTENT_MM, self.PLANE_STEP_MM, float(self.plane_z)

        def _project(p_w):
            p_cam = R @ np.asarray(p_w, dtype=np.float64).reshape(3) + t_flat
            if p_cam[2] <= 1e-6:
                return None
            uv = K @ p_cam
            u, v = int(uv[0] / uv[2]), int(uv[1] / uv[2])
            return (u, v) if (0 <= u < w_c and 0 <= v < h_c) else None

        def _seg(p0, p1, color, thick):
            """长线段沿线采样投影连线 (自动处理出画与近裁剪)"""
            prev = None
            for k in range(25):
                s = k / 24.0
                p = (p0[0] + (p1[0] - p0[0]) * s,
                     p0[1] + (p1[1] - p0[1]) * s,
                     p0[2] + (p1[2] - p0[2]) * s)
                uv = _project(p)
                if uv is not None and prev is not None:
                    cv2.line(canvas, prev, uv, color, thick, cv2.LINE_AA)
                prev = uv

        # 平行线网格 (绘制高度 z0): 平行于 X 轴 (Y=-ext..ext) 与平行于 Y 轴 (X=-ext..ext) 两组
        for i in range(-ext, ext + 1, step):
            _seg((-ext, i, z0), (ext, i, z0), COL_GRAY, 1)
            _seg((i, -ext, z0), (i, ext, z0), COL_GRAY, 1)
        # 坐标轴加粗高亮: X 红 / Y 绿 (随平面高度) / Z 蓝 (0→600mm), 明显可见
        _seg((-ext, 0, z0), (ext, 0, z0), (60, 60, 245), 3)
        _seg((0, -ext, z0), (0, ext, z0), COL_GREEN, 3)
        _seg((0, 0, 0), (0, 0, self.PLANE_Z_MM), COL_BLUE, 4)
        # Tag 等高辅助红线: 平面高度与某锚定标靶中心 Z 重合且该靶不在原点时,
        # 平移一条红色 X 轴穿过该标靶 (如 Z=196 平面过 Tag 1); Tag 0 在原点, 主 X 轴已穿过
        for tid, c_t in self.anchor_positions.items():
            if abs(float(c_t[2]) - z0) < 2.0 and (abs(float(c_t[0])) > 1.0 or abs(float(c_t[1])) > 1.0):
                _seg((c_t[0] - ext, c_t[1], z0), (c_t[0] + ext, c_t[1], z0), (60, 60, 245), 2)
                uv = _project((c_t[0] + ext, c_t[1], z0))
                if uv is not None:
                    draw_text(canvas, f"X (Tag {tid})", (uv[0] + 6, uv[1] - 8), 13, (60, 60, 245), True)
        # Z 轴高度刻度 (每 100mm) + 顶端箭头: 俯视相机下高度轴指向镜头呈放射状,
        # 刻度数值让"向上生长"方向一目了然, 消除透视歧义
        for hz in range(100, self.PLANE_Z_MM, 100):
            tp = _project((0, 0, hz))
            if tp is not None:
                cv2.line(canvas, (tp[0] - 5, tp[1]), (tp[0] + 5, tp[1]), COL_BLUE, 2, cv2.LINE_AA)
                draw_text(canvas, str(hz), (tp[0] + 8, tp[1] - 6), 12, COL_BLUE, True)
        p_top = _project((0, 0, self.PLANE_Z_MM))
        p_base = _project((0, 0, 0))
        if p_top is not None and p_base is not None:
            d = np.array(p_top, dtype=np.float64) - np.array(p_base, dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n > 24:
                d /= n
                perp = np.array([-d[1], d[0]])
                tip = np.array(p_top, dtype=np.float64)
                wing = 14.0 * d
                arrow = np.array([tip, tip - wing + 6.0 * perp, tip - wing - 6.0 * perp],
                                 dtype=np.int32)
                cv2.fillPoly(canvas, [arrow], COL_BLUE)
        for label, p, col in (("X", (ext + 70, 0, z0), (60, 60, 245)),
                              ("Y", (0, ext + 70, z0), COL_GREEN),
                              ("Z", (0, 0, self.PLANE_Z_MM + 70), COL_BLUE),
                              ("0", (0, 0, 0), COL_WHITE)):
            uv = _project(p)
            if uv is not None:
                draw_text(canvas, label, (uv[0] + 6, uv[1] - 8), 15, col, True)

    def _draw_info_panel(self, canvas, y_off=0):
        """左上信息面板 (实测 / 理论 / 偏差 / 世界系状态 / 机械臂状态)"""
        x1, y1 = 14, y_off + 14
        x2, y2 = 478, y_off + 296
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
        if self.sampling:
            draw_text(canvas, f"世界坐标系: {self.sample_stage}", (x1 + 14, y), 16, COL_YELLOW, True)
        elif self.world_locked:
            draw_text(canvas, f"世界坐标系: 已锁定 ({self.lock_info})", (x1 + 14, y), 16, COLOR_ACCENT)
        else:
            draw_text(canvas, "世界坐标系: 未确定 [确定世界坐标系]", (x1 + 14, y), 16, COL_GRAY)
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
        elif btn_id == "TOGGLE_ANCHORS":
            self.show_anchors_on = not self.show_anchors_on
            self.set_toast("已显示已知 Tag 位置 (绿=BA理论 / 蓝=实测)" if self.show_anchors_on
                           else "已知 Tag 位置显示已关闭")
        elif btn_id == "TOGGLE_RECOG":
            self.recog_tag2_on = not self.recog_tag2_on
            if not self.recog_tag2_on:
                self.support_ids = []
            self.set_toast(f"识别 Tag {self.target_tag_id} 已开启" if self.recog_tag2_on
                           else f"识别 Tag {self.target_tag_id} 已关闭 (纯预览)")
        elif btn_id == "TOGGLE_LOCK":
            self.toggle_world_lock()
        elif btn_id == "TOGGLE_PLANE_DD":
            self.active_dropdown = None if self.active_dropdown == "PLANE_DROPDOWN" else "PLANE_DROPDOWN"
        elif btn_id.startswith("DD_PLANE_"):
            self.active_dropdown = None
            if payload is None:
                self.show_xy_plane_on = False
                self.set_toast("XY 平面网格已关闭")
            else:
                self.show_xy_plane_on = True
                self.plane_z = int(payload)
                label = self.PLANE_Z_LABELS.get(self.plane_z, "")
                self.set_toast(f"XY 平面已重绘至 Z={self.plane_z} mm{label}")
        elif btn_id == "TRIGGER_RECOG":
            self.trigger_recognize()
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
        # Windows 原生 Unicode API 注入中文标题, 彻底消除标题栏问号乱码 (借鉴 GuiWindowManager)
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(None, win_name)
                if hwnd:
                    ctypes.windll.user32.SetWindowTextW(hwnd, win_name)
            except Exception:
                pass
        cv2.setMouseCallback(win_name, self._on_mouse)
        print("\n" + "=" * 68)
        print(" Robot 在线跟踪 (GUI 已启动, 相机未开启)")
        print("   顶部工具栏: 相机类型 → 分辨率 → [开启] → [识别] → [确定世界坐标系] → [XY平面▼] → [显示已知Tag] → [识别 Tag 2]")
        print("   [识别] 一键单帧闭环: 开相机→拍一张→关相机→识别Tag(蓝棱柱)→确定世界坐标系→地图白名单绿棱柱")
        print("   开启相机后为纯预览; [确定世界坐标系] 一键执行: 采样30帧→滤波→求解零点→锁定")
        print("   [XY平面▼]: 不绘制 / Z 405(Tag0) 196(Tag1) 350 300...0 mm 透视网格+三轴, Tag 等高平面附加红色 X 轴")
        print("   快捷键: [S] 一键识别 | [P] XY平面下拉 | [A] 显示已知Tag | [R] 识别Tag2 | [L] 确定/解除世界坐标系 | [C] 连接机械臂 | [T] 跟踪 | [X] 退出")
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
                        # 实时叠加直接画在原始帧上 (帧坐标), 再与工具栏拼合, 保证与画面内容对齐
                        if self.recog_tag2_on or self.show_anchors_on or self.show_xy_plane_on:
                            det = self.solve_frame(frame)
                            if det is not None:
                                self._draw_overlay(frame, det)
                                self._draw_xy_plane_overlay(frame, det)
                                self._draw_anchor_overlay(frame, det)
                        canvas = self._compose_canvas(frame)
                        self._draw_info_panel(canvas, y_off=TOOLBAR_H)
                elif self.static_frame is not None:
                    # 单帧识别结果静态显示 (相机已关闭): 蓝=当帧实测棱柱 / 绿=地图理论棱柱
                    disp = self.static_frame.copy()
                    self._draw_xy_plane_overlay(disp, None)
                    self._draw_recognition_overlay(disp)
                    canvas = self._compose_canvas(disp)
                    self._draw_info_panel(canvas, y_off=TOOLBAR_H)
                    draw_text(canvas, "单帧识别结果 (相机已关闭): 蓝=当帧实测 / 绿=地图理论",
                              (14, TOOLBAR_H + 310), 15, COL_CYAN, True)
                else:
                    # 相机未开启: 占位画面 (尺寸跟随所选分辨率, 保证开启前后工具栏视觉一致)
                    cw, ch = self.frame_w, self.frame_h
                    canvas = np.full((ch + TOOLBAR_H, cw, 3), COLOR_BG, dtype=np.uint8)
                    draw_text(canvas, "相机未开启",
                              (cw // 2 - 120, ch // 2 - 50), 32, COLOR_ACCENT, True)
                    draw_text(canvas, "请先选择相机类型和分辨率，然后点击 [开启] 按钮",
                              (cw // 2 - 250, ch // 2 + 10), 18, COLOR_TEXT_SUB)

                self._draw_toolbar(canvas)
                self._draw_toast(canvas)
                cv2.imshow(win_name, canvas)

                # 画布尺寸变化时同步窗口尺寸, 避免工具栏被缩放变小
                cur_size = (canvas.shape[1], canvas.shape[0])
                if cur_size != self._last_canvas_size:
                    cv2.resizeWindow(win_name, cur_size[0], cur_size[1])
                    self._last_canvas_size = cur_size

                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                key = cv2.waitKeyEx(30)
                if key == -1:
                    continue
                k = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""
                if k in ("x", "q") or key == 27:
                    break
                elif k == "a":
                    self._handle_action("TOGGLE_ANCHORS", None)
                elif k == "s":
                    self.trigger_recognize()
                elif k == "r":
                    self._handle_action("TOGGLE_RECOG", None)
                elif k == "p":
                    self._handle_action("TOGGLE_PLANE_DD", None)
                elif k == "l":
                    self.toggle_world_lock()
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
