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
from tools.tracker.camera_controller import CameraController
from tools.tracker.common import (
    COLOR_ACCENT, COLOR_BG, COLOR_TEXT_SUB, COL_CYAN, COL_YELLOW,
    TOOLBAR_H, draw_text)
from tools.tracker.renderer import TrackerRenderer

try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "robot_online_tracker"


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

        # 3. 相机硬件控制器 (类型/分辨率状态机 + 取流启停; GUI 先行, 不自动开相机)
        self.camera = CameraController(self.engine)

        # 4. 工具栏状态 (借鉴 d435_viewer: 相机类型 → 分辨率 → 开关; 绘制由 TrackerRenderer 负责)
        self.active_dropdown = None     # "CAMERA_TYPE_DROPDOWN" | "RES_DROPDOWN" | "PLANE_DROPDOWN" | None
        self.plane_z = 0                # XY 平面绘制高度 (mm, 下拉框选择)
        self.plane_options = [
            (None, "不绘制 XY 平面")
        ] + [
            (z, f"Z {z} mm" + self.PLANE_Z_LABELS.get(z, ""))
            for z in self.PLANE_Z_CHOICES
        ]
        self.renderer = TrackerRenderer(self)  # UI 渲染器 (工具栏/叠加层/按钮命中表)
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
                self.camera.camera_type = state["camera_type"]
            if any(k == state.get("resolution") for k, _ in self.camera.resolution_options):
                self.camera.resolution = state["resolution"]
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
                "camera_type": self.camera.camera_type,
                "resolution": self.camera.resolution,
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
        if cam_key == self.camera.camera_type:
            return
        if self.camera.pipeline_running:
            self._toggle_camera(force_off=True)
        self.camera.camera_type = cam_key
        self._save_viewer_state()
        print(f"[INFO] 相机类型已切换为: {dict(self.camera.camera_options).get(cam_key, cam_key)}")

    def _change_resolution(self, res_key):
        """切换分辨率：运行中则先停再按新分辨率重启 (任务执行中禁止)"""
        if res_key == self.camera.resolution:
            return
        if self.sampling or self.tracking or self.recognizing:
            self.set_toast("任务执行中, 禁止切换分辨率", True)
            return
        was_running = self.camera.pipeline_running
        if was_running:
            self._toggle_camera(force_off=True)
        self.camera.set_resolution_key(res_key)
        self._save_viewer_state()
        if was_running:
            self._toggle_camera()
        print(f"[INFO] 分辨率已切换: {res_key}")

    def _toggle_camera(self, force_off=False, _internal=False):
        """开启或关闭相机取流 (采样/跟踪/识别任务执行中禁止手动开关; 内部调用与强制关闭除外)"""
        if not force_off and not _internal and (self.sampling or self.tracking or self.recognizing):
            self.set_toast("任务执行中, 禁止切换相机状态", True)
            return
        if force_off or self.camera.pipeline_running:
            self.camera.stop()
            self._release_world_lock(silent=True)
            self.recog_tag2_on = False
            self.set_toast("相机已关闭")
            print("[INFO] 相机已关闭")
        else:
            # 重新开启相机: 单帧识别静态结果让位于实时预览
            self.static_frame = None
            self.static_det = None
            try:
                self.camera.start()
                self.set_toast(f"相机已开启: {dict(self.camera.camera_options).get(self.camera.camera_type)}"
                               f" @ {self.camera.resolution}")
            except Exception as e:
                print(f"[ERROR] 相机开启失败: {e}")
                self.set_toast(f"相机开启失败: {e}", True)

    def _release_world_lock(self, silent=False):
        """解除世界系锁定 (相机关闭/切换时位姿失效)"""
        if self.world_locked or self.lock_info:
            self.world_locked = False
            self.locked_rvec = None
            self.locked_tvec = None
            self.lock_info = ""
            if not silent:
                self.set_toast("世界坐标系锁定已解除, 可重新确定")

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

    def get_world_pose(self, det):
        """当前相机世界位姿 (世界->相机): 优先锁定位姿, 未锁定用当帧锚定 PnP (渲染器叠加层使用)"""
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
        if not self.camera.pipeline_running:
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
                frame = self.camera.read_frame()
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
            if not self.camera.pipeline_running:
                self.recog_stage = "开启相机..."
                self._toggle_camera(_internal=True)
                if not self.camera.pipeline_running:
                    return
            # 2. 采集一帧 (连读数帧待曝光稳定, 取最后一帧)
            self.recog_stage = "采集照片..."
            frame = None
            for _ in range(8):
                f = self.camera.read_frame()
                if f is not None:
                    frame = f
                time.sleep(0.05)
            # 3. 关相机 (拍完即关, 完整闭环)
            self.recog_stage = "关闭相机..."
            if self.camera.pipeline_running:
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

    # ------------------------------ 界面渲染 (由 TrackerRenderer 负责) ------------------------------

    # ------------------------------ 鼠标交互 ------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.renderer.on_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            hit = self.renderer.hit_test(x, y)
            if hit is not None:
                self._handle_action(*hit)
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
                if self.camera.pipeline_running:
                    frame = self.camera.read_frame()
                    if frame is None:
                        canvas = np.full((self.camera.frame_h + TOOLBAR_H, self.camera.frame_w, 3),
                                         COLOR_BG, dtype=np.uint8)
                        draw_text(canvas, "取流中...", (self.camera.frame_w // 2 - 60,
                                                       self.camera.frame_h // 2), 22, COL_YELLOW, True)
                    else:
                        # 实时叠加直接画在原始帧上 (帧坐标), 再与工具栏拼合, 保证与画面内容对齐
                        if self.recog_tag2_on or self.show_anchors_on or self.show_xy_plane_on:
                            det = self.solve_frame(frame)
                            if det is not None:
                                self.renderer.draw_overlay(frame, det)
                                self.renderer.draw_xy_plane_overlay(frame, det)
                                self.renderer.draw_anchor_overlay(frame, det)
                        canvas = self.renderer.compose_canvas(frame)
                        self.renderer.draw_info_panel(canvas, y_off=TOOLBAR_H)
                elif self.static_frame is not None:
                    # 单帧识别结果静态显示 (相机已关闭): 蓝=当帧实测棱柱 / 绿=地图理论棱柱
                    disp = self.static_frame.copy()
                    self.renderer.draw_xy_plane_overlay(disp, None)
                    self.renderer.draw_recognition_overlay(disp)
                    canvas = self.renderer.compose_canvas(disp)
                    self.renderer.draw_info_panel(canvas, y_off=TOOLBAR_H)
                    draw_text(canvas, "单帧识别结果 (相机已关闭): 蓝=当帧实测 / 绿=地图理论",
                              (14, TOOLBAR_H + 310), 15, COL_CYAN, True)
                else:
                    # 相机未开启: 占位画面 (尺寸跟随所选分辨率, 保证开启前后工具栏视觉一致)
                    cw, ch = self.camera.frame_w, self.camera.frame_h
                    canvas = np.full((ch + TOOLBAR_H, cw, 3), COLOR_BG, dtype=np.uint8)
                    draw_text(canvas, "相机未开启",
                              (cw // 2 - 120, ch // 2 - 50), 32, COLOR_ACCENT, True)
                    draw_text(canvas, "请先选择相机类型和分辨率，然后点击 [开启] 按钮",
                              (cw // 2 - 250, ch // 2 + 10), 18, COLOR_TEXT_SUB)

                self.renderer.draw_toolbar(canvas)
                self.renderer.draw_toast(canvas)
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
