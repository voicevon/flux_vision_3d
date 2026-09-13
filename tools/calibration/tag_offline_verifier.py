#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线标定精度体检与 Leave-One-Out 盲测批量验证 GUI 工作台 (Tag Offline Verifier Workbench)
=====================================================================================
核心设计哲学：【独立于 BA 求解的第三方审判官 + 闭环自愈质检控制台】

功能特性：
  1. 闭环质检交互工作台 (OpenCV Interactive GUI)：
     - 逐帧全分辨率检视：清晰呈现实测角点虚线、盲测反推实线、15倍放大残差向量与误差标牌；
     - 顶部 HUD 仪表盘：当前帧指标、全局中位空间偏差 (mm)、系统性偏差嫌疑标靶、放行门限指示灯；
     - 底部触控/鼠标/快捷键工具栏：平滑翻页、仅看回审帧 (F)、审核微调 (R)、重新平差 (B)、进入 AR (V)；
  2. 多工序分层协同与生命周期管理：
     - 【体检 <-> 审核 (R)】：模态挂起握手，带入问题帧与问题 Tag 直达审核画板，微调退出后原地刷新；
     - 【体检 -> 平差 (B)】：就地静默触发两阶段 BA 全局平差，原地热重载地图数据，无需切窗；
     - 【体检 -> 在线 AR (V)】：门限放行守门员机制，达标后优雅销毁窗口，干净移交拉起 AR 验证器；
  3. 双模运行支持：
     - GUI 交互模式（默认）：面向工程师直观质检与闭环迭代；
     - 无头批处理模式 (`--headless`)：面向 CI/自动化测试与脚本批量质检。
=====================================================================================
"""

import os
import sys
import math
import time
import glob
import copy
import yaml
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import cv2

# Windows 终端 UTF-8 编码适配
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MAP_PATH = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")
VERIFICATION_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_verification")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

try:
    from src.utils.window_helper import force_window_focus
except ImportError:
    force_window_focus = None

from src.calibration.verification_reporter import VerificationReporter
from src.calibration.offline_engine import OfflineVerificationEngine
from src.calibration.verification_visualizer import VerificationVisualizer

try:
    from src.utils.viewport_manager import (
        ViewportManager, get_safe_screen_size,
        draw_styled_button, draw_segmented_toggle,
        draw_dropdown_box
    )
except ImportError:
    ViewportManager = None
    get_safe_screen_size = None
    draw_styled_button = None
    draw_segmented_toggle = None
    draw_dropdown_box = None


class TagOfflineVerifier:
    """离线标定精度体检引擎：Leave-One-Out 盲测批量验证与 GUI 工作台"""

    def __init__(self,
                 map_path: str = DEFAULT_MAP_PATH,
                 image_dir: str = DEFAULT_IMAGE_DIR,
                 marker_size_mm: float = 50.0,
                 source: str = "auto",
                 caller_ar_instance: Any = None):
        """
        初始化离线体检引擎
        :param map_path: tags_map.yaml 路径
        :param image_dir: 采集样本图像目录
        :param marker_size_mm: 标靶物理边长 (mm)，会被地图中的值覆盖
        :param source: 数据源 'auto' (优先清单), 'manifest' (仅已审核清单), 'images' (原图重检测)
        :param caller_ar_instance: 若由 AR 验证器唤起，传入其实例用于避免递归调用
        """
        self.map_path = map_path
        self.image_dir = image_dir
        self.marker_size_mm = marker_size_mm
        self.source = (source or "auto").lower()
        self.caller_ar_instance = caller_ar_instance

        # 加载标靶空间立体地图
        self.tags_map = None
        self.mapped_tag_ids = []
        self._load_tags_map()

        # 加载相机内参
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()

        # 加载白名单
        self.valid_tag_ids = self._load_valid_tag_ids()

        # 数据源策略判定
        self.manifest_path = os.path.join(self.image_dir, "tag_observations.yaml")
        self.manifest_data = None
        self._load_manifest()

        # 初始化双路互补融合检测器 (与 TagMapBuilder 一致)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self.detector_bright, self.detector_dark = self._build_detectors()
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        # 标靶局部坐标系角点
        s = self.marker_size_mm / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

        # 实例化离线标定精度体检纯算法引擎 (解耦纯计算与 GUI 交互)
        self.engine = OfflineVerificationEngine(
            tags_map=self.tags_map,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm,
            valid_tag_ids=self.valid_tag_ids
        )

        # 实例化视觉呈现与双模态渲染器 (解耦渲染与交互)
        self.visualizer = VerificationVisualizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )

        # 确保输出目录存在
        os.makedirs(VERIFICATION_DIR, exist_ok=True)
        self.vis_dir = os.path.join(self.image_dir, "visualized")
        os.makedirs(self.vis_dir, exist_ok=True)

        # GUI 交互状态
        self.window_name = "AprilTag Calibration Offline Verifier Workbench (GUI Edition)"
        self.is_running_gui = False
        self.current_idx = 0
        self.image_files: List[str] = []
        self.frame_results_cache: Dict[str, List[Dict]] = {}
        self.all_results: List[Dict] = []
        self.stats: Dict[str, Any] = {}
        self.filter_flagged_only = False
        self.flagged_frames: List[str] = []
        self.flagged_tags: List[int] = []
        self.is_filter_dropdown_open: bool = False

        # 鼠标交互
        self.mouse_pos = (-1, -1)
        self.gui_buttons: List[Tuple[str, Tuple[int, int, int, int], str, Tuple[int, int, int]]] = []

        # Toast 通知
        self.toast_msg = ""
        self.toast_time = 0.0

        # 门限防呆与二次确认时间戳
        self.last_ar_warn_time = 0.0

        # 自适应工作区与分层视口管理器 (彻底解决超屏与按钮模糊问题)
        if get_safe_screen_size:
            self.win_w, self.win_h = get_safe_screen_size(preferred_w=1280, preferred_h=720)
        else:
            self.win_w, self.win_h = 1280, 720

        if ViewportManager:
            self.viewport = ViewportManager(win_w=self.win_w, win_h=self.win_h, top_bar_h=62, bottom_bar_h=58)
        else:
            self.viewport = None

        # 3D 棱柱虚实对比视图模式: True 为 3D 双四棱柱对比视图 (默认开启)，False 为 2D 角点残差矢量视图
        self.view_mode_3d = True

    def _load_manifest(self):
        """加载已审核观测清单"""
        if os.path.exists(self.manifest_path) and self.source in ("auto", "manifest"):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest_data = yaml.safe_load(f)
                self.actual_source = "manifest"
            except Exception as e:
                print(f"[WARN] 无法解析观测清单: {e}，回退至图像重检测模式")
                self.actual_source = "images"
        else:
            self.actual_source = "images"

    def _load_tags_map(self):
        """加载 tags_map.yaml 空间立体地图"""
        if not os.path.exists(self.map_path):
            raise FileNotFoundError(f"找不到标靶地图文件: {self.map_path}\n请先运行 BA 求解生成地图！")

        with open(self.map_path, "r", encoding="utf-8") as f:
            self.tags_map = yaml.safe_load(f)

        self.marker_size_mm = float(self.tags_map.get("marker_size_mm", self.marker_size_mm))
        self.mapped_tag_ids = sorted([int(tid) for tid in self.tags_map.get("tags", {}).keys()])
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.tags_map = self.tags_map
            self.engine.mapped_tag_ids = self.mapped_tag_ids
            self.engine.marker_size_mm = self.marker_size_mm
        print(f"[OK] 已加载标靶空间地图: {self.map_path} (包含 {len(self.mapped_tag_ids)} 个标靶: {self.mapped_tag_ids})")

    def _load_camera_intrinsics(self):
        """加载相机内参，优先从 config_guard 读取"""
        if resolve_camera_intrinsics is not None:
            K, dist, _ = resolve_camera_intrinsics(CONFIG_PATH)
            return K, dist
        else:
            K = np.array([
                [1363.68, 0.0, 971.19],
                [0.0, 1361.19, 566.26],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            return K, np.zeros((5, 1), dtype=np.float64)

    def _load_valid_tag_ids(self) -> List[int]:
        """从 config.yaml 加载白名单"""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                ids = cfg.get("calibration", {}).get("valid_tag_ids", [])
                if ids:
                    return [int(x) for x in ids]
            except Exception:
                pass
        return list(range(30))

    def _build_detectors(self):
        """构建双路互补检测器 (与 TagMapBuilder 一致)"""
        def make_params(thresh_c, min_otsu):
            p = cv2.aruco.DetectorParameters()
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 43
            p.adaptiveThreshWinSizeStep = 8
            p.adaptiveThreshConstant = thresh_c
            p.minOtsuStdDev = min_otsu
            p.minMarkerPerimeterRate = 0.008
            p.maxMarkerPerimeterRate = 4.0
            p.polygonalApproxAccuracyRate = 0.09
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            p.perspectiveRemovePixelPerCell = 10
            p.errorCorrectionRate = 0.50
            p.perspectiveRemoveIgnoredMarginPerCell = 0.15
            p.maxErroneousBitsInBorderRate = 0.30
            return p

        det_bright = cv2.aruco.ArucoDetector(self.dictionary, make_params(5.5, 0.55))
        det_dark = cv2.aruco.ArucoDetector(self.dictionary, make_params(2.5, 0.45))
        return det_bright, det_dark

    def detect_tags(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """双路互补融合检测 (委托给 OfflineVerificationEngine)"""
        return self.engine.detect_tags(image)

    def _refine_corners(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """亚像素角点精修 (委托给 OfflineVerificationEngine)"""
        return self.engine._refine_corners(gray, corners)

    def _get_tag_world_transform(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶在世界坐标系下的 4x4 变换矩阵 (委托给 OfflineVerificationEngine)"""
        return self.engine.get_tag_world_transform(tag_id)

    def _get_tag_world_corners(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶 4 个角点在世界坐标系下的 3D 位置 (委托给 OfflineVerificationEngine)"""
        return self.engine.get_tag_world_corners(tag_id)

    def _solve_camera_pose(self, tag_corners_pairs: List[Tuple[int, np.ndarray]]) -> Optional[Dict]:
        """超定 PnP 求解相机位姿 (委托给 OfflineVerificationEngine)"""
        return self.engine.solve_camera_pose(tag_corners_pairs)

    def _solve_single_tag_pnp(self, corners_2d: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """解算单标靶实测相机外参位姿 (委托给 OfflineVerificationEngine)"""
        return self.engine.solve_single_tag_pnp(corners_2d)

    def _compute_loo_error(self, rvec, tvec, tag_id: int, observed_corners: np.ndarray) -> Optional[Dict]:
        """计算单个盲测目标的重投影误差 (委托给 OfflineVerificationEngine)"""
        return self.engine.compute_loo_error(rvec, tvec, tag_id, observed_corners)

    def _verify_single_frame(self, img_path: str) -> List[Dict]:
        """对单帧执行全标靶 Leave-One-Out 盲测循环 (委托给 OfflineVerificationEngine)"""
        return self.engine.verify_single_frame(img_path, self.manifest_data, self.actual_source)

    def render_tag_dual_prisms(self, img: np.ndarray,
                               ba_rvec: Optional[np.ndarray], ba_tvec: Optional[np.ndarray],
                               obs_rvec: Optional[np.ndarray], obs_tvec: Optional[np.ndarray],
                               tag_id: int, err_px: float, err_mm: float,
                               observed_corners: Optional[np.ndarray] = None):
        """绘制 3D 双四棱柱空间对比 (委托给 VerificationVisualizer)"""
        return self.visualizer.render_tag_dual_prisms(
            img=img, ba_rvec=ba_rvec, ba_tvec=ba_tvec,
            obs_rvec=obs_rvec, obs_tvec=obs_tvec,
            tag_id=tag_id, err_px=err_px, err_mm=err_mm,
            observed_corners=observed_corners
        )

    def _render_verification_frame(self, img_path: str, frame_results: List[Dict], out_path: Optional[str] = None) -> np.ndarray:
        """渲染单帧 LOO 盲测可视化图 (委托给 VerificationVisualizer)"""
        return self.visualizer.render_verification_frame(
            img_path=img_path,
            frame_results=frame_results,
            view_mode_3d=self.view_mode_3d,
            manifest_data=self.manifest_data,
            solve_single_tag_pnp_func=self._solve_single_tag_pnp,
            out_path=out_path
        )

    def _aggregate_statistics(self, all_results: List[Dict]) -> Dict:
        """汇总 Per-Tag / Per-Frame 统计 (委托给 VerificationReporter)"""
        return VerificationReporter.aggregate_statistics(all_results)

    def _export_markdown_report(self, stats: Dict, all_results: List[Dict]) -> str:
        """输出完整 Markdown 精度体检报告 (委托给 VerificationReporter)"""
        return VerificationReporter.export_markdown_report(
            stats=stats,
            all_results=all_results,
            map_path=self.map_path,
            image_dir=self.image_dir,
            actual_source=self.actual_source,
            vis_dir=self.vis_dir
        )

    def recompute_all(self):
        """重新扫描全部图像并重新计算全部盲测指标与统计"""
        self._load_manifest()
        self.image_files = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.endswith("_annotated.png")
            and not f.endswith("_quiver.png")
            and not f.endswith("_loo_verify.png")
            and "visualized" not in f
        ])

        self.all_results = []
        self.frame_results_cache = {}

        for fpath in self.image_files:
            fname = os.path.basename(fpath)
            res = self._verify_single_frame(fpath)
            self.frame_results_cache[fname] = res
            self.all_results.extend(res)

            # 同步写盘可视化静态图
            vis_path = os.path.join(self.vis_dir, os.path.splitext(fname)[0] + "_loo_verify.png")
            self._render_verification_frame(fpath, res, vis_path)

        self.stats = self._aggregate_statistics(self.all_results)
        self.flagged_frames = self.stats.get("flagged_frames", [])
        self.flagged_tags = self.stats.get("flagged_tags", [])
        self._export_markdown_report(self.stats, self.all_results)

    def run_full_verification(self) -> Dict:
        """无头批处理模式主入口：扫描 -> 计算 -> 渲染 -> 导出报告"""
        self.recompute_all()
        source_label = "已审核观测清单 (tag_observations.yaml)" if self.actual_source == "manifest" else "原始采图重新检测 (端到端独立复检)"
        print("\n" + "=" * 80)
        print("  【离线标定精度体检与 Leave-One-Out 盲测批量验证】")
        print("=" * 80)
        print(f"  -> 标靶地图     : {self.map_path} ({len(self.mapped_tag_ids)} 个标靶)")
        print(f"  -> 采图目录     : {self.image_dir} ({len(self.image_files)} 张图像)")
        print(f"  -> 验证数据源   : {source_label}")
        print(f"  -> 验证策略     : 鲁棒 RANSAC PnP + 逐 Tag Leave-One-Out 盲测")
        print(f"  -> 输出报告目录 : {VERIFICATION_DIR}")
        print("=" * 80 + "\n")

        grade_label = self.stats.get("grade_label", "N/A")
        print("\n" + "=" * 80)
        print("  【离线精度体检汇总大成报表】")
        print("=" * 80)
        print(f"  -> 总处理帧数   : {len(self.image_files)} 张")
        print(f"  -> LOO 盲测总数 : {self.stats.get('total_tests', 0)} 次")
        print(f"  -> 中位像元误差 : {self.stats.get('global_median_px', 0):.3f} px")
        print(f"  -> 中位空间偏差 : {self.stats.get('global_median_mm', 0):.3f} mm")
        if self.flagged_tags:
            print(f"  -> 偏差嫌疑标靶 : {self.flagged_tags} (建议重点检查)")
        if self.flagged_frames:
            print(f"  -> 建议回审帧   : {self.flagged_frames}")
        print(f"  -> 综合评级     : {grade_label}")
        print("=" * 80 + "\n")
        return self.stats

    # =========================================================================
    # GUI 交互工作台系统 (Interactive GUI Workbench)
    # =========================================================================

    def set_toast(self, msg: str):
        """设置屏幕悬浮 Toast 气泡提示"""
        self.toast_msg = msg
        self.toast_time = time.time()

    def _on_mouse_gui(self, event, x, y, flags, param):
        """鼠标交互处理：点击工具栏、悬停高亮、滚轮缩放、中键平移拖拽"""
        self.mouse_pos = (x, y)

        # 0. 滚轮以鼠标为中心平滑缩放与中键平移拖拽 (通用视口引擎支持)
        if event == cv2.EVENT_MOUSEWHEEL:
            if self.viewport and self.viewport.handle_mouse_wheel(x, y, flags):
                pass
            return

        if event == cv2.EVENT_MBUTTONDOWN:
            if self.viewport:
                self.viewport.start_pan(x, y)
            return
        elif event == cv2.EVENT_MBUTTONUP:
            if self.viewport:
                self.viewport.end_pan()
            return
        elif event == cv2.EVENT_MBUTTONDBLCLK:
            if self.viewport and self.viewport.reset_zoom():
                self.set_toast("视口已复位 (1.0x)")
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if self.viewport and self.viewport.is_panning:
                self.viewport.update_pan(x, y)
                return

        # 同时支持按下 (LBUTTONDOWN) 与松开 (LBUTTONUP) 触发，确保所有鼠标驱动与触控习惯均能 100% 触发点击
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            w, h = self.win_w, self.win_h
            top_bar_h = self.viewport.top_bar_h if self.viewport else 62
            bot_bar_h = self.viewport.bottom_bar_h if self.viewport else 58

            # A. 退出指令全域绝对优先捕获 (零防抖冷却限制，仅保留右上角关闭热区与标记为 EXIT 的按钮)
            # 1) 顶部 HUD 右上角关闭区域 (w - 120 <= x <= w 且 y <= top_bar_h + 8)
            # 2) 遍历命中了任何标记为 EXIT 的按钮矩形 (右上角关闭按钮，带 8px 扩充容差)
            is_exit_clicked = False
            if (w - 120) <= x and 0 <= y <= (top_bar_h + 8):
                is_exit_clicked = True
            else:
                for btn_id, (bx1, by1, bx2, by2), _, _ in self.gui_buttons:
                    if btn_id == "EXIT":
                        if (bx1 - 8) <= x <= (bx2 + 8) and (by1 - 8) <= y <= (by2 + 8):
                            is_exit_clicked = True
                            break

            if is_exit_clicked:
                print("\n[*] 收到鼠标退出点击指令，离线精度体检 GUI 工作台即将关闭...")
                self.is_running_gui = False
                return

            # B. 其他业务按钮响应 (仅在实际命中业务按钮后更新防抖冷却，避免误吞点击)
            now = time.time()
            if now - getattr(self, "_last_btn_click_time", 0.0) < 0.20:
                return

            # C. 下拉菜单优先拦截 (若当前处于展开状态，优先捕获菜单项点击或外部空白点击关闭)
            if self.is_filter_dropdown_open:
                self._last_btn_click_time = now
                # 1) 检查是否命中下拉选项
                for btn_id, (bx1, by1, bx2, by2), _, _ in self.gui_buttons:
                    if btn_id.startswith("SELECT_"):
                        if bx1 <= x <= bx2 and by1 <= y <= by2:
                            if btn_id == "SELECT_ALL":
                                self.set_flagged_filter(False)
                            elif btn_id == "SELECT_FLAGGED":
                                self.set_flagged_filter(True)
                            return
                # 2) 检查是否点击了下拉框主按钮 (再次点击收起)
                for btn_id, (bx1, by1, bx2, by2), _, _ in self.gui_buttons:
                    if btn_id == "DROPDOWN_TOGGLE":
                        if bx1 <= x <= bx2 and by1 <= y <= by2:
                            self.is_filter_dropdown_open = False
                            return
                # 3) 点击了菜单与主框外部 (Click Outside)，自动收起且不向下穿透误触
                self.is_filter_dropdown_open = False
                return

            # D. 常态业务按钮响应 (下拉菜单未展开时)
            for btn_id, (bx1, by1, bx2, by2), label, _ in self.gui_buttons:
                pad = 5
                if (bx1 - pad) <= x <= (bx2 + pad) and (by1 - pad) <= y <= (by2 + pad):
                    self._last_btn_click_time = now
                    if btn_id == "DROPDOWN_TOGGLE":
                        self.is_filter_dropdown_open = True
                    elif btn_id == "PREV":
                        self.prev_image()
                    elif btn_id == "NEXT":
                        self.next_image()
                    elif btn_id == "TOGGLE_FILTER":
                        self.toggle_flagged_filter()
                    elif btn_id == "TOGGLE_3D":
                        self.toggle_view_mode()
                    elif btn_id == "OPEN_REVIEWER":
                        self.open_reviewer_modal()
                    elif btn_id == "RUN_BA":
                        self.run_in_place_bundle_adjustment()
                    elif btn_id == "ENTER_AR":
                        self.handover_to_ar_verifier()
                    elif btn_id == "EXPORT":
                        self.export_report_action()
                    return

    def prev_image(self):
        """上一张图片"""
        if not self.image_files:
            return
        if self.filter_flagged_only and self.flagged_frames:
            cur_fname = os.path.basename(self.image_files[self.current_idx])
            try:
                curr_sub = self.flagged_frames.index(cur_fname)
                next_sub = (curr_sub - 1) % len(self.flagged_frames)
            except ValueError:
                next_sub = len(self.flagged_frames) - 1
            target_fname = self.flagged_frames[next_sub]
            for idx, p in enumerate(self.image_files):
                if os.path.basename(p) == target_fname:
                    self.current_idx = idx
                    break
        else:
            self.current_idx = (self.current_idx - 1) % len(self.image_files)

    def next_image(self):
        """下一张图片"""
        if not self.image_files:
            return
        if self.filter_flagged_only and self.flagged_frames:
            cur_fname = os.path.basename(self.image_files[self.current_idx])
            try:
                curr_sub = self.flagged_frames.index(cur_fname)
                next_sub = (curr_sub + 1) % len(self.flagged_frames)
            except ValueError:
                next_sub = 0
            target_fname = self.flagged_frames[next_sub]
            for idx, p in enumerate(self.image_files):
                if os.path.basename(p) == target_fname:
                    self.current_idx = idx
                    break
        else:
            self.current_idx = (self.current_idx + 1) % len(self.image_files)

    def toggle_view_mode(self):
        """切换 3D 双四棱柱对比模式 与 2D 角点残差矢量模式"""
        self.view_mode_3d = not self.view_mode_3d
        mode_str = "【3D双棱柱空间对比】(BA真值 vs 实测)" if self.view_mode_3d else "【2D角点残差矢量】"
        self.set_toast(f"视图切换: {mode_str}")

    def set_flagged_filter(self, only_flagged: bool):
        """显式设置是否仅浏览回审帧"""
        self.is_filter_dropdown_open = False
        if only_flagged:
            if not self.flagged_frames:
                self.set_toast("当前无任何建议回审帧，无需过滤！")
                self.filter_flagged_only = False
                return
            self.filter_flagged_only = True
            target_fname = self.flagged_frames[0]
            for idx, p in enumerate(self.image_files):
                if os.path.basename(p) == target_fname:
                    self.current_idx = idx
                    break
            self.set_toast(f"已开启【仅回审模式】(共 {len(self.flagged_frames)} 帧)")
        else:
            self.filter_flagged_only = False
            self.set_toast(f"已恢复【全量浏览模式】(共 {len(self.image_files)} 帧)")

    def toggle_flagged_filter(self):
        """切换是否仅浏览建议回审帧"""
        self.is_filter_dropdown_open = False
        self.set_flagged_filter(not self.filter_flagged_only)

    def open_reviewer_modal(self):
        """模态呼出审核画板，定向跳转到当前帧，退出后自动原地热重载"""
        if not self.image_files:
            return
        cur_file = os.path.basename(self.image_files[self.current_idx])
        self.set_toast(f"正在唤起人工审核画板 (定向定位: {cur_file})...")

        # 先刷新一次画面把 Toast 渲染出去
        canvas = self._render_gui_canvas()
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(20)

        from tools.calibration.tag_manifest_reviewer import TagManifestReviewer
        try:
            cur_results = self.frame_results_cache.get(cur_file, [])
            focus_tid = None
            if cur_results:
                worst = max(cur_results, key=lambda r: r["err_px"])
                focus_tid = worst["blind_tag_id"]

            reviewer = TagManifestReviewer(
                manifest_path=self.manifest_path,
                focus_tag_id=focus_tid,
                initial_frame=cur_file
            )
            reviewer.run()

            # 审核画板关闭后，确保离线体检台窗口与鼠标事件绑定彻底满血恢复
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.win_w, self.win_h)
            cv2.setMouseCallback(self.window_name, self._on_mouse_gui)
            if force_window_focus:
                force_window_focus(self.window_name)

            if reviewer.trigger_verify_and_ba:
                self.set_toast("画板请求立即求解 BA 平差，正在就地执行...")
                self.run_in_place_bundle_adjustment()
            elif getattr(reviewer, "has_modified_manifest", False) or reviewer.has_unsaved_changes:
                self.set_toast("已同步画板修改，正在重新计算盲测指标...")
                self.recompute_all()
            else:
                self.set_toast("已从审核画板返回离线体检工作台")
        except Exception as e:
            self.set_toast(f"唤起审核画板失败: {e}")

    def run_in_place_bundle_adjustment(self):
        """就地重新执行全局 BA 平差优化，并热重载地图"""
        from tools.calibration.tag_map_builder import TagMapBuilder
        self.set_toast("正在后台重新执行两阶段 BA 全局平差优化，请稍候...")
        canvas = self._render_gui_canvas()
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(20)

        try:
            builder = TagMapBuilder(marker_size_mm=self.marker_size_mm)
            map_data = builder.build_map_from_manifest(manifest_path=self.manifest_path)
            if map_data:
                builder.save_map(map_data, self.map_path)
                self._load_tags_map()
                self.recompute_all()
                self.set_toast("BA 平差优化完成！新地图已原地热重载，体检指标已就地刷新！")
            else:
                self.set_toast("BA 平差未收敛，请在画板中检查连通拓扑！")
        except Exception as e:
            self.set_toast(f"BA 平差执行异常: {e}")

    def handover_to_ar_verifier(self):
        """门限准入与优雅移交至在线 AR 验证"""
        med_mm = self.stats.get("global_median_mm", 999.0)
        med_px = self.stats.get("global_median_px", 999.0)
        flagged_tags = self.flagged_tags
        flagged_frames = self.flagged_frames

        is_passed = (med_mm <= 2.0 and med_px <= 3.0 and len(flagged_tags) == 0 and len(flagged_frames) == 0)

        now = time.time()
        if not is_passed and (now - self.last_ar_warn_time > 4.0):
            self.last_ar_warn_time = now
            self.set_toast(f"[门限阻断] 中位偏差 {med_mm:.2f}mm/偏差Tag:{flagged_tags}/回审帧:{len(flagged_frames)}！再次点击可强制放行")
            return

        self.set_toast("正在保存体检报告并移交至在线 AR 验证系统...")
        canvas = self._render_gui_canvas()
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(20)

        self._export_markdown_report(self.stats, self.all_results)

        # 若是由 AR 验证器反向唤起的子体检台，保存报告后直接优雅退出，把控制权还给 AR 验证器
        if self.caller_ar_instance is not None:
            print("[*] [HANDSHAKE] 体检完成/已放行，正在返回在线 AR 验证器...")
            self.is_running_gui = False
            return

        cv2.destroyAllWindows()

        from tools.calibration.tag_calibration_verifier import TagCalibrationVerifier
        try:
            verifier = TagCalibrationVerifier(map_path=self.map_path)
            verifier.run()
        except Exception as e:
            print(f"[ERROR] 在线 AR 验证器异常: {e}")

        # 从 AR 验证器返回后，重新恢复体检 GUI 窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_w, self.win_h)
        cv2.setMouseCallback(self.window_name, self._on_mouse_gui)
        if force_window_focus:
            force_window_focus(self.window_name)
        self.set_toast("已从在线 AR 验证器返回离线体检工作台")

    def export_report_action(self):
        """导出 Markdown 精度体检报告"""
        rep = self._export_markdown_report(self.stats, self.all_results)
        self.set_toast(f"已导出体检报告: {os.path.basename(rep)}")

    def _render_gui_canvas(self) -> np.ndarray:
        """渲染完整 GUI 界面：底图等比贴入视口 + 顶部 HUD + 底部工具栏 + 悬浮 Toast"""
        w, h = self.win_w, self.win_h
        if not self.image_files:
            empty = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(empty, "No Calibration Images Found", (w // 2 - 180, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return empty

        cur_path = self.image_files[self.current_idx]
        cur_fname = os.path.basename(cur_path)
        cur_results = self.frame_results_cache.get(cur_fname, [])

        # 1. 渲染原始全分辨率图像画布 (1920x1080 原生高清角点、盲测反推框与残差标牌)
        content_frame = self._render_verification_frame(cur_path, cur_results)

        # 2. 创建窗口画布并利用 ViewportManager 将原图等比无畸变贴入中间视口
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if self.viewport:
            self.viewport.render_viewport(canvas, content_frame)
            top_bar_h = self.viewport.top_bar_h
            bot_bar_h = self.viewport.bottom_bar_h
        else:
            top_bar_h = 62
            bot_bar_h = 58
            canvas = cv2.resize(content_frame, (w, h))

        # 3. 绘制顶部 HUD 仪表盘 (1:1 独立物理像素渲染，绝不缩水发糊)
        top_overlay = canvas.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, top_bar_h), (18, 18, 22), -1)
        cv2.addWeighted(top_overlay, 0.90, canvas, 0.10, 0, canvas)
        cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (60, 65, 75), 1)

        # 帧信息与单帧状态
        is_cur_flagged = cur_fname in self.flagged_frames
        if cur_results:
            cur_mean_px = np.mean([r["err_px"] for r in cur_results])
            cur_max_px = np.max([r["err_px"] for r in cur_results])
            worst_tid = max(cur_results, key=lambda r: r["err_px"])["blind_tag_id"]
        else:
            cur_mean_px, cur_max_px, worst_tid = 0.0, 0.0, -1

        if is_cur_flagged:
            conclusion_str = "结论: 需回审"
            conclusion_color = (0, 90, 255)   # 醒目警示橙红
        elif cur_mean_px <= 1.5:
            conclusion_str = "结论: PASS"
            conclusion_color = (0, 240, 120)  # 达标翠绿
        else:
            conclusion_str = "结论: 需复核"
            conclusion_color = (0, 190, 245)  # 预警亮黄

        # 初始化按钮列表 (涵盖顶部整套/全局操作栏与底部当前单帧控制栏)
        self.gui_buttons = []

        # 计算全局准入门限与统计指标
        med_mm = self.stats.get("global_median_mm", 0.0)
        med_px = self.stats.get("global_median_px", 0.0)
        gate_ok = (med_mm <= 2.0 and med_px <= 3.0 and len(self.flagged_tags) == 0 and len(self.flagged_frames) == 0)
        gate_tag = "[PASS]" if gate_ok else "[FAIL]"
        gate_label = f"{gate_tag} 准入门限: 达标 (允许进入AR)" if gate_ok else f"{gate_tag} 准入门限: 超标 (需审核/重平差)"
        gate_color = (0, 240, 120) if gate_ok else (0, 100, 255)

        # -------------------------------------------------------------------------
        # 3. 顶部 HUD: 整套样本 / 全局业务操作栏 (全量/回审过滤、BA平差、在线验证、导出报告、关闭)
        # -------------------------------------------------------------------------
        top_by1 = 12
        top_by2 = top_bar_h - 12
        top_bx = 12

        # A. 全局浏览模式下拉选择框 (Drop-down ComboBox，更加紧凑现代)
        flagged_cnt = len(self.flagged_frames)
        total_cnt = len(self.image_files)
        filter_opts = [
            ("ALL", f"全量浏览 ({total_cnt}帧)"),
            ("FLAGGED", f"仅回审 ({flagged_cnt}帧)")
        ]
        active_key = "FLAGGED" if self.filter_flagged_only else "ALL"
        active_c = (35, 45, 175) if self.filter_flagged_only else (150, 95, 20)
        dd_w = 175
        dd_rect = (top_bx, top_by1, top_bx + dd_w, top_by2)
        if draw_dropdown_box:
            dd_items, _ = draw_dropdown_box(
                canvas, dd_rect, filter_opts, active_key,
                is_open=self.is_filter_dropdown_open,
                mouse_pos=self.mouse_pos,
                shortcut="F",
                active_color=active_c
            )
            for key, irect in dd_items:
                self.gui_buttons.append((key, irect, key, (255, 255, 255)))
        else:
            self.gui_buttons.append(("DROPDOWN_TOGGLE", dd_rect, "TOGGLE", (255, 255, 255)))
        top_bx += dd_w + 8

        # B. 全局业务操作按钮：[BA平差 (B)]、[在线验证 (V)]、[导出报告 (S)]
        global_action_btns = [
            ("RUN_BA", "[BA平差 (B)]", 108, "primary"),
            ("ENTER_AR", "[在线验证 (V)]", 120, "success" if gate_ok else "normal"),
            ("EXPORT", "[导出报告 (S)]", 116, "normal"),
        ]
        for btn_id, label, bw, btype in global_action_btns:
            btn_rect = (top_bx, top_by1, top_bx + bw, top_by2)
            if draw_styled_button:
                draw_styled_button(canvas, btn_rect, label, self.mouse_pos, btn_type=btype)
            self.gui_buttons.append((btn_id, btn_rect, label, (255, 255, 255)))
            top_bx += bw + 6

        # C. 右上角关闭按钮 [关闭 (Q)]
        top_exit_rect = (w - 105, 12, w - 12, top_bar_h - 12)
        if draw_styled_button:
            draw_styled_button(canvas, top_exit_rect, "[关闭 (Q)]", self.mouse_pos, btn_type="danger")
        self.gui_buttons.append(("EXIT", top_exit_rect, "退出", (0, 0, 255)))

        # D. 右上角仪表盘指标 (位于关闭按钮左侧)
        dash_rx = w - 120
        cv2.putText(canvas, gate_label, (dash_rx - 340, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, gate_color, 2, cv2.LINE_AA)
        glob_str = f"全局中位: {med_mm:.2f}mm ({med_px:.2f}px) | 嫌疑Tag: {self.flagged_tags or '无'} | 回审: {len(self.flagged_frames)} 帧"
        cv2.putText(canvas, glob_str, (dash_rx - 450, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # -------------------------------------------------------------------------
        # 4. 绘制底部控制栏: 当前单帧专属操作栏 (上张、下张、3D/2D残差、审核此帧、单帧指标)
        # -------------------------------------------------------------------------
        bot_overlay = canvas.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_bar_h), (w, h), (20, 22, 28), -1)
        cv2.addWeighted(bot_overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.line(canvas, (0, h - bot_bar_h), (w, h), (50, 54, 66), 1)

        bot_by1 = h - bot_bar_h + 8
        bot_by2 = h - 8
        bot_bx = 12

        # A. 翻页按钮组 (A / D)
        for btn_id, label, bw in [("PREV", "< 上张 (A)", 96), ("NEXT", "下张 (D) >", 96)]:
            btn_rect = (bot_bx, bot_by1, bot_bx + bw, bot_by2)
            if draw_styled_button:
                draw_styled_button(canvas, btn_rect, label, self.mouse_pos, btn_type="normal")
            self.gui_buttons.append((btn_id, btn_rect, label, (255, 255, 255)))
            bot_bx += bw + 6

        # B. 3D双棱柱 / 2D残差 乒乓切换
        btn_label_3d = "[3D双棱柱(T)]" if self.view_mode_3d else "[2D残差(T)]"
        btn_type_3d = "info" if self.view_mode_3d else "normal"
        rect_3d = (bot_bx, bot_by1, bot_bx + 125, bot_by2)
        if draw_styled_button:
            draw_styled_button(canvas, rect_3d, btn_label_3d, self.mouse_pos, btn_type=btn_type_3d)
        self.gui_buttons.append(("TOGGLE_3D", rect_3d, btn_label_3d, (255, 255, 255)))
        bot_bx += 125 + 6

        # C. 审核此帧 (R)
        rect_rev = (bot_bx, bot_by1, bot_bx + 120, bot_by2)
        if draw_styled_button:
            draw_styled_button(canvas, rect_rev, "[审核此帧 (R)]", self.mouse_pos, btn_type="purple")
        self.gui_buttons.append(("OPEN_REVIEWER", rect_rev, "[审核此帧 (R)]", (255, 255, 255)))
        bot_bx += 120 + 12

        # D. 右侧展示当前单帧指标与状态结论 (背景与底栏/文件名一致，彻底消除按钮视觉误认)
        # 1) 右侧靠边以醒目纯文本呈现结论 (无按钮外框与实心底板)
        (cw, ch), _ = cv2.getTextSize(conclusion_str, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
        concl_x = w - cw - 16
        concl_y = bot_by1 + (bot_by2 - bot_by1 + ch) // 2
        cv2.putText(canvas, conclusion_str, (concl_x, concl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, conclusion_color, 2, cv2.LINE_AA)

        # 2) 结论左侧展示当前帧索引、文件名与误差指标明细
        info_rx = concl_x - 18
        idx_str = f"[{self.current_idx + 1:02d}/{len(self.image_files):02d}] {cur_fname}"
        detail_str = f"帧均: {cur_mean_px:.2f}px | 最大: {cur_max_px:.2f}px (Tag#{worst_tid}) | 盲测: {len(cur_results)} 靶"
        (iw, _), _ = cv2.getTextSize(idx_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        (dw, _), _ = cv2.getTextSize(detail_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        content_w = max(iw, dw)
        info_tx = max(bot_bx + 12, info_rx - content_w)

        cv2.putText(canvas, idx_str, (info_tx, bot_by1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 235, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, detail_str, (info_tx, bot_by1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)

        # 5. 浮动 Toast 气泡
        if self.toast_msg and (time.time() - self.toast_time < 3.5):
            tw_box, th_box = min(580, w - 80), 38
            tx1 = (w - tw_box) // 2
            ty1 = h - bot_bar_h - 52
            toast_overlay = canvas.copy()
            cv2.rectangle(toast_overlay, (tx1, ty1), (tx1 + tw_box, ty1 + th_box), (15, 15, 15), -1)
            cv2.addWeighted(toast_overlay, 0.85, canvas, 0.15, 0, canvas)
            cv2.rectangle(canvas, (tx1, ty1), (tx1 + tw_box, ty1 + th_box), (0, 220, 255), 1)
            cv2.putText(canvas, self.toast_msg, (tx1 + 16, ty1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

        return canvas

    def run_gui(self):
        """启动离线标定精度体检 GUI 交互工作台"""
        print("\n[*] 正在启动离线标定精度体检 GUI 交互工作台...")
        self.recompute_all()

        if not self.image_files:
            print("[WARN] 未在采图目录中检测到有效图像文件！")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_w, self.win_h)
        cv2.setMouseCallback(self.window_name, self._on_mouse_gui)

        if force_window_focus:
            force_window_focus(self.window_name)

        self.is_running_gui = True
        self.set_toast("已载入离线体检台 [T 切换3D棱柱对比 | 滚轮缩放/中键漫游 | 0复位 | R审核 | B重平差]")

        while self.is_running_gui:
            # 实时动态感知用户拖拽 Resize 或点击最大化窗口后的实际物理尺寸
            if self.viewport and self.viewport.sync_window_size(self.window_name):
                self.win_w = self.viewport.win_w
                self.win_h = self.viewport.win_h

            canvas = self._render_gui_canvas()
            cv2.imshow(self.window_name, canvas)

            if not self.is_running_gui:
                break

            key = cv2.waitKey(25) & 0xFF

            # 状态退出判定与窗口右上角系统关闭按钮检测 (WND_PROP_VISIBLE < 1)
            if not self.is_running_gui:
                break
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            if key in (ord('a'), ord('A'), 81):  # A / Left
                self.prev_image()
            elif key in (ord('d'), ord('D'), 83):  # D / Right
                self.next_image()
            elif key in (ord('f'), ord('F')):  # F
                self.toggle_flagged_filter()
            elif key in (ord('t'), ord('T')):  # T 切换 3D 棱柱 / 2D 残差
                self.toggle_view_mode()
            elif key in (ord('r'), ord('R')):  # R
                self.open_reviewer_modal()
            elif key in (ord('b'), ord('B')):  # B
                self.run_in_place_bundle_adjustment()
            elif key in (ord('v'), ord('V')):  # V
                self.handover_to_ar_verifier()
            elif key in (ord('s'), ord('S')):  # S
                self.export_report_action()
            elif key in (ord('0'), ord('z'), ord('Z')):  # 0 / Z 复位视口
                if self.viewport and self.viewport.reset_zoom():
                    self.set_toast("视口已复位 (1.0x)")
            elif key == 27:  # ESC: 若下拉菜单展开则优先收起，否则退出工作台
                if self.is_filter_dropdown_open:
                    self.is_filter_dropdown_open = False
                    continue
                break
            elif key in (ord('q'), ord('Q')):  # Q 退出
                break

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        print("[EXIT] 离线精度体检 GUI 工作台已安全退出。")


def main():
    parser = argparse.ArgumentParser(description="AprilTag 离线标定精度体检与 Leave-One-Out 盲测批量验证 GUI 工作台")
    parser.add_argument("--map", type=str, default=DEFAULT_MAP_PATH, help="tags_map.yaml 路径")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="采集样本图像目录")
    parser.add_argument("--marker_size", type=float, default=50.0, help="标靶物理边长 (mm)")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "manifest", "images"],
                        help="数据源: auto (优先清单), manifest (仅已审核清单), images (原图重检测)")
    parser.add_argument("--headless", action="store_true", help="无头批处理模式 (仅输出报告与图片，不弹 GUI 窗口)")
    args = parser.parse_args()

    verifier = TagOfflineVerifier(
        map_path=args.map,
        image_dir=args.image_dir,
        marker_size_mm=args.marker_size,
        source=args.source
    )
    if args.headless:
        verifier.run_full_verification()
    else:
        verifier.run_gui()


if __name__ == "__main__":
    main()
