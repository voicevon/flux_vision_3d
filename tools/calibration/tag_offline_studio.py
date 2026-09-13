#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 离线标定与建图综合工作站 (Tag Offline Studio)
=====================================================
旗舰级资产中心化 (Asset-Centric) 离线标定工作台：
  - 100% 装配复用底层核心领域模型：
      - ManifestRepository (清单与地图仓储 / 剔除状态同步)
      - OfflineEngine (PnP 求解 / 重投影残差 / LOO 盲测)
      - BundleAdjustmentOptimizer (两阶段 Cauchy BA 全局平差)
      - VerificationReporter (残差分析 / MAD 粗差诊断 / 质检报告)
      - VerificationVisualizer (3D 双四棱柱立体对比 / 2D 残差矢量)
  - 1920x1080 工业级三栏排版 (左侧高密紧凑帧列表、中央高清视口、右侧属性诊断面板、底栏全局调度)
  - 异步 BA 全局平差，前台丝滑响应，求解完成后就地热重载地图并即时刷新全量帧残差数值。
"""

import os
import sys
import glob
import time
import math
import threading
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

# Windows 终端色彩与编码适配
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.manifest_repository import ManifestRepository
from src.calibration.offline_engine import OfflineVerificationEngine, OfflineEngine
from src.calibration.ba_optimizer import BundleAdjustmentOptimizer
from src.calibration.verification_reporter import VerificationReporter
from src.calibration.verification_visualizer import VerificationVisualizer
from tools.calibration.studio import (
    StudioDataManager,
    StudioViewportInteractor,
    StudioBARunner,
    StudioUIRenderer,
    draw_dropdown_button,
    VIEW_MODE_OPTIONS,
    FILTER_MODE_OPTIONS,
    SORT_MODE_OPTIONS,
    BA_VIEW_OPTIONS,
    OBS_VIEW_OPTIONS
)
from src.utils.viewport_manager import (
    ViewportManager,
    draw_styled_button,
    draw_segmented_toggle
)

try:
    from src.utils.window_helper import force_window_focus
except ImportError:
    force_window_focus = None


try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

CALIB_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")
DEFAULT_MAP_PATH = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
MANIFEST_PATH = os.path.join(CALIB_IMAGES_DIR, "tag_observations.yaml")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


class TagOfflineStudio:

    """AprilTag 离线标定综合工作站 (Offline Studio) 控制器"""

    def __init__(
        self,
        map_path: str = DEFAULT_MAP_PATH,
        image_dir: str = CALIB_IMAGES_DIR,
        marker_size_mm: float = 50.0,
        win_w: int = 1920,
        win_h: int = 1080
    ):
        self.map_path = map_path
        self.image_dir = image_dir
        self.marker_size_mm = marker_size_mm
        self.win_w = win_w
        self.win_h = win_h

        # 1. 初始化视口管理器与物理布局尺寸
        self.viewport = ViewportManager(win_w=win_w, win_h=win_h, top_bar_h=44, bottom_bar_h=52)
        self.left_bar_w = 340   # 左侧紧凑列表宽度
        self.right_bar_w = 180  # 右侧精简属性栏宽度 (瘦身至 180px，充分释放主视口空间)

        # 2. 相机内参与领域模型装配
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()
        self.manifest_path = MANIFEST_PATH
        self.manifest_repo = ManifestRepository()

        self.engine = OfflineVerificationEngine(
            tags_map={},
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm
        )
        self.visualizer = VerificationVisualizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        self.reporter = VerificationReporter()
        self.optimizer = BundleAdjustmentOptimizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm
        )

        # 3. 领域与数据状态管理器 (从主类中独立解耦)
        self.data_mgr = StudioDataManager(
            map_path=self.map_path,
            image_dir=self.image_dir,
            manifest_path=self.manifest_path,
            engine=self.engine,
            marker_size_mm=self.marker_size_mm
        )

        # 视口显示双独立正交模式 (用户指定默认 3d)
        self.ba_view_mode: str = "3d"    # BA 理论值: "3d" (翡翠绿棱柱), "2d" (投影框), "off" (隐藏)
        self.obs_view_mode: str = "3d"   # 实测识别值: "3d" (天蓝棱柱), "2d" (实测角点框), "off" (隐藏)
        self.view_mode: str = "3d"       # 兼容器

        # 下拉菜单展开态标识 ("FILTER_DROPDOWN", "SORT_DROPDOWN", "BA_VIEW_DROPDOWN", "OBS_VIEW_DROPDOWN" 或 None)
        self.active_dropdown: Optional[str] = None
        self.dropdown_boxes: Dict[str, Dict[str, Any]] = {}

        # 5. 异步 BA 全局平差任务调度器
        self.ba_runner = StudioBARunner(
            data_mgr=self.data_mgr,
            optimizer=self.optimizer,
            manifest_repo=self.manifest_repo,
            map_path=self.map_path,
            manifest_path=self.manifest_path,
            marker_size_mm=self.marker_size_mm,
            on_status_change=self.set_toast
        )

        # 6. 单帧深度病因切片诊断开关
        self.show_frame_diagnostics = False

        # 7. 浮层通知 (Toast)
        self.status_toast = "欢迎进入 Offline Studio 离线标定工作站"
        self.status_toast_time = time.time()

        # 8. GUI 交互按钮注册表
        self.gui_buttons: List[Tuple[str, Tuple[int, int, int, int], Any]] = []
        self.mouse_pos = (-1, -1)
        self.is_running = True

        # 9. 视口几何变换与鼠标交互控制器 (Viewport Zoom & Pan)
        self.viewport = StudioViewportInteractor(win_w=self.win_w, win_h=self.win_h)

        # 10. UI 界面排版与视觉渲染器
        self.ui_renderer = StudioUIRenderer()

        # 首次预热并计算全集残差指标
        self.refresh_all_frame_metrics()


    def reset_viewport_zoom(self):
        """重置中间视口缩放与平移状态为适应屏幕 (1.0x)"""
        self.viewport.reset()
        self.set_toast("视口缩放已重置 (1.0x 适应视口)")

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()

    def _load_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载相机内参和畸变参数"""
        if resolve_camera_intrinsics is not None:
            K, dist, _ = resolve_camera_intrinsics(CONFIG_PATH)
            return K, dist
        else:
            K = np.array([
                [1363.68, 0.0, 971.19],
                [0.0, 1361.19, 566.26],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            return K, dist

    # ================= 状态与数据管理属性代理 (透明转发至 data_mgr) =================
    @property
    def image_files(self) -> List[str]:
        return self.data_mgr.image_files

    @image_files.setter
    def image_files(self, val: List[str]):
        self.data_mgr.image_files = val

    @property
    def current_img_idx(self) -> int:
        return self.data_mgr.current_img_idx

    @current_img_idx.setter
    def current_img_idx(self, val: int):
        self.data_mgr.current_img_idx = val

    @property
    def scroll_offset(self) -> int:
        return self.data_mgr.scroll_offset

    @scroll_offset.setter
    def scroll_offset(self, val: int):
        self.data_mgr.scroll_offset = val

    @property
    def filter_mode(self) -> str:
        return self.data_mgr.filter_mode

    @filter_mode.setter
    def filter_mode(self, val: str):
        self.data_mgr.filter_mode = val

    @property
    def sort_mode(self) -> str:
        return self.data_mgr.sort_mode

    @sort_mode.setter
    def sort_mode(self, val: str):
        self.data_mgr.sort_mode = val

    @property
    def manifest_data(self) -> Dict[str, Any]:
        return self.data_mgr.manifest_data

    @manifest_data.setter
    def manifest_data(self, val: Dict[str, Any]):
        self.data_mgr.manifest_data = val

    @property
    def tags_map_data(self) -> Dict[str, Any]:
        return self.data_mgr.tags_map_data

    @tags_map_data.setter
    def tags_map_data(self, val: Dict[str, Any]):
        self.data_mgr.tags_map_data = val

    @property
    def frame_metrics_cache(self) -> Dict[str, Dict[str, Any]]:
        return self.data_mgr.frame_metrics_cache

    @frame_metrics_cache.setter
    def frame_metrics_cache(self, val: Dict[str, Dict[str, Any]]):
        self.data_mgr.frame_metrics_cache = val

    # ================= 视口几何交互属性代理 (透明转发至 viewport) =================
    @property
    def zoom_level(self) -> float:
        return self.viewport.zoom_level

    @zoom_level.setter
    def zoom_level(self, val: float):
        self.viewport.zoom_level = val

    @property
    def pan_offset_x(self) -> float:
        return self.viewport.pan_offset_x

    @pan_offset_x.setter
    def pan_offset_x(self, val: float):
        self.viewport.pan_offset_x = val

    @property
    def pan_offset_y(self) -> float:
        return self.viewport.pan_offset_y

    @pan_offset_y.setter
    def pan_offset_y(self, val: float):
        self.viewport.pan_offset_y = val

    @property
    def is_panning(self) -> bool:
        return self.viewport.is_panning

    @is_panning.setter
    def is_panning(self, val: bool):
        self.viewport.is_panning = val

    @property
    def pan_start_pos(self) -> Tuple[int, int]:
        return self.viewport.pan_start_pos

    @pan_start_pos.setter
    def pan_start_pos(self, val: Tuple[int, int]):
        self.viewport.pan_start_pos = val

    @property
    def global_rmse(self) -> float:
        return self.data_mgr.global_rmse

    @global_rmse.setter
    def global_rmse(self, val: float):
        self.data_mgr.global_rmse = val

    # ================= 异步 BA 平差属性代理 (透明转发至 ba_runner) =================
    @property
    def is_ba_running(self) -> bool:
        return self.ba_runner.is_ba_running

    @is_ba_running.setter
    def is_ba_running(self, val: bool):
        self.ba_runner.is_ba_running = val

    @property
    def ba_thread(self) -> Optional[threading.Thread]:
        return self.ba_runner.ba_thread

    @ba_thread.setter
    def ba_thread(self, val: Optional[threading.Thread]):
        self.ba_runner.ba_thread = val

    @property
    def ba_result_queue(self) -> Optional[Tuple[bool, str]]:
        return self.ba_runner.ba_result_queue

    @ba_result_queue.setter
    def ba_result_queue(self, val: Optional[Tuple[bool, str]]):
        self.ba_runner.ba_result_queue = val

    @property
    def ba_progress(self) -> float:
        return self.ba_runner.ba_progress

    @ba_progress.setter
    def ba_progress(self, val: float):
        self.ba_runner.ba_progress = val

    @property
    def ba_stage_text(self) -> str:
        return self.ba_runner.ba_stage_text

    @ba_stage_text.setter
    def ba_stage_text(self, val: str):
        self.ba_runner.ba_stage_text = val

    @property
    def ba_sub_progress(self) -> float:
        return self.ba_runner.ba_sub_progress

    @ba_sub_progress.setter
    def ba_sub_progress(self, val: float):
        self.ba_runner.ba_sub_progress = val

    @property
    def ba_sub_text(self) -> str:
        return self.ba_runner.ba_sub_text

    @ba_sub_text.setter
    def ba_sub_text(self, val: str):
        self.ba_runner.ba_sub_text = val

    # ================= 领域数据方法委托 =================
    def _save_manifest(self):
        self.data_mgr._save_manifest()

    def _load_tags_map(self):
        self.data_mgr._load_tags_map()

    def is_image_excluded(self, base_name: str) -> bool:
        return self.data_mgr.is_image_excluded(base_name)

    def toggle_image_exclusion(self, base_name: str) -> bool:
        return self.data_mgr.toggle_image_exclusion(base_name)

    def toggle_observation_keep(self, base_name: str, target_tag_id: int) -> bool:
        return self.data_mgr.toggle_observation_keep(base_name, target_tag_id)

    def get_observations_for_image(self, base_name: str) -> List[Dict[str, Any]]:
        return self.data_mgr.get_observations_for_image(base_name)

    def get_tag_world_corners(self, tag_id: int) -> Optional[np.ndarray]:
        return self.data_mgr.get_tag_world_corners(tag_id)

    def get_tag_transform(self, tag_id: int) -> Optional[np.ndarray]:
        return self.data_mgr.get_tag_transform(tag_id)

    def refresh_all_frame_metrics(self):
        self.data_mgr.refresh_all_frame_metrics()

    def _evaluate_frame_reprojection(self, observations: List[Dict[str, Any]]):
        return self.data_mgr._evaluate_frame_reprojection(observations)

    def _get_filtered_indices(self) -> List[int]:
        return self.data_mgr.get_filtered_indices()

    def toggle_current_frame_exclusion(self):
        bname, is_excl = self.data_mgr.toggle_current_frame_exclusion()
        if bname:
            status_str = "已标记为 [剔除/EXCLUDED]" if is_excl else "已恢复为 [保留/ACTIVE]"
            self.set_toast(f"帧 {bname} {status_str}")
            print(f"[*] 帧状态翻转: {bname} -> {status_str}")

    def toggle_tag_exclusion_in_current_frame(self, target_tag_id: int):
        bname, is_kept = self.data_mgr.toggle_tag_exclusion_in_current_frame(target_tag_id)
        if bname:
            t_str = "已保留" if is_kept else "已剔除 (打叉)"
            self.set_toast(f"标靶 Tag #{target_tag_id} 在本帧中 {t_str}")

    def start_async_bundle_adjustment(self):
        """启动后台线程执行两阶段全局 BA 平差优化，前台持续平滑响应"""
        return self.ba_runner.start()



    def export_verification_report(self):
        """导出 Markdown 全景精度质检单"""
        report_dir = os.path.join(PROJECT_ROOT, "data", "tag_calibration_verification")
        os.makedirs(report_dir, exist_ok=True)
        ts = int(time.time())
        report_path = os.path.join(report_dir, f"studio_qa_report_{ts}.md")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# AprilTag 离线标定与建图全景质检单 (Offline Studio)\n\n")
                f.write(f"- **质检时间**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
                f.write(f"- **总采图集**: `{len(self.image_files)} 帧`\n")
                f.write(f"- **全景 RMSE**: `{self.global_rmse:.3f} px`\n")
                f.write(f"- **已知标靶数**: `{len(self.tags_map_data.get('tags', {}))} 个`\n")
                f.write(f"- **空间地图**: `{self.map_path}`\n\n")
                f.write(f"## 图像帧逐项质检明细\n\n")
                f.write(f"| 图像帧 | 观测标靶数 | 平均残差 | 最大残差 | 状态 |\n")
                f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")

                for p in self.image_files:
                    bname = os.path.basename(p)
                    meta = self.frame_metrics_cache.get(bname, {})
                    status_str = "❌ 已剔除" if meta.get("is_excluded", False) else "✅ 参与解算"
                    f.write(f"| `{bname}` | {meta.get('tag_count', 0)} | {meta.get('mean_err', 0.0):.2f} px | {meta.get('max_err', 0.0):.2f} px | {status_str} |\n")

            self.set_toast("全景质检报告已成功导出至 data/tag_calibration_verification/！")
            print(f"[OK] 质检报告导出成功: {report_path}")
        except Exception as e:
            self.set_toast(f"导出质检报告失败: {e}")
            print(f"[ERROR] 导出质检报告异常: {e}")

    # ===================== 渲染管线 (三栏自适应排版) =====================

    def render(self, canvas: np.ndarray):
        """完整渲染 Offline Studio 的顶栏、左栏列表、中间视口、右栏诊断与底栏"""
        self.ui_renderer.render(self, canvas)

    def _render_active_dropdown(self, canvas: np.ndarray):
        if self.active_dropdown and self.active_dropdown in self.dropdown_boxes:
            box = self.dropdown_boxes[self.active_dropdown]
            self.ui_renderer.render_dropdown_popup(self, canvas, self.active_dropdown, box["rect"], box["options"], box["active_key"])

    def _render_top_bar(self, canvas: np.ndarray, w: int, top_h: int):
        self.ui_renderer.render_top_bar(self, canvas, w, top_h)

    def _render_bottom_toolbar(self, canvas: np.ndarray, w: int, h: int, bot_h: int):
        self.ui_renderer.render_bottom_toolbar(self, canvas, w, h, bot_h)

    def _render_left_frame_list(self, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        self.ui_renderer.render_left_frame_list(self, canvas, x, y, w, h)

    def _render_center_viewport(self, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        self.ui_renderer.render_center_viewport(self, canvas, x, y, w, h)

    def _overlay_visual_elements(
        self,
        disp_frame: np.ndarray,
        observations: List[Dict[str, Any]],
        is_frame_excluded: bool,
        meta: Optional[Dict[str, Any]] = None
    ):
        self.ui_renderer.overlay_visual_elements(self, disp_frame, observations, is_frame_excluded, meta=meta)



    def _render_right_inspector(self, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        self.ui_renderer.render_right_inspector(self, canvas, x, y, w, h)

    def _render_ba_loading_card(self, canvas: np.ndarray, w: int, h: int):
        self.ui_renderer.render_ba_loading_card(self, canvas, w, h)

    def _render_toast(self, canvas: np.ndarray, w: int, h: int, bot_h: int):
        self.ui_renderer.render_toast(self, canvas, w, h, bot_h)

    # ===================== 事件分发与主循环 =====================

    def _on_mouse(self, event, mx, my, flags, param):
        self.mouse_pos = (mx, my)

        top_h = self.viewport.top_bar_h if self.viewport else 44
        bot_h = self.viewport.bottom_bar_h if self.viewport else 52
        content_y1 = top_h
        content_y2 = self.win_h - bot_h

        left_x1, left_x2 = 0, self.left_bar_w
        mid_x1, mid_x2 = self.left_bar_w, self.win_w - self.right_bar_w

        # 1. 鼠标滚轮事件 (精准区分：左侧列表滚动 vs 中间视口以鼠标为中心缩放)
        if event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮判定方向: flags > 0 为向上滚, flags < 0 为向下滚
            wheel_up = (flags > 0)

            # A. 鼠标光标位于左栏：上下滚动帧资产列表
            if left_x1 <= mx < left_x2:
                if wheel_up:
                    self.scroll_offset = max(0, self.scroll_offset - 2)
                else:
                    self.scroll_offset += 2
                return

            # B. 鼠标光标位于中间画布视口：执行以光标为中心的精准缩放 (Zoom In/Out)
            elif mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                self.viewport.zoom_at(mx, my, wheel_up, (mid_x1, content_y1, mid_x2 - mid_x1, content_y2 - content_y1))
                return

        # 2. 拖拽平移事件 (支持鼠标右键或中键按住平移)
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            if mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                self.viewport.start_pan(mx, my)
                return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.viewport.update_pan(mx, my):
                return
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            if self.viewport.is_panning:
                self.viewport.end_pan()
                return

        # 3. 双击事件 (双击左键或右键一键重置缩放)
        if event in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDBLCLK):
            if mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                self.reset_viewport_zoom()
                return

        # 4. 鼠标左键点击事件 (GUI 按钮分发，优先命中置顶下拉层)
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_any = False
            for btn_id, (bx1, by1, bx2, by2), extra in reversed(self.gui_buttons):
                if bx1 <= mx <= bx2 and by1 <= my <= by2:
                    self._handle_button_click(btn_id, extra, mx, my)
                    clicked_any = True
                    return

            # 若未点击任何已注册按钮，且当前有下拉菜单展开，则自动收起 (Click-outside)
            if not clicked_any and self.active_dropdown:
                self.active_dropdown = None
                return

            # 5. 检查是否直接点击在中间视口图片的标靶区域上 (画布直接打叉剔除 / 恢复审核模式)
            if mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                if self.image_files and 0 <= self.current_img_idx < len(self.image_files):
                    cur_file = self.image_files[self.current_img_idx]
                    bname = os.path.basename(cur_file)
                    meta = self.frame_metrics_cache.get(bname, {})
                    obs_list = meta.get("observations", [])

                    if obs_list:
                        bgr = cv2.imread(cur_file)
                        if bgr is not None:
                            frame_h, frame_w = bgr.shape[:2]
                            hit_tid = self.viewport.hit_test_tag(
                                mx, my, obs_list,
                                (mid_x1, content_y1, mid_x2 - mid_x1, content_y2 - content_y1),
                                frame_w, frame_h
                            )
                            if hit_tid is not None:
                                self.toggle_tag_exclusion_in_current_frame(hit_tid)
                                return


    def _handle_button_click(self, btn_id: str, extra: Any, mx: int, my: int):
        if btn_id == "EXIT":
            self.is_running = False
        elif btn_id == "RUN_BA":
            self.start_async_bundle_adjustment()
        elif btn_id == "RECOMPUTE_METRICS":
            self.refresh_all_frame_metrics()
            self.set_toast("已全量重算并刷新所有帧残差指标")
        elif btn_id == "EXPORT_REPORT":
            self.export_verification_report()
        elif btn_id == "SAVE_MAP":
            self.manifest_repo.save_tags_map(self.map_path, self.tags_map_data)
            self.set_toast(f"空间立体地图已成功保存至 {self.map_path}")
        elif btn_id == "TOGGLE_BA_VIEW_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "BA_VIEW_DROPDOWN" else "BA_VIEW_DROPDOWN"
        elif btn_id == "TOGGLE_OBS_VIEW_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "OBS_VIEW_DROPDOWN" else "OBS_VIEW_DROPDOWN"
        elif btn_id == "TOGGLE_VIEW_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "BA_VIEW_DROPDOWN" else "BA_VIEW_DROPDOWN"
        elif btn_id == "TOGGLE_FILTER_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "FILTER_DROPDOWN" else "FILTER_DROPDOWN"
        elif btn_id == "TOGGLE_SORT_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "SORT_DROPDOWN" else "SORT_DROPDOWN"
        elif btn_id.startswith("DD_SELECT_"):
            dd_name, selected_val = extra
            if dd_name == "BA_VIEW_DROPDOWN":
                self.ba_view_mode = selected_val
                self.view_mode = selected_val
                lbl = dict(BA_VIEW_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"BA 理论显示已切换为: {lbl}")
            elif dd_name == "OBS_VIEW_DROPDOWN":
                self.obs_view_mode = selected_val
                lbl = dict(OBS_VIEW_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"实测识别显示已切换为: {lbl}")
            elif dd_name == "VIEW_DROPDOWN":
                self.view_mode = selected_val
                if selected_val == "3d":
                    self.ba_view_mode = "3d"
                    self.obs_view_mode = "3d"
                elif selected_val == "2d":
                    self.ba_view_mode = "2d"
                    self.obs_view_mode = "2d"
                lbl = dict(VIEW_MODE_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"显示模式已切换为: {lbl}")
            elif dd_name == "FILTER_DROPDOWN":
                self.filter_mode = selected_val
                self.scroll_offset = 0
                lbl = dict(FILTER_MODE_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"筛选模式已切换为: {lbl}")
            elif dd_name == "SORT_DROPDOWN":
                self.sort_mode = selected_val
                self.scroll_offset = 0
                lbl = dict(SORT_MODE_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"排序方式已切换为: {lbl}")
            self.active_dropdown = None
        elif btn_id.startswith("SELECT_FRAME_"):
            orig_idx = int(extra)
            self.current_img_idx = orig_idx
            self.set_toast(f"已选中帧: {os.path.basename(self.image_files[orig_idx])}")
            self.active_dropdown = None
        elif btn_id == "TOGGLE_FRAME_STATUS":
            self.toggle_current_frame_exclusion()
        elif btn_id.startswith("TOGGLE_TAG_"):
            tid = int(extra)
            self.toggle_tag_exclusion_in_current_frame(tid)

        elif btn_id == "SUPER_EXTRACT_FRAME":
            self.set_toast("正在对当前单帧执行超精重提取...")
            # 单帧重提取
            if self.image_files:
                cur_file = self.image_files[self.current_img_idx]
                bname = os.path.basename(cur_file)
                bgr = cv2.imread(cur_file)
                if bgr is not None:
                    tags_dict = self.engine.detect_tags(bgr)
                    obs = []
                    for tid, c in tags_dict.items():
                        obs.append({
                            "tag_id": int(tid),
                            "corners": c.reshape((4, 2)).tolist(),
                            "keep": True
                        })
                    self.manifest_data.setdefault("images", {})[bname] = {"observations": obs}
                    self._save_manifest()
                    self.refresh_all_frame_metrics()
                    self.set_toast(f"超精重提取完成: 识别到 {len(obs)} 个标靶")
        elif btn_id == "DIAGNOSE_FRAME":
            self.set_toast("已启动当前帧切片漏检病因诊断 (请查看终端输出)")
            print(f"\n[*] [STUDIO DIAGNOSTICS] 正在切片诊断当前选定帧: {os.path.basename(self.image_files[self.current_img_idx])}")

    def run(self):
        """进入 Studio 主交互渲染循环"""
        window_name = "AprilTag Offline Studio (Integrated Edition)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.win_w, self.win_h)
        cv2.setMouseCallback(window_name, self._on_mouse)

        if force_window_focus:
            force_window_focus(window_name)

        print("\n" + "=" * 80)
        print("         AprilTag 离线标定与空间建图综合工作站 (Offline Studio)")
        print("=" * 80)
        print(f" [采图资产目录] : {self.image_dir} (共 {len(self.image_files)} 帧)")
        print(f" [空间立体地图] : {self.map_path}")
        print(" [工作流指南]   :")
        print("   - [↑] / [↓] 或 [W] / [S] : 上下顺序切换当前选定的图像帧")
        print("   - [V]                    : 循环切换视口模式 (混合 ⇋ 3D双棱柱 ⇋ 2D残差矢量)")
        print("   - [滚轮 (中间画布)]      : 以鼠标为中心实时精准放大/缩小图像 (0.4x ~ 15.0x)")
        print("   - [右键/中键拖拽]        : 在中间画布中自由平移浏览图像细节")
        print("   - [滚轮 (左侧栏)]        : 上下滚动浏览帧序列列表")
        print("   - [双击画布] / [Z] / [0] : 一键重置图像缩放和平移为适应视口 (1.0x)")
        print("   - [T] / [Space]          : 翻转当前帧有效性状态 (保留 ⇋ 剔除)")
        print("   - [B]                    : 异步执行全局平差优化 (BA) 并就地热重载")
        print("   - [P]                    : 全量重算并刷新所有帧精度体检残差指标")
        print("   - [R]                    : 导出离线全景精度体检 Markdown 质检单")
        print("   - [S]                    : 保存当前优化后的空间立体地图")
        print("   - [Q] / [ESC]            : 安全退出工作台返回控制台")
        print("=" * 80 + "\n")

        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        try:
            while self.is_running:
                # 视口自适应物理尺寸
                if self.viewport and self.viewport.sync_window_size(window_name):
                    self.win_w = self.viewport.win_w
                    self.win_h = self.viewport.win_h
                    canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

                # 渲染整帧
                self.render(canvas)

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('w'), ord('W'), 82):  # 上一帧 (W / Up)
                    if self.image_files:
                        self.current_img_idx = (self.current_img_idx - 1) % len(self.image_files)
                        self.set_toast(f"选定帧: {os.path.basename(self.image_files[self.current_img_idx])}")
                elif key in (ord('s'), ord('S'), 84):  # 下一帧 (S / Down)
                    if self.image_files:
                        self.current_img_idx = (self.current_img_idx + 1) % len(self.image_files)
                        self.set_toast(f"选定帧: {os.path.basename(self.image_files[self.current_img_idx])}")
                elif key in (ord('v'), ord('V')):      # V 键 -> 循环切换视口预设模式
                    presets = [
                        ("3d", "3d", "全 3D 双棱柱空间对比 (BA 3D + 实测 3D)"),
                        ("2d", "2d", "全 2D 重投影与残差矢量 (BA 2D + 实测 2D)"),
                        ("off", "2d", "仅单帧实测识别角点框"),
                        ("3d", "off", "仅 BA 空间理论 3D 棱柱"),
                        ("off", "off", "纯净原始采图 (全隐藏)")
                    ]
                    curr_idx = -1
                    for idx, (b_m, o_m, _) in enumerate(presets):
                        if self.ba_view_mode == b_m and self.obs_view_mode == o_m:
                            curr_idx = idx
                            break
                    next_idx = (curr_idx + 1) % len(presets)
                    self.ba_view_mode, self.obs_view_mode, desc = presets[next_idx]
                    self.view_mode = self.ba_view_mode
                    self.set_toast(f"视口模式: {desc}")
                elif key in (ord('z'), ord('Z'), ord('0')):  # Z / 0 键 -> 重置缩放
                    self.reset_viewport_zoom()
                elif key in (ord('t'), ord('T'), 32):  # T 键或空格键 -> 翻转状态
                    self.toggle_current_frame_exclusion()
                elif key in (ord('b'), ord('B')):      # B 键 -> 一键 BA
                    self.start_async_bundle_adjustment()
                elif key in (ord('p'), ord('P')):      # P 键 -> 重算体检
                    self.refresh_all_frame_metrics()
                    self.set_toast("已全量重算体检指标")
                elif key in (ord('r'), ord('R')):      # R 键 -> 导出报告
                    self.export_verification_report()
                elif key in (ord('s'), ord('S')):      # S 键 -> 保存地图
                    ManifestRepository.save_map(self.tags_map_data, self.map_path)
                    self.set_toast("空间立体地图已保存！")




        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AprilTag 离线标定与空间建图综合工作站 (Offline Studio)")
    parser.add_argument("--map", type=str, default=DEFAULT_MAP_PATH, help="标靶空间立体地图路径")
    parser.add_argument("--images", type=str, default=CALIB_IMAGES_DIR, help="标定采图目录")
    parser.add_argument("--marker_size", type=float, default=50.0, help="标靶物理边长 (mm)")
    args = parser.parse_args()

    studio = TagOfflineStudio(
        map_path=args.map,
        image_dir=args.images,
        marker_size_mm=args.marker_size
    )
    studio.run()


if __name__ == "__main__":
    main()
