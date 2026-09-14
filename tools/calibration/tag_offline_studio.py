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
        win_h: int = 1080,
        manifest_path: Optional[str] = None
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
        self.manifest_path = manifest_path or os.path.join(self.image_dir, "tag_observations.yaml")
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

        # 4. 文件列表多轮残差演进矩阵视图模式 (Matrix View)
        self.matrix_view_mode: bool = False

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

        # 11. 异步全量超精提取任务状态调度
        self.is_extracting_all: bool = False
        self.extract_progress: float = 0.0
        self.extract_stage_text: str = ""
        self.extract_thread: Optional[threading.Thread] = None
        self.extract_result_queue: Optional[Tuple[bool, str]] = None

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

    @property
    def global_median_mm(self) -> float:
        return self.data_mgr.global_median_mm

    @property
    def global_mean_mm(self) -> float:
        return self.data_mgr.global_mean_mm

    @property
    def gate_status(self) -> str:
        return self.data_mgr.gate_status

    @property
    def topology_status(self) -> Dict[str, Any]:
        return self.data_mgr.topology_status

    @property
    def current_diagnostics(self) -> Dict[str, Any]:
        return self.data_mgr.current_diagnostics

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
    def is_auto_pruning(self) -> bool:
        return self.ba_runner.is_auto_pruning

    @property
    def prune_settlement_data(self) -> Optional[Dict[str, Any]]:
        return self.ba_runner.prune_settlement_data

    @prune_settlement_data.setter
    def prune_settlement_data(self, val: Optional[Dict[str, Any]]):
        self.ba_runner.prune_settlement_data = val

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

    def super_extract_current_frame(self) -> Tuple[str, int]:
        """对当前选中帧执行工序 3 工业级超精重提取并持久化"""
        return self.data_mgr.super_extract_current_frame(self.current_img_idx)

    def super_extract_all_frames(self, progress_callback: Optional[Any] = None) -> Tuple[int, int]:
        """对所有采图帧清空原有角点与观测，从头重提取超精标靶并持久化"""
        return self.data_mgr.super_extract_all_frames(progress_callback=progress_callback)

    def start_async_super_extract_all(self) -> bool:
        """启动后台异步线程执行全量采图工序 3 工业级超精重提取并从头重建"""
        if self.is_extracting_all:
            self.set_toast("全量超精提取已在后台运行中，请稍候...")
            return False
        if self.is_ba_running:
            self.set_toast("全局平差计算中，请待平差完成后再执行提取")
            return False
        if not self.image_files:
            self.set_toast("未扫描到采图文件，无法执行超精重提取")
            return False

        self.is_extracting_all = True
        self.extract_progress = 0.01
        self.extract_stage_text = f"正在启动全局全量超精提取 (共 {len(self.image_files)} 帧)..."
        self.set_toast(self.extract_stage_text)

        def _worker():
            try:
                def on_progress(cur, total, bname, count):
                    self.extract_progress = cur / max(1, total)
                    self.extract_stage_text = f"全量超精提取 ({cur}/{total}): {bname} (检出 {count} 个标靶)"

                total_frames, total_tags = self.data_mgr.super_extract_all_frames(progress_callback=on_progress)
                self.extract_progress = 1.0
                msg = f"全局超精提取完成！处理 {total_frames} 帧，累计提取 {total_tags} 个高精标靶"
                self.extract_result_queue = (True, msg)
            except Exception as e:
                self.extract_result_queue = (False, f"全量超精重提取失败: {e}")

        self.extract_thread = threading.Thread(target=_worker, daemon=True)
        self.extract_thread.start()
        return True

    def poll_super_extract_result(self) -> Optional[Tuple[bool, str]]:
        """检查异步全量超精提取任务是否完成"""
        if self.extract_result_queue is not None:
            res = self.extract_result_queue
            self.extract_result_queue = None
            self.is_extracting_all = False
            return res
        return None

    @property
    def dynamic_left_bar_w(self) -> int:
        """根据当前是否处于矩阵视图动态计算左栏排版宽度"""
        if not self.matrix_view_mode:
            return 340
        headers = getattr(self.data_mgr, "convergence_headers", [])
        num_cols = max(1, len(headers))
        # 基础列宽 175px (状态点+文件名+标靶数) + 各轮次列 (56px/列) + 降幅列 (72px)
        calc_w = 175 + num_cols * 56 + 72
        return min(960, max(640, calc_w))

    def toggle_matrix_view_mode(self):
        """一键切换左栏文件列表的单列紧凑视图与多轮残差演进矩阵宽表大视图"""
        self.matrix_view_mode = not self.matrix_view_mode
        self.left_bar_w = self.dynamic_left_bar_w
        if self.matrix_view_mode:
            self.set_toast("已切换为: 逐帧多轮残差演进矩阵大表 (Matrix View)")
        else:
            self.set_toast("已切换为: 紧凑图像帧列表 (Compact View)")

    def start_auto_prune_ba(self) -> bool:
        """启动全自动基于边际收益与共视拓扑守门的残差剪枝平差"""
        if self.is_ba_running or self.is_extracting_all:
            self.set_toast("后台任务正在计算中，请稍候...")
            return False
        # 联动质检视角: 自动将左侧图像序列切换为【残差降序 (最差优先 ↓)】并展开多轮残差矩阵视图
        self.sort_mode = "err_desc"
        self.matrix_view_mode = True
        self.left_bar_w = self.dynamic_left_bar_w
        res = self.ba_runner.start_auto_prune(max_rounds=10, min_improvement_px=0.01)
        if res:
            # 自动将主视口聚焦至残差最大、最亟待排查的首张图像
            f_indices = self._get_filtered_indices()
            if f_indices:
                self.current_img_idx = f_indices[0]
        return res

    def accept_prune_results(self):
        """采纳智能剪枝平差结果并清空结算单"""
        self.ba_runner.prune_settlement_data = None
        self.set_toast("已采纳智能剪枝平差结果！可按 [M] 保存为最新地图")
        print("[*] [STUDIO] 操作员确认采纳智能剪枝平差结果。")

    def undo_prune_results(self):
        """一键无损撤销智能剪枝，回滚至快照状态"""
        succ = self.data_mgr.restore_manifest_snapshot()
        self.ba_runner.prune_settlement_data = None
        if succ:
            self.set_toast("已撤销智能剪枝！观测清单与地图已完全恢复至剪枝前状态")
            print("[*] [STUDIO] 操作员已撤销智能剪枝，状态已无损回滚。")
        else:
            self.set_toast("未找到有效快照，撤销未执行")

    def reset_map(self) -> bool:
        """一键复位清空空间立体地图 (自动备份为 tags_map.yaml.bak)"""
        return self.data_mgr.reset_map()

    def reset_all_keep_status(self) -> int:
        """一键复位全量观测保留状态"""
        return self.data_mgr.reset_all_keep_status()

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

                # 2. 智能剪枝平差逐帧多轮残差收敛矩阵 (若存在多轮历史)
                headers = getattr(self.data_mgr, "convergence_headers", [])
                matrix = getattr(self.data_mgr, "frame_convergence_matrix", {})
                if headers and matrix and len(headers) >= 1:
                    f.write(f"\n## 2. 智能剪枝平差逐帧多轮残差收敛矩阵 (Per-Frame Convergence Matrix)\n\n")
                    f.write(f"> 记录各图像帧在每一轮平差求解后的残差演进变化情况：\n\n")
                    header_cols = ["图像帧", "标靶数"] + headers + ["累计降幅"]
                    f.write("| " + " | ".join(header_cols) + " |\n")
                    f.write("| " + " | ".join([":---"] + [":---:"] * (len(header_cols) - 1)) + " |\n")

                    for p in self.image_files:
                        bname = os.path.basename(p)
                        meta = self.frame_metrics_cache.get(bname, {})
                        tag_cnt = meta.get("tag_count", 0)
                        row_vals = matrix.get(bname, [])
                        r_strs = []
                        for val in row_vals:
                            r_strs.append(f"{val:.2f} px" if val is not None else "--")
                        while len(r_strs) < len(headers):
                            r_strs.append("--")

                        first_val = row_vals[0] if (row_vals and row_vals[0] is not None) else None
                        last_val = None
                        for v in reversed(row_vals):
                            if v is not None:
                                last_val = v
                                break
                        if first_val is not None and last_val is not None and first_val > 0.001:
                            drop_px = first_val - last_val
                            drop_pct = (drop_px / first_val) * 100.0
                            drop_str = f"↓{drop_pct:.1f}% ({drop_px:+.2f}px)"
                        else:
                            drop_str = "--"

                        f.write(f"| `{bname}` | {tag_cnt} | " + " | ".join(r_strs) + f" | {drop_str} |\n")

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
            ManifestRepository.save_map(self.tags_map_data, self.map_path)
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
            self.set_toast("正在执行工序 3 工业级超精重提取 (多尺度CLAHE+2x超分+0.01px亚像素精修)...")
            bname, cnt = self.super_extract_current_frame()
            if bname:
                self.set_toast(f"帧 {bname} 超精重提取完成并已原子持久化: 检出 {cnt} 个标靶")
        elif btn_id == "SUPER_EXTRACT_ALL":
            self.start_async_super_extract_all()
        elif btn_id == "RESET_MAP":
            self.reset_map()
            self.set_toast("立体地图已复位清空 (备份为 .bak)，恢复为纯观测模式")
        elif btn_id == "RESET_KEEP_ALL":
            restored = self.reset_all_keep_status()
            self.set_toast(f"已一键复位所有观测有效状态 (恢复 {restored} 个标靶)")
        elif btn_id == "DIAGNOSE_FRAME":
            self.toggle_frame_diagnostics()
        elif btn_id == "LAUNCH_AR":
            self.launch_online_ar_verifier()
        elif btn_id == "RUN_AUTO_PRUNE_BA":
            self.start_auto_prune_ba()
        elif btn_id == "TOGGLE_MATRIX_VIEW":
            self.toggle_matrix_view_mode()
        elif btn_id == "ACCEPT_PRUNE":
            self.accept_prune_results()
        elif btn_id == "UNDO_PRUNE":
            self.undo_prune_results()
        elif btn_id == "STOP_PRUNE":
            self.ba_runner.request_stop_pruning()

    def toggle_frame_diagnostics(self):
        """唤起/关闭当前选定帧的漏检病因深度切片诊断视图"""
        self.show_frame_diagnostics = not self.show_frame_diagnostics
        if self.show_frame_diagnostics:
            diag = self.data_mgr.diagnose_frame(self.current_img_idx)
            bname = os.path.basename(self.image_files[self.current_img_idx])
            c_g = diag.get("contrast_grade", "")
            s_g = diag.get("sharpness_grade", "")
            rej_n = diag.get("rejected_quads_count", 0)
            miss_n = len(diag.get("missing_theoretical_tags", []))
            self.set_toast(f"[{bname}] 病因切片: 对比度 {c_g} | 清晰度 {s_g} | 拒检 {rej_n} | 理论漏检 {miss_n}")
            print("\n" + "=" * 70)
            print(f"[*] [STUDIO DIAGNOSTICS] 图像深度病因切片: {bname}")
            print(f"    - 对比度 (灰度标准差): {diag.get('contrast', 0.0):.1f} ({c_g})")
            print(f"    - 亮度均值: {diag.get('brightness', 0.0):.1f} ({diag.get('brightness_grade', '')})")
            print(f"    - 图像清晰度 (拉普拉斯梯度): {diag.get('sharpness', 0.0):.1f} ({s_g})")
            print(f"    - 算法被拒候选四边形: {rej_n} 个 (过小: {diag.get('rej_small', 0)}, 长宽失真: {diag.get('rej_aspect', 0)})")
            if diag.get("missing_theoretical_tags"):
                print(f"    - 视场理论可见但漏检的标靶: {[m['tag_id'] for m in diag['missing_theoretical_tags']]}")
            print("=" * 70 + "\n")
        else:
            self.set_toast("已退出病因切片诊断模式，返回常规视口")

    def launch_online_ar_verifier(self):
        """一键跨工序启动工序 7 在线 AR 姿态重投影验证器"""
        print("\n[*] [STUDIO] 正在启动工序 7 在线 AR 验证器 (tag_calibration_verifier.py)...")
        self.set_toast("正在启动工序 7 在线 AR 验证器...")
        import subprocess
        subprocess.Popen([sys.executable, "tools/calibration/tag_calibration_verifier.py"])

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
        print("   - [E]                    : 执行工序 3 工业级超精重提取 (5路增强+2x超分+0.01px精修)")
        print("   - [D]                    : 运行当前帧漏检病因切片诊断")
        print("   - [V]                    : 循环切换视口模式 (混合 ⇋ 3D双棱柱 ⇋ 2D残差矢量)")
        print("   - [滚轮 (中间画布)]      : 以鼠标为中心实时精准放大/缩小图像 (0.4x ~ 15.0x)")
        print("   - [右键/中键拖拽]        : 在中间画布中自由平移浏览图像细节")
        print("   - [滚轮 (左侧栏)]        : 上下滚动浏览帧序列列表")
        print("   - [双击画布] / [Z] / [0] : 一键重置图像缩放和平移为适应视口 (1.0x)")
        print("   - [T] / [Space]          : 翻转当前帧有效性状态 (保留 ⇋ 剔除)")
        print("   - [X]                    : 切换左栏视图 (紧凑列表 ⇋ 逐帧多轮残差演进矩阵宽表)")
        print("   - [B]                    : 异步执行全局平差优化 (全量批处理 Batch BA) 并就地热重载")
        print("   - [P]                    : 全量重算并刷新所有帧精度体检残差指标")
        print("   - [R]                    : 导出离线全景精度体检 Markdown 质检单")
        print("   - [M]                    : 保存当前优化后的空间立体地图")
        print("   - [Backspace] / [Delete] : 一键复位清空空间立体地图 (重置为未建图纯观测状态)")
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
                raw_key = cv2.waitKey(20)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if raw_key == -1:
                    continue
                key = raw_key & 0xFF

                # 优先拦截结算确认卡片按键交互
                if self.ba_runner.prune_settlement_data is not None:
                    if key in (10, 13):  # Enter 键 -> 采纳结果
                        self.accept_prune_results()
                        continue
                    elif key == 27:      # ESC 键 -> 撤销还原
                        self.undo_prune_results()
                        continue

                # 运行中支持空格急停
                if self.ba_runner.is_auto_pruning:
                    if key == 32:  # 空格键 -> 急停
                        self.ba_runner.request_stop_pruning()
                        continue

                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('a'), ord('A')):      # A 键 -> 智能迭代剪枝平差
                    self.start_auto_prune_ba()
                elif key in (ord('x'), ord('X')):      # X 键 -> 切换多轮残差矩阵视图
                    self.toggle_matrix_view_mode()
                elif key in (ord('w'), ord('W'), 82):  # 上一帧 (W / Up)
                    if self.image_files:
                        self.current_img_idx = (self.current_img_idx - 1) % len(self.image_files)
                        self.set_toast(f"选定帧: {os.path.basename(self.image_files[self.current_img_idx])}")
                        if self.show_frame_diagnostics:
                            self.data_mgr.diagnose_frame(self.current_img_idx)
                elif key in (ord('s'), ord('S'), 84):  # 下一帧 (S / Down)
                    if self.image_files:
                        self.current_img_idx = (self.current_img_idx + 1) % len(self.image_files)
                        self.set_toast(f"选定帧: {os.path.basename(self.image_files[self.current_img_idx])}")
                        if self.show_frame_diagnostics:
                            self.data_mgr.diagnose_frame(self.current_img_idx)
                elif key in (ord('e'), ord('E')):      # E 键 -> 单帧超精重提取
                    self.set_toast("正在执行单帧工业级超精重提取...")
                    bname, cnt = self.super_extract_current_frame()
                    if bname:
                        self.set_toast(f"帧 {bname} 超精提取完成并永久持久化: 检出 {cnt} 个标靶")
                elif key in (ord('d'), ord('D')):      # D 键 -> 漏检病因切片诊断
                    self.toggle_frame_diagnostics()
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
                elif key in (ord('m'), ord('M')):      # M 键 -> 保存地图
                    ManifestRepository.save_map(self.tags_map_data, self.map_path)
                    self.set_toast("空间立体地图已保存！")
                elif key in (8, 127):                  # Backspace 或 Delete (DEL) -> 一键复位地图
                    self.reset_map()
                    self.set_toast("立体地图已复位清空 (备份为 .bak)，恢复为纯观测模式")




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
