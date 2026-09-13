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

VIEW_MODE_OPTIONS = [
    ("mix", "混合透视 (2D+3D)"),
    ("3d", "3D 双四棱柱对比"),
    ("2d", "2D 识别框与残差矢量")
]

FILTER_MODE_OPTIONS = [
    ("all", "全部帧"),
    ("warning", "高残差 (>0.5px)"),
    ("excluded", "已剔除帧")
]

SORT_MODE_OPTIONS = [
    ("name_asc", "文件名升序"),
    ("err_desc", "残差降序 (最差优先 ↓)"),
    ("err_asc", "残差升序 (最优优先 ↑)"),
    ("tags_desc", "标靶数量降序")
]


def draw_dropdown_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    is_open: bool,
    mouse_pos: Tuple[int, int],
    prefix: str = ""
):
    """绘制现代扁平化微质感下拉菜单头部按钮"""
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

    if is_open:
        bg_col = (48, 56, 72)
        border_col = (0, 220, 255)
        text_col = (255, 255, 255)
        arrow = "▲"
    elif is_hover:
        bg_col = (36, 40, 52)
        border_col = (0, 180, 220)
        text_col = (240, 240, 240)
        arrow = "▼"
    else:
        bg_col = (28, 30, 38)
        border_col = (55, 60, 75)
        text_col = (200, 200, 200)
        arrow = "▼"

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

    display_txt = f"{prefix}{label} {arrow}" if prefix else f"{label} {arrow}"
    (tw, th), _ = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    tx = x1 + max(6, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(canvas, display_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)


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
        self.right_bar_w = 340  # 右侧属性诊断栏宽度

        # 2. 相机内参与领域模型装配
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()

        self.manifest_path = MANIFEST_PATH
        self.manifest_data: Dict[str, Any] = {}
        self._load_manifest()

        self.tags_map_data: Dict[str, Any] = {}
        self._load_tags_map()

        self.manifest_repo = ManifestRepository()
        self.engine = OfflineVerificationEngine(
            tags_map=self.tags_map_data,
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

        # 3. 图像资产扫描与状态列表
        self.image_files: List[str] = []
        self._scan_images()
        self.current_img_idx = 0
        self.scroll_offset = 0  # 左侧列表滚动偏移行数

        # 筛选与排序模式
        self.filter_mode: str = "all"
        self.sort_mode: str = "name_asc"

        # 视口显示模式: "mix" (混合 2D+3D), "3d" (仅 3D 棱柱), "2d" (仅 2D 识别与残差矢量)
        self.view_mode: str = "mix"

        # 下拉菜单展开态标识 ("FILTER_DROPDOWN", "SORT_DROPDOWN", "VIEW_DROPDOWN" 或 None)
        self.active_dropdown: Optional[str] = None
        self.dropdown_boxes: Dict[str, Dict[str, Any]] = {}

        # 4. 单帧与全集指标缓存
        self.frame_metrics_cache: Dict[str, Dict[str, Any]] = {}
        self.global_rmse = 0.0

        # 5. 异步 BA 全局平差任务状态与进度
        self.is_ba_running = False
        self.ba_thread: Optional[threading.Thread] = None
        self.ba_result_queue: Optional[Tuple[bool, str]] = None
        self.ba_progress: float = 0.0
        self.ba_stage_text: str = ""
        self.ba_sub_progress: float = 0.0
        self.ba_sub_text: str = ""

        # 6. 单帧深度病因切片诊断开关
        self.show_frame_diagnostics = False

        # 7. 浮层通知 (Toast)
        self.status_toast = "欢迎进入 Offline Studio 离线标定工作站"
        self.status_toast_time = time.time()

        # 8. GUI 交互按钮注册表
        self.gui_buttons: List[Tuple[str, Tuple[int, int, int, int], Any]] = []
        self.mouse_pos = (-1, -1)
        self.is_running = True

        # 9. 中间画布视口变换与平移缩放状态 (Viewport Zoom & Pan)
        self.zoom_level: float = 1.0
        self.pan_offset_x: float = 0.0
        self.pan_offset_y: float = 0.0
        self.is_panning: bool = False
        self.pan_start_pos: Tuple[int, int] = (0, 0)

        # 首次预热并计算全集残差指标
        self.refresh_all_frame_metrics()


    def reset_viewport_zoom(self):
        """重置中间视口缩放与平移状态为适应屏幕 (1.0x)"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
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

    def _load_manifest(self):
        """加载观测清单文件"""
        if os.path.exists(self.manifest_path):
            try:
                import yaml
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest_data = yaml.safe_load(f) or {}
            except Exception:
                self.manifest_data = {"images": {}}
        else:
            self.manifest_data = {"images": {}}

    def _save_manifest(self):
        """保存观测清单文件"""
        os.makedirs(os.path.dirname(os.path.abspath(self.manifest_path)), exist_ok=True)
        try:
            import yaml
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(self.manifest_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except Exception as e:
            print(f"[WARN] 保存观测清单失败: {e}")

    def is_image_excluded(self, base_name: str) -> bool:
        """检查指定图像是否已被整帧剔除"""
        img_info = self.manifest_data.get("images", {}).get(base_name, {})
        return not img_info.get("enabled", True)

    def toggle_image_exclusion(self, base_name: str) -> bool:
        """翻转整帧的保留/剔除状态"""
        img_info = self.manifest_data.setdefault("images", {}).setdefault(base_name, {})
        cur_enabled = img_info.get("enabled", True)
        new_enabled = not cur_enabled
        img_info["enabled"] = new_enabled
        self._save_manifest()
        return not new_enabled

    def toggle_observation_keep(self, base_name: str, target_tag_id: int) -> bool:
        """翻转单帧内特定标靶的保留/剔除状态"""
        obs_list = self.get_observations_for_image(base_name)
        new_keep = True
        for obs in obs_list:
            if obs["tag_id"] == target_tag_id:
                obs["keep"] = not obs.get("keep", True)
                new_keep = obs["keep"]
                break
        self._save_manifest()
        return new_keep

    def get_observations_for_image(self, base_name: str) -> List[Dict[str, Any]]:
        """获取指定图像的所有标靶观测"""
        return self.manifest_data.get("images", {}).get(base_name, {}).get("observations", [])

    def get_tag_world_corners(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶角点在当前地图中的世界坐标"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return None
        tag_data = self.tags_map_data["tags"].get(tag_id)
        if tag_data is None:
            return None
        s = self.marker_size_mm / 2.0
        local_c = np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)
        if "transform_matrix" in tag_data:
            T = np.array(tag_data["transform_matrix"], dtype=np.float64)
            return (T @ local_c.T).T[:, :3]
        return None

    def get_tag_transform(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶 4x4 齐次变换矩阵"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return None
        tag_data = self.tags_map_data["tags"].get(tag_id)
        if tag_data is None:
            return None
        if "transform_matrix" in tag_data:
            return np.array(tag_data["transform_matrix"], dtype=np.float64)
        return None

    def _scan_images(self):
        """扫描采图目录中全部 view_*.png 图像文件"""
        if os.path.exists(self.image_dir):
            self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "view_*.png")))
        else:
            self.image_files = []

    def _load_tags_map(self):
        """加载空间立体地图"""
        if os.path.exists(self.map_path):
            try:
                import yaml
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.tags_map_data = yaml.safe_load(f) or {}
                if self.tags_map_data and "tags" in self.tags_map_data:
                    print(f"[OK] Studio 成功装载地图: {self.map_path} (共 {len(self.tags_map_data['tags'])} 个标靶)")
            except Exception as e:
                print(f"[WARN] 无法读取地图: {e}")
                self.tags_map_data = {}
        else:
            self.tags_map_data = {}

    def refresh_all_frame_metrics(self):
        """重新遍历计算所有可用帧的残差与有效性指标"""
        self.frame_metrics_cache.clear()
        all_reproj_errors = []

        if not self.tags_map_data or "tags" not in self.tags_map_data:
            for img_path in self.image_files:
                base_name = os.path.basename(img_path)
                obs_list = self.get_observations_for_image(base_name)
                is_excl = self.is_image_excluded(base_name)
                self.frame_metrics_cache[base_name] = {
                    "tag_count": len(obs_list),
                    "mean_err": 0.0,
                    "max_err": 0.0,
                    "is_excluded": is_excl,
                    "observations": obs_list
                }
            self.global_rmse = 0.0
            return

        for img_path in self.image_files:
            base_name = os.path.basename(img_path)
            obs_list = self.get_observations_for_image(base_name)
            is_excl = self.is_image_excluded(base_name)

            if not obs_list:
                # 若清单未录入，尝试快速单帧提取
                img = cv2.imread(img_path)
                if img is not None:
                    tags_dict = self.engine.detect_tags(img)
                    obs_list = []
                    for tid, c in tags_dict.items():
                        obs_list.append({
                            "tag_id": int(tid),
                            "corners": c.reshape((4, 2)).tolist(),
                            "keep": True
                        })


            # 计算该帧在当前地图下的位姿与重投影残差
            mean_err, max_err, errors_dict = self._evaluate_frame_reprojection(obs_list)
            if not is_excl and errors_dict:
                for tid, err in errors_dict.items():
                    all_reproj_errors.append(err)

            self.frame_metrics_cache[base_name] = {
                "tag_count": len(obs_list),
                "mean_err": mean_err,
                "max_err": max_err,
                "is_excluded": is_excl,
                "observations": obs_list,
                "tag_errors": errors_dict
            }

        if all_reproj_errors:
            self.global_rmse = float(np.sqrt(np.mean(np.array(all_reproj_errors) ** 2)))
        else:
            self.global_rmse = 0.0

    def _evaluate_frame_reprojection(self, observations: List[Dict[str, Any]]) -> Tuple[float, float, Dict[int, float]]:
        """计算单帧中所有标靶的重投影残差"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return 0.0, 0.0, {}

        obj_pts = []
        img_pts = []
        valid_tids = []

        for obs in observations:
            if not obs.get("keep", True):
                continue
            tid = obs["tag_id"]
            w_corners = self.get_tag_world_corners(tid)
            if w_corners is not None:
                c_arr = np.array(obs["corners"], dtype=np.float64).reshape((4, 2))
                obj_pts.append(w_corners)
                img_pts.append(c_arr)
                valid_tids.append(tid)

        if not obj_pts:
            return 0.0, 0.0, {}

        obj_flat = np.concatenate(obj_pts, axis=0)
        img_flat = np.concatenate(img_pts, axis=0)

        rvec, tvec, success = self.engine.solve_pnp(obj_flat, img_flat)
        if not success:
            return 0.0, 0.0, {}

        proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, self.engine.camera_matrix, self.engine.dist_coeffs)
        dists = np.linalg.norm(img_flat - proj_pts.reshape((-1, 2)), axis=1)

        errors_dict = {}
        for idx, tid in enumerate(valid_tids):
            errors_dict[tid] = float(np.mean(dists[idx * 4:(idx + 1) * 4]))

        mean_val = float(np.mean(dists))
        max_val = float(np.max(dists))
        return mean_val, max_val, errors_dict


    def _get_filtered_indices(self) -> List[int]:
        """依据当前的过滤模式与排序模式获取最终展示的帧索引列表"""
        matched = []
        for idx, p in enumerate(self.image_files):
            base_name = os.path.basename(p)
            meta = self.frame_metrics_cache.get(base_name, {})
            if self.filter_mode == "all":
                matched.append(idx)
            elif self.filter_mode == "warning":
                if meta.get("mean_err", 0.0) > 0.5 and not meta.get("is_excluded", False):
                    matched.append(idx)
            elif self.filter_mode == "excluded":
                if meta.get("is_excluded", False):
                    matched.append(idx)

        # 排序规则处理
        if self.sort_mode == "name_asc":
            matched.sort(key=lambda i: os.path.basename(self.image_files[i]))
        elif self.sort_mode == "err_desc":
            matched.sort(key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("mean_err", 0.0), reverse=True)
        elif self.sort_mode == "err_asc":
            matched.sort(key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("mean_err", 0.0))
        elif self.sort_mode == "tags_desc":
            matched.sort(key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("tag_count", 0), reverse=True)

        return matched


    def toggle_current_frame_exclusion(self):
        """翻转当前选中帧的保留/剔除状态"""
        if not self.image_files:
            return
        cur_file = self.image_files[self.current_img_idx]
        base_name = os.path.basename(cur_file)

        is_now_excluded = self.toggle_image_exclusion(base_name)
        # 同步更新内存缓存
        if base_name in self.frame_metrics_cache:
            self.frame_metrics_cache[base_name]["is_excluded"] = is_now_excluded

        status_str = "已标记为 [剔除/EXCLUDED]" if is_now_excluded else "已恢复为 [保留/ACTIVE]"
        self.set_toast(f"帧 {base_name} {status_str}")
        print(f"[*] 帧状态翻转: {base_name} -> {status_str}")

    def toggle_tag_exclusion_in_current_frame(self, target_tag_id: int):
        """翻转当前帧中特定标靶的保留/剔除状态"""
        if not self.image_files:
            return
        cur_file = self.image_files[self.current_img_idx]
        base_name = os.path.basename(cur_file)

        is_kept = self.toggle_observation_keep(base_name, target_tag_id)
        # 刷新当前帧残差
        meta = self.frame_metrics_cache.get(base_name, {})
        obs_list = self.get_observations_for_image(base_name)
        mean_err, max_err, errors_dict = self._evaluate_frame_reprojection(obs_list)
        meta["observations"] = obs_list
        meta["mean_err"] = mean_err
        meta["max_err"] = max_err
        meta["tag_errors"] = errors_dict

        t_str = "已保留" if is_kept else "已剔除"
        self.set_toast(f"标靶 Tag #{target_tag_id} 在本帧中 {t_str}")

    def start_async_bundle_adjustment(self):
        """启动后台线程执行两阶段全局 BA 平差优化，前台持续平滑响应"""
        if self.is_ba_running:
            self.set_toast("BA 全局平差优化已在运行中，请稍候...")
            return

        self.is_ba_running = True
        self.ba_progress = 0.05
        self.ba_sub_progress = 0.0
        self.ba_stage_text = "正在启动两阶段全局 BA 平差优化计算..."
        self.ba_sub_text = "初始化优化工作空间..."
        self.set_toast("正在启动两阶段全局 BA 平差优化计算...")
        print("\n[*] [STUDIO] 正在启动异步 BA 全局平差优化计算...")

        def _worker():
            try:
                self.ba_progress = 0.10
                self.ba_sub_progress = 0.30
                self.ba_stage_text = "阶段 1/4: 准备观测清单与共视拓扑分析..."
                self.ba_sub_text = "校验数据清单并分析共视连通性图..."
                self._save_manifest()
                time.sleep(0.05)

                self.ba_progress = 0.25
                self.ba_sub_progress = 0.60
                self.ba_stage_text = "阶段 2/4: 过滤已剔除样本并构建全局初值..."
                self.ba_sub_text = "构建超定 PnP 初始机位与标靶三维姿态..."
                frame_detections, valid_frame_names, _ = self.manifest_repo.load_manifest(self.manifest_path)
                if len(frame_detections) < 2:
                    self.ba_result_queue = (False, "有效图像帧不足 2 帧，无法执行 BA 平差")
                    return

                self.ba_progress = 0.40
                self.ba_sub_progress = 0.0
                self.ba_stage_text = "阶段 3/4: 两阶段 Cauchy 稳健核平差全局收敛求解..."
                self.ba_sub_text = "准备启动非线性平差求解器..."

                def on_ba_callback(info: Dict[str, Any]):
                    stg = info.get("stage", 1)
                    stg_name = info.get("stage_name", "")
                    cur_it = info.get("iter", 0)
                    max_it = info.get("max_iter", 1)
                    cur_rmse = info.get("rmse", 0.0)
                    sub_pct = max(0.0, min(1.0, info.get("sub_progress", 0.0)))

                    # 全局大进度条在阶段 3/4 期间平滑推进：阶段一占 40%~62%，阶段二占 62%~84%
                    if stg == 1:
                        self.ba_progress = 0.40 + 0.22 * sub_pct
                    else:
                        self.ba_progress = 0.62 + 0.22 * sub_pct

                    self.ba_sub_progress = sub_pct
                    self.ba_sub_text = f"[{stg_name}] 轮次 #{cur_it}/{max_it} | 实时 RMSE: {cur_rmse:.3f} px"

                opt_res = self.optimizer.optimize(frame_detections, valid_frame_names, callback=on_ba_callback)

                self.ba_progress = 0.88
                self.ba_sub_progress = 0.50
                self.ba_stage_text = "阶段 4/4: 重映射空间立体地图与外参反算..."
                self.ba_sub_text = "对齐世界原点并写入立体几何地图缓存..."
                if opt_res and "final_tag_poses_aligned" in opt_res:
                    tags_dict = {}
                    for tid, T in opt_res["final_tag_poses_aligned"].items():
                        tags_dict[tid] = {
                            "transform_matrix": T.tolist(),
                            "position_mm": T[:3, 3].tolist()
                        }
                    new_map = {
                        "marker_size_mm": self.marker_size_mm,
                        "tags": tags_dict,
                        "rmse_px": opt_res.get("final_rmse", 0.0)
                    }
                    ManifestRepository.save_map(new_map, self.map_path)
                    self.tags_map_data = new_map
                    self.ba_progress = 1.0
                    self.ba_sub_progress = 1.0
                    self.ba_stage_text = "全局平差完成！正在同步就地热更新..."
                    self.ba_sub_text = f"最终全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px (已就地生效)"
                    msg = f"BA 优化成功！新全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px (地图已就地热重载)"
                    self.ba_result_queue = (True, msg)
                else:
                    self.ba_result_queue = (False, "BA 优化未能收敛，请检查有效观测标靶数")
            except Exception as e:
                self.ba_result_queue = (False, f"BA 平差优化异常: {e}")

        self.ba_thread = threading.Thread(target=_worker, daemon=True)
        self.ba_thread.start()



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
        w, h = self.win_w, self.win_h
        top_h = self.viewport.top_bar_h
        bot_h = self.viewport.bottom_bar_h
        self.gui_buttons.clear()

        # 异步 BA 结果检查
        if self.ba_result_queue is not None:
            succ, msg = self.ba_result_queue
            self.ba_result_queue = None
            self.is_ba_running = False
            self.set_toast(msg)
            if succ:
                self.refresh_all_frame_metrics()

        # 1. 顶栏 (Top Navigation Bar)
        self._render_top_bar(canvas, w, top_h)

        # 2. 底栏 (Bottom Control Toolbar)
        self._render_bottom_toolbar(canvas, w, h, bot_h)

        # 3. 中间工作区尺寸
        content_y1 = top_h
        content_y2 = h - bot_h
        content_h = content_y2 - content_y1

        # 左栏：紧凑帧序列列表 (x: 0 -> left_bar_w)
        self._render_left_frame_list(canvas, 0, content_y1, self.left_bar_w, content_h)

        # 右栏：属性与单帧诊断面板 (x: w - right_bar_w -> w)
        self._render_right_inspector(canvas, w - self.right_bar_w, content_y1, self.right_bar_w, content_h)

        # 中栏：高清工作视口 (x: left_bar_w -> w - right_bar_w)
        mid_x1 = self.left_bar_w
        mid_w = w - self.left_bar_w - self.right_bar_w
        self._render_center_viewport(canvas, mid_x1, content_y1, mid_w, content_h)

        # 4. 居中展示异步 BA 运行中科技感浮层
        if self.is_ba_running:
            self._render_ba_loading_card(canvas, w, h)

        # 5. Toast 浮层
        if time.time() - self.status_toast_time < 3.0 and self.status_toast:
            self._render_toast(canvas, w, h, bot_h)

        # 6. 置顶渲染展开状态的下拉菜单浮层 (最高 Z-Index, 绝不被下层视口截断)
        if self.active_dropdown:
            self._render_active_dropdown(canvas)

    def _render_active_dropdown(self, canvas: np.ndarray):
        """在全局最顶层 (Z-Index 最高) 渲染当前展开的下拉选项列表"""
        if not self.active_dropdown or self.active_dropdown not in self.dropdown_boxes:
            return

        box = self.dropdown_boxes[self.active_dropdown]
        rect = box["rect"]
        options = box["options"]
        active_key = box["active_key"]

        x1, y1, x2, y2 = rect
        item_h = 32
        pop_w = max(x2 - x1, 175)
        pop_h = len(options) * item_h + 6
        pop_x1 = x1
        pop_y1 = y2 + 2
        pop_x2 = pop_x1 + pop_w
        pop_y2 = pop_y1 + pop_h

        # 防超出画布左右与底部边界
        if pop_x2 > self.win_w - 6:
            pop_x1 = self.win_w - pop_w - 6
            pop_x2 = pop_x1 + pop_w
        if pop_y2 > self.win_h - 10:
            pop_y1 = y1 - pop_h - 2
            pop_y2 = pop_y1 + pop_h

        # 浮层半透明背景与青色发光边框
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), (0, 200, 255), 1)

        mx, my = self.mouse_pos
        for idx, (opt_key, opt_label) in enumerate(options):
            iy1 = pop_y1 + 3 + idx * item_h
            iy2 = iy1 + item_h - 1
            is_active = (opt_key == active_key)
            is_hover = (pop_x1 <= mx <= pop_x2 and iy1 <= my <= iy2)

            if is_active:
                row_bg = (52, 45, 20)
                txt_col = (0, 230, 255)
            elif is_hover:
                row_bg = (36, 42, 56)
                txt_col = (255, 255, 255)
            else:
                row_bg = (22, 25, 32)
                txt_col = (190, 190, 190)

            cv2.rectangle(canvas, (pop_x1 + 3, iy1), (pop_x2 - 3, iy2), row_bg, -1)
            prefix = "✔ " if is_active else "  "
            (tw, th), _ = cv2.getTextSize(prefix + opt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(canvas, prefix + opt_label, (pop_x1 + 8, iy1 + (item_h + th) // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

            # 注册选项点击按钮
            btn_id = f"DD_SELECT_{self.active_dropdown}_{opt_key}"
            self.gui_buttons.append((btn_id, (pop_x1, iy1, pop_x2, iy2), (self.active_dropdown, opt_key)))

    def _render_top_bar(self, canvas: np.ndarray, w: int, top_h: int):

        cv2.rectangle(canvas, (0, 0), (w, top_h), (24, 26, 32), -1)
        cv2.line(canvas, (0, top_h), (w, top_h), (55, 60, 72), 1)

        cv2.putText(canvas, "OFFLINE STUDIO", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "| AprilTag 离线标定与空间建图综合工作站", (185, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        # 状态指标
        tag_num = len(self.tags_map_data.get("tags", {}))
        stat_txt = f"采图集: {len(self.image_files)} 帧  |  已知标靶: {tag_num} 个  |  全局 RMSE: {self.global_rmse:.2f} px"
        (tw, _), _ = cv2.getTextSize(stat_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(canvas, stat_txt, (w - tw - 20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1, cv2.LINE_AA)

    def _render_bottom_toolbar(self, canvas: np.ndarray, w: int, h: int, bot_h: int):
        y1 = h - bot_h
        cv2.rectangle(canvas, (0, y1), (w, h), (20, 22, 28), -1)
        cv2.line(canvas, (0, y1), (w, y1), (60, 65, 78), 1)

        btn_y_top = y1 + 8
        btn_y_bot = h - 8
        bx = 16
        mx, my = self.mouse_pos

        # [B] 全局平差
        ba_w = 135
        ba_type = "warning" if self.is_ba_running else "primary"
        ba_txt = "正在平差..." if self.is_ba_running else "全局平差 (B)"
        draw_styled_button(canvas, (bx, btn_y_top, bx + ba_w, btn_y_bot), ba_txt,
                           mouse_pos=(mx, my), btn_type=ba_type)
        self.gui_buttons.append(("RUN_BA", (bx, btn_y_top, bx + ba_w, btn_y_bot), "RUN_BA"))
        bx += ba_w + 10

        # [P] 全量精度体检重算
        p_w = 125
        draw_styled_button(canvas, (bx, btn_y_top, bx + p_w, btn_y_bot), "全量体检 (P)",
                           mouse_pos=(mx, my), btn_type="normal")
        self.gui_buttons.append(("RECOMPUTE_METRICS", (bx, btn_y_top, bx + p_w, btn_y_bot), "RECOMPUTE_METRICS"))
        bx += p_w + 10

        # [R] 导出质检报告
        r_w = 115
        draw_styled_button(canvas, (bx, btn_y_top, bx + r_w, btn_y_bot), "导出报告 (R)",
                           mouse_pos=(mx, my), btn_type="normal")
        self.gui_buttons.append(("EXPORT_REPORT", (bx, btn_y_top, bx + r_w, btn_y_bot), "EXPORT_REPORT"))
        bx += r_w + 10

        # [S] 保存/发布地图
        s_w = 115
        draw_styled_button(canvas, (bx, btn_y_top, bx + s_w, btn_y_bot), "保存地图 (S)",
                           mouse_pos=(mx, my), btn_type="success")
        self.gui_buttons.append(("SAVE_MAP", (bx, btn_y_top, bx + s_w, btn_y_bot), "SAVE_MAP"))
        bx += s_w + 10

        # 右侧 [Q] 退出工作台
        exit_w = 90
        exit_x1 = w - exit_w - 16
        draw_styled_button(canvas, (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "退出 (Q)",
                           mouse_pos=(mx, my), btn_type="danger")
        self.gui_buttons.append(("EXIT", (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "EXIT"))

    def _render_left_frame_list(self, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """左栏：高信息密度垂直紧凑帧列表"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x + w, y), (x + w, y + h), (50, 54, 66), 1)

        # 顶部并排双下拉菜单 (筛选范围 + 排序方式, 高度 34px)
        header_h = 36
        dd_y1 = y + 6
        dd_y2 = y + header_h
        dd1_w = (w - 24) // 2
        dd1_x1 = x + 8
        dd1_x2 = dd1_x1 + dd1_w
        dd2_x1 = dd1_x2 + 8
        dd2_x2 = x + w - 8

        # 1. 筛选范围下拉框
        cur_filter_label = dict(FILTER_MODE_OPTIONS).get(self.filter_mode, "全部帧")
        is_f_open = (self.active_dropdown == "FILTER_DROPDOWN")
        draw_dropdown_button(canvas, (dd1_x1, dd_y1, dd1_x2, dd_y2), cur_filter_label,
                             is_open=is_f_open, mouse_pos=self.mouse_pos)
        self.dropdown_boxes["FILTER_DROPDOWN"] = {
            "rect": (dd1_x1, dd_y1, dd1_x2, dd_y2),
            "options": FILTER_MODE_OPTIONS,
            "active_key": self.filter_mode
        }
        self.gui_buttons.append(("TOGGLE_FILTER_DROPDOWN", (dd1_x1, dd_y1, dd1_x2, dd_y2), "FILTER_DROPDOWN"))

        # 2. 排序方式下拉框
        cur_sort_label = dict(SORT_MODE_OPTIONS).get(self.sort_mode, "文件名升序")
        short_sort = cur_sort_label.split(" ")[0] if "(" in cur_sort_label else cur_sort_label
        is_s_open = (self.active_dropdown == "SORT_DROPDOWN")
        draw_dropdown_button(canvas, (dd2_x1, dd_y1, dd2_x2, dd_y2), short_sort,
                             is_open=is_s_open, mouse_pos=self.mouse_pos)
        self.dropdown_boxes["SORT_DROPDOWN"] = {
            "rect": (dd2_x1, dd_y1, dd2_x2, dd_y2),
            "options": SORT_MODE_OPTIONS,
            "active_key": self.sort_mode
        }
        self.gui_buttons.append(("TOGGLE_SORT_DROPDOWN", (dd2_x1, dd_y1, dd2_x2, dd_y2), "SORT_DROPDOWN"))

        # 帧列表区域
        list_y = y + header_h + 8
        item_h = 36
        visible_count = (h - header_h - 16) // item_h


        filtered_indices = self._get_filtered_indices()
        if not filtered_indices:
            cv2.putText(canvas, "当前筛选条件下无图像", (x + 60, list_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1, cv2.LINE_AA)
            return

        # 修正滚动偏移量
        self.scroll_offset = max(0, min(self.scroll_offset, len(filtered_indices) - visible_count))

        for row_idx in range(visible_count):
            list_idx = self.scroll_offset + row_idx
            if list_idx >= len(filtered_indices):
                break

            orig_img_idx = filtered_indices[list_idx]
            p = self.image_files[orig_img_idx]
            bname = os.path.basename(p)
            meta = self.frame_metrics_cache.get(bname, {})

            iy1 = list_y + row_idx * item_h
            iy2 = iy1 + item_h - 2
            is_selected = (orig_img_idx == self.current_img_idx)
            is_hover = (x + 4 <= self.mouse_pos[0] <= x + w - 4 and iy1 <= self.mouse_pos[1] <= iy2)

            # 行底色
            if is_selected:
                row_bg = (55, 45, 20)  # 选中高亮琥珀金
                border_col = (0, 200, 255)
            elif is_hover:
                row_bg = (38, 42, 52)
                border_col = (70, 75, 90)
            else:
                row_bg = (28, 30, 38)
                border_col = (42, 45, 56)

            cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), row_bg, -1)
            cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), border_col, 1)
            self.gui_buttons.append((f"SELECT_FRAME_{orig_img_idx}", (x + 6, iy1, x + w - 6, iy2), orig_img_idx))

            # 状态圆点 (绿色正常 / 黄色残差偏高 / 灰色已剔除)
            dot_x = x + 18
            dot_y = iy1 + (item_h // 2) - 1
            if meta.get("is_excluded", False):
                cv2.circle(canvas, (dot_x, dot_y), 4, (100, 100, 100), -1)
                text_col = (110, 115, 125)
            elif meta.get("mean_err", 0.0) > 0.5:
                cv2.circle(canvas, (dot_x, dot_y), 4, (0, 215, 255), -1)
                text_col = (0, 220, 255)
            else:
                cv2.circle(canvas, (dot_x, dot_y), 4, (0, 230, 80), -1)
                text_col = (230, 230, 230)

            # 文本排版: view_XXXX  [4T]  0.18px
            cv2.putText(canvas, bname, (x + 30, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_col, 1, cv2.LINE_AA)

            tag_badge = f"[{meta.get('tag_count', 0)}T]"
            cv2.putText(canvas, tag_badge, (x + 165, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)

            if meta.get("is_excluded", False):
                cv2.putText(canvas, "EXCL", (x + 230, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1, cv2.LINE_AA)
            else:
                err_str = f"{meta.get('mean_err', 0.0):.2f}px"
                err_col = (0, 200, 255) if meta.get('mean_err', 0.0) > 0.5 else (0, 240, 100)
                cv2.putText(canvas, err_str, (x + 225, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, err_col, 1, cv2.LINE_AA)

    def _render_center_viewport(self, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """中栏：高清工作视口，等比居中自适应渲染"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (14, 15, 18), -1)

        if not self.image_files:
            cv2.putText(canvas, "未扫描到采图图像 (data/tag_calibration_images/ 为空)", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
            return

        cur_file = self.image_files[self.current_img_idx]
        bgr = cv2.imread(cur_file)
        if bgr is None:
            cv2.putText(canvas, f"读取图像文件失败: {cur_file}", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            return

        disp_frame = bgr.copy()
        base_name = os.path.basename(cur_file)
        meta = self.frame_metrics_cache.get(base_name, {})
        obs_list = meta.get("observations", [])

        # 叠加标靶与 3D 双棱柱
        self._overlay_visual_elements(disp_frame, obs_list, meta.get("is_excluded", False), meta=meta)

        # 视口等比与平移缩放渲染
        frame_h, frame_w = disp_frame.shape[:2]
        base_scale = min(w / frame_w, h / frame_h)
        curr_scale = base_scale * self.zoom_level
        target_w = int(round(frame_w * curr_scale))
        target_h = int(round(frame_h * curr_scale))

        center_x = x + w / 2.0 + self.pan_offset_x
        center_y = y + h / 2.0 + self.pan_offset_y

        img_x1 = int(round(center_x - target_w / 2.0))
        img_y1 = int(round(center_y - target_h / 2.0))
        img_x2 = img_x1 + target_w
        img_y2 = img_y1 + target_h

        # 计算视口矩形 [x, y, x + w, y + h] 与目标虚拟矩形 [img_x1, img_y1, img_x2, img_y2] 的求交
        dst_x1 = max(x, img_x1)
        dst_y1 = max(y, img_y1)
        dst_x2 = min(x + w, img_x2)
        dst_y2 = min(y + h, img_y2)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            rel_x1 = (dst_x1 - img_x1) / float(target_w)
            rel_y1 = (dst_y1 - img_y1) / float(target_h)
            rel_x2 = (dst_x2 - img_x1) / float(target_w)
            rel_y2 = (dst_y2 - img_y1) / float(target_h)

            src_x1 = max(0, min(frame_w - 1, int(round(rel_x1 * frame_w))))
            src_y1 = max(0, min(frame_h - 1, int(round(rel_y1 * frame_h))))
            src_x2 = max(src_x1 + 1, min(frame_w, int(round(rel_x2 * frame_w))))
            src_y2 = max(src_y1 + 1, min(frame_h, int(round(rel_y2 * frame_h))))

            src_roi = disp_frame[src_y1:src_y2, src_x1:src_x2]
            dst_w = dst_x2 - dst_x1
            dst_h = dst_y2 - dst_y1

            if dst_w > 0 and dst_h > 0 and src_roi.size > 0:
                interp = cv2.INTER_LINEAR if self.zoom_level > 1.2 else cv2.INTER_AREA
                resized_roi = cv2.resize(src_roi, (dst_w, dst_h), interpolation=interp)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized_roi

        # 视口外边框
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 60, 70), 1)

        # 视口左上角：显示模式下拉菜单 (2D / 3D / 混合)
        vx1 = x + 12
        vy1 = y + 10
        vx2 = vx1 + 175
        vy2 = vy1 + 28
        cur_view_label = dict(VIEW_MODE_OPTIONS).get(self.view_mode, "混合透视")
        is_v_open = (self.active_dropdown == "VIEW_DROPDOWN")
        draw_dropdown_button(canvas, (vx1, vy1, vx2, vy2), cur_view_label,
                             is_open=is_v_open, mouse_pos=self.mouse_pos, prefix="模式: ")
        self.dropdown_boxes["VIEW_DROPDOWN"] = {
            "rect": (vx1, vy1, vx2, vy2),
            "options": VIEW_MODE_OPTIONS,
            "active_key": self.view_mode
        }
        self.gui_buttons.append(("TOGGLE_VIEW_DROPDOWN", (vx1, vy1, vx2, vy2), "VIEW_DROPDOWN"))

        # 视口右上角悬浮提示胶囊
        zoom_badge = f"缩放: {self.zoom_level:.1f}x | 切换模式: V | 拖拽: 右键/中键 | 双击/Z: 重置"
        (zw, zh), _ = cv2.getTextSize(zoom_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        bx1 = x + w - zw - 24
        by1 = y + 10
        bx2 = bx1 + zw + 14
        by2 = by1 + zh + 10
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 24, 32), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (70, 75, 88), 1)
        cv2.putText(canvas, zoom_badge, (bx1 + 7, by1 + zh + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 210, 230), 1, cv2.LINE_AA)


    def _overlay_visual_elements(
        self,
        disp_frame: np.ndarray,
        observations: List[Dict[str, Any]],
        is_frame_excluded: bool,
        meta: Optional[Dict[str, Any]] = None
    ):

        """在工作底图上依据 view_mode 分离渲染 2D 识别框、3D 棱柱与重投影残差矢量"""
        show_2d = self.view_mode in ("2d", "mix")
        show_3d = self.view_mode in ("3d", "mix")

        obj_pts = []
        img_pts = []
        valid_obs = []

        for obs in observations:
            tid = obs["tag_id"]
            pts = np.array(obs["corners"], dtype=np.int32).reshape((-1, 2))
            keep = obs.get("keep", True) and not is_frame_excluded

            # 2D 识别框与标牌 (仅在 2d 或 mix 模式下绘制)
            if show_2d:
                box_col = (0, 230, 80) if keep else (80, 80, 80)
                thick = 2 if keep else 1
                cv2.polylines(disp_frame, [pts], isClosed=True, color=box_col, thickness=thick, lineType=cv2.LINE_AA)

                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                badge_txt = f"Tag #{tid}" if keep else f"Tag #{tid} [EXCL]"
                cv2.putText(disp_frame, badge_txt, (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 2, cv2.LINE_AA)

            # 收集参与解算的标靶
            if keep:
                w_c = self.get_tag_world_corners(tid)
                if w_c is not None:
                    obj_pts.append(w_c)
                    img_pts.append(np.array(obs["corners"], dtype=np.float64))
                    valid_obs.append(obs)

        # 3D 棱柱与残差矢量投影
        if len(obj_pts) >= 1:
            obj_flat = np.concatenate(obj_pts, axis=0)
            img_flat = np.concatenate(img_pts, axis=0)
            rvec, tvec, success = self.engine.solve_pnp(obj_flat, img_flat)
            if success:
                # 绘制 3D 双棱柱 (仅在 3d 或 mix 模式下绘制)
                if show_3d:
                    for obs in valid_obs:
                        tid = obs["tag_id"]
                        T_w_t = self.get_tag_transform(tid)
                        if T_w_t is not None:
                            R_c_w, _ = cv2.Rodrigues(rvec)
                            T_c_w = np.eye(4, dtype=np.float64)
                            T_c_w[:3, :3] = R_c_w
                            T_c_w[:3, 3] = tvec.flatten()
                            T_c_t = T_c_w @ T_w_t
                            r_tag, _ = cv2.Rodrigues(T_c_t[:3, :3])
                            t_tag = T_c_t[:3, 3].reshape((3, 1))

                            # 解算单标靶本地实测位姿 (用于 3D 蓝色实测棱柱)
                            c_arr = np.array(obs["corners"], dtype=np.float64).reshape((4, 2))
                            succ_single, obs_r, obs_t = self.engine.solve_single_tag_pnp(c_arr)

                            # 计算空间位移误差 (mm) 与 2D 重投影残差 (px)
                            err_mm = 0.0
                            if t_tag is not None and succ_single and obs_t is not None:
                                err_mm = float(np.linalg.norm(t_tag - obs_t))
                            err_px = (meta or {}).get("tag_errors", {}).get(tid, 0.2)

                            self.visualizer.render_tag_dual_prisms(
                                img=disp_frame,
                                ba_rvec=r_tag,
                                ba_tvec=t_tag,
                                obs_rvec=obs_r if succ_single else None,
                                obs_tvec=obs_t if succ_single else None,
                                tag_id=tid,
                                err_px=err_px,
                                err_mm=err_mm,
                                observed_corners=c_arr
                            )


                # 绘制亚像素残差红色放大矢量箭头 (仅在 2d 或 mix 模式下绘制)
                if show_2d:
                    proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, self.engine.camera_matrix, self.engine.dist_coeffs)
                    proj_flat = proj_pts.reshape((-1, 2))
                    if hasattr(self.visualizer, "draw_reprojection_vectors"):
                        self.visualizer.draw_reprojection_vectors(disp_frame, img_flat, proj_flat, scale_factor=40.0)



    def _render_right_inspector(self, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """右栏：属性、细目列表与漏检切片诊断面板"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x, y), (x, y + h), (50, 54, 66), 1)

        if not self.image_files:
            return

        cur_file = self.image_files[self.current_img_idx]
        bname = os.path.basename(cur_file)
        meta = self.frame_metrics_cache.get(bname, {})

        # 1. 顶部当前帧摘要卡片 (高度: 110px)
        cv2.putText(canvas, "当前选定帧属性", (x + 12, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, bname, (x + 12, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        # 状态切换大按钮 (保留 / 剔除)
        is_excl = meta.get("is_excluded", False)
        btn_w = w - 24
        btn_y1 = y + 62
        btn_y2 = btn_y1 + 34
        b_type = "danger" if is_excl else "success"
        b_label = "[已剔除] 点击恢复保留 (T)" if is_excl else "[保留中] 点击剔除此帧 (T)"
        draw_styled_button(canvas, (x + 12, btn_y1, x + 12 + btn_w, btn_y2), b_label,
                           mouse_pos=self.mouse_pos, btn_type=b_type)
        self.gui_buttons.append(("TOGGLE_FRAME_STATUS", (x + 12, btn_y1, x + 12 + btn_w, btn_y2), bname))

        # 2. 标靶细目清单 (表格展示)
        list_y = y + 115
        cv2.line(canvas, (x + 10, list_y), (x + w - 10, list_y), (45, 48, 58), 1)
        cv2.putText(canvas, "本帧标靶细目与残差", (x + 12, list_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

        obs_list = meta.get("observations", [])
        tag_errors = meta.get("tag_errors", {})

        row_y = list_y + 32
        row_h = 32
        for obs in obs_list:
            tid = obs["tag_id"]
            keep = obs.get("keep", True)
            err_val = tag_errors.get(tid, 0.0)

            # 行底色
            rx1, ry1, rx2, ry2 = x + 10, row_y, x + w - 10, row_y + row_h - 2
            is_hover = (rx1 <= self.mouse_pos[0] <= rx2 and ry1 <= self.mouse_pos[1] <= ry2)
            bg_col = (34, 38, 48) if is_hover else (26, 28, 36)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), bg_col, -1)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (48, 52, 64), 1)
            self.gui_buttons.append((f"TOGGLE_TAG_{tid}", (rx1, ry1, rx2, ry2), tid))

            # 文本
            t_col = (240, 240, 240) if keep else (110, 110, 110)
            cv2.putText(canvas, f"Tag #{tid}", (rx1 + 10, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, t_col, 1, cv2.LINE_AA)

            err_c = (0, 200, 255) if err_val > 0.5 else (0, 230, 80)
            cv2.putText(canvas, f"{err_val:.2f} px", (rx1 + 110, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, err_c, 1, cv2.LINE_AA)

            btn_act = "[剔除]" if keep else "[恢复]"
            act_c = (0, 160, 255) if keep else (0, 220, 100)
            cv2.putText(canvas, btn_act, (rx2 - 55, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, act_c, 1, cv2.LINE_AA)

            row_y += row_h
            if row_y > y + h - 140:
                break

        # 3. 底部快捷动作与漏检切片诊断卡片
        diag_y = y + h - 120
        cv2.line(canvas, (x + 10, diag_y), (x + w - 10, diag_y), (45, 48, 58), 1)
        cv2.putText(canvas, "单帧快捷动作", (x + 12, diag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

        act_btn_w = (w - 30) // 2
        act_y1 = diag_y + 30
        act_y2 = act_y1 + 36

        # 按钮 1: 超精重提取
        draw_styled_button(canvas, (x + 10, act_y1, x + 10 + act_btn_w, act_y2), "超精提取 (E)",
                           mouse_pos=self.mouse_pos, btn_type="primary")
        self.gui_buttons.append(("SUPER_EXTRACT_FRAME", (x + 10, act_y1, x + 10 + act_btn_w, act_y2), bname))

        # 按钮 2: 漏检病因
        draw_styled_button(canvas, (x + 15 + act_btn_w, act_y1, x + w - 10, act_y2), "病因诊断 (D)",
                           mouse_pos=self.mouse_pos, btn_type="warning")
        self.gui_buttons.append(("DIAGNOSE_FRAME", (x + 15 + act_btn_w, act_y1, x + w - 10, act_y2), bname))

        # 诊断提示小字
        if len(obs_list) < 3:
            cv2.putText(canvas, "* 提示: 本帧标靶较少，可按 D 键切片分析漏检原因", (x + 12, act_y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)

    def _render_ba_loading_card(self, canvas: np.ndarray, w: int, h: int):
        """居中展示异步 BA 全局平差双轨进度卡片 (大阶段主进度条 + 求解器子进度条与实时收敛指标)"""
        card_w, card_h = 640, 162
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (0, 220, 255), 2)

        # ---------------- 1. 主阶段总流程进度条 ----------------
        pct = max(0.0, min(1.0, self.ba_progress))
        pct_int = int(round(pct * 100))

        # 主标题与主进度百分比
        cv2.putText(canvas, "[BA] 全局平差整体收敛进度", (cx1 + 22, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA)

        # 主进度条槽位 (高 12px)
        bar_x1 = cx1 + 22
        bar_y1 = cy1 + 34
        bar_x2 = cx1 + card_w - 22
        bar_y2 = bar_y1 + 12
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (55, 62, 78), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 210, 255), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (180, 245, 255), 1)

        # 主阶段当前说明
        stage_txt = self.ba_stage_text or "准备进入优化平差管线..."
        cv2.putText(canvas, stage_txt, (cx1 + 22, cy1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

        # ---------------- 细微居中分割线 ----------------
        cv2.line(canvas, (cx1 + 22, cy1 + 74), (cx1 + card_w - 22, cy1 + 74), (45, 48, 60), 1)

        # ---------------- 2. 求解器内部子阶段与迭代进度条 ----------------
        sub_pct = max(0.0, min(1.0, self.ba_sub_progress))
        sub_pct_int = int(round(sub_pct * 100))

        # 子阶段标题与百分比
        cv2.putText(canvas, "优化器实时迭代收敛监控 (Sub-Iteration)", (cx1 + 22, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 205, 215), 1, cv2.LINE_AA)
        sub_pct_str = f"{sub_pct_int}%"
        (spw, _), _ = cv2.getTextSize(sub_pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        cv2.putText(canvas, sub_pct_str, (cx1 + card_w - 22 - spw, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 180, 255), 2, cv2.LINE_AA)

        # 子进度条槽位 (高 10px)
        sbar_y1 = cy1 + 102
        sbar_y2 = sbar_y1 + 10
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (25, 28, 36), -1)
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (48, 54, 68), 1)

        sub_fill_w = int(bar_w * sub_pct)
        if sub_fill_w > 0:
            # 琥珀金 / 亮橙高光填充，与主进度条形成高辨识度层次区分
            cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y2), (0, 160, 255), -1)
            cv2.line(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y1), (120, 220, 255), 1)

        # 底部动态收敛指标与轮次
        sub_txt = self.ba_sub_text or "等待当前阶段迭代步进推进..."
        cv2.putText(canvas, sub_txt, (cx1 + 22, cy1 + 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)


    def _render_toast(self, canvas: np.ndarray, w: int, h: int, bot_h: int):
        (tw, _), _ = cv2.getTextSize(self.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        tx1 = (w - tw) // 2 - 16
        ty1 = h - bot_h - 46
        tx2 = tx1 + tw + 32
        ty2 = ty1 + 32

        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (120, 30, 100), -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (220, 60, 180), 1)
        cv2.putText(canvas, self.status_toast, (tx1 + 16, ty1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

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
                zoom_factor = 1.15 if wheel_up else (1.0 / 1.15)
                new_zoom = max(0.4, min(15.0, self.zoom_level * zoom_factor))
                ratio = new_zoom / self.zoom_level

                # 当前视口几何中心
                mid_w = mid_x2 - mid_x1
                mid_h = content_y2 - content_y1
                curr_center_x = mid_x1 + mid_w / 2.0 + self.pan_offset_x
                curr_center_y = content_y1 + mid_h / 2.0 + self.pan_offset_y

                # 保持当前鼠标指向的图像局部坐标在缩放前后完全重合
                dx = mx - curr_center_x
                dy = my - curr_center_y
                new_center_x = mx - dx * ratio
                new_center_y = my - dy * ratio

                self.pan_offset_x = new_center_x - (mid_x1 + mid_w / 2.0)
                self.pan_offset_y = new_center_y - (content_y1 + mid_h / 2.0)
                self.zoom_level = new_zoom
                return

        # 2. 拖拽平移事件 (支持鼠标右键或中键按住平移)
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            if mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                self.is_panning = True
                self.pan_start_pos = (mx, my)
                return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                dx = mx - self.pan_start_pos[0]
                dy = my - self.pan_start_pos[1]
                self.pan_offset_x += dx
                self.pan_offset_y += dy
                self.pan_start_pos = (mx, my)
                return
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            if self.is_panning:
                self.is_panning = False
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
        elif btn_id == "TOGGLE_VIEW_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "VIEW_DROPDOWN" else "VIEW_DROPDOWN"
        elif btn_id == "TOGGLE_FILTER_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "FILTER_DROPDOWN" else "FILTER_DROPDOWN"
        elif btn_id == "TOGGLE_SORT_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "SORT_DROPDOWN" else "SORT_DROPDOWN"
        elif btn_id.startswith("DD_SELECT_"):
            dd_name, selected_val = extra
            if dd_name == "VIEW_DROPDOWN":
                self.view_mode = selected_val
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
                elif key in (ord('v'), ord('V')):      # V 键 -> 循环切换显示模式
                    modes = ["mix", "3d", "2d"]
                    curr_i = modes.index(self.view_mode) if self.view_mode in modes else 0
                    self.view_mode = modes[(curr_i + 1) % len(modes)]
                    lbl = dict(VIEW_MODE_OPTIONS).get(self.view_mode, self.view_mode)
                    self.set_toast(f"显示模式已切换为: {lbl}")
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
