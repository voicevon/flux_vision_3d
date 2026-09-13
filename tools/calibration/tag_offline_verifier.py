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

try:
    from src.utils.viewport_manager import (
        ViewportManager, get_safe_screen_size,
        draw_styled_button, draw_segmented_toggle
    )
except ImportError:
    ViewportManager = None
    get_safe_screen_size = None
    draw_styled_button = None
    draw_segmented_toggle = None


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
        """双路互补融合检测 (与 TagMapBuilder.detect_tags 一致)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        results = {}

        # 路 1: 高光路
        c1, ids1, _ = self.detector_bright.detectMarkers(gray)
        if ids1 is not None:
            for idx, tag_id in enumerate(ids1.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                results[tid] = c1[idx].reshape((4, 2))

        # 路 2: 低反差路 (动态拉伸)
        p_low, p_high = np.percentile(gray[::4, ::4], (2, 98))
        if p_high > p_low + 10:
            gray_s = np.clip((gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            gray_s = gray

        c2, ids2, _ = self.detector_dark.detectMarkers(gray_s)
        if ids2 is not None:
            for idx, tag_id in enumerate(ids2.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                if tid not in results:
                    results[tid] = c2[idx].reshape((4, 2))

        # 亚像素精修
        for tid in list(results.keys()):
            results[tid] = self._refine_corners(gray, results[tid])

        return results

    def _refine_corners(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """亚像素角点精修"""
        try:
            pts = corners.reshape((4, 2)).astype(np.float32)
            side = (np.linalg.norm(pts[0] - pts[1]) + np.linalg.norm(pts[1] - pts[2])) / 2.0
            hw = int(np.clip(side * 0.06, 3, 9))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            refined = cv2.cornerSubPix(gray, pts.copy(), (hw, hw), (-1, -1), criteria)
            if np.max(np.linalg.norm(refined - pts, axis=1)) > 2.5:
                return pts.astype(np.float64)
            return refined.astype(np.float64)
        except Exception:
            return corners.astype(np.float64)

    def _get_tag_world_transform(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶在世界坐标系下的 4x4 变换矩阵"""
        tag_data = self.tags_map.get("tags", {}).get(tag_id)
        if tag_data is None:
            tag_data = self.tags_map.get("tags", {}).get(str(tag_id))
        if tag_data is None:
            return None
        if "transform_matrix" in tag_data:
            return np.array(tag_data["transform_matrix"], dtype=np.float64)
        elif "position_mm" in tag_data:
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = np.array(tag_data["position_mm"], dtype=np.float64)
            return T
        return None

    def _get_tag_world_corners(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶 4 个角点在世界坐标系下的 3D 位置"""
        T = self._get_tag_world_transform(tag_id)
        if T is None:
            return None
        s = self.marker_size_mm / 2.0
        local = np.array([
            [-s, s, 0.0, 1.0], [s, s, 0.0, 1.0],
            [s, -s, 0.0, 1.0], [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)
        return (T @ local.T).T[:, :3]

    def _solve_camera_pose(self, tag_corners_pairs: List[Tuple[int, np.ndarray]]) -> Optional[Dict]:
        """
        给定一组 (tag_id, corners_2d) 对，超定 PnP 求解相机位姿。
        集成物理正深度前置校验与 SQPNP -> EPNP -> ITERATIVE 多级回退机制，彻底杜绝翻转伪解。
        :return: {"rvec", "tvec", "rmse"} 或 None
        """
        all_obj = []
        all_img = []
        for tid, c2d in tag_corners_pairs:
            wc = self._get_tag_world_corners(tid)
            if wc is not None:
                all_obj.append(wc)
                all_img.append(c2d.reshape(4, 2))

        if len(all_obj) < 1:
            return None

        obj_flat = np.concatenate(all_obj, axis=0)
        img_flat = np.concatenate(all_img, axis=0)

        best_rvec, best_tvec = None, None
        best_rmse = float("inf")

        def try_candidate(r, t):
            nonlocal best_rvec, best_tvec, best_rmse
            if r is None or t is None:
                return
            tz = float(t[2, 0])
            if tz <= 100.0 or tz > 3500.0:
                return
            proj, _ = cv2.projectPoints(obj_flat, r, t, self.camera_matrix, self.dist_coeffs)
            rmse = float(np.sqrt(np.mean((img_flat - proj.reshape(-1, 2)) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_rvec = r.copy()
                best_tvec = t.copy()

        # 1. 尝试 RANSAC SQPNP (标靶数 >= 3 时抗噪效果佳)
        if len(all_obj) >= 3:
            try:
                succ_r, r_r, t_r, inliers = cv2.solvePnPRansac(
                    obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                    reprojectionError=4.0, flags=cv2.SOLVEPNP_SQPNP
                )
                if succ_r:
                    try_candidate(r_r, t_r)
            except Exception:
                pass

        # 2. 尝试标准 SQPNP
        try:
            succ_sq, r_sq, t_sq = cv2.solvePnP(
                obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )
            if succ_sq:
                try_candidate(r_sq, t_sq)
        except Exception:
            pass

        # 3. 若 SQPNP 解算不佳或出现负深度/翻转，尝试 EPNP
        if best_rmse > 5.0 or best_rvec is None:
            try:
                succ_ep, r_ep, t_ep = cv2.solvePnP(
                    obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP
                )
                if succ_ep:
                    try_candidate(r_ep, t_ep)
            except Exception:
                pass

        # 4. 尝试 ITERATIVE
        if best_rmse > 10.0 or best_rvec is None:
            try:
                succ_it, r_it, t_it = cv2.solvePnP(
                    obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if succ_it:
                    try_candidate(r_it, t_it)
            except Exception:
                pass

        if best_rvec is None or best_tvec is None:
            return None

        # 5. 采用非线性最小二乘极限精修 (ITERATIVE with extrinsic guess)
        if len(all_obj) >= 2:
            try:
                succ_ref, r_ref, t_ref = cv2.solvePnP(
                    obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                    rvec=best_rvec, tvec=best_tvec, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if succ_ref and 100.0 < float(t_ref[2, 0]) <= 3500.0:
                    proj, _ = cv2.projectPoints(obj_flat, r_ref, t_ref, self.camera_matrix, self.dist_coeffs)
                    ref_rmse = float(np.sqrt(np.mean((img_flat - proj.reshape(-1, 2)) ** 2)))
                    if ref_rmse <= best_rmse * 1.5:
                        best_rvec, best_tvec, best_rmse = r_ref, t_ref, ref_rmse
            except Exception:
                pass

        return {"rvec": best_rvec, "tvec": best_tvec, "rmse": best_rmse}

    def _solve_single_tag_pnp(self, corners_2d: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """根据单帧检出的 4 个 2D 角点解算单标靶实测相机外参位姿 (优先 IPPE_SQUARE，兜底 ITERATIVE)"""
        try:
            c = corners_2d.reshape((4, 2)).astype(np.float64)
            succ, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                self.obj_points, c, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if succ and len(rvecs) > 0:
                best_r, best_t, min_err = None, None, float("inf")
                for r, t in zip(rvecs, tvecs):
                    if t[2, 0] <= 0:
                        continue
                    proj, _ = cv2.projectPoints(self.obj_points, r, t, self.camera_matrix, self.dist_coeffs)
                    err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - c, axis=1))
                    if err < min_err:
                        min_err = err
                        best_r, best_t = r, t
                if best_r is not None:
                    return True, best_r, best_t
        except Exception:
            pass

        try:
            c = corners_2d.reshape((4, 2)).astype(np.float64)
            succ, r, t = cv2.solvePnP(self.obj_points, c, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if succ and t[2, 0] > 0:
                return True, r, t
        except Exception:
            pass

        return False, None, None

    def _compute_loo_error(self, rvec, tvec, tag_id: int, observed_corners: np.ndarray) -> Optional[Dict]:
        """
        计算单个盲测目标的重投影误差，并记录 BA 理论位姿与单帧实测位姿
        :return: 包含 2D 误差、3D 误差与双位姿向量的字典
        """
        T_w_t = self._get_tag_world_transform(tag_id)
        if T_w_t is None:
            return None

        T_c_w = np.eye(4, dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        T_c_w[:3, :3] = R
        T_c_w[:3, 3] = tvec.flatten()
        T_c_t = T_c_w @ T_w_t
        r_t, _ = cv2.Rodrigues(T_c_t[:3, :3])
        t_t = T_c_t[:3, 3].reshape(3, 1)

        proj, _ = cv2.projectPoints(self.obj_points, r_t, t_t, self.camera_matrix, self.dist_coeffs)
        proj_2d = proj.reshape(4, 2)

        obs = observed_corners.reshape(4, 2)
        per_corner_err = np.linalg.norm(proj_2d - obs, axis=1)
        mean_err_px = float(np.mean(per_corner_err))

        # 空间误差估算: err_mm ≈ err_px * depth / fx
        depth = float(t_t[2, 0])
        fx = self.camera_matrix[0, 0]
        err_mm = (mean_err_px * abs(depth)) / fx if fx > 0 else 0.0

        # 解算单帧实测位姿用于 3D 双棱柱虚实比对
        ok_obs, obs_r, obs_t = self._solve_single_tag_pnp(obs)

        return {
            "err_px": mean_err_px,
            "err_mm": err_mm,
            "depth_mm": depth,
            "per_corner_px": per_corner_err.tolist(),
            "proj_corners": proj_2d,
            "obs_corners": obs,
            "ba_rvec": r_t,
            "ba_tvec": t_t,
            "obs_rvec": obs_r if ok_obs else None,
            "obs_tvec": obs_t if ok_obs else None
        }

    def _verify_single_frame(self, img_path: str) -> List[Dict]:
        """
        对单帧执行全标靶 Leave-One-Out 盲测循环
        :return: 该帧所有 LOO 测试结果列表
        """
        fname = os.path.basename(img_path)
        known_tags = {}

        # 优先从已审核观测清单中提取有效样本 (保持与 BA 求解输入完全一致)
        if self.actual_source == "manifest" and self.manifest_data:
            img_entry = self.manifest_data.get("images", {}).get(fname, {})
            if not img_entry.get("enabled", True):
                return []
            for obs in img_entry.get("observations", []):
                tid = obs.get("tag_id")
                if obs.get("keep", True) and tid in self.mapped_tag_ids:
                    c = np.array(obs["corners"], dtype=np.float64)
                    area = cv2.contourArea(c.reshape(4, 2).astype(np.float32))
                    if area >= 250.0:  # 过滤远景噪点与像素过小的极度畸变标靶
                        known_tags[tid] = c

        # 若非清单模式或清单无数据，则对图像进行实时检测
        if not known_tags:
            img = cv2.imread(img_path)
            if img is None:
                return []
            detected = self.detect_tags(img)
            for tid, c in detected.items():
                if tid in self.mapped_tag_ids:
                    area = cv2.contourArea(c.reshape(4, 2).astype(np.float32))
                    if area >= 250.0:
                        known_tags[tid] = c

        if len(known_tags) < 2:
            return []

        results = []
        for blind_tid in sorted(known_tags.keys()):
            # 构建排除盲测目标后的求解集
            solver_pairs = [(tid, c) for tid, c in known_tags.items() if tid != blind_tid]
            if len(solver_pairs) < 1:
                continue

            pose = self._solve_camera_pose(solver_pairs)
            if pose is None or pose["rmse"] > 30.0:
                continue

            loo = self._compute_loo_error(pose["rvec"], pose["tvec"], blind_tid, known_tags[blind_tid])
            if loo is None:
                continue

            results.append({
                "image": os.path.basename(img_path),
                "blind_tag_id": blind_tid,
                "solver_tags": [tid for tid, _ in solver_pairs],
                "solver_count": len(solver_pairs),
                "solver_rmse_px": pose["rmse"],
                **loo
            })

        return results

    def render_tag_dual_prisms(self, img: np.ndarray,
                               ba_rvec: Optional[np.ndarray], ba_tvec: Optional[np.ndarray],
                               obs_rvec: Optional[np.ndarray], obs_tvec: Optional[np.ndarray],
                               tag_id: int, err_px: float, err_mm: float,
                               observed_corners: Optional[np.ndarray] = None):
        """
        绘制全局 BA 平差理论位姿 (绿色) 与单帧实测抓取位姿 (金色/橙红) 的 3D 双四棱柱空间对比
        """
        try:
            hw = 15.0   # 截面半宽 15mm，整体截面 30.0mm x 30.0mm
            L = 75.0    # 柱体高度 75mm (与实际尺寸协调)

            pts_3d = np.array([
                # 底面 4 点 (Z=0)
                [-hw, -hw, 0.0],
                [ hw, -hw, 0.0],
                [ hw,  hw, 0.0],
                [-hw,  hw, 0.0],
                # 顶面 4 点 (Z=L)
                [-hw, -hw, L],
                [ hw, -hw, L],
                [ hw,  hw, L],
                [-hw,  hw, L],
                # 顶面中心
                [0.0, 0.0, L]
            ], dtype=np.float64)

            # 1. 投影 BA 理论棱柱
            proj_ba = None
            if ba_rvec is not None and ba_tvec is not None:
                p, _ = cv2.projectPoints(pts_3d, ba_rvec, ba_tvec, self.camera_matrix, self.dist_coeffs)
                proj_ba = p.reshape((-1, 2)).astype(int)

            # 2. 投影实测观测棱柱
            proj_obs = None
            if obs_rvec is not None and obs_tvec is not None:
                p, _ = cv2.projectPoints(pts_3d, obs_rvec, obs_tvec, self.camera_matrix, self.dist_coeffs)
                proj_obs = p.reshape((-1, 2)).astype(int)

            is_good = (err_px <= 1.5 and err_mm <= 1.5)
            is_moderate = (err_px <= 3.0 and err_mm <= 2.5)

            overlay = img.copy()

            # A. 良好达标样本 (BA 与实测高度吻合，渲染巍峨稳健的纯正翠绿棱柱，给用户强烈的踏实感)
            if is_good:
                pts = proj_ba if proj_ba is not None else proj_obs
                if pts is not None:
                    b_pts = pts[0:4]
                    t_pts = pts[4:8]
                    top_c = tuple(pts[8])

                    # 翠绿色半透明实心柱体
                    side_color = (0, 190, 50)
                    cap_color = (80, 255, 120)
                    for i in range(4):
                        next_i = (i + 1) % 4
                        side_poly = np.array([b_pts[i], b_pts[next_i], t_pts[next_i], t_pts[i]], dtype=np.int32)
                        cv2.fillPoly(overlay, [side_poly], side_color)
                    cv2.fillPoly(overlay, [t_pts], cap_color)
                    cv2.addWeighted(overlay, 0.42, img, 0.58, 0, img)

                    # 纯净亮白/亮绿棱线描边
                    edge_c = (0, 245, 100)
                    cv2.polylines(img, [b_pts], True, edge_c, 2, cv2.LINE_AA)
                    cv2.polylines(img, [t_pts], True, (255, 255, 255), 2, cv2.LINE_AA)
                    for i in range(4):
                        cv2.line(img, tuple(b_pts[i]), tuple(t_pts[i]), edge_c, 2, cv2.LINE_AA)

                    # 顶盖中心与标识
                    cv2.circle(img, top_c, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(img, "BA", (top_c[0] + 5, top_c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

                    # 底面实测角点连线与四色圆点
                    if observed_corners is not None:
                        c_int = observed_corners.reshape((4, 2)).astype(np.int32)
                        cv2.polylines(img, [c_int], True, (0, 255, 100), 2, cv2.LINE_AA)
                        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                        for pt_i, pt in enumerate(c_int):
                            cv2.circle(img, tuple(pt), 4, dot_colors[pt_i], -1)

                    # 悬浮高对比度稳态绿色标牌 (彻底告别原本看不清的暗灰色)
                    min_x = min(np.min(b_pts[:, 0]), np.min(t_pts[:, 0]))
                    min_y = min(np.min(b_pts[:, 1]), np.min(t_pts[:, 1]))
                    bx = max(10, int(min_x - 10))
                    by = max(40, int(min_y - 14))

                    label = f"Tag#{tag_id} [PASS] {err_mm:.2f}mm ({err_px:.2f}px)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), (10, 42, 16), -1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), (0, 240, 90), 1)
                    cv2.putText(img, label, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # B. 偏差/需回审样本 (渲染绿色 BA 棱柱 vs 金黄/橙红实测棱柱，形成鲜明空间错位对比)
            else:
                # 1. 渲染绿色 BA 理论棱柱 (Truth / 理论目标位置)
                if proj_ba is not None:
                    ba_b = proj_ba[0:4]
                    ba_t = proj_ba[4:8]
                    ba_c = tuple(proj_ba[8])

                    side_c = (0, 180, 80)
                    for i in range(4):
                        next_i = (i + 1) % 4
                        side_poly = np.array([ba_b[i], ba_b[next_i], ba_t[next_i], ba_t[i]], dtype=np.int32)
                        cv2.fillPoly(overlay, [side_poly], side_c)
                    cv2.fillPoly(overlay, [ba_t], (50, 250, 140))
                    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

                    # 绿色棱线
                    cv2.polylines(img, [ba_b], True, (0, 220, 80), 2, cv2.LINE_AA)
                    cv2.polylines(img, [ba_t], True, (120, 255, 160), 2, cv2.LINE_AA)
                    for i in range(4):
                        cv2.line(img, tuple(ba_b[i]), tuple(ba_t[i]), (0, 220, 80), 2, cv2.LINE_AA)
                    cv2.circle(img, ba_c, 4, (0, 255, 100), -1, cv2.LINE_AA)
                    cv2.putText(img, "BA", (ba_c[0] + 6, ba_c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)

                # 2. 渲染金黄/橙红实测棱柱 (Observed / 实测抓拍位置)
                if proj_obs is not None:
                    obs_b = proj_obs[0:4]
                    obs_t = proj_obs[4:8]
                    obs_c = tuple(proj_obs[8])

                    warn_c = (0, 80, 240) if not is_moderate else (0, 160, 255)
                    overlay2 = img.copy()
                    for i in range(4):
                        next_i = (i + 1) % 4
                        side_poly = np.array([obs_b[i], obs_b[next_i], obs_t[next_i], obs_t[i]], dtype=np.int32)
                        cv2.fillPoly(overlay2, [side_poly], warn_c)
                    cv2.fillPoly(overlay2, [obs_t], (0, 210, 255))
                    cv2.addWeighted(overlay2, 0.30, img, 0.70, 0, img)

                    # 实测棱线
                    cv2.polylines(img, [obs_b], True, warn_c, 2, cv2.LINE_AA)
                    cv2.polylines(img, [obs_t], True, (255, 255, 255), 2, cv2.LINE_AA)
                    for i in range(4):
                        cv2.line(img, tuple(obs_b[i]), tuple(obs_t[i]), warn_c, 2, cv2.LINE_AA)
                    cv2.circle(img, obs_c, 4, (0, 200, 255), -1, cv2.LINE_AA)
                    cv2.putText(img, "OBS", (obs_c[0] + 6, obs_c[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)

                    # 3. 顶盖中心拉出 3D 空间残差偏转连线
                    if proj_ba is not None:
                        cv2.line(img, obs_c, ba_c, (0, 50, 255), 3, cv2.LINE_AA)
                        cv2.circle(img, obs_c, 5, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(img, ba_c, 5, (0, 255, 0), -1, cv2.LINE_AA)

                # 4. 醒目报警标牌
                anchor_pts = proj_ba if proj_ba is not None else proj_obs
                if anchor_pts is not None:
                    min_x = np.min(anchor_pts[:, 0])
                    min_y = np.min(anchor_pts[:, 1])
                    bx = max(10, int(min_x - 10))
                    by = max(40, int(min_y - 14))
                    tag_status = "[WARN]" if is_moderate else "[FAIL-回审]"
                    border_c = (0, 160, 255) if is_moderate else (0, 0, 255)
                    bg_c = (15, 30, 60) if is_moderate else (15, 15, 65)
                    label = f"Tag#{tag_id} {tag_status} {err_mm:.2f}mm ({err_px:.2f}px)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), bg_c, -1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), border_c, 2)
                    cv2.putText(img, label, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    def _render_verification_frame(self, img_path: str, frame_results: List[Dict], out_path: Optional[str] = None) -> np.ndarray:
        """渲染单帧 LOO 盲测可视化图 (支持 3D 双棱柱空间对比视图 与 2D 角点残差矢量视图)"""
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
        disp = img.copy()
        h, w = disp.shape[:2]

        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

        # 若当前帧 frame_results 为空（如孤立单靶帧），依然尽可能从清单提取有效样本渲染实测 3D 棱柱，杜绝空白
        if not frame_results and self.manifest_data:
            fname = os.path.basename(img_path)
            img_entry = self.manifest_data.get("images", {}).get(fname, {})
            for obs in img_entry.get("observations", []):
                if obs.get("keep", True):
                    tid = int(obs.get("tag_id", -1))
                    c = np.array(obs.get("corners", []), dtype=np.float64)
                    if len(c) == 4:
                        ok_pnp, r_obs, t_obs = self._solve_single_tag_pnp(c)
                        if ok_pnp:
                            self.render_tag_dual_prisms(
                                disp, ba_rvec=None, ba_tvec=None,
                                obs_rvec=r_obs, obs_tvec=t_obs,
                                tag_id=tid, err_px=0.0, err_mm=0.0,
                                observed_corners=c
                            )

        if self.view_mode_3d:
            # =================================================================
            # 模式 A: 3D 双四棱柱虚实位姿对比模式 (BA 理论真值 vs 实测抓取)
            # =================================================================
            for r in frame_results:
                tid = r["blind_tag_id"]
                err_px = r["err_px"]
                err_mm = r["err_mm"]
                ba_r = r.get("ba_rvec")
                ba_t = r.get("ba_tvec")
                obs_r = r.get("obs_rvec")
                obs_t = r.get("obs_tvec")
                obs_c = r.get("obs_corners")

                self.render_tag_dual_prisms(
                    disp, ba_rvec=ba_r, ba_tvec=ba_t,
                    obs_rvec=obs_r, obs_tvec=obs_t,
                    tag_id=tid, err_px=err_px, err_mm=err_mm,
                    observed_corners=obs_c
                )

            # 左下方悬浮模式胶囊提示条
            mode_badge = "[ 视图: 3D双四棱柱空间对比模式 (按 T 键切换 2D 残差矢量) ]"
            (mw, mh), _ = cv2.getTextSize(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (18, 22, 30), -1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (0, 220, 180), 1)
            cv2.putText(disp, mode_badge, (22, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 200), 1, cv2.LINE_AA)

        else:
            # =================================================================
            # 模式 B: 2D 角点残差矢量模式 (高对比度强化版，好样本同样清晰醒目)
            # =================================================================
            for r in frame_results:
                obs = r["obs_corners"].astype(np.int32)
                proj = r["proj_corners"].astype(np.int32)
                tid = r["blind_tag_id"]
                err_px = r["err_px"]
                err_mm = r["err_mm"]

                is_good = (err_px <= 1.5)
                is_moderate = (err_px <= 3.0)

                # 观测轮廓 (好样本为翡翠绿，超标样本为深灰色)
                obs_line_color = (0, 230, 80) if is_good else ((180, 180, 180) if is_moderate else (120, 120, 120))
                cv2.polylines(disp, [obs], True, obs_line_color, 2 if is_good else 1, cv2.LINE_AA)

                # 反推投影轮廓
                if is_good:
                    proj_color = (0, 255, 100)
                elif is_moderate:
                    proj_color = (0, 180, 255)
                else:
                    proj_color = (0, 0, 255)

                cv2.polylines(disp, [proj], True, proj_color, 2, cv2.LINE_AA)

                # 逐角点残差箭头 (放大 15 倍)
                scale = 15.0
                for i in range(4):
                    p_obs = r["obs_corners"][i]
                    p_proj = r["proj_corners"][i]
                    dx = (p_proj[0] - p_obs[0]) * scale
                    dy = (p_proj[1] - p_obs[1]) * scale
                    pt_s = (int(p_obs[0]), int(p_obs[1]))
                    pt_e = (int(p_obs[0] + dx), int(p_obs[1] + dy))
                    cv2.circle(disp, pt_s, 4, dot_colors[i], -1)
                    cv2.arrowedLine(disp, pt_s, pt_e, proj_color, 2, tipLength=0.25)

                # 误差标牌 (好样本赋予高对比度绿底白字)
                cx = int(np.mean(obs[:, 0]))
                min_y = int(np.min(obs[:, 1]))
                badge_y = max(75, min_y - 14)
                badge_x = max(10, cx - 85)

                status_text = "[PASS]" if is_good else ("[WARN]" if is_moderate else "[FAIL]")
                label = f"Tag#{tid} {status_text} {err_px:.2f}px / {err_mm:.2f}mm"
                badge_bg = (12, 42, 16) if is_good else ((15, 35, 75) if is_moderate else (15, 15, 75))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 18), (badge_x + tw + 8, badge_y + 4), badge_bg, -1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 18), (badge_x + tw + 8, badge_y + 4), proj_color, 1 if is_good else 2)
                cv2.putText(disp, label, (badge_x, badge_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # 左下方悬浮模式胶囊提示条
            mode_badge = "[ 视图: 2D角点残差矢量模式 (按 T 键切换 3D 双棱柱) ]"
            (mw, mh), _ = cv2.getTextSize(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (18, 22, 30), -1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (0, 200, 255), 1)
            cv2.putText(disp, mode_badge, (22, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        if out_path is not None:
            cv2.imwrite(out_path, disp)

        return disp

    def _aggregate_statistics(self, all_results: List[Dict]) -> Dict:
        """汇总 Per-Tag / Per-Frame 统计，识别系统性偏差与建议回审帧"""
        if not all_results:
            return {"pass": False, "message": "无有效盲测结果", "per_tag": {}, "per_frame": {}, "flagged_frames": [], "flagged_tags": []}

        # Per-Tag 统计
        tag_errors = {}
        for r in all_results:
            tid = r["blind_tag_id"]
            if tid not in tag_errors:
                tag_errors[tid] = {"err_px": [], "err_mm": []}
            tag_errors[tid]["err_px"].append(r["err_px"])
            tag_errors[tid]["err_mm"].append(r["err_mm"])

        per_tag = {}
        for tid, data in sorted(tag_errors.items()):
            arr_px = np.array(data["err_px"])
            arr_mm = np.array(data["err_mm"])
            per_tag[tid] = {
                "count": len(arr_px),
                "median_px": float(np.median(arr_px)),
                "mean_px": float(np.mean(arr_px)),
                "max_px": float(np.max(arr_px)),
                "std_px": float(np.std(arr_px)),
                "median_mm": float(np.median(arr_mm)),
                "mean_mm": float(np.mean(arr_mm)),
                "max_mm": float(np.max(arr_mm)),
            }

        # Per-Frame 统计
        frame_errors = {}
        for r in all_results:
            fname = r["image"]
            if fname not in frame_errors:
                frame_errors[fname] = {"err_px": [], "tags": []}
            frame_errors[fname]["err_px"].append(r["err_px"])
            frame_errors[fname]["tags"].append(r["blind_tag_id"])

        per_frame = {}
        for fname, data in sorted(frame_errors.items()):
            arr = np.array(data["err_px"])
            per_frame[fname] = {
                "tag_count": len(arr),
                "mean_px": float(np.mean(arr)),
                "max_px": float(np.max(arr)),
                "tags_tested": sorted(data["tags"])
            }

        # 全局统计
        all_px = np.array([r["err_px"] for r in all_results])
        all_mm = np.array([r["err_mm"] for r in all_results])
        global_median_px = float(np.median(all_px))
        global_median_mm = float(np.median(all_mm))
        global_mean_px = float(np.mean(all_px))
        global_std_px = float(np.std(all_px))

        # 系统性偏差检测: Tag 中位误差 > 全局中位数 × 2
        flagged_tags = []
        for tid, stats in per_tag.items():
            if stats["median_px"] > max(2.5, global_median_px * 2.0):
                flagged_tags.append(tid)

        # 坏帧检测 (基于稳健统计门限)
        mad_px = float(np.median(np.abs(all_px - global_median_px)))
        outlier_thresh = max(5.0, min(15.0, global_median_px + 3.0 * (1.4826 * mad_px if mad_px > 0.1 else 1.0)))
        flagged_frames = []
        for fname, stats in per_frame.items():
            if stats["mean_px"] > outlier_thresh or stats["max_px"] > 25.0:
                flagged_frames.append(fname)

        # 综合放行评级 (Pass / Acceptable / Fail)
        if global_median_px <= 1.5 and global_median_mm <= 1.0 and len(flagged_tags) == 0 and len(flagged_frames) == 0:
            grade = "PASS"
            grade_label = "合格 (PASS) — 精度极佳，允许直接上线"
        elif global_median_px <= 3.0 and global_median_mm <= 2.5 and len(flagged_tags) <= 1:
            grade = "ACCEPTABLE"
            grade_label = "基本合格 (ACCEPTABLE) — 可准入 AR 验证"
        else:
            grade = "FAIL"
            grade_label = "不合格 (FAIL) — 建议回审后重新 BA 求解"

        return {
            "pass": grade in ("PASS", "ACCEPTABLE"),
            "grade": grade,
            "grade_label": grade_label,
            "total_tests": len(all_results),
            "global_median_px": global_median_px,
            "global_mean_px": global_mean_px,
            "global_std_px": global_std_px,
            "global_median_mm": global_median_mm,
            "global_mean_mm": float(np.mean(all_mm)),
            "outlier_threshold_px": outlier_thresh,
            "per_tag": per_tag,
            "per_frame": per_frame,
            "flagged_tags": flagged_tags,
            "flagged_frames": flagged_frames,
        }

    def _export_markdown_report(self, stats: Dict, all_results: List[Dict]) -> str:
        """输出完整 Markdown 精度体检报告"""
        report_path = os.path.join(VERIFICATION_DIR, "offline_verification_report.md")
        grade = stats.get("grade", "N/A")
        grade_label = stats.get("grade_label", "N/A")
        grade_emoji = "[PASS]" if grade == "PASS" else ("[WARN]" if grade == "ACCEPTABLE" else "[FAIL]")

        status_median_px = "[PASS]" if stats['global_median_px'] <= 1.5 else ("[WARN]" if stats['global_median_px'] <= 3.0 else "[FAIL]")
        status_median_mm = "[PASS]" if stats['global_median_mm'] <= 1.0 else ("[WARN]" if stats['global_median_mm'] <= 2.5 else "[FAIL]")
        status_flagged_tags = "[PASS]" if len(stats['flagged_tags']) == 0 else ("[WARN]" if len(stats['flagged_tags']) <= 2 else "[FAIL]")
        status_flagged_frames = "[PASS]" if len(stats['flagged_frames']) == 0 else "[WARN]"

        source_label = "已审核观测清单 (tag_observations.yaml)" if self.actual_source == "manifest" else "原始采图重新检测 (端到端独立复检)"

        lines = [
            "# 离线标定精度体检与 Leave-One-Out 盲测批量验证报告",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"> 地图文件: `{self.map_path}`  ",
            f"> 采图目录: `{self.image_dir}`  ",
            f"> 验证数据源: **{source_label}**  ",
            f"> 评审结论: **{grade_emoji} {grade_label}**",
            "",
            "---",
            "",
            "## 1. 全局核心指标",
            "",
            "| 指标 | 测量值 | PASS 门限 | ACCEPTABLE 门限 | 状态 |",
            "| :--- | :--- | :--- | :--- | :---: |",
            f"| **LOO 盲测中位像元误差** | **{stats['global_median_px']:.3f} px** | ≤ 1.50 px | ≤ 3.00 px | {status_median_px} |",
            f"| **LOO 盲测中位空间偏差** | **{stats['global_median_mm']:.3f} mm** | ≤ 1.00 mm | ≤ 2.50 mm | {status_median_mm} |",
            f"| **LOO 盲测均值像元误差** | {stats['global_mean_px']:.3f} px | — | — | — |",
            f"| **系统性偏差嫌疑 Tag 数** | {len(stats['flagged_tags'])} 个 | 0 个 | ≤ 2 个 | {status_flagged_tags} |",
            f"| **建议回审帧数** | {len(stats['flagged_frames'])} 帧 | 0 帧 | — | {status_flagged_frames} |",
            f"| **盲测执行总数** | {stats['total_tests']} 次 | — | — | — |",
            "",
            "## 2. Per-Tag 盲测精度统计",
            "",
            "| Tag ID | 盲测次数 | 中位误差 (px) | 均值 (px) | 最大 (px) | 标准差 (px) | 中位空间 (mm) | 状态 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for tid, s in sorted(stats["per_tag"].items()):
            is_flagged = tid in stats["flagged_tags"]
            status = "[FAIL] 偏差嫌疑" if is_flagged else ("[PASS]" if s["median_px"] <= 1.5 else "[WARN]")
            lines.append(
                f"| Tag #{tid:2d} | {s['count']:3d} | {s['median_px']:6.2f} | {s['mean_px']:6.2f} | "
                f"{s['max_px']:6.2f} | {s['std_px']:5.2f} | {s['median_mm']:6.2f} | {status} |"
            )

        # Per-Frame 统计
        lines.extend([
            "",
            "## 3. Per-Frame 盲测精度统计",
            "",
            "| 图像文件 | 盲测 Tag 数 | 帧均误差 (px) | 帧最大误差 (px) | 状态 |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for fname, s in sorted(stats["per_frame"].items()):
            is_flagged = fname in stats["flagged_frames"]
            status = "[FAIL] 建议回审" if is_flagged else ("[PASS]" if s["mean_px"] <= 1.5 else "[WARN]")
            lines.append(
                f"| {fname} | {s['tag_count']:2d} | {s['mean_px']:6.2f} | {s['max_px']:6.2f} | {status} |"
            )

        # 系统性偏差 Tag 详情
        if stats["flagged_tags"]:
            lines.extend([
                "",
                "## 4. 系统性偏差嫌疑标靶",
                "",
                "> [!WARNING]",
                f"> 以下 {len(stats['flagged_tags'])} 个标靶的盲测中位误差显著高于全局中位数的 2 倍 ({stats['global_median_px']:.2f} px × 2)，",
                "> 可能是 BA 地图中该 Tag 的空间坐标不够准确，建议检查相关观测样本并回审。",
                "",
            ])
            for tid in stats["flagged_tags"]:
                s = stats["per_tag"][tid]
                lines.append(f"- **Tag #{tid}**: 中位 {s['median_px']:.2f} px / {s['median_mm']:.2f} mm (测试 {s['count']} 次)")

        # 建议回审帧
        if stats["flagged_frames"]:
            lines.extend([
                "",
                "## 5. 建议回审帧",
                "",
                "> [!WARNING]",
                f"> 以下帧的平均盲测误差超过异常阈值 ({stats['outlier_threshold_px']:.2f} px)，建议在审核画板中检查并考虑剔除。",
                "",
            ])
            for fname in stats["flagged_frames"]:
                s = stats["per_frame"][fname]
                lines.append(f"- **{fname}**: 帧均 {s['mean_px']:.2f} px (最大: {s['max_px']:.2f} px)")

        # 可视化指引
        lines.extend([
            "",
            "## 6. 逐帧 LOO 盲测可视化图",
            "",
            f"分析图像已输出至: `{self.vis_dir}`",
            "",
            "- 浅灰虚线轮廓 = 物理实测角点位置",
            "- 彩色实线轮廓 = LOO 盲测反推投影位置 (绿色=误差达标, 橙色=误差偏高)",
            "- 彩色箭头 = 逐角点残差矢量 (已放大 15 倍)",
            "",
            "---",
            f"*报告由 `tag_offline_verifier.py` 自动生成*",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[OK] 离线精度体检报告已生成: {report_path}")
        return report_path

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

            # A. 退出指令全域绝对优先捕获 (零防抖冷却限制，只要点在退出热区立刻毫秒级关闭)
            # 1) 顶部 HUD 右上角关闭区域 (w - 120 <= x <= w 且 y <= top_bar_h + 8)
            # 2) 底部工具栏右下角区域 (x >= w - 150 且 y >= h - bot_bar_h - 8)
            # 3) 遍历命中了任何标记为 EXIT 的按钮矩形 (带 8px 扩充容差)
            is_exit_clicked = False
            if (w - 120) <= x and 0 <= y <= (top_bar_h + 8):
                is_exit_clicked = True
            elif (w - 150) <= x and (h - bot_bar_h - 8) <= y:
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

            for btn_id, (bx1, by1, bx2, by2), label, _ in self.gui_buttons:
                pad = 5
                if (bx1 - pad) <= x <= (bx2 + pad) and (by1 - pad) <= y <= (by2 + pad):
                    self._last_btn_click_time = now
                    if btn_id == "PREV":
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

    def toggle_flagged_filter(self):
        """切换是否仅浏览建议回审帧"""
        if not self.flagged_frames:
            self.set_toast("当前无任何建议回审帧，无需过滤！")
            self.filter_flagged_only = False
            return
        self.filter_flagged_only = not self.filter_flagged_only
        if self.filter_flagged_only:
            target_fname = self.flagged_frames[0]
            for idx, p in enumerate(self.image_files):
                if os.path.basename(p) == target_fname:
                    self.current_idx = idx
                    break
            self.set_toast(f"已开启【仅看回审帧模式】(共 {len(self.flagged_frames)} 帧)")
        else:
            self.set_toast(f"已恢复【全量浏览模式】(共 {len(self.image_files)} 帧)")

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

        status_tag = "FAIL (回审)" if is_cur_flagged else ("PASS" if cur_mean_px <= 1.5 else "WARN")
        status_bg = (0, 0, 180) if is_cur_flagged else ((0, 180, 0) if cur_mean_px <= 1.5 else (0, 140, 220))

        # 绘制帧状态标牌
        cv2.rectangle(canvas, (12, 10), (105, 48), status_bg, -1)
        cv2.rectangle(canvas, (12, 10), (105, 48), (255, 255, 255), 1)
        cv2.putText(canvas, status_tag, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # 帧文字明细
        idx_str = f"[{self.current_idx + 1:02d}/{len(self.image_files):02d}] {cur_fname}"
        cv2.putText(canvas, idx_str, (118, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 235, 255), 2, cv2.LINE_AA)
        detail_str = f"帧均: {cur_mean_px:.2f}px | 最大: {cur_max_px:.2f}px (Tag#{worst_tid}) | 盲测: {len(cur_results)} 靶"
        cv2.putText(canvas, detail_str, (118, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

        # 右侧全局仪表盘
        med_mm = self.stats.get("global_median_mm", 0.0)
        med_px = self.stats.get("global_median_px", 0.0)
        gate_ok = (med_mm <= 2.0 and med_px <= 3.0 and len(self.flagged_tags) == 0 and len(self.flagged_frames) == 0)
        gate_tag = "[PASS]" if gate_ok else "[FAIL]"
        gate_label = f"{gate_tag} 准入门限: 达标 (允许进入AR)" if gate_ok else f"{gate_tag} 准入门限: 超标 (需审核/重平差)"
        gate_color = (0, 240, 120) if gate_ok else (0, 100, 255)

        cv2.putText(canvas, gate_label, (w - 455, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, gate_color, 2, cv2.LINE_AA)
        glob_str = f"全局中位: {med_mm:.2f}mm ({med_px:.2f}px) | 嫌疑Tag: {self.flagged_tags or '无'} | 回审: {len(self.flagged_frames)} 帧"
        cv2.putText(canvas, glob_str, (w - 565, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # 初始化按钮列表 (涵盖顶部快捷关闭与底部全量控制栏)
        self.gui_buttons = []

        # 顶部 HUD 右上角快捷关闭按钮 (双保险交互设计，无论是右上角还是右下角均可秒退)
        top_exit_rect = (w - 105, 12, w - 12, top_bar_h - 12)
        if draw_styled_button:
            draw_styled_button(canvas, top_exit_rect, "[关闭 (Q)]", self.mouse_pos, btn_type="danger")
        self.gui_buttons.append(("EXIT", top_exit_rect, "退出", (0, 0, 255)))

        # 4. 绘制底部控制栏 (1:1 独立物理像素渲染，牢牢常驻在屏幕视线正下方)
        bot_overlay = canvas.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_bar_h), (w, h), (20, 22, 28), -1)
        cv2.addWeighted(bot_overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.line(canvas, (0, h - bot_bar_h), (w, h), (50, 54, 66), 1)

        # 组织底栏按钮与分段乒乓开关
        bx = 12
        by1 = h - bot_bar_h + 8
        by2 = h - 8
        mx, my = self.mouse_pos

        # A. 翻页按钮组
        for btn_id, label, bw in [("PREV", "< 上张 (A)", 96), ("NEXT", "下张 (D) >", 96)]:
            btn_rect = (bx, by1, bx + bw, by2)
            if draw_styled_button:
                draw_styled_button(canvas, btn_rect, label, self.mouse_pos, btn_type="normal")
            self.gui_buttons.append((btn_id, btn_rect, label, (255, 255, 255)))
            bx += bw + 6

        # B. 核心分段胶囊乒乓开关：[ 全量浏览 | 仅回审(N) (F) ]
        flagged_cnt = len(self.flagged_frames)
        filter_opts = [("ALL", "全量浏览"), ("FLAGGED", f"仅回审({flagged_cnt})")]
        active_idx = 1 if self.filter_flagged_only else 0
        seg_w = 210
        seg_rect = (bx, by1, bx + seg_w, by2)
        active_c = (35, 45, 175) if self.filter_flagged_only else (150, 95, 20)
        if draw_segmented_toggle:
            sub_rects = draw_segmented_toggle(canvas, seg_rect, filter_opts, active_idx, self.mouse_pos, shortcut="F", active_color=active_c)
            for key, srect in sub_rects:
                self.gui_buttons.append(("TOGGLE_FILTER", srect, "TOGGLE_FILTER", (255, 255, 255)))
        else:
            self.gui_buttons.append(("TOGGLE_FILTER", seg_rect, "TOGGLE_FILTER", (255, 255, 255)))
        bx += seg_w + 8

        # C. 业务操作按钮
        btn_label_3d = "[3D双棱柱(T)]" if self.view_mode_3d else "[2D残差(T)]"
        btn_type_3d = "info" if self.view_mode_3d else "normal"
        action_btns = [
            ("TOGGLE_3D", btn_label_3d, 125, btn_type_3d),
            ("OPEN_REVIEWER", "[审核此帧 (R)]", 120, "purple"),
            ("RUN_BA", "[BA平差 (B)]", 110, "primary"),
            ("ENTER_AR", "[在线验证 (V)]", 125, "success" if gate_ok else "normal"),
            ("EXPORT", "[导出报告 (S)]", 115, "normal"),
        ]
        for btn_id, label, bw, btype in action_btns:
            btn_rect = (bx, by1, bx + bw, by2)
            if draw_styled_button:
                draw_styled_button(canvas, btn_rect, label, self.mouse_pos, btn_type=btype)
            self.gui_buttons.append((btn_id, btn_rect, label, (255, 255, 255)))
            bx += bw + 6

        # D. 右侧退出按钮 (加宽至 110px，高辨识度，确保鼠标极易点击)
        exit_bw = 110
        exit_bx = w - exit_bw - 10
        exit_rect = (exit_bx, by1, exit_bx + exit_bw, by2)
        if draw_styled_button:
            draw_styled_button(canvas, exit_rect, "[退出 (Q)]", self.mouse_pos, btn_type="danger")
        self.gui_buttons.append(("EXIT", exit_rect, "退出", (0, 0, 255)))

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
            elif key in (ord('q'), ord('Q'), 27):  # Q / ESC
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
