#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线标定精度体检与 Leave-One-Out 盲测批量验证工具 (Tag Offline Verifier)
======================================================================
核心设计哲学：【独立于 BA 求解的第三方审判官】

本工具拿着 BA 求解完成的 tags_map.yaml，回头逐帧审判采集样本原图：
  1. 对原图重新检测标靶（独立于 BA 输入的观测清单，验证独立性最大化）；
  2. 对每帧每个可见 Tag 执行 Leave-One-Out 盲测：
     排除当前目标 Tag → 用其余 Tag 超定求解相机位姿 → 反推盲测目标的投影位置；
  3. 计算重投影像元误差 (px) 与空间绝对偏差 (mm)；
  4. 汇总 Per-Tag / Per-Frame 统计，自动识别系统性偏差与建议回审帧；
  5. 输出综合 Markdown 精度体检报告与逐帧可视化分析图。

纯离线批处理：不依赖相机硬件，仅需 config/tags_map.yaml + 采集样本图像。
======================================================================
"""

import os
import sys
import math
import time
import glob
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


class TagOfflineVerifier:
    """离线标定精度体检引擎：Leave-One-Out 盲测批量验证"""

    def __init__(self,
                 map_path: str = DEFAULT_MAP_PATH,
                 image_dir: str = DEFAULT_IMAGE_DIR,
                 marker_size_mm: float = 50.0,
                 source: str = "auto"):
        """
        初始化离线体检引擎
        :param map_path: tags_map.yaml 路径
        :param image_dir: 采集样本图像目录
        :param marker_size_mm: 标靶物理边长 (mm)，会被地图中的值覆盖
        :param source: 数据源 'auto' (优先清单), 'manifest' (仅已审核清单), 'images' (原图重新检测)
        """
        self.map_path = map_path
        self.image_dir = image_dir
        self.marker_size_mm = marker_size_mm
        self.source = (source or "auto").lower()

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

    def _compute_loo_error(self, rvec, tvec, tag_id: int, observed_corners: np.ndarray) -> Optional[Dict]:
        """
        计算单个盲测目标的重投影误差
        :return: {"err_px": float, "err_mm": float, "depth_mm": float, "proj_corners": np.ndarray}
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

        return {
            "err_px": mean_err_px,
            "err_mm": err_mm,
            "depth_mm": depth,
            "per_corner_px": per_corner_err.tolist(),
            "proj_corners": proj_2d,
            "obs_corners": obs
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

    def _render_verification_frame(self, img_path: str, frame_results: List[Dict], out_path: str):
        """渲染单帧 LOO 盲测可视化图"""
        img = cv2.imread(img_path)
        if img is None:
            return
        disp = img.copy()
        h, w = disp.shape[:2]

        # 顶部标题栏
        cv2.rectangle(disp, (0, 0), (w, 48), (18, 18, 18), -1)
        cv2.line(disp, (0, 48), (w, 48), (65, 65, 65), 1)

        base_name = os.path.basename(img_path)
        if frame_results:
            avg_err = np.mean([r["err_px"] for r in frame_results])
            max_item = max(frame_results, key=lambda r: r["err_px"])
            status = "PASS" if avg_err <= 1.5 else ("WARN" if avg_err <= 3.0 else "FAIL")
            status_color = (0, 255, 0) if status == "PASS" else ((0, 200, 255) if status == "WARN" else (0, 0, 255))
            header = f"LOO Blind Test | {base_name} | Avg: {avg_err:.2f}px | Worst: Tag#{max_item['blind_tag_id']} ({max_item['err_px']:.2f}px) | [{status}]"
        else:
            header = f"LOO Blind Test | {base_name} | Insufficient tags for blind test"
            status_color = (120, 120, 120)

        cv2.putText(disp, header, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

        for r in frame_results:
            obs = r["obs_corners"].astype(np.int32)
            proj = r["proj_corners"].astype(np.int32)
            tid = r["blind_tag_id"]
            err_px = r["err_px"]
            err_mm = r["err_mm"]

            # 观测轮廓 (浅灰虚线)
            cv2.polylines(disp, [obs], True, (160, 160, 160), 1, cv2.LINE_AA)

            # 反推投影轮廓 (绿/橙)
            is_good = err_px <= 1.5
            proj_color = (0, 255, 0) if is_good else (0, 80, 255)
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

            # 误差标牌
            cx = int(np.mean(obs[:, 0]))
            min_y = int(np.min(obs[:, 1]))
            badge_y = max(65, min_y - 14)
            badge_x = max(10, cx - 75)

            label = f"Tag#{tid} LOO: {err_px:.2f}px / {err_mm:.2f}mm"
            badge_bg = (20, 80, 20) if is_good else (20, 20, 80)
            badge_border = (0, 255, 0) if is_good else (0, 80, 255)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(disp, (badge_x - 4, badge_y - 16), (badge_x + tw + 6, badge_y + 4), badge_bg, -1)
            cv2.rectangle(disp, (badge_x - 4, badge_y - 16), (badge_x + tw + 6, badge_y + 4), badge_border, 1)
            cv2.putText(disp, label, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(out_path, disp)

    def _aggregate_statistics(self, all_results: List[Dict]) -> Dict:
        """
        汇总 Per-Tag / Per-Frame 统计，识别系统性偏差与建议回审帧
        """
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
            if stats["median_px"] > max(2.0, global_median_px * 2.0):
                flagged_tags.append(tid)

        # 坏帧检测 (基于 MAD 稳健统计，避免超大离群值掩盖坏样本):
        # 门限 = max(5.0, min(15.0, median + 3.0 * 1.4826 * MAD))
        mad_px = float(np.median(np.abs(all_px - global_median_px)))
        outlier_thresh = max(5.0, min(15.0, global_median_px + 3.0 * (1.4826 * mad_px if mad_px > 0.1 else 1.0)))
        flagged_frames = []
        for fname, stats in per_frame.items():
            if stats["mean_px"] > outlier_thresh or stats["max_px"] > 25.0:
                flagged_frames.append(fname)

        # 综合评级
        flagged_tag_count = len(flagged_tags)
        if global_median_px <= 1.5 and global_median_mm <= 1.0 and flagged_tag_count == 0:
            grade = "PASS"
            grade_label = "优异 (PASS)"
        elif global_median_px <= 3.0 and global_median_mm <= 2.5 and flagged_tag_count <= 2:
            grade = "ACCEPTABLE"
            grade_label = "良好 (ACCEPTABLE)"
        else:
            grade = "FAIL"
            grade_label = "不合格 (FAIL) — 建议回审后重新 BA 求解"

        return {
            "pass": grade != "FAIL",
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
        grade_emoji = "\u2705" if grade == "PASS" else ("\u26a0\ufe0f" if grade == "ACCEPTABLE" else "\u274c")

        status_median_px = "✅" if stats['global_median_px'] <= 1.5 else ("⚠️" if stats['global_median_px'] <= 3.0 else "❌")
        status_median_mm = "✅" if stats['global_median_mm'] <= 1.0 else ("⚠️" if stats['global_median_mm'] <= 2.5 else "❌")
        status_flagged_tags = "✅" if len(stats['flagged_tags']) == 0 else ("⚠️" if len(stats['flagged_tags']) <= 2 else "❌")
        status_flagged_frames = "✅" if len(stats['flagged_frames']) == 0 else "⚠️"

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
            f"| **LOO 盲测中位像元误差** | **{stats['global_median_px']:.3f} px** | \u2264 1.50 px | \u2264 3.00 px | {status_median_px} |",
            f"| **LOO 盲测中位空间偏差** | **{stats['global_median_mm']:.3f} mm** | \u2264 1.00 mm | \u2264 2.50 mm | {status_median_mm} |",
            f"| **LOO 盲测均值像元误差** | {stats['global_mean_px']:.3f} px | — | — | — |",
            f"| **系统性偏差嫌疑 Tag 数** | {len(stats['flagged_tags'])} 个 | 0 个 | \u2264 2 个 | {status_flagged_tags} |",
            f"| **建议回审帧数** | {len(stats['flagged_frames'])} 帧 | 0 帧 | — | {status_flagged_frames} |",
            f"| **盲测执行总数** | {stats['total_tests']} 次 | — | — | — |",
            "",
        ]

        # Per-Tag 统计
        lines.extend([
            "## 2. Per-Tag 盲测精度统计",
            "",
            "| Tag ID | 盲测次数 | 中位误差 (px) | 均值 (px) | 最大 (px) | 标准差 (px) | 中位空间 (mm) | 状态 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for tid, s in sorted(stats["per_tag"].items()):
            is_flagged = tid in stats["flagged_tags"]
            status = "\u274c \u504f\u5dee\u5acc\u7591" if is_flagged else ("\u2705" if s["median_px"] <= 1.5 else "\u26a0\ufe0f")
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
            status = "\u274c \u5efa\u8bae\u56de\u5ba1" if is_flagged else ("\u2705" if s["mean_px"] <= 1.5 else "\u26a0\ufe0f")
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
                f"> 以下 {len(stats['flagged_tags'])} 个标靶的盲测中位误差显著高于全局中位数的 2 倍 ({stats['global_median_px']:.2f} px \u00d7 2)，",
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

    def run_full_verification(self) -> Dict:
        """
        主入口：扫描全部采集图 → 逐帧重新检测 → LOO 盲测 → 汇总 → 渲染 → 报告
        """
        # 收集采集图像 (排除标注图和可视化图)
        disk_files = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.endswith("_annotated.png")
            and not f.endswith("_quiver.png")
            and not f.endswith("_loo_verify.png")
            and "visualized" not in f
        ])

        if not disk_files:
            print(f"[WARN] 采图目录为空: {self.image_dir}")
            return {}

        source_label = "已审核观测清单 (tag_observations.yaml)" if self.actual_source == "manifest" else "原始采图重新检测 (端到端独立复检)"

        print("\n" + "=" * 80)
        print("  【离线标定精度体检与 Leave-One-Out 盲测批量验证】")
        print("=" * 80)
        print(f"  -> 标靶地图     : {self.map_path} ({len(self.mapped_tag_ids)} 个标靶)")
        print(f"  -> 采图目录     : {self.image_dir} ({len(disk_files)} 张图像)")
        print(f"  -> 验证数据源   : {source_label}")
        print(f"  -> 验证策略     : 鲁棒 RANSAC PnP + 逐 Tag Leave-One-Out 盲测")
        print(f"  -> 输出报告目录 : {VERIFICATION_DIR}")
        print("=" * 80 + "\n")

        all_results = []
        t0 = time.time()

        for idx, fpath in enumerate(disk_files):
            fname = os.path.basename(fpath)
            t_f = time.time()

            frame_results = self._verify_single_frame(fpath)
            all_results.extend(frame_results)

            # 渲染可视化图
            vis_path = os.path.join(self.vis_dir, os.path.splitext(fname)[0] + "_loo_verify.png")
            self._render_verification_frame(fpath, frame_results, vis_path)

            dt = (time.time() - t_f) * 1000.0
            if frame_results:
                errs = [r["err_px"] for r in frame_results]
                print(f"  [{idx+1:02d}/{len(disk_files):02d}] {fname:<14} -> {len(frame_results):2d} LOO \u76f2\u6d4b | "
                      f"\u5747: {np.mean(errs):.2f}px | \u6700\u5927: {np.max(errs):.2f}px | ({dt:.0f}ms)")
            else:
                print(f"  [{idx+1:02d}/{len(disk_files):02d}] {fname:<14} -> \u53ef\u89c1\u6807\u9776\u4e0d\u8db3 2 \u4e2a\uff0c\u8df3\u8fc7 ({dt:.0f}ms)")

        total_time = time.time() - t0

        # 汇总统计
        stats = self._aggregate_statistics(all_results)

        # 输出报告
        self._export_markdown_report(stats, all_results)

        # 终端汇总
        grade_label = stats.get("grade_label", "N/A")
        print("\n" + "=" * 80)
        print("  \u3010\u79bb\u7ebf\u7cbe\u5ea6\u4f53\u68c0\u6c47\u603b\u5927\u6210\u62a5\u8868\u3011")
        print("=" * 80)
        print(f"  -> \u603b\u5904\u7406\u5e27\u6570   : {len(disk_files)} \u5f20 (\u603b\u8017\u65f6: {total_time:.1f}s)")
        print(f"  -> LOO \u76f2\u6d4b\u603b\u6570 : {stats.get('total_tests', 0)} \u6b21")
        print(f"  -> \u4e2d\u4f4d\u50cf\u5143\u8bef\u5dee : {stats.get('global_median_px', 0):.3f} px")
        print(f"  -> \u4e2d\u4f4d\u7a7a\u95f4\u504f\u5dee : {stats.get('global_median_mm', 0):.3f} mm")
        if stats.get("flagged_tags"):
            print(f"  -> \u504f\u5dee\u5acc\u7591\u6807\u9776 : {stats['flagged_tags']} (\u5efa\u8bae\u91cd\u70b9\u68c0\u67e5)")
        if stats.get("flagged_frames"):
            print(f"  -> \u5efa\u8bae\u56de\u5ba1\u5e27   : {stats['flagged_frames']}")
        print(f"  -> \u7efc\u5408\u8bc4\u7ea7     : {grade_label}")
        print("=" * 80 + "\n")

        return stats


def main():
    parser = argparse.ArgumentParser(description="离线标定精度体检与 Leave-One-Out 盲测批量验证")
    parser.add_argument("--map", type=str, default=DEFAULT_MAP_PATH, help="tags_map.yaml 路径")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="采集样本图像目录")
    parser.add_argument("--marker_size", type=float, default=50.0, help="标靶物理边长 (mm)")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "manifest", "images"],
                        help="数据源: auto (优先清单), manifest (仅已审核清单), images (原图重检测)")
    args = parser.parse_args()

    verifier = TagOfflineVerifier(
        map_path=args.map,
        image_dir=args.image_dir,
        marker_size_mm=args.marker_size,
        source=args.source
    )
    verifier.run_full_verification()


if __name__ == "__main__":
    main()
