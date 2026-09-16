# -*- coding: utf-8 -*-
"""
离线标定精度体检与 Leave-One-Out 盲测纯计算引擎 (OfflineVerificationEngine)
========================================================================
单一职责设计：
  1. 仅负责纯三维视觉几何计算、超定 PnP 位姿解算与正深度约束校验；
  2. 独立执行 Leave-One-Out (LOO) 逐 Tag 盲测重投影与空间毫米偏差度量；
  3. 绝不掺杂任何 GUI 窗口渲染、鼠标事件、快捷键调度等交互逻辑；
  4. 既可作为 GUI 交互工作台的计算内核，亦可作为 CI/CD 自动化流水线的无头引擎。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class OfflineVerificationEngine:
    """离线标定精度体检纯计算引擎"""

    def __init__(
        self,
        tags_map: Dict[str, Any],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_size_mm: float = 50.0,
        valid_tag_ids: Optional[List[int]] = None
    ):
        """
        初始化纯计算引擎
        :param tags_map: tags_map.yaml 字典对象
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数向量
        :param marker_size_mm: 标靶物理边长 (mm)
        :param valid_tag_ids: 允许参与计算的有效 Tag ID 白名单
        """
        self.tags_map = tags_map
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        self.marker_size_mm = float(marker_size_mm)
        self.valid_tag_ids = valid_tag_ids

        self.mapped_tag_ids = sorted([int(tid) for tid in self.tags_map.get("tags", {}).keys()])

        # 标靶局部坐标系角点 (顺序: 左上, 右上, 右下, 左下)
        s = self.marker_size_mm / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

        # 双路互补检测器初始化 (用于独立原图复检模式)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self.detector_bright, self.detector_dark = self._build_detectors()

    def set_marker_size_mm(self, size_mm: float) -> None:
        """更新标靶物理边长并重建单靶 PnP 物理角点模型 (与地图 BA 反算真实边长保持一致)"""
        if not size_mm or float(size_mm) <= 0:
            return
        self.marker_size_mm = float(size_mm)
        s = self.marker_size_mm / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

    def _build_detectors(self):
        """构建双路互补检测器 (高光路与暗部动态拉伸路)"""
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
        """双路互补融合检测标靶角点"""
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

        # 路 2: 低反差路 (动态百分比拉伸)
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

    def get_tag_world_transform(self, tag_id: int) -> Optional[np.ndarray]:
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

    def get_tag_world_corners(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶 4 个角点在世界坐标系下的 3D 位置"""
        T = self.get_tag_world_transform(tag_id)
        if T is None:
            return None
        s = self.marker_size_mm / 2.0
        local = np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)
        return (T @ local.T).T[:, :3]

    def solve_camera_pose(self, tag_corners_pairs: List[Tuple[int, np.ndarray]]) -> Optional[Dict[str, Any]]:
        """
        给定一组 (tag_id, corners_2d) 对，超定 PnP 求解相机位姿。
        集成物理正深度前置校验与 SQPNP -> EPNP -> ITERATIVE 多级回退机制，彻底杜绝翻转伪解。
        :return: {"rvec", "tvec", "rmse"} 或 None
        """
        all_obj = []
        all_img = []
        for tid, c2d in tag_corners_pairs:
            wc = self.get_tag_world_corners(tid)
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

    def solve_pnp(self, obj_flat: np.ndarray, img_flat: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """标准两阶段 SQPNP -> ITERATIVE 稳健 PnP 求解"""
        try:
            succ, rvec, tvec = cv2.solvePnP(obj_flat, img_flat, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            if succ:
                succ, rvec, tvec = cv2.solvePnP(obj_flat, img_flat, self.camera_matrix, self.dist_coeffs, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            return rvec, tvec, succ
        except Exception:
            return None, None, False

    def solve_single_tag_pnp(self, corners_2d: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:

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

    def compute_loo_error(self, rvec: np.ndarray, tvec: np.ndarray, tag_id: int, observed_corners: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        计算单个盲测目标的重投影误差，并记录 BA 理论位姿与单帧实测位姿
        """
        T_w_t = self.get_tag_world_transform(tag_id)
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

        depth = float(t_t[2, 0])
        fx = self.camera_matrix[0, 0]
        err_mm = (mean_err_px * abs(depth)) / fx if fx > 0 else 0.0

        ok_obs, obs_r, obs_t = self.solve_single_tag_pnp(obs)

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
            "obs_tvec": obs_t if ok_obs else None,
        }

    def verify_single_frame(
        self,
        img_path: str,
        manifest_data: Optional[Dict] = None,
        source: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        对单帧执行全标靶 Leave-One-Out 盲测循环
        :param img_path: 图像路径
        :param manifest_data: 观测清单缓存数据
        :param source: "auto" / "manifest" / "images"
        :return: 该帧所有 LOO 测试结果列表
        """
        fname = os.path.basename(img_path)
        known_tags = {}

        if source in ("auto", "manifest") and manifest_data:
            img_entry = manifest_data.get("images", {}).get(fname, {})
            if not img_entry.get("enabled", True):
                return []
            for obs in img_entry.get("observations", []):
                tid = obs.get("tag_id")
                if obs.get("keep", True) and tid in self.mapped_tag_ids:
                    c = np.array(obs["corners"], dtype=np.float64)
                    area = cv2.contourArea(c.reshape(4, 2).astype(np.float32))
                    if area >= 250.0:
                        known_tags[tid] = c

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
            solver_pairs = [(tid, c) for tid, c in known_tags.items() if tid != blind_tid]
            if len(solver_pairs) < 1:
                continue

            pose = self.solve_camera_pose(solver_pairs)
            if pose is None or pose["rmse"] > 30.0:
                continue

            loo = self.compute_loo_error(pose["rvec"], pose["tvec"], blind_tid, known_tags[blind_tid])
            if loo is None:
                continue

            results.append({
                "image": fname,
                "blind_tag_id": blind_tid,
                "solver_tags": [tid for tid, _ in solver_pairs],
                "solver_count": len(solver_pairs),
                "solver_rmse_px": pose["rmse"],
                **loo
            })

        return results


# 别名导出 (向前向后兼容)
OfflineEngine = OfflineVerificationEngine
