# -*- coding: utf-8 -*-
"""
离线标定纯几何计算引擎 (OfflineVerificationEngine)
==================================================
单一职责设计：
  1. 仅负责纯三维视觉几何计算、PnP 位姿解算与正深度约束校验；
  2. 绝不掺杂任何 GUI 窗口渲染、鼠标事件、快捷键调度等交互逻辑；
  3. 既可作为 GUI 交互工作台的计算内核，亦可作为 CI/CD 自动化流水线的无头引擎。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class OfflineVerificationEngine:
    """离线标定纯几何计算引擎"""

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

    def solve_pnp(self, obj_flat: np.ndarray, img_flat: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """标准两阶段 SQPNP -> ITERATIVE 稳健 PnP 求解"""
        try:
            succ, rvec, tvec = cv2.solvePnP(obj_flat, img_flat, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            if succ:
                succ, rvec, tvec = cv2.solvePnP(obj_flat, img_flat, self.camera_matrix, self.dist_coeffs, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            return rvec, tvec, succ
        except Exception:
            return None, None, False

    def solve_single_tag_pnp(self, corners_2d: np.ndarray,
                             expected_z_cam: Optional[np.ndarray] = None) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:

        """根据单帧检出的 4 个 2D 角点解算单标靶实测相机外参位姿 (优先 IPPE_SQUARE，兜底 ITERATIVE)

        平面靶 PnP 天然存在二义性: IPPE 返回的两个候选解沿靶面内一轴相差约 180°
        (标靶法向/Z 轴翻转), 斜视 (约 45°) 时两解重投影误差之差缩小到像素噪声量级,
        仅按误差择优会间歇性选中翻转解。expected_z_cam 为标靶法向 (Z 轴) 在相机系下的
        先验方向 (来自地图理论位姿或"标靶朝向天空"先验), 提供后剔除法向与先验反向
        (dot<=0) 的翻转解, 再按重投影误差择优; 先验下无同向合格解时拒绝输出 (防错优先)。
        """
        z_exp = None
        if expected_z_cam is not None:
            z_exp = np.asarray(expected_z_cam, dtype=np.float64).reshape(3)
            n = float(np.linalg.norm(z_exp))
            z_exp = z_exp / n if n > 1e-9 else None
        try:
            c = corners_2d.reshape((4, 2)).astype(np.float64)
            succ, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                self.obj_points, c, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if succ and len(rvecs) > 0:
                best_r, best_t, min_err = None, None, float("inf")
                prior_r, prior_t, min_err_prior = None, None, float("inf")
                for r, t in zip(rvecs, tvecs):
                    if t[2, 0] <= 0:
                        continue
                    proj, _ = cv2.projectPoints(self.obj_points, r, t, self.camera_matrix, self.dist_coeffs)
                    err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - c, axis=1))
                    if err < min_err:
                        min_err = err
                        best_r, best_t = r, t
                    if z_exp is not None:
                        R_c, _ = cv2.Rodrigues(r)
                        if float(R_c[:, 2] @ z_exp) > 0.0 and err < min_err_prior:
                            min_err_prior = err
                            prior_r, prior_t = r, t
                if prior_r is not None:
                    return True, prior_r, prior_t
                if z_exp is None and best_r is not None:
                    return True, best_r, best_t
                # 有先验但候选解全部反向/深度非法: 落入兜底 (兜底同样做先验校验)
        except Exception:
            pass

        try:
            c = corners_2d.reshape((4, 2)).astype(np.float64)
            succ, r, t = cv2.solvePnP(self.obj_points, c, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if succ and t[2, 0] > 0:
                if z_exp is not None:
                    R_c, _ = cv2.Rodrigues(r)
                    if float(R_c[:, 2] @ z_exp) <= 0.0:
                        return False, None, None   # 与先验反向的翻转解, 拒绝输出
                return True, r, t
        except Exception:
            pass

        return False, None, None

# 别名导出 (向前向后兼容)
OfflineEngine = OfflineVerificationEngine
