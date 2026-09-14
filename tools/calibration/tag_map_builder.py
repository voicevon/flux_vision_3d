#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标靶 (AprilTag 16h5) 空间地图建图与全局平差求解工具 (Tag Map Builder)
- 基于多视角重叠图像构建标靶共视连通图 (Co-visibility Graph)
- 基于 Bundle Adjustment (BA) 联合优化静止标靶 3D 空间位姿与相机位姿
- 支持 Tag 0 (SCARA J1 旋转中心原点) 与 Tag 1 (世界 +X 轴基准) 的刚体对齐闭环
- 导出 config/tags_map.yaml 供运行时毫秒级在线相机定位
"""

import os
import sys
import glob
import math
import yaml
import argparse
from datetime import datetime
import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional, Any, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

from src.calibration.covisibility_graph import (
    CovisibilityGraphAnalyzer,
    CovisibilityGraphError,
    print_topology_report as _print_topology_report_impl
)
from src.calibration.manifest_repository import ManifestRepository
from src.calibration.ba_optimizer import BundleAdjustmentOptimizer


class TagMapBuilder:
    def __init__(self, 
                 marker_size_mm: float = 50.0,
                 tag_family: int = cv2.aruco.DICT_APRILTAG_16h5,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        初始化建图求解器
        :param marker_size_mm: 标靶黑白边框物理边长 (毫米)
        :param tag_family: OpenCV ArUco 字典枚举 (默认 AprilTag 16h5)
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数
        """
        self.marker_size_mm = float(marker_size_mm)
        self.dictionary = cv2.aruco.getPredefinedDictionary(tag_family)
        
        # 现代 OpenCV 4.7+ ArucoDetector 接口与工业级高灵敏度超参数
        # 尝试读取 config.yaml 中的反差门限配置
        contrast_boost = 1.6
        clahe_clip = 4.0
        min_otsu = 1.5
        min_perim = 0.006
        thresh_const = 3.0
        valid_tag_ids = []
        try:
            cfg_file = "config.yaml"
            if os.path.exists(cfg_file):
                with open(cfg_file, "r", encoding="utf-8") as f:
                    c = yaml.safe_load(f) or {}
                valid_tag_ids = [int(x) for x in c.get("calibration", {}).get("valid_tag_ids", [])]
                d_cfg = c.get("calibration", {}).get("tag_detection", {})
                contrast_boost = float(d_cfg.get("contrast_boost", contrast_boost))
                clahe_clip = float(d_cfg.get("clahe_clip_limit", clahe_clip))
                min_otsu = float(d_cfg.get("min_otsu_std_dev", min_otsu))
                min_perim = float(d_cfg.get("min_perimeter_rate", min_perim))
                thresh_const = float(d_cfg.get("adaptive_thresh_constant", thresh_const))
        except Exception:
            pass

        self.valid_tag_ids = valid_tag_ids
        self.contrast_boost = contrast_boost
        self.min_otsu_std_dev = min_otsu

        # 基础参数模版 (兼顾高精度亚像素角点与大透视/低反差/发灰墨色)
        def make_params(thresh_c, min_otsu):
            p = cv2.aruco.DetectorParameters()
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 43
            p.adaptiveThreshWinSizeStep = 8
            p.adaptiveThreshConstant = thresh_c
            p.minOtsuStdDev = min_otsu
            p.minMarkerPerimeterRate = max(0.008, min_perim)
            p.maxMarkerPerimeterRate = 4.0
            p.polygonalApproxAccuracyRate = 0.09
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            p.perspectiveRemovePixelPerCell = 10
            p.errorCorrectionRate = 0.50
            p.perspectiveRemoveIgnoredMarginPerCell = 0.15
            p.maxErroneousBitsInBorderRate = 0.30
            return p

        # 构建两路互补检测器：
        # 路 A: 抗反光/高亮清晰路 (C=5.5, 适合高光、灯光直射视角)
        self.params_bright = make_params(thresh_c=5.5, min_otsu=0.55)
        self.detector_bright = cv2.aruco.ArucoDetector(self.dictionary, self.params_bright)
        
        # 路 B: 低反差/黑度不纯路 (C=2.5, min_otsu=0.45, 适合背光、发灰视角)
        self.params_dark = make_params(thresh_c=2.5, min_otsu=0.45)
        self.detector_dark = cv2.aruco.ArucoDetector(self.dictionary, self.params_dark)
        self.detector = self.detector_bright

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

        # 相机内参与畸变 (若未指定，优先从 config_guard 加载并自适应)
        if camera_matrix is None:
            if resolve_camera_intrinsics is not None:
                K, dist, _ = resolve_camera_intrinsics("config.yaml")
                self.camera_matrix = K
                self.dist_coeffs = dist
            else:
                self.camera_matrix = np.array([
                    [1363.68, 0.0, 971.19],
                    [0.0, 1361.19, 566.26],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float64)

        # 标靶局部坐标系下的 4 个角点物理坐标 (逆时针, Z=0)
        s = self.marker_size_mm / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

        # 清单与地图仓储管理器
        self.repository = ManifestRepository(builder=self)

        # 专业 BA 平差优化求解器
        self.ba_optimizer = BundleAdjustmentOptimizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm,
            obj_points=self.obj_points,
            builder=self
        )

    def detect_tags(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """
        双路互补融合全景检测（彻底解决反光、黑度不纯、倾斜与远景漏检）：
        路 1 (抗反光/高光路): 原图灰度 + 较严二值门限 C=5.5，精准捕获高光、灯光直射区域标靶
        路 2 (低反差/暗部路): 动态直方图拉伸 + 宽松二值门限 C=2.5 + minOtsu=0.45，攻克发灰、低反差、暗部标靶
        白名单机制硬锁保底，两路检测结果快速取并集，杜绝误检与漏检。
        :return: {tag_id: corners_4x2}
        """
        if len(image.shape) == 3:
            gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_raw = image

        results = {}

        # 路 1: 针对清晰/高光/反光区域
        c1, ids1, _ = self.detector_bright.detectMarkers(gray_raw)
        if ids1 is not None and len(ids1) > 0:
            for idx, tag_id in enumerate(ids1.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                results[tid] = c1[idx].reshape((4, 2))

        # 路 2: 针对暗部/低反差/打印黑度不够纯区域 (动态拉伸 + 宽松门限)
        p_low, p_high = np.percentile(gray_raw[::4, ::4], (2, 98))
        if p_high > p_low + 10:
            gray_stretch = np.clip((gray_raw.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            gray_stretch = gray_raw

        c2, ids2, _ = self.detector_dark.detectMarkers(gray_stretch)
        if ids2 is not None and len(ids2) > 0:
            for idx, tag_id in enumerate(ids2.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                if tid not in results:
                    results[tid] = c2[idx].reshape((4, 2))

        # 极致高精度优化：对检测到的所有标靶执行局部梯度协方差高阶亚像素二次精修
        for tid in list(results.keys()):
            results[tid] = self.refine_corners_subpix(gray_raw, results[tid])

        return results

    def refine_corners_subpix(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        基于梯度自相关矩阵对 AprilTag 4 个角点进行高阶亚像素二次精修：
        - 窗口大小基于标靶平均边长自适应计算 (标靶尺度的 6%，夹紧在 3~9 像素之间)
        - 零区域 (-1, -1) 规避自相关矩阵退化
        - 严格迭代终止准则：40 次迭代或精度达到 0.001 像素
        - 包含异常漂移防呆：若精修漂移超过 2.5 像素，安全回退到原角点
        """
        try:
            pts = corners.reshape((4, 2)).astype(np.float32)
            side1 = np.linalg.norm(pts[0] - pts[1])
            side2 = np.linalg.norm(pts[1] - pts[2])
            avg_side = (side1 + side2) / 2.0
            
            half_win = int(np.clip(avg_side * 0.06, 3, 9))
            win_size = (half_win, half_win)
            zero_zone = (-1, -1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            
            # cornerSubPix 要求输入 float32
            refined = cv2.cornerSubPix(gray, pts.copy(), win_size, zero_zone, criteria)
            diff = np.linalg.norm(refined - pts, axis=1)
            if np.max(diff) > 2.5:
                # 局部边缘或反光导致非正常跳变，安全保底
                return pts.astype(np.float64)
            return refined.astype(np.float64)
        except Exception:
            return corners.astype(np.float64)

    def compute_observation_weight(self, corners: np.ndarray, 
                                   rvec: Optional[np.ndarray] = None, 
                                   tvec: Optional[np.ndarray] = None) -> float:
        """
        计算单次观测标靶的信息矩阵综合权重 w in [0.0, 1.0]：
        1. 像面有效像素面积权重 (远景小标靶像素信噪比低，适度衰减)
        2. 正对入射夹角权重 (大倾角透视掠射视角不确定度剧增，强力衰减)
        3. 径向畸变边缘衰减权重 (图像四周边缘相差区域适度衰减)
        4. 极小噪点/虚警直接置 0 (拒绝面积 < 100px^2 或 Z > 2500mm 的离群杂波)
        """
        pts = corners.reshape((4, 2)).astype(np.float64)
        area = float(cv2.contourArea(pts.astype(np.float32)))
        
        # 守门员：面积微小严重退化
        if area < 100.0:
            return 0.0

        if rvec is None or tvec is None:
            succ, rvec, tvec = self.solve_single_tag_pnp(pts)
            if not succ:
                return 0.1

        z = float(tvec[2][0])
        # 守门员：单靶解算深度异常（超过 2.5 米或小于 150mm）
        if z > 2500.0 or z < 150.0:
            return 0.0

        # 1. 面积权重 (以 3600 px^2 约 60x60px 为标准)
        w_area = float(np.clip(area / 3600.0, 0.25, 1.0))

        # 2. 正对夹角权重 (法向量与光轴夹角)
        R, _ = cv2.Rodrigues(rvec)
        # 标靶法向量在相机系下的朝向为 R[:, 2]
        # 当正对相机时，其 Z 分量朝向相机光心即 R[2, 2] 接近 -1
        cos_theta = abs(float(R[2, 2]))
        if cos_theta >= 0.85:
            w_angle = 1.0
        elif cos_theta >= 0.50:
            w_angle = 0.5 + 0.5 * (cos_theta - 0.50) / 0.35
        else:
            w_angle = max(0.15, cos_theta / 0.50 * 0.5)

        # 3. 径向边缘衰减 (根据中心距离)
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        center = np.mean(pts, axis=0)
        dist_from_center = np.linalg.norm(center - np.array([cx, cy]))
        max_radius = np.sqrt(cx**2 + cy**2)
        r_norm = dist_from_center / max_radius
        if r_norm <= 0.65:
            w_radial = 1.0
        else:
            w_radial = float(np.clip(1.0 - (r_norm - 0.65) * 1.2, 0.4, 1.0))

        return float(np.clip(w_area * w_angle * w_radial, 0.1, 1.0))

    def solve_single_tag_pnp(self, corners: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        对单标靶执行 PnP 获得其在相机系下的位姿 (rvec, tvec)
        集成 IPPE_SQUARE 翻转二义性智能消歧 (Planar Ambiguity Disambiguation):
        当由于图像噪点导致对称翻转的伪解重投影误差极小时，
        基于物理几何先验（相机俯视拍摄工作台，标靶法向量 Z 轴必须向上立起即 R[1, 2] < 0）
        精准筛选物理真实解，彻底杜绝“Z 轴倒栽葱”或“反向刺入工作台”。
        """
        corners_2d = corners.reshape((4, 2)).astype(np.float64)
        retval, rvecs, tvecs, reprojErrors = cv2.solvePnPGeneric(
            self.obj_points,
            corners_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not retval or len(rvecs) == 0:
            # 降级尝试 ITERATIVE
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corners_2d,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            return success, rvec, tvec

        if len(rvecs) == 1:
            return True, rvecs[0], tvecs[0]

        # 当存在 2 个对称候选解时（典型 IPPE 平面双解）
        err0 = reprojErrors[0][0] if reprojErrors is not None else 0.0
        err1 = reprojErrors[1][0] if reprojErrors is not None else 0.0

        # 如果两个解的误差非常接近（相差在 2.5 像素以内，处于典型二义性退化带）
        # 使用工作台法向量向上先验 (在 OpenCV 相机坐标系中，俯视工作台时，向上法向量的 Y 分量 R[1, 2] < 0)
        best_idx = 0
        if abs(err0 - err1) < 2.5:
            R0, _ = cv2.Rodrigues(rvecs[0])
            R1, _ = cv2.Rodrigues(rvecs[1])
            # R[:, 2] 为标靶 Z 轴 (法向量) 在相机系下的方向
            # R[1, 2] 对应相机坐标系 Y 轴 (向下) 的投影。若标靶向上挺拔，则 R[1, 2] 应为负数 (朝向天空)
            if R0[1, 2] >= 0 and R1[1, 2] < 0:
                best_idx = 1
            elif R1[1, 2] >= 0 and R0[1, 2] < 0:
                best_idx = 0
            else:
                best_idx = 0 if err0 <= err1 else 1
        else:
            best_idx = 0 if err0 <= err1 else 1

        return True, rvecs[best_idx], tvecs[best_idx]

    def render_tag_3d_axes(self, img: np.ndarray, corners: np.ndarray, tag_id: int):
        """
        在图像上绘制标靶 3D 空间坐标系与实心正四棱柱：
        - X 轴 (红色, 25mm), Y 轴 (绿色, 25mm)
        - Z 轴指示: 边长 20.0mm x 20.0mm (原始标靶 1/2)、长 120.0mm 的实心正四棱柱 (半透明实心柱体 + 12条高亮棱线 + 顶盖透视截面)
        """
        try:
            corners_2d = corners.reshape((4, 2)).astype(np.float64)
            ok, rvec, tvec = self.solve_single_tag_pnp(corners_2d)
            if not ok:
                return

            hw = 15.0   # 截面半宽 15mm，整体截面边长为 30.0mm x 30.0mm
            L = 80.0    # 柱体长度 80mm (约原高度 2/3)

            # 8 个 3D 角点: 底面 4 点 (Z=0), 顶面 4 点 (Z=L)
            pts_3d = np.array([
                # 底面 4 点
                [-hw, -hw, 0.0],
                [ hw, -hw, 0.0],
                [ hw,  hw, 0.0],
                [-hw,  hw, 0.0],
                # 顶面 4 点
                [-hw, -hw, L],
                [ hw, -hw, L],
                [ hw,  hw, L],
                [-hw,  hw, L],
                # 顶面中心
                [0.0, 0.0, L],
                # X 轴与 Y 轴参考端点 (从中心伸出 25mm，突出棱柱外侧)
                [25.0, 0.0, 0.0],
                [0.0, 25.0, 0.0],
                [0.0, 0.0, 0.0]
            ], dtype=np.float64)

            proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            proj = proj.reshape((-1, 2)).astype(int)

            b_pts = proj[0:4] # 底面 4 点
            t_pts = proj[4:8] # 顶面 4 点
            top_center = tuple(proj[8])
            p_x = tuple(proj[9])
            p_y = tuple(proj[10])
            p_orig = tuple(proj[11])

            # 1. 绘制 X 轴 (红色) 和 Y 轴 (绿色)
            cv2.line(img, p_orig, p_x, (0, 0, 240), 2, cv2.LINE_AA)
            cv2.putText(img, 'X', p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(img, p_orig, p_y, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(img, 'Y', p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 2. 半透明填充 4 个侧面与顶面 (呈现实心方柱体立体质感)
            overlay = img.copy()
            for i in range(4):
                next_i = (i + 1) % 4
                side_poly = np.array([b_pts[i], b_pts[next_i], t_pts[next_i], t_pts[i]], dtype=np.int32)
                cv2.fillPoly(overlay, [side_poly], (240, 160, 30)) # BGR: 浅蓝/青
            cv2.fillPoly(overlay, [t_pts], (255, 220, 90))         # 顶面高光
            cv2.addWeighted(overlay, 0.42, img, 0.58, 0, img)

            # 3. 绘制 12 条棱线 (高清晰边框)
            cv2.polylines(img, [b_pts], isClosed=True, color=(180, 80, 0), thickness=2, lineType=cv2.LINE_AA)
            for i in range(4):
                cv2.line(img, tuple(b_pts[i]), tuple(t_pts[i]), (255, 130, 0), 2, cv2.LINE_AA)
            cv2.polylines(img, [t_pts], isClosed=True, color=(255, 240, 120), thickness=2, lineType=cv2.LINE_AA)

            # 4. 顶面中心标注点与文字 (简洁工业标定, 仅保留 Z 轴标识)
            cv2.circle(img, top_center, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, 'Z', (top_center[0] + 5, top_center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 80), 2, cv2.LINE_AA)
        except Exception:
            pass

    @staticmethod
    def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """旋转向量与平移向量转 4x4 齐次矩阵"""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    @staticmethod
    def matrix_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """4x4 齐次矩阵转旋转向量与平移向量"""
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3].reshape((3, 1))
        return rvec, tvec

    def compute_tag_metrics(self, corners: np.ndarray) -> Dict[str, Any]:
        """
        根据标靶 4 个角点物理投影计算空间像元分辨率及几何指标
        """
        pts = corners.reshape((4, 2)).astype(np.float64)
        l01 = float(np.linalg.norm(pts[1] - pts[0]))
        l12 = float(np.linalg.norm(pts[2] - pts[1]))
        l23 = float(np.linalg.norm(pts[3] - pts[2]))
        l30 = float(np.linalg.norm(pts[0] - pts[3]))
        cell_w = int(round((l01 + l23) / 12.0))
        cell_h = int(round((l30 + l12) / 12.0))
        center_x = float(np.mean(pts[:, 0]))
        center_y = float(np.mean(pts[:, 1]))
        area = float(cv2.contourArea(pts.astype(np.float32)))
        return {
            "cell_size_px": [cell_w, cell_h],
            "center_px": [round(center_x, 1), round(center_y, 1)],
            "area_px": round(area, 1),
            "edge_lengths_px": [round(l01, 1), round(l12, 1), round(l23, 1), round(l30, 1)]
        }

    def render_annotated_frame(self, image: np.ndarray, detected_tags: Dict[int, np.ndarray]) -> np.ndarray:
        """
        渲染完整图示化分析图像 (带绿色轮廓线、彩色角点、像元分辨率标牌与 3D 正四棱柱)
        """
        disp = image.copy()
        dot_colors = [
            (0, 0, 255),    # 角点 0: 红色
            (0, 255, 0),    # 角点 1: 绿色
            (255, 0, 0),    # 角点 2: 蓝色
            (0, 255, 255)   # 角点 3: 黄色
        ]
        for tag_id, corners in detected_tags.items():
            pts = corners.reshape((4, 2)).astype(np.int32)
            # 绘制绿色轮廓线
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            # 绘制 4 个角点彩色圆点
            for pt_idx, pt in enumerate(pts):
                cv2.circle(disp, tuple(pt), 5, dot_colors[pt_idx], -1)

            # 计算机械标靶单元方格像素尺寸 (AprilTag 16h5 为 6x6 网格)
            metrics = self.compute_tag_metrics(corners)
            cell_w, cell_h = metrics["cell_size_px"]

            # 绘制 ID 与最小单元方格像素分辨率标牌
            cx = int(metrics["center_px"][0])
            min_y = int(np.min(pts[:, 1]))
            tag_text = f"Tag {tag_id}" + (" [ORIGIN]" if tag_id == 0 else "")
            cell_text = f"Cell: {cell_w}x{cell_h}px"
            
            badge_x = cx - 50
            badge_y = max(42, min_y - 12)
            cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 106, badge_y + 8), (20, 20, 20), -1)
            cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 106, badge_y + 8), (0, 255, 255), 1)
            cv2.putText(disp, tag_text, (badge_x, badge_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, cell_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

            # 绘制 3D 空间坐标系 (实心正四棱柱)
            self.render_tag_3d_axes(disp, corners, tag_id)

        return disp

    def export_observations_manifest(self, 
                                     image_paths: List[str], 
                                     manifest_path: str = "data/tag_calibration_images/tag_observations.yaml",
                                     generate_visualized: bool = True) -> str:
        """
        两阶段建图流水线 - 阶段一：
        扫描多视角图像，提取标靶观测数据并导出为结构化审核清单 (YAML)
        委托至专职仓储类 ManifestRepository 处理
        """
        return self.repository.export_manifest(
            image_paths=image_paths,
            manifest_path=manifest_path,
            generate_visualized=generate_visualized
        )

    def load_observations_manifest(self, manifest_path: str = "data/tag_calibration_images/tag_observations.yaml") -> Tuple[List[Dict[int, np.ndarray]], List[str], Dict[str, Any]]:
        """
        两阶段建图流水线 - 阶段二：
        从审核清单中加载已审核的标靶观测数据，并过滤掉 keep: false 的坏样本。
        委托至专职仓储类 ManifestRepository 处理
        """
        return self.repository.load_manifest(manifest_path=manifest_path)

    def validate_covisibility(self, 
                              frame_detections: List[Dict[int, np.ndarray]], 
                              frame_names: Optional[List[str]] = None,
                              origin_tag_id: int = 0,
                              x_align_tag_id: int = 1) -> Dict[str, Any]:
        """
        共视连通性安全守门员 (Co-visibility Graph Connectivity Guard)
        委托至独立领域类 CovisibilityGraphAnalyzer 处理
        """
        return CovisibilityGraphAnalyzer.analyze(
            frame_detections=frame_detections,
            frame_names=frame_names,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id
        )

    def optimize_bundle_adjustment(self, 
                                   frame_detections: List[Dict[int, np.ndarray]], 
                                   active_frame_names: Optional[List[str]] = None,
                                   origin_tag_id: int = 0,
                                   x_align_tag_id: int = 1,
                                   baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict[str, Any]:
        """
        基于非线性最小二乘 (Bundle Adjustment) 联合优化所有标靶位姿与相机位姿
        委托至独立专业求解器 BundleAdjustmentOptimizer 处理
        """
        self.ba_optimizer.camera_matrix = self.camera_matrix
        self.ba_optimizer.dist_coeffs = self.dist_coeffs
        self.ba_optimizer.marker_size_mm = self.marker_size_mm
        self.ba_optimizer.obj_points = self.obj_points

        result = self.ba_optimizer.optimize(
            frame_detections=frame_detections,
            active_frame_names=active_frame_names,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )
        self.marker_size_mm = self.ba_optimizer.marker_size_mm
        return result

    def compute_3d_uncertainties(self, jacobian, static_tags, base_id, sigma_res_px) -> Dict[int, Dict[str, float]]:
        """委托计算 3D 标靶空间坐标一阶协方差置信区间"""
        return self.ba_optimizer.compute_3d_uncertainties(jacobian, static_tags, base_id, sigma_res_px)

    def export_diagnostic_report(self,
                                 final_tags_map: Dict[str, Any],
                                 detailed_obs_res: List[Dict[str, Any]],
                                 active_frame_names: List[str],
                                 tag_uncertainties: Dict[int, Dict[str, float]],
                                 outliers_detected: Set[Tuple[int, int]],
                                 rmse_px: float,
                                 report_dir: str = "data/tag_calibration_verification") -> str:
        """委托生成 2D 像面 Quiver 残差矢量场并输出 Markdown 诊断报告"""
        return self.ba_optimizer.export_diagnostic_report(
            final_tags_map=final_tags_map,
            detailed_obs_res=detailed_obs_res,
            active_frame_names=active_frame_names,
            tag_uncertainties=tag_uncertainties,
            outliers_detected=outliers_detected,
            rmse_px=rmse_px,
            report_dir=report_dir
        )

    def build_map_from_manifest(self, 
                                manifest_path: str = "data/tag_calibration_images/tag_observations.yaml",
                                origin_tag_id: int = 0,
                                x_align_tag_id: int = 1,
                                baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict[str, Any]:
        """
        从审核清单直接加载过滤后的观测数据并执行 BA 优化建图
        """
        frame_detections, valid_frames, stats = self.load_observations_manifest(manifest_path)
        print(f"[+] 从审核清单成功载入: {len(valid_frames)} 张有效图像，保留观测 {stats['total_kept']} 次，排除观测 {stats['total_excluded']} 次")
        if stats["total_excluded"] > 0:
            print(f"    【已人工剔除的坏样本】:")
            for exc in stats["excluded_items"]:
                note_str = f" (备注: {exc['note']})" if exc['note'] else ""
                print(f"      - [{exc['image']}] Tag #{exc['tag_id']}{note_str}")
        if stats["dropped_single_tag_frames"]:
            for d_name, d_cnt in stats["dropped_single_tag_frames"]:
                print(f"[WARN] 图像 {d_name} 有效标靶少于 2 个 (实际={d_cnt})，已自动不参与相对刚体平差约束")

        return self.optimize_bundle_adjustment(
            frame_detections=frame_detections,
            active_frame_names=valid_frames,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )

    def build_map_from_images(self, 
                              image_paths: List[str],
                              origin_tag_id: int = 0,
                              x_align_tag_id: int = 1,
                              baseline_pair: Optional[Tuple[int, int, float]] = None,
                              manifest_path: str = "data/tag_calibration_images/tag_observations.yaml",
                              use_manifest: bool = True) -> Dict[str, Any]:
        """
        从一组多视角图像构建标靶全局地图并进行 BA 全局平差优化
        自动联动审核清单流水线 (若清单不存在则先导出，若存在则使用清单的过滤规则)
        """
        print(f"[*] 开始处理 {len(image_paths)} 张多视角标定图像...")
        
        # 探测输入图像真实分辨率，执行内参动态自适应与防呆校验
        for path in image_paths:
            probe_img = cv2.imread(path)
            if probe_img is not None:
                if resolve_camera_intrinsics is not None:
                    self.camera_matrix, self.dist_coeffs, _ = resolve_camera_intrinsics(
                        "config.yaml", actual_image_shape=probe_img.shape[:2]
                    )
                break

        if use_manifest:
            if not os.path.exists(manifest_path):
                print(f"[*] 未检测到观测清单，正在生成初始观测数据清单: {manifest_path} ...")
                self.export_observations_manifest(image_paths, manifest_path=manifest_path)
            return self.build_map_from_manifest(
                manifest_path=manifest_path,
                origin_tag_id=origin_tag_id,
                x_align_tag_id=x_align_tag_id,
                baseline_pair=baseline_pair
            )

        # 直接全量检测（备用快速通道）
        frame_detections = []
        valid_frames = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            tags = self.detect_tags(img)
            if len(tags) >= 2:
                frame_detections.append(tags)
                valid_frames.append(os.path.basename(path))

        return self.optimize_bundle_adjustment(
            frame_detections=frame_detections,
            active_frame_names=valid_frames,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )

    def apply_baseline_scale(self, 
                             tag_poses: Dict[int, np.ndarray],
                             tag_id_a: int, 
                             tag_id_b: int, 
                             real_distance_mm: float) -> Tuple[Dict[int, np.ndarray], float, float]:
        """利用双标靶距离锁定绝对尺度（委托专职求解器）"""
        self.ba_optimizer.marker_size_mm = self.marker_size_mm
        scaled_poses, scale_factor, real_marker_size = self.ba_optimizer.apply_baseline_scale(
            tag_poses=tag_poses,
            tag_id_a=tag_id_a,
            tag_id_b=tag_id_b,
            real_distance_mm=real_distance_mm
        )
        self.marker_size_mm = real_marker_size
        return scaled_poses, scale_factor, real_marker_size

    def _align_to_scara_world(self, 
                              tag_poses: Dict[int, np.ndarray], 
                              origin_tag_id: int, 
                              x_align_tag_id: int) -> Dict[str, Any]:
        """对齐 SCARA 世界坐标系（委托专职求解器）"""
        return self.ba_optimizer.align_to_scara_world(
            tag_poses=tag_poses,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id
        )

    def save_map(self, map_data: Dict, output_path: str = "config/tags_map.yaml"):
        """保存标靶地图至 YAML 文件（委托专职仓储处理）"""
        return self.repository.save_map(map_data=map_data, output_path=output_path)


def print_topology_report(covis_report: Dict[str, Any], stats: Dict[str, Any], valid_frames: List[str]):
    """打印详细共视拓扑分析诊断报告（委托转发）"""
    CovisibilityGraphAnalyzer.print_topology_report(covis_report, stats, valid_frames)


def interactive_workflow(args, builder: TagMapBuilder, image_paths: List[str], baseline_pair: Optional[Tuple[int, int, float]]):
    """交互式两阶段建图与人工质量审核工作流"""
    manifest_path = args.manifest
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "visualized")

    # 1. 若清单不存在，先自动导出
    if not os.path.exists(manifest_path):
        print(f"[*] 首次运行，正在自动提取并生成观测数据清单: {manifest_path} ...")
        builder.export_observations_manifest(image_paths, manifest_path=manifest_path)

    while True:
        # 加载清单与状态
        try:
            frame_detections, valid_frames, stats = builder.load_observations_manifest(manifest_path)
            covis_report = builder.validate_covisibility(frame_detections, valid_frames, args.origin_id, args.x_axis_id)
        except Exception as e:
            print(f"[ERROR] 读取或解析观测清单异常: {e}")
            covis_report = {"is_valid": False, "message": str(e), "all_tags": [], "critical_bridges": []}
            stats = {"total_images": 0, "total_observations": 0, "total_kept": 0, "total_excluded": 0, "excluded_items": []}

        # 终端横幅
        print("\n" + "=" * 76)
        print("     【AprilTag 两阶段人工审核与 BA 空间平差控制台】(Tag Map Builder)")
        print("=" * 76)
        print(f" 采图目录: {args.image_dir} ({len(image_paths)} 帧)")
        print(f" 审核清单: {manifest_path}")
        print(f" 观测统计: 共 {stats['total_observations']} 次标靶检出 | 保留: {stats['total_kept']} | 人工剔除: {stats['total_excluded']}")
        
        if covis_report["is_valid"]:
            status_tag = f"\033[92m[ 连通网健康 (PASS) ]\033[0m"
            print(f" 拓扑状态: {status_tag} 全部 {len(covis_report['all_tags'])} 个标靶完全连通 (有效参与: {len(valid_frames)} 帧)")
        else:
            status_tag = f"\033[91m[ 拓扑断网告警 (FAIL) ]\033[0m"
            print(f" 拓扑状态: {status_tag} {covis_report['message']}")

        if covis_report.get("critical_bridges"):
            print(f" 关键桥梁: 发现 {len(covis_report['critical_bridges'])} 对标靶仅由单图连接: {covis_report['critical_bridges']} (注意不要剔除桥梁帧)")

        print("-" * 76)
        print(" 【第一阶段：数据审核与拓扑评估】")
        print("   [1] 启动交互审核画板 (鼠标点击标靶直接剔除/恢复，实时拓扑安全红绿灯)")
        print("   [2] 查看共视拓扑结构深度诊断报告 (节点度数、共视重叠帧数、关键桥梁)")
        print("")
        print(" 【辅助维护工具集 (无固定顺序，按需调用)】")
        print("   [3] 刷新/重新扫描图像并更新清单 (自动保留已有 keep: false 与备注)")
        print("   [4] 在系统资源管理器中打开可视化标注图目录 (visualized/) 查看像元标牌")
        print("   [5] (或 [E]) 在系统文本编辑器中直接编辑清单 (tag_observations.yaml)")
        print("")
        print(" 【最终收官：终审求解与成果导出】")
        print("   [6] (或 [B]) 立即执行 BA 全局平差优化并导出地图 (config/tags_map.yaml)")
        print("----------------------------------------------------------------------------")
        print("   [Q] 退出")
        print("=" * 76)

        try:
            choice = input("请输入操作编号 [1-6, E, B, Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出] 操作已终止。")
            break

        if choice == '1':
            try:
                from tools.calibration.tag_manifest_reviewer import TagManifestReviewer
                print("\n[启动] 正在启动 AprilTag 观测样本交互审核画板...")
                reviewer = TagManifestReviewer(manifest_path=manifest_path, builder=builder)
                reviewer.run()
            except Exception as e:
                print(f"\n[WARN] 启动交互画板失败或当前环境无显示服务: {e}")
                print("已自动回退到调用系统文本编辑器打开 YAML 清单。")
                abs_manifest = os.path.abspath(manifest_path)
                try:
                    if sys.platform == "win32":
                        os.startfile(abs_manifest)
                    elif sys.platform == "darwin":
                        import subprocess
                        subprocess.run(["open", abs_manifest])
                    else:
                        import subprocess
                        subprocess.run(["xdg-open", abs_manifest])
                except Exception:
                    pass
                try:
                    input("\n修改保存完毕后，请按回车键刷新...")
                except (EOFError, KeyboardInterrupt):
                    pass

        elif choice == '2':
            print_topology_report(covis_report, stats, valid_frames)
            try:
                input("\n按回车键返回菜单...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == '3':
            print(f"\n[*] 正在重新扫描所有标定图像并刷新清单 (历史人工标记将安全保留)...")
            builder.export_observations_manifest(image_paths, manifest_path=manifest_path)
            try:
                input("\n清单刷新完成，按回车键继续...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == '4':
            os.makedirs(vis_dir, exist_ok=True)
            print(f"\n[浏览] 正在打开可视化标注目录: {vis_dir} ...")
            try:
                if sys.platform == "win32":
                    os.startfile(vis_dir)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", vis_dir])
                else:
                    import subprocess
                    subprocess.run(["xdg-open", vis_dir])
            except Exception as e:
                print(f"[WARN] 打开目录失败: {e}")
            try:
                input("\n按回车键返回菜单...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice in ('5', 'E'):
            abs_manifest = os.path.abspath(manifest_path)
            print(f"\n[打开] 正在尝试调用系统默认编辑器打开: {abs_manifest} ...")
            try:
                if sys.platform == "win32":
                    os.startfile(abs_manifest)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", abs_manifest])
                else:
                    import subprocess
                    subprocess.run(["xdg-open", abs_manifest])
            except Exception as e:
                print(f"[WARN] 无法自动拉起编辑器: {e}，请手动打开该文件。")
            try:
                input("\n修改保存完毕后，请按回车键重新读取并刷新拓扑连通状态...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice in ('6', 'B'):
            if not covis_report["is_valid"]:
                print(f"\n\033[91m[拦截] 共视拓扑检查未通过，禁止执行 BA 优化！\033[0m")
                print(f"原因: {covis_report['message']}")
                print("请先选择选项 [1] 打开交互画板，恢复失联标靶的关键视角后再继续。")
                try:
                    input("\n按回车键返回菜单...")
                except (EOFError, KeyboardInterrupt):
                    pass
                continue

            try:
                tags_map = builder.optimize_bundle_adjustment(
                    frame_detections=frame_detections,
                    active_frame_names=valid_frames,
                    origin_tag_id=args.origin_id,
                    x_align_tag_id=args.x_axis_id,
                    baseline_pair=baseline_pair
                )
                builder.save_map(tags_map, args.output)
                print(f"\n\033[92m[SUCCESS] 标靶空间地图优化求解大功告成！已成功输出至: {args.output}\033[0m")
                break
            except Exception as e:
                print(f"\n\033[91m[ERROR] BA 平差求解失败: {e}\033[0m")
                try:
                    input("\n按回车键返回菜单...")
                except (EOFError, KeyboardInterrupt):
                    pass

        elif choice in ('Q', '0'):
            print("\n[退出] 已退出建图工具。")
            break


def main():
    parser = argparse.ArgumentParser(description="AprilTag 16h5 多标靶离线两阶段建图与 BA 平差工具")
    # 从 config.yaml 动态加载基准标靶 ID
    def_origin_id = 0
    def_x_axis_id = 28
    try:
        cfg_file = os.path.join(PROJECT_ROOT, "config.yaml")
        if os.path.exists(cfg_file):
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_obj = yaml.safe_load(f) or {}
            c_sec = cfg_obj.get("calibration", {})
            def_origin_id = int(c_sec.get("origin_tag_id", 0))
            def_x_axis_id = int(c_sec.get("x_axis_tag_id", 28))
    except Exception:
        pass

    def_image_dir = "data/tag_calibration_images"
    def_manifest = "data/tag_calibration_images/tag_observations.yaml"
    def_output = "config/tags_map.yaml"
    try:
        from src.calibration.scene_manager import CalibrationSceneManager
        active_sc = CalibrationSceneManager().get_active_scene()
        def_image_dir = active_sc.raw_images_dir
        def_manifest = active_sc.manifest_path
        def_output = active_sc.map_path
    except Exception:
        pass

    parser.add_argument("--image_dir", type=str, default=def_image_dir, help="多视角标定图片目录")
    parser.add_argument("--manifest", type=str, default=def_manifest, help="观测数据审核清单路径")
    parser.add_argument("--marker_size", type=float, default=50.0, help="标靶黑白边框名义边长 (mm)")
    parser.add_argument("--origin_id", type=int, default=def_origin_id, help="SCARA 原点锚定标靶 ID")
    parser.add_argument("--x_axis_id", type=int, default=def_x_axis_id, help="世界 X 轴对齐基准标靶 ID (默认与 config.yaml 一致)")
    parser.add_argument("--baseline_pair", nargs=3, type=float, metavar=('TAG_A', 'TAG_B', 'DIST_MM'),
                        help="双标靶基线尺度校准参数: TAG_A TAG_B 真实距离(mm), 例如: --baseline_pair 1 5 620.5")
    parser.add_argument("--output", type=str, default=def_output, help="导出的图谱文件路径")
    parser.add_argument("--export-manifest", action="store_true", help="仅扫描图像导出观测数据清单并退出")
    parser.add_argument("--solve-manifest", action="store_true", help="直接读取清单执行 BA 平差 (非交互批处理)")
    parser.add_argument("--inspect", action="store_true", help="仅执行连通性深度诊断并输出报告后退出")
    parser.add_argument("--no-interactive", action="store_true", help="静默非交互模式运行")
    args = parser.parse_args()

    # 寻找图像文件
    patterns = [os.path.join(args.image_dir, ext) for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat))

    if not image_paths and not os.path.exists(args.manifest):
        print(f"[!] 目录 '{args.image_dir}' 下未找到任何标定图像，且未找到已有清单 '{args.manifest}'！")
        print(f"[*] 提示：请使用相机采集覆盖多标靶的图像放入该目录后重试。")
        sys.exit(1)

    builder = TagMapBuilder(marker_size_mm=args.marker_size)

    baseline_pair = None
    if args.baseline_pair is not None:
        baseline_pair = (int(args.baseline_pair[0]), int(args.baseline_pair[1]), float(args.baseline_pair[2]))

    # 单独模式 1: 仅导出清单
    if args.export_manifest:
        builder.export_observations_manifest(image_paths, manifest_path=args.manifest)
        print("[OK] 清单导出完毕，已退出。")
        return

    # 单独模式 2: 仅深度诊断
    if args.inspect:
        if not os.path.exists(args.manifest):
            builder.export_observations_manifest(image_paths, manifest_path=args.manifest)
        f_det, v_frames, stats = builder.load_observations_manifest(args.manifest)
        report = builder.validate_covisibility(f_det, v_frames, args.origin_id, args.x_axis_id)
        print_topology_report(report, stats, v_frames)
        return

    # 单独模式 3: 直接静默求解
    if args.solve_manifest or args.no_interactive or (not sys.stdin.isatty()):
        if not os.path.exists(args.manifest):
            builder.export_observations_manifest(image_paths, manifest_path=args.manifest)
        tags_map = builder.build_map_from_manifest(
            manifest_path=args.manifest,
            origin_tag_id=args.origin_id,
            x_align_tag_id=args.x_axis_id,
            baseline_pair=baseline_pair
        )
        builder.save_map(tags_map, args.output)
        return

    # 默认模式: 启动交互式审核与建图控制台
    interactive_workflow(args, builder, image_paths, baseline_pair)


if __name__ == "__main__":
    main()
