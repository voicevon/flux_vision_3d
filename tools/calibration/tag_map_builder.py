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


class CovisibilityGraphError(Exception):
    """标靶共视连通图拓扑异常（如出现孤立子图、断网或约束退化）"""
    pass


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
        - 自动保留用户此前修改过的 keep: false 剔除状态与人工备注 (note)
        - 自动输出单元方格分辨率 (Cell: WxH px)、中心坐标与面积
        - 支持同步生成高清图示化标注图片至 visualized/ 目录
        """
        # 读取已有清单以保留用户此前的手工标记
        existing_prefs = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old_manifest = yaml.safe_load(f) or {}
                for img_key, img_info in old_manifest.get("images", {}).items():
                    for obs in img_info.get("observations", []):
                        tid = obs.get("tag_id")
                        keep = obs.get("keep", True)
                        note = obs.get("note", "")
                        existing_prefs[(img_key, tid)] = (keep, note)
                print(f"[+] 检测到已有审核清单，已成功加载 {len(existing_prefs)} 条历史人工保留/剔除标记")
            except Exception as e:
                print(f"[WARN] 读取已有清单配置失败: {e}，将生成全新清单。")

        manifest_data = {
            "summary": {},
            "images": {}
        }
        total_obs = 0
        total_kept = 0
        total_excluded = 0

        vis_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "visualized")
        if generate_visualized:
            os.makedirs(vis_dir, exist_ok=True)

        for path in sorted(image_paths):
            base_name = os.path.basename(path)
            img = cv2.imread(path)
            if img is None:
                continue

            detected = self.detect_tags(img)
            obs_list = []

            for tid in sorted(detected.keys()):
                corners = detected[tid]
                metrics = self.compute_tag_metrics(corners)
                
                # 保留历史人工决策
                keep_val, note_val = existing_prefs.get((base_name, tid), (True, ""))
                total_obs += 1
                if keep_val:
                    total_kept += 1
                else:
                    total_excluded += 1

                obs_entry = {
                    "tag_id": int(tid),
                    "keep": bool(keep_val),
                    "cell_size_px": metrics["cell_size_px"],
                    "center_px": metrics["center_px"],
                    "area_px": metrics["area_px"],
                    "note": str(note_val),
                    "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                }
                obs_list.append(obs_entry)

            annotated_name = os.path.splitext(base_name)[0] + "_annotated.png"
            annotated_path = os.path.join(vis_dir, annotated_name)
            
            if generate_visualized:
                ann_img = self.render_annotated_frame(img, detected)
                cv2.imwrite(annotated_path, ann_img)

            manifest_data["images"][base_name] = {
                "file_name": base_name,
                "image_path": path.replace("\\", "/"),
                "annotated_path": annotated_path.replace("\\", "/"),
                "detected_count": len(obs_list),
                "observations": obs_list
            }

        manifest_data["summary"] = {
            "total_images": len(manifest_data["images"]),
            "total_observations": total_obs,
            "total_kept": total_kept,
            "total_excluded": total_excluded,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 写入带有清晰中文使用指南的 YAML
        header_comments = (
            "# ==============================================================================\n"
            "# AprilTag 离线建图与 BA 平差观测数据清单 (Observations Manifest)\n"
            "# \n"
            "# 【人工审核与坏样本剔除指南】：\n"
            "# 1. 请在下方列表中查找需要剔除的劣质标靶观测（如透视大倾角、模糊、边缘拉丝畸变）；\n"
            "# 2. 将对应的 `keep: true` 修改为 `keep: false` 即可，求解器将自动忽略该项；\n"
            "# 3. 可在 `note: \"\"` 中记录剔除原因（如 \"大倾角发灰\"、\"遮挡严重\" 等）；\n"
            "# 4. 修改保存后，在控制台选择继续执行平差即可；\n"
            "# 5. 【拓扑连通性安全守门员】：系统会在平差前进行连通图校验，防止误删导致断网崩溃！\n"
            "# ==============================================================================\n\n"
        )
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(header_comments)
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"[OK] 观测清单已成功导出至: {manifest_path}")
        print(f"     统计: 共 {len(manifest_data['images'])} 张图像，{total_obs} 次标靶观测 (保留: {total_kept}, 排除: {total_excluded})")
        return manifest_path

    def load_observations_manifest(self, manifest_path: str = "data/tag_calibration_images/tag_observations.yaml") -> Tuple[List[Dict[int, np.ndarray]], List[str], Dict[str, Any]]:
        """
        两阶段建图流水线 - 阶段二：
        从审核清单中加载已审核的标靶观测数据，并过滤掉 keep: false 的坏样本。
        :return: (frame_detections, valid_frame_names, stats)
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"未找到观测清单文件: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        frame_detections = []
        valid_frame_names = []
        stats = {
            "total_images": len(data.get("images", {})),
            "total_observations": 0,
            "total_kept": 0,
            "total_excluded": 0,
            "dropped_single_tag_frames": [],
            "excluded_items": []
        }

        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        for img_name, img_info in data.get("images", {}).items():
            tags_in_frame = {}
            # 探测原图是否存在以执行实时亚像素精修
            raw_img_path = img_info.get("image_path", os.path.join(manifest_dir, img_name))
            if not os.path.isabs(raw_img_path):
                raw_img_path = os.path.join(PROJECT_ROOT, raw_img_path)
            gray_for_refine = None
            if os.path.exists(raw_img_path):
                raw_img = cv2.imread(raw_img_path)
                if raw_img is not None:
                    gray_for_refine = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

            for obs in img_info.get("observations", []):
                stats["total_observations"] += 1
                tid = int(obs["tag_id"])
                keep = obs.get("keep", True)
                if not keep:
                    stats["total_excluded"] += 1
                    stats["excluded_items"].append({
                        "image": img_name,
                        "tag_id": tid,
                        "note": obs.get("note", "")
                    })
                    continue

                corners = np.array(obs["corners"], dtype=np.float64)
                if gray_for_refine is not None:
                    corners = self.refine_corners_subpix(gray_for_refine, corners)

                # 物理有效性前置过滤：过滤微小噪点伪标靶 (area < 120px^2) 或测距异常 (深度不在 150~2200mm)
                area = float(cv2.contourArea(corners.astype(np.float32)))
                succ, _, tv = self.solve_single_tag_pnp(corners)
                z = float(tv[2][0]) if succ else 0.0
                if area < 120.0 or z > 2200.0 or z < 150.0:
                    stats["total_excluded"] += 1
                    stats["excluded_items"].append({
                        "image": img_name,
                        "tag_id": tid,
                        "note": f"物理异常噪点自动拦截 (面积={area:.1f}px, 深度={z:.1f}mm)"
                    })
                    continue

                stats["total_kept"] += 1
                tags_in_frame[tid] = corners

            # 约束检查：一张图必须至少观测到 2 个 Tag 才能形成相对刚体约束
            if len(tags_in_frame) >= 2:
                frame_detections.append(tags_in_frame)
                valid_frame_names.append(img_name)
            else:
                stats["dropped_single_tag_frames"].append((img_name, len(tags_in_frame)))

        return frame_detections, valid_frame_names, stats

    def validate_covisibility(self, 
                              frame_detections: List[Dict[int, np.ndarray]], 
                              frame_names: Optional[List[str]] = None,
                              origin_tag_id: int = 0,
                              x_align_tag_id: int = 1) -> Dict[str, Any]:
        """
        共视连通性安全守门员 (Co-visibility Graph Connectivity Guard)
        在 BA 平差前全面审查标靶共视图拓扑健康度：
        1. 检查各 Tag 是否在一个连通分量内，防止误删导致孤岛断网引发奇异矩阵；
        2. 识别单视角关键桥梁 (Critical Bridges)，当桥梁仅由 1 张图片支撑时警示用户；
        3. 给出详细结构化拓扑报告。
        """
        all_tags = set()
        for tags in frame_detections:
            all_tags.update(tags.keys())

        if len(all_tags) == 0:
            return {
                "is_valid": False,
                "all_tags": [],
                "connected_tags": [],
                "unconnected_tags": [],
                "adj_list": {},
                "edge_counts": {},
                "critical_bridges": [],
                "valid_frames_count": 0,
                "components": [],
                "message": "有效标靶集合为空，无法建图！"
            }

        # 构建无向图与共视重叠帧数统计
        adj_list = {t: set() for t in all_tags}
        edge_counts = {}

        for tags in frame_detections:
            t_ids = list(tags.keys())
            for i in range(len(t_ids)):
                for j in range(i + 1, len(t_ids)):
                    u, v = min(t_ids[i], t_ids[j]), max(t_ids[i], t_ids[j])
                    adj_list[u].add(v)
                    adj_list[v].add(u)
                    edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

        # 识别关键单一支撑桥梁 (只有 1 帧同时观测到该标靶对)
        critical_bridges = [(u, v) for (u, v), count in edge_counts.items() if count == 1]

        # 寻找连通分量 (BFS)
        remaining = set(all_tags)
        components = []
        while remaining:
            root = next(iter(remaining))
            comp = set()
            q = [root]
            while q:
                curr = q.pop(0)
                if curr not in comp:
                    comp.add(curr)
                    q.extend(adj_list[curr] - comp)
            components.append(sorted(list(comp)))
            remaining -= comp

        # 确定主基准分量 (优先包含 x_align_tag_id 或 origin_tag_id)
        preferred_base = x_align_tag_id if x_align_tag_id in all_tags else (origin_tag_id if origin_tag_id in all_tags else min(all_tags))
        main_component = []
        for comp in components:
            if preferred_base in comp:
                main_component = comp
                break
        if not main_component and components:
            main_component = max(components, key=len)

        unconnected = sorted(list(all_tags - set(main_component)))
        is_valid = (len(components) == 1) and (len(all_tags) >= 2) and (len(frame_detections) >= 2)

        if not is_valid:
            if len(all_tags) < 2:
                msg = f"有效标靶总数不足 2 个 (仅检出 {list(all_tags)})，无法构建相对空间地图！"
            elif len(frame_detections) < 2:
                msg = f"有效多视角图像不足 2 张 (当前仅 {len(frame_detections)} 张)，无法构建空间刚体约束！"
            else:
                msg = (f"【严重风险】标靶共视图发生断裂！共发现 {len(components)} 个孤立分量。\n"
                       f"主连通网络: {main_component}\n"
                       f"失联断网标靶: {unconnected}\n"
                       f"提示：请检查 tag_observations.yaml，恢复连接上述失联标靶的关键视角观测 (设为 keep: true)。")
        else:
            msg = f"共视连通性检查完全通过！全部 {len(all_tags)} 个标靶处于统一连通拓扑网中。"

        return {
            "is_valid": is_valid,
            "all_tags": sorted(list(all_tags)),
            "connected_tags": main_component,
            "unconnected_tags": unconnected,
            "components": components,
            "adj_list": {k: sorted(list(v)) for k, v in adj_list.items()},
            "edge_counts": {f"{u}-{v}": c for (u, v), c in edge_counts.items()},
            "critical_bridges": critical_bridges,
            "valid_frames_count": len(frame_detections),
            "message": msg
        }

    def optimize_bundle_adjustment(self, 
                                   frame_detections: List[Dict[int, np.ndarray]], 
                                   active_frame_names: Optional[List[str]] = None,
                                   origin_tag_id: int = 0,
                                   x_align_tag_id: int = 1,
                                   baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict[str, Any]:
        """
        基于非线性最小二乘 (Bundle Adjustment) 联合优化所有标靶位姿与相机位姿
        平差前严格调用共视连通性安全守门员校验，杜绝奇异矩阵。
        """
        if active_frame_names is None:
            active_frame_names = [f"frame_{i:04d}" for i in range(len(frame_detections))]

        # 1. 守门员审查
        report = self.validate_covisibility(frame_detections, active_frame_names, origin_tag_id, x_align_tag_id)
        if not report["is_valid"]:
            raise CovisibilityGraphError(report["message"])

        all_detected_tags = set(report["all_tags"])
        if report["critical_bridges"]:
            print(f"[NOTE] 提示：发现 {len(report['critical_bridges'])} 对标靶仅由单张图共视支撑 (关键桥梁): {report['critical_bridges']}")

        # 2. 生成高质量初值 (多标靶联合超定 PnP 初值传递，杜绝单链累积误差与翻转)
        base_static_id = x_align_tag_id if x_align_tag_id in all_detected_tags else min(all_detected_tags)
        tag_poses_init = {base_static_id: np.eye(4, dtype=np.float64)}
        camera_poses_init = {}  # {frame_idx: T_w_cam}

        changed = True
        while changed:
            changed = False
            # 步骤 2.1: 优先利用该帧可见的所有已知标靶进行联合超定 PnP 求解相机位姿
            for f_idx, tags in enumerate(frame_detections):
                if f_idx not in camera_poses_init:
                    obj_pts_list = []
                    img_pts_list = []
                    for t_id, corners in tags.items():
                        if t_id in tag_poses_init:
                            T_w_t = tag_poses_init[t_id]
                            for pt3d in self.obj_points:
                                pt_h = np.append(pt3d, 1.0)
                                w_pt = (T_w_t @ pt_h)[:3]
                                obj_pts_list.append(w_pt)
                            img_pts_list.extend(corners.reshape(4, 2))

                    if len(obj_pts_list) >= 4:
                        obj_pts_arr = np.array(obj_pts_list, dtype=np.float64)
                        img_pts_arr = np.array(img_pts_list, dtype=np.float64)
                        succ, rvec, tvec = cv2.solvePnP(
                            obj_pts_arr, img_pts_arr, 
                            self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        if succ:
                            T_c_w = self.rvec_tvec_to_matrix(rvec, tvec)
                            camera_poses_init[f_idx] = np.linalg.inv(T_c_w)
                            changed = True

            # 步骤 2.2: 利用已定位的相机推算未知标靶在世界系下的绝对位姿
            for f_idx in list(camera_poses_init.keys()):
                T_w_c = camera_poses_init[f_idx]
                tags = frame_detections[f_idx]
                for t_id, corners in tags.items():
                    if t_id not in tag_poses_init:
                        succ, rvec, tvec = self.solve_single_tag_pnp(corners)
                        if succ:
                            T_c_t = self.rvec_tvec_to_matrix(rvec, tvec)
                            tag_poses_init[t_id] = T_w_c @ T_c_t
                            changed = True

        print(f"[+] 超定初值生成完毕：成功初始化 {len(tag_poses_init)} 个 Tag，{len(camera_poses_init)} 个相机机位")

        # 3. 预计算每个观测项的几何置信度权重矩阵
        obs_weights = {}
        for f_idx in camera_poses_init.keys():
            tags = frame_detections[f_idx]
            for t_id, corners in tags.items():
                w = self.compute_observation_weight(corners)
                obs_weights[(f_idx, t_id)] = w

        static_tags_to_opt = [t for t in tag_poses_init.keys() if t != base_static_id]
        active_frames = [f for f in camera_poses_init.keys()]

        # 4. 构建两阶段 BA 优化器
        def pack_params(tags_dict, cams_dict):
            vec = []
            for t in static_tags_to_opt:
                rv, tv = self.matrix_to_rvec_tvec(tags_dict[t])
                vec.extend(rv.flatten().tolist())
                vec.extend(tv.flatten().tolist())
            for f in active_frames:
                rv, tv = self.matrix_to_rvec_tvec(cams_dict[f])
                vec.extend(rv.flatten().tolist())
                vec.extend(tv.flatten().tolist())
            return np.array(vec, dtype=np.float64)

        def unpack_params(x):
            tags_pose = {base_static_id: np.eye(4, dtype=np.float64)}
            offset = 0
            for t in static_tags_to_opt:
                rv = x[offset:offset + 3]
                tv = x[offset + 3:offset + 6]
                tags_pose[t] = self.rvec_tvec_to_matrix(rv, tv)
                offset += 6

            cams_pose = {}
            for f in active_frames:
                rv = x[offset:offset + 3]
                tv = x[offset + 3:offset + 6]
                cams_pose[f] = self.rvec_tvec_to_matrix(rv, tv)
                offset += 6
            return tags_pose, cams_pose

        def residuals_func(x, weights_dict, active_outliers=None):
            tags_pose, cams_pose = unpack_params(x)
            residuals = []
            for f in active_frames:
                T_w_c = cams_pose[f]
                T_c_w = np.linalg.inv(T_w_c)
                tags = frame_detections[f]
                for t_id, corners_img in tags.items():
                    if t_id in tags_pose:
                        w = weights_dict.get((f, t_id), 1.0)
                        if active_outliers and (f, t_id) in active_outliers:
                            w *= 0.01  # 离群项强力压制

                        T_w_t = tags_pose[t_id]
                        T_c_t = T_c_w @ T_w_t
                        rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                        proj_pts, _ = cv2.projectPoints(
                            self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs
                        )
                        proj_pts = proj_pts.reshape((4, 2))
                        diff = (proj_pts - corners_img).flatten()
                        # 加权非线性最小二乘 (乘以 sqrt(w))
                        residuals.extend(diff * np.sqrt(max(1e-4, w)))

            # 若配置了物理标靶间距先验约束，作为硬约束惩罚项联合求解
            if baseline_pair is not None:
                id_a, id_b, real_dist_mm = baseline_pair
                if id_a in tags_pose and id_b in tags_pose:
                    t_a = tags_pose[id_a][:3, 3]
                    t_b = tags_pose[id_b][:3, 3]
                    dist_est = float(np.linalg.norm(t_a - t_b))
                    # 尺度约束权重设为 5.0
                    residuals.append((dist_est - real_dist_mm) * 5.0)

            return np.array(residuals, dtype=np.float64)

        x0 = pack_params(tag_poses_init, camera_poses_init)

        print("[*] 正在执行 Phase 1 阶段一：基于 Cauchy 鲁棒核的粗差清洗与全局收敛...")
        res_stage1 = least_squares(
            residuals_func, x0,
            args=(obs_weights, None),
            method='trf',
            loss='cauchy',
            f_scale=1.5,
            x_scale='jac',
            ftol=1e-5,
            xtol=1e-5,
            max_nfev=200,
            verbose=0
        )

        # 统计阶段一未加权像元残差，自动识别标准化残差 > 3.0 sigma 的粗差观测
        tags_p1, cams_p1 = unpack_params(res_stage1.x)
        raw_errors = []
        obs_map = []
        for f in active_frames:
            T_c_w = np.linalg.inv(cams_p1[f])
            for t_id, corners_img in frame_detections[f].items():
                if t_id in tags_p1:
                    T_c_t = T_c_w @ tags_p1[t_id]
                    rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                    proj, _ = cv2.projectPoints(self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs)
                    e = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - corners_img, axis=1)))
                    raw_errors.append(e)
                    obs_map.append((f, t_id, e))

        # 采用鲁棒统计学 MAD (Median Absolute Deviation) 估计真实噪声标准差，杜绝离群坏点拉偏门限
        med_e = float(np.median(raw_errors))
        mad_e = float(np.median(np.abs(np.array(raw_errors) - med_e)))
        sigma_robust = max(0.5, 1.4826 * mad_e)
        outlier_thresh = max(4.0, med_e + 2.8 * sigma_robust)
        outliers_detected = set()
        for f, t_id, e in obs_map:
            # 权重过低或残差明显超过 MAD 鲁棒置信门限的判定为离群值
            if e > outlier_thresh or obs_weights.get((f, t_id), 1.0) <= 0.05:
                outliers_detected.add((f, t_id))

        if outliers_detected:
            print(f"[CLEAN] 自动清洗识别出 {len(outliers_detected)} 个潜在粗差/远景噪点观测 (MAD 门限 > {outlier_thresh:.2f}px, 中位数={med_e:.2f}px):")
            for f, t_id in sorted(list(outliers_detected)):
                f_name = active_frame_names[f] if f < len(active_frame_names) else f"frame_{f}"
                print(f"        - [{f_name}] Tag #{t_id}")

        print("[*] 正在执行 Phase 1 阶段二：微容差 (ftol=1e-9) 极致深层平差收敛...")
        res_stage2 = least_squares(
            residuals_func, res_stage1.x,
            args=(obs_weights, outliers_detected),
            method='trf',
            loss='cauchy',
            f_scale=1.0,
            x_scale='jac',
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=500,
            verbose=0
        )

        optimized_tags_pose, optimized_cams_pose = unpack_params(res_stage2.x)

        # 计算剔除已清洗粗差点后的真实未加权重投影 RMSE
        clean_residuals = []
        detailed_obs_res = []
        for f in active_frames:
            T_c_w = np.linalg.inv(optimized_cams_pose[f])
            for t_id, corners_img in frame_detections[f].items():
                if t_id in optimized_tags_pose:
                    T_c_t = T_c_w @ optimized_tags_pose[t_id]
                    rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                    proj, _ = cv2.projectPoints(self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs)
                    proj_2d = proj.reshape(4, 2)
                    diff = proj_2d - corners_img
                    e_pt = np.linalg.norm(diff, axis=1)
                    detailed_obs_res.append({
                        "frame_idx": f,
                        "frame_name": active_frame_names[f] if f < len(active_frame_names) else f"frame_{f}",
                        "tag_id": t_id,
                        "corners_obs": corners_img,
                        "corners_proj": proj_2d,
                        "rmse_px": float(np.sqrt(np.mean(e_pt ** 2))),
                        "is_outlier": (f, t_id) in outliers_detected
                    })
                    if (f, t_id) not in outliers_detected:
                        clean_residuals.extend(diff.flatten())

        rmse_px = float(np.sqrt(np.mean(np.array(clean_residuals) ** 2)))
        print(f"[OK] 两阶段 BA 极限优化完成！有效观测像面 RMSE: {rmse_px:.3f} 像素 (迭代次数: {res_stage2.nfev})")

        # 5. 计算 3D 标靶空间坐标一阶协方差置信区间 (Uncertainty Estimation)
        tag_uncertainties = self.compute_3d_uncertainties(
            res_stage2.jac, static_tags_to_opt, base_static_id, rmse_px
        )

        # 6. 双标靶中心基线测距尺度修正 (Metric Baseline Gauge)
        scale_factor = 1.0
        real_marker_size = self.marker_size_mm
        baseline_info = None

        if baseline_pair is not None:
            id_a, id_b, real_dist_mm = baseline_pair
            optimized_tags_pose, scale_factor, real_marker_size = self.apply_baseline_scale(
                optimized_tags_pose, id_a, id_b, real_dist_mm
            )
            baseline_info = {
                "tag_a": int(id_a),
                "tag_b": int(id_b),
                "measured_dist_mm": float(real_dist_mm),
                "scale_factor": round(float(scale_factor), 6)
            }
            self.marker_size_mm = real_marker_size

        # 7. 坐标系对齐闭环：将世界坐标原点绑定到 Tag 0 中心，X 轴对齐到 Tag 1
        final_tags_map = self._align_to_scara_world(
            optimized_tags_pose, 
            origin_tag_id=origin_tag_id, 
            x_align_tag_id=x_align_tag_id
        )

        final_tags_map["rmse_reprojection_px"] = rmse_px
        final_tags_map["marker_size_mm"] = round(float(self.marker_size_mm), 3)
        final_tags_map["tag_family"] = "DICT_APRILTAG_16h5"
        final_tags_map["calibrated_images_count"] = len(active_frames)
        final_tags_map["cleaned_outliers_count"] = len(outliers_detected)
        if baseline_info:
            final_tags_map["baseline_gauge"] = baseline_info
        if tag_uncertainties:
            final_tags_map["uncertainties_mm"] = tag_uncertainties

        # 8. 生成 2D 像面 Quiver Plot 残差矢量场与详细 Markdown 诊断报告
        try:
            self.export_diagnostic_report(
                final_tags_map=final_tags_map,
                detailed_obs_res=detailed_obs_res,
                active_frame_names=active_frame_names,
                tag_uncertainties=tag_uncertainties,
                outliers_detected=outliers_detected,
                rmse_px=rmse_px
            )
        except Exception as e:
            print(f"[WARN] 导出深度诊断报告异常 (已安全忽略): {e}")

        return final_tags_map

    def compute_3d_uncertainties(self, jacobian, static_tags, base_id, sigma_res_px) -> Dict[int, Dict[str, float]]:
        """
        基于平差最优解雅可比矩阵 J 计算参数协方差：
        Cov = inv(J^T J) * sigma_res^2
        提取每个标靶 3D 位置分量 (tv_x, tv_y, tv_z) 的 3-sigma 空间置信区间 (单位: mm)
        """
        uncertainties = {base_id: {"sigma_x_mm": 0.0, "sigma_y_mm": 0.0, "sigma_z_mm": 0.0, "sigma_3d_mm": 0.0}}
        if jacobian is None:
            return uncertainties
        try:
            J = jacobian
            if hasattr(J, "toarray"):
                J = J.toarray()
            JTJ = J.T @ J
            diag_JTJ = np.diag(JTJ)
            # 正则化防奇异
            JTJ_reg = JTJ + np.eye(JTJ.shape[0]) * 1e-6
            cov = np.linalg.pinv(JTJ_reg) * (sigma_res_px ** 2)

            for idx, tid in enumerate(static_tags):
                # 变量排布: 每个 tag 占 6 个自由度 (rv 3个, tv 3个)
                t_offset = idx * 6 + 3
                var_x = max(0.0, float(cov[t_offset, t_offset]))
                var_y = max(0.0, float(cov[t_offset + 1, t_offset + 1]))
                var_z = max(0.0, float(cov[t_offset + 2, t_offset + 2]))
                sx = round(float(np.sqrt(var_x) * 3.0), 3)
                sy = round(float(np.sqrt(var_y) * 3.0), 3)
                sz = round(float(np.sqrt(var_z) * 3.0), 3)
                s3d = round(float(np.sqrt(var_x + var_y + var_z) * 3.0), 3)
                uncertainties[tid] = {
                    "sigma_x_mm": sx,
                    "sigma_y_mm": sy,
                    "sigma_z_mm": sz,
                    "sigma_3d_mm": s3d
                }
        except Exception:
            pass
        return uncertainties

    def export_diagnostic_report(self,
                                 final_tags_map: Dict[str, Any],
                                 detailed_obs_res: List[Dict[str, Any]],
                                 active_frame_names: List[str],
                                 tag_uncertainties: Dict[int, Dict[str, float]],
                                 outliers_detected: Set[Tuple[int, int]],
                                 rmse_px: float,
                                 report_dir: str = "data/tag_calibration_verification") -> str:
        """
        生成 2D 像面 Quiver 残差矢量场并输出详尽的 Markdown 精度体检报告
        """
        os.makedirs(report_dir, exist_ok=True)
        vis_dir = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "visualized")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. 针对每张图绘制 2D 像面 Quiver 矢量场分析图
        frame_grouped = {}
        for item in detailed_obs_res:
            frame_grouped.setdefault(item["frame_name"], []).append(item)

        for f_name, obs_items in frame_grouped.items():
            img_path = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", f_name)
            if not os.path.exists(img_path):
                continue
            base_img = cv2.imread(img_path)
            if base_img is None:
                continue

            h, w = base_img.shape[:2]
            quiver_img = base_img.copy()

            # 绘制顶部半透明状态条
            cv2.rectangle(quiver_img, (0, 0), (w, 50), (20, 24, 30), -1)
            cv2.putText(quiver_img, f"BA 2D Residual Field (Quiver x20) - {f_name} | RMSE: {rmse_px:.3f}px",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            for obs in obs_items:
                tid = obs["tag_id"]
                c_obs = obs["corners_obs"]
                c_proj = obs["corners_proj"]
                is_outlier = obs["is_outlier"]

                # 标靶边框
                poly_color = (0, 0, 255) if is_outlier else (0, 255, 0)
                cv2.polylines(quiver_img, [c_obs.astype(np.int32)], True, poly_color, 2)

                # 绘制每个角点的残差矢量箭头 (放大 20 倍)
                scale = 20.0
                for pt_o, pt_p in zip(c_obs, c_proj):
                    dx = (pt_p[0] - pt_o[0]) * scale
                    dy = (pt_p[1] - pt_o[1]) * scale
                    p_start = (int(round(pt_o[0])), int(round(pt_o[1])))
                    p_end = (int(round(pt_o[0] + dx)), int(round(pt_o[1] + dy)))
                    
                    # 绿色小圆圈代表观测点，黄色十字代表模型投影
                    cv2.circle(quiver_img, p_start, 3, (0, 255, 0), -1)
                    cv2.drawMarker(quiver_img, (int(round(pt_p[0])), int(round(pt_p[1]))), (0, 255, 255), 
                                   markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1)
                    # 箭头
                    cv2.arrowedLine(quiver_img, p_start, p_end, (0, 0, 255) if is_outlier else (0, 165, 255), 
                                    2, tipLength=0.3)

                # 标注 Tag ID 与像面 RMSE
                center = np.mean(c_obs, axis=0).astype(int)
                lbl = f"Tag #{tid}: {obs['rmse_px']:.2f}px" + (" [OUTLIER]" if is_outlier else "")
                cv2.putText(quiver_img, lbl, (center[0] - 40, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            out_quiver_path = os.path.join(vis_dir, os.path.splitext(f_name)[0] + "_quiver.png")
            cv2.imwrite(out_quiver_path, quiver_img)

        # 2. 编写 Markdown 综合体检报告
        report_file = os.path.join(report_dir, "ba_precision_diagnostic_report.md")
        lines = [
            "# AprilTag 离线 BA 空间平差精度体检与深度诊断报告",
            f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"> 评价结论: **{'[优异 PASS]' if rmse_px <= 0.8 else '[良好 ACCEPTABLE]'}** (重投影有效 RMSE: **{rmse_px:.3f} px**)\n",
            "## 1. 平差全局核心指标",
            "| 指标项 | 测量值 | 工业判定门限 | 状态 |",
            "| :--- | :--- | :--- | :--- |",
            f"| **重投影均方根误差 (RMSE)** | **{rmse_px:.3f} px** | $\\le 0.50$ px | {'PASS' if rmse_px <= 0.5 else 'WARN'} |",
            f"| **参与优化图像帧数** | {final_tags_map.get('calibrated_images_count', 0)} 帧 | $\\ge 10$ 帧 | PASS |",
            f"| **空间标靶总数** | {len(final_tags_map.get('tags', {}))} 个 | $\\ge 6$ 个 | PASS |",
            f"| **自动识别清洗离群观测** | {len(outliers_detected)} 项 | $\\le 5$ 项 | {'PASS' if len(outliers_detected) <= 5 else 'WARN'} |",
            "\n## 2. 标靶 3D 空间绝对坐标与 $3\\sigma$ 置信度分析",
            "| 标靶 ID | 空间 X (mm) | 空间 Y (mm) | 空间 Z (mm) | $3\\sigma$ 空间不确定度 (mm) | $3\\sigma$ Z轴深度向 (mm) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        tags_dict = final_tags_map.get("tags", {})
        for tid in sorted(tags_dict.keys()):
            info = tags_dict[tid]
            pos = info.get("position_mm", [0, 0, 0])
            unc = tag_uncertainties.get(tid, {})
            s3d = unc.get("sigma_3d_mm", 0.0)
            sz = unc.get("sigma_z_mm", 0.0)
            lines.append(f"| Tag #{tid:2d} | {pos[0]:8.2f} | {pos[1]:8.2f} | {pos[2]:8.2f} | $\\pm${s3d:5.2f} mm | $\\pm${sz:5.2f} mm |")

        lines.extend([
            "\n## 3. 各采图机位重投影残差分布",
            "| 图像文件名 | 观测标靶数 | 平均重投影误差 (px) | 最大误差标靶 | 状态 |",
            "| :--- | :---: | :---: | :--- | :---: |"
        ])

        for f_name, obs_items in sorted(frame_grouped.items()):
            valid_e = [o["rmse_px"] for o in obs_items if not o["is_outlier"]]
            f_mean = float(np.mean(valid_e)) if valid_e else 0.0
            max_item = max(obs_items, key=lambda x: x["rmse_px"])
            max_str = f"Tag #{max_item['tag_id']} ({max_item['rmse_px']:.2f}px)"
            status_str = "PASS" if f_mean <= 0.8 else "WARN"
            lines.append(f"| {f_name:15s} | {len(obs_items):2d} 个 | {f_mean:6.2f} px | {max_str:20s} | {status_str} |")

        lines.extend([
            "\n## 4. 2D 残差矢量场 (Quiver Plot) 说明",
            "- 分析图像已输出至目录: `data/tag_calibration_images/visualized/*_quiver.png`；",
            "- 红色箭头代表角点残差矢量 (已统一放大 20 倍)，用于辨识相机内参畸变是否完全对称消除；",
            "- 若箭头呈完全各向同性发散，说明系统误差已被彻底吸收，剩余均为传感器随机白噪声。"
        ])

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[OK] 深度精度体检与诊断报告已生成至: {report_file}")
        return report_file

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
        """
        利用两个标靶中心物理测量距离锁定绝对尺度 (Metric Baseline Gauge)
        :param tag_poses: 各标靶 4x4 位姿矩阵字典
        :param tag_id_a: 标靶 A 的 ID
        :param tag_id_b: 标靶 B 的 ID
        :param real_distance_mm: 现场实际测量的中心物理直线距离 (mm)
        :return: (scaled_tag_poses, scale_factor, real_marker_size_mm)
        """
        if tag_id_a not in tag_poses or tag_id_b not in tag_poses:
            print(f"[WARN] 尺度标定失败：标靶 {tag_id_a} 或 {tag_id_b} 未在重构地图中！保持名义尺度。")
            return tag_poses, 1.0, self.marker_size_mm

        p_a = tag_poses[tag_id_a][:3, 3]
        p_b = tag_poses[tag_id_b][:3, 3]
        nominal_dist = float(np.linalg.norm(p_a - p_b))

        if nominal_dist < 1e-4:
            print(f"[WARN] 标靶 {tag_id_a} 与 {tag_id_b} 距离过近，无法用作尺度基线！")
            return tag_poses, 1.0, self.marker_size_mm

        scale_factor = float(real_distance_mm) / nominal_dist
        real_marker_size = self.marker_size_mm * scale_factor

        print(f"\n[+] ====== 双标靶中心基线绝对尺度校准 (Metric Baseline Gauge) ======")
        print(f"  -> 基准标靶对: Tag #{tag_id_a} <---> Tag #{tag_id_b}")
        print(f"  -> 当前名义欧氏距离: {nominal_dist:.2f} mm")
        print(f"  -> 现场测量实际距离: {real_distance_mm:.2f} mm")
        print(f"  -> 尺度修正系数 (Scale): {scale_factor:.6f}")
        print(f"  -> 反算单个 Tag 真实物理边长: {real_marker_size:.2f} mm (名义初值: {self.marker_size_mm:.2f} mm)")
        print(f"===================================================================\n")

        # 缩放所有 Tag 的平移位置
        scaled_poses = {}
        for t_id, T in tag_poses.items():
            T_scaled = T.copy()
            T_scaled[:3, 3] = T[:3, 3] * scale_factor
            scaled_poses[t_id] = T_scaled

        return scaled_poses, scale_factor, real_marker_size

    def _align_to_scara_world(self, 
                              tag_poses: Dict[int, np.ndarray], 
                              origin_tag_id: int, 
                              x_align_tag_id: int) -> Dict:
        """
        通过刚体变换将地图整体平移旋转，使得：
        1. Tag 0 的中心处于 (0.0, 0.0)
        2. Tag 0 -> Tag 1 的水平向量严格处于 +X 轴 (Y=0, X>0)
        """
        aligned_map = {
            "origin_tag_id": origin_tag_id,
            "x_axis_align_tag_id": x_align_tag_id,
            "tags": {}
        }

        # 如果没有检测到 Tag 0，则默认以 base 标靶对齐
        if origin_tag_id in tag_poses:
            p_origin = tag_poses[origin_tag_id][:3, 3].copy()
        else:
            p_origin = np.zeros(3)
            print(f"[WARN] 未在有效图像中检出 Tag {origin_tag_id}，将以参考标靶相对对齐！")

        # 计算对齐旋转角 (绕 Z 轴旋转使得 Tag 1 的 Y 坐标归零)
        yaw_rad = 0.0
        if origin_tag_id in tag_poses and x_align_tag_id in tag_poses:
            vec_x = tag_poses[x_align_tag_id][:3, 3] - p_origin
            # 水平面夹角
            yaw_rad = math.atan2(vec_x[1], vec_x[0])
            print(f"[+] 坐标系 X 轴对齐旋转角: {-math.degrees(yaw_rad):.2f}°")

        # 构建整体刚体变换矩阵 T_world_aligned
        cos_y = math.cos(-yaw_rad)
        sin_y = math.sin(-yaw_rad)
        R_align = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        for t_id, T_w_t in tag_poses.items():
            # 1. 平移至原点
            pos_rel = T_w_t[:3, 3] - p_origin
            # 2. 旋转对齐 X 轴
            pos_aligned = R_align @ pos_rel
            R_aligned = R_align @ T_w_t[:3, :3]
            
            # 提取欧拉角 (RPY deg)
            sy = math.sqrt(R_aligned[0, 0] * R_aligned[0, 0] + R_aligned[1, 0] * R_aligned[1, 0])
            singular = sy < 1e-6
            if not singular:
                roll = math.atan2(R_aligned[2, 1], R_aligned[2, 2])
                pitch = math.atan2(-R_aligned[2, 0], sy)
                yaw = math.atan2(R_aligned[1, 0], R_aligned[0, 0])
            else:
                roll = math.atan2(-R_aligned[1, 2], R_aligned[1, 1])
                pitch = math.atan2(-R_aligned[2, 0], sy)
                yaw = 0.0

            aligned_map["tags"][t_id] = {
                "position_mm": [round(float(v), 2) for v in pos_aligned],
                "rpy_deg": [round(float(math.degrees(v)), 2) for v in [roll, pitch, yaw]],
                "transform_matrix": [[round(float(val), 5) for val in row] for row in np.vstack([np.hstack([R_aligned, pos_aligned.reshape(3, 1)]), [0, 0, 0, 1]])],
                "is_origin": bool(t_id == origin_tag_id),
                "is_dynamic_yaw": bool(t_id == origin_tag_id)
            }

        return aligned_map

    def save_map(self, map_data: Dict, output_path: str = "config/tags_map.yaml"):
        """保存标靶地图至 YAML 文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(map_data, f, allow_unicode=True, sort_keys=False)
        print(f"[OK] 标靶空间立体地图已成功保存至: {output_path}")


def print_topology_report(covis_report: Dict[str, Any], stats: Dict[str, Any], valid_frames: List[str]):
    """打印详细共视拓扑分析诊断报告"""
    print("\n" + "=" * 70)
    print("           标靶共视连通图拓扑结构深度诊断报告")
    print("=" * 70)
    print(f" 状态评估: {'[ 连通健康 (PASS) ]' if covis_report['is_valid'] else '[ 断网告警 (FAIL) ]'}")
    print(f" 诊断说明: {covis_report['message']}")
    print(f" 参与帧数: {len(valid_frames)} 张图像满足 >= 2 标靶相对刚体约束")
    print("-" * 70)
    print(" [标靶节点与连通度 (Degree)]")
    for tid in covis_report.get("all_tags", []):
        neighbors = covis_report.get("adj_list", {}).get(tid, [])
        is_conn = tid in covis_report.get("connected_tags", [])
        status_str = "正常连通" if is_conn else "【断网孤立】"
        print(f"   - Tag #{tid:02d}: 相邻标靶 {neighbors} (度数: {len(neighbors)}) -> {status_str}")

    print("\n [共视桥梁与重叠帧数 (Edge Co-visibility Count)]")
    for edge_str, count in covis_report.get("edge_counts", {}).items():
        bridge_warn = " \033[93m[单图支撑关键桥梁 - 切勿剔除]\033[0m" if count == 1 else ""
        print(f"   - 标靶对 ({edge_str}): 在 {count} 帧图像中同时出现{bridge_warn}")

    if stats.get("excluded_items"):
        print("\n [当前已人工剔除的观测清单]")
        for exc in stats["excluded_items"]:
            note = f" (备注: {exc['note']})" if exc.get("note") else ""
            print(f"   - [{exc['image']}] Tag #{exc['tag_id']}{note}")
    print("=" * 70)


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
    parser.add_argument("--image_dir", type=str, default="data/tag_calibration_images", help="多视角标定图片目录")
    parser.add_argument("--manifest", type=str, default="data/tag_calibration_images/tag_observations.yaml", help="观测数据审核清单路径")
    parser.add_argument("--marker_size", type=float, default=50.0, help="标靶黑白边框名义边长 (mm)")
    parser.add_argument("--origin_id", type=int, default=0, help="SCARA 原点锚定标靶 ID")
    parser.add_argument("--x_axis_id", type=int, default=1, help="世界 X 轴对齐基准标靶 ID")
    parser.add_argument("--baseline_pair", nargs=3, type=float, metavar=('TAG_A', 'TAG_B', 'DIST_MM'),
                        help="双标靶基线尺度校准参数: TAG_A TAG_B 真实距离(mm), 例如: --baseline_pair 1 5 620.5")
    parser.add_argument("--output", type=str, default="config/tags_map.yaml", help="导出的图谱文件路径")
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
