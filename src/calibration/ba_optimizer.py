#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标靶空间 Bundle Adjustment (BA) 全局平差优化求解器 (Bundle Adjustment Optimizer)
- 纯面向对象单一职责设计，专注于空间静止标靶位姿与多视角相机位姿的非线性联合平差优化
- 阶段一：基于 Cauchy 鲁棒核函数的粗差自动识别与清洗 (MAD 鲁棒离群统计)
- 阶段二：微容差极致深层收敛求解
- 标靶物理间距先验约束惩罚项 (Metric Baseline Gauge)
- 基于雅可比矩阵逆的一阶 3D 空间置信度 (Uncertainty Estimation) 分析
- 世界坐标系对齐闭环 (FR-9.6 绝对坐标锚定 / Origin 锚定与 X 轴水平对齐)
- 2D 像面 Quiver Plot 残差矢量场与 Markdown 诊断报告输出
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
import numpy as np
import cv2
from scipy.optimize import least_squares

from src.calibration.covisibility_graph import CovisibilityGraphAnalyzer, CovisibilityGraphError
from src.calibration.ba_report import compute_3d_uncertainties, export_diagnostic_report

from src.utils.logger import get_logger

log = get_logger(__name__)

# 两阶段 BA 最小二乘统一收敛容差 (ftol/xtol/gtol 三项同值)
_BA_CONVERGE_TOL = 1e-5

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class BundleAdjustmentOptimizer:
    """
    BA 联合平差优化求解器
    """

    def __init__(self, 
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 marker_size_mm: float,
                 obj_points: Optional[np.ndarray] = None,
                 builder: Optional[Any] = None):
        """
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数
        :param marker_size_mm: 标靶物理边长 (mm)
        :param obj_points: 标靶局部坐标系 4 角点物理坐标 (4, 3)
        :param builder: 宿主 TagMapBuilder 实例（用于辅助观测加权等接口）
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        self.marker_size_mm = float(marker_size_mm)
        self.builder = builder

        if obj_points is not None:
            self.obj_points = np.array(obj_points, dtype=np.float64)
        else:
            s = self.marker_size_mm / 2.0
            self.obj_points = np.array([
                [-s,  s, 0.0],
                [ s,  s, 0.0],
                [ s, -s, 0.0],
                [-s, -s, 0.0]
            ], dtype=np.float64)

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

    def compute_observation_weight(self, corners: np.ndarray) -> float:
        """计算角点观测权重 (如果宿主 builder 有实现则委托，否则使用内置几何加权)"""
        if self.builder is not None and hasattr(self.builder, "compute_observation_weight"):
            return float(self.builder.compute_observation_weight(corners))

        pts = corners.reshape((4, 2)).astype(np.float64)
        area = float(cv2.contourArea(pts.astype(np.float32)))
        w_area = float(np.clip(area / 1200.0, 0.2, 1.0))

        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        center = np.mean(pts, axis=0)
        dist_from_center = np.linalg.norm(center - np.array([cx, cy]))
        max_radius = np.sqrt(cx**2 + cy**2)
        r_norm = dist_from_center / max_radius
        w_radial = 1.0 if r_norm <= 0.65 else float(np.clip(1.0 - (r_norm - 0.65) * 1.2, 0.4, 1.0))

        return float(np.clip(w_area * w_radial, 0.1, 1.0))

    def optimize(self, 
                 frame_detections: List[Dict[int, np.ndarray]], 
                 active_frame_names: Optional[List[str]] = None,
                 origin_tag_id: int = 0,
                 x_align_tag_id: int = 1,
                 baseline_pair: Optional[Tuple[int, int, float]] = None,
                 anchor_tags: Optional[Any] = None,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        基于非线性最小二乘 (Bundle Adjustment) 联合优化所有标靶位姿与相机位姿
        平差前严格调用共视连通性安全守门员校验，杜绝奇异矩阵。
        """
        if active_frame_names is None:
            active_frame_names = [f"frame_{i:04d}" for i in range(len(frame_detections))]

        # 1. 守门员审查
        report = CovisibilityGraphAnalyzer.analyze(frame_detections, active_frame_names, origin_tag_id, x_align_tag_id)
        if not report["is_valid"]:
            raise CovisibilityGraphError(report["message"])

        all_detected_tags = set(report["all_tags"])
        if report["critical_bridges"]:
            log.info(f"[NOTE] 提示：发现 {len(report['critical_bridges'])} 对标靶仅由单张图共视支撑 (关键桥梁): {report['critical_bridges']}")

        # 2. 生成高质量初值 (多标靶联合超定 PnP 初值传递，杜绝单链累积误差与翻转)
        if origin_tag_id in all_detected_tags:
            base_static_id = origin_tag_id
        elif x_align_tag_id in all_detected_tags:
            base_static_id = x_align_tag_id
        else:
            base_static_id = min(all_detected_tags)
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
                T_c_w = np.linalg.inv(T_w_c)
                tags = frame_detections[f_idx]
                for t_id, corners in tags.items():
                    if t_id not in tag_poses_init:
                        corners_2d = corners.reshape(4, 2).astype(np.float64)
                        succ, rvec, tvec = cv2.solvePnP(
                            self.obj_points, corners_2d,
                            self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                        if succ:
                            T_c_t = self.rvec_tvec_to_matrix(rvec, tvec)
                            T_w_t = T_w_c @ T_c_t
                            tag_poses_init[t_id] = T_w_t
                            changed = True

        missing_tags = all_detected_tags - set(tag_poses_init.keys())
        if missing_tags:
            raise RuntimeError(f"以下标靶未能完成初值初始化: {missing_tags}")

        active_frames = sorted(list(camera_poses_init.keys()))
        log.info(f"[+] 初值推导完成: 成功初始化 {len(tag_poses_init)} 个标靶位姿，{len(active_frames)} 个采图机位位姿")

        # 3. 计算每个观测点的初始权重 (观测加权)
        obs_weights = {}
        for f in active_frames:
            tags = frame_detections[f]
            for t_id, corners in tags.items():
                w = self.compute_observation_weight(corners)
                obs_weights[(f, t_id)] = w

        # 4. 构建两阶段非线性优化变量 (固定 base_static_id 位姿作为 Gauge Freedom 锚点)
        static_tags_to_opt = sorted([t for t in tag_poses_init.keys() if t != base_static_id])

        def pack_params(tags_dict, cams_dict):
            params = []
            for t in static_tags_to_opt:
                rv, tv = self.matrix_to_rvec_tvec(tags_dict[t])
                params.extend(rv.flatten())
                params.extend(tv.flatten())
            for f in active_frames:
                rv, tv = self.matrix_to_rvec_tvec(cams_dict[f])
                params.extend(rv.flatten())
                params.extend(tv.flatten())
            return np.array(params, dtype=np.float64)

        def unpack_params(x):
            tags_pose = {base_static_id: tag_poses_init[base_static_id].copy()}
            cams_pose = {}
            offset = 0
            for t in static_tags_to_opt:
                rv = x[offset:offset + 3]
                tv = x[offset + 3:offset + 6]
                tags_pose[t] = self.rvec_tvec_to_matrix(rv, tv)
                offset += 6
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
                        residuals.extend(diff * np.sqrt(max(1e-4, w)))

            # 若配置了物理标靶间距先验约束，作为硬约束惩罚项联合求解
            if baseline_pair is not None:
                id_a, id_b, real_dist_mm = baseline_pair
                if id_a in tags_pose and id_b in tags_pose:
                    t_a = tags_pose[id_a][:3, 3]
                    t_b = tags_pose[id_b][:3, 3]
                    dist_est = float(np.linalg.norm(t_a - t_b))
                    residuals.append((dist_est - real_dist_mm) * 5.0)

            return np.array(residuals, dtype=np.float64)

        # 内部轻量迭代步进与残差监控器
        class _OptimizationMonitor:
            def __init__(self, stage: int, stage_name: str, max_iters: int, cb=None):
                self.stage = stage
                self.stage_name = stage_name
                self.max_iters = max(max_iters, 50)
                self.cb = cb
                self.call_count = 0
                self.iter_count = 0
                self.last_x = None

            def wrap_residuals(self, base_func, weights_dict, active_outliers):
                def _wrapped(x):
                    self.call_count += 1
                    res = base_func(x, weights_dict, active_outliers)

                    # 判断是否为新的优化主步 (过滤雅可比差分时的微摄动)
                    is_new_step = False
                    if self.last_x is None:
                        is_new_step = True
                        self.last_x = x.copy()
                    else:
                        diff = np.linalg.norm(x - self.last_x)
                        if diff > 1e-4:
                            is_new_step = True
                            self.last_x = x.copy()

                    if is_new_step:
                        self.iter_count += 1
                        # 动态自适应调整最大轮次：分母永不小于分子，若超过预设则自适应平滑扩充
                        if self.iter_count > self.max_iters:
                            import math
                            self.max_iters = int(math.ceil(self.iter_count / 10.0) * 10)

                        rmse = float(np.sqrt(np.mean(res ** 2))) if len(res) > 0 else 0.0
                        # 迭代运行中进度条最高逼近 95%，收敛完成时由回调置 100%
                        sub_pct = min(0.95, self.iter_count / float(max(1, self.max_iters)))
                        if self.cb:
                            try:
                                self.cb({
                                    "stage": self.stage,
                                    "stage_name": self.stage_name,
                                    "iter": self.iter_count,
                                    "max_iter": self.max_iters,
                                    "rmse": rmse,
                                    "sub_progress": sub_pct,
                                    "call_count": self.call_count
                                })
                            except Exception as e:
                                log.warning(f"[BA] 优化进度回调异常 (已忽略): {e}")
                    return res
                return _wrapped

        x0 = pack_params(tag_poses_init, camera_poses_init)

        stage1_est_max = 60
        if callback:
            callback({
                "stage": 1,
                "stage_name": "粗差清洗与收敛",
                "iter": 0,
                "max_iter": stage1_est_max,
                "rmse": 1.0,
                "sub_progress": 0.0,
                "call_count": 0
            })

        log.info("[*] 正在执行 Phase 1 阶段一：基于 Cauchy 鲁棒核的粗差清洗与全局收敛...")
        monitor1 = _OptimizationMonitor(stage=1, stage_name="粗差清洗收敛", max_iters=stage1_est_max, cb=callback)
        res_stage1 = least_squares(
            monitor1.wrap_residuals(residuals_func, obs_weights, None), x0,
            method='trf',
            loss='cauchy',
            f_scale=1.5,
            x_scale='jac',
            ftol=_BA_CONVERGE_TOL,
            xtol=_BA_CONVERGE_TOL,
            gtol=_BA_CONVERGE_TOL,
            max_nfev=200,
            verbose=0
        )

        final_iter1 = max(1, monitor1.iter_count)
        final_max1 = max(monitor1.max_iters, final_iter1)
        if callback:
            callback({
                "stage": 1,
                "stage_name": "粗差清洗收敛",
                "iter": final_iter1,
                "max_iter": final_max1,
                "rmse": float(np.sqrt(np.mean(res_stage1.fun ** 2))) if len(res_stage1.fun) > 0 else 0.0,
                "sub_progress": 1.0,
                "call_count": monitor1.call_count
            })

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

        med_e = float(np.median(raw_errors))
        mad_e = float(np.median(np.abs(np.array(raw_errors) - med_e)))
        sigma_robust = max(0.5, 1.4826 * mad_e)
        outlier_thresh = max(4.0, med_e + 2.8 * sigma_robust)
        outliers_detected = set()
        for f, t_id, e in obs_map:
            if e > outlier_thresh or obs_weights.get((f, t_id), 1.0) <= 0.05:
                outliers_detected.add((f, t_id))

        if outliers_detected:
            log.info(f"[CLEAN] 自动清洗识别出 {len(outliers_detected)} 个潜在粗差/远景噪点观测 (MAD 门限 > {outlier_thresh:.2f}px, 中位数={med_e:.2f}px):")
            for f, t_id in sorted(list(outliers_detected)):
                f_name = active_frame_names[f] if f < len(active_frame_names) else f"frame_{f}"
                log.info(f"        - [{f_name}] Tag #{t_id}")

        max_iters_p2 = 60
        if callback:
            callback({
                "stage": 2,
                "stage_name": "微容差深度平差",
                "iter": 0,
                "max_iter": max_iters_p2,
                "rmse": float(np.sqrt(np.mean(res_stage1.fun ** 2))) if len(res_stage1.fun) > 0 else 0.0,
                "sub_progress": 0.0,
                "call_count": 0
            })

        log.info("[*] 正在执行 Phase 1 阶段二：微容差 (ftol=1e-5) 极致深层平差收敛...")
        monitor2 = _OptimizationMonitor(stage=2, stage_name="微容差深度平差", max_iters=max_iters_p2, cb=callback)
        res_stage2 = least_squares(
            monitor2.wrap_residuals(residuals_func, obs_weights, outliers_detected), res_stage1.x,
            method='trf',
            loss='cauchy',
            f_scale=1.0,
            x_scale='jac',
            ftol=_BA_CONVERGE_TOL,
            xtol=_BA_CONVERGE_TOL,
            gtol=_BA_CONVERGE_TOL,
            max_nfev=200,
            verbose=0
        )

        final_iter2 = max(1, monitor2.iter_count)
        final_max2 = max(monitor2.max_iters, final_iter2)
        if callback:
            callback({
                "stage": 2,
                "stage_name": "微容差深度平差",
                "iter": final_iter2,
                "max_iter": final_max2,
                "rmse": float(np.sqrt(np.mean(res_stage2.fun ** 2))) if len(res_stage2.fun) > 0 else 0.0,
                "sub_progress": 1.0,
                "call_count": monitor2.call_count
            })

        optimized_tags_pose, optimized_cams_pose = unpack_params(res_stage2.x)

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
        log.info(f"[OK] 两阶段 BA 极限优化完成！有效观测像面 RMSE: {rmse_px:.3f} 像素 (迭代次数: {res_stage2.nfev})")

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

        # 7. 坐标系解耦逻辑:
        #    - 若未指定 anchor_tags: 纯自由平差 (阶段一), 输出纯视觉相对几何地图 (以 base_static_id 为相对原点)
        #    - 若指定了 anchor_tags: 世界绝对锚定 (阶段二), 求解 3D 相似变换变换至世界系
        if anchor_tags:
            final_tags_map = self.anchor_to_absolute_world(
                optimized_tags_pose, anchor_tags,
                origin_tag_id=origin_tag_id, x_align_tag_id=x_align_tag_id,
                strict=True
            )
        else:
            # 阶段一: 纯自由平差相对地图
            tags_dict = {}
            for tid, T in optimized_tags_pose.items():
                pos = T[:3, 3]
                roll, pitch, yaw = self._rotation_to_rpy_deg(T[:3, :3])
                tags_dict[tid] = {
                    "position_mm": [round(float(v), 2) for v in pos],
                    "rpy_deg": [round(float(math.degrees(v)), 2) for v in [roll, pitch, yaw]],
                    "transform_matrix": [[round(float(val), 5) for val in row] for row in T],
                    "is_origin": bool(tid == base_static_id),
                    "is_dynamic_yaw": bool(tid == base_static_id)
                }
            final_tags_map = {
                "origin_tag_id": base_static_id,
                "x_axis_align_tag_id": x_align_tag_id,
                "anchor_mode": "unaligned",
                "tags": tags_dict
            }
            log.info(f"[+] [FREE_BA] 阶段一纯视觉自由平差完成: 基准标靶 Tag #{base_static_id}, 相对构型已固化 (尚未校准世界系)")

        # 始终保存一份未经世界变换的纯相对矩阵 (供后续随时独立做世界系校准)
        final_tags_map["raw_relative_poses"] = {
            int(tid): [[round(float(val), 5) for val in row] for row in T]
            for tid, T in optimized_tags_pose.items()
        }

        final_tags_map["rmse_reprojection_px"] = rmse_px
        final_tags_map["marker_size_mm"] = round(float(self.marker_size_mm), 3)
        final_tags_map["tag_family"] = "DICT_APRILTAG_16h5"
        final_tags_map["calibrated_images_count"] = len(active_frames)
        final_tags_map["cleaned_outliers_count"] = len(outliers_detected)
        final_tags_map["final_rmse"] = rmse_px
        final_tags_map["final_tag_poses_aligned"] = {
            tid: np.array(t_info["transform_matrix"], dtype=np.float64)
            for tid, t_info in final_tags_map.get("tags", {}).items()
        }
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
            log.warning(f"[WARN] 导出深度诊断报告异常 (已安全忽略): {e}")

        return final_tags_map

    def compute_3d_uncertainties(self, jacobian, static_tags, base_id, sigma_res_px) -> Dict[int, Dict[str, float]]:
        """标靶 3D 置信区间计算（委托 ba_report 纯函数实现）"""
        return compute_3d_uncertainties(jacobian, static_tags, base_id, sigma_res_px)

    def export_diagnostic_report(self,
                                 final_tags_map: Dict[str, Any],
                                 detailed_obs_res: List[Dict[str, Any]],
                                 active_frame_names: List[str],
                                 tag_uncertainties: Dict[int, Dict[str, float]],
                                 outliers_detected: Set[Tuple[int, int]],
                                 rmse_px: float,
                                 report_dir: Optional[str] = None) -> str:
        """Quiver 残差矢量场与 Markdown 精度体检报告生成（委托 ba_report 纯函数实现）"""
        return export_diagnostic_report(
            final_tags_map=final_tags_map,
            detailed_obs_res=detailed_obs_res,
            active_frame_names=active_frame_names,
            tag_uncertainties=tag_uncertainties,
            outliers_detected=outliers_detected,
            rmse_px=rmse_px,
            report_dir=report_dir
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
            log.warning(f"[WARN] 尺度标定失败：标靶 {tag_id_a} 或 {tag_id_b} 未在重构地图中！保持名义尺度。")
            return tag_poses, 1.0, self.marker_size_mm

        p_a = tag_poses[tag_id_a][:3, 3]
        p_b = tag_poses[tag_id_b][:3, 3]
        nominal_dist = float(np.linalg.norm(p_a - p_b))

        if nominal_dist < 1e-4:
            log.warning(f"[WARN] 标靶 {tag_id_a} 与 {tag_id_b} 距离过近，无法用作尺度基线！")
            return tag_poses, 1.0, self.marker_size_mm

        scale_factor = float(real_distance_mm) / nominal_dist
        real_marker_size = self.marker_size_mm * scale_factor

        log.info(f"\n[+] ====== 双标靶中心基线绝对尺度校准 (Metric Baseline Gauge) ======")
        log.info(f"  -> 基准标靶对: Tag #{tag_id_a} <---> Tag #{tag_id_b}")
        log.info(f"  -> 当前名义欧氏距离: {nominal_dist:.2f} mm")
        log.info(f"  -> 现场测量实际距离: {real_distance_mm:.2f} mm")
        log.info(f"  -> 尺度修正系数 (Scale): {scale_factor:.6f}")
        log.info(f"  -> 反算单个 Tag 真实物理边长: {real_marker_size:.2f} mm (名义初值: {self.marker_size_mm:.2f} mm)")
        log.info(f"===================================================================\n")

        scaled_poses = {}
        for t_id, T in tag_poses.items():
            T_scaled = T.copy()
            T_scaled[:3, 3] = T[:3, 3] * scale_factor
            scaled_poses[t_id] = T_scaled

        return scaled_poses, scale_factor, real_marker_size

    @staticmethod
    def _rotation_to_rpy_deg(R: np.ndarray) -> Tuple[float, float, float]:
        """3x3 旋转矩阵 -> (roll, pitch, yaw) 弧度 (含万向锁奇异保护)"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    # ------------------------------------------------------------------
    # FR-9.6 世界系绝对锚定 (约束积累式: 支持任意数量全知/部分已知锚点)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_anchor_tags(anchor_input: Any) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        锚点配置格式归一化 (兼容两种输入):
        - 新格式: {tag_id: {"xyz_mm": [x,y,z], "known": [b,b,b]}} (known 缺省视为三轴全知)
        - 旧格式: {"origin_tag_id": i, "origin_xyz_mm": [...], "align_tag_id": j, "align_xyz_mm": [...]}
        :return: 统一新格式字典; 无有效锚点返回 None
        """
        if not anchor_input or not isinstance(anchor_input, dict):
            return None
        out: Dict[int, Dict[str, Any]] = {}

        def _add(tid: Any, xyz: Any, known: Any) -> None:
            try:
                tid_i = int(tid)
                xyz_f = [float(v) for v in xyz]
            except (TypeError, ValueError):
                return
            if len(xyz_f) != 3:
                return
            known_raw = list(known) if isinstance(known, (list, tuple)) else []
            known_b = [bool(v) for v in (known_raw + [True, True, True])[:3]]
            if not any(known_b):
                return
            out[tid_i] = {"xyz_mm": xyz_f, "known": known_b}

        if "origin_xyz_mm" in anchor_input or "align_xyz_mm" in anchor_input:
            # 旧双锚点格式: origin/align 均视为三轴全知
            _add(anchor_input.get("origin_tag_id", 0), anchor_input.get("origin_xyz_mm"), [True, True, True])
            _add(anchor_input.get("align_tag_id", 1), anchor_input.get("align_xyz_mm"), [True, True, True])
        else:
            for k, v in anchor_input.items():
                if isinstance(v, dict):
                    _add(k, v.get("xyz_mm"), v.get("known"))
        return out or None

    @staticmethod
    def evaluate_anchor_dof(anchor_tags: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        配置级自由度记账 (UI 实时状态与求解器共用, 与当帧实际检出无关)。
        重力先验 (BA 系 Z 轴指向天) 固定 roll/pitch 后剩 5 DoF:
        - 尺度 s: 存在共同已知轴且距离非零的锚点对 (中位数聚合)
        - 偏航 yaw: 存在共同已知 XY 的锚点对 (连线方向)
        - t_x / t_y / t_z: 任一锚点已知对应轴
        :return: {"mode": full|partial|none, "dof_solved": 0~5, "dof_total": 5,
                  "scale_pairs": [(ia, ib)...], "yaw_pairs": [...],
                  "t_axes": [has_x, has_y, has_z], "reason": str}
        """
        tids = sorted(anchor_tags.keys())
        scale_pairs: List[Tuple[int, int]] = []
        yaw_pairs: List[Tuple[int, int]] = []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = anchor_tags[tids[i]], anchor_tags[tids[j]]
                common = [k for k in range(3) if a["known"][k] and b["known"][k]]
                if not common:
                    continue
                d_w = math.sqrt(sum((a["xyz_mm"][k] - b["xyz_mm"][k]) ** 2 for k in common))
                if d_w < 1e-6:
                    continue  # 共同已知轴上重合, 该对无尺度信息
                scale_pairs.append((tids[i], tids[j]))
                if all(a["known"][k] and b["known"][k] for k in (0, 1)):
                    d_xy = math.hypot(a["xyz_mm"][0] - b["xyz_mm"][0], a["xyz_mm"][1] - b["xyz_mm"][1])
                    if d_xy >= 1e-6:
                        yaw_pairs.append((tids[i], tids[j]))
        has_x = any(a["known"][0] for a in anchor_tags.values())
        has_y = any(a["known"][1] for a in anchor_tags.values())
        has_z = any(a["known"][2] for a in anchor_tags.values())
        dof = (1 if scale_pairs else 0) + (1 if yaw_pairs else 0) + int(has_x) + int(has_y) + int(has_z)

        if scale_pairs and yaw_pairs and has_x and has_y:
            mode = "full" if has_z else "partial"
        else:
            mode = "none"
        reason = ""
        if mode == "none":
            if not anchor_tags:
                reason = "未配置任何世界锚点"
            elif not scale_pairs:
                reason = "无共同已知轴且距离非零的锚点对, 尺度不可解 (不允许打印边长兜底)"
            elif not yaw_pairs:
                reason = "无共同已知 XY 的锚点对, 偏航不可解"
            else:
                reason = "X/Y 平移约束不足 (需至少一枚已知 X 与一枚已知 Y 的锚点)"
        return {"mode": mode, "dof_solved": dof, "dof_total": 5,
                "scale_pairs": scale_pairs, "yaw_pairs": yaw_pairs,
                "t_axes": [has_x, has_y, has_z], "reason": reason}

    @staticmethod
    def _umeyama_alignment(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Umeyama 算法求解最小二乘 3D 相似变换: dst = s * (src @ R.T) + t
        :param src_pts: Nx3 (BA 坐标点云)
        :param dst_pts: Nx3 (世界目标坐标真值)
        :return: (s, R, t) 其中 R in SO(3), det(R) = +1
        """
        n, m = src_pts.shape
        mu_src = np.mean(src_pts, axis=0)
        mu_dst = np.mean(dst_pts, axis=0)

        src_centered = src_pts - mu_src
        dst_centered = dst_pts - mu_dst

        var_src = np.mean(np.sum(src_centered ** 2, axis=1))
        if var_src < 1e-9:
            return 1.0, np.eye(3), mu_dst - mu_src

        H = (dst_centered.T @ src_centered) / n
        U, D, Vt = np.linalg.svd(H)
        S = np.eye(m)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[m - 1, m - 1] = -1

        R = U @ S @ Vt
        s = float((1.0 / var_src) * np.trace(np.diag(D) @ S))
        t = mu_dst - s * (R @ mu_src)
        return s, R, t

    @staticmethod
    def solve_similarity_from_anchors(tag_poses: Dict[int, np.ndarray],
                                      anchor_tags: Dict[int, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        自适应求解 BA 系 -> 世界系相似变换 (世界坐标系优先):
        - 优先分支 (Umeyama 3D): 当存在 >=3 枚三轴全知且非共线锚点时, 采用闭式解析 Umeyama 算法
          求解全局最优 3D 刚体旋转 R in SO(3) 与尺度/平移, 彻底解除世界系法向对单个基准 Tag
          自身平贴倾角的绑架, 使得世界坐标系严格以用户标定的 3D 地面真值为绝对基准!
        - 降级分支 (Planar 2D + Z平移): 当仅有 2 枚锚点或仅已知部分轴时, 以共同轴测距求解尺度,
          以共同 XY 方向求解偏航角 yaw 并独立平移各轴.
        - 锚点一致性守门: 对所有锚点对的世界几何距离与相机重构距离进行相对形变校验, 发现严重录入
          冲突时记录警告, 杜绝错误几何污染全图.
        :return: (mode, info); mode in full|partial|none
        """
        usable = {tid: a for tid, a in anchor_tags.items() if tid in tag_poses}
        missing = sorted(tid for tid in anchor_tags if tid not in tag_poses)
        if missing:
            log.warning(f"[ANCHOR] 配置的世界锚点标靶未参与本次平差解算 (已忽略其约束): Tag {missing}")
        if not usable:
            return "none", {"reason": "配置的世界锚点标靶均未参与本次平差解算"}

        p_ba = {tid: tag_poses[tid][:3, 3].astype(np.float64) for tid in usable}
        tids = sorted(usable.keys())

        # ① 收集全部可用锚点对在共同已知轴下的距离比 (尺度观测)
        scale_obs: List[Tuple[int, int, float, float, float]] = []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                ia, ib = tids[i], tids[j]
                a, b = usable[ia], usable[ib]
                common = [k for k in range(3) if a["known"][k] and b["known"][k]]
                if not common:
                    continue
                d_w = math.sqrt(sum((a["xyz_mm"][k] - b["xyz_mm"][k]) ** 2 for k in common))
                d_b = math.sqrt(sum((p_ba[ia][k] - p_ba[ib][k]) ** 2 for k in common))
                if d_w < 1e-6 or d_b < 1e-6:
                    continue  # 世界系或 BA 系在共同已知轴上重合, 该对无尺度信息
                scale_obs.append((ia, ib, d_w / d_b, d_w, d_b))

        if not scale_obs:
            return "none", {"reason": "无任何锚点对存在共同已知轴 (或全部重合), 尺度不可解 — 不允许以打印边长兜底"}

        ratios = [r[2] for r in scale_obs]
        scale_median = float(np.median(ratios))

        # 锚点几何形变与录入冲突校验
        conflict_pairs = []
        for ia, ib, ratio, d_w, d_b in scale_obs:
            expected_dw = scale_median * d_b
            abs_diff = abs(d_w - expected_dw)
            rel_diff = abs_diff / max(1e-3, expected_dw)
            if abs_diff > 15.0 and rel_diff > 0.15:
                conflict_pairs.append({
                    "pair": (ia, ib),
                    "world_dist_mm": round(d_w, 2),
                    "measured_dist_mm": round(expected_dw, 2),
                    "diff_mm": round(abs_diff, 2),
                    "rel_error": round(rel_diff, 3)
                })
                log.warning(
                    f"[ANCHOR CONFLICT] 锚点几何严重冲突! Tag #{ia} 与 Tag #{ib} 间输入的世界距离为 {d_w:.1f} mm, "
                    f"但相机视觉重构等效距离约为 {expected_dw:.1f} mm (偏差 {abs_diff:.1f} mm, 相对误差 {rel_diff*100:.1f}%)! "
                    f"请务必核对工位锚点/白名单中这两个标靶的已知世界坐标输入!"
                )

        # ② 检查是否有 >=3 枚三轴全知锚点且在 3D 空间有效非共线 -> 优先使用 Umeyama 3D 相似变换
        full_3d_tids = [tid for tid in tids if all(usable[tid]["known"])]
        use_umeyama = False
        if len(full_3d_tids) >= 3:
            src_test = np.array([p_ba[tid] for tid in full_3d_tids])
            src_c = src_test - np.mean(src_test, axis=0)
            _, sv_src, _ = np.linalg.svd(src_c)
            # 有效空间秩 >= 2 (非单一退化直线)
            if len(sv_src) >= 2 and sv_src[1] > 1e-2:
                use_umeyama = True

        if use_umeyama:
            src = np.array([p_ba[tid] for tid in full_3d_tids])
            dst = np.array([usable[tid]["xyz_mm"] for tid in full_3d_tids])
            scale, R, t = BundleAdjustmentOptimizer._umeyama_alignment(src, dst)
            mode = "full"
            yaw = math.atan2(R[1, 0], R[0, 0])
            yaw_pairs_count = len(full_3d_tids) * (len(full_3d_tids) - 1) // 2
            solver_type = "umeyama_3d"
        else:
            # 降级分支: 2D 偏航 + 独立 Z 平移 (兼容双锚点及部分已知轴模式)
            yaw_obs: List[float] = []
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    ia, ib = tids[i], tids[j]
                    a, b = usable[ia], usable[ib]
                    if all(a["known"][k] and b["known"][k] for k in (0, 1)):
                        wdx, wdy = a["xyz_mm"][0] - b["xyz_mm"][0], a["xyz_mm"][1] - b["xyz_mm"][1]
                        bdx, bdy = p_ba[ia][0] - p_ba[ib][0], p_ba[ia][1] - p_ba[ib][1]
                        if math.hypot(wdx, wdy) >= 1e-6 and math.hypot(bdx, bdy) >= 1e-6:
                            yaw_obs.append(math.atan2(wdy, wdx) - math.atan2(bdy, bdx))

            if not yaw_obs:
                return "none", {"reason": "无任何共同已知 XY 的锚点对, 偏航不可解"}

            scale = scale_median
            yaw = math.atan2(sum(math.sin(v) for v in yaw_obs), sum(math.cos(v) for v in yaw_obs))
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            R = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

            # ③ 平移: 逐轴对已知锚点残差取均值 (缺失轴为 None)
            t_axes: List[Optional[float]] = []
            for axis in range(3):
                vals = [a["xyz_mm"][axis] - scale * float((R @ p_ba[tid])[axis])
                        for tid, a in usable.items() if a["known"][axis]]
                t_axes.append(float(np.mean(vals)) if vals else None)
            if t_axes[0] is None or t_axes[1] is None:
                return "none", {"reason": "X/Y 平移约束不足 (需至少一枚已知 X 与一枚已知 Y 的锚点)"}
            t = np.array([v if v is not None else 0.0 for v in t_axes], dtype=np.float64)
            mode = "full" if t_axes[2] is not None else "partial"
            yaw_pairs_count = len(yaw_obs)
            solver_type = "planar_2d"

        # ④ 残差: 各锚点已知轴的变换后偏差 (锚定质量指标)
        per_tag: Dict[str, List[float]] = {}
        all_res: List[float] = []
        for tid, a in usable.items():
            p_w = scale * (R @ p_ba[tid]) + t
            res = [float(p_w[k] - a["xyz_mm"][k]) for k in range(3) if a["known"][k]]
            if res:
                per_tag[str(tid)] = [round(v, 3) for v in res]
                all_res.extend(res)
        residuals = {
            "mean_mm": round(float(np.mean(np.abs(all_res))), 3) if all_res else 0.0,
            "max_mm": round(float(np.max(np.abs(all_res))), 3) if all_res else 0.0,
            "per_tag": per_tag
        }
        info = {
            "scale_factor": scale, "yaw_rad": yaw, "R": R, "t": t,
            "scale_pair_count": len(scale_obs), "yaw_pair_count": yaw_pairs_count,
            "residuals": residuals,
            "solver_type": solver_type,
            "conflict_pairs": conflict_pairs
        }
        return mode, info

    def _anchor_fallback_relative(self,
                                  tag_poses: Dict[int, np.ndarray],
                                  origin_tag_id: int,
                                  x_align_tag_id: int,
                                  reason: str) -> Dict[str, Any]:
        """锚定不可行时的统一退化出口: 相对对齐 + none 模式标记"""
        log.warning(f"[ANCHOR] 世界锚定不可行: {reason} — 退化为相对对齐！")
        rel = self.align_to_scara_world(tag_poses, origin_tag_id, x_align_tag_id)
        rel["anchor_mode"] = "none"
        rel["anchor_skip_reason"] = reason
        return rel

    def anchor_to_absolute_world(self,
                                 tag_poses: Dict[int, np.ndarray],
                                 anchor_input: Any,
                                 origin_tag_id: int = 0,
                                 x_align_tag_id: int = 1,
                                 strict: bool = False) -> Dict[str, Any]:
        """
        FR-9.6 世界坐标系绝对锚定 (约束积累式):
        支持任意数量全知/部分已知锚点 (逐轴 known 标记), 重力先验 (BA 系 Z 轴指向天) 固定 roll/pitch,
        分阶段闭式求解相似变换 (尺度 s + 偏航 yaw + 平移 t) 将整张 BA 平差地图变换到机械臂世界坐标系:
        - full:    5/5 DoF 全部解算, 绝对世界系地图
        - partial: 仅 XY 链可解 (t_z 悬空), XY 绝对锚定 + Z 保持 BA 尺度相对坐标, 下游须按 anchor_mode 守门
        - none:    约束不足或退化, strict=True 抛错终止, strict=False 退化为 align_to_scara_world 相对对齐
        :param anchor_input: 新格式 {tid: {"xyz_mm","known"}} 或旧双锚点格式 (自动归一化)
        :param strict: 是否严格模式 (平差主干默认 True, 禁止任何静默兜底)
        """
        anchor_tags = self.normalize_anchor_tags(anchor_input)
        if not anchor_tags:
            if strict:
                raise ValueError("anchor_to_absolute_world: 锚点配置为空或字段不合法 (FR-9.6 禁止兜底)")
            return self._anchor_fallback_relative(tag_poses, origin_tag_id, x_align_tag_id, "锚点配置为空或字段不合法")

        dof = self.evaluate_anchor_dof(anchor_tags)
        if dof["mode"] == "none":
            if strict:
                raise ValueError(
                    f"锚点 DoF 约束不足 ({dof['dof_solved']}/5): {dof['reason']}"
                    " — 请增配已知世界坐标的 tag 或放宽当前部分已知标记 (FR-9.6 禁止兜底)"
                )
            return self._anchor_fallback_relative(tag_poses, origin_tag_id, x_align_tag_id, dof["reason"])

        mode, solve = self.solve_similarity_from_anchors(tag_poses, anchor_tags)
        if mode == "none":
            if strict:
                raise ValueError(
                    f"锚点求解退化: {solve['reason']}"
                    " — 请检查锚点 tag 之间的已知轴距离与 XY 共线方向 (FR-9.6 禁止兜底)"
                )
            return self._anchor_fallback_relative(tag_poses, origin_tag_id, x_align_tag_id, solve["reason"])

        scale = solve["scale_factor"]
        R = solve["R"]
        t_vec = solve["t"]

        # 尺度修正同步作用于边长模型: BA 以名义边长建模, 真实边长 = 名义 × 锚定尺度
        # (与 apply_baseline_scale 的 Metric Baseline Gauge 语义一致, 否则世界角点云与
        #  单靶 PnP 模型比例失真, 导致绿/蓝棱柱尺寸与空间偏差系统性错误)
        self.marker_size_mm = self.marker_size_mm * scale

        res = solve["residuals"]
        solver_type = solve.get("solver_type", "planar_2d")
        conflict_pairs = solve.get("conflict_pairs", [])
        aligned_map = {
            "origin_tag_id": origin_tag_id,
            "x_axis_align_tag_id": x_align_tag_id,
            "anchor_mode": mode,
            "world_anchor": {
                "anchor_mode": mode,
                "solver_type": solver_type,
                "scale_factor": round(float(scale), 6),
                "yaw_deg": round(float(math.degrees(solve["yaw_rad"])), 3),
                "t_xyz_mm": [round(float(v), 3) for v in t_vec],
                "scale_pair_count": int(solve["scale_pair_count"]),
                "yaw_pair_count": int(solve["yaw_pair_count"]),
                "anchor_residual_mm": res,
                "anchor_tags": {str(tid): {"xyz_mm": [round(float(v), 3) for v in a["xyz_mm"]],
                                           "known": [bool(v) for v in a["known"]]}
                                for tid, a in sorted(anchor_tags.items())},
                "conflict_pairs": conflict_pairs,
                "real_marker_size_mm": round(float(self.marker_size_mm), 3)
            },
            "tags": {}
        }
        log.info(f"[+] [ANCHOR] FR-9.6 世界系锚定完成 (mode={mode}, solver={solver_type}): "
              f"尺度因子: {scale:.6f} ({solve['scale_pair_count']} 对), 偏航: {math.degrees(solve['yaw_rad']):.2f}°, "
              f"锚点残差 mean/max: {res['mean_mm']:.2f}/{res['max_mm']:.2f} mm, "
              f"反算真实边长: {self.marker_size_mm:.2f} mm")
        if conflict_pairs:
            log.warning(f"[ANCHOR] ⚠️ 注意：检测到 {len(conflict_pairs)} 组锚点输入存在严重几何形变冲突，可能影响世界对齐精度！请检查锚点录入。")
        if mode == "partial":
            log.warning("[ANCHOR] partial 模式: XY 已绝对锚定, Z 轴保持 BA 尺度相对坐标 (t_z 悬空) — 下游须按 anchor_mode 守门！")

        for t_id, T_w_t in tag_poses.items():
            pos_aligned = scale * (R @ T_w_t[:3, 3]) + t_vec
            R_aligned = R @ T_w_t[:3, :3]
            roll, pitch, yaw = self._rotation_to_rpy_deg(R_aligned)

            aligned_map["tags"][t_id] = {
                "position_mm": [round(float(v), 2) for v in pos_aligned],
                "rpy_deg": [round(float(math.degrees(v)), 2) for v in [roll, pitch, yaw]],
                "transform_matrix": [[round(float(val), 5) for val in row] for row in np.vstack([np.hstack([R_aligned, pos_aligned.reshape(3, 1)]), [0, 0, 0, 1]])],
                "is_origin": bool(t_id == origin_tag_id),
                "is_dynamic_yaw": bool(t_id == origin_tag_id)
            }

        return aligned_map

    def align_relative_map_to_world(self,
                                    relative_map: Dict[str, Any],
                                    anchor_tags: Dict[int, Dict[str, Any]],
                                    origin_tag_id: int = 0,
                                    x_align_tag_id: int = 1,
                                    strict: bool = True) -> Dict[str, Any]:
        """
        【阶段二独立解算核心】将自由平差产出的相对几何底图对齐到世界坐标系
        :param relative_map: 阶段一产出的相对底图 (包含 raw_relative_poses 或 tags 的位姿矩阵)
        :param anchor_tags: 用户录入的世界锚点真值表
        :param origin_tag_id: 原点标靶 ID
        :param x_align_tag_id: X 轴对齐标靶 ID
        :param strict: 是否严格模式 (发现冲突或锚点不足时抛错，否则降级)
        :return: 具有绝对世界坐标的完整 tags_map 生产字典
        """
        # 提取标靶 4x4 位姿矩阵 (优先使用 raw_relative_poses)
        tag_poses: Dict[int, np.ndarray] = {}
        if "raw_relative_poses" in relative_map:
            for tid, mat in relative_map["raw_relative_poses"].items():
                tag_poses[int(tid)] = np.array(mat, dtype=np.float64)
        elif "tags" in relative_map:
            for tid, t_info in relative_map["tags"].items():
                if "transform_matrix" in t_info:
                    tag_poses[int(tid)] = np.array(t_info["transform_matrix"], dtype=np.float64)

        if not tag_poses:
            raise ValueError("align_relative_map_to_world: 相对底图中未找到任何有效的标靶位姿矩阵")

        # 同步名义边长
        if "marker_size_mm" in relative_map:
            self.marker_size_mm = float(relative_map["marker_size_mm"])

        aligned = self.anchor_to_absolute_world(
            tag_poses, anchor_tags,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            strict=strict
        )

        # 完整继承阶段一的所有平差质检与元数据属性
        result_map = dict(relative_map)
        result_map["tags"] = aligned["tags"]
        result_map["world_anchor"] = aligned.get("world_anchor", {})
        result_map["anchor_mode"] = aligned.get("anchor_mode", "full")
        result_map["marker_size_mm"] = round(float(self.marker_size_mm), 3)
        if "raw_relative_poses" not in result_map:
            result_map["raw_relative_poses"] = {
                int(tid): [[round(float(val), 5) for val in row] for row in T]
                for tid, T in tag_poses.items()
            }
        return result_map

    def align_to_scara_world(self,
                             tag_poses: Dict[int, np.ndarray], 
                             origin_tag_id: int, 
                             x_align_tag_id: int) -> Dict[str, Any]:
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

        if origin_tag_id in tag_poses:
            p_origin = tag_poses[origin_tag_id][:3, 3].copy()
        else:
            p_origin = np.zeros(3)
            log.warning(f"[WARN] 未在有效图像中检出 Tag {origin_tag_id}，将以参考标靶相对对齐！")

        yaw_rad = 0.0
        aligned_x_target_id = x_align_tag_id

        if origin_tag_id in tag_poses and x_align_tag_id in tag_poses:
            vec_x = tag_poses[x_align_tag_id][:3, 3] - p_origin
            yaw_rad = math.atan2(vec_x[1], vec_x[0])
            log.info(f"[+] [ALIGN] 成功锚定基准 Tag {origin_tag_id} -> Tag {x_align_tag_id}，坐标系 X 轴对齐旋转角: {-math.degrees(yaw_rad):.2f}°")
        else:
            # 指定对齐标靶缺失，打印显式告警
            available_tags = [tid for tid in tag_poses.keys() if tid != origin_tag_id]
            log.warning(f"[WARN] [ALIGN] 指定的 X 轴对齐标靶 Tag {x_align_tag_id} 不在解算标靶中 (可用静态标靶: {sorted(available_tags)})！")
            
            # 自适应寻找候选远端标靶（水平距离最大且在有效范围内的标靶）
            if available_tags and origin_tag_id in tag_poses:
                candidate_dists = []
                for tid in available_tags:
                    dist_xy = np.linalg.norm(tag_poses[tid][:2, 3] - p_origin[:2])
                    candidate_dists.append((dist_xy, tid))
                candidate_dists.sort(reverse=True)
                fallback_id = candidate_dists[0][1]
                aligned_x_target_id = fallback_id
                vec_x = tag_poses[fallback_id][:3, 3] - p_origin
                yaw_rad = math.atan2(vec_x[1], vec_x[0])
                log.warning(f"[!] [ALIGN] 自动降级使用最远端刚体标靶 Tag {fallback_id} (距离 {candidate_dists[0][0]:.1f}mm) 进行 X 轴定向校正: {-math.degrees(yaw_rad):.2f}°")
            else:
                log.error(f"[ERROR] [ALIGN] 无法进行世界 X 轴对齐，世界系方向将退化保持为基准标靶印刷朝向！")

        aligned_map["x_axis_align_tag_id"] = aligned_x_target_id

        cos_y = math.cos(-yaw_rad)
        sin_y = math.sin(-yaw_rad)
        R_align = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        for t_id, T_w_t in tag_poses.items():
            pos_rel = T_w_t[:3, 3] - p_origin
            pos_aligned = R_align @ pos_rel
            R_aligned = R_align @ T_w_t[:3, :3]
            roll, pitch, yaw = self._rotation_to_rpy_deg(R_aligned)

            aligned_map["tags"][t_id] = {
                "position_mm": [round(float(v), 2) for v in pos_aligned],
                "rpy_deg": [round(float(math.degrees(v)), 2) for v in [roll, pitch, yaw]],
                "transform_matrix": [[round(float(val), 5) for val in row] for row in np.vstack([np.hstack([R_aligned, pos_aligned.reshape(3, 1)]), [0, 0, 0, 1]])],
                "is_origin": bool(t_id == origin_tag_id),
                "is_dynamic_yaw": bool(t_id == origin_tag_id)
            }

        return aligned_map
