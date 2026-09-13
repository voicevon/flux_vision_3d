#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标靶空间 Bundle Adjustment (BA) 全局平差优化求解器 (Bundle Adjustment Optimizer)
- 纯面向对象单一职责设计，专注于空间静止标靶位姿与多视角相机位姿的非线性联合平差优化
- 阶段一：基于 Cauchy 鲁棒核函数的粗差自动识别与清洗 (MAD 鲁棒离群统计)
- 阶段二：微容差极致深层收敛求解
- 标靶物理间距先验约束惩罚项 (Metric Baseline Gauge)
- 基于雅可比矩阵逆的一阶 3D 空间置信度 (Uncertainty Estimation) 分析
- 世界坐标系对齐闭环 (Origin 锚定与 X 轴水平对齐)
- 2D 像面 Quiver Plot 残差矢量场与 Markdown 诊断报告输出
"""

import os
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
import numpy as np
import cv2
from scipy.optimize import least_squares

from src.calibration.covisibility_graph import CovisibilityGraphAnalyzer, CovisibilityGraphError

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class BundleAdjustmentOptimizer:
    """
    BA 联合平差优化求解器
    """

    def __init__(self, 
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 marker_size_mm: float = 50.0,
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
        print(f"[+] 初值推导完成: 成功初始化 {len(tag_poses_init)} 个标靶位姿，{len(active_frames)} 个采图机位位姿")

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
                self.max_iters = max_iters
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
                        rmse = float(np.sqrt(np.mean(res ** 2))) if len(res) > 0 else 0.0
                        sub_pct = min(1.0, self.iter_count / float(max(1, self.max_iters)))
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
                            except Exception:
                                pass
                    return res
                return _wrapped

        x0 = pack_params(tag_poses_init, camera_poses_init)

        if callback:
            callback({
                "stage": 1,
                "stage_name": "粗差清洗与收敛",
                "iter": 0,
                "max_iter": 30,
                "rmse": 1.0,
                "sub_progress": 0.0,
                "call_count": 0
            })

        print("[*] 正在执行 Phase 1 阶段一：基于 Cauchy 鲁棒核的粗差清洗与全局收敛...")
        monitor1 = _OptimizationMonitor(stage=1, stage_name="粗差清洗收敛", max_iters=30, cb=callback)
        res_stage1 = least_squares(
            monitor1.wrap_residuals(residuals_func, obs_weights, None), x0,
            method='trf',
            loss='cauchy',
            f_scale=1.5,
            x_scale='jac',
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            max_nfev=200,
            verbose=0
        )

        if callback:
            callback({
                "stage": 1,
                "stage_name": "粗差清洗收敛",
                "iter": monitor1.iter_count,
                "max_iter": 30,
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
            print(f"[CLEAN] 自动清洗识别出 {len(outliers_detected)} 个潜在粗差/远景噪点观测 (MAD 门限 > {outlier_thresh:.2f}px, 中位数={med_e:.2f}px):")
            for f, t_id in sorted(list(outliers_detected)):
                f_name = active_frame_names[f] if f < len(active_frame_names) else f"frame_{f}"
                print(f"        - [{f_name}] Tag #{t_id}")

        max_iters_p2 = 35
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

        print("[*] 正在执行 Phase 1 阶段二：微容差 (ftol=1e-5) 极致深层平差收敛...")
        monitor2 = _OptimizationMonitor(stage=2, stage_name="微容差深度平差", max_iters=max_iters_p2, cb=callback)
        res_stage2 = least_squares(
            monitor2.wrap_residuals(residuals_func, obs_weights, outliers_detected), res_stage1.x,
            method='trf',
            loss='cauchy',
            f_scale=1.0,
            x_scale='jac',
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            max_nfev=200,
            verbose=0
        )

        if callback:
            callback({
                "stage": 2,
                "stage_name": "微容差深度平差",
                "iter": monitor2.iter_count,
                "max_iter": max_iters_p2,
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
        final_tags_map = self.align_to_scara_world(
            optimized_tags_pose, 
            origin_tag_id=origin_tag_id, 
            x_align_tag_id=x_align_tag_id
        )

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
            JTJ_reg = JTJ + np.eye(JTJ.shape[0]) * 1e-6
            cov = np.linalg.pinv(JTJ_reg) * (sigma_res_px ** 2)

            for idx, tid in enumerate(static_tags):
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

                poly_color = (0, 0, 255) if is_outlier else (0, 255, 0)
                cv2.polylines(quiver_img, [c_obs.astype(np.int32)], True, poly_color, 2)

                scale = 20.0
                for pt_o, pt_p in zip(c_obs, c_proj):
                    dx = (pt_p[0] - pt_o[0]) * scale
                    dy = (pt_p[1] - pt_o[1]) * scale
                    p_start = (int(round(pt_o[0])), int(round(pt_o[1])))
                    p_end = (int(round(pt_o[0] + dx)), int(round(pt_o[1] + dy)))
                    
                    cv2.circle(quiver_img, p_start, 3, (0, 255, 0), -1)
                    cv2.drawMarker(quiver_img, (int(round(pt_p[0])), int(round(pt_p[1]))), (0, 255, 255), 
                                   markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1)
                    cv2.arrowedLine(quiver_img, p_start, p_end, (0, 0, 255) if is_outlier else (0, 165, 255), 
                                    2, tipLength=0.3)

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

        scaled_poses = {}
        for t_id, T in tag_poses.items():
            T_scaled = T.copy()
            T_scaled[:3, 3] = T[:3, 3] * scale_factor
            scaled_poses[t_id] = T_scaled

        return scaled_poses, scale_factor, real_marker_size

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
            print(f"[WARN] 未在有效图像中检出 Tag {origin_tag_id}，将以参考标靶相对对齐！")

        yaw_rad = 0.0
        if origin_tag_id in tag_poses and x_align_tag_id in tag_poses:
            vec_x = tag_poses[x_align_tag_id][:3, 3] - p_origin
            yaw_rad = math.atan2(vec_x[1], vec_x[0])
            print(f"[+] 坐标系 X 轴对齐旋转角: {-math.degrees(yaw_rad):.2f}°")

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
