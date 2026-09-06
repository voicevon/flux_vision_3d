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
import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional


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
        
        # 现代 OpenCV 4.7+ ArucoDetector 接口
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # 相机内参与畸变
        if camera_matrix is None:
            # 默认 RealSense D435 标称内参 (RGB 1280x720)
            self.camera_matrix = np.array([
                [615.0, 0.0, 320.0],
                [0.0, 615.0, 240.0],
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
        在单帧图像中检测所有 Tag
        :return: {tag_id: corners_4x2}
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        corners, ids, _ = self.detector.detectMarkers(gray)
        results = {}
        if ids is not None and len(ids) > 0:
            for idx, tag_id in enumerate(ids.flatten()):
                results[int(tag_id)] = corners[idx].reshape((4, 2))
        return results

    def solve_single_tag_pnp(self, corners: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        对单标靶执行 PnP 获得其在相机系下的位姿 (rvec, tvec)
        """
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return success, rvec, tvec

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

    def build_map_from_images(self, 
                             image_paths: List[str],
                             origin_tag_id: int = 0,
                             x_align_tag_id: int = 1,
                             baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict:
        """
        从一组多视角图像构建标靶全局地图并进行 BA 全局平差优化
        :param image_paths: 图像文件路径列表
        :param origin_tag_id: SCARA 原点锚定标靶 ID (默认 0)
        :param x_align_tag_id: 世界 X 轴对齐基准标靶 ID (默认 1)
        :param baseline_pair: (tag_a, tag_b, real_dist_mm) 用于绝对物理尺度锁定的标靶对
        :return: 优化后的标靶地图字典
        """
        print(f"[*] 开始处理 {len(image_paths)} 张多视角标定图像...")
        
        # 1. 检测每张图像中的标靶
        # frame_detections: [ {tag_id: corners_4x2} ]
        frame_detections = []
        valid_frames = []
        all_detected_tags = set()

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            tags = self.detect_tags(img)
            # 过滤：每张图像必须至少观测到 2 个 Tag 才能构成相对约束
            if len(tags) >= 2:
                frame_detections.append(tags)
                valid_frames.append(path)
                all_detected_tags.update(tags.keys())
            else:
                print(f"[WARN] 图像 {os.path.basename(path)} 仅检测到 {len(tags)} 个 Tag (需要 >= 2)，已跳过")

        if len(frame_detections) < 2:
            raise ValueError(f"有效图像数量不足 ({len(frame_detections)})，无法构建多视角约束！")

        print(f"[+] 提取到 {len(valid_frames)} 张有效多视角图像，累计发现 {len(all_detected_tags)} 个不同 Tag: {sorted(list(all_detected_tags))}")

        # 2. 构建共视连通图 (Co-visibility Graph)
        adj_list = {t: set() for t in all_detected_tags}
        for tags in frame_detections:
            t_ids = list(tags.keys())
            for i in range(len(t_ids)):
                for j in range(i + 1, len(t_ids)):
                    adj_list[t_ids[i]].add(t_ids[j])
                    adj_list[t_ids[j]].add(t_ids[i])

        # 检查连通性 (以 x_align_tag_id 或首个静止 Tag 为基准进行 BFS)
        base_static_id = x_align_tag_id if x_align_tag_id in all_detected_tags else min(all_detected_tags)
        visited = set()
        queue = [base_static_id]
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                queue.extend(adj_list[curr] - visited)

        unconnected = all_detected_tags - visited
        if unconnected:
            print(f"[WARN] 警告：以下标靶未与主连通图连通: {unconnected}，将无法求解绝对位姿！")

        # 3. 生成初值 (以 base_static_id 为临时参考系 T_world_to_tag)
        # 维护静止 Tag 位姿字典: {tag_id: T_w_tag}
        tag_poses_init = {base_static_id: np.eye(4, dtype=np.float64)}
        camera_poses_init = {} # {frame_idx: T_w_cam}

        # 迭代传递位姿初值
        changed = True
        while changed:
            changed = False
            for f_idx, tags in enumerate(frame_detections):
                # 检查当前帧是否能定出相机初值
                if f_idx not in camera_poses_init:
                    for t_id, corners in tags.items():
                        if t_id in tag_poses_init:
                            # 通过已定位的 Tag 解算相机初值
                            succ, rvec, tvec = self.solve_single_tag_pnp(corners)
                            if succ:
                                T_c_t = self.rvec_tvec_to_matrix(rvec, tvec)
                                # T_w_t = T_w_c * T_c_t  =>  T_w_c = T_w_t * T_c_t^(-1)
                                T_w_c = tag_poses_init[t_id] @ np.linalg.inv(T_c_t)
                                camera_poses_init[f_idx] = T_w_c
                                changed = True
                                break

                # 若相机位姿已知，利用它为其他未知 Tag 赋予初值
                if f_idx in camera_poses_init:
                    T_w_c = camera_poses_init[f_idx]
                    for t_id, corners in tags.items():
                        if t_id not in tag_poses_init:
                            succ, rvec, tvec = self.solve_single_tag_pnp(corners)
                            if succ:
                                T_c_t = self.rvec_tvec_to_matrix(rvec, tvec)
                                T_w_t = T_w_c @ T_c_t
                                tag_poses_init[t_id] = T_w_t
                                changed = True

        print(f"[+] 初值生成完毕：成功初始化 {len(tag_poses_init)} 个 Tag，{len(camera_poses_init)} 个相机机位")

        # 4. 全局 Bundle Adjustment (光束平差优化)
        # 优化变量：
        # - 所有除 base_static_id 外的 Tag 位姿 (6DoF: rx, ry, rz, tx, ty, tz)
        # - 所有有效相机机位的位姿 (6DoF)
        # 注意：Tag 0 在不同帧中如果是旋转的，其三维中心点位置仍是刚性的
        static_tags_to_opt = [t for t in tag_poses_init.keys() if t != base_static_id]
        active_frames = [f for f in camera_poses_init.keys()]

        tag_param_idx = {t: i for i, t in enumerate(static_tags_to_opt)}
        frame_param_idx = {f: i for i, f in enumerate(active_frames)}

        # 打包初值向量
        x0 = []
        for t in static_tags_to_opt:
            rv, tv = self.matrix_to_rvec_tvec(tag_poses_init[t])
            x0.extend(rv.flatten().tolist())
            x0.extend(tv.flatten().tolist())

        for f in active_frames:
            rv, tv = self.matrix_to_rvec_tvec(camera_poses_init[f])
            x0.extend(rv.flatten().tolist())
            x0.extend(tv.flatten().tolist())

        x0 = np.array(x0, dtype=np.float64)

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

        def residuals_func(x):
            tags_pose, cams_pose = unpack_params(x)
            residuals = []
            for f in active_frames:
                T_w_c = cams_pose[f]
                T_c_w = np.linalg.inv(T_w_c)
                tags = frame_detections[f]
                for t_id, corners_img in tags.items():
                    if t_id in tags_pose:
                        T_w_t = tags_pose[t_id]
                        T_c_t = T_c_w @ T_w_t
                        rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                        # 投影 4 个角点
                        proj_pts, _ = cv2.projectPoints(
                            self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs
                        )
                        proj_pts = proj_pts.reshape((4, 2))
                        # 计算重投影残差 (dx, dy)
                        diff = (proj_pts - corners_img).flatten()
                        residuals.extend(diff)
            return np.array(residuals, dtype=np.float64)

        print("[*] 开始进行非线性最小二乘 (Bundle Adjustment) 全局平差优化...")
        res = least_squares(residuals_func, x0, method='trf', loss='huber', f_scale=1.0, verbose=1)
        
        # 解算优化后位姿
        optimized_tags_pose, optimized_cams_pose = unpack_params(res.x)
        final_residuals = residuals_func(res.x)
        rmse_px = float(np.sqrt(np.mean(final_residuals ** 2)))
        print(f"[OK] BA 优化完成！重投影均方根误差 RMSE: {rmse_px:.3f} 像素")

        # 5. 双标靶中心基线测距尺度修正 (Metric Baseline Gauge)
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

        # 6. 坐标系对齐闭环：将世界坐标原点绑定到 Tag 0 中心，X 轴对齐到 Tag 1
        final_tags_map = self._align_to_scara_world(
            optimized_tags_pose, 
            origin_tag_id=origin_tag_id, 
            x_align_tag_id=x_align_tag_id
        )

        final_tags_map["rmse_reprojection_px"] = rmse_px
        final_tags_map["marker_size_mm"] = round(float(self.marker_size_mm), 3)
        final_tags_map["tag_family"] = "DICT_APRILTAG_16h5"
        final_tags_map["calibrated_images_count"] = len(active_frames)
        if baseline_info:
            final_tags_map["baseline_gauge"] = baseline_info
        return final_tags_map

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


def main():
    parser = argparse.ArgumentParser(description="AprilTag 16h5 多标靶离线建图与平差工具")
    parser.add_argument("--image_dir", type=str, default="data/tag_calibration_images", help="多视角标定图片目录")
    parser.add_argument("--marker_size", type=float, default=50.0, help="标靶黑白边框名义边长 (mm)")
    parser.add_argument("--origin_id", type=int, default=0, help="SCARA 原点锚定标靶 ID")
    parser.add_argument("--x_axis_id", type=int, default=1, help="世界 X 轴对齐基准标靶 ID")
    parser.add_argument("--baseline_pair", nargs=3, type=float, metavar=('TAG_A', 'TAG_B', 'DIST_MM'),
                        help="双标靶基线尺度校准参数: TAG_A TAG_B 真实距离(mm), 例如: --baseline_pair 1 5 620.5")
    parser.add_argument("--output", type=str, default="config/tags_map.yaml", help="导出的图谱文件路径")
    args = parser.parse_args()

    # 寻找图像文件
    patterns = [os.path.join(args.image_dir, ext) for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat))

    if not image_paths:
        print(f"[!] 目录 '{args.image_dir}' 下未找到任何标定图像！")
        print(f"[*] 提示：请使用相机采集约 20 张覆盖多标靶的图像放入该目录后重试。")
        sys.exit(1)

    baseline_pair = None
    if args.baseline_pair is not None:
        baseline_pair = (int(args.baseline_pair[0]), int(args.baseline_pair[1]), float(args.baseline_pair[2]))
    else:
        # 尝试交互式提示输入 (若在交互终端中)
        if sys.stdin.isatty():
            print(f"\n[?] 是否需要输入双标靶实际物理中心距离以锁定绝对尺度？")
            print(f"    (例如现场用卷尺测量 Tag 1 和 Tag 5 的中心距离为 620.5mm)")
            try:
                user_inp = input("    请输入 [ID_A ID_B 真实距离mm] (直接回车跳过使用名义尺寸): ").strip()
                if user_inp:
                    parts = user_inp.split()
                    if len(parts) == 3:
                        baseline_pair = (int(parts[0]), int(parts[1]), float(parts[2]))
            except (EOFError, KeyboardInterrupt):
                pass

    builder = TagMapBuilder(marker_size_mm=args.marker_size)
    tags_map = builder.build_map_from_images(
        image_paths=image_paths,
        origin_tag_id=args.origin_id,
        x_align_tag_id=args.x_axis_id,
        baseline_pair=baseline_pair
    )
    builder.save_map(tags_map, args.output)


if __name__ == "__main__":
    main()
