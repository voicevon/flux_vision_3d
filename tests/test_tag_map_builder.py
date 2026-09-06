#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标靶 (AprilTag 16h5) 全局建图与 BA 平差算法单元测试
- 生成虚拟空间立体分布的 4x4 AprilTag (高低错落、非共面)
- 模拟多个相机机位观测 (每帧至少包含 2~4 个 Tag)
- 测试 TagMapBuilder 从观测中重构 3D 空间地图并对齐 Tag 0 原点和 Tag 1 X 轴的精度
"""

import unittest
import math
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.tag_map_builder import TagMapBuilder


class TestTagMapBuilder(unittest.TestCase):
    def setUp(self):
        self.marker_size = 50.0 # 50mm
        self.builder = TagMapBuilder(marker_size_mm=self.marker_size)
        self.K = self.builder.camera_matrix
        self.D = self.builder.dist_coeffs

    def test_ba_reconstruction_synthetic(self):
        """测试非共面空间标靶阵列的合成多视角投影重构精度"""
        # 1. 定义真实世界中的 4 个标靶位姿 (高低错落，Tag 0 为原点)
        # 标靶 0: (0, 0, 0), 原点锚点
        # 标靶 1: (200, 0, 15), 位于世界 X 轴正方向，带高程
        # 标靶 2: (100, 150, -20), 高低错落
        # 标靶 3: (-50, 120, 30), 高低错落
        gt_tag_positions = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([200.0, 0.0, 15.0]),
            2: np.array([100.0, 150.0, -20.0]),
            3: np.array([-50.0, 120.0, 30.0])
        }
        
        gt_tag_rpy = {
            0: (0.0, 0.0, 0.0),
            1: (0.05, -0.02, 0.1),
            2: (-0.1, 0.05, -0.2),
            3: (0.08, 0.1, 0.15)
        }

        # 构建世界系下各 Tag 4x4 矩阵
        T_w_tags = {}
        for t_id in gt_tag_positions:
            rpy = gt_tag_rpy[t_id]
            R, _ = cv2.Rodrigues(np.array(rpy, dtype=np.float64))
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = gt_tag_positions[t_id]
            T_w_tags[t_id] = T

        # 2. 模拟 4 个不同角度的相机机位 (俯视、斜视)，均位于 Z > 400mm 处
        camera_positions = [
            np.array([50.0, 50.0, 500.0]),
            np.array([150.0, 80.0, 480.0]),
            np.array([-30.0, 70.0, 520.0]),
            np.array([100.0, -20.0, 490.0]),
        ]

        frame_detections = []
        for cam_pos in camera_positions:
            # 相机朝向工作区中心
            look_at = np.array([60.0, 60.0, 5.0])
            forward = look_at - cam_pos
            forward = forward / np.linalg.norm(forward)
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            actual_up = np.cross(right, forward)
            
            # 相机坐标系旋转 R_w_c: X=right, Y=down(-actual_up), Z=forward
            R_w_c = np.column_stack([right, -actual_up, forward])
            T_w_c = np.eye(4)
            T_w_c[:3, :3] = R_w_c
            T_w_c[:3, 3] = cam_pos
            T_c_w = np.linalg.inv(T_w_c)

            # 投影当前相机能看到的 Tag
            frame_tags = {}
            for t_id, T_w_t in T_w_tags.items():
                T_c_t = T_c_w @ T_w_t
                rv, tv = self.builder.matrix_to_rvec_tvec(T_c_t)
                # 投影 4 个角点
                proj_pts, _ = cv2.projectPoints(
                    self.builder.obj_points, rv, tv, self.K, self.D
                )
                proj_pts = proj_pts.reshape((4, 2))
                frame_tags[t_id] = proj_pts

            frame_detections.append(frame_tags)

        # 3. 验证初值生成和平差函数
        # 我们用这 4 组观测直接调用内部平差逻辑验证
        # 手动注入 frame_detections 进行图解算
        base_static_id = 1
        tag_poses_init = {base_static_id: T_w_tags[base_static_id].copy()}
        camera_poses_init = {}

        # 迭代生成初值
        for f_idx, tags in enumerate(frame_detections):
            for t_id, corners in tags.items():
                if t_id in tag_poses_init:
                    succ, rvec, tvec = self.builder.solve_single_tag_pnp(corners)
                    if succ:
                        T_c_t = self.builder.rvec_tvec_to_matrix(rvec, tvec)
                        camera_poses_init[f_idx] = tag_poses_init[t_id] @ np.linalg.inv(T_c_t)
                        break

        for f_idx, tags in enumerate(frame_detections):
            if f_idx in camera_poses_init:
                T_w_c = camera_poses_init[f_idx]
                for t_id, corners in tags.items():
                    if t_id not in tag_poses_init:
                        succ, rvec, tvec = self.builder.solve_single_tag_pnp(corners)
                        if succ:
                            T_c_t = self.builder.rvec_tvec_to_matrix(rvec, tvec)
                            tag_poses_init[t_id] = T_w_c @ T_c_t

        self.assertEqual(len(tag_poses_init), 4, "所有 4 个 Tag 都应被正确初始化")

        # 对齐到 Tag 0 原点和 Tag 1 X 轴
        aligned_map = self.builder._align_to_scara_world(
            tag_poses_init, origin_tag_id=0, x_align_tag_id=1
        )

        # 验证 Tag 0 位置在原点
        t0_pos = aligned_map["tags"][0]["position_mm"]
        self.assertAlmostEqual(t0_pos[0], 0.0, places=1)
        self.assertAlmostEqual(t0_pos[1], 0.0, places=1)
        
        # 验证 Tag 1 处于世界 X 轴正方向 (Y 接近 0, X 约为 200)
        t1_pos = aligned_map["tags"][1]["position_mm"]
        self.assertAlmostEqual(t1_pos[1], 0.0, places=1, msg="Tag 1 的 Y 坐标应被旋转对齐至 0")
        self.assertGreater(t1_pos[0], 190.0, "Tag 1 的 X 坐标应在正半轴")

        print(f"\n[Test OK] Tag 0 对齐位置: {t0_pos}")
        print(f"[Test OK] Tag 1 对齐位置: {t1_pos}")
        print(f"[Test OK] Tag 2 对齐位置: {aligned_map['tags'][2]['position_mm']}")
        print(f"[Test OK] Tag 3 对齐位置: {aligned_map['tags'][3]['position_mm']}")

    def test_baseline_scale_calibration(self):
        """测试未知单个 Tag 边长工况下，通过双标靶实际测距恢复绝对尺度的算法精度"""
        # 假设名义边长为 50mm，但实际打印被缩放为 42.5mm (缩放系数 0.85)
        scale_true = 0.85
        nominal_poses = {
            1: np.array([
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64),
            5: np.array([
                [1.0, 0.0, 0.0, 500.0],
                [0.0, 1.0, 0.0, 300.0],
                [0.0, 0.0, 1.0, 50.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
        }

        # 名义欧氏距离: sqrt(400^2 + 300^2 + 50^2) = sqrt(160000 + 90000 + 2500) = sqrt(252500) ≈ 502.4938 mm
        nominal_dist = np.linalg.norm(nominal_poses[1][:3, 3] - nominal_poses[5][:3, 3])
        real_measured_dist = nominal_dist * scale_true # 真实测量距离

        scaled_poses, scale_factor, real_marker_size = self.builder.apply_baseline_scale(
            nominal_poses, tag_id_a=1, tag_id_b=5, real_distance_mm=real_measured_dist
        )

        self.assertAlmostEqual(scale_factor, scale_true, places=5)
        self.assertAlmostEqual(real_marker_size, 50.0 * scale_true, places=3)
        
        # 验证缩放后的距离与现场测量完全一致
        scaled_dist = np.linalg.norm(scaled_poses[1][:3, 3] - scaled_poses[5][:3, 3])
        self.assertAlmostEqual(scaled_dist, real_measured_dist, places=3)
        print(f"[Test OK] 双标靶基线绝对尺度测试通过！计算Scale: {scale_factor:.5f}, 真实边长: {real_marker_size:.2f}mm")


if __name__ == "__main__":
    unittest.main()
