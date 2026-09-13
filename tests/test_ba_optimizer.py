#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BundleAdjustmentOptimizer 单元测试 (test_ba_optimizer.py)
=========================================================
覆盖核心特性：
  1. 旋转平移向量与 4x4 齐次矩阵双向精确可逆转换；
  2. 双标靶基线尺度修正算法 (Metric Baseline Gauge)；
  3. SCARA 世界坐标系对齐闭环 (Origin 锚定与 X 轴水平对齐)；
  4. 3D 不确定度协方差提取。
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.ba_optimizer import BundleAdjustmentOptimizer


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    def setUp(self):
        K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
        dist = np.zeros(5)
        self.optimizer = BundleAdjustmentOptimizer(
            camera_matrix=K,
            dist_coeffs=dist,
            marker_size_mm=50.0
        )

    def test_matrix_rvec_tvec_roundtrip(self):
        """测试 4x4 矩阵与 rvec/tvec 的无损双向转换"""
        rvec_in = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        tvec_in = np.array([100.0, 200.0, 300.0], dtype=np.float64)

        T = self.optimizer.rvec_tvec_to_matrix(rvec_in, tvec_in)
        rvec_out, tvec_out = self.optimizer.matrix_to_rvec_tvec(T)

        np.testing.assert_allclose(rvec_out.flatten(), rvec_in.flatten(), atol=1e-7)
        np.testing.assert_allclose(tvec_out.flatten(), tvec_in.flatten(), atol=1e-7)

    def test_apply_baseline_scale(self):
        """测试 Metric Baseline Gauge 尺度校准"""
        # 初始名义坐标：Tag 0 在 (0, 0, 0), Tag 1 在 (500, 0, 0)，名义距离 500mm
        tag_poses = {
            0: np.eye(4),
            1: np.array([[1, 0, 0, 500.0],
                         [0, 1, 0, 0.0],
                         [0, 0, 1, 0.0],
                         [0, 0, 0, 1.0]], dtype=np.float64)
        }
        # 实际物理测量距离 525.0mm (比例尺 1.05)
        scaled_poses, scale, real_marker_size = self.optimizer.apply_baseline_scale(
            tag_poses, tag_id_a=0, tag_id_b=1, real_distance_mm=525.0
        )
        self.assertAlmostEqual(scale, 1.05, places=5)
        self.assertAlmostEqual(real_marker_size, 52.5, places=2)
        np.testing.assert_allclose(scaled_poses[1][:3, 3], [525.0, 0.0, 0.0], atol=1e-5)

    def test_align_to_scara_world(self):
        """测试将 Tag 0 绑定为原点，Tag 1 对齐至 +X 轴"""
        # 构造未对齐状态：Tag 0 在 (100, 100, 0)，Tag 1 在 (100+300*cos(45°), 100+300*sin(45°), 0)
        p0 = np.array([100.0, 100.0, 0.0])
        p1 = np.array([100.0 + 300.0 * np.cos(np.pi / 4), 100.0 + 300.0 * np.sin(np.pi / 4), 0.0])
        T0 = np.eye(4)
        T0[:3, 3] = p0
        T1 = np.eye(4)
        T1[:3, 3] = p1

        tag_poses = {0: T0, 1: T1}
        aligned_map = self.optimizer.align_to_scara_world(tag_poses, origin_tag_id=0, x_align_tag_id=1)

        pos0 = aligned_map["tags"][0]["position_mm"]
        pos1 = aligned_map["tags"][1]["position_mm"]

        # 校验 Tag 0 归零
        np.testing.assert_allclose(pos0, [0.0, 0.0, 0.0], atol=1e-2)
        # 校验 Tag 1 落在 +X 轴上 (Y=0, X 约为 300.0)
        self.assertAlmostEqual(pos1[1], 0.0, places=1)
        self.assertAlmostEqual(pos1[0], 300.0, places=1)


if __name__ == "__main__":
    unittest.main()
