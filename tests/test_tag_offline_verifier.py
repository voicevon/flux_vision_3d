#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线标定精度体检系统单元测试 (test_tag_offline_verifier.py)
======================================================
覆盖核心特性：
  1. 引擎初始化、空间地图加载与数据源自动探测；
  2. 超定 PnP 求解与物理正深度前置校验；
  3. Leave-One-Out (LOO) 盲测重投影与空间毫米误差解算；
  4. 偏差统计聚合、系统性偏差识别与综合评审定级。
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.calibration.tag_offline_verifier import TagOfflineVerifier


class TestTagOfflineVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = TagOfflineVerifier()

    def test_initialization(self):
        """测试离线体检引擎初始化与依赖加载"""
        self.assertIsNotNone(self.verifier.tags_map, "必须成功加载 tags_map.yaml")
        self.assertGreater(len(self.verifier.mapped_tag_ids), 0, "地图中标靶数量应大于 0")
        self.assertIsNotNone(self.verifier.camera_matrix, "相机内参必须成功加载")
        self.assertEqual(self.verifier.actual_source, "manifest", "存在清单时应优先选用 manifest 数据源")

    def test_solve_camera_pose_positive_depth(self):
        """测试 PnP 位姿解算与正深度约束"""
        if len(self.verifier.mapped_tag_ids) < 3:
            self.skipTest("地图中标靶数量不足 3 个，跳过测试")

        # 构造虚拟相机位姿: 位于 (150, 0, 1000) 俯视工作台
        rvec_true = np.array([0.1, 0.05, 0.0], dtype=np.float64)
        tvec_true = np.array([50.0, -100.0, 950.0], dtype=np.float64)

        tag_pairs = []
        for tid in self.verifier.mapped_tag_ids[:4]:
            wc = self.verifier._get_tag_world_corners(tid)
            self.assertIsNotNone(wc, f"标靶 {tid} 世界角点不应为空")
            proj, _ = cv2.projectPoints(wc, rvec_true, tvec_true, self.verifier.camera_matrix, self.verifier.dist_coeffs)
            tag_pairs.append((tid, proj.reshape(4, 2)))

        # 解算位姿
        pose = self.verifier._solve_camera_pose(tag_pairs)
        self.assertIsNotNone(pose, "PnP 解算应成功")
        self.assertGreater(pose["tvec"][2, 0], 0.0, "求解位姿必须满足物理正深度 (z > 0)")
        self.assertLess(pose["rmse"], 0.1, "无噪点合成投影下 RMSE 应接近 0")

    def test_compute_loo_error(self):
        """测试单个盲测目标的像元误差与空间毫米偏差计算"""
        target_tid = self.verifier.mapped_tag_ids[0]
        wc = self.verifier._get_tag_world_corners(target_tid)

        rvec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec = np.array([0.0, 0.0, 1000.0], dtype=np.float64)

        proj, _ = cv2.projectPoints(wc, rvec, tvec, self.verifier.camera_matrix, self.verifier.dist_coeffs)
        obs_corners = proj.reshape(4, 2)

        # 完美对齐场景
        loo_perfect = self.verifier._compute_loo_error(rvec, tvec, target_tid, obs_corners)
        self.assertIsNotNone(loo_perfect)
        self.assertAlmostEqual(loo_perfect["err_px"], 0.0, places=3)
        self.assertAlmostEqual(loo_perfect["err_mm"], 0.0, places=3)

        # 注入 2.0 像素偏差
        obs_noisy = obs_corners + 2.0
        loo_noisy = self.verifier._compute_loo_error(rvec, tvec, target_tid, obs_noisy)
        self.assertGreater(loo_noisy["err_px"], 1.5)
        self.assertGreater(loo_noisy["err_mm"], 1.0)

    def test_aggregate_statistics_grading(self):
        """测试综合评级判定与异常标记"""
        dummy_results = [
            {"image": "view_0001.png", "blind_tag_id": 18, "err_px": 1.2, "err_mm": 0.8},
            {"image": "view_0001.png", "blind_tag_id": 19, "err_px": 1.4, "err_mm": 0.9},
            {"image": "view_0002.png", "blind_tag_id": 18, "err_px": 1.1, "err_mm": 0.7},
            {"image": "view_0002.png", "blind_tag_id": 19, "err_px": 1.3, "err_mm": 0.8},
        ]
        stats = self.verifier._aggregate_statistics(dummy_results)
        self.assertEqual(stats["grade"], "PASS", "中位误差 <= 1.5px 且 <= 1.0mm 应评定为 PASS")
        self.assertEqual(len(stats["flagged_tags"]), 0, "不应有偏差嫌疑标靶")
        self.assertEqual(len(stats["flagged_frames"]), 0, "不应有建议回审帧")


if __name__ == "__main__":
    unittest.main()
