#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tag Calibration Verifier 单元测试
==================================
验证：
  1. 初始化与 config/tags_map.yaml 地图加载；
  2. 留一法盲测 (Leave-One-Out) 数学投影与位姿推算正确性；
  3. 鼠标点选 GUI 按钮与标靶轮廓命中的状态切换；
  4. 3D 正四棱柱立体轴 (render_tag_3d_axes) 渲染无崩溃。
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.tag_calibration_verifier import TagCalibrationVerifier


class TestTagCalibrationVerifier(unittest.TestCase):
    def setUp(self):
        self.map_path = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
        self.verifier = TagCalibrationVerifier(map_path=self.map_path, mock_mode=True)

    def test_initialization_and_map_loading(self):
        """测试地图加载与基础数据结构"""
        self.assertIsNotNone(self.verifier.tags_map)
        self.assertIn("tags", self.verifier.tags_map)
        self.assertGreater(len(self.verifier.mapped_tag_ids), 0)
        self.assertIn(26, self.verifier.mapped_tag_ids)

    def test_blind_projection_math(self):
        """测试留一盲测隔空推算数学逻辑"""
        # 假设已知世界系下的标靶 Tag 26
        T_w_26 = self.verifier._get_tag_world_transform(26)
        self.assertIsNotNone(T_w_26)

        # 构造一个虚拟的相机位姿 T_c_w (相机在世界系下的绝对位姿)
        R_c_w = np.eye(3)
        t_c_w = np.array([0.0, 0.0, 600.0]) # 距离原点 600mm
        T_c_w = np.eye(4)
        T_c_w[:3, :3] = R_c_w
        T_c_w[:3, 3] = t_c_w

        # 隔空推算 Tag 26 在相机系下的相对位姿
        T_c_26 = T_c_w @ T_w_26
        R_c_26 = T_c_26[:3, :3]
        t_c_26 = T_c_26[:3, 3].reshape((3, 1))
        rvec_26, _ = cv2.Rodrigues(R_c_26)

        # 投影角点
        s = 25.0
        corners_3d = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

        pred_pts, _ = cv2.projectPoints(corners_3d, rvec_26, t_c_26, 
                                        self.verifier.camera_matrix, self.verifier.dist_coeffs)
        pred_pts = pred_pts.reshape((4, 2))
        self.assertEqual(pred_pts.shape, (4, 2))
        # 确认投影在合理像素范围内
        self.assertTrue(np.all(pred_pts[:, 0] > 0))
        self.assertTrue(np.all(pred_pts[:, 1] > 0))

    def test_gui_mouse_click_interaction(self):
        """测试 GUI 按钮点击切换盲测目标逻辑"""
        # 模拟设置 GUI 按钮热区
        self.verifier.gui_buttons = [
            (None, (10, 60, 100, 90), "ALL"),
            (26, (110, 60, 180, 90), "Tag #26"),
            (24, (190, 60, 260, 90), "Tag #24")
        ]

        # 1. 初始状态为 None
        self.assertIsNone(self.verifier.blind_target_tag_id)

        # 2. 点击 Tag 26 按钮 (x=130, y=75)
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 130, 75, 0, None)
        self.assertEqual(self.verifier.blind_target_tag_id, 26)

        # 3. 再次点击 Tag 26 按钮 -> 应该取消盲测恢复 None
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 130, 75, 0, None)
        self.assertIsNone(self.verifier.blind_target_tag_id)

        # 4. 点击 Tag 24 按钮 (x=210, y=75)
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 210, 75, 0, None)
        self.assertEqual(self.verifier.blind_target_tag_id, 24)

        # 5. 点击 ALL 按钮 (x=30, y=75) -> 恢复 None
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 30, 75, 0, None)
        self.assertIsNone(self.verifier.blind_target_tag_id)

    def test_render_3d_axes(self):
        """测试 3D 正四棱柱实心轴渲染"""
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)

        # 盲测高光渲染
        self.verifier.render_tag_3d_axes(dummy_img, rvec, tvec, tag_id=26, is_blind_projection=True)
        # 普通渲染
        self.verifier.render_tag_3d_axes(dummy_img, rvec, tvec, tag_id=0, is_blind_projection=False)
        self.assertGreater(np.count_nonzero(dummy_img), 0)


if __name__ == "__main__":
    unittest.main()
