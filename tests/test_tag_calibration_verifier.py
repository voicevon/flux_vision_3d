#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 在线 AR 综合验证系统单元测试 (test_tag_calibration_verifier.py)
===================================================================
覆盖核心特性：
  1. 系统初始化与默认运行模式 (实时动态 live_mode == True)；
  2. 乒乓开关流转 (LIVE ⇋ STATIC LOCKED)；
  3. 批处理采样采足 30 帧 -> 集中时域去噪 -> 一次性解算并绝对锁死位姿 (Fixate & Lock)；
  4. 全 GUI 工具栏按钮点击与盲测事件分发。
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.calibration.tag_calibration_verifier import TagCalibrationVerifier


class TestTagCalibrationVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = TagCalibrationVerifier(mock_mode=True)

    def test_initialization(self):
        """测试系统初始化与默认运行状态"""
        self.assertTrue(self.verifier.live_mode, "默认启动模式应为【⚡ 实时动态 LIVE】")
        self.assertIsNotNone(self.verifier.camera_matrix, "相机内参矩阵必须成功加载")
        self.assertGreater(len(self.verifier.mapped_tag_ids), 0, "应成功加载已知标靶列表")
        self.assertEqual(self.verifier.batch_target_frames, 30, "默认批次采样深度应为 30 帧")
        self.assertIsNone(self.verifier.locked_pose, "初始状态下位姿应尚未锁定")

    def test_ping_pong_toggle_mode(self):
        """测试核心乒乓开关切换逻辑"""
        # 初始为实时动态
        self.assertTrue(self.verifier.live_mode)

        # 第一次切换 -> 静态锁定模式 (未锁定时应自动触发批次采样)
        self.verifier.toggle_mode()
        self.assertFalse(self.verifier.live_mode, "切换后应为【🎯 静态滤波锁定模式】")
        self.assertTrue(self.verifier.is_collecting_batch, "初次进入应触发批次静止采样")

        # 第二次切换 -> 回到实时动态
        self.verifier.toggle_mode()
        self.assertTrue(self.verifier.live_mode, "再次切换应回到【⚡ 实时动态模式】")
        self.assertIn("实时动态", self.verifier.status_toast)

    def test_batch_collection_and_lock_pose(self):
        """测试静止采足 30 帧后，一次性集中去噪解算并彻底锁定机制"""
        self.verifier.live_mode = False
        self.verifier.start_batch_collection()
        self.assertTrue(self.verifier.is_collecting_batch)

        # 构造基准世界标靶 (至少需要 2 枚已知标靶以支持 PnP 超定解算)
        test_tids = self.verifier.mapped_tag_ids[:2]
        self.assertGreaterEqual(len(test_tids), 2, "地图中至少应有 2 枚标靶")

        # 模拟 30 帧带散粒高斯噪声的角点采样
        np.random.seed(42)
        true_corners = {}
        for tid in test_tids:
            # 投影无噪声真值角点 (使用仿真位姿)
            w_c = self.verifier._get_tag_world_corners(tid)
            # 假定相机处于原点附近
            rvec_dummy = np.array([0.05, -0.02, 0.0], dtype=np.float64)
            tvec_dummy = np.array([50.0, 30.0, 800.0], dtype=np.float64)
            proj_pts, _ = cv2.projectPoints(w_c, rvec_dummy, tvec_dummy, self.verifier.camera_matrix, self.verifier.dist_coeffs)
            true_corners[tid] = proj_pts.reshape((4, 2))

        # 填充 30 帧带噪声数据
        for _ in range(30):
            for tid in test_tids:
                noise = np.random.normal(0, 0.4, size=(4, 2))
                noisy_c = true_corners[tid] + noise
                if tid not in self.verifier.batch_corner_buffer:
                    self.verifier.batch_corner_buffer[tid] = []
                self.verifier.batch_corner_buffer[tid].append(noisy_c)

        # 触发集中去噪解算与位姿锁定
        self.verifier._compute_and_lock_pose()

        # 核心断言：
        # 1. 采样状态结束 (停止采样与滤波)
        self.assertFalse(self.verifier.is_collecting_batch, "解算后必须彻底停止采样与滤波")
        # 2. locked_pose 成功建立并存入成果
        self.assertIsNotNone(self.verifier.locked_pose, "位姿必须成功锁定")
        self.assertIn('pos_w', self.verifier.locked_pose)
        self.assertIn('rvec', self.verifier.locked_pose)
        self.assertIn('tvec', self.verifier.locked_pose)
        self.assertLess(self.verifier.locked_pose['rmse'], 1.0, "去噪后解算的重投影 RMSE 应极小 (< 1.0 px)")
        print(f"\n[TEST] 30 帧静止集中去噪锁定成功！RMSE: {self.verifier.locked_pose['rmse']:.3f} px | 相机位置: {self.verifier.locked_pose['pos_w']}")

    def test_gui_mouse_interactions(self):
        """测试底部工具栏按钮点击事件响应"""
        self.verifier.gui_buttons = [
            ("TOGGLE_MODE", (10, 100, 200, 140), "TOGGLE"),
            ("CYCLE_BATCH", (210, 100, 300, 140), "BATCH"),
            ("RESAMPLE_LOCK", (310, 100, 450, 140), "RESAMPLE"),
            (18, (10, 10, 80, 40), "Tag 18"),
            (None, (90, 10, 150, 40), "ALL")
        ]

        # 1. 点击乒乓开关
        self.assertTrue(self.verifier.live_mode)
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 120, 0, None)
        self.assertFalse(self.verifier.live_mode)

        # 2. 点击批次帧数切换
        old_target = self.verifier.batch_target_frames
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 250, 120, 0, None)
        self.assertNotEqual(self.verifier.batch_target_frames, old_target)

        # 3. 点击重新采样锁定
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 350, 120, 0, None)
        self.assertTrue(self.verifier.is_collecting_batch)

        # 4. 点击顶栏 Tag 18 盲测
        self.assertIsNone(self.verifier.blind_target_tag_id)
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 30, 20, 0, None)
        self.assertEqual(self.verifier.blind_target_tag_id, 18)

        # 5. 点击 ALL 恢复全量
        self.verifier._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 20, 0, None)
        self.assertIsNone(self.verifier.blind_target_tag_id)

    def test_full_render_loop_robustness(self):
        """测试完整 GUI 帧渲染、视口自适应、HUD折叠、工具栏各按钮绘制无异常"""
        # 1. 测试空白灰色网格帧下的全套 UI 渲染
        frame = self.verifier.get_frame(0)
        self.assertEqual(frame.shape, (1080, 1920, 3))

        canvas = np.zeros((self.verifier.win_h, self.verifier.win_w, 3), dtype=np.uint8)
        self.verifier.viewport.render_viewport(canvas, frame)

        # 2. 模拟触发 Toast 与 HUD 展开
        self.verifier.set_toast("单元测试浮层通知提示")
        self.verifier.show_hud_terminal = True
        self.verifier.hud_terminal_lines = ["[INFO] 测试终端行 1", "[WARN] 测试终端行 2"]
        self.verifier.render_hud_terminal(canvas)

        # 3. 模拟工具栏与顶栏各按钮渲染 (包含鼠标坐标悬停)
        w_img, h_img = self.verifier.win_w, self.verifier.win_h
        top_bar_h = self.verifier.viewport.top_bar_h
        bottom_bar_h = self.verifier.viewport.bottom_bar_h
        btn_y_top = h_img - bottom_bar_h + 8
        btn_y_bot = h_img - 8
        mx, my = 150, btn_y_top + 10

        from src.utils.viewport_manager import draw_styled_button, draw_segmented_toggle
        draw_styled_button(canvas, (10, btn_y_top, 100, btn_y_bot), "测试按钮",
                           mouse_pos=(mx, my), btn_type="primary")
        draw_segmented_toggle(canvas, (110, btn_y_top, 250, btn_y_bot),
                              [("live", "实时"), ("locked", "锁定")],
                              mouse_pos=(mx, my), active_key="live")

        self.assertGreater(canvas.shape[0], 0)
        self.assertGreater(canvas.shape[1], 0)


if __name__ == "__main__":
    unittest.main()

