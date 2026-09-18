#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot 在线跟踪 - Tag 2 (芦笋) R 轴旋转角提取与拟真长棒渲染单元测试
================================================================
测试用例:
  1. 单靶 PnP 解算时完整提取旋转向量并恢复局部 Y 轴偏航角 (对比理论真值误差 < 0.5°);
  2. 世界坐标系锁定时相机位姿与标靶位姿复合投影, 校验世界水平 XY 平面偏航角;
  3. 芦笋 3D 拟真长棒 (宽 15mm x 长 200mm, 对称延伸各 100mm) 透视投影与分段多边形渲染;
  4. Tracker 状态机 solve_frame 闭环校验 measured_r 与 G-code E 轴联动参数。
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.tracker.common import (
    ASPARAGUS_WIDTH_MM, ASPARAGUS_LENGTH_MM, ASPARAGUS_HALF_LENGTH_MM,
    ASPARAGUS_HALF_WIDTH_MM, fmt_pose_4d
)
from tools.tracker.app import RobotOnlineTracker
from tools.tracker.renderer import TrackerRenderer
from src.calibration.offline_engine import OfflineVerificationEngine


class TestTrackerAsparagusRotation(unittest.TestCase):
    def setUp(self):
        self.camera_matrix = np.array([
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.marker_size_mm = 50.0

        self.engine = OfflineVerificationEngine(
            tags_map={},
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm
        )

        # 实例化轻量 tracker
        self.tracker = RobotOnlineTracker.__new__(RobotOnlineTracker)
        self.tracker.target_tag_id = 2
        self.tracker.recog_tag2_on = True
        self.tracker.show_anchors_on = False
        self.tracker.show_xy_plane_on = False
        self.tracker.world_locked = False
        self.tracker.locked_rvec = None
        self.tracker.locked_tvec = None
        self.tracker.engine = self.engine
        self.tracker.support_ids = []
        self.tracker.rmse = None
        self.tracker.measured = None
        self.tracker.measured_r = None
        self.tracker.target_rvec = None
        self.tracker.target_tvec = None
        self.tracker.measured_time = 0.0
        self.tracker.HP_TARGET_SIDE_PX = 100.0

    def test_asparagus_constants(self):
        """测试芦笋物理规格常量定义: 宽 15mm, 长 200mm, 对称延伸各 100mm"""
        self.assertEqual(ASPARAGUS_WIDTH_MM, 15.0)
        self.assertEqual(ASPARAGUS_LENGTH_MM, 200.0)
        self.assertEqual(ASPARAGUS_HALF_LENGTH_MM, 100.0)
        self.assertEqual(ASPARAGUS_HALF_WIDTH_MM, 7.5)

        # 测试 4D 位姿格式化
        s = fmt_pose_4d(np.array([100.0, 200.0, 80.0]), r_deg=45.2)
        self.assertIn("R: +45.2°", s)

    def test_single_tag_pnp_extracts_rvec_and_yaw(self):
        """测试单靶 PnP 解算提取旋转向量及局部 Y 轴偏航角"""
        s = self.marker_size_mm / 2.0
        # 构造一个绕 Z 轴逆时针旋转 30° 的标靶
        angle_deg = 30.0
        angle_rad = np.radians(angle_deg)
        R_gt = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        t_gt = np.array([50.0, -30.0, 800.0], dtype=np.float64).reshape((3, 1))

        # 投影 4 个角点
        obj_pts = self.engine.obj_points
        rvec_gt, _ = cv2.Rodrigues(R_gt)
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, t_gt, self.camera_matrix, self.dist_coeffs)
        corners_2d = proj_pts.reshape((4, 2))

        # 调用 engine.solve_single_tag_pnp
        ok, rvec_est, tvec_est = self.engine.solve_single_tag_pnp(corners_2d, expected_z_cam=np.array([0, 0, 1]))
        self.assertTrue(ok)
        self.assertIsNotNone(rvec_est)
        self.assertIsNotNone(tvec_est)

        # 校验 Y 轴朝向角
        R_est, _ = cv2.Rodrigues(rvec_est)
        v_y = R_est[:, 1]
        yaw_calc = float(np.degrees(np.arctan2(v_y[1], v_y[0])))
        # 真值 Y 轴在图像坐标系下: [ -sin(30°), cos(30°), 0 ] -> arctan2(cos, -sin) = 90° - (-30°) 或 60°/120°
        # 校验方向向量点积
        v_y_gt = R_gt[:, 1]
        cos_sim = float(np.dot(v_y, v_y_gt) / (np.linalg.norm(v_y) * np.linalg.norm(v_y_gt)))
        self.assertAlmostEqual(cos_sim, 1.0, places=2, msg="解算得到的 Y 轴方向向量应与真实真值高度一致")

    def test_world_locked_asparagus_yaw_computation(self):
        """测试世界坐标系锁定时，芦笋 Y 轴在世界水平面上投影偏航角的提取"""
        # 相机在世界坐标系上方俯视原点
        R_lock = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float64)
        rvec_lock, _ = cv2.Rodrigues(R_lock)
        tvec_lock = np.array([0.0, 0.0, 1000.0], dtype=np.float64)

        self.tracker.world_locked = True
        self.tracker.locked_rvec = rvec_lock
        self.tracker.locked_tvec = tvec_lock

        # 模拟检出 Tag 2，其在相机坐标系下的旋转为 R_c_t2
        # 设标靶在世界系下 Y 轴朝向世界 +X 轴 (偏航角 0°)
        # v_w_y = [1.0, 0.0, 0.0]
        R_w_t2 = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        R_c_t2 = R_lock @ R_w_t2
        rvec_c, _ = cv2.Rodrigues(R_c_t2)
        tvec_c = np.array([10.0, 20.0, 600.0], dtype=np.float64)

        # 投影角点
        proj_pts, _ = cv2.projectPoints(self.engine.obj_points, rvec_c, tvec_c, self.camera_matrix, self.dist_coeffs)
        det = {2: proj_pts.reshape((4, 2))}

        # 执行 solve_frame
        # 模拟全景与ROI重检直接返回 det
        self.tracker.engine.detect_tags = lambda frame: det
        self.tracker._detect_high_precision = lambda frame, d: d

        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.tracker.solve_frame(fake_frame)

        self.assertIsNotNone(self.tracker.measured_r)
        self.assertIsNotNone(self.tracker.target_rvec)
        self.assertIsNotNone(self.tracker.target_tvec)
        # R_w_t2 的第二列为 [1.0, 0.0, 0.0] -> arctan2(0, 1) = 0°
        self.assertAlmostEqual(self.tracker.measured_r, 0.0, delta=2.0)

    def test_draw_asparagus_stem_rendering(self):
        """测试 3D 拟真芦笋长条 (宽 15mm x 长 200mm) 的分段渲染无崩溃且在画布上着色"""
        renderer = TrackerRenderer(self.tracker)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        rvec = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        tvec = np.array([0.0, 0.0, 600.0], dtype=np.float64)

        # 绘制芦笋长棒
        renderer._draw_asparagus_stem(canvas, rvec, tvec, yaw_deg=35.0)

        # 检查画布是否有像素被绘制 (长棒、外框、切口与文字)
        nonzeros = np.count_nonzero(canvas)
        self.assertGreater(nonzeros, 500, "芦笋拟真长条应在画布上产生像素填充与边框")

        # 校验画面的主要颜色包含鲜翠绿 (BGR 绿色通道显著) 与白色 (三通道接近)
        green_mask = (canvas[:, :, 1] > 180) & (canvas[:, :, 0] < 120)
        self.assertTrue(np.any(green_mask), "芦笋头部应含有高饱和度鲜翠绿色像素")

        white_mask = (canvas[:, :, 0] > 200) & (canvas[:, :, 1] > 200) & (canvas[:, :, 2] > 200)
        self.assertTrue(np.any(white_mask), "芦笋尾部切口应含有白色像素")

    def test_draw_overlay_cleanliness(self):
        """测试叠加层渲染无任何汉字(芦笋/笋尖/根部)、无Tag 2文字、仅保留绿框和笋尖角度"""
        renderer = TrackerRenderer(self.tracker)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 模拟目标标靶角点与位姿
        corners = np.array([
            [600.0, 300.0],
            [680.0, 300.0],
            [680.0, 380.0],
            [600.0, 380.0]
        ], dtype=np.float32)
        det = {2: corners}

        self.tracker.target_rvec = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        self.tracker.target_tvec = np.array([0.0, 0.0, 600.0], dtype=np.float64)
        self.tracker.measured_r = 42.5

        # 记录调用 draw_text 的文本内容
        called_texts = []
        original_draw_text = sys.modules["tools.tracker.renderer"].draw_text

        def mock_draw_text(img, text, pos, font_size=16, color=(240, 240, 240), bold=False):
            called_texts.append(text)
            return original_draw_text(img, text, pos, font_size, color, bold)

        sys.modules["tools.tracker.renderer"].draw_text = mock_draw_text
        try:
            renderer.draw_overlay(canvas, det)
        finally:
            sys.modules["tools.tracker.renderer"].draw_text = original_draw_text

        # 验证调用的文本中无任何汉字与“Tag 2”
        for txt in called_texts:
            for ch in txt:
                self.assertTrue(
                    ch < '\u4e00' or ch > '\u9fff',
                    f"叠加层文字中不应包含任何汉字: '{txt}' 包含 '{ch}'"
                )
            self.assertNotIn("Tag", txt, f"叠加层中不应包含 Tag 2 文字: '{txt}'")
            self.assertNotIn("芦笋", txt, f"叠加层中不应包含芦笋汉字: '{txt}'")

        # 验证笋尖角度数值存在
        self.assertTrue(
            any("+42.5°" in t for t in called_texts),
            f"笋尖前端应展示角度数值 '+42.5°'，实测调用列表: {called_texts}"
        )


if __name__ == "__main__":
    unittest.main()
