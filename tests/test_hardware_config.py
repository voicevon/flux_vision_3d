"""
硬件环境配置自动化单元测试 (tests/test_hardware_config.py)
==========================================================
验证选型默认值、下拉选择持久化、画布矢量渲染与下拉浮层交互
"""

import os
import json
import tempfile
import unittest

from tools.hardware_config import HardwareConfigApp


class TestHardwareConfig(unittest.TestCase):

    def _make_app(self, tmpdir):
        cfg = os.path.join(tmpdir, "hardware_env.json")
        app = HardwareConfigApp(config_file=cfg)
        # 固定画布尺寸, 屏蔽本机窗口偏好记忆对渲染断言的干扰
        app.win_mgr.canvas_w, app.win_mgr.canvas_h = 760, 560
        return app, cfg

    def test_defaults(self):
        """无配置文件时应回落默认选型 (RealSense / 1280x720 / SCARA / 未选串口)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            app, _ = self._make_app(tmpdir)
            self.assertEqual(app.config["camera_type"], "realsense")
            self.assertEqual(app.config["camera_resolution"], "1280x720")
            self.assertEqual(app.config["arm_type"], "scara")
            self.assertEqual(app.config["arm_port"], "")

    def test_select_and_persistence(self):
        """下拉选型即改即存, 二次启动自动恢复"""
        with tempfile.TemporaryDirectory() as tmpdir:
            app, cfg = self._make_app(tmpdir)
            app.select("cam", "usb")
            app.select("res", "640x480")
            app.select("arm", "delta")
            app.select("port", "COM7")

            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["camera_type"], "usb")
            self.assertEqual(data["camera_resolution"], "640x480")
            self.assertEqual(data["arm_type"], "delta")
            self.assertEqual(data["arm_port"], "COM7")
            self.assertIn("updated_at", data)

            app2, _ = self._make_app(tmpdir)
            self.assertEqual(app2.config["camera_type"], "usb")
            self.assertEqual(app2.config["camera_resolution"], "640x480")
            self.assertEqual(app2.config["arm_type"], "delta")
            self.assertEqual(app2.config["arm_port"], "COM7")

    def test_render_canvas_vector_reflow(self):
        """真矢量渲染: 画布随窗口物理尺寸 1:1 重绘无畸变"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np
            app, _ = self._make_app(tmpdir)
            canvas = app.render()
            self.assertIsInstance(canvas, np.ndarray)
            self.assertEqual(canvas.shape, (560, 760, 3))
            self.assertTrue(np.any(canvas > 0))

            # 模拟放大窗口 → 布局按比例重排
            app.win_mgr.canvas_w, app.win_mgr.canvas_h = 1140, 840
            canvas2 = app.render()
            self.assertEqual(canvas2.shape, (840, 1140, 3))

    def test_dropdown_toggle_and_option_select(self):
        """下拉浮层开合与选项命中选型 (选中项高亮 + 即改即存)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            app, _ = self._make_app(tmpdir)
            app.render()
            self.assertIsNone(app.active_dropdown)

            # 命中摄像机类型下拉按钮 → 浮层打开
            toggle = next(a for _, a in app._buttons if a[0] == "toggle" and a[1] == "cam")
            app._toggle_dropdown(toggle[1])
            self.assertEqual(app.active_dropdown, "cam")
            app.render()

            # 命中 "usb" 选项 → 选型更新并自动关闭浮层
            opt = next(a for _, a in app._buttons if a[0] == "opt" and a[1] == "cam" and a[2] == "usb")
            app.select(opt[1], opt[2])
            self.assertEqual(app.config["camera_type"], "usb")

            # 再次点击同一下拉按钮 → 浮层关闭
            app._toggle_dropdown("cam")
            self.assertIsNone(app.active_dropdown)

    def test_option_labels(self):
        """选项标签映射与串口缺省显示"""
        with tempfile.TemporaryDirectory() as tmpdir:
            app, _ = self._make_app(tmpdir)
            self.assertEqual(app._option_label("cam", "realsense"), "RealSense D435")
            self.assertEqual(app._option_label("arm", "scara"), "SCARA (串联)")
            self.assertEqual(app._option_label("port", ""), "未选择")
            self.assertEqual(app._option_label("port", "COM3"), "COM3")


if __name__ == "__main__":
    unittest.main()
