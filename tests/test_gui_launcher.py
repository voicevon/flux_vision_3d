"""
GUI 控制中心自动化单元测试 (tests/test_gui_launcher.py)
=====================================================
验证 3D Vision Suite GUI 控制中心的卡片目录、状态采集、碰撞测试与渲染稳定性
"""

import unittest
import numpy as np
import cv2

from tools.gui_launcher import GuiLauncherApp, build_tools_catalog, ToolCardMeta


class TestGuiLauncher(unittest.TestCase):

    def setUp(self):
        self.app = GuiLauncherApp()

    def test_tools_catalog_integrity(self):
        """测试工具目录数据结构完整性与快捷键不重复"""
        catalog = build_tools_catalog()
        self.assertGreaterEqual(len(catalog), 14)

        seen_keys = set()
        seen_shortcuts = set()
        for tool in catalog:
            self.assertIsInstance(tool, ToolCardMeta)
            self.assertTrue(len(tool.title) > 0)
            self.assertTrue(len(tool.summary) > 0)
            self.assertTrue(len(tool.details) > 0)
            self.assertTrue(len(tool.inputs) > 0)
            self.assertTrue(len(tool.outputs) > 0)
            self.assertIn(tool.category, ["生产与工况", "视觉标定", "测试运维"])

            # 键值唯一性
            self.assertNotIn(tool.key_id, seen_keys, f"重复的 key_id: {tool.key_id}")
            seen_keys.add(tool.key_id)

            # 快捷键唯一性
            self.assertNotIn(tool.shortcut, seen_shortcuts, f"重复的 shortcut: {tool.shortcut}")
            seen_shortcuts.add(tool.shortcut)

    def test_app_initialization_and_status(self):
        """测试 Launcher 应用初始化与系统状态探针"""
        self.assertEqual(self.app.canvas_w, 1280)
        self.assertEqual(self.app.canvas_h, 720)
        self.assertTrue(self.app._running)
        self.assertIsNotNone(self.app.system_status)

    def test_canvas_rendering(self):
        """测试 1280x720 双缓冲画布离线渲染稳定性"""
        canvas = self.app._render_canvas()
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape, (720, 1280, 3))

    def test_hit_test_cards(self):
        """测试鼠标卡片网格碰撞检测"""
        # 第一张卡片大约在 (15, 66)
        hit_0 = self.app._hit_test_cards(100, 100)
        self.assertEqual(hit_0, 0)

        # 第二张卡片 (右列第一张) 大约在 (15 + 370 + 12 = 397, 66)
        hit_1 = self.app._hit_test_cards(450, 100)
        self.assertEqual(hit_1, 1)

        # 越界区域 (如右侧说明栏 x=900) 应该返回 -1
        hit_none = self.app._hit_test_cards(900, 300)
        self.assertEqual(hit_none, -1)

    def test_keyboard_navigation(self):
        """测试键盘方向键导航与选择变更"""
        self.app.selected_tool_idx = 0
        # 按右键 (39)
        self.app._handle_keyboard(39)
        self.assertEqual(self.app.selected_tool_idx, 1)

        # 按下键 (40)
        self.app._handle_keyboard(40)
        self.assertEqual(self.app.selected_tool_idx, 3)

        # 按左键 (37)
        self.app._handle_keyboard(37)
        self.assertEqual(self.app.selected_tool_idx, 2)

        # 按上键 (38)
        self.app._handle_keyboard(38)
        self.assertEqual(self.app.selected_tool_idx, 0)

    def test_toast_message(self):
        """测试动态 Toast 提示设置"""
        self.app.set_toast("测试提示消息", duration=2.0)
        self.assertEqual(self.app.toast_msg, "测试提示消息")


if __name__ == "__main__":
    unittest.main()
