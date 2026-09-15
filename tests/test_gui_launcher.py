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
        self.assertEqual(len(catalog), 9)

        seen_keys = set()
        seen_shortcuts = set()
        valid_categories = {
            "A — 场景总控",
            "B — 标定流水线",
            "C — 感知层",
            "D — 生产执行",
            "E — 系统运维"
        }
        for tool in catalog:
            self.assertIsInstance(tool, ToolCardMeta)
            self.assertTrue(len(tool.title) > 0)
            self.assertTrue(len(tool.summary) > 0)
            self.assertTrue(len(tool.details) > 0)
            self.assertTrue(len(tool.inputs) > 0)
            self.assertTrue(len(tool.outputs) > 0)
            self.assertIn(tool.category, valid_categories)

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
        """测试鼠标卡片网格碰撞检测 (基于五分组布局)"""
        # 第一张卡片 idx=0 (顶部全宽) 覆盖 (15~767, 86~156)
        hit_0 = self.app._hit_test_cards(100, 100)
        self.assertEqual(hit_0, 0)

        # 第二张卡片 idx=1 (B分组左列) 在 (15, 216)
        hit_1 = self.app._hit_test_cards(100, 230)
        self.assertEqual(hit_1, 1)

        # 第三张卡片 idx=2 (B分组右列) 在 (397, 216)
        hit_2 = self.app._hit_test_cards(450, 230)
        self.assertEqual(hit_2, 2)

        # 越界区域 (如右侧说明栏 x=900) 应该返回 -1
        hit_none = self.app._hit_test_cards(900, 300)
        self.assertEqual(hit_none, -1)

    def test_keyboard_navigation(self):
        """测试键盘方向键导航与选择变更 (五分组布局)"""
        self.app.selected_tool_idx = 0
        # 从 row=0 (idx=0) 按下键 (40) -> row=1, col=0 (idx=1)
        self.app._handle_keyboard(40)
        self.assertEqual(self.app.selected_tool_idx, 1)

        # 从 row=1, col=0 按右键 (39) -> row=1, col=1 (idx=2)
        self.app._handle_keyboard(39)
        self.assertEqual(self.app.selected_tool_idx, 2)

        # 从 row=1, col=1 按下键 (40) -> row=2, col=1 (idx=4)
        self.app._handle_keyboard(40)
        self.assertEqual(self.app.selected_tool_idx, 4)

        # 从 row=2, col=1 按左键 (37) -> row=2, col=0 (idx=3)
        self.app._handle_keyboard(37)
        self.assertEqual(self.app.selected_tool_idx, 3)

        # 从 row=2, col=0 按上键 (38) -> row=1, col=0 (idx=1)
        self.app._handle_keyboard(38)
        self.assertEqual(self.app.selected_tool_idx, 1)

        # 从 row=1, col=0 按上键 (38) -> row=0 (idx=0)
        self.app._handle_keyboard(38)
        self.assertEqual(self.app.selected_tool_idx, 0)

    def test_toast_message(self):
        """测试动态 Toast 提示设置"""
        self.app.set_toast("测试提示消息", duration=2.0)
        self.assertEqual(self.app.toast_msg, "测试提示消息")

    def test_dynamic_resize_canvas_rendering(self):
        """测试用户拖拽缩放窗口大小时，物理分辨率自适应与矢量重绘无锯齿"""
        # 模拟窗口拉大到 1600x900
        self.app.canvas_w = 1600
        self.app.canvas_h = 900
        canvas = self.app._render_canvas()
        self.assertEqual(canvas.shape, (900, 1600, 3))

        # 模拟全高清 1920x1080
        self.app.canvas_w = 1920
        self.app.canvas_h = 1080
        canvas_fhd = self.app._render_canvas()
        self.assertEqual(canvas_fhd.shape, (1080, 1920, 3))

    def test_zoom_shortcuts_laptop_compatibility(self):
        """全面测试笔记本电脑各种键位、Ctrl组合与逆向坐标映射"""
        self.app._force_ctrl_pressed = True

        # 1. 放大 (Zoom In): 兼容未按Shift '=' 与按Shift '+'
        self.app.scale_pct = 100
        self.app._handle_keyboard(ord('='))
        self.assertEqual(self.app.scale_pct, 110)

        self.app._handle_keyboard(ord('+'))
        self.assertEqual(self.app.scale_pct, 120)

        self.app._handle_keyboard(187)  # VK_OEM_PLUS
        self.assertEqual(self.app.scale_pct, 130)

        # 2. 缩小 (Zoom Out): 兼容未按Shift '-' 与按Shift '_' (下划线)
        self.app._handle_keyboard(ord('-'))
        self.assertEqual(self.app.scale_pct, 120)

        self.app._handle_keyboard(ord('_'))
        self.assertEqual(self.app.scale_pct, 110)

        self.app._handle_keyboard(189)  # VK_OEM_MINUS
        self.assertEqual(self.app.scale_pct, 100)

        self.app._handle_keyboard(31)   # Ctrl+- 控制字符
        self.assertEqual(self.app.scale_pct, 90)

        # 3. 复位 (Reset): '0' 或 VK_0
        self.app._handle_keyboard(ord('0'))
        self.assertEqual(self.app.scale_pct, 100)

        # 4. 边界保护: 50% ~ 200%
        for _ in range(15):
            self.app._handle_keyboard(ord('-'))
        self.assertEqual(self.app.scale_pct, 50)

        for _ in range(25):
            self.app._handle_keyboard(ord('+'))
        self.assertEqual(self.app.scale_pct, 200)

        # 5. 鼠标与当前物理视口坐标 1:1 原生对齐验证 (真矢量无畸变模式)
        self.app.scale_pct = 150
        self.app.canvas_w, self.app.canvas_h = 1920, 1080
        self.app._on_mouse(0, 960, 540, 0, None)
        self.assertEqual((self.app.mouse_x, self.app.mouse_y), (960, 540))

        # 6. Ctrl + 鼠标滚轮向上/向下平滑缩放
        self.app.scale_pct = 100
        self.app._on_mouse(10, 500, 300, 1, None)   # 向上滚
        self.assertEqual(self.app.scale_pct, 110)
        self.app._on_mouse(10, 500, 300, -1, None)  # 向下滚
        self.assertEqual(self.app.scale_pct, 100)


if __name__ == "__main__":
    unittest.main()
