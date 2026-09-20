# -*- coding: utf-8 -*-
"""
单元测试：跨平台原生对话框适配工具 (src/utils/dialog_utils.py)
验证 Windows 原生调用、Linux zenity 分支、兜底机制及业务解耦。
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from src.utils.dialog_utils import prompt_confirm, prompt_input_text


class TestDialogUtils(unittest.TestCase):
    """测试对话框工具"""

    def test_windows_confirm_yes(self):
        """测试 Windows 原生 MessageBoxW 返回确定 (IDYES=6)"""
        with patch("sys.platform", "win32"), \
             patch("ctypes.windll.user32.MessageBoxW", return_value=6):
            res = prompt_confirm("测试标题", "测试内容")
            self.assertTrue(res)

    def test_windows_confirm_no(self):
        """测试 Windows 原生 MessageBoxW 返回取消 (IDNO=7)"""
        with patch("sys.platform", "win32"), \
             patch("ctypes.windll.user32.MessageBoxW", return_value=7):
            res = prompt_confirm("测试标题", "测试内容")
            self.assertFalse(res)

    def test_linux_zenity_confirm(self):
        """测试 Linux 环境优先调用 zenity 确认框"""
        mock_res = MagicMock()
        mock_res.returncode = 0
        with patch("sys.platform", "linux"), \
             patch("shutil.which", return_value="/usr/bin/zenity"), \
             patch("subprocess.run", return_value=mock_res) as mock_run:
            res = prompt_confirm("Linux标题", "Linux内容")
            self.assertTrue(res)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            self.assertIn("zenity", args)
            self.assertIn("--question", args)

    def test_linux_zenity_input_text(self):
        """测试 Linux 环境优先调用 zenity 输入框获取中文内容"""
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "测试工位_01\n"
        with patch("sys.platform", "linux"), \
             patch("shutil.which", return_value="/usr/bin/zenity"), \
             patch("subprocess.run", return_value=mock_res) as mock_run:
            val = prompt_input_text("输入标题", "提示信息", initial="默认值")
            self.assertEqual(val, "测试工位_01")
            args = mock_run.call_args[0][0]
            self.assertIn("zenity", args)
            self.assertIn("--entry", args)

    def test_input_text_cli_fallback(self):
        """测试无图形环境或异常时自动降级到控制台输入"""
        with patch("sys.platform", "linux"), \
             patch("shutil.which", return_value=None), \
             patch("tkinter.Tk", side_effect=Exception("No DISPLAY")), \
             patch("builtins.input", return_value="控制台输入内容"):
            val = prompt_input_text("标题", "提示")
            self.assertEqual(val, "控制台输入内容")


if __name__ == "__main__":
    unittest.main()
