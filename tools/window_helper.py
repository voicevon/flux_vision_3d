#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
窗口焦点与置顶管理助手 (Window Focus & Forefront Helper)
解决 Windows 环境下从终端/CLI 启动 OpenCV 窗口时无法自动获取键盘焦点的痛点。

注: 属 GUI 工具层能力, 位于 tools/ (核心库 src/ 不含 GUI 逻辑)。
"""

import sys
import cv2


def force_window_focus(window_name: str, keep_topmost: bool = False):
    """
    强制激活指定的 OpenCV 窗口并夺取键盘焦点。
    :param window_name: cv2.namedWindow / cv2.imshow 的窗口标题
    :param keep_topmost: 是否保持永远置顶 (默认为 False，即仅在激活瞬间置顶夺取焦点后恢复普通层级)
    """
    if not window_name:
        return

    # 1. OpenCV 原生属性脉冲激活
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        if not keep_topmost:
            # 延迟短暂脉冲后恢复非强行置顶，确保窗口既在前台又不死锁屏幕
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass  # GUI 可选功能：窗口置顶失败不影响主流程

    # 2. Windows 平台原生 API 穿透防抢焦点限制 (AttachThreadInput)
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            hwnd = user32.FindWindowW(None, window_name)
            if not hwnd:
                # 尝试通过模糊或类名寻找，OpenCV 默认类名通常为 OpenCV
                return

            # 显示窗口
            user32.ShowWindow(hwnd, 5)  # SW_SHOW
            user32.BringWindowToTop(hwnd)

            # 绕过 Windows Foreground Lockout
            fore_hwnd = user32.GetForegroundWindow()
            if fore_hwnd and fore_hwnd != hwnd:
                fore_thread_id = user32.GetWindowThreadProcessId(fore_hwnd, None)
                cur_thread_id = kernel32.GetCurrentThreadId()
                if fore_thread_id != cur_thread_id:
                    user32.AttachThreadInput(cur_thread_id, fore_thread_id, True)
                    user32.SetForegroundWindow(hwnd)
                    user32.SetFocus(hwnd)
                    user32.AttachThreadInput(cur_thread_id, fore_thread_id, False)
                else:
                    user32.SetForegroundWindow(hwnd)
                    user32.SetFocus(hwnd)
            else:
                user32.SetForegroundWindow(hwnd)
                user32.SetFocus(hwnd)
        except Exception:
            pass  # GUI 可选功能：Win32 焦点穿透失败不影响主流程
