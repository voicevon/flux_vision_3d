#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot 在线跟踪 - 共享视觉样式常量与绘制工具 (主控制器与渲染器共用)"""

import numpy as np

from src.utils.text_rendering import draw_text

# ============================ 视觉样式常量 (BGR, 与 d435_viewer 同源工业深色主题) ============================
COLOR_BG = (15, 17, 21)         # 工具栏 / 占位背景
COLOR_CARD_BG = (22, 26, 33)    # 按钮常态底色
COLOR_CARD_SEL = (28, 44, 58)   # 乒乓开关激活底色
COLOR_BORDER = (38, 46, 58)     # 常态描边
COLOR_BORDER_SEL = (0, 240, 200)  # 激活描边
COLOR_ACCENT = (0, 210, 180)    # 主题强调色
COLOR_TEXT_SUB = (155, 170, 185)
COL_GREEN = (110, 220, 90)      # 实测 / 目标标靶高亮
COL_YELLOW = (90, 200, 245)     # 理论值
COL_CYAN = (235, 205, 80)       # 偏差 / 支撑标靶
COL_BLUE = (255, 170, 0)        # 单帧实测棱柱 (蓝, 与已知Tag叠加的实测色一致)
COL_RED = (85, 85, 245)         # 错误提示
COL_WHITE = (240, 240, 240)
COL_GRAY = (165, 165, 170)
COL_PANEL_BG = (26, 26, 30)
COL_PANEL_EDGE = (95, 95, 105)

TOOLBAR_H = 44  # 顶部工具栏高度


def fmt_point(p, signed=False):
    """三维坐标一行式格式化 (一位小数, 右对齐, 无 X/Y/Z 前缀)"""
    if p is None:
        return "    --      --      --"
    if signed:
        return f"{p[0]:+8.1f} {p[1]:+8.1f} {p[2]:+8.1f}"
    return f"{p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}"


def _tag_local_frame(wc):
    """由标靶世界角点构造局部坐标系 (X右/Y上/Z=面法向朝镜头, 右手系) 与中心点。
    角点顺序: 左上, 右上, 右下, 左下; Z = X×Y 朝外 (与单靶 PnP 位姿同向)。"""
    x = wc[1] - wc[0]
    x = x / (np.linalg.norm(x) + 1e-9)
    y0 = wc[0] - wc[3]
    y0 = y0 / (np.linalg.norm(y0) + 1e-9)
    z = np.cross(x, y0)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    return np.column_stack([x, y, z]), wc.mean(axis=0)


PRISM_HW_MM = 15.0        # 棱柱截面半宽 mm (截面 30x30, 与 Offline Studio 一致)
PRISM_HEIGHT_MM = 75.0    # 棱柱生长高度 mm (与 Offline Studio 一致)
