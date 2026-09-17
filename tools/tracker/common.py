#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot 在线跟踪 - 共享视觉样式常量与绘制工具 (主控制器与渲染器共用)"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

_FONT_CACHE = {}


def draw_text(img, text, pos, font_size=16, color=COL_WHITE, bold=False):
    """在 OpenCV BGR 图像上绘制中文/西文 (局部轻量 Patch 贴图)"""
    if not text:
        return
    if any(ord(c) > 127 for c in text):
        key = (font_size, bold)
        if key not in _FONT_CACHE:
            try:
                font_path = "C:/Windows/Fonts/msyh.ttc"
                if not os.path.exists(font_path):
                    font_path = "C:/Windows/Fonts/simhei.ttf"
                _FONT_CACHE[key] = ImageFont.truetype(font_path, font_size)
            except Exception:
                _FONT_CACHE[key] = ImageFont.load_default()
        font = _FONT_CACHE[key]
        x, y = pos
        ih, iw = img.shape[:2]
        if x >= iw or y >= ih or x < 0 or y < 0:
            return
        text_w = int(len(text) * font_size * 1.15) + 12
        text_h = int(font_size * 1.5) + 6
        x2 = min(iw, x + text_w)
        y2 = min(ih, y + text_h)
        if x2 <= x or y2 <= y:
            return
        patch_bgr = img[y:y2, x:x2]
        pil_img = Image.fromarray(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((0, 0), text, font=font, fill=(color[2], color[1], color[0]))
        img[y:y2, x:x2] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        scale = font_size / 28.0
        cv2.putText(img, text, (pos[0], pos[1] + int(font_size * 0.85)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)


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
_PRISM_PTS = np.array([
    # 底面 4 点 (Z=0) / 顶面 4 点 (Z=生长高) / 顶面中心
    [-PRISM_HW_MM, -PRISM_HW_MM, 0.0], [PRISM_HW_MM, -PRISM_HW_MM, 0.0],
    [PRISM_HW_MM, PRISM_HW_MM, 0.0], [-PRISM_HW_MM, PRISM_HW_MM, 0.0],
    [-PRISM_HW_MM, -PRISM_HW_MM, PRISM_HEIGHT_MM], [PRISM_HW_MM, -PRISM_HW_MM, PRISM_HEIGHT_MM],
    [PRISM_HW_MM, PRISM_HW_MM, PRISM_HEIGHT_MM], [-PRISM_HW_MM, PRISM_HW_MM, PRISM_HEIGHT_MM],
    [0.0, 0.0, PRISM_HEIGHT_MM],
], dtype=np.float64)
