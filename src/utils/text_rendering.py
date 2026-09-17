#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用文字渲染工具: TrueType 中文字体缓存 + OpenCV BGR 图像上的 PIL 高质量抗锯齿文本贴图。

下沉自 tools/gui_launcher (消除 src→tools 逆向依赖):
src 库层 (报告可视化) 与全部 GUI 工具统一从本模块导入, 依赖方向保持 tools→src 单向。
"""

import os
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 字体缓存
_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}


def get_cached_font(font_size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont:
    """获取缓存的 TrueType 中文字体"""
    key = (font_size, bold)
    if key not in _FONT_CACHE:
        font_paths = [
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simheittc.ttc" if bold else "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], font_size: int = 16,
              color: Tuple[int, int, int] = (240, 240, 240), bold: bool = False):
    """在 OpenCV BGR 图像上绘制高质量抗锯齿矢量文本 (支持中文)"""
    if not text:
        return
    x, y = pos
    if x >= img.shape[1] or y >= img.shape[0]:
        return

    font = get_cached_font(font_size, bold)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    patch_w = tw + 20
    patch_h = th + 14

    rx2 = min(img.shape[1], x + patch_w)
    ry2 = min(img.shape[0], y + patch_h)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if rx2 <= x or ry2 <= y:
        return

    sub_bgr = img[y:ry2, x:rx2]
    sub_rgb = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(sub_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((0, 0), text, font=font, fill=(color[2], color[1], color[0]))
    res_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[y:ry2, x:rx2] = res_bgr
