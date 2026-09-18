#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局基础 GUI 交互控件库 (gui_components)
================================================================================
跨应用通用的标准视觉交互控件：
1. draw_dropdown_button: 统一现代微质感下拉菜单头部按钮 (支持 Hover 微光、展开态高亮、强调色)
2. render_dropdown_popup: 统一置顶悬浮下拉选项浮层 (半透明磨砂遮罩、当前选中态发光、自动计算弹窗坐标)
3. draw_dashboard_button: 统一工业风碳灰卡片按钮 (支持语义侧条、运行态金色发光、禁用态)

所有控件均深度绑定 src.utils.gui_theme.GuiTheme 单源调色板。
"""

from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, get_cached_font, measure_text, put_text


def draw_dropdown_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    is_open: bool,
    mouse_pos: Tuple[int, int] = (-1, -1),
    prefix: str = "",
    theme_color: Optional[Tuple[int, int, int]] = None,
    font_size: int = 13,
) -> bool:
    """绘制统一的现代微质感下拉菜单头部按钮

    Args:
        canvas: 目标画布 (BGR)
        rect: (x1, y1, x2, y2) 坐标范围
        label: 按钮文本
        is_open: 当前是否处于展开状态
        mouse_pos: 当前鼠标坐标 (mx, my)，用于计算 hover 状态
        prefix: 前缀文本 (例如 "场景: ")
        theme_color: 自定义高亮色 (BGR)，默认使用 GuiTheme.BORDER_SEL
        font_size: 字体大小

    Returns:
        bool: 当前鼠标是否悬停在该按钮上
    """
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

    active_col = theme_color if theme_color is not None else GuiTheme.BORDER_SEL

    if is_open:
        bg_col = GuiTheme.CARD_SEL
        border_col = active_col
        text_col = GuiTheme.WHITE
        arrow = "▲"
    elif is_hover:
        bg_col = GuiTheme.CARD_HOVER
        border_col = active_col
        text_col = GuiTheme.BTN_TEXT_HOVER
        arrow = "▼"
    else:
        bg_col = GuiTheme.CARD_BG
        border_col = GuiTheme.BORDER
        text_col = GuiTheme.BTN_TEXT
        arrow = "▼"

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

    display_txt = f"{prefix}{label} {arrow}" if prefix else f"{label} {arrow}"

    try:
        font = get_cached_font(font_size, bold=(is_open or is_hover))
        bbox = font.getbbox(display_txt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x1 + max(6, (x2 - x1 - tw) // 2 - bbox[0])
        ty = y1 + (y2 - y1 - th) // 2 - bbox[1]
        draw_text(canvas, display_txt, (tx, ty), font_size=font_size, color=text_col, bold=(is_open or is_hover))
    except Exception:
        (tw, th), _ = measure_text(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tx = x1 + max(6, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2
        put_text(canvas, display_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)

    return is_hover


def render_dropdown_popup(
    canvas: np.ndarray,
    anchor_rect: Tuple[int, int, int, int],
    options: Sequence[Tuple[str, str]],
    active_key: str,
    btn_prefix: str = "DD_",
    item_h: int = 30,
    min_width: int = 210,
    max_visible: int = 15,
) -> List[Tuple[str, Tuple[int, int, int, int], str]]:
    """在画布上渲染置顶悬浮下拉选项列表浮层

    Args:
        canvas: 目标画布 (BGR)
        anchor_rect: 触发按钮的矩形 (x1, y1, x2, y2)，用于决定弹窗位置
        options: 选项序列 [(key, display_label), ...]
        active_key: 当前激活的选项 key
        btn_prefix: 注册的按钮 ID 前缀，生成的按钮 ID 为 f"{btn_prefix}{idx}"
        item_h: 每项高度 (px)
        min_width: 浮层最小宽度 (px)
        max_visible: 最多展示行数

    Returns:
        List[Tuple[str, Tuple[int, int, int, int], str]]: 生成的交互按钮注册列表 [(btn_id, item_rect, key), ...]
    """
    if not options:
        return []

    rx1, ry1, rx2, ry2 = anchor_rect
    pop_w = max(rx2 - rx1, min_width)
    pop_x1 = rx1
    pop_y1 = ry2 + 2

    ch, cw = canvas.shape[:2]
    if pop_x1 + pop_w > cw - 8:
        pop_x1 = max(8, cw - pop_w - 8)

    disp_opts = options[:max_visible]
    pop_x2 = pop_x1 + pop_w
    pop_y2 = min(ch - 8, pop_y1 + len(disp_opts) * item_h + 6)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (24, 28, 36), -1)
    cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
    cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), GuiTheme.BORDER_SEL, 1)

    registered_buttons = []
    for i, (key, label) in enumerate(disp_opts):
        iy1 = pop_y1 + 3 + i * item_h
        iy2 = iy1 + item_h
        if iy2 > pop_y2 - 2:
            break

        is_active = (key == active_key)
        item_rect = (pop_x1 + 2, iy1, pop_x2 - 2, iy2)

        if is_active:
            cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), GuiTheme.CARD_SEL, -1)
            text_color = GuiTheme.ACCENT
        else:
            text_color = GuiTheme.WHITE

        draw_text(
            canvas,
            label,
            (pop_x1 + 10, iy1 + (item_h - 16) // 2 - 2),
            font_size=14,
            color=text_color,
            bold=is_active,
        )
        btn_id = f"{btn_prefix}{i}"
        registered_buttons.append((btn_id, item_rect, key))

    return registered_buttons


def draw_dashboard_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    mouse_pos: Tuple[int, int] = (-1, -1),
    accent: Optional[Tuple[int, int, int]] = None,
    is_running: bool = False,
    font_size: int = 12,
    disabled: bool = False,
) -> bool:
    """绘制与 Dashboard 同源的工业风碳灰卡片式按钮:
    深色底 + 沉稳边框, 悬停冷青微光, 左缘语义色条, 运行中金色高亮, 禁用态置灰

    Returns:
        bool: 当前鼠标是否悬停在按钮上
    """
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2) and not disabled

    if disabled:
        bg_col = GuiTheme.BTN_DISABLED_BG
        border_col = GuiTheme.BTN_DISABLED_BORDER
        text_col = GuiTheme.TEXT_DISABLED
        border_th = 1
    elif is_running:
        bg_col = GuiTheme.CARD_HOVER
        border_col = GuiTheme.GOLD
        text_col = (250, 225, 140)
        border_th = 2
    elif is_hover:
        bg_col = GuiTheme.CARD_HOVER
        border_col = GuiTheme.BORDER_HOVER
        text_col = GuiTheme.WHITE
        border_th = 2
    else:
        bg_col = GuiTheme.CARD_BG
        border_col = GuiTheme.BORDER
        text_col = GuiTheme.BTN_TEXT
        border_th = 1

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_th)

    if accent is not None and not disabled:
        cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x1 + 4, y2 - 1), accent, -1)

    bbox = get_cached_font(font_size, bold=True).getbbox(label)
    tw, t_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x1 + max(4, ((x2 - x1) - tw) // 2 - bbox[0])
    ty = y1 + ((y2 - y1) - t_h) // 2 - bbox[1]
    draw_text(canvas, label, (tx, ty), font_size=font_size, color=text_col, bold=True)

    return is_hover
