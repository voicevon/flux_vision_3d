#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局基础 GUI 交互控件库 (gui_components)
================================================================================
跨应用通用的标准视觉交互控件：
1. draw_dropdown_button: 统一现代微质感下拉菜单头部按钮 (支持 Hover 微光、展开态高亮、强调色)
2. render_dropdown_popup: 统一置顶悬浮下拉选项浮层 (半透明磨砂遮罩、当前选中态发光、自动计算弹窗坐标)
3. draw_dashboard_button: 统一工业风碳灰卡片按钮 (支持语义侧条、运行态金色发光、禁用态)
4. render_floating_tooltip: 统一高对比度科技悬浮气泡浮层 (智能边界贴靠避让与翻转、语义前缀高亮、磨砂半透明融合、圆角质感)
5. draw_rounded_rectangle: 统一抗锯齿圆角矩形绘制函数 (支持实体圆角填充与平滑发光描边)

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


_CACHED_LOGO: Optional[np.ndarray] = None
_CACHED_LOGO_SIZE: int = -1


def get_cached_logo(size: int = 32) -> Optional[np.ndarray]:
    """读取并缓存标准 LOGO 图像 (assets/logo.png)"""
    global _CACHED_LOGO, _CACHED_LOGO_SIZE
    if _CACHED_LOGO is not None and _CACHED_LOGO_SIZE == size:
        return _CACHED_LOGO
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logo_path = os.path.join(base_dir, "assets", "logo.png")
    if os.path.exists(logo_path):
        img = cv2.imread(logo_path, cv2.IMREAD_COLOR)
        if img is not None:
            _CACHED_LOGO = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            _CACHED_LOGO_SIZE = size
            return _CACHED_LOGO
    return None


def draw_app_header(
    canvas: np.ndarray,
    x: int = 12,
    y: int = 6,
    sub_title: str = "",
    icon_size: int = 32,
) -> int:
    """在 GUI 顶部统一绘制科技感品牌 LOGO、主标题与子系统模块名称

    Args:
        canvas: 目标画布 (BGR)
        x: 左侧起始 X
        y: 顶部起始 Y
        sub_title: 子模块名 (例如 "AprilTag 管理器", "标定场景中心", "空间位姿追踪器")
        icon_size: 图标尺寸 (默认 32x32)

    Returns:
        int: 标题组件右侧边缘的 X 坐标 (方便后续横向排布工具栏按钮)
    """
    logo = get_cached_logo(icon_size)
    curr_x = x
    if logo is not None:
        h, w = logo.shape[:2]
        ch, cw = canvas.shape[:2]
        if y + h <= ch and curr_x + w <= cw:
            canvas[y:y + h, curr_x:curr_x + w] = logo
            cv2.rectangle(canvas, (curr_x, y), (curr_x + w, y + h), (0, 216, 180), 1)
        curr_x += w + 10
    else:
        cv2.circle(canvas, (curr_x + icon_size // 2, y + icon_size // 2), icon_size // 2 - 2, (0, 216, 180), 2)
        cv2.circle(canvas, (curr_x + icon_size // 2, y + icon_size // 2), 3, (0, 255, 255), -1)
        curr_x += icon_size + 10

    # 绘制主标题 "FluxVision 3D"
    draw_text(canvas, "FluxVision 3D", (curr_x, y - 2), font_size=15, color=(0, 240, 220), bold=True)

    # 绘制副标题 (芦笋上料自动化 | <sub_title>)
    sub_text = f"芦笋上料自动化 | {sub_title}" if sub_title else "芦笋上料自动化"
    draw_text(canvas, sub_text, (curr_x, y + 17), font_size=11, color=(140, 160, 180))

    try:
        font1 = get_cached_font(15, bold=True)
        w1 = font1.getbbox("FluxVision 3D")[2]
        font2 = get_cached_font(11, bold=False)
        w2 = font2.getbbox(sub_text)[2]
        text_w = max(w1, w2)
    except Exception:
        text_w = 160

    return curr_x + text_w + 18


def draw_rounded_rectangle(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    radius: int = 8,
    thickness: int = 1,
    fill: bool = False,
) -> None:
    """在画布上绘制抗锯齿圆角矩形 (支持实体填充或平滑发光描边)

    Args:
        canvas: 目标画布图像 (BGR)
        rect: (x, y, w, h) 矩形范围
        color: 绘制颜色 (B, G, R)
        radius: 圆角半径 (像素，默认 8)
        thickness: 描边线宽 (fill=True 时忽略)
        fill: 是否填充内部
    """
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    r = max(0, min(radius, w // 2, h // 2))

    if r <= 0:
        if fill:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        return

    if fill:
        # 十字中心填充
        cv2.rectangle(canvas, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(canvas, (x, y + r), (x + w, y + h - r), color, -1)
        # 四角实心抗锯齿圆
        cv2.circle(canvas, (x + r, y + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x + w - r, y + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x + r, y + h - r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x + w - r, y + h - r), r, color, -1, lineType=cv2.LINE_AA)
    else:
        # 四条直边
        cv2.line(canvas, (x + r, y), (x + w - r, y), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x + r, y + h), (x + w - r, y + h), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x, y + r), (x, y + h - r), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x + w, y + r), (x + w, y + h - r), color, thickness, lineType=cv2.LINE_AA)
        # 四角抗锯齿圆弧
        cv2.ellipse(canvas, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)



def render_floating_tooltip(
    canvas: np.ndarray,
    title: str,
    lines: Sequence[Any],
    anchor_pos: Tuple[int, int],
    max_width: int = 560,
    theme_color: Optional[Tuple[int, int, int]] = None,
    font_size: int = 11,
    line_height: int = 20,
    corner_radius: int = 8,
) -> Tuple[int, int, int, int]:
    """绘制高对比度、科技质感的悬浮气泡框 (自动贴靠锚点并计算视口边界避让、翻转与圆角质感)

    Args:
        canvas: 目标画布 (BGR)
        title: 气泡标题文本
        lines: 提示正文列表 (支持纯文本 str，或结构化元组 (tag, text) / (tag, color, text))
        anchor_pos: (ax, ay) 锚点坐标 (通常为当前鼠标或控件基准点)
        max_width: 气泡框最大宽度 (像素，默认 560)
        theme_color: 强调色/发光边框色 (BGR)，默认使用 (0, 240, 200) 科技青
        font_size: 正文字体大小 (默认 11)
        line_height: 行高步进 (默认 20)
        corner_radius: 气泡圆角半径 (默认 8)

    Returns:
        Tuple[int, int, int, int]: 实际渲染的气泡矩形 (tx, ty, tw, th)
    """
    ch, cw = canvas.shape[:2]
    ax, ay = anchor_pos

    # 动态测量最宽行以自适应气泡宽度，避免空白或文字溢出
    measured_max_w = 0
    for item in lines:
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                tag, _, text = item
                text_for_w = f"[{tag}]  {text}"
            elif len(item) >= 2:
                tag, text = item[0], item[1]
                text_for_w = f"[{tag}]  {text}"
            else:
                text_for_w = str(item[0]) if item else ""
        else:
            text_for_w = str(item)
        if text_for_w:
            try:
                (tw_line, _), _ = measure_text(text_for_w, font_size=font_size)
                if tw_line > measured_max_w:
                    measured_max_w = tw_line
            except Exception:
                measured_max_w = max(measured_max_w, len(text_for_w) * 11)

    try:
        (title_w, _), _ = measure_text(f"★ {title}", font_size=13, bold=True)
        measured_max_w = max(measured_max_w, title_w)
    except Exception:
        measured_max_w = max(measured_max_w, len(title) * 13)

    content_w = measured_max_w + 34
    tw = max(280, min(max_width, content_w))
    tw = min(tw, cw - 32)
    th = 38 + len(lines) * line_height + 12

    # 横向边界避让
    tx = max(16, min(cw - tw - 16, ax + 15))

    # 纵向计算与底部溢出自动向上翻转
    ty = ay + 15
    if ty + th > ch - 16:
        ty = max(16, ay - th - 10)
    ty = max(16, min(ch - th - 16, ty))

    accent_col = theme_color if theme_color is not None else (0, 240, 200)

    # 区域半透明暗色磨砂背景融合 (带有平滑圆角 Mask 遮罩)
    sub = canvas[ty:ty + th, tx:tx + tw]
    bg = np.full_like(sub, (16, 20, 28))
    blended = cv2.addWeighted(bg, 0.94, sub, 0.06, 0)
    card_mask = np.zeros((th, tw), dtype=np.uint8)
    draw_rounded_rectangle(card_mask, (0, 0, tw, th), 255, radius=corner_radius, fill=True)
    np.copyto(sub, blended, where=(card_mask[:, :, None] > 0))
    canvas[ty:ty + th, tx:tx + tw] = sub

    # 标题栏底色 (带顶部两处圆角)
    title_h = 30
    title_sub = canvas[ty:ty + title_h, tx:tx + tw]
    title_bg = np.full_like(title_sub, (24, 34, 44))
    t_mask = np.zeros((title_h, tw), dtype=np.uint8)
    r = min(corner_radius, title_h, tw // 2)
    cv2.rectangle(t_mask, (0, r), (tw, title_h), 255, -1)
    cv2.rectangle(t_mask, (r, 0), (tw - r, r), 255, -1)
    cv2.circle(t_mask, (r, r), r, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(t_mask, (tw - r, r), r, 255, -1, lineType=cv2.LINE_AA)
    np.copyto(title_sub, title_bg, where=(t_mask[:, :, None] > 0))
    canvas[ty:ty + title_h, tx:tx + tw] = title_sub

    # 标题栏底部分割线
    cv2.line(canvas, (tx + 1, ty + title_h), (tx + tw - 1, ty + title_h), (0, 200, 160), 1)

    # 外层发光圆角细边框与内衬双边
    draw_rounded_rectangle(canvas, (tx, ty, tw, th), accent_col, radius=corner_radius, thickness=1)
    inner_r = max(2, corner_radius - 1)
    draw_rounded_rectangle(canvas, (tx + 1, ty + 1, tw - 2, th - 2), (30, 60, 55), radius=inner_r, thickness=1)

    # 标题文字
    draw_text(canvas, f"★ {title}", (tx + 12, ty + 6), font_size=13, color=(0, 255, 220), bold=True)

    # 正文内容逐行语义着色
    cur_y = ty + 38
    for item in lines:
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                tag, val_col, text = item
            elif len(item) >= 2:
                tag, text = item[0], item[1]
                val_col = (220, 235, 245)
            else:
                tag, val_col, text = "", (220, 235, 245), str(item[0]) if item else ""

            tag_label = f"[{tag}] " if tag else ""
            if tag_label:
                (lw, _), _ = measure_text(tag_label, font_size=font_size, bold=True)
                draw_text(canvas, tag_label, (tx + 14, cur_y), font_size=font_size, color=(0, 240, 210), bold=True)
                draw_text(canvas, str(text), (tx + 14 + lw + 4, cur_y), font_size=font_size, color=val_col, bold=False)
            else:
                draw_text(canvas, str(text), (tx + 14, cur_y), font_size=font_size, color=val_col, bold=False)
            cur_y += line_height
            continue

        line = str(item)
        if not line:
            cur_y += 6
            continue
        col = (220, 235, 245)
        bold = False
        if line.startswith("【") or line.startswith("["):
            col = (0, 240, 210)
            bold = True
        elif line.startswith("•") or line.startswith("-"):
            col = (180, 210, 230)
        elif "★" in line or "注意" in line:
            col = (255, 205, 80)
        elif "!" in line or "警告" in line or "错误" in line:
            col = (80, 90, 255)
        draw_text(canvas, line, (tx + 14, cur_y), font_size=font_size, color=col, bold=bold)
        cur_y += line_height

    return (tx, ty, tw, th)



