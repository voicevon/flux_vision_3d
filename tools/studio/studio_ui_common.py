"""
AprilTag 离线标定工作站 - UI 渲染共享常量与模块级绘制函数 (studio_ui_common)
================================================================================
存放 StudioUIRenderer 核心调度类与各分区 Mixin (帧列表 / 中栏视口 / 右栏检视)
共同引用的下拉菜单选项常量与模块级按钮绘制函数：
1. 下拉菜单选项常量 (视图模式 / 筛选 / 排序 / BA 理论值与实测值显示)
2. draw_dropdown_button: 现代扁平化下拉菜单头部按钮
3. draw_dashboard_button: Dashboard (gui_launcher) 同源碳灰卡片式按钮
本模块不 import 任何 tools.studio 模块, 供各渲染分区单向引用, 避免循环依赖。
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import draw_text, get_cached_font, measure_text, put_text


# 下拉菜单选项定义
VIEW_MODE_OPTIONS = [
    ("3d", "3D 双四棱柱对比"),
    ("2d", "2D 识别框与残差矢量"),
    ("mix", "混合透视模式")
]

FILTER_MODE_OPTIONS = [
    ("all", "全部帧"),
    ("warning", "高残差 (>0.5px)"),
    ("excluded", "已剔除帧")
]

SORT_MODE_OPTIONS = [
    ("name_asc", "文件名升序"),
    ("err_desc", "残差降序 (最差优先 ↓)"),
    ("err_asc", "残差升序 (最优优先 ↑)"),
    ("tags_desc", "标靶数量降序")
]

BA_VIEW_OPTIONS = [
    ("3d", "3D 翡翠绿棱柱"),
    ("2d", "2D 理论投影框"),
    ("off", "隐藏 (关闭显示)")
]

OBS_VIEW_OPTIONS = [
    ("3d", "3D 科技天蓝棱柱"),
    ("2d", "2D 实测识别框"),
    ("off", "隐藏 (关闭显示)")
]


def draw_dropdown_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    is_open: bool,
    mouse_pos: Tuple[int, int],
    prefix: str = "",
    theme_color: Optional[Tuple[int, int, int]] = None,
):
    """绘制现代扁平化微质感下拉菜单头部按钮"""
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

    active_col = theme_color if theme_color is not None else (0, 220, 255)

    if is_open:
        bg_col = (48, 56, 72)
        border_col = active_col
        text_col = (255, 255, 255)
        arrow = "▲"
    elif is_hover:
        bg_col = (36, 40, 52)
        border_col = active_col
        text_col = (240, 240, 240)
        arrow = "▼"
    else:
        bg_col = (28, 30, 38)
        border_col = (55, 60, 75)
        text_col = (200, 200, 200)
        arrow = "▼"

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

    display_txt = f"{prefix}{label} {arrow}" if prefix else f"{label} {arrow}"
    (tw, th), _ = measure_text(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    tx = x1 + max(6, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    put_text(canvas, display_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)


def draw_dashboard_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    mouse_pos: Tuple[int, int] = (-1, -1),
    accent: Optional[Tuple[int, int, int]] = None,
    is_running: bool = False,
    font_size: int = 12,
):
    """绘制与 Dashboard (gui_launcher) 同源的碳灰卡片式按钮:
    深色底 + 沉稳边框, 悬停冷青微光, 左缘语义色条 (danger 红 / success 绿), 运行中金色高亮"""
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

    if is_running:
        bg_col, border_col, text_col, border_th = (30, 38, 50), (210, 175, 60), (250, 225, 140), 2
    elif is_hover:
        bg_col, border_col, text_col, border_th = (30, 38, 50), (0, 220, 180), (242, 245, 248), 2
    else:
        bg_col, border_col, text_col, border_th = (22, 26, 33), (38, 46, 58), (205, 215, 225), 1

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_th)
    if accent is not None:
        cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x1 + 4, y2 - 1), accent, -1)

    bbox = get_cached_font(font_size, bold=True).getbbox(label)
    tw, t_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x1 + max(4, ((x2 - x1) - tw) // 2 - bbox[0])
    ty = y1 + ((y2 - y1) - t_h) // 2 - bbox[1]
    draw_text(canvas, label, (tx, ty), font_size=font_size, color=text_col, bold=True)
