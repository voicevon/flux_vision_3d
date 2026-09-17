"""
AprilTag 离线标定工作站 - UI 界面渲染器 (StudioUIRenderer)
================================================================================
负责工作站现代深色全景三栏界面的几何排版与所有视觉元素绘制：
1. 顶栏 (Top Navigation Bar: LOGO + 全局快捷动作按钮, Dashboard 同源风格)
2. 底栏 (Bottom Status Bar: 放行门限 / 拓扑连通度 / 采图与精度状态指标)
3. 左栏 (高信息密度垂直紧凑帧列表 + 双下拉菜单过滤与排序)
4. 中栏视口 (自适应平移缩放、ROI 裁剪、3D 双棱柱与残差矢量投影)
5. 右栏 (180px 瘦身属性面板、逐 Tag 剔除打叉、单帧病因切片诊断)
6. 浮层 (科技感居中异步 BA 双轨进度卡片、Toast 提示、置顶下拉菜单)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.viewport_manager import draw_styled_button
from src.utils.text_rendering import draw_text, get_cached_font


# ============================================================
# Hover Tooltip 定义 (按钮 id -> 帮助文字)
# ============================================================
HOVER_TOOLTIPS: Dict[str, List[str]] = {
    "SAVE_MAP": [
        "【保存地图】",
        "",
        "将当前工作站 BA 全局平差后的 Tag 空间立体地图",
        "写入 config/tags_map.yaml (覆盖保存)。",
        "",
        "地图内容包含:",
        "  • 每个 Tag 的世界坐标系位姿 (x, y, z, 四元数)",
        "  • 观测置信度、重投影误差统计、参与帧数",
        "  • 全局 RMSE / 物理偏差 / 迭代次数等元信息",
        "",
        "下游 (tag_capture_wizard / tag_studio / robot_tracker)",
        "启动时会自动加载此文件作为已知空间基准。",
        "",
        "快捷键: [M]  建议每次 BA 平差后立即保存",
    ],
    "RUN_AUTO_PRUNE_BA": [
        "【智能残差剪枝平差 (A)】",
        "",
        "运行 BA 全局平差 + 自动剔除残差最大的离群观测",
        "保留 RMSE 最优的 Tag 组合，迭代收敛。",
        "",
        "剪枝依据: 单帧残差 / 单 Tag 离群 / 共视拓扑连通性",
        "快捷键: [A]",
    ],
    "RECOMPUTE_METRICS": [
        "【全量体检重算 (P)】",
        "",
        "重新计算所有帧 / 所有 Tag 的精度体检指标:",
        "  • RMSE 重投影误差",
        "  • 空间物理偏差 (mm)",
        "  • 清晰度 / 对比度 / 亮度 / 畸变",
        "快捷键: [P]",
    ],
    "EXPORT_REPORT": [
        "【导出质检报告 (R)】",
        "",
        "将当前工作站的标定结果导出为 Markdown 报告:",
        "  • 收敛曲线 / 残差分布图 / 三维位姿",
        "  • 各项精度指标汇总",
        "  • 质检结论 (是否可发布)",
        "快捷键: [R]",
    ],
}


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
    prefix: str = ""
):
    """绘制现代扁平化微质感下拉菜单头部按钮"""
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

    if is_open:
        bg_col = (48, 56, 72)
        border_col = (0, 220, 255)
        text_col = (255, 255, 255)
        arrow = "▲"
    elif is_hover:
        bg_col = (36, 40, 52)
        border_col = (0, 180, 220)
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
    (tw, th), _ = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    tx = x1 + max(6, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(canvas, display_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)


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


class StudioUIRenderer:
    """Offline Studio UI 渲染器"""

    def __init__(self):
        pass

    def render(self, studio: Any, canvas: np.ndarray):
        """完整渲染 Offline Studio 顶栏、左栏、中栏、右栏与底栏"""
        w, h = studio.win_w, studio.win_h
        top_h = studio.viewport.top_bar_h
        bot_h = studio.viewport.bottom_bar_h
        studio.gui_buttons.clear()

        # 异步 BA 结果轮询
        ba_res = studio.ba_runner.poll_result()
        if ba_res is not None:
            succ, msg = ba_res
            studio.set_toast(msg)
            if succ:
                studio.refresh_all_frame_metrics()

        # 异步全量超精提取结果轮询
        if hasattr(studio, "poll_super_extract_result"):
            ext_res = studio.poll_super_extract_result()
            if ext_res is not None:
                succ, msg = ext_res
                studio.set_toast(msg)
                if succ:
                    studio.refresh_all_frame_metrics()

        # 1. 顶栏
        self.render_top_bar(studio, canvas, w, top_h)

        # 2. 底栏 (全局状态信息条)
        self.render_bottom_status_bar(studio, canvas, w, h, bot_h)

        # 3. 工作区尺寸与左栏自适应宽度
        studio.left_bar_w = getattr(studio, "dynamic_left_bar_w", studio.left_bar_w)
        content_y1 = top_h
        content_y2 = h - bot_h
        content_h = content_y2 - content_y1

        # 左栏：紧凑帧序列列表或逐帧多轮残差演进矩阵宽表
        self.render_left_frame_list(studio, canvas, 0, content_y1, studio.left_bar_w, content_h)

        # 右栏：180px 瘦身属性与单帧诊断面板
        self.render_right_inspector(studio, canvas, w - studio.right_bar_w, content_y1, studio.right_bar_w, content_h)

        # 中栏：视口
        mid_x1 = studio.left_bar_w
        mid_w = w - studio.left_bar_w - studio.right_bar_w
        self.render_center_viewport(studio, canvas, mid_x1, content_y1, mid_w, content_h)

        # 4. 居中展示浮层卡片 (优先级: 结算对比卡片 > 智能剪枝进度卡片 > BA进度卡片 > 超精提取进度卡片)
        if getattr(studio, "prune_settlement_data", None) is not None:
            self.render_prune_settlement_card(studio, canvas, w, h)
        elif getattr(studio, "is_auto_pruning", False):
            self.render_prune_ba_card(studio, canvas, w, h)
        elif studio.is_ba_running:
            self.render_ba_loading_card(studio, canvas, w, h)
        elif getattr(studio, "is_extracting_all", False):
            self.render_extract_loading_card(studio, canvas, w, h)

        # 5. Toast 浮层
        if time.time() - studio.status_toast_time < 3.0 and studio.status_toast:
            self.render_toast(studio, canvas, w, h, bot_h)

        # 6. 置顶悬浮下拉列表
        if studio.active_dropdown and studio.active_dropdown in studio.dropdown_boxes:
            dd_info = studio.dropdown_boxes[studio.active_dropdown]
            self.render_dropdown_popup(studio, canvas, studio.active_dropdown,
                                      dd_info["rect"], dd_info["options"], dd_info["active_key"])

        # 7. Hover 帮助气泡 (最后绘制, 覆盖在所有面板之上, 不自动关闭)
        mx, my = studio.mouse_pos
        self._render_hover_tooltip(canvas, studio, mx, my, w, h)

    def render_top_bar(self, studio: Any, canvas: np.ndarray, w: int, top_h: int):
        """顶栏：LOGO + 紧随其后的全局快捷动作按钮 (Dashboard 同源风格)"""
        cv2.rectangle(canvas, (0, 0), (w, top_h), (24, 26, 32), -1)
        cv2.line(canvas, (0, top_h), (w, top_h), (55, 60, 72), 1)

        # 1. LOGO
        cv2.putText(canvas, "OFFLINE STUDIO", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
        (logo_w, _), _ = cv2.getTextSize("OFFLINE STUDIO", cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)

        # 2. LOGO 右侧紧邻的全局快捷动作按钮
        mx, my = studio.mouse_pos
        btn_y_top, btn_y_bot = 7, top_h - 7
        bx = 16 + logo_w + 26

        # 1. 复位地图 (清空已知平差地图)
        rst_map_w = 105
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + rst_map_w, btn_y_bot), "复位地图",
                              mouse_pos=(mx, my), accent=(70, 60, 210))
        studio.gui_buttons.append(("RESET_MAP", (bx, btn_y_top, bx + rst_map_w, btn_y_bot), "RESET_MAP"))
        bx += rst_map_w + 10

        # 2. 全局全量超精提取 (清空旧角点并从头重提取)
        ext_w = 135
        is_ext = getattr(studio, "is_extracting_all", False)
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + ext_w, btn_y_bot),
                              "正在超精提取..." if is_ext else "全局超精提取",
                              mouse_pos=(mx, my), is_running=is_ext)
        studio.gui_buttons.append(("SUPER_EXTRACT_ALL", (bx, btn_y_top, bx + ext_w, btn_y_bot), "SUPER_EXTRACT_ALL"))
        bx += ext_w + 10

        # 3. [B] 全局平差
        ba_w = 135
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + ba_w, btn_y_bot),
                              "正在平差..." if studio.is_ba_running else "全局平差 (B)",
                              mouse_pos=(mx, my), is_running=studio.is_ba_running)
        studio.gui_buttons.append(("RUN_BA", (bx, btn_y_top, bx + ba_w, btn_y_bot), "RUN_BA"))
        bx += ba_w + 10

        # 4. [A] 智能残差剪枝平差
        prune_w = 135
        is_prune = getattr(studio, "is_auto_pruning", False)
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + prune_w, btn_y_bot),
                              "正在剪枝..." if is_prune else "剪枝平差 (A)",
                              mouse_pos=(mx, my), is_running=is_prune)
        studio.gui_buttons.append(("RUN_AUTO_PRUNE_BA", (bx, btn_y_top, bx + prune_w, btn_y_bot), "RUN_AUTO_PRUNE_BA"))
        bx += prune_w + 10

        # 5. [M] 保存/发布地图
        s_w = 115
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + s_w, btn_y_bot), "保存地图 (M)",
                              mouse_pos=(mx, my), accent=(0, 215, 90))
        studio.gui_buttons.append(("SAVE_MAP", (bx, btn_y_top, bx + s_w, btn_y_bot), "SAVE_MAP"))
        bx += s_w + 10

        # 6. [P] 全程/全量精度体检重算
        p_w = 125
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + p_w, btn_y_bot), "全量体检 (P)",
                              mouse_pos=(mx, my))
        studio.gui_buttons.append(("RECOMPUTE_METRICS", (bx, btn_y_top, bx + p_w, btn_y_bot), "RECOMPUTE_METRICS"))
        bx += p_w + 10

        # 7. [R] 导出质检报告
        r_w = 115
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + r_w, btn_y_bot), "导出报告 (R)",
                              mouse_pos=(mx, my))
        studio.gui_buttons.append(("EXPORT_REPORT", (bx, btn_y_top, bx + r_w, btn_y_bot), "EXPORT_REPORT"))
        bx += r_w + 10

        # 8. 复位保留 (一键恢复所有剔除的观测为有效)
        rst_keep_w = 105
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + rst_keep_w, btn_y_bot), "复位保留",
                              mouse_pos=(mx, my))
        studio.gui_buttons.append(("RESET_KEEP_ALL", (bx, btn_y_top, bx + rst_keep_w, btn_y_bot), "RESET_KEEP_ALL"))

        # 9. 右侧 [Q] 退出工作台 (最右侧退出不动)
        exit_w = 90
        exit_x1 = w - exit_w - 14
        draw_dashboard_button(canvas, (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "退出 (Q)",
                              mouse_pos=(mx, my), accent=(70, 60, 210))
        studio.gui_buttons.append(("EXIT", (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "EXIT"))

    def render_bottom_status_bar(self, studio: Any, canvas: np.ndarray, w: int, h: int, bot_h: int):
        """底栏：全局状态信息条 (放行门限徽章 / 拓扑连通度 / 采图与精度统计)"""
        y1 = h - bot_h
        cv2.rectangle(canvas, (0, y1), (w, h), (20, 22, 28), -1)
        cv2.line(canvas, (0, y1), (w, y1), (60, 65, 78), 1)

        badge_y1, badge_y2 = y1 + 10, h - 10
        badge_cy = (badge_y1 + badge_y2) // 2
        curr_x = 16

        # 1. 质量放行门限徽章 (Gate Verdict)
        gate = getattr(studio, "gate_status", "REVIEW")
        verdict = gate if isinstance(gate, str) else gate.get("gate_verdict", "REVIEW")
        if verdict == "PASS":
            v_txt = "放行: PASS"
            v_bg, v_border, v_fg = (20, 70, 30), (40, 180, 70), (160, 255, 180)
        elif verdict == "ACCEPTABLE":
            v_txt = "放行: 可接受"
            v_bg, v_border, v_fg = (20, 60, 80), (30, 160, 220), (140, 230, 255)
        else:
            v_txt = "放行: 建议回审"
            v_bg, v_border, v_fg = (30, 20, 80), (50, 40, 220), (180, 160, 255)

        vb = get_cached_font(12).getbbox(v_txt)
        v_box_w = (vb[2] - vb[0]) + 16
        cv2.rectangle(canvas, (curr_x, badge_y1), (curr_x + v_box_w, badge_y2), v_bg, -1)
        cv2.rectangle(canvas, (curr_x, badge_y1), (curr_x + v_box_w, badge_y2), v_border, 1)
        draw_text(canvas, v_txt, (curr_x + 8, badge_cy - (vb[3] - vb[1]) // 2 - vb[1]),
                  font_size=12, color=v_fg)
        curr_x += v_box_w + 10

        # 2. 拓扑连通度徽章 (Topology Status)
        topo = getattr(studio, "topology_status", {})
        unconnected = topo.get("unconnected_tags", [])
        if unconnected:
            t_txt = f"拓扑: 孤岛 #{unconnected[0]}"
            t_bg, t_border, t_fg = (20, 20, 75), (40, 40, 200), (140, 140, 255)
        elif not topo.get("is_valid", True):
            t_txt = "拓扑: 弱连通"
            t_bg, t_border, t_fg = (20, 50, 75), (40, 130, 200), (140, 210, 255)
        else:
            t_txt = "拓扑: 全连通"
            t_bg, t_border, t_fg = (20, 60, 35), (40, 160, 80), (160, 255, 190)

        tb = get_cached_font(12).getbbox(t_txt)
        t_box_w = (tb[2] - tb[0]) + 16
        cv2.rectangle(canvas, (curr_x, badge_y1), (curr_x + t_box_w, badge_y2), t_bg, -1)
        cv2.rectangle(canvas, (curr_x, badge_y1), (curr_x + t_box_w, badge_y2), t_border, 1)
        draw_text(canvas, t_txt, (curr_x + 8, badge_cy - (tb[3] - tb[1]) // 2 - tb[1]),
                  font_size=12, color=t_fg)
        curr_x += t_box_w + 14

        # 3. 统计指标文字 (采图数 | 标靶数 | 全局 RMSE / 空间毫米偏差)
        tag_num = len(studio.tags_map_data.get("tags", {}))
        med_mm = getattr(studio, "global_median_mm", 0.0)
        stat_txt = f"采图集: {len(studio.image_files)} 帧 | 标靶: {tag_num} 个 | 全局 RMSE: {studio.global_rmse:.2f}px ({med_mm:.2f}mm)"
        sb = get_cached_font(12).getbbox(stat_txt)
        draw_text(canvas, stat_txt, (curr_x, badge_cy - (sb[3] - sb[1]) // 2 - sb[1]),
                  font_size=12, color=(0, 255, 180))

        # 4. 右侧产品副标题 (muted)
        sub_txt = "AprilTag 离线标定与空间建图综合工作站"
        ub = get_cached_font(11).getbbox(sub_txt)
        draw_text(canvas, sub_txt, (w - 16 - (ub[2] - ub[0]), badge_cy - (ub[3] - ub[1]) // 2 - ub[1]),
                  font_size=11, color=(115, 130, 145))

    def _render_hover_tooltip(self, canvas: np.ndarray, studio: Any, mx: int, my: int,
                               cw: int, ch: int):
        """检测鼠标是否悬停在有帮助文字的按钮上, 若有则画气泡面板"""
        if mx < 0 or my < 0:
            return
        for btn_id, (bx1, by1, bx2, by2), _ in getattr(studio, "gui_buttons", []):
            if btn_id not in HOVER_TOOLTIPS:
                continue
            if bx1 <= mx <= bx2 and by1 <= my <= by2:
                lines = HOVER_TOOLTIPS[btn_id]
                # 测量尺寸
                font = get_cached_font(11, bold=False)
                bold_font = get_cached_font(14, bold=True)
                lh = 18
                pad_x, pad_y = 14, 12
                inner_w = max((len(l) + 4) * 11 for l in lines)
                tip_w = min(inner_w + pad_x * 2, 520)
                tip_h = len(lines) * lh + pad_y * 2

                # 定位: 优先在按钮正上方, 不够空间则在下方
                gap = 8
                if by1 - tip_h - gap >= 0:
                    ty2 = by1 - gap
                    ty1 = ty2 - tip_h
                else:
                    ty1 = by2 + gap
                    ty2 = ty1 + tip_h
                tx1 = max(8, min(bx1, cw - tip_w - 8))
                tx2 = tx1 + tip_w

                # 半透明暗色底
                overlay = canvas.copy()
                cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (22, 24, 32), -1)
                cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, dst=canvas)
                cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (0, 210, 180), 2)

                # 画文字
                ly = ty1 + pad_y + 6
                for i, line in enumerate(lines):
                    if line.startswith("【") and line.endswith("】"):
                        draw_text(canvas, line, (tx1 + pad_x, ly), font_size=13,
                                  color=(0, 230, 200), bold=True)
                    elif line == "":
                        pass
                    else:
                        draw_text(canvas, line, (tx1 + pad_x, ly), font_size=11,
                                  color=(220, 225, 235), bold=False)
                    ly += lh
                return

    def render_left_frame_list(self, studio: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """左栏：高信息密度垂直紧凑帧列表 或 逐帧多轮残差演进矩阵宽表大视图"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x + w, y), (x + w, y + h), (50, 54, 66), 1)

        is_matrix = getattr(studio, "matrix_view_mode", False)
        headers = getattr(studio.data_mgr, "convergence_headers", [])
        matrix = getattr(studio.data_mgr, "frame_convergence_matrix", {})

        header_h = 36
        dd_y1 = y + 6
        dd_y2 = y + header_h

        # 1. 顶部操作栏自适应排版：切换按钮 + 筛选下拉框 + 排序下拉框
        btn_w = 98 if is_matrix else 86
        btn_x1 = x + w - btn_w - 8
        btn_x2 = x + w - 8

        avail_w = max(120, btn_x1 - (x + 8) - 8)
        dd1_w = avail_w // 2
        dd2_w = avail_w - dd1_w - 6

        dd1_x1 = x + 8
        dd1_x2 = dd1_x1 + dd1_w
        dd2_x1 = dd1_x2 + 6
        dd2_x2 = dd2_x1 + dd2_w

        # 筛选范围下拉框
        cur_filter_label = dict(FILTER_MODE_OPTIONS).get(studio.filter_mode, "全部帧")
        is_f_open = (studio.active_dropdown == "FILTER_DROPDOWN")
        draw_dropdown_button(canvas, (dd1_x1, dd_y1, dd1_x2, dd_y2), cur_filter_label,
                             is_open=is_f_open, mouse_pos=studio.mouse_pos)
        studio.dropdown_boxes["FILTER_DROPDOWN"] = {
            "rect": (dd1_x1, dd_y1, dd1_x2, dd_y2),
            "options": FILTER_MODE_OPTIONS,
            "active_key": studio.filter_mode
        }
        studio.gui_buttons.append(("TOGGLE_FILTER_DROPDOWN", (dd1_x1, dd_y1, dd1_x2, dd_y2), "FILTER_DROPDOWN"))

        # 排序方式下拉框
        cur_sort_label = dict(SORT_MODE_OPTIONS).get(studio.sort_mode, "文件名升序")
        short_sort = cur_sort_label.split(" ")[0] if "(" in cur_sort_label else cur_sort_label
        is_s_open = (studio.active_dropdown == "SORT_DROPDOWN")
        draw_dropdown_button(canvas, (dd2_x1, dd_y1, dd2_x2, dd_y2), short_sort,
                             is_open=is_s_open, mouse_pos=studio.mouse_pos)
        studio.dropdown_boxes["SORT_DROPDOWN"] = {
            "rect": (dd2_x1, dd_y1, dd2_x2, dd_y2),
            "options": SORT_MODE_OPTIONS,
            "active_key": studio.sort_mode
        }
        studio.gui_buttons.append(("TOGGLE_SORT_DROPDOWN", (dd2_x1, dd_y1, dd2_x2, dd_y2), "SORT_DROPDOWN"))

        # 矩阵模式切换按钮 (一键展开/收起 10 轮残差对比大表)
        btn_txt = "⊟ 紧凑 (X)" if is_matrix else "⊞ 矩阵 (X)"
        btn_type = "primary" if is_matrix else "secondary"
        draw_styled_button(canvas, (btn_x1, dd_y1, btn_x2, dd_y2), btn_txt,
                           mouse_pos=studio.mouse_pos, btn_type=btn_type)
        studio.gui_buttons.append(("TOGGLE_MATRIX_VIEW", (btn_x1, dd_y1, btn_x2, dd_y2), "TOGGLE_MATRIX_VIEW"))

        filtered_indices = studio._get_filtered_indices()
        if not filtered_indices:
            cv2.putText(canvas, "当前筛选条件下无图像", (x + 60, y + header_h + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1, cv2.LINE_AA)
            return

        # 2. 列表内容区布局与渲染
        if is_matrix:
            # ==================== 【模式 A: 逐帧多轮残差演进矩阵宽表】 ====================
            th_h = 24
            th_y1 = y + header_h + 6
            th_y2 = th_y1 + th_h
            list_y = th_y2 + 4
            item_h = 30
            visible_count = (h - (list_y - y) - 10) // item_h

            studio.scroll_offset = max(0, min(studio.scroll_offset, len(filtered_indices) - visible_count))

            # 绘制矩阵表头背景
            cv2.rectangle(canvas, (x + 6, th_y1), (x + w - 6, th_y2), (32, 36, 46), -1)
            cv2.rectangle(canvas, (x + 6, th_y1), (x + w - 6, th_y2), (52, 58, 72), 1)

            c_name_w = 110
            c_tag_w = 34
            c_round_w = 54
            c_drop_w = 68

            # 表头固定列
            cv2.putText(canvas, "图像帧", (x + 14, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Tag", (x + 14 + c_name_w, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)

            cur_col_x = x + 14 + c_name_w + c_tag_w
            # 动态平差轮次表头列 (R0, R1, R2, ..., R10)
            active_headers = headers if headers else ["基准(R0)"]
            for col_idx, h_name in enumerate(active_headers):
                is_latest_col = (col_idx == len(active_headers) - 1)
                th_color = (0, 240, 255) if is_latest_col else (180, 185, 195)
                # 最新一轮列高亮底框
                if is_latest_col and len(active_headers) > 1:
                    cv2.rectangle(canvas, (cur_col_x - 2, th_y1 + 2), (cur_col_x + c_round_w - 4, th_y2 - 2), (25, 60, 75), -1)
                cv2.putText(canvas, h_name, (cur_col_x + 6, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, th_color, 1 if not is_latest_col else 2, cv2.LINE_AA)
                cur_col_x += c_round_w

            # 累计改善降幅列
            cv2.putText(canvas, "累计降幅", (cur_col_x + 4, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 120), 1, cv2.LINE_AA)

            # 绘制逐帧矩阵数据行
            for row_idx in range(visible_count):
                list_idx = studio.scroll_offset + row_idx
                if list_idx >= len(filtered_indices):
                    break

                orig_img_idx = filtered_indices[list_idx]
                p = studio.image_files[orig_img_idx]
                bname = os.path.basename(p)
                meta = studio.frame_metrics_cache.get(bname, {})

                iy1 = list_y + row_idx * item_h
                iy2 = iy1 + item_h - 2
                is_selected = (orig_img_idx == studio.current_img_idx)
                is_hover = (x + 6 <= studio.mouse_pos[0] <= x + w - 6 and iy1 <= studio.mouse_pos[1] <= iy2)

                if is_selected:
                    row_bg = (48, 42, 28)
                    border_c = (0, 220, 255)
                elif is_hover:
                    row_bg = (35, 38, 48)
                    border_c = (55, 60, 75)
                else:
                    row_bg = (24, 26, 33) if row_idx % 2 == 0 else (20, 22, 28)
                    border_c = (36, 38, 48)

                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), row_bg, -1)
                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), border_c, 1)

                btn_id = f"SELECT_FRAME_{orig_img_idx}"
                studio.gui_buttons.append((btn_id, (x + 6, iy1, x + w - 6, iy2), orig_img_idx))

                # 状态小圆点
                dot_y = iy1 + item_h // 2
                is_excl = meta.get("is_excluded", False)
                if is_excl:
                    dot_c = (0, 0, 240)
                elif meta.get("mean_err", 0.0) > 0.5:
                    dot_c = (0, 180, 255)
                else:
                    dot_c = (0, 230, 80)
                cv2.circle(canvas, (x + 14, dot_y), 3, dot_c, -1)

                # 文件名 (截取前10个字符)
                name_stem = bname.replace(".png", "").replace(".jpg", "")
                short_name = name_stem if len(name_stem) <= 10 else name_stem[:9] + "…"
                name_col = (255, 255, 255) if is_selected else (200, 205, 215)
                cv2.putText(canvas, short_name, (x + 22, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, name_col, 1, cv2.LINE_AA)

                # Tag 数量
                tag_cnt = meta.get("tag_count", 0)
                cv2.putText(canvas, f"{tag_cnt}T", (x + 14 + c_name_w, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 150, 165), 1, cv2.LINE_AA)

                # 各轮次残差单元格值 (R0, R1, ..., R10)
                row_vals = matrix.get(bname, [])
                r_col_x = x + 14 + c_name_w + c_tag_w

                for c_idx in range(len(active_headers)):
                    is_latest_c = (c_idx == len(active_headers) - 1)
                    val = row_vals[c_idx] if c_idx < len(row_vals) else None

                    # 最新一轮单元格微高亮底框
                    if is_latest_c and len(active_headers) > 1 and not is_selected:
                        cv2.rectangle(canvas, (r_col_x - 2, iy1 + 2), (r_col_x + c_round_w - 6, iy2 - 2), (20, 42, 54), -1)

                    if val is None or is_excl:
                        v_txt = "EXCL" if is_excl else "--"
                        v_col = (90, 95, 110) if not is_excl else (0, 0, 200)
                        cv2.putText(canvas, v_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, v_col, 1, cv2.LINE_AA)
                    else:
                        v_txt = f"{val:.1f}" if val >= 100.0 else f"{val:.2f}"
                        if val > 1.0:
                            v_col = (0, 180, 255)
                        elif val > 0.5:
                            v_col = (0, 220, 255)
                        else:
                            v_col = (0, 240, 100)
                        cv2.putText(canvas, v_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, v_col, 1, cv2.LINE_AA)
                    r_col_x += c_round_w

                # 累计降幅百分比
                first_v = row_vals[0] if (row_vals and row_vals[0] is not None) else None
                last_v = None
                for rv in reversed(row_vals):
                    if rv is not None:
                        last_v = rv
                        break

                if first_v is not None and last_v is not None and first_v > 0.001:
                    drop_val = first_v - last_v
                    drop_pct = (drop_val / first_v) * 100.0
                    if drop_pct >= 5.0:
                        pct_txt = f"↓{drop_pct:.0f}%" if drop_pct >= 10.0 else f"↓{drop_pct:.1f}%"
                        pct_col = (0, 240, 255)
                    elif drop_pct <= -5.0:
                        pct_txt = f"↑{abs(drop_pct):.0f}%"
                        pct_col = (0, 100, 255)
                    else:
                        pct_txt = "0%"
                        pct_col = (150, 160, 175)
                else:
                    pct_txt = "--"
                    pct_col = (110, 115, 125)

                cv2.putText(canvas, pct_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pct_col, 1, cv2.LINE_AA)

        else:
            # ==================== 【模式 B: 高信息密度垂直紧凑帧列表】 ====================
            list_y = y + header_h + 8
            item_h = 36
            visible_count = (h - header_h - 16) // item_h

            studio.scroll_offset = max(0, min(studio.scroll_offset, len(filtered_indices) - visible_count))

            for row_idx in range(visible_count):
                list_idx = studio.scroll_offset + row_idx
                if list_idx >= len(filtered_indices):
                    break

                orig_img_idx = filtered_indices[list_idx]
                p = studio.image_files[orig_img_idx]
                bname = os.path.basename(p)
                meta = studio.frame_metrics_cache.get(bname, {})

                iy1 = list_y + row_idx * item_h
                iy2 = iy1 + item_h - 2
                is_selected = (orig_img_idx == studio.current_img_idx)
                is_hover = (x + 4 <= studio.mouse_pos[0] <= x + w - 4 and iy1 <= studio.mouse_pos[1] <= iy2)

                if is_selected:
                    row_bg = (48, 42, 28)
                    border_c = (0, 220, 255)
                elif is_hover:
                    row_bg = (35, 38, 48)
                    border_c = (55, 60, 75)
                else:
                    row_bg = (25, 27, 34)
                    border_c = (38, 40, 50)

                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), row_bg, -1)
                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), border_c, 1)

                btn_id = f"SELECT_FRAME_{orig_img_idx}"
                studio.gui_buttons.append((btn_id, (x + 6, iy1, x + w - 6, iy2), orig_img_idx))

                # 状态小圆点
                dot_y = iy1 + item_h // 2
                is_excl = meta.get("is_excluded", False)
                if is_excl:
                    dot_c = (0, 0, 240)
                elif meta.get("mean_err", 0.0) > 0.5:
                    dot_c = (0, 180, 255)
                else:
                    dot_c = (0, 230, 80)
                cv2.circle(canvas, (x + 18, dot_y), 4, dot_c, -1)

                # 文件名
                txt_col = (255, 255, 255) if is_selected else (200, 200, 200)
                short_name = bname if len(bname) <= 15 else bname[:12] + "..."
                cv2.putText(canvas, short_name, (x + 28, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

                # Tag 计数
                tag_cnt = meta.get("tag_count", 0)
                t_str = f"{tag_cnt}T"
                cv2.putText(canvas, t_str, (x + 168, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

                # 平均残差数值与降幅
                if is_excl:
                    cv2.putText(canvas, "EXCL", (x + 218, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 240), 1, cv2.LINE_AA)
                else:
                    err_val = meta.get('mean_err', 0.0)
                    err_str = f"{err_val:.2f}px"
                    err_col = (0, 200, 255) if err_val > 0.5 else (0, 240, 100)
                    cv2.putText(canvas, err_str, (x + 218, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, err_col, 1, cv2.LINE_AA)

    def render_center_viewport(self, studio: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """中栏：高清工作视口，等比居中自适应渲染"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (14, 15, 18), -1)

        if not studio.image_files:
            cv2.putText(canvas, "未扫描到采图图像 (data/tag_calibration_images/ 为空)", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
            return

        cur_file = studio.image_files[studio.current_img_idx]
        bgr = cv2.imread(cur_file)
        if bgr is None:
            cv2.putText(canvas, f"读取图像文件失败: {cur_file}", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            return

        disp_frame = bgr.copy()
        base_name = os.path.basename(cur_file)
        meta = studio.frame_metrics_cache.get(base_name, {})
        obs_list = meta.get("observations", [])

        # 叠加标靶与 3D 双棱柱
        self.overlay_visual_elements(studio, disp_frame, obs_list, meta.get("is_excluded", False), meta=meta,
                                     panel_rect=(x, y, w, h))

        # 视口等比与平移缩放渲染 (委托给 viewport 控制器)
        frame_h, frame_w = disp_frame.shape[:2]
        rois = studio.viewport.compute_viewport_render_rois((x, y, w, h), frame_w, frame_h)
        if rois is not None:
            (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = rois
            src_roi = disp_frame[src_y1:src_y2, src_x1:src_x2]
            dst_w = dst_x2 - dst_x1
            dst_h = dst_y2 - dst_y1
            if dst_w > 0 and dst_h > 0 and src_roi.size > 0:
                interp = cv2.INTER_LINEAR if studio.zoom_level > 1.2 else cv2.INTER_AREA
                resized_roi = cv2.resize(src_roi, (dst_w, dst_h), interpolation=interp)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized_roi

        # 视口外边框
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 60, 70), 1)

        # 视口左上角：双独立下拉菜单 (BA 理论值控制 + 单帧实测值控制)
        ba_x1 = x + 12
        ba_y1 = y + 10
        ba_x2 = ba_x1 + 148
        ba_y2 = ba_y1 + 28
        cur_ba_label = dict(BA_VIEW_OPTIONS).get(studio.ba_view_mode, "3D 翡翠绿棱柱")
        is_ba_open = (studio.active_dropdown == "BA_VIEW_DROPDOWN")
        draw_dropdown_button(canvas, (ba_x1, ba_y1, ba_x2, ba_y2), cur_ba_label,
                             is_open=is_ba_open, mouse_pos=studio.mouse_pos, prefix="BA理论: ")
        studio.dropdown_boxes["BA_VIEW_DROPDOWN"] = {
            "rect": (ba_x1, ba_y1, ba_x2, ba_y2),
            "options": BA_VIEW_OPTIONS,
            "active_key": studio.ba_view_mode
        }
        studio.gui_buttons.append(("TOGGLE_BA_VIEW_DROPDOWN", (ba_x1, ba_y1, ba_x2, ba_y2), "BA_VIEW_DROPDOWN"))

        obs_x1 = ba_x2 + 8
        obs_y1 = y + 10
        obs_x2 = obs_x1 + 148
        obs_y2 = obs_y1 + 28
        cur_obs_label = dict(OBS_VIEW_OPTIONS).get(studio.obs_view_mode, "3D 科技天蓝棱柱")
        is_obs_open = (studio.active_dropdown == "OBS_VIEW_DROPDOWN")
        draw_dropdown_button(canvas, (obs_x1, obs_y1, obs_x2, obs_y2), cur_obs_label,
                             is_open=is_obs_open, mouse_pos=studio.mouse_pos, prefix="实测识别: ")
        studio.dropdown_boxes["OBS_VIEW_DROPDOWN"] = {
            "rect": (obs_x1, obs_y1, obs_x2, obs_y2),
            "options": OBS_VIEW_OPTIONS,
            "active_key": studio.obs_view_mode
        }
        studio.gui_buttons.append(("TOGGLE_OBS_VIEW_DROPDOWN", (obs_x1, obs_y1, obs_x2, obs_y2), "OBS_VIEW_DROPDOWN"))

        # 视口右上角悬浮提示胶囊
        zoom_badge = f"缩放: {studio.zoom_level:.1f}x | 点击Tag: 剔除/恢复(打叉) | 切换模式: V | 拖拽: 右键/中键 | 双击/Z: 重置"
        (zw, zh), _ = cv2.getTextSize(zoom_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        bx1 = x + w - zw - 24
        by1 = y + 10
        bx2 = bx1 + zw + 14
        by2 = by1 + zh + 10
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 24, 32), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (70, 75, 88), 1)
        cv2.putText(canvas, zoom_badge, (bx1 + 7, by1 + zh + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 210, 230), 1, cv2.LINE_AA)

    def overlay_visual_elements(
        self,
        studio: Any,
        disp_frame: np.ndarray,
        observations: List[Dict[str, Any]],
        is_frame_excluded: bool,
        meta: Optional[Dict[str, Any]] = None,
        panel_rect: Optional[Tuple[int, int, int, int]] = None
    ):
        """依据 ba_view_mode 与 obs_view_mode 双独立维度解耦渲染，剔除标靶显著打红叉"""
        ba_mode = studio.ba_view_mode
        obs_mode = studio.obs_view_mode

        # 画布鼠标坐标 -> 原始帧坐标 (悬停展开标靶详情, 高密度场景防遮挡)
        mouse_frame = None
        if panel_rect is not None and getattr(studio, "mouse_pos", None):
            try:
                fh, fw = disp_frame.shape[:2]
                img_rect = studio.viewport.compute_image_rect(panel_rect, fw, fh)
                ix1, iy1, ix2, iy2 = img_rect[0], img_rect[1], img_rect[2], img_rect[3]
                if ix2 > ix1 and iy2 > iy1:
                    mfx = (studio.mouse_pos[0] - ix1) / float(ix2 - ix1) * fw
                    mfy = (studio.mouse_pos[1] - iy1) / float(iy2 - iy1) * fh
                    if 0 <= mfx < fw and 0 <= mfy < fh:
                        mouse_frame = (mfx, mfy)
            except Exception:
                mouse_frame = None

        obj_pts = []
        img_pts = []
        valid_obs = []

        # 1. 第一阶段：绘制实测观测标注（有效标靶记录用于 PnP，剔除标靶绘制红叉审核标记）
        obs_map = {}
        for obs in observations:
            tid = obs["tag_id"]
            obs_map[tid] = obs
            pts = np.array(obs["corners"], dtype=np.int32).reshape((-1, 2))
            keep = obs.get("keep", True) and not is_frame_excluded

            # 剔除状态下在标靶实测位置绘制鲜红显著的大叉号 (打叉审核模式)
            if not keep:
                cv2.line(disp_frame, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0, 0, 235), 3, cv2.LINE_AA)
                cv2.line(disp_frame, (pts[1][0], pts[1][1]), (pts[3][0], pts[3][1]), (0, 0, 235), 3, cv2.LINE_AA)
                cv2.polylines(disp_frame, [pts], isClosed=True, color=(40, 40, 180), thickness=2, lineType=cv2.LINE_AA)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.putText(disp_frame, f"Tag #{tid} [EXCL]", (cx - 42, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 240), 2, cv2.LINE_AA)
            else:
                # 收集参与三维相机位姿解算的已知有效标靶
                w_c = studio.get_tag_world_corners(tid)
                if w_c is not None:
                    obj_pts.append(w_c)
                    img_pts.append(np.array(obs["corners"], dtype=np.float64))
                    valid_obs.append(obs)

        rendered_tids = set()

        # 2. 第二阶段：解算当前相机位姿 (PnP)
        rvec = None
        tvec = None
        success = False
        if len(obj_pts) >= 1:
            obj_flat = np.concatenate(obj_pts, axis=0)
            img_flat = np.concatenate(img_pts, axis=0)
            rvec, tvec, success = studio.engine.solve_pnp(obj_flat, img_flat)

        # 若当前无足够有效点 (如标靶全被剔除)，尝试复用 meta 缓存的相机外参
        if not success and meta is not None:
            rvec_c = meta.get("rvec")
            tvec_c = meta.get("tvec")
            if rvec_c is not None and tvec_c is not None:
                rvec = np.array(rvec_c, dtype=np.float64)
                tvec = np.array(tvec_c, dtype=np.float64)
                success = True

        # 3. 第三阶段：3D 棱柱与残差立体渲染 (无论标靶是否被剔除，只要开启 ba_mode=='3d'，绿色 BA 理论棱柱全量呈现！)
        if success:
            R_c_w, _ = cv2.Rodrigues(rvec)
            T_c_w = np.eye(4, dtype=np.float64)
            T_c_w[:3, :3] = R_c_w
            T_c_w[:3, 3] = tvec.flatten()

            need_3d = (ba_mode == "3d" or obs_mode == "3d")
            if need_3d:
                # 收集候选标靶：
                # (a) 当前帧观测到的所有标靶 (不论保留还是已剔除)
                # (b) 如果开启了 ba_mode == "3d"，还包含地图中已建图的其余已知标靶
                candidate_tids = list(obs_map.keys())
                if ba_mode == "3d":
                    tags_dict = getattr(studio, "tags_map_data", {}).get("tags", {})
                    if not tags_dict and hasattr(studio, "data_mgr"):
                        tags_dict = studio.data_mgr.tags_map_data.get("tags", {})
                    for m_tid in tags_dict.keys():
                        if m_tid not in obs_map:
                            candidate_tids.append(m_tid)

                h_f, w_f = disp_frame.shape[:2]
                # FR-9.6 世界系位姿元数据 (平差锚定后每枚标靶的 XYZ 与 RPY)
                tags_meta = (getattr(studio, "tags_map_data", {}) or {}).get("tags", {})
                if not tags_meta and hasattr(studio, "data_mgr"):
                    tags_meta = (getattr(studio.data_mgr, "tags_map_data", {}) or {}).get("tags", {})
                for tid in candidate_tids:
                    T_w_t = studio.get_tag_transform(tid)
                    if T_w_t is None:
                        continue

                    # 计算标靶在当前相机系下的理论位姿
                    T_c_t = T_c_w @ T_w_t
                    t_tag_center = T_c_t[:3, 3]

                    # 标靶必须位于相机正前方
                    if t_tag_center[2] <= 50.0:
                        continue

                    # 理论 BA 位姿 (翡翠绿)
                    r_tag, t_tag = None, None
                    if ba_mode == "3d":
                        r_tag, _ = cv2.Rodrigues(T_c_t[:3, :3])
                        t_tag = t_tag_center.reshape((3, 1))

                    obs = obs_map.get(tid)
                    obs_r, obs_t = None, None
                    c_arr = None
                    succ_single = False
                    is_kept = False

                    if obs is not None:
                        c_arr = np.array(obs["corners"], dtype=np.float64).reshape((4, 2))
                        is_kept = obs.get("keep", True) and not is_frame_excluded
                        # 仅在有效保留且 obs_mode=='3d' 下才计算并显示实测蓝色棱柱
                        if is_kept and obs_mode == "3d":
                            # 传入地图理论法向, 消除 IPPE 平面二义性 180° 翻转
                            succ_single, obs_r, obs_t = studio.engine.solve_single_tag_pnp(
                                c_arr, expected_z_cam=T_c_t[:3, :3][:, 2])

                    # 如果既不画理论绿色棱柱，也不画实测蓝色棱柱，跳过
                    if r_tag is None and obs_r is None:
                        continue

                    # 若当前标靶未检出 (纯理论)，检查理论中心是否在像面可视范围内
                    if obs is None and r_tag is not None:
                        p_center, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), r_tag, t_tag, studio.engine.camera_matrix, studio.engine.dist_coeffs)
                        cu, cv = p_center.reshape(-1)
                        if not (-80 <= cu <= w_f + 80 and -80 <= cv <= h_f + 80):
                            continue

                    err_mm = 0.0
                    if t_tag is not None and succ_single and obs_t is not None:
                        err_mm = float(np.linalg.norm(t_tag - obs_t))
                    err_px = (meta or {}).get("tag_errors", {}).get(tid, 0.2)

                    # 悬停命中检测 (帧坐标, 48px 半径): 实测以观测角点中心, 纯理论以投影中心
                    tag_center_f = None
                    if c_arr is not None:
                        tag_center_f = (float(np.mean(c_arr[:, 0])), float(np.mean(c_arr[:, 1])))
                    elif r_tag is not None:
                        p_c, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), r_tag, t_tag,
                                                   studio.engine.camera_matrix, studio.engine.dist_coeffs)
                        tag_center_f = (float(p_c.reshape(-1)[0]), float(p_c.reshape(-1)[1]))
                    is_hovered = (mouse_frame is not None and tag_center_f is not None
                                  and (mouse_frame[0] - tag_center_f[0]) ** 2 + (mouse_frame[1] - tag_center_f[1]) ** 2 < 48.0 ** 2)

                    # 状态提示文案
                    status_hint = None
                    if obs is not None and not is_kept:
                        status_hint = "[BA理论:实测已剔除]"
                    elif obs is None:
                        status_hint = "[BA理论:未检出/遮挡]"

                    studio.visualizer.render_tag_dual_prisms(
                        img=disp_frame,
                        ba_rvec=r_tag if ba_mode == "3d" else None,
                        ba_tvec=t_tag if ba_mode == "3d" else None,
                        obs_rvec=obs_r if (obs_mode == "3d" and succ_single) else None,
                        obs_tvec=obs_t if (obs_mode == "3d" and succ_single) else None,
                        tag_id=tid,
                        err_px=err_px,
                        err_mm=err_mm,
                        observed_corners=c_arr,
                        tag_status_hint=status_hint,
                        world_position_mm=(tags_meta.get(tid) or {}).get("position_mm"),
                        world_rpy_deg=(tags_meta.get(tid) or {}).get("rpy_deg"),
                        ba_center_xyz=(t_tag_center.tolist() if r_tag is not None else None),
                        obs_center_xyz=(obs_t.flatten().tolist() if obs_t is not None else None),
                        hovered=is_hovered
                    )
                    rendered_tids.add(tid)

            # 4. 2D 理论重投影框与残差矢量
            if ba_mode == "2d" and len(valid_obs) > 0:
                proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, studio.engine.camera_matrix, studio.engine.dist_coeffs)
                proj_flat = proj_pts.reshape((-1, 2))
                for i in range(len(valid_obs)):
                    p4 = proj_flat[i * 4:(i + 1) * 4].astype(np.int32)
                    cv2.polylines(disp_frame, [p4], isClosed=True, color=(0, 210, 255), thickness=1, lineType=cv2.LINE_AA)

                if obs_mode == "2d" and hasattr(studio.visualizer, "draw_reprojection_vectors"):
                    studio.visualizer.draw_reprojection_vectors(disp_frame, img_flat, proj_flat, scale_factor=40.0)

        # 4. 保底渲染：对所有提取到但未被 3D 棱柱覆盖的有效保留标靶，保底绘制 2D 实测角点多边形与编号标签
        if obs_mode != "off":
            for obs in observations:
                if not obs.get("keep", True) or is_frame_excluded:
                    continue
                tid = obs["tag_id"]
                if obs_mode == "2d" or tid not in rendered_tids:
                    pts = np.array(obs["corners"], dtype=np.int32).reshape((-1, 2))
                    cv2.polylines(disp_frame, [pts], isClosed=True, color=(0, 230, 80), thickness=2, lineType=cv2.LINE_AA)
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    in_map = (studio.get_tag_world_corners(tid) is not None)
                    tag_lbl = f"Tag #{tid}" if in_map else f"Tag #{tid} [未入图]"
                    cv2.putText(disp_frame, tag_lbl, (cx - 38, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 80), 2, cv2.LINE_AA)
                    rendered_tids.add(tid)

        # 5. 若处于病因切片诊断模式，叠加视野内预测但实测漏检的标靶框 (橙黄色矩形与 Tag 标注)
        if getattr(studio, "show_frame_diagnostics", False):
            diag = getattr(studio, "current_diagnostics", {})
            missing = diag.get("missing_projected_tags", []) or diag.get("missing_theoretical_tags", [])
            for m in missing:
                tid = m.get("tag_id")
                c_pts = m.get("proj_corners") or m.get("predicted_corners")
                if c_pts is not None and len(c_pts) == 4:
                    pts_i = np.array(c_pts, dtype=np.int32)
                    cv2.polylines(disp_frame, [pts_i], isClosed=True, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
                    mcx, mcy = int(np.mean(pts_i[:, 0])), int(np.mean(pts_i[:, 1]))
                    cv2.putText(disp_frame, f"? Tag #{tid} [漏检预测]", (mcx - 45, mcy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2, cv2.LINE_AA)

    def render_right_inspector(self, studio: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """右栏：精简属性与标靶残差清单/病因切片诊断双模面板 (瘦身宽度: 180px)"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x, y), (x, y + h), (50, 54, 66), 1)

        if not studio.image_files:
            return

        cur_file = studio.image_files[studio.current_img_idx]
        bname = os.path.basename(cur_file)
        meta = studio.frame_metrics_cache.get(bname, {})

        # 1. 顶部当前帧摘要卡片
        cv2.putText(canvas, bname, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        # 状态切换按钮 (保留 / 剔除)
        is_excl = meta.get("is_excluded", False)
        b_type = "danger" if is_excl else "success"
        b_label = "恢复此帧 (T)" if is_excl else "剔除此帧 (T)"
        draw_styled_button(canvas, (x + 8, y + 30, x + w - 8, y + 54), b_label,
                           mouse_pos=studio.mouse_pos, btn_type=b_type)
        studio.gui_buttons.append(("TOGGLE_FRAME_STATUS", (x + 8, y + 30, x + w - 8, y + 54), bname))

        # 底部动作区域基准 Y (预留 3 个紧凑按钮: 超精提取、病因诊断/常规面板、Robot 跟踪)
        diag_y = y + h - 88

        # 2. 中间区域：根据 show_frame_diagnostics 模式切换
        is_diag_mode = getattr(studio, "show_frame_diagnostics", False)
        if is_diag_mode:
            # 渲染【单帧深度切片病因诊断】面板
            diag_top_y = y + 62
            cv2.line(canvas, (x + 8, diag_top_y), (x + w - 8, diag_top_y), (45, 48, 58), 1)
            cv2.putText(canvas, "单帧病因切片诊断", (x + 8, diag_top_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)

            diag = getattr(studio, "current_diagnostics", {})
            cy = diag_top_y + 36

            # 指标1：清晰度 Laplace
            lap = diag.get("sharpness", diag.get("laplacian_var", 0.0))
            lap_g = diag.get("sharpness_grade", "")
            cv2.putText(canvas, f"清晰度: {lap:.1f} {lap_g}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 215), 1, cv2.LINE_AA)
            cy += 20

            # 指标2：对比度 RMS
            c_rms = diag.get("contrast", diag.get("contrast_rms", 0.0))
            c_g = diag.get("contrast_grade", "")
            cv2.putText(canvas, f"对比度: {c_rms:.1f} {c_g}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 215), 1, cv2.LINE_AA)
            cy += 20

            # 指标3：平均亮度 Mean
            m_lum = diag.get("brightness", diag.get("mean_intensity", 0.0))
            b_g = diag.get("brightness_grade", "")
            cv2.putText(canvas, f"亮度: {m_lum:.1f} {b_g}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 215), 1, cv2.LINE_AA)
            cy += 24

            # 拒检四边形候选
            rej_c = diag.get("rejected_quads_count", diag.get("false_rejections_count", 0))
            cv2.putText(canvas, f"畸变/超小候选: {rej_c}个", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 175, 185), 1, cv2.LINE_AA)
            cy += 20

            # 理论漏检标靶
            missing = diag.get("missing_projected_tags", []) or diag.get("missing_theoretical_tags", [])
            if missing:
                m_tids = ",".join([f"#{m['tag_id']}" for m in missing])
                cv2.putText(canvas, f"理论漏检: {len(missing)}个", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 140, 255), 1, cv2.LINE_AA)
                cy += 18
                cv2.putText(canvas, f"目标: {m_tids}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 180, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "理论漏检: 无 (全捕获)", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 100), 1, cv2.LINE_AA)

        else:
            # 渲染常规【标靶与残差清单】面板 (按照残差降序排序: 离差最大排在最上方)
            list_y = y + 62
            cv2.line(canvas, (x + 8, list_y), (x + w - 8, list_y), (45, 48, 58), 1)
            obs_list = meta.get("observations", [])
            tag_errors = meta.get("tag_errors", {})
            # FR-9.6 世界系坐标 (平差锚定后每枚标靶的 XYZ)
            tags_meta = (getattr(studio, "tags_map_data", {}) or {}).get("tags", {})
            if not tags_meta and hasattr(studio, "data_mgr"):
                tags_meta = (getattr(studio.data_mgr, "tags_map_data", {}) or {}).get("tags", {})
            cv2.putText(canvas, f"标靶残差+世界XYZ ({len(obs_list)}) 降序↓", (x + 8, list_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)

            # 按残差降序排序：离差最大的坏标靶置顶优先显示
            sorted_obs_list = sorted(
                obs_list,
                key=lambda o: (tag_errors.get(o.get("tag_id", -1), -1.0), -o.get("tag_id", 0)),
                reverse=True
            )

            row_y = list_y + 24
            row_h = 38
            for obs in sorted_obs_list:
                tid = obs["tag_id"]
                keep = obs.get("keep", True)
                err_val = tag_errors.get(tid, 0.0)

                rx1, ry1, rx2, ry2 = x + 6, row_y, x + w - 6, row_y + row_h - 2
                is_hover = (rx1 <= studio.mouse_pos[0] <= rx2 and ry1 <= studio.mouse_pos[1] <= ry2)
                bg_col = (34, 38, 48) if is_hover else (26, 28, 36)
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), bg_col, -1)
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (48, 52, 64), 1)
                studio.gui_buttons.append((f"TOGGLE_TAG_{tid}", (rx1, ry1, rx2, ry2), tid))

                dot_c = (0, 220, 80) if keep else (0, 0, 220)
                cv2.circle(canvas, (rx1 + 10, ry1 + 11), 3, dot_c, -1)

                t_col = (230, 230, 230) if keep else (120, 120, 120)
                cv2.putText(canvas, f"#{tid}", (rx1 + 18, ry1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, t_col, 1, cv2.LINE_AA)

                err_str = f"{err_val:.2f}px" if keep else "EXCL"
                if not keep:
                    err_c = (120, 120, 120)
                elif err_val > 1.0:
                    err_c = (0, 100, 255)  # 离差严重 (>1px) 鲜艳红色
                elif err_val > 0.5:
                    err_c = (0, 200, 255)  # 离差偏大 (>0.5px) 醒目金黄
                else:
                    err_c = (0, 230, 80)   # 优良 (<=0.5px) 荧光绿
                (ew, _), _ = cv2.getTextSize(err_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.putText(canvas, err_str, (rx2 - ew - 6, ry1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, err_c, 1, cv2.LINE_AA)

                # 第二行: FR-9.6 世界系坐标 XYZ (mm)
                rec = tags_meta.get(tid) or {}
                pos_mm = rec.get("position_mm")
                if pos_mm and len(pos_mm) >= 3:
                    xyz_str = f"X{pos_mm[0]:.0f} Y{pos_mm[1]:.0f} Z{pos_mm[2]:.0f} mm"
                    xyz_c = (140, 190, 220)
                else:
                    xyz_str = "XYZ: --"
                    xyz_c = (110, 115, 125)
                cv2.putText(canvas, xyz_str, (rx1 + 18, ry1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.36, xyz_c, 1, cv2.LINE_AA)

                row_y += row_h
                if row_y > diag_y - 12:
                    break

        # 3. 底部 3 个紧凑快捷动作
        cv2.line(canvas, (x + 8, diag_y), (x + w - 8, diag_y), (45, 48, 58), 1)

        # 按钮 1: 超精提取 (E)
        draw_styled_button(canvas, (x + 8, diag_y + 4, x + w - 8, diag_y + 28), "超精提取 (E)",
                           mouse_pos=studio.mouse_pos, btn_type="primary")
        studio.gui_buttons.append(("SUPER_EXTRACT_FRAME", (x + 8, diag_y + 4, x + w - 8, diag_y + 28), bname))

        # 按钮 2: 病因诊断 (D) / 常规面板 (D)
        d_lbl = "常规面板 (D)" if is_diag_mode else "病因诊断 (D)"
        d_typ = "normal" if is_diag_mode else "warning"
        draw_styled_button(canvas, (x + 8, diag_y + 32, x + w - 8, diag_y + 56), d_lbl,
                           mouse_pos=studio.mouse_pos, btn_type=d_typ)
        studio.gui_buttons.append(("DIAGNOSE_FRAME", (x + 8, diag_y + 32, x + w - 8, diag_y + 56), bname))

        # 按钮 3: Robot 在线跟踪
        draw_styled_button(canvas, (x + 8, diag_y + 60, x + w - 8, diag_y + 84), "Robot 跟踪",
                           mouse_pos=studio.mouse_pos, btn_type="success")
        studio.gui_buttons.append(("LAUNCH_TRACKER", (x + 8, diag_y + 60, x + w - 8, diag_y + 84), "LAUNCH_TRACKER"))

    def render_ba_loading_card(self, studio: Any, canvas: np.ndarray, w: int, h: int):
        """居中展示异步 BA 全局平差双轨进度卡片 (大阶段主进度条 + 求解器子进度条与实时收敛指标)"""
        card_w, card_h = 640, 162
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (0, 220, 255), 2)

        # 1. 主阶段总流程进度条
        pct = max(0.0, min(1.0, studio.ba_progress))
        pct_int = int(round(pct * 100))

        cv2.putText(canvas, "[BA] 全局平差整体收敛进度", (cx1 + 22, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA)

        bar_x1 = cx1 + 22
        bar_y1 = cy1 + 34
        bar_x2 = cx1 + card_w - 22
        bar_y2 = bar_y1 + 12
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (55, 62, 78), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 210, 255), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (180, 245, 255), 1)

        stage_txt = studio.ba_stage_text or "准备进入优化平差管线..."
        cv2.putText(canvas, stage_txt, (cx1 + 22, cy1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

        cv2.line(canvas, (cx1 + 22, cy1 + 74), (cx1 + card_w - 22, cy1 + 74), (45, 48, 60), 1)

        # 2. 求解器内部子阶段与迭代进度条
        sub_pct = max(0.0, min(1.0, studio.ba_sub_progress))
        sub_pct_int = int(round(sub_pct * 100))

        cv2.putText(canvas, "优化器实时迭代收敛监控 (Sub-Iteration)", (cx1 + 22, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 205, 215), 1, cv2.LINE_AA)
        sub_pct_str = f"{sub_pct_int}%"
        (spw, _), _ = cv2.getTextSize(sub_pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        cv2.putText(canvas, sub_pct_str, (cx1 + card_w - 22 - spw, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 180, 255), 2, cv2.LINE_AA)

        sbar_y1 = cy1 + 102
        sbar_y2 = sbar_y1 + 10
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (25, 28, 36), -1)
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (48, 54, 68), 1)

        sub_fill_w = int(bar_w * sub_pct)
        if sub_fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y2), (0, 160, 255), -1)
            cv2.line(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y1), (120, 220, 255), 1)

        sub_txt = studio.ba_sub_text or "等待当前阶段迭代步进推进..."
        cv2.putText(canvas, sub_txt, (cx1 + 22, cy1 + 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

    def render_extract_loading_card(self, studio: Any, canvas: np.ndarray, w: int, h: int):
        """工序 3 全量超精提取进行中：居中磨砂半透明高科技进度卡片"""
        card_w, card_h = 520, 110
        cx1 = (w - card_w) // 2
        cy1 = (h - card_h) // 2
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (20, 24, 32), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 200, 255), 2)

        pct = max(0.0, min(1.0, getattr(studio, "extract_progress", 0.0)))
        pct_int = int(round(pct * 100))

        cv2.putText(canvas, "工序 3: 全局全量图像超精重提取", (cx1 + 22, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA)

        bar_x1 = cx1 + 22
        bar_y1 = cy1 + 38
        bar_x2 = cx1 + card_w - 22
        bar_y2 = bar_y1 + 14
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (55, 62, 78), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 210, 255), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (180, 245, 255), 1)

        stage_txt = getattr(studio, "extract_stage_text", "") or "正在全量调用多尺度增强与正交亚像素精修..."
        cv2.putText(canvas, stage_txt, (cx1 + 22, cy1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

    def render_toast(self, studio: Any, canvas: np.ndarray, w: int, h: int, bot_h: int):
        (tw, _), _ = cv2.getTextSize(studio.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        tx1 = (w - tw) // 2 - 16
        ty1 = h - bot_h - 46
        tx2 = tx1 + tw + 32
        ty2 = ty1 + 32

        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (120, 30, 100), -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (220, 60, 180), 1)
        cv2.putText(canvas, studio.status_toast, (tx1 + 16, ty1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    def render_dropdown_popup(
        self,
        studio: Any,
        canvas: np.ndarray,
        pop_name: str,
        rect: Tuple[int, int, int, int],
        options: List[Tuple[str, str]],
        active_key: str
    ):
        """置顶悬浮下拉列表浮层"""
        rx1, ry1, rx2, ry2 = rect
        item_h = 32
        pop_w = max(rx2 - rx1, 190)
        pop_h = len(options) * item_h + 6
        pop_x1 = rx1
        pop_y1 = ry2 + 2
        pop_x2 = pop_x1 + pop_w
        pop_y2 = pop_y1 + pop_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), (0, 200, 255), 1)

        mx, my = studio.mouse_pos
        for idx, (opt_key, opt_label) in enumerate(options):
            iy1 = pop_y1 + 3 + idx * item_h
            iy2 = iy1 + item_h - 1
            is_active = (opt_key == active_key)
            is_hover = (pop_x1 <= mx <= pop_x2 and iy1 <= my <= iy2)

            if is_active:
                row_bg = (52, 45, 20)
                txt_col = (0, 230, 255)
            elif is_hover:
                row_bg = (36, 42, 56)
                txt_col = (255, 255, 255)
            else:
                row_bg = (22, 25, 32)
                txt_col = (190, 190, 190)

            cv2.rectangle(canvas, (pop_x1 + 3, iy1), (pop_x2 - 3, iy2), row_bg, -1)
            prefix = "✔ " if is_active else "  "
            (tw, th), _ = cv2.getTextSize(prefix + opt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(canvas, prefix + opt_label, (pop_x1 + 8, iy1 + (item_h + th) // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

            btn_id = f"DD_SELECT_{studio.active_dropdown}_{opt_key}"
            studio.gui_buttons.append((btn_id, (pop_x1, iy1, pop_x2, iy2), (studio.active_dropdown, opt_key)))

    def render_prune_ba_card(self, studio: Any, canvas: np.ndarray, w: int, h: int):
        """居中展示工序 5-Auto: 迭代残差剪枝平差运行中多轮收敛监控卡片 (集成实时多轮报告列表)"""
        card_w, card_h = 760, 360
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (255, 140, 0), 2)

        # 1. 标题与急停按钮
        r = getattr(studio.ba_runner, "prune_round", 1)
        max_r = getattr(studio.ba_runner, "max_prune_rounds", 10)
        title_txt = f"工序 5-Auto: 迭代残差剪枝平差监控 (第 {r}/{max_r} 轮)..."
        cv2.putText(canvas, title_txt, (cx1 + 20, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # 急停按钮 (右上角)
        btn_w, btn_h = 105, 24
        btn_x1 = cx1 + card_w - btn_w - 18
        btn_y1 = cy1 + 12
        btn_x2 = btn_x1 + btn_w
        btn_y2 = btn_y1 + btn_h
        draw_styled_button(canvas, (btn_x1, btn_y1, btn_x2, btn_y2), "急停 (Space)",
                           mouse_pos=studio.mouse_pos, btn_type="danger")
        studio.gui_buttons.append(("STOP_PRUNE", (btn_x1, btn_y1, btn_x2, btn_y2), "STOP_PRUNE"))

        # 2. 进度条与百分比
        pct = max(0.0, min(1.0, studio.ba_progress))
        pct_int = int(round(pct * 100))
        pct_str = f"{pct_int}%"
        (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        cv2.putText(canvas, pct_str, (btn_x1 - pw - 14, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 180, 0), 2, cv2.LINE_AA)

        bar_x1 = cx1 + 20
        bar_y1 = cy1 + 42
        bar_x2 = cx1 + card_w - 20
        bar_y2 = bar_y1 + 10
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 72, 88), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (255, 140, 0), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (255, 210, 140), 1)

        # 3. 统计指标胶囊栏 (淘汰数、基准RMSE、当前RMSE、累计改善)
        cap_y1 = cy1 + 58
        cap_y2 = cap_y1 + 28
        cv2.rectangle(canvas, (bar_x1, cap_y1), (bar_x2, cap_y2), (25, 28, 36), -1)
        cv2.rectangle(canvas, (bar_x1, cap_y1), (bar_x2, cap_y2), (48, 54, 68), 1)

        history: List[Dict[str, Any]] = getattr(studio.ba_runner, "prune_history", [])
        total_pruned = sum(len(item.get("pruned", [])) for item in history)
        init_rmse = getattr(studio.ba_runner, "initial_rmse", studio.data_mgr.global_rmse)
        curr_rmse = studio.data_mgr.global_rmse
        cum_delta = max(0.0, init_rmse - curr_rmse) if init_rmse > 0 else 0.0
        cum_pct = (cum_delta / init_rmse * 100.0) if init_rmse > 0 else 0.0

        txt_p = f"累计淘汰: {total_pruned} 个"
        txt_i = f"初始基准: {init_rmse:.2f}px"
        txt_c = f"当前残差: {curr_rmse:.2f}px"
        txt_d = f"累计改善: ↓{cum_delta:.2f}px ({cum_pct:.1f}%)"

        cv2.putText(canvas, txt_p, (bar_x1 + 14, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt_i, (bar_x1 + 160, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 185, 195), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt_c, (bar_x1 + 330, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 210, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt_d, (bar_x1 + 500, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 120), 1, cv2.LINE_AA)

        # 4. 当前运行主阶段与动态细节
        stg_txt = studio.ba_stage_text or "智能迭代剪枝平差管线推进中..."
        sub_txt = studio.ba_sub_text or "正在执行全场景 BA 平差与共视安全守门..."
        cv2.putText(canvas, stg_txt, (cx1 + 20, cy1 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, sub_txt, (cx1 + 20, cy1 + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 190, 80), 1, cv2.LINE_AA)

        # 5. 【核心实时报告表格】 (Live Settlement Table)
        tbl_x1 = bar_x1
        tbl_y1 = cy1 + 134
        tbl_w = bar_w
        th_h = 24
        tr_h = 24

        # 表头
        cv2.rectangle(canvas, (tbl_x1, tbl_y1), (tbl_x1 + tbl_w, tbl_y1 + th_h), (34, 38, 48), -1)
        cv2.rectangle(canvas, (tbl_x1, tbl_y1), (tbl_x1 + tbl_w, tbl_y1 + th_h), (55, 62, 78), 1)

        c_round_w = 60
        c_item_w = 320
        c_before_w = 110
        c_after_w = 110
        c_delta_w = tbl_w - c_round_w - c_item_w - c_before_w - c_after_w

        tx0 = tbl_x1 + 8
        tx1 = tbl_x1 + c_round_w
        tx2 = tx1 + c_item_w
        tx3 = tx2 + c_before_w
        tx4 = tx3 + c_after_w

        cv2.putText(canvas, "轮次", (tx0, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        cv2.putText(canvas, "淘汰坏样本 (图像 / Tag / 离差)", (tx1 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        cv2.putText(canvas, "平差前残差", (tx2 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        cv2.putText(canvas, "平差后残差", (tx3 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        cv2.putText(canvas, "进步幅度", (tx4 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)

        # 动态行组合: 历史完成轮次 + 当前正在求解轮次
        display_rows = []
        for h_item in history:
            rnd = f"#{h_item.get('round', 1)}"
            pruned_info = ", ".join([f"{bn} #{tid} ({err:.1f}px)" for bn, tid, err, _ in h_item.get("pruned", [])])
            b_rmse = f"{h_item.get('rmse_before', 0.0):.2f}px"
            a_rmse = f"{h_item.get('rmse_after', 0.0):.2f}px"
            d_rmse = h_item.get('delta_rmse', 0.0)
            d_txt = f"↓{d_rmse:+.2f}px" if d_rmse >= 0 else f"{d_rmse:+.2f}px"
            display_rows.append((rnd, pruned_info, b_rmse, a_rmse, d_txt, False))

        # 当前正在求解的进行中行
        curr_target = getattr(studio.ba_runner, "current_pruning_target", "")
        if curr_target:
            curr_row = (f"#{r}", curr_target, f"{curr_rmse:.2f}px", "--", "求解中...", True)
            display_rows.append(curr_row)

        # 仅展示最新的 5 行，保证排版整洁
        rows_to_show = display_rows[-5:] if len(display_rows) > 5 else display_rows
        max_rows = 5
        curr_row_y = tbl_y1 + th_h

        if not rows_to_show:
            cv2.rectangle(canvas, (tbl_x1, curr_row_y), (tbl_x1 + tbl_w, curr_row_y + tr_h * 2), (20, 22, 28), -1)
            cv2.rectangle(canvas, (tbl_x1, curr_row_y), (tbl_x1 + tbl_w, curr_row_y + tr_h * 2), (45, 50, 62), 1)
            cv2.putText(canvas, "正在执行首轮共视拓扑分析与基准残差排查，即将生成实时对比明细...",
                        (tbl_x1 + 18, curr_row_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 150, 165), 1, cv2.LINE_AA)
            curr_row_y += tr_h * 2
        else:
            for idx, (rnd_s, item_s, b_s, a_s, d_s, is_active) in enumerate(rows_to_show):
                ry1 = curr_row_y + idx * tr_h
                ry2 = ry1 + tr_h
                row_bg = (32, 28, 20) if is_active else ((24, 27, 34) if idx % 2 == 0 else (20, 22, 28))
                row_border = (255, 140, 0) if is_active else (40, 44, 54)

                cv2.rectangle(canvas, (tbl_x1, ry1), (tbl_x1 + tbl_w, ry2), row_bg, -1)
                cv2.rectangle(canvas, (tbl_x1, ry1), (tbl_x1 + tbl_w, ry2), row_border, 1)

                col_txt = (255, 180, 0) if is_active else (210, 215, 225)
                delta_col = (0, 240, 120) if not is_active else (255, 200, 80)

                # 限制文字长度避免溢出
                s_item = item_s if len(item_s) <= 38 else item_s[:35] + "..."

                cv2.putText(canvas, rnd_s, (tx0, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_txt, 1, cv2.LINE_AA)
                cv2.putText(canvas, s_item, (tx1 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col_txt, 1, cv2.LINE_AA)
                cv2.putText(canvas, b_s, (tx2 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 170, 185), 1, cv2.LINE_AA)
                cv2.putText(canvas, a_s, (tx3 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_txt, 1, cv2.LINE_AA)
                cv2.putText(canvas, d_s, (tx4 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, delta_col, 1 if is_active else 2, cv2.LINE_AA)

        # 6. 底栏停机说明
        tip_txt = "收敛准则: 单轮改善 < 0.010 px 触发边际最优收敛 | 共视拓扑守门确保几何不退化 | 随时按 Space 急停"
        cv2.putText(canvas, tip_txt, (cx1 + 20, cy1 + card_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 140, 155), 1, cv2.LINE_AA)

    def render_prune_settlement_card(self, studio: Any, canvas: np.ndarray, w: int, h: int):
        """居中展示工序 5-Auto: 智能剪枝平差结算单对比卡片 (支持一键采纳或无损撤销)"""
        s_data = getattr(studio, "prune_settlement_data", None)
        if not s_data:
            return

        card_w, card_h = 700, 340
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (0, 230, 100), 2)

        # 1. 顶部标题与收敛徽章
        cv2.putText(canvas, "智能残差剪枝平差结算单 (Auto-Prune Settlement)", (cx1 + 22, cy1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        reason = s_data.get("stop_reason", "最优收敛")
        (rw, _), _ = cv2.getTextSize(f"[{reason}]", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, f"[{reason}]", (cx1 + card_w - 22 - rw, cy1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)

        cv2.line(canvas, (cx1 + 22, cy1 + 44), (cx1 + card_w - 22, cy1 + 44), (50, 58, 72), 1)

        # 2. 关键前后指标对比
        init_rmse = s_data.get("initial_rmse", 0.0)
        final_rmse = s_data.get("final_rmse", 0.0)
        init_mm = s_data.get("initial_mm", 0.0)
        final_mm = s_data.get("final_mm", 0.0)
        rounds = s_data.get("rounds_executed", 0)
        pruned_cnt = s_data.get("total_pruned_count", 0)

        drop_pct = ((init_rmse - final_rmse) / max(0.001, init_rmse)) * 100.0 if init_rmse > 0 else 0.0

        y_c = cy1 + 72
        # RMSE 对比
        cv2.putText(canvas, "全局像面 RMSE:", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
        rmse_str = f"{init_rmse:.3f} px  ->  {final_rmse:.3f} px"
        cv2.putText(canvas, rmse_str, (cx1 + 175, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 100), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"(误差显著降低 {drop_pct:.1f}%)", (cx1 + 445, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        y_c += 28
        # 物理毫米对比
        cv2.putText(canvas, "空间物理偏差 (中位):", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
        mm_str = f"{init_mm:.2f} mm  ->  {final_mm:.2f} mm"
        cv2.putText(canvas, mm_str, (cx1 + 175, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 100), 2, cv2.LINE_AA)

        y_c += 28
        # 轮次与剔除汇总
        cv2.putText(canvas, f"迭代执行: {rounds} 轮", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 185, 195), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"累计淘汰外点: {pruned_cnt} 个 (已受共视拓扑严格保护)", (cx1 + 175, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 0), 1, cv2.LINE_AA)

        y_c += 16
        cv2.line(canvas, (cx1 + 22, y_c), (cx1 + card_w - 22, y_c), (45, 52, 65), 1)
        y_c += 20

        # 3. 逐轮剔除明细 (最多展示最近 3 轮)
        cv2.putText(canvas, "各轮剪枝与收敛明细:", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 165, 175), 1, cv2.LINE_AA)
        y_c += 18
        hist = s_data.get("history", [])
        show_hist = hist[-3:] if len(hist) > 3 else hist
        for h_item in show_hist:
            r_num = h_item.get("round", 1)
            p_list = h_item.get("pruned", [])
            p_str = ", ".join([f"{bname}的#{tid}({err:.2f}px)" for bname, tid, err, _ in p_list])
            d_rmse = h_item.get("delta_rmse", 0.0)
            a_rmse = h_item.get("rmse_after", 0.0)
            log_line = f"轮次 #{r_num}: 淘汰 [{p_str}] -> RMSE降至 {a_rmse:.3f}px (改善: {d_rmse:.3f}px)"
            cv2.putText(canvas, log_line, (cx1 + 32, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 210, 220), 1, cv2.LINE_AA)
            y_c += 20

        # 4. 底部决策操作与提示
        card_hint = "左侧列表已自动展开逐帧多轮残差演进矩阵大表 (按 X 键可自由收放)"
        cv2.putText(canvas, card_hint, (cx1 + 24, cy1 + card_h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)

        btn_y1 = cy1 + card_h - 44
        btn_y2 = btn_y1 + 32

        # 采纳按钮 (Enter)
        b1_w = 200
        b1_x1 = cx1 + 40
        b1_x2 = b1_x1 + b1_w
        draw_styled_button(canvas, (b1_x1, btn_y1, b1_x2, btn_y2), "采纳成果 (Enter)",
                           mouse_pos=studio.mouse_pos, btn_type="success")
        studio.gui_buttons.append(("ACCEPT_PRUNE", (b1_x1, btn_y1, b1_x2, btn_y2), "ACCEPT_PRUNE"))

        # 导出报告按钮 (R)
        b2_w = 190
        b2_x1 = b1_x2 + 25
        b2_x2 = b2_x1 + b2_w
        draw_styled_button(canvas, (b2_x1, btn_y1, b2_x2, btn_y2), "导出质检单 (R)",
                           mouse_pos=studio.mouse_pos, btn_type="primary")
        studio.gui_buttons.append(("EXPORT_REPORT", (b2_x1, btn_y1, b2_x2, btn_y2), "EXPORT_REPORT"))

        # 撤销还原按钮 (Esc)
        b3_w = 170
        b3_x1 = b2_x2 + 25
        b3_x2 = b3_x1 + b3_w
        draw_styled_button(canvas, (b3_x1, btn_y1, b3_x2, btn_y2), "撤销还原 (Esc)",
                           mouse_pos=studio.mouse_pos, btn_type="danger")
        studio.gui_buttons.append(("UNDO_PRUNE", (b3_x1, btn_y1, b3_x2, btn_y2), "UNDO_PRUNE"))
