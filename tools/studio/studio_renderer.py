"""
AprilTag 离线标定工作站 - UI 界面渲染器 (StudioUIRenderer, 核心调度类)
================================================================================
负责工作站现代深色全景三栏界面的整体排版调度, 并直接承担以下分区绘制：
1. 顶栏 (Top Navigation Bar: LOGO + 全局快捷动作按钮, Dashboard 同源风格)
2. 底栏 (Bottom Status Bar: 放行门限 / 拓扑连通度 / 采图与精度状态指标)
3. 浮层 (科技感居中异步 BA 双轨进度卡片、Toast 提示、置顶下拉菜单、Hover 气泡)

拆分结构 (上帝文件拆分重构, 对外行为与拆分前完全一致)：
- tools/studio/studio_ui_common.py   共享常量 (下拉选项) 与模块级按钮绘制函数
- tools/studio/studio_frame_list.py  StudioFrameListMixin   左栏帧列表/矩阵宽表
- tools/studio/studio_center_view.py StudioCenterViewMixin  中栏视口与 3D 棱柱叠加
- tools/studio/studio_inspector.py   StudioInspectorMixin   右栏检视与剪枝卡片
本类通过多继承组合三个 Mixin, 跨分区方法调用由 MRO 解析；
外部模块仍通过 self.ui_renderer.<method> 唯一通道访问, 并可从本模块
re-import 共享常量与绘制函数 (保持既有导入路径兼容)。
"""

import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import draw_text, get_cached_font, measure_text, put_text

# 共享常量与模块级绘制函数已迁移至 studio_ui_common, 此处 re-import 保持
# 既有外部导入路径 (from tools.studio.studio_renderer import ...) 兼容可用。
from src.utils.gui_theme import GuiTheme
from src.utils.gui_components import (
    draw_dashboard_button,
    draw_dropdown_button,
    render_dropdown_popup as common_render_dropdown_popup,
)
from tools.studio.studio_ui_common import (
    VIEW_MODE_OPTIONS,
    FILTER_MODE_OPTIONS,
    SORT_MODE_OPTIONS,
    BA_VIEW_OPTIONS,
    OBS_VIEW_OPTIONS,
)
from tools.studio.studio_frame_list import StudioFrameListMixin
from tools.studio.studio_center_view import StudioCenterViewMixin
from tools.studio.studio_inspector import StudioInspectorMixin


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
        "下游 (capture_wizard / tag_studio / robot_tracker)",
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


class StudioUIRenderer(StudioFrameListMixin, StudioCenterViewMixin, StudioInspectorMixin):
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
        put_text(canvas, "OFFLINE STUDIO", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
        (logo_w, _), _ = measure_text("OFFLINE STUDIO", cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)

        # 2. LOGO 右侧紧邻的全局快捷动作按钮
        mx, my = studio.mouse_pos
        btn_y_top, btn_y_bot = 7, top_h - 7
        bx = 16 + logo_w + 20

        # 0. 选择场景下拉框 (首要核心位置)
        sc_w = 145
        cur_sc_label = getattr(studio, "current_scene_name", "默认场景")
        is_sc_open = (studio.active_dropdown == "SCENE_DROPDOWN")
        draw_dropdown_button(canvas, (bx, btn_y_top, bx + sc_w, btn_y_bot), f"场景: {cur_sc_label}",
                             is_open=is_sc_open, mouse_pos=(mx, my),
                             theme_color=(0, 255, 180))
        studio.dropdown_boxes["SCENE_DROPDOWN"] = {
            "rect": (bx, btn_y_top, bx + sc_w, btn_y_bot),
            "options": getattr(studio, "scene_options", []),
            "active_key": getattr(studio, "current_scene_id", "")
        }
        studio.gui_buttons.append(("TOGGLE_SCENE_DROPDOWN", (bx, btn_y_top, bx + sc_w, btn_y_bot), "SCENE_DROPDOWN"))
        bx += sc_w + 5

        # 0.5. [绘制XY平面] 透视网格开关与 [Z轴特殊点] 下拉选择 (移植自在线跟踪)
        xy_on = getattr(studio, "show_xy_plane_on", False)
        xy_lbl = "√ XY平面" if xy_on else "绘制XY平面"
        xy_accent = (0, 255, 180) if xy_on else None
        xy_w = 88
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + xy_w, btn_y_bot), xy_lbl,
                              mouse_pos=(mx, my), accent=xy_accent)
        studio.gui_buttons.append(("TOGGLE_DRAW_XY_PLANE", (bx, btn_y_top, bx + xy_w, btn_y_bot), "TOGGLE_DRAW_XY_PLANE"))
        bx += xy_w + 5

        # Z 轴特殊点下拉按钮 (显示当前选定高度或特殊点)
        z_w = 120
        cur_z_lbl = studio.get_current_plane_z_label() if hasattr(studio, "get_current_plane_z_label") else "Z轴特殊点"
        is_z_open = (studio.active_dropdown == "PLANE_Z_DROPDOWN")
        draw_dropdown_button(canvas, (bx, btn_y_top, bx + z_w, btn_y_bot), cur_z_lbl,
                             is_open=is_z_open, mouse_pos=(mx, my),
                             theme_color=(0, 220, 255) if xy_on else (140, 160, 180))
        plane_opts = [(str(val) if val is not None else "NONE", lbl)
                      for val, lbl in studio.get_plane_z_options()] if hasattr(studio, "get_plane_z_options") else []
        active_z_key = str(studio.plane_z) if (xy_on and hasattr(studio, "plane_z")) else "NONE"
        studio.dropdown_boxes["PLANE_Z_DROPDOWN"] = {
            "rect": (bx, btn_y_top, bx + z_w, btn_y_bot),
            "options": plane_opts,
            "active_key": active_z_key
        }
        studio.gui_buttons.append(("TOGGLE_PLANE_Z_DROPDOWN", (bx, btn_y_top, bx + z_w, btn_y_bot), "PLANE_Z_DROPDOWN"))
        bx += z_w + 5

        # 1. 全局全量超精提取 (清空旧角点并从头重提取)
        ext_w = 85
        is_ext = getattr(studio, "is_extracting_all", False)
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + ext_w, btn_y_bot),
                              "提取中..." if is_ext else "超精提取",
                              mouse_pos=(mx, my), is_running=is_ext)
        studio.gui_buttons.append(("SUPER_EXTRACT_ALL", (bx, btn_y_top, bx + ext_w, btn_y_bot), "SUPER_EXTRACT_ALL"))
        bx += ext_w + 5

        # 2. [B] 全局平差
        ba_w = 88
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + ba_w, btn_y_bot),
                              "平差中..." if studio.is_ba_running else "全局平差",
                              mouse_pos=(mx, my), is_running=studio.is_ba_running)
        studio.gui_buttons.append(("RUN_BA", (bx, btn_y_top, bx + ba_w, btn_y_bot), "RUN_BA"))
        bx += ba_w + 5

        # 3. [A] 智能残差剪枝平差
        prune_w = 88
        is_prune = getattr(studio, "is_auto_pruning", False)
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + prune_w, btn_y_bot),
                              "剪枝中..." if is_prune else "剪枝平差",
                              mouse_pos=(mx, my), is_running=is_prune)
        studio.gui_buttons.append(("RUN_AUTO_PRUNE_BA", (bx, btn_y_top, bx + prune_w, btn_y_bot), "RUN_AUTO_PRUNE_BA"))
        bx += prune_w + 5

        # 4. [M] 保存/发布地图
        s_w = 86
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + s_w, btn_y_bot), "保存地图",
                              mouse_pos=(mx, my), accent=(0, 215, 90))
        studio.gui_buttons.append(("SAVE_MAP", (bx, btn_y_top, bx + s_w, btn_y_bot), "SAVE_MAP"))
        bx += s_w + 5

        # 5. [P] 全程/全量精度体检重算
        p_w = 86
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + p_w, btn_y_bot), "全量体检",
                              mouse_pos=(mx, my))
        studio.gui_buttons.append(("RECOMPUTE_METRICS", (bx, btn_y_top, bx + p_w, btn_y_bot), "RECOMPUTE_METRICS"))
        bx += p_w + 5

        # 6. [R] 导出质检报告
        r_w = 86
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + r_w, btn_y_bot), "导出报告",
                              mouse_pos=(mx, my))
        studio.gui_buttons.append(("EXPORT_REPORT", (bx, btn_y_top, bx + r_w, btn_y_bot), "EXPORT_REPORT"))
        bx += r_w + 5

        # 7. 复位保留 (一键恢复所有剔除的观测为有效)
        rst_keep_w = 76
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + rst_keep_w, btn_y_bot), "复位保留",
                              mouse_pos=(mx, my))
        studio.gui_buttons.append(("RESET_KEEP_ALL", (bx, btn_y_top, bx + rst_keep_w, btn_y_bot), "RESET_KEEP_ALL"))
        bx += rst_keep_w + 5

        # 8. 复位地图 (清空已知平差地图)
        rst_map_w = 76
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + rst_map_w, btn_y_bot), "复位地图",
                              mouse_pos=(mx, my), accent=(70, 60, 210))
        studio.gui_buttons.append(("RESET_MAP", (bx, btn_y_top, bx + rst_map_w, btn_y_bot), "RESET_MAP"))

        # 9. 右侧 [Q] 退出工作台 (最右侧退出不动)
        exit_w = 85
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

        put_text(canvas, "[BA] 全局平差整体收敛进度", (cx1 + 22, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = measure_text(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        put_text(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 26),
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
        put_text(canvas, stage_txt, (cx1 + 22, cy1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

        cv2.line(canvas, (cx1 + 22, cy1 + 74), (cx1 + card_w - 22, cy1 + 74), (45, 48, 60), 1)

        # 2. 求解器内部子阶段与迭代进度条
        sub_pct = max(0.0, min(1.0, studio.ba_sub_progress))
        sub_pct_int = int(round(sub_pct * 100))

        put_text(canvas, "优化器实时迭代收敛监控 (Sub-Iteration)", (cx1 + 22, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 205, 215), 1, cv2.LINE_AA)
        sub_pct_str = f"{sub_pct_int}%"
        (spw, _), _ = measure_text(sub_pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        put_text(canvas, sub_pct_str, (cx1 + card_w - 22 - spw, cy1 + 94),
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
        put_text(canvas, sub_txt, (cx1 + 22, cy1 + 132),
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

        put_text(canvas, "工序 3: 全局全量图像超精重提取", (cx1 + 22, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = measure_text(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        put_text(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 28),
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
        put_text(canvas, stage_txt, (cx1 + 22, cy1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

    def render_toast(self, studio: Any, canvas: np.ndarray, w: int, h: int, bot_h: int):
        (tw, _), _ = measure_text(studio.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        tx1 = (w - tw) // 2 - 16
        ty1 = h - bot_h - 46
        tx2 = tx1 + tw + 32
        ty2 = ty1 + 32

        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (120, 30, 100), -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (220, 60, 180), 1)
        put_text(canvas, studio.status_toast, (tx1 + 16, ty1 + 21),
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
        """置顶悬浮下拉列表浮层 (统一委托给公共 gui_components)"""
        reg_btns = common_render_dropdown_popup(
            canvas,
            anchor_rect=rect,
            options=options,
            active_key=active_key,
            btn_prefix=f"DD_SELECT_{pop_name}_",
            item_h=32,
            min_width=190,
        )
        for _, item_rect, opt_key in reg_btns:
            btn_id = f"DD_SELECT_{pop_name}_{opt_key}"
            studio.gui_buttons.append((btn_id, item_rect, (pop_name, opt_key)))
