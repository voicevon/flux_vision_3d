#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) - 界面渲染模块
=====================================================
负责真矢量自适应画布重绘、三栏布局面板、视口切片映射、目标卡片与置顶菜单渲染。
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.gui_theme import GuiTheme
from src.utils.gui_components import draw_dropdown_button, render_dropdown_popup
from src.utils.text_rendering import draw_text, measure_text
from tools.asparagus_pose_studio.data_io import BASE_W, CALIB_LABELS


class AsparagusPoseStudioRenderer:
    """芦笋位姿工作室专属 UI 渲染器"""

    @staticmethod
    def compute_metrics(canvas_w: int) -> Dict[str, Any]:
        """计算物理画布与逻辑基准尺寸的动态缩放度量"""
        s = canvas_w / BASE_W
        return {
            "s": s,
            "L": int(12 * s),            # 全局左边距
            "list_w": int(140 * s),      # 左侧样本列表宽
            "right_w": int(236 * s),     # 右侧结果面板宽
            "header_h": int(52 * s),
            "bottom_h": int(46 * s),
            "row_h": int(32 * s),        # 样本行高
            "btn_h": int(30 * s),
            "fs_title": max(14, int(20 * s)),
            "fs_sub": max(10, int(12 * s)),
            "fs_body": max(11, int(13 * s)),
            "fs_small": max(9, int(11 * s)),
            "fs_gcode": max(9, int(11 * s)),
        }

    @staticmethod
    def compute_panels(canvas_w: int, canvas_h: int, m: Dict[str, Any]):
        """计算三栏面板物理矩形: (样本列表, 图像视口, 右侧结果)"""
        top = m["header_h"]
        bottom = canvas_h - m["bottom_h"]
        x0 = m["L"]
        x1 = x0 + m["list_w"]
        x4 = canvas_w - m["L"]
        x3 = x4 - m["right_w"]
        return (
            (x0, top, x1, bottom),
            (x1 + int(10 * m["s"]), top, x3 - int(10 * m["s"]), bottom),
            (x3, top, x4, bottom)
        )

    @staticmethod
    def draw_panel_bg(canvas: np.ndarray, rect: Tuple[int, int, int, int], title: str, m: Dict[str, Any]) -> int:
        """绘制带标题栏的标准卡片面板背景，返回内容区域起始 Y 坐标"""
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.CARD_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)
        header_h = int(26 * m["s"])
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + header_h), (26, 30, 38), -1)
        draw_text(canvas, title, (x1 + int(8 * m["s"]), y1 + int(6 * m["s"])), m["fs_body"],
                  GuiTheme.ACCENT, bold=True)
        return y1 + int(30 * m["s"])

    @staticmethod
    def draw_button(
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        mouse_pos: Tuple[int, int],
        m: Dict[str, Any],
        enabled: bool = True,
        active: bool = False
    ) -> bool:
        """绘制标准功能按钮并检测悬停高亮"""
        x1, y1, x2, y2 = rect
        mx, my = mouse_pos
        hover = enabled and (x1 <= mx <= x2 and y1 <= my <= y2)

        if not enabled:
            bg, border, col = GuiTheme.BTN_DISABLED_BG, GuiTheme.BTN_DISABLED_BORDER, GuiTheme.TEXT_DISABLED
        elif active:
            bg, border, col = GuiTheme.CARD_SEL, GuiTheme.BORDER_SEL, GuiTheme.WHITE
        elif hover:
            if "退出" in label:
                bg, border, col = (45, 38, 75), (80, 80, 220), (230, 230, 255)
            else:
                bg, border, col = GuiTheme.BTN_HOVER, GuiTheme.BORDER_HOVER, GuiTheme.BTN_TEXT_HOVER
        else:
            if "退出" in label:
                bg, border, col = GuiTheme.BTN, (60, 60, 110), GuiTheme.BTN_TEXT
            else:
                bg, border, col = GuiTheme.BTN, GuiTheme.BTN_BORDER, GuiTheme.BTN_TEXT

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if hover else 1)
        (tw, th), _ = measure_text(label, font_size=m["fs_sub"])
        draw_text(
            canvas, label,
            (x1 + ((x2 - x1) - tw) // 2, y1 + ((y2 - y1) - th) // 2),
            m["fs_sub"], col,
            bold=(hover and GuiTheme.BTN_BEHAVIOR["HOVER_BOLD"]) or active
        )
        return enabled

    @staticmethod
    def draw_view_pill(
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        is_active: bool,
        mouse_pos: Tuple[int, int],
        m: Dict[str, Any]
    ):
        """绘制算法中间步骤药丸按钮 (互斥激活、发光小圆点指示)"""
        x1, y1, x2, y2 = rect
        mx, my = mouse_pos
        hover = (x1 <= mx <= x2 and y1 <= my <= y2)

        if is_active:
            bg = (32, 68, 48)            # 翡翠绿底色
            border = (0, 235, 120)        # 高光绿边框
            text_col = (255, 255, 255)
        elif hover:
            bg = (34, 38, 46)
            border = (110, 130, 155)
            text_col = (235, 240, 245)
        else:
            bg = (24, 28, 34)
            border = (52, 58, 68)
            text_col = (165, 175, 185)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if is_active else 1)

        (tw, th), _ = measure_text(label, font_size=m["fs_sub"])
        if is_active:
            dot_x = x1 + int(10 * m["s"])
            dot_y = y1 + (y2 - y1) // 2
            cv2.circle(canvas, (dot_x, dot_y), int(3.5 * m["s"]), (0, 235, 120), -1)
            cv2.circle(canvas, (dot_x, dot_y), int(5 * m["s"]), (0, 235, 120), 1)
            tx = dot_x + int(8 * m["s"])
        else:
            tx = x1 + ((x2 - x1) - tw) // 2

        ty = y1 + ((y2 - y1) - th) // 2
        draw_text(canvas, label, (tx, ty), m["fs_sub"], text_col, bold=is_active)

    @classmethod
    def render_scene(
        cls,
        canvas: np.ndarray,
        app_state: Any
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        """
        完整渲染场景画布并返回可交互区域映射:
        返回 (buttons, sample_rows, result_rows, dd_items)
        """
        buttons = []
        sample_rows = []
        result_rows = []
        dd_items = []

        m = cls.compute_metrics(canvas.shape[1])
        W, H = canvas.shape[1], canvas.shape[0]

        # 1. 标题文字
        draw_text(canvas, "芦笋位姿工作室 - Asparagus Pose Studio",
                  (m["L"], int(16 * m["s"])), m["fs_title"], GuiTheme.TEXT, bold=True)

        list_p, img_p, right_p = cls.compute_panels(W, H, m)

        # 2. 渲染三栏内容
        cls._draw_sample_list(canvas, m, list_p, app_state, sample_rows)
        cls._draw_image_area(canvas, m, img_p, app_state)
        cls._draw_result_panel(canvas, m, right_p, app_state, result_rows)

        # 3. 顶部右侧主要功能按钮组
        btn_w = int(112 * m["s"])
        bx = W - m["L"]
        top_btns = [
            ("退出 [X]", "exit", True),
            ("导出G-code [E]", "export", bool(app_state.gcode_text)),
            ("识别定位 [空格]", "analyze", bool(app_state.samples and app_state.sel_idx >= 0)),
        ]
        for label, _act, enabled in top_btns:
            bx -= btn_w + int(8 * m["s"])
            rect = (bx, int(14 * m["s"]), bx + btn_w, int(14 * m["s"]) + m["btn_h"])
            cls.draw_button(canvas, rect, label, app_state.mouse_pos, m, enabled=enabled)
            if enabled:
                buttons.append((rect, ("btn", label)))

        # 4. 动态算法步骤药丸 (由 pipeline.get_steps() 动态产生)
        steps = app_state.pipeline.get_steps() if app_state.pipeline else []
        pill_w = int(58 * m["s"])
        for s_obj in reversed(steps):
            bx -= pill_w + int(4 * m["s"])
            pill_rect = (bx, int(14 * m["s"]), bx + pill_w, int(14 * m["s"]) + m["btn_h"])
            cls.draw_view_pill(
                canvas, pill_rect, s_obj.name,
                is_active=(app_state.active_step_key == s_obj.key),
                mouse_pos=app_state.mouse_pos, m=m
            )
            buttons.append((pill_rect, ("set_step", s_obj.key)))

        if steps:
            bx -= int(38 * m["s"])
            draw_text(canvas, "显示:", (bx, int(22 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)

        # 5. 算法路线下拉按钮
        pipe_w = int(185 * m["s"])
        bx -= pipe_w + int(8 * m["s"])
        pipeline_rect = (bx, int(14 * m["s"]), bx + pipe_w, int(14 * m["s"]) + m["btn_h"])
        app_state._pipeline_rect = pipeline_rect
        is_pipe_open = (app_state.active_dropdown == "PIPELINE_DROPDOWN")
        curr_pipe_name = app_state.current_pipeline_name
        short_pipe = curr_pipe_name.split(":")[0] if ":" in curr_pipe_name else curr_pipe_name
        draw_dropdown_button(canvas, pipeline_rect, f"算法: {short_pipe}", is_pipe_open, app_state.mouse_pos, font_size=m["fs_sub"])
        buttons.append((pipeline_rect, ("toggle_dd", "PIPELINE_DROPDOWN")))

        # 6. 工位地图下拉按钮
        sc_w = int(160 * m["s"])
        bx -= sc_w + int(8 * m["s"])
        workspace_rect = (bx, int(14 * m["s"]), bx + sc_w, int(14 * m["s"]) + m["btn_h"])
        app_state._workspace_rect = workspace_rect
        is_sc_open = (app_state.active_dropdown == "WORKSPACE_DROPDOWN")
        draw_dropdown_button(canvas, workspace_rect, f"地图: {app_state.current_workspace_name}", is_sc_open, app_state.mouse_pos, font_size=m["fs_sub"])
        buttons.append((workspace_rect, ("toggle_dd", "WORKSPACE_DROPDOWN")))

        # 7. 底部状态栏
        yb = H - m["bottom_h"] + int(8 * m["s"])
        calib = CALIB_LABELS.get(
            app_state.targets[0].calibration_source if app_state.targets else "uncalibrated", "-"
        )
        n3d = sum(1 for smp in app_state.samples if smp.get("depth"))
        status = (
            f"工位【{app_state.current_workspace_name}】 · "
            f"样本 {len(app_state.samples)} 个 (3D成对 {n3d} / 2D {len(app_state.samples) - n3d}) · "
            f"标定: {calib}"
        )
        draw_text(canvas, status, (m["L"], yb), m["fs_sub"], GuiTheme.TEXT_SUB)
        draw_text(
            canvas,
            "[↑↓] 样本  ·  [空格] 识别定位  ·  右键拖拽  ·  滚轮无级缩放  ·  双击复位",
            (W - int(460 * m["s"]), yb), m["fs_sub"], GuiTheme.TEXT_MUTED
        )

        # 8. 悬浮 Toast
        if app_state._toast_msg and time.time() < app_state._toast_until:
            (tw, th), _ = measure_text(app_state._toast_msg, font_size=m["fs_body"])
            tx, ty = (W - tw) // 2, H - int(76 * m["s"])
            pad = int(10 * m["s"])
            cv2.rectangle(canvas, (tx - pad, ty - int(6 * m["s"])),
                          (tx + tw + pad, ty + th + int(8 * m["s"])), (40, 34, 26), -1)
            cv2.rectangle(canvas, (tx - pad, ty - int(6 * m["s"])),
                          (tx + tw + pad, ty + th + int(8 * m["s"])), GuiTheme.WARN, 1)
            draw_text(canvas, app_state._toast_msg, (tx, ty), m["fs_body"], GuiTheme.WARN, bold=True)

        # 9. 置顶弹出下拉框
        if app_state.active_dropdown == "WORKSPACE_DROPDOWN" and app_state._workspace_rect:
            item_h = int(28 * m["s"])
            btns = render_dropdown_popup(
                canvas, anchor_rect=app_state._workspace_rect,
                options=app_state.workspace_options,
                active_key=app_state.current_workspace_id,
                btn_prefix="DD_MAP_", item_h=item_h, min_width=int(260 * m["s"])
            )
            dd_items = [(r, k) for _, r, k in btns]
        elif app_state.active_dropdown == "PIPELINE_DROPDOWN" and getattr(app_state, "_pipeline_rect", None):
            item_h = int(28 * m["s"])
            btns = render_dropdown_popup(
                canvas, anchor_rect=app_state._pipeline_rect,
                options=app_state.pipeline_options,
                active_key=app_state.pipeline_key,
                btn_prefix="DD_PIPE_", item_h=item_h, min_width=int(260 * m["s"])
            )
            dd_items = [(r, k) for _, r, k in btns]

        return buttons, sample_rows, result_rows, dd_items

    @classmethod
    def _draw_sample_list(cls, canvas, m, rect, app_state, sample_rows):
        y = cls.draw_panel_bg(canvas, rect, f"样本列表 ({len(app_state.samples)})", m)
        if not app_state.samples:
            draw_text(canvas, "当前工位无生产样本", (rect[0] + int(10 * m["s"]), y + int(10 * m["s"])),
                      m["fs_body"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "请在采集向导中拍摄生产样本,",
                      (rect[0] + int(10 * m["s"]), y + int(32 * m["s"])), m["fs_small"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "或通过上方地图下拉框切换工位",
                      (rect[0] + int(10 * m["s"]), y + int(50 * m["s"])), m["fs_small"], GuiTheme.TEXT_MUTED)
            return

        x1, _, x2, y2 = rect
        row_h = m["row_h"]
        visible = max(1, (y2 - y - int(6 * m["s"])) // (row_h + int(4 * m["s"])))
        app_state.scroll_off = max(0, min(app_state.scroll_off, len(app_state.samples) - visible))

        for row_i in range(visible):
            idx = app_state.scroll_off + row_i
            if idx >= len(app_state.samples):
                break
            smp = app_state.samples[idx]
            ry1 = y + row_i * (row_h + int(4 * m["s"]))
            ry2 = ry1 + row_h
            is_sel = (idx == app_state.sel_idx)

            if is_sel:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_SEL, -1)
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.BORDER_SEL, 1)
            elif x1 < app_state.mouse_pos[0] < x2 and ry1 <= app_state.mouse_pos[1] <= ry2:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_HOVER, -1)

            raw_name = smp["name"]
            short_name = raw_name.replace(".png", "").replace(".jpg", "").replace("view_", "")
            if len(short_name) > 10:
                short_name = short_name[-10:]

            draw_text(canvas, short_name, (x1 + int(8 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB, bold=is_sel)

            tag = "3D" if smp.get("depth") else "2D"
            tag_col = GuiTheme.OK if smp.get("depth") else GuiTheme.TEXT_MUTED
            draw_text(canvas, tag, (x2 - int(24 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], tag_col, bold=is_sel)
            sample_rows.append(((x1 + 2, ry1, x2 - 2, ry2), idx))

    @classmethod
    def _draw_image_area(cls, canvas, m, rect, app_state):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 12, 16), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)

        if app_state.vis_img is None:
            msg = app_state.error if app_state.error else ("请在左侧选择样本" if app_state.samples else "当前工位无样本")
            draw_text(canvas, msg, (x1 + int(16 * m["s"]), y1 + int(16 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if app_state.error else GuiTheme.TEXT_MUTED)
            return

        vx, vy, vw, vh = x1 + 2, y1 + 2, x2 - x1 - 4, y2 - y1 - 4
        ih, iw = app_state.vis_img.shape[:2]
        rois = app_state.viewport.compute_viewport_render_rois((vx, vy, vw, vh), iw, ih)

        if rois:
            (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = rois
            src_crop = app_state.vis_img[src_y1:src_y2, src_x1:src_x2]
            dw, dh = dst_x2 - dst_x1, dst_y2 - dst_y1
            if dw > 0 and dh > 0 and src_crop.size > 0:
                interp = cv2.INTER_LINEAR if app_state.viewport.zoom_level > 1.0 else cv2.INTER_AREA
                disp = cv2.resize(src_crop, (dw, dh), interpolation=interp)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = disp

        # 视口底部模式指示
        if app_state.targets:
            mode_txt = (
                f"{'3D 完整链路' if app_state.mode == '3d' else '2D 预览'} — "
                f"提取前 {len(app_state.targets)} 位目标 "
                f"(已高亮目标 #{app_state.targets[app_state.sel_target].id})"
            )
            mode_col = GuiTheme.OK if app_state.mode == "3d" else GuiTheme.WARN
        else:
            mode_txt = "原图已载入 — 点击上方【识别定位】(或按空格键) 开始位姿解算"
            mode_col = GuiTheme.GOLD
        draw_text(canvas, mode_txt, (x1 + int(8 * m["s"]), y2 - int(20 * m["s"])),
                  m["fs_small"], mode_col, bold=True)

        # 视口右上角缩放比例悬浮指示
        if abs(app_state.viewport.zoom_level - 1.0) > 0.01 or app_state.viewport.is_panning:
            zoom_badge = f"缩放: {app_state.viewport.zoom_level:.1f}x [右键拖拽/双击复位]"
            (zw, zh), _ = measure_text(zoom_badge, font_size=m["fs_small"])
            cv2.rectangle(canvas, (x2 - zw - int(16 * m["s"]), y1 + int(8 * m["s"])),
                          (x2 - int(6 * m["s"]), y1 + zh + int(14 * m["s"])), (20, 24, 30), -1)
            cv2.rectangle(canvas, (x2 - zw - int(16 * m["s"]), y1 + int(8 * m["s"])),
                          (x2 - int(6 * m["s"]), y1 + zh + int(14 * m["s"])), GuiTheme.BORDER, 1)
            draw_text(canvas, zoom_badge, (x2 - zw - int(11 * m["s"]), y1 + int(11 * m["s"])),
                      m["fs_small"], GuiTheme.ACCENT)

    @classmethod
    def _draw_result_panel(cls, canvas, m, rect, app_state, result_rows):
        perf_tag = f" · {app_state.pipeline_result.elapsed_ms:.0f}ms" if app_state.pipeline_result else ""
        y = cls.draw_panel_bg(canvas, rect, f"识别结果 (前3位){perf_tag}", m)
        x1, _, x2, y2 = rect

        if not app_state.targets:
            msg = app_state.error if app_state.error else (
                "点击【识别定位】开始分析" if (app_state.samples and app_state.sel_idx >= 0) else "未检出目标"
            )
            draw_text(canvas, msg, (x1 + int(10 * m["s"]), y + int(14 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if app_state.error else GuiTheme.TEXT_MUTED)
            cls._draw_gcode_box(canvas, m, rect, app_state)
            return

        card_h = int(60 * m["s"])
        for list_idx, t in enumerate(app_state.targets):
            ry1 = y + list_idx * (card_h + int(8 * m["s"]))
            if ry1 + card_h > y2 - int(190 * m["s"]):
                break
            ry2 = ry1 + card_h
            is_sel = (list_idx == app_state.sel_target)
            is_top = t.is_topmost

            bg_col = (28, 38, 32) if is_sel else (20, 24, 30)
            border_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else (55, 65, 80))
            cv2.rectangle(canvas, (x1 + 4, ry1), (x2 - 4, ry2), bg_col, -1)
            cv2.rectangle(canvas, (x1 + 4, ry1), (x2 - 4, ry2), border_col, 2 if is_sel else 1)

            badge = "#1最优" if is_top else f"#{t.id}候选"
            badge_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else GuiTheme.TEXT_SUB)
            draw_text(canvas, f"{badge} D:{t.diam_mm} L:{int(t.length_mm)}mm",
                      (x1 + int(8 * m["s"]), ry1 + int(5 * m["s"])), m["fs_small"], badge_col, bold=True)

            h_str = f"+{t.rel_height_mm}mm" if t.rel_height_mm > 0 else (f"Z:{int(t.grip_z)}" if t.grip_z > 0 else "--")
            draw_text(canvas, f"方向:{t.yaw_deg}° 凸起:{h_str}",
                      (x1 + int(8 * m["s"]), ry1 + int(23 * m["s"])), m["fs_small"],
                      GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB)

            draw_text(canvas, f"S:({int(t.robot_x)},{int(t.robot_y)},{int(t.robot_z)}) R:{int(t.robot_r)}°",
                      (x1 + int(8 * m["s"]), ry1 + int(41 * m["s"])), m["fs_small"],
                      GuiTheme.ACCENT if is_sel else GuiTheme.TEXT_MUTED)

            result_rows.append(((x1 + 4, ry1, x2 - 4, ry2), list_idx))

        cls._draw_gcode_box(canvas, m, rect, app_state)

    @classmethod
    def _draw_gcode_box(cls, canvas, m, rect, app_state):
        x1, _, x2, y2 = rect
        gh = int(185 * m["s"])
        gy1 = y2 - gh - int(6 * m["s"])
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      (16, 19, 24), -1)
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      GuiTheme.BORDER, 1)

        if app_state.gcode_text:
            target_id = app_state.targets[app_state.sel_target].id if app_state.targets else 1
            title = f"G-code 预览 (目标 #{target_id}) [E] 导出"
            draw_text(canvas, title, (x1 + int(12 * m["s"]), gy1 + int(5 * m["s"])),
                      m["fs_small"], GuiTheme.GOLD, bold=True)
            yy = gy1 + int(24 * m["s"])
            line_h = int(13 * m["s"])
            for line in app_state.gcode_text.splitlines():
                if yy + line_h > y2 - int(8 * m["s"]):
                    draw_text(canvas, "... (完整内容见 [E] 导出文件)",
                              (x1 + int(12 * m["s"]), yy), m["fs_small"], GuiTheme.TEXT_MUTED)
                    break
                draw_text(canvas, line, (x1 + int(12 * m["s"]), yy), m["fs_gcode"], GuiTheme.TEXT_SUB)
                yy += line_h
        else:
            note = "2D 预览无深度, 不生成抓取 G-code" if app_state.mode == "2d" else "未检出可抓取目标"
            draw_text(canvas, note, (x1 + int(12 * m["s"]), gy1 + int(8 * m["s"])),
                      m["fs_small"], GuiTheme.TEXT_MUTED)
