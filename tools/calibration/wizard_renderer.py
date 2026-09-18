#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AprilTag 采图向导 - 渲染器:
顶部单排工具栏 (相机类型/分辨率/开启, 与 Robot 在线跟踪第一排左半部分同款) /
下拉浮层 / 画布合成 / Toast。
只读 TagCaptureWizard 的状态并绘制, 不修改业务状态;
按钮命中表 (buttons) 每帧由 draw_toolbar 重建, 供主控制器鼠标分发使用。
"""

import time

import cv2
import numpy as np

from src.utils.gui_theme import (
    GuiTheme,
)
from src.utils.text_rendering import draw_text

# 视觉样式常量 (BGR, 统一取自 GuiTheme 主题单源)
COLOR_BG = GuiTheme.BG                    # 工具栏 / 占位背景
COLOR_CARD_BG = GuiTheme.CARD_BG          # 按钮常态底色
COLOR_CARD_SEL = GuiTheme.CARD_SEL        # 乒乓开关激活底色
COLOR_BORDER = GuiTheme.BORDER            # 常态描边
COLOR_BORDER_HOVER = GuiTheme.BORDER_HOVER  # 悬停描边
COLOR_BORDER_SEL = GuiTheme.BORDER_SEL    # 激活描边
COLOR_BTN_HOVER = GuiTheme.BTN_HOVER      # 悬停底色
COLOR_BTN_TEXT_HOVER = GuiTheme.BTN_TEXT_HOVER  # 按钮悬停文字
COLOR_TEXT_SUB = GuiTheme.TEXT_SUB        # 副文字
COLOR_ACCENT = GuiTheme.ACCENT            # 主题强调色
COL_WHITE = GuiTheme.WHITE
COL_YELLOW = (90, 200, 245)               # 提示文字 (数据可视化色, 本地保留)
COL_PANEL_BG = (26, 26, 30)               # Toast 底色 (数据可视化色, 本地保留)

TOOLBAR_H = 44  # 顶部工具栏高度 (单排: 相机类型/分辨率/开启 ... 退出)


class WizardRenderer:
    """采图向导渲染器: 输入主控制器状态, 输出画布与按钮命中表"""

    def __init__(self, wizard):
        self.wiz = wizard               # 主控制器状态引用 (只读)
        self.buttons = []               # [(btn_id, (x1,y1,x2,y2), payload), ...] 每帧重建
        self.mouse_pos = (-1, -1)
        self._camera_type_rect = None
        self._resolution_rect = None

    # ------------------------------ 鼠标辅助 ------------------------------
    def on_mouse_move(self, x, y):
        self.mouse_pos = (x, y)

    def hit_test(self, x, y):
        """命中检测: 返回 (btn_id, payload) 或 None"""
        for btn_id, rect, payload in self.buttons:
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return btn_id, payload
        return None

    def _is_hover(self, rect):
        """鼠标是否悬停在 rect 上 (与 _draw_dropdown_button 的 hover 判定一致)"""
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        return x1 <= mx <= x2 and y1 <= my <= y2

    def _hover_text(self, hovered, size, bold=False):
        """hover 文字行为参数单源 (GuiTheme.BTN_BEHAVIOR): hover 时按主题加粗/放大"""
        if not hovered:
            return size, bold
        bh = GuiTheme.BTN_BEHAVIOR
        return int(size * bh["HOVER_SCALE"]), (bold or bh["HOVER_BOLD"])

    # ------------------------------ 下拉控件 ------------------------------
    def _draw_dropdown_button(self, canvas, rect, label, is_open):
        """扁平化下拉按钮 (与 tracker renderer 同款)"""
        x1, y1, x2, y2 = rect
        is_hover = self._is_hover(rect)
        if is_open:
            bg_col, border_col, text_col = COLOR_BTN_HOVER, COLOR_BORDER_SEL, COL_WHITE
            arrow = "▲"
        elif is_hover:
            bg_col, border_col, text_col = COLOR_BTN_HOVER, COLOR_BORDER_HOVER, COL_WHITE
            arrow = "▼"
        else:
            bg_col, border_col, text_col = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB
            arrow = "▼"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)
        dd_size, dd_bold = self._hover_text(is_hover, 14)
        draw_text(canvas, f"{label} {arrow}", (x1 + 8, y1 + (y2 - y1) // 2 - 8),
                  dd_size, text_col, dd_bold)

    def _render_dropdown_popup(self, canvas, rect, options, active_key, btn_prefix):
        """置顶悬浮下拉列表浮层 (与 tracker renderer 同款)"""
        rx1, ry1, rx2, ry2 = rect
        item_h = 30
        pop_w = max(rx2 - rx1, 210)
        pop_x1, pop_y1 = rx1, ry2 + 2
        pop_x2, pop_y2 = pop_x1 + pop_w, pop_y1 + len(options) * item_h + 6
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (30, 34, 42), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), COLOR_BORDER_SEL, 1)

        for i, (key, label) in enumerate(options):
            iy1 = pop_y1 + 3 + i * item_h
            iy2 = iy1 + item_h
            is_active = (key == active_key)
            if is_active:
                cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), COLOR_CARD_SEL, -1)
            draw_text(canvas, label, (pop_x1 + 10, iy1 + (item_h - 16) // 2 - 2), 15,
                      COLOR_ACCENT if is_active else COL_WHITE, bold=is_active)
            self.buttons.append((f"{btn_prefix}{i}", (pop_x1 + 2, iy1, pop_x2 - 2, iy2), key))

    # ------------------------------ 工具栏 ------------------------------
    def draw_toolbar(self, canvas):
        """顶部单排工具栏 (与 Robot 在线跟踪第一排左半部分同款):
        [相机类型 ▼] [分辨率 ▼] [开启/关闭] ... [退出 X]
        """
        wiz = self.wiz
        tw = canvas.shape[1]
        self.buttons = []
        cv2.rectangle(canvas, (0, 0), (tw, TOOLBAR_H), COLOR_BG, -1)
        cv2.line(canvas, (0, TOOLBAR_H - 1), (tw, TOOLBAR_H - 1), COLOR_BORDER, 1)

        y1, y2 = 6, 38
        gap = 6

        # 1. 相机类型下拉 (最左)
        cam_x1, cam_x2 = 8, 8 + 150
        cam_label = dict(wiz.camera_options).get(wiz.camera_type, wiz.camera_type)
        self._draw_dropdown_button(canvas, (cam_x1, y1, cam_x2, y2), cam_label,
                                   is_open=(wiz.active_dropdown == "CAMERA_TYPE_DROPDOWN"))
        self.buttons.append(("TOGGLE_CAM_DD", (cam_x1, y1, cam_x2, y2), "CAMERA_TYPE_DROPDOWN"))
        self._camera_type_rect = (cam_x1, y1, cam_x2, y2)

        # 2. 分辨率下拉
        res_x1 = cam_x2 + gap
        res_x2 = res_x1 + 110
        self._draw_dropdown_button(canvas, (res_x1, y1, res_x2, y2), wiz.resolution,
                                   is_open=(wiz.active_dropdown == "RES_DROPDOWN"))
        self.buttons.append(("TOGGLE_RES_DD", (res_x1, y1, res_x2, y2), "RES_DROPDOWN"))
        self._resolution_rect = (res_x1, y1, res_x2, y2)

        # 3. 开启/关闭乒乓按钮 (悬停高亮, 与 tracker 同款三态)
        sw_x1 = res_x2 + gap
        sw_x2 = sw_x1 + 70
        sw_hover = self._is_hover((sw_x1, y1, sw_x2, y2))
        if wiz.pipeline_running:
            sw_bg, sw_border, sw_txt, sw_label = (55, 45, 30), (255, 160, 40), (255, 200, 80), "关闭"
        else:
            sw_bg, sw_border, sw_txt, sw_label = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB, "开启"
        if sw_hover:   # 悬停高亮不覆盖开/关语义色 (仅底色提亮 + 主题悬停描边)
            sw_bg, sw_border = COLOR_BTN_HOVER, COLOR_BORDER_HOVER
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_bg, -1)
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_border, 1)
        sw_size, _ = self._hover_text(sw_hover, 15, True)
        draw_text(canvas, sw_label, (sw_x1 + 22, y1 + (y2 - y1 - 16) // 2 - 1), sw_size, sw_txt, True)
        self.buttons.append(("TOGGLE_CAMERA", (sw_x1, y1, sw_x2, y2), None))

        # 4. 退出按钮 (最右, 悬停高亮)
        exit_x1, exit_x2 = tw - 90, tw - 8
        q_hover = self._is_hover((exit_x1, y1, exit_x2, y2))
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2),
                      COLOR_BTN_HOVER if q_hover else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2),
                      COLOR_BORDER_HOVER if q_hover else COLOR_BORDER, 1)
        q_size, _ = self._hover_text(q_hover, 14, True)
        draw_text(canvas, "退出 X", (exit_x1 + 14, y1 + (y2 - y1 - 16) // 2 - 1), q_size,
                  COLOR_BTN_TEXT_HOVER if q_hover else (190, 190, 200), True)
        self.buttons.append(("QUIT", (exit_x1, y1, exit_x2, y2), None))

        # 展开的下拉浮层 (置顶最后绘制)
        if wiz.active_dropdown == "CAMERA_TYPE_DROPDOWN" and self._camera_type_rect:
            self._render_dropdown_popup(canvas, self._camera_type_rect,
                                        wiz.camera_options, wiz.camera_type, "DD_CAM_")
        elif wiz.active_dropdown == "RES_DROPDOWN" and self._resolution_rect:
            self._render_dropdown_popup(canvas, self._resolution_rect,
                                        wiz.resolution_options, wiz.resolution, "DD_RES_")

    # ------------------------------ 画布合成 ------------------------------
    def make_canvas(self):
        """按当前窗口物理尺寸生成底板画布 (imshow 严格 1:1, 鼠标坐标零偏移)"""
        cw = self.wiz.win_mgr.canvas_w
        ch = self.wiz.win_mgr.canvas_h
        return np.full((ch, cw, 3), COLOR_BG, dtype=np.uint8)

    def compose_canvas(self, frame):
        """窗口尺寸底板 + 顶部工具栏区 + 视频帧等比缩放居中 (真矢量模式)"""
        cw = self.wiz.win_mgr.canvas_w
        ch = self.wiz.win_mgr.canvas_h
        canvas = np.full((ch, cw, 3), COLOR_BG, dtype=np.uint8)

        # 视频帧等比 contain 缩放到工具栏下方区域并居中 (不裁剪, 保证 Tag 不出画)
        area_h = ch - TOOLBAR_H
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, area_h / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame2 = cv2.resize(frame, (nw, nh), interpolation=interp)
        x0, y0 = (cw - nw) // 2, TOOLBAR_H + (area_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = frame2
        return canvas

    def draw_toast(self, canvas):
        """底部居中临时通知 (沿用向导原 Toast 样式, 绘制在最终画布上)"""
        wiz = self.wiz
        h, w = canvas.shape[:2]
        if time.time() - wiz.status_toast_time < 2.5 and wiz.status_toast:
            (tw, _), _ = self._measure(wiz.status_toast)
            toast_x = (w - tw) // 2
            cv2.rectangle(canvas, (toast_x - 12, h - 60), (toast_x + tw + 12, h - 25),
                          (0, 120, 0), -1)
            draw_text(canvas, wiz.status_toast, (toast_x, h - 52), 16, COL_WHITE)

    @staticmethod
    def _measure(text):
        """文字测宽 (draw_text 同源渲染路径)"""
        from src.utils.text_rendering import measure_text
        return measure_text(text, font_size=16, bold=False)
