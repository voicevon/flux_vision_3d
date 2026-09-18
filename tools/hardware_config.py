#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件环境配置 (Hardware Config) — Dashboard 2 号卡片
====================================================
统一选型硬件环境并持久化, 供全系统生产工具启动时自动读取:
  ① 摄像机: 类型 (RealSense D435 / USB 普通摄像头) + 默认分辨率;
  ② 机械臂: 类型 (SCARA 串联 / Delta 并联) + 默认串口。
配置落盘 config/hardware_env.json, 下拉选择后自动保存, 即改即生效。
窗口偏好 (缩放/尺寸) 经 GuiWindowManager 归档 config/gui_settings.json。
"""

import os
import sys
import json
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.gui_theme import GuiTheme
from src.utils.gui_window_manager import GuiWindowManager
from src.utils.text_rendering import draw_text, measure_text
from src.utils.logger import get_logger
from tools.tracker.common import list_serial_ports

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "hardware_env.json")
WINDOW_KEY = "HardwareConfig"   # cv2 窗口内部 key (纯 ASCII, 中文标题经 SetWindowTextW 注入)
APP_ID = "hardware_config"
BASE_W, BASE_H = 760, 560       # 基准逻辑画布 (真矢量模式按窗口物理尺寸重绘)

log = get_logger(__name__)


class HardwareConfigApp:
    """硬件环境配置 GUI: 摄像机与机械臂选型下拉 + 自动持久化"""

    def __init__(self, config_file=None):
        self.config_file = config_file or CONFIG_PATH
        self.win_mgr = GuiWindowManager(app_id=APP_ID, base_w=BASE_W, base_h=BASE_H,
                                        min_w=520, min_h=400)

        # 选项集 (与 tracker CameraController 保持一致)
        self.camera_options = [
            ("realsense", "RealSense D435"),
            ("usb",       "USB 普通摄像头"),
        ]
        self.resolution_options = [
            ("1280x720",  "1280 × 720  (推荐)"),
            ("1920x1080", "1920 × 1080"),
            ("848x480",   "848 × 480"),
            ("640x480",   "640 × 480"),
        ]
        self.arm_options = [
            ("scara", "SCARA (串联)"),
            ("delta", "Delta (并联)"),
        ]
        self.port_options = []          # 打开下拉时实时枚举

        self.config = self._load_config()
        self.active_dropdown = None     # "cam" | "res" | "arm" | "port" | None
        self.mouse_pos = (-1, -1)
        self._buttons = []              # [(rect, action), ...] 每帧重建, action=("toggle",组)|("opt",组,值)
        self._toast_msg = None
        self._toast_until = 0.0
        self._running = True

    # ------------------------------ 配置持久化 ------------------------------
    def _load_config(self):
        """读取硬件环境配置, 缺省时回落默认选型"""
        defaults = {
            "camera_type": "realsense",
            "camera_resolution": "1280x720",
            "arm_type": "scara",
            "arm_port": "",
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    defaults.update({k: loaded[k] for k in defaults if k in loaded})
            except Exception as exc:
                log.warning("硬件环境配置读取失败, 使用默认选型: %s", exc)
        return defaults

    def save_config(self):
        """原子化落盘硬件环境配置 (即改即存)"""
        self.config["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            log.error("硬件环境配置保存失败: %s", exc)
            self.set_toast("保存失败! 详见日志", True)

    def select(self, group, value):
        """更新选型并立即持久化 (group: cam/res/arm/port)"""
        key_map = {"cam": "camera_type", "res": "camera_resolution",
                   "arm": "arm_type", "port": "arm_port"}
        self.config[key_map[group]] = value
        self.save_config()
        self.set_toast(f"已保存: {self._group_title(group)} = {self._option_label(group, value)}")

    # ------------------------------ 选项辅助 ------------------------------
    GROUP_TITLES = {"cam": "摄像机类型", "res": "默认分辨率",
                    "arm": "机械臂类型", "port": "默认串口"}

    def _group_title(self, group):
        return self.GROUP_TITLES[group]

    def _options(self, group):
        if group == "cam":
            return self.camera_options
        if group == "res":
            return self.resolution_options
        if group == "arm":
            return self.arm_options
        return self.port_options

    def _option_label(self, group, value):
        for key, label in self._options(group):
            if key == value:
                return label
        if group == "port":
            return value if value else "未选择"
        return value

    def _toggle_dropdown(self, group):
        """开/关下拉浮层 (打开串口下拉时实时刷新枚举)"""
        if self.active_dropdown == group:
            self.active_dropdown = None
            return
        if group == "port":
            ports = list_serial_ports()
            self.port_options = [(p, p) for p in ports] or [("", "未检测到串口")]
        self.active_dropdown = group

    # ------------------------------ 布局与渲染 ------------------------------
    def _metrics(self):
        s = self.win_mgr.canvas_w / BASE_W
        return {
            "s": s,
            "pad_x": int(40 * s),
            "label_w": int(180 * s),
            "btn_w": int(300 * s),
            "btn_h": int(40 * s),
            "row_gap": int(20 * s),
            "sec_h": int(38 * s),
            "fs_title": max(18, int(26 * s)),
            "fs_sub": max(12, int(14 * s)),
            "fs_sec": max(14, int(18 * s)),
            "fs_label": max(13, int(16 * s)),
            "fs_value": max(13, int(15 * s)),
        }

    def _draw_dropdown_button(self, canvas, rect, label, is_open):
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        hover = x1 <= mx <= x2 and y1 <= my <= y2
        if is_open:
            bg, border, col = GuiTheme.CARD_SEL, GuiTheme.BORDER_SEL, GuiTheme.WHITE
            arrow = "▲"
        elif hover:
            bg, border, col = GuiTheme.CARD_HOVER, GuiTheme.BORDER_HOVER, GuiTheme.WHITE
            arrow = "▼"
        else:
            bg, border, col = GuiTheme.CARD_BG, GuiTheme.BORDER, GuiTheme.TEXT_SUB
            arrow = "▼"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
        m = self._metrics()
        draw_text(canvas, label, (x1 + int(10 * m["s"]), y1 + (y2 - y1 - int(18 * m["s"])) // 2),
                  m["fs_value"], col)
        # 箭头右对齐
        (tw, _), _ = measure_text(arrow, font_size=m["fs_value"])
        draw_text(canvas, arrow, (x2 - tw - int(10 * m["s"]),
                                  y1 + (y2 - y1 - int(18 * m["s"])) // 2),
                  m["fs_value"], col)

    def _render_dropdown_popup(self, canvas, rect, group):
        """置顶悬浮下拉列表 (底部溢出时自动向上展开)"""
        m = self._metrics()
        rx1, ry1, rx2, ry2 = rect
        item_h = int(36 * m["s"])
        options = self._options(group)
        total = len(options) * item_h + int(6 * m["s"])
        pop_x1, pop_y1, pop_x2 = rx1, ry2 + 2, rx2
        if pop_y1 + total > canvas.shape[0] - int(8 * m["s"]):   # 向下溢出 → 向上展开
            pop_y1 = ry1 - total - 2
        pop_y2 = pop_y1 + total

        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (30, 34, 42), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), GuiTheme.BORDER_SEL, 1)

        current = {"cam": "camera_type", "res": "camera_resolution",
                   "arm": "arm_type", "port": "arm_port"}[group]
        for i, (key, label) in enumerate(options):
            iy1 = pop_y1 + int(3 * m["s"]) + i * item_h
            iy2 = iy1 + item_h
            is_active = (key == self.config[current])
            mx, my = self.mouse_pos
            is_hover = pop_x1 < mx < pop_x2 and iy1 <= my <= iy2
            if is_active:
                cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), GuiTheme.CARD_SEL, -1)
            elif is_hover:
                cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), GuiTheme.CARD_HOVER, -1)
            draw_text(canvas, label,
                      (pop_x1 + int(10 * m["s"]), iy1 + (item_h - int(18 * m["s"])) // 2),
                      m["fs_value"], GuiTheme.ACCENT if is_active else GuiTheme.WHITE,
                      bold=is_active)
            if key or group != "port":   # "未检测到串口" 占位项不可选
                self._buttons.append(((pop_x1 + 2, iy1, pop_x2 - 2, iy2), ("opt", group, key)))

    def render(self):
        """真矢量渲染: 画布按窗口物理尺寸 1:1 重绘 (imshow 零缩放, 鼠标坐标零偏移)"""
        m = self._metrics()
        W, H = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        canvas = np.full((H, W, 3), GuiTheme.BG, dtype=np.uint8)
        self._buttons = []

        # 标题区
        draw_text(canvas, "硬件环境配置", (m["pad_x"], int(24 * m["s"])), m["fs_title"],
                  GuiTheme.TEXT, bold=True)
        draw_text(canvas, "统一选型硬件环境 · 选择后自动保存 · 全系统工具启动时自动读取",
                  (m["pad_x"], int(72 * m["s"])), m["fs_sub"], GuiTheme.TEXT_SUB)
        cv2.line(canvas, (m["pad_x"], int(102 * m["s"])), (W - m["pad_x"], int(102 * m["s"])),
                 GuiTheme.BORDER, 1)

        # 行布局: 分组标题 + 下拉行
        rows = [
            ("sec", "① 摄像机", None),
            ("row", "cam", self._option_label("cam", self.config["camera_type"])),
            ("row", "res", self._option_label("res", self.config["camera_resolution"])),
            ("sec", "② 机械臂", None),
            ("row", "arm", self._option_label("arm", self.config["arm_type"])),
            ("row", "port", self._option_label("port", self.config["arm_port"])),
        ]
        y = int(124 * m["s"])
        row_rects = {}
        for kind, text, value in rows:
            if kind == "sec":
                draw_text(canvas, text, (m["pad_x"], y), m["fs_sec"], GuiTheme.ACCENT, bold=True)
                y += m["sec_h"]
            else:
                draw_text(canvas, self._group_title(text),
                          (m["pad_x"], y + (m["btn_h"] - int(20 * m["s"])) // 2),
                          m["fs_label"], GuiTheme.TEXT_SUB)
                bx1 = m["pad_x"] + m["label_w"]
                rect = (bx1, y, bx1 + m["btn_w"], y + m["btn_h"])
                row_rects[text] = rect
                self._buttons.append((rect, ("toggle", text)))
                y += m["btn_h"] + m["row_gap"]

        for group, rect in row_rects.items():
            label = self._option_label(group, self.config[{
                "cam": "camera_type", "res": "camera_resolution",
                "arm": "arm_type", "port": "arm_port"}[group]])
            self._draw_dropdown_button(canvas, rect, label,
                                       is_open=(self.active_dropdown == group))
        if self.active_dropdown:
            self._render_dropdown_popup(canvas, row_rects[self.active_dropdown],
                                        self.active_dropdown)

        # 底部: 状态行 + 操作提示
        updated = self.config.get("updated_at", "")
        try:
            cfg_display = os.path.relpath(self.config_file, PROJECT_ROOT)
        except ValueError:   # 跨盘符 (如测试临时目录) 退回绝对路径
            cfg_display = self.config_file
        status = f"配置文件: {cfg_display}"
        if updated:
            status += f"  ·  上次保存 {updated}"
        draw_text(canvas, status, (m["pad_x"], H - int(58 * m["s"])), m["fs_sub"],
                  GuiTheme.TEXT_MUTED)
        draw_text(canvas, "下拉选择后自动保存  |  [ESC]/[X] 退出  |  Ctrl+滚轮/± 缩放",
                  (m["pad_x"], H - int(32 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)

        # Toast (底部居中)
        if self._toast_msg and time.time() < self._toast_until:
            (tw, th), _ = measure_text(self._toast_msg, font_size=m["fs_value"])
            tx = (W - tw) // 2
            ty = H - int(96 * m["s"])
            cv2.rectangle(canvas, (tx - int(12 * m["s"]), ty - int(8 * m["s"])),
                          (tx + tw + int(12 * m["s"]), ty + th + int(10 * m["s"])),
                          (40, 34, 26), -1)
            cv2.rectangle(canvas, (tx - int(12 * m["s"]), ty - int(8 * m["s"])),
                          (tx + tw + int(12 * m["s"]), ty + th + int(10 * m["s"])),
                          GuiTheme.WARN, 1)
            draw_text(canvas, self._toast_msg, (tx, ty), m["fs_value"], GuiTheme.WARN, bold=True)

        return canvas

    # ------------------------------ 交互 ------------------------------
    def set_toast(self, msg, sticky=False, duration=2.2):
        self._toast_msg = msg
        self._toast_until = time.time() + (3600 if sticky else duration)

    def hit_test(self, x, y):
        """命中检测: 倒序遍历 (浮层项优先), 返回 action 或 None"""
        for rect, action in reversed(self._buttons):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return action
        return None

    def _on_mouse(self, event, x, y, flags, param):
        # 画布按窗口物理尺寸 1:1 重绘, 窗口坐标即画布坐标 (零偏移)
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
            return

        handled, toast = self.win_mgr.handle_mouse_wheel(event, flags)
        if handled:
            self.set_toast(toast)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = (x, y)
            hit = self.hit_test(x, y)
            if hit is None:
                self.active_dropdown = None      # 点击空白关闭浮层
                return
            kind = hit[0]
            if kind == "toggle":
                self._toggle_dropdown(hit[1])
            elif kind == "opt":
                self.select(hit[1], hit[2])
                self.active_dropdown = None

    # ------------------------------ 主循环 ------------------------------
    def run(self):
        self.win_mgr.setup_window(WINDOW_KEY, self._on_mouse)
        self.win_mgr.set_unicode_title("硬件环境配置 - Hardware Config")
        log.info("硬件环境配置已启动: %s", self.config_file)

        while self._running:
            key = cv2.waitKey(30)
            poll = self.win_mgr.poll_events(key & 0xFFFF if key > 0 else -1)
            if poll.should_quit:
                break
            if key > 0 and (key & 0xFF) in (ord('x'), ord('X')):
                break
            if poll.toast_msg:
                self.set_toast(poll.toast_msg)

            cv2.imshow(WINDOW_KEY, self.render())

        cv2.destroyWindow(WINDOW_KEY)
        log.info("硬件环境配置已退出")


def main():
    app = HardwareConfigApp()
    app.run()


if __name__ == "__main__":
    main()
