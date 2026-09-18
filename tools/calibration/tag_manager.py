#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 管理器 — 左右两栏布局的 cv2 原生 GUI
左侧 sidebar 为 Tab 卡片切换，右侧为内容区域。
Tab 1: 图纸生成 (import generate_tags() 函数)
Tab 2: 白名单管理 (30 个 Tag ID toggle + 保存 config.yaml)
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.gui_window_manager import GuiWindowManager
from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text
from src.utils.config_guard import load_raw_config

# 复用旧代码的图纸生成函数 (不修改旧代码)
from tools.calibration.generate_apriltags import generate_tags

from src.utils.logger import get_logger

log = get_logger(__name__)


class TagManager:
    """AprilTag 管理器 GUI"""

    # —— 工业暗色主题 (与 d435_viewer / gui_launcher 一致) ——
    # 调色板: 统一取自 GuiTheme 主题单源 (原略有差异的局部色值已归一到全局色板)
    COLOR_BG = GuiTheme.BG
    COLOR_SIDEBAR = GuiTheme.CARD_BG
    COLOR_CARD_BG = GuiTheme.CARD_HOVER
    COLOR_CARD_SEL = GuiTheme.CARD_SEL
    COLOR_BORDER = GuiTheme.BORDER
    COLOR_BORDER_SEL = GuiTheme.BORDER_SEL
    COLOR_TEXT = GuiTheme.TEXT
    COLOR_TEXT_SUB = GuiTheme.TEXT_SUB
    COLOR_ACCENT = GuiTheme.ACCENT
    COLOR_ACCENT2 = (0, 170, 255)      # 次强调 (橙), 本地保留
    COLOR_OK = GuiTheme.OK
    COLOR_WARN = GuiTheme.WARN
    COLOR_ERR = GuiTheme.ERR
    COLOR_TAG_ON = (0, 160, 140)       # 白名单内 Tag (青绿色), 本地保留
    COLOR_TAG_OFF = (75, 80, 95)       # 白名单外 Tag, 本地保留

    # 布局常量
    SIDEBAR_W = 220
    TOOLBAR_H = 0                       # 不设顶部工具栏, sidebar 承担切换
    TAB_CONTENT_H = 40                  # 右侧子工具栏高度
    MARGIN = 12

    # Tag 16h5 共 30 个 (ID 0~29)
    TAG_COUNT = 30
    TAG_PRESET_DEFAULT = [0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
    SETTINGS_FILE = os.path.join(PROJECT_ROOT, "data", "gui_settings.json")
    APP_ID = "tag_manager"

    def __init__(self):
        self.win_mgr = GuiWindowManager(
            self.APP_ID,
            base_w=1100, base_h=680,
            min_w=860, min_h=500,
            settings_file=self.SETTINGS_FILE,
        )
        self.app_id = self.APP_ID

        # —— 状态 ——
        self.active_tab = "generator"     # "generator" | "whitelist"
        self._quit_requested = False
        self.mouse_pos = (0, 0)
        self.gui_buttons = []             # 每帧重建, 用于 hit-test

        # —— Tab 定义 ——
        self.tabs = [
            {"key": "generator", "label": "📐 图纸生成",  "desc": "生成 AprilTag PDF/PNG"},
            {"key": "whitelist", "label": "✅ 白名单管理", "desc": "0~29 Tag ID 开关"},
        ]

        # —— Tab 1: 图纸生成参数 ——
        self.gen_tag_count = 30
        self.gen_pixel_size = 800
        self.gen_border_bits = 2
        self.gen_output_dir = os.path.join(PROJECT_ROOT, "data", "apriltags_16h5")
        self.gen_status = ""              # 生成进度/结果信息
        self.gen_preview = None           # 生成后的总览网格图
        self.gen_running = False

        # —— Tab 2: 白名单 ——
        self.valid_tag_ids = self._load_valid_tag_ids()

        self._load_settings()

    # ================================================================
    # 持久化
    # ================================================================
    def _load_settings(self):
        try:
            if not os.path.exists(self.SETTINGS_FILE):
                return
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            st = root.get(self.APP_ID, {})
            if "gen_tag_count" in st:
                self.gen_tag_count = int(st["gen_tag_count"])
            if "gen_pixel_size" in st:
                self.gen_pixel_size = int(st["gen_pixel_size"])
        except Exception as e:
            log.warning(f"恢复标靶生成器设置失败，使用默认值: {e}")

    def _save_settings(self):
        try:
            os.makedirs(os.path.dirname(self.SETTINGS_FILE), exist_ok=True)
            root = {}
            if os.path.exists(self.SETTINGS_FILE):
                try:
                    with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                        root = json.load(f)
                except Exception:
                    pass  # 已有设置读取失败时回退为空字典，避免阻塞保存
            st = root.setdefault(self.APP_ID, {})
            st["gen_tag_count"] = self.gen_tag_count
            st["gen_pixel_size"] = self.gen_pixel_size
            st["active_tab"] = self.active_tab
            st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存标靶生成器设置失败: {e}")

    def _load_valid_tag_ids(self):
        cfg = load_raw_config(self.CONFIG_PATH)
        ids = cfg.get("calibration", {}).get("valid_tag_ids", [])
        return [int(x) for x in ids] if ids else []

    def _save_valid_tag_ids(self):
        """写回 config.yaml"""
        try:
            with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("calibration", {})["valid_tag_ids"] = sorted(self.valid_tag_ids)
            with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            self.gen_status = f"✅ 白名单已保存: {sorted(self.valid_tag_ids)}"
        except Exception as e:
            self.gen_status = f"❌ 保存失败: {e}"

    # ================================================================
    # 事件处理
    # ================================================================
    def on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._hit_test(x, y)

    def _hit_test(self, x, y):
        # 反算: mouse 回调返回的是窗口物理坐标, 如果 render 尺寸和窗口不一致需要映射
        cw = self.win_mgr.canvas_w
        ch = self.win_mgr.canvas_h
        full_w = getattr(self, "_last_full_w", cw)
        full_h = getattr(self, "_last_full_h", ch)
        if full_w > 0 and full_h > 0 and (cw != full_w or ch != full_h):
            scale = min(cw / float(full_w), ch / float(full_h))
            pad_x = (cw - int(full_w * scale)) // 2
            pad_y = (ch - int(full_h * scale)) // 2
            x = int((x - pad_x) / max(1e-6, scale))
            y = int((y - pad_y) / max(1e-6, scale))

        for btn_id, (bx1, by1, bx2, by2), payload in self.gui_buttons:
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self._on_click(btn_id, payload)
                return

    def _on_click(self, btn_id, payload):
        if btn_id == "QUIT":
            self._quit_requested = True
            return
        if btn_id.startswith("TAB_"):
            self.active_tab = payload
            self._save_settings()
        elif btn_id == "GENERATE":
            self._do_generate()
        elif btn_id == "WHITELIST_ALL":
            self.valid_tag_ids = list(range(self.TAG_COUNT))
            self._save_valid_tag_ids()
        elif btn_id == "WHITELIST_CLEAR":
            self.valid_tag_ids = []
            self._save_valid_tag_ids()
        elif btn_id == "WHITELIST_DEFAULT":
            self.valid_tag_ids = list(self.TAG_PRESET_DEFAULT)
            self._save_valid_tag_ids()
        elif btn_id == "WHITELIST_SAVE":
            self._save_valid_tag_ids()
        elif btn_id.startswith("TAG_"):
            tid = int(payload)
            if tid in self.valid_tag_ids:
                self.valid_tag_ids.remove(tid)
            else:
                self.valid_tag_ids.append(tid)

    def _is_hover(self, bx1, by1, bx2, by2):
        x, y = self.mouse_pos
        return bx1 <= x <= bx2 and by1 <= y <= by2

    def _do_generate(self):
        """调用 generate_tags() 生成图纸"""
        if self.gen_running:
            return
        self.gen_running = True
        self.gen_status = "⏳ 正在生成..."
        try:
            generate_tags(
                output_dir=self.gen_output_dir,
                tag_count=self.gen_tag_count,
                tag_pixel_size=self.gen_pixel_size,
                border_bits=self.gen_border_bits,
            )
            # 尝试加载生成的网格预览图
            grid_path = os.path.join(self.gen_output_dir, "apriltags_16h5_all_grid.png")
            if os.path.exists(grid_path):
                self.gen_preview = cv2.imread(grid_path)
            self.gen_status = f"✅ 完成: {self.gen_tag_count} 个 Tag → {self.gen_output_dir}"
        except Exception as e:
            self.gen_status = f"❌ 生成失败: {e}"
        finally:
            self.gen_running = False
            self._save_settings()

    # ================================================================
    # 渲染
    # ================================================================
    def _render(self, canvas_w, canvas_h):
        canvas = np.full((canvas_h, canvas_w, 3), self.COLOR_BG, dtype=np.uint8)
        self.gui_buttons = []

        # —— Sidebar ——
        cv2.rectangle(canvas, (0, 0), (self.SIDEBAR_W, canvas_h), self.COLOR_SIDEBAR, -1)
        cv2.line(canvas, (self.SIDEBAR_W - 1, 0), (self.SIDEBAR_W - 1, canvas_h), self.COLOR_BORDER, 1)

        # 标题
        draw_text(canvas, "AprilTag", (16, 14), font_size=16, color=self.COLOR_ACCENT, bold=True)
        draw_text(canvas, "管理器", (16, 34), font_size=14, color=self.COLOR_TEXT)
        draw_text(canvas, "v1.0", (16, 52), font_size=10, color=self.COLOR_TEXT_SUB)
        cv2.line(canvas, (12, 64), (self.SIDEBAR_W - 12, 64), self.COLOR_BORDER, 1)

        # Tab 卡片列表
        card_h = 66
        cy = 74
        for tab in self.tabs:
            active = (tab["key"] == self.active_tab)
            x1, x2 = 10, self.SIDEBAR_W - 10
            y1, y2 = cy, cy + card_h
            hover = self._is_hover(x1, y1, x2, y2)

            if active:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_CARD_SEL, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_ACCENT, 2)
                draw_text(canvas, tab["label"], (x1 + 14, y1 + 22), font_size=14, color=self.COLOR_ACCENT, bold=True)
                draw_text(canvas, tab["desc"], (x1 + 14, y1 + 46), font_size=11, color=self.COLOR_TEXT)
                cv2.rectangle(canvas, (x1, y1 + 6), (x1 + 3, y2 - 6), self.COLOR_ACCENT, -1)
            elif hover:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (42, 48, 60), -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_BORDER_SEL, 1)
                draw_text(canvas, tab["label"], (x1 + 14, y1 + 22), font_size=14, color=self.COLOR_TEXT, bold=True)
                draw_text(canvas, tab["desc"], (x1 + 14, y1 + 46), font_size=11, color=self.COLOR_TEXT_SUB)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_CARD_BG, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_BORDER, 1)
                draw_text(canvas, tab["label"], (x1 + 14, y1 + 22), font_size=14, color=self.COLOR_TEXT, bold=False)
                draw_text(canvas, tab["desc"], (x1 + 14, y1 + 46), font_size=11, color=self.COLOR_TEXT_SUB)

            self.gui_buttons.append((f"TAB_{tab['key']}", (x1, y1, x2, y2), tab["key"]))
            cy += card_h + 10

        # —— 右侧内容区 ——
        rx1 = self.SIDEBAR_W
        rx2 = canvas_w
        ry1 = 0
        ry2 = canvas_h

        if self.active_tab == "generator":
            self._render_generator(canvas, rx1, ry1, rx2, ry2)
        elif self.active_tab == "whitelist":
            self._render_whitelist(canvas, rx1, ry1, rx2, ry2)

        # 退出按钮 —— 固定在 sidebar 底部
        ex1, ex2 = 10, self.SIDEBAR_W - 10
        ey2 = canvas_h - 12
        ey1 = ey2 - 40
        hover_q = self._is_hover(ex1, ey1, ex2, ey2)
        q_bg = (60, 30, 30) if hover_q else (35, 28, 28)
        q_border = self.COLOR_ERR if hover_q else (90, 50, 50)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), q_bg, -1)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), q_border, 2)
        draw_text(canvas, "✕ 退出", (ex1 + 14, ey1 + 25), font_size=13, color=(255, 140, 140), bold=True)
        self.gui_buttons.append(("QUIT", (ex1, ey1, ex2, ey2), None))

        return canvas

    def _render_generator(self, canvas, x1, y1, x2, y2):
        """图纸生成 Tab"""
        content_area = (x1 + self.MARGIN, y1 + self.MARGIN + 30,
                        x2 - self.MARGIN, y2 - self.MARGIN)

        # 子工具栏
        cv2.line(canvas, (x1 + 8, y1 + 30), (x2 - 8, y1 + 30), self.COLOR_BORDER, 1)
        draw_text(canvas, "📐 图纸生成", (x1 + self.MARGIN, y1 + 20), font_size=14, color=self.COLOR_ACCENT, bold=True)

        # —— 参数面板 (左半) ——
        param_x1, param_x2 = content_area[0], content_area[0] + 340
        param_y1, param_y2 = content_area[1], content_area[3]
        cv2.rectangle(canvas, (param_x1, param_y1), (param_x2, param_y2), self.COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (param_x1, param_y1), (param_x2, param_y2), self.COLOR_BORDER, 1)

        draw_text(canvas, "生成参数", (param_x1 + 14, param_y1 + 22), font_size=12, color=self.COLOR_TEXT, bold=True)
        cv2.line(canvas, (param_x1 + 14, param_y1 + 34), (param_x2 - 14, param_y1 + 34), self.COLOR_BORDER, 1)

        # 参数行
        ly = param_y1 + 54
        for label, value in [
            ("标靶数量",  f"{self.gen_tag_count} 个 (ID 00 ~ {self.gen_tag_count - 1:02d})"),
            ("像素尺寸",  f"{self.gen_pixel_size} × {self.gen_pixel_size}"),
            ("边框宽度",  f"{self.gen_border_bits} bits"),
            ("输出目录",  self.gen_output_dir),
        ]:
            draw_text(canvas, label, (param_x1 + 14, ly), font_size=11, color=self.COLOR_TEXT_SUB)
            draw_text(canvas, value, (param_x1 + 110, ly), font_size=11, color=self.COLOR_TEXT)
            ly += 24

        # 生成按钮
        btn_y = ly + 16
        btn_h = 36
        btn_x1 = param_x1 + 14
        btn_x2 = param_x2 - 14
        hover_g = self._is_hover(btn_x1, btn_y, btn_x2, btn_y + btn_h)
        if self.gen_running:
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), (40, 45, 55), -1)
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), self.COLOR_BORDER, 1)
            draw_text(canvas, "⏳ 生成中...", (btn_x1 + (btn_x2 - btn_x1) // 2 - 40, btn_y + 23),
                      font_size=12, color=self.COLOR_TEXT_SUB, bold=True)
        else:
            g_bg = (30, 95, 80) if hover_g else (20, 70, 60)
            g_border = self.COLOR_ACCENT if hover_g else (0, 170, 150)
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), g_bg, -1)
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), g_border, 2)
            draw_text(canvas, "▶ 开始生成 PDF / PNG", (btn_x1 + 14, btn_y + 23),
                      font_size=12, color=self.COLOR_ACCENT, bold=True)
        self.gui_buttons.append(("GENERATE", (btn_x1, btn_y, btn_x2, btn_y + btn_h), None))

        # 状态行
        status_y = btn_y + btn_h + 14
        status_color = self.COLOR_OK if self.gen_status.startswith("✅") else \
                       self.COLOR_ERR if self.gen_status.startswith("❌") else \
                       self.COLOR_TEXT_SUB
        draw_text(canvas, self.gen_status or "点击上方按钮开始生成", (param_x1 + 14, status_y),
                  font_size=10, color=status_color)

        # —— 预览区 (右半) ——
        prev_x1 = param_x2 + self.MARGIN
        prev_x2 = content_area[2]
        prev_y1 = content_area[1]
        prev_y2 = content_area[3]
        cv2.rectangle(canvas, (prev_x1, prev_y1), (prev_x2, prev_y2), self.COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (prev_x1, prev_y1), (prev_x2, prev_y2), self.COLOR_BORDER, 1)

        draw_text(canvas, "📋 预览 / 输出", (prev_x1 + 14, prev_y1 + 22), font_size=12, color=self.COLOR_TEXT, bold=True)
        cv2.line(canvas, (prev_x1 + 14, prev_y1 + 34), (prev_x2 - 14, prev_y1 + 34), self.COLOR_BORDER, 1)

        prev_inner_y = prev_y1 + 44
        prev_h = prev_y2 - prev_inner_y - 10
        prev_w = prev_x2 - prev_x1 - 20

        if self.gen_preview is not None:
            # letterbox 保持比例贴入
            h, w = self.gen_preview.shape[:2]
            scale = min(prev_w / float(w), prev_h / float(h))
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(self.gen_preview, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = prev_x1 + 10 + (prev_w - nw) // 2
            oy = prev_inner_y + (prev_h - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized
            cv2.rectangle(canvas, (prev_x1 + 10, prev_inner_y), (prev_x2 - 10, prev_y2 - 10), self.COLOR_BORDER, 1)
        else:
            # 占位
            cv2.rectangle(canvas, (prev_x1 + 10, prev_inner_y), (prev_x2 - 10, prev_y2 - 10), self.COLOR_BG, -1)
            cv2.rectangle(canvas, (prev_x1 + 10, prev_inner_y), (prev_x2 - 10, prev_y2 - 10), self.COLOR_BORDER, 1)
            draw_text(canvas, "生成后总览网格图将显示在这里",
                      (prev_x1 + prev_w // 2 - 140, prev_inner_y + prev_h // 2 - 6),
                      font_size=11, color=self.COLOR_TEXT_SUB)

    def _render_whitelist(self, canvas, x1, y1, x2, y2):
        """白名单管理 Tab — 30 个 Tag 切换方块"""
        cv2.line(canvas, (x1 + 8, y1 + 30), (x2 - 8, y1 + 30), self.COLOR_BORDER, 1)
        draw_text(canvas, "✅ 白名单管理", (x1 + self.MARGIN, y1 + 20), font_size=14, color=self.COLOR_ACCENT, bold=True)

        content_area = (x1 + self.MARGIN, y1 + self.MARGIN + 30,
                        x2 - self.MARGIN, y2 - self.MARGIN)

        # —— 顶部操作条 ——
        bar_y = content_area[1]
        bar_h = 34
        cv2.rectangle(canvas, (content_area[0], bar_y), (content_area[2], bar_y + bar_h), self.COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (content_area[0], bar_y), (content_area[2], bar_y + bar_h), self.COLOR_BORDER, 1)

        btn_specs = [
            ("WHITELIST_ALL",     "全选",      (80, 220, 120)),
            ("WHITELIST_CLEAR",   "清空",      (220, 80, 80)),
            ("WHITELIST_DEFAULT", "预设 0+18~29", (180, 160, 40)),
            ("WHITELIST_SAVE",    "💾 保存",   (0, 200, 180)),
        ]
        bx = content_area[0] + 14
        for bid, blabel, bcol in btn_specs:
            bw = 110 if bid != "WHITELIST_DEFAULT" else 140
            bh1, bh2 = bx, bx + bw
            by1, by2 = bar_y + 5, bar_y + bar_h - 5
            hh = self._is_hover(bh1, by1, bh2, by2)
            if hh:
                bright = tuple(min(255, int(c * 1.2)) for c in bcol)
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), bright, -1)
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), (255, 255, 255), 2)
            else:
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), bcol, -1)
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), self.COLOR_BORDER, 1)
            draw_text(canvas, blabel, (bh1 + 14, by1 + 16), font_size=11, color=(15, 17, 21), bold=True)
            self.gui_buttons.append((bid, (bh1, by1, bh2, by2), None))
            bx += bw + 8

        # 当前状态显示
        status_text = f"当前白名单: {len(self.valid_tag_ids)} 个  →  {sorted(self.valid_tag_ids) if self.valid_tag_ids else '(空 = 全量探索)'}"
        draw_text(canvas, status_text, (bx + 6, bar_y + 23), font_size=10, color=self.COLOR_TEXT_SUB)

        # —— Tag 网格 (5 列 × 6 行) ——
        grid_y = bar_y + bar_h + self.MARGIN
        cols, rows = 5, 6
        gap = 10
        avail_w = content_area[2] - content_area[0] - 28
        avail_h = content_area[3] - grid_y - 28
        card_w = (avail_w - gap * (cols - 1)) // cols
        card_h = (avail_h - gap * (rows - 1)) // rows

        for i in range(self.TAG_COUNT):
            col = i % cols
            row = i // cols
            cx = content_area[0] + 14 + col * (card_w + gap)
            cy = grid_y + row * (card_h + gap)
            is_on = i in self.valid_tag_ids
            hover_t = self._is_hover(cx, cy, cx + card_w, cy + card_h)

            if is_on:
                bg = self.COLOR_TAG_ON
                border = (255, 255, 255) if hover_t else self.COLOR_ACCENT
                bw = 3 if hover_t else 2
            else:
                bg = (100, 110, 130) if hover_t else self.COLOR_TAG_OFF
                border = self.COLOR_TEXT if hover_t else self.COLOR_BORDER
                bw = 2 if hover_t else 1

            cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), border, bw)

            # Tag ID 大号
            id_str = f"#{i:02d}"
            draw_text(canvas, id_str, (cx + card_w // 2 - 22, cy + card_h // 2 - 4),
                      font_size=16, color=(255, 255, 255), bold=True)

            # 底部状态小标签
            status = "✓ IN" if is_on else "✗ OUT"
            sc = self.COLOR_OK if is_on else self.COLOR_TEXT_SUB
            draw_text(canvas, status, (cx + card_w // 2 - 26, cy + card_h - 10),
                      font_size=9, color=sc)

            self.gui_buttons.append((f"TAG_{i}", (cx, cy, cx + card_w, cy + card_h), str(i)))

        # 底部提示
        tip = "💡 点击方块切换白名单 | 空名单 = 放行所有 30 个 Tag (探索模式)"
        draw_text(canvas, tip, (content_area[0] + 14, content_area[3] - 8),
                  font_size=10, color=self.COLOR_TEXT_SUB)

    # ================================================================
    # 主循环
    # ================================================================
    def run(self):
        self._quit_requested = False
        window_name = "tag_manager"  # 窗口 key 纯 ASCII (namedWindow ANSI API)
        self.win_mgr.setup_window(window_name, self.on_mouse)
        self.win_mgr.set_unicode_title("AprilTag 管理器")

        # 恢复 active_tab
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    root = json.load(f)
                saved_tab = root.get(self.APP_ID, {}).get("active_tab")
                if saved_tab in ("generator", "whitelist"):
                    self.active_tab = saved_tab
        except Exception as e:
            log.warning(f"恢复上次活动 Tab 设置失败: {e}")

        print("\n" + "=" * 52)
        print(" AprilTag 管理器 (cv2 GUI)")
        print("   左侧 Tab 卡片切换 | 点击退出或 [Q/ESC] 关闭")
        print("   Tab 1: 图纸生成 (import generate_tags)")
        print("   Tab 2: 白名单管理 (0~29 ID toggle)")
        print("=" * 52 + "\n")

        try:
            while True:
                if self._quit_requested:
                    break

                canvas_w = self.win_mgr.canvas_w
                canvas_h = self.win_mgr.canvas_h
                full = self._render(canvas_w, canvas_h)
                self._last_full_w = full.shape[1]
                self._last_full_h = full.shape[0]

                cv2.imshow(window_name, full)

                poll_res = self.win_mgr.poll_events()
                if poll_res.should_quit or self._quit_requested:
                    break

                raw_key = cv2.waitKeyEx(1)
                if raw_key == -1:
                    continue
                if raw_key & 0xFF in (ord('q'), 27):
                    break
                if (raw_key & 0xFF) == ord('1'):
                    self.active_tab = "generator"
                elif (raw_key & 0xFF) == ord('2'):
                    self.active_tab = "whitelist"

        finally:
            self._save_settings()
            self.win_mgr.save_settings()
            cv2.destroyAllWindows()
            log.info("AprilTag 管理器已安全退出。")


def main():
    mgr = TagManager()
    mgr.run()


if __name__ == "__main__":
    main()
