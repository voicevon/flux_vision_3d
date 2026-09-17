# -*- coding: utf-8 -*-
"""
SCARA 机械臂调试终端 — 纯渲染层 (ScaraDebugRenderer)
====================================================
按 flux_vision_3d 深色工业风在 1280x800 逻辑画布上绘制三列布局：
  左列: 串口连接 / 实时状态 / 限位诊断
  中列: 步长档位 / 笛卡尔与关节点动 / Z 轴快捷 / 夹爪舵机
  右列: 原点与维护 / 直达坐标 / 工位跳转 / 搬运宏 / G-code 透传
底部: 滚动日志区
无任何业务逻辑，按钮命中区域由 buttons() 统一给出。
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.utils.text_rendering import draw_text

# 逻辑画布尺寸
LOGIC_W = 1280
LOGIC_H = 800

# 调色板 (与 Suite Dashboard 一致的低饱和冷色系)
COL_BG = (15, 17, 21)
COL_PANEL = (22, 26, 33)
COL_PANEL_HOVER = (30, 38, 50)
COL_BORDER = (38, 46, 58)
COL_ACCENT = (0, 210, 180)
COL_TEXT = (242, 245, 248)
COL_SUB = (155, 170, 185)
COL_MUTED = (115, 130, 145)
COL_BTN = (28, 34, 44)
COL_BTN_BORDER = (52, 64, 80)
COL_BTN_HOVER = (40, 52, 66)
COL_OK = (80, 190, 115)
COL_WARN = (220, 145, 60)
COL_ERR = (210, 80, 80)
COL_GOLD = (210, 175, 60)

Btn = Tuple[str, int, int, int, int]  # (btn_id, x, y, w, h)


class ScaraDebugRenderer:
    """SCARA 调试终端纯渲染器：布局绘制与按钮命中区域定义"""

    def __init__(self):
        self.mouse_x = -1
        self.mouse_y = -1
        self._buttons: List[Btn] = []

    # ------------------------------------------------------------------
    # 命中区域
    # ------------------------------------------------------------------
    @property
    def buttons(self) -> List[Btn]:
        return self._buttons

    def hit_test(self, x: int, y: int) -> str:
        """返回命中的按钮 id，未命中返回空串"""
        for bid, bx, by, bw, bh in self._buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return bid
        return ""

    # ------------------------------------------------------------------
    # 主渲染入口
    # ------------------------------------------------------------------
    def render(self, app) -> np.ndarray:
        canvas = np.full((LOGIC_H, LOGIC_W, 3), COL_BG, dtype=np.uint8)
        self._buttons = []

        self._render_top_bar(canvas, app)
        self._render_left_column(canvas, app)
        self._render_mid_column(canvas, app)
        self._render_right_column(canvas, app)
        self._render_log_area(canvas, app)
        self._render_footer(canvas, app)
        return canvas

    # ------------------------------------------------------------------
    # 基础绘制元件
    # ------------------------------------------------------------------
    def _panel(self, canvas, x, y, w, h, title: str, accent=COL_ACCENT) -> int:
        """绘制区块面板与标题，返回内容起始 y"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_BORDER, 1)
        cv2.rectangle(canvas, (x, y), (x + 4, y + h), accent, -1)
        draw_text(canvas, title, (x + 12, y + 7), font_size=13, color=(192, 206, 222), bold=True)
        return y + 32

    def _button(self, canvas, bid: str, label: str, x, y, w, h,
                enabled: bool = True, accent=None, hoverable: bool = True) -> None:
        """绘制单个按钮并登记命中区"""
        self._buttons.append((bid, x, y, w, h))
        hovered = hoverable and (x <= self.mouse_x <= x + w and y <= self.mouse_y <= y + h)
        bg = COL_BTN_HOVER if hovered else COL_BTN
        border = (accent or COL_BTN_BORDER) if hovered else COL_BTN_BORDER
        if not enabled:
            bg, border = (20, 23, 28), (32, 38, 46)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)
        tcol = COL_MUTED if not enabled else ((accent or COL_TEXT) if hovered else (205, 215, 225))
        # 文本居中
        font = cv2.FONT_HERSHEY_SIMPLEX
        draw_text(canvas, label, (x + 8, y + (h - 16) // 2 + 2), font_size=13, color=tcol,
                  bold=hovered)

    # ------------------------------------------------------------------
    # 顶栏
    # ------------------------------------------------------------------
    def _render_top_bar(self, canvas, app):
        cv2.rectangle(canvas, (0, 0), (LOGIC_W, 46), (17, 20, 26), -1)
        cv2.line(canvas, (0, 46), (LOGIC_W, 46), COL_BORDER, 1)
        cv2.circle(canvas, (24, 23), 5, COL_ACCENT, -1)
        draw_text(canvas, "SCARA 机械臂调试终端 (Flux Loader)", (40, 8),
                  font_size=16, color=COL_ACCENT, bold=True)
        draw_text(canvas, "MKS Base V1.6 / Marlin 2.0", (40, 27), font_size=11, color=COL_SUB)

        # 连接状态胶囊
        conn = app.robot.is_connected()
        ccol = COL_OK if conn else COL_WARN
        ctext = f"已连接: {app.current_port}" if conn else "未连接"
        cv2.rectangle(canvas, (560, 8), (790, 38), (20, 26, 34), -1)
        cv2.rectangle(canvas, (560, 8), (790, 38), ccol, 1)
        cv2.circle(canvas, (576, 23), 4, ccol, -1)
        draw_text(canvas, ctext, (588, 13), font_size=13, color=ccol, bold=True)

        if app.mock_mode:
            draw_text(canvas, "[仿真 MOCK]", (806, 13), font_size=13, color=COL_GOLD, bold=True)

        # 退出按钮
        self._button(canvas, "quit", "[X] 退出 [ESC]", 1120, 8, 148, 30, accent=COL_ERR)

    # ------------------------------------------------------------------
    # 左列: 串口 / 状态 / 限位
    # ------------------------------------------------------------------
    def _render_left_column(self, canvas, app):
        x, w = 12, 396
        # 1. 串口连接
        cy = self._panel(canvas, x, 58, w, 240, "串口连接 (Serial)")
        ports = app.port_list
        for i, p in enumerate(ports[:5]):
            sel = (p == app.selected_port)
            bg = (24, 40, 44) if sel else COL_BTN
            border = COL_ACCENT if sel else COL_BTN_BORDER
            by = cy + i * 30
            cv2.rectangle(canvas, (x + 10, by), (x + w - 10, by + 26), bg, -1)
            cv2.rectangle(canvas, (x + 10, by), (x + w - 10, by + 26), border, 1)
            mark = "* " if sel else "  "
            draw_text(canvas, f"{mark}{p}", (x + 20, by + 5), font_size=13,
                      color=COL_TEXT if sel else COL_SUB, bold=sel)
            self._buttons.append((f"port:{p}", x + 10, by, w - 20, 26))
        if not ports:
            draw_text(canvas, "(未检测到串口，可输入手动连接)", (x + 20, cy + 6),
                      font_size=12, color=COL_MUTED)
        by = cy + 155
        self._button(canvas, "refresh_ports", "[刷新端口]", x + 10, by, 118, 30)
        self._button(canvas, "connect", "[连接]", x + 134, by, 118, 30,
                     enabled=not app.robot.is_connected() and bool(app.selected_port),
                     accent=COL_OK)
        self._button(canvas, "disconnect", "[断开]", x + 258, by, 118, 30,
                     enabled=app.robot.is_connected(), accent=COL_ERR)

        # 2. 实时状态
        cy = self._panel(canvas, x, 310, w, 190, "实时状态 (M114)", accent=COL_GOLD)
        p, a = app.robot.current_pose, app.robot.current_angles
        draw_text(canvas, f"X: {p.x:8.2f} mm", (x + 20, cy), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"Y: {p.y:8.2f} mm", (x + 200, cy), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"Z: {p.z:8.2f} mm", (x + 20, cy + 30), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"R: {p.r:8.2f} deg", (x + 200, cy + 30), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"大臂 θ: {a.theta:7.2f}°", (x + 20, cy + 66), font_size=16, color=COL_SUB)
        draw_text(canvas, f"小臂 ψ: {a.psi:7.2f}°", (x + 200, cy + 66), font_size=16, color=COL_SUB)
        self._button(canvas, "refresh_pos", "[刷新坐标 M114]", x + 20, cy + 100, 200, 30)
        draw_text(canvas, f"步长: {app.jog.step_info}", (x + 20, cy + 140),
                  font_size=12, color=COL_MUTED)

        # 3. 限位诊断
        cy = self._panel(canvas, x, 512, w, 168, "限位与传感器 (M119)", accent=COL_WARN)
        self._button(canvas, "m119", "[限位诊断 M119]", x + 10, cy, 180, 28)
        lines = app.limit_lines[-4:] if app.limit_lines else ["(点击上方按钮执行诊断)"]
        for i, ln in enumerate(lines):
            col = COL_OK if "open" in ln.lower() else (COL_ERR if "triggered" in ln.lower() else COL_SUB)
            draw_text(canvas, ln[:46], (x + 14, cy + 36 + i * 26), font_size=12, color=col)

    # ------------------------------------------------------------------
    # 中列: 步长 / 点动 / Z 快捷 / 夹爪
    # ------------------------------------------------------------------
    def _render_mid_column(self, canvas, app):
        x, w = 420, 396
        conn = app.robot.is_connected()

        # 1. 步长档位
        cy = self._panel(canvas, x, 58, w, 92, "步长档位 (Jog Step)")
        for i, (key, label) in enumerate([("1", "1mm/1°"), ("2", "10mm/5°"), ("3", "50mm/15°")]):
            active = (app.jog.step_linear_mm == float(label.split("m")[0]) or
                      (key == "1" and app.jog.step_linear_mm == 1.0) or
                      (key == "2" and app.jog.step_linear_mm == 10.0) or
                      (key == "3" and app.jog.step_linear_mm == 50.0))
            self._button(canvas, f"step:{key}", f"[{key}] {label}", x + 10 + i * 126, cy, 120, 30,
                         accent=COL_ACCENT if active else None)

        # 2. 笛卡尔点动 (2 列 × 4 行: Y/W-S, X/A-D, Z/U-J, R/Q-E)
        cy = self._panel(canvas, x, 162, w, 218, "笛卡尔点动 (Cartesian Jog)")
        rows = [
            ("jog:w", "W: Y+ 前进", "jog:s", "S: Y- 后退"),
            ("jog:d", "D: X+ 右移", "jog:a", "A: X- 左移"),
            ("jog:u", "U: Z+ 升", "jog:j", "J: Z- 降"),
            ("jog:q", "Q: R+ 旋+", "jog:e", "E: R- 旋-"),
        ]
        for r, (lft, llabel, rgt, rlabel) in enumerate(rows):
            ry = cy + r * 44
            self._button(canvas, lft, llabel, x + 10, ry, 180, 36, enabled=conn)
            self._button(canvas, rgt, rlabel, x + 202, ry, 180, 36, enabled=conn)

        # 3. 关节点动
        cy = self._panel(canvas, x, 392, w, 114, "关节角独立点动 (Joint Jog)")
        self._button(canvas, "jog:o", "O: 大臂θ+", x + 10, cy, 180, 36, enabled=conn)
        self._button(canvas, "jog:l", "L: 大臂θ-", x + 202, cy, 180, 36, enabled=conn)
        self._button(canvas, "jog:i", "I: 小臂ψ+", x + 10, cy + 42, 180, 36, enabled=conn)
        self._button(canvas, "jog:k", "K: 小臂ψ-", x + 202, cy + 42, 180, 36, enabled=conn)

        # 4. Z 轴快捷 + 夹爪
        cy = self._panel(canvas, x, 518, w, 162, "Z 轴快捷与夹爪舵机 (End-Effector)")
        self._button(canvas, "z_up", "Z升至 100", x + 10, cy, 118, 30, enabled=conn)
        self._button(canvas, "z_down", "Z降至 20", x + 134, cy, 118, 30, enabled=conn)
        self._button(canvas, "z_input", "指定 Z", x + 258, cy, 118, 30, enabled=conn)
        gy = cy + 40
        self._button(canvas, "grip_close", "双夹爪闭合", x + 10, gy, 118, 30, enabled=conn, accent=COL_WARN)
        self._button(canvas, "grip_open", "双夹爪打开", x + 134, gy, 118, 30, enabled=conn, accent=COL_OK)
        self._button(canvas, "grip1_close", "夹1闭", x + 258, gy, 57, 30, enabled=conn)
        self._button(canvas, "grip1_open", "夹1开", x + 319, gy, 57, 30, enabled=conn)
        self._button(canvas, "grip2_close", "夹2闭", x + 258, gy + 36, 57, 30, enabled=conn)
        self._button(canvas, "grip2_open", "夹2开", x + 319, gy + 36, 57, 30, enabled=conn)
        self._button(canvas, "m84", "[M84 释放电机]", x + 10, gy + 36, 236, 30, enabled=conn)

    # ------------------------------------------------------------------
    # 右列: 原点维护 / 直达 / 工位 / 宏 / 透传
    # ------------------------------------------------------------------
    def _render_right_column(self, canvas, app):
        x, w = 828, 440
        conn = app.robot.is_connected()

        # 1. 原点与维护
        cy = self._panel(canvas, x, 58, w, 100, "原点标定与维护 (Homing)")
        self._button(canvas, "home", "[G28 一键三轴回零]", x + 10, cy, 200, 32, enabled=conn, accent=COL_ACCENT)
        self._button(canvas, "g92", "[G92 设当前位置为零点]", x + 220, cy, 208, 32, enabled=conn)
        self._button(canvas, "reconnect", "[重连/切换串口]", x + 10, cy + 38, 200, 28)

        # 2. 直达坐标
        cy = self._panel(canvas, x, 170, w, 66, "直达目标坐标 (G1 X Y Z R F)")
        self._button(canvas, "goto", "输入目标坐标 (如 X100 Y300 Z50 R0)...",
                     x + 10, cy, w - 20, 30, enabled=conn, accent=COL_GOLD)

        # 3. 工位跳转
        presets = app.presets.list_presets()
        n_show = min(len(presets), 5)
        cy = self._panel(canvas, x, 248, w, 60 + n_show * 34, "预设工位跳转 (Workstations)")
        keys = list(presets.keys())
        for i, name in enumerate(keys[:5]):
            pose = presets[name]
            label = f"{name[:12]} ({pose.x:.0f},{pose.y:.0f},{pose.z:.0f},{pose.r:.0f})"
            self._button(canvas, f"preset:{i}", label, x + 10, cy + i * 34, w - 96, 30, enabled=conn)
            self._button(canvas, f"preset_del:{i}", "删", x + w - 78, cy + i * 34, 28, 30,
                         enabled=True, accent=COL_ERR, hoverable=True)
        by = cy + n_show * 34 + 4
        self._button(canvas, "preset_save", "[存当前位置为预设]", x + 10, by, 200, 28)

        # 4. 搬运宏
        macro_top = 248 + 60 + n_show * 34 + 40
        cy = self._panel(canvas, x, macro_top, w, 100, "芦笋搬运节拍宏 (Pick & Place)")
        self._button(canvas, "macro_minus", "-", x + 10, cy, 36, 32, enabled=conn)
        draw_text(canvas, f"x{app.macro_cycles}", (x + 56, cy + 6), font_size=18,
                  color=COL_GOLD, bold=True)
        self._button(canvas, "macro_plus", "+", x + 108, cy, 36, 32, enabled=conn)
        self._button(canvas, "macro_run", f"[执行 {app.macro_cycles} 次搬运循环]",
                     x + 156, cy, 272, 32, enabled=conn, accent=COL_WARN)

        # 5. G-code 透传
        cy = self._panel(canvas, x, macro_top + 112, w, 66, "原生 G-code 透传 (Raw Terminal)")
        self._button(canvas, "gcode_input", "输入 G-code 指令透传至 Marlin (如 M119 / G28)...",
                     x + 10, cy, w - 20, 30, enabled=conn)

    # ------------------------------------------------------------------
    # 日志区与底栏
    # ------------------------------------------------------------------
    def _render_log_area(self, canvas, app):
        x, y, w, h = 12, 692, 1256, 84
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (16, 19, 24), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_BORDER, 1)
        draw_text(canvas, "通信日志", (x + 10, y + 4), font_size=11, color=COL_MUTED, bold=True)
        lines = list(app.log_lines)[-3:]
        for i, ln in enumerate(lines):
            col = COL_SUB
            if ln.startswith("<"):
                col = (120, 200, 160)
            elif "ERR" in ln or "失败" in ln:
                col = COL_ERR
            elif ln.startswith(">"):
                col = (150, 190, 230)
            draw_text(canvas, ln[:88], (x + 10, y + 22 + i * 20), font_size=12, color=col)

    def _render_footer(self, canvas, app):
        y = LOGIC_H - 22
        cv2.line(canvas, (0, y - 2), (LOGIC_W, y - 2), COL_BORDER, 1)
        draw_text(canvas,
                  "键盘: W/S/A/D/X/Y 点动 | Q/E 旋转 | O/L 大臂 | I/K 小臂 | 1/2/3 步长 | 空格 刷新坐标 | ESC 退出",
                  (12, y + 2), font_size=12, color=COL_MUTED)
        draw_text(canvas, f"日志 {len(app.log_lines)} 条",
                  (LOGIC_W - 140, y + 2), font_size=12, color=COL_MUTED)
