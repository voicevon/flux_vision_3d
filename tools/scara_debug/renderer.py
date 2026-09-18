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

from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text

# 逻辑画布尺寸
LOGIC_W = 1280
LOGIC_H = 800

# 调色板: 统一取自 GuiTheme 主题单源
COL_BG = GuiTheme.BG
COL_PANEL = GuiTheme.CARD_BG
COL_PANEL_HOVER = GuiTheme.CARD_HOVER
COL_BORDER = GuiTheme.BORDER
COL_ACCENT = GuiTheme.ACCENT
COL_TEXT = GuiTheme.TEXT
COL_SUB = GuiTheme.TEXT_SUB
COL_MUTED = GuiTheme.TEXT_MUTED
COL_BTN = GuiTheme.BTN
COL_BTN_BORDER = GuiTheme.BTN_BORDER
COL_BTN_HOVER = GuiTheme.BTN_HOVER
COL_OK = GuiTheme.OK
COL_WARN = GuiTheme.WARN
COL_ERR = GuiTheme.ERR
COL_GOLD = GuiTheme.GOLD

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
        """返回命中的按钮 id，未命中返回空串 (后登记的浮层按钮优先)"""
        for bid, bx, by, bw, bh in reversed(self._buttons):
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
        self._render_dropdowns(canvas, app)
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
    # 顶栏: 标题 / 串口下拉框 + 连接切换 / M84 / G28 / G92 / 退出
    # ------------------------------------------------------------------
    def _render_top_bar(self, canvas, app):
        cv2.rectangle(canvas, (0, 0), (LOGIC_W, 46), (17, 20, 26), -1)
        cv2.line(canvas, (0, 46), (LOGIC_W, 46), COL_BORDER, 1)
        cv2.circle(canvas, (24, 23), 5, COL_ACCENT, -1)
        draw_text(canvas, "SCARA 机械臂调试终端 (Flux Loader)", (40, 8),
                  font_size=16, color=COL_ACCENT, bold=True)
        draw_text(canvas, "MKS Base V1.6 / Marlin 2.0", (40, 27), font_size=11, color=COL_SUB)

        if app.mock_mode:
            draw_text(canvas, "[仿真 MOCK]", (408, 13), font_size=13, color=COL_GOLD, bold=True)

        conn = app.robot.is_connected()

        # 串口下拉框 (选端口 / 刷新端口)
        dd_label = f"串口: {app.current_port} ✓" if conn else \
            f"串口: {app.selected_port or '未连接'} ▼"
        self._button(canvas, "dd_open:serial", dd_label[:26], 496, 8, 190, 30,
                     accent=COL_OK if conn else COL_WARN)

        # 连接/断开切换按钮三态: 连接 -> 正在连接... -> 已连接 (再按断开)
        if app.connecting:
            self._button(canvas, "conn_toggle", "正在连接...", 694, 8, 92, 30, accent=COL_WARN)
        else:
            self._button(canvas, "conn_toggle", "已连接" if conn else "连接", 694, 8, 92, 30,
                         accent=COL_OK)

        # M84 释放电机
        self._button(canvas, "m84", "[M84 释放]", 794, 8, 100, 30,
                     enabled=conn, accent=COL_WARN)

        # G28 回零 / G92 设零
        self._button(canvas, "home", "[G28 回零]", 902, 8, 100, 30,
                     enabled=conn, accent=COL_ACCENT)
        self._button(canvas, "g92", "[G92 设零]", 1010, 8, 92, 30, enabled=conn)

        # 退出按钮
        self._button(canvas, "quit", "[X] 退出", 1110, 8, 158, 30, accent=COL_ERR)

    # ------------------------------------------------------------------
    # 左列: 实时状态 (含自动刷新) / 限位诊断
    # ------------------------------------------------------------------
    def _render_left_column(self, canvas, app):
        x, w = 12, 396
        # 1. 实时状态
        cy = self._panel(canvas, x, 58, w, 250, "实时状态 (M114)", accent=COL_GOLD)
        p, a = app.robot.current_pose, app.robot.current_angles
        draw_text(canvas, f"X: {p.x:8.2f} mm", (x + 20, cy), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"Y: {p.y:8.2f} mm", (x + 200, cy), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"Z: {p.z:8.2f} mm", (x + 20, cy + 30), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"R: {p.r:8.2f} deg", (x + 200, cy + 30), font_size=18, color=COL_TEXT, bold=True)
        draw_text(canvas, f"大臂 θ: {a.theta:7.2f}°", (x + 20, cy + 66), font_size=16, color=COL_SUB)
        draw_text(canvas, f"小臂 ψ: {a.psi:7.2f}°", (x + 200, cy + 66), font_size=16, color=COL_SUB)
        # 自动刷新 checkbox
        cbx, cby = x + 20, cy + 100
        checked = app.auto_refresh
        cb_hover = (cbx <= self.mouse_x <= cbx + 200 and cby <= self.mouse_y <= cby + 26)
        cv2.rectangle(canvas, (cbx, cby), (cbx + 22, cby + 22),
                      COL_PANEL_HOVER if cb_hover else COL_BTN, -1)
        cv2.rectangle(canvas, (cbx, cby), (cbx + 22, cby + 22),
                      COL_ACCENT if checked else COL_BTN_BORDER, 2 if checked else 1)
        if checked:
            cv2.line(canvas, (cbx + 5, cby + 12), (cbx + 9, cby + 16), COL_ACCENT, 2)
            cv2.line(canvas, (cbx + 9, cby + 16), (cbx + 17, cby + 5), COL_ACCENT, 2)
        draw_text(canvas, "自动刷新坐标 (0.3s)", (cbx + 32, cby + 3),
                  font_size=13, color=COL_TEXT if checked else COL_SUB, bold=checked)
        self._buttons.append(("auto_refresh", cbx, cby, 200, 26))
        draw_text(canvas, f"步长: {app.jog.step_info}", (cbx, cby + 36),
                  font_size=12, color=COL_MUTED)
        self._button(canvas, "refresh_pos", "[手动刷新 M114]", cbx, cby + 62, 200, 28)

        # 2. 限位诊断
        cy = self._panel(canvas, x, 320, w, 250, "限位与传感器 (M119)", accent=COL_WARN)
        self._button(canvas, "m119", "[限位诊断 M119]", x + 10, cy, 180, 28)
        lines = app.limit_lines[-7:] if app.limit_lines else ["(点击上方按钮执行诊断)"]
        for i, ln in enumerate(lines):
            col = COL_OK if "open" in ln.lower() else (COL_ERR if "triggered" in ln.lower() else COL_SUB)
            draw_text(canvas, ln[:46], (x + 14, cy + 36 + i * 26), font_size=12, color=col)

    # ------------------------------------------------------------------
    # 中列: 步长 / 点动 / Z 快捷 / 夹爪
    # ------------------------------------------------------------------
    def _render_mid_column(self, canvas, app):
        x, w = 420, 396
        conn = app.robot.is_connected()

        # 1. 步长档位 (下拉框)
        cy = self._panel(canvas, x, 58, w, 70, "步长档位 (Jog Step)")
        step_label = f"步长: {app.jog.step_linear_mm:g}mm / {app.jog.step_rot_deg:g}°"
        self._button(canvas, "dd_open:step", f"{step_label} {'▲' if app.dd_step_open else '▼'}",
                     x + 10, cy, w - 20, 30)

        # 2. 笛卡尔点动十字盘 (X± 左右 / Y± 上下 / Z 竖条右侧 Z+上 Z-下)
        cy = self._panel(canvas, x, 140, w, 210, "笛卡尔点动十字盘 (Cartesian Jog)")
        bw, bh = 80, 38
        c0, c2, cz = x + 10, x + 228, x + 316
        cx, cw = x + 90, 136  # 中央坐标格 (加宽, 深蓝高亮)
        c_mid = cx + (cw - bw) // 2  # Y± 按钮在中央格上方居中
        r1, r2, r3 = cy, cy + 42, cy + 84
        self._button(canvas, "jog:w", "Y+ (W)", c_mid, r1, bw, bh, enabled=conn, accent=COL_ACCENT)
        self._button(canvas, "jog:u", "Z+ (U)", cz, r1, 70, bh, enabled=conn)
        self._button(canvas, "jog:a", "X- (A)", c0, r2, bw, bh, enabled=conn)
        # 中央格: 实时坐标向量 [X, Y, Z], R°
        p = app.robot.current_pose
        coord_text = f"[{p.x:.1f},{p.y:.1f},{p.z:.1f}],{p.r:.1f}°"
        cv2.rectangle(canvas, (cx, r2), (cx + cw, r2 + bh), (30, 55, 85), -1)
        cv2.rectangle(canvas, (cx, r2), (cx + cw, r2 + bh), (90, 160, 220), 1)
        draw_text(canvas, coord_text, (cx + 3, r2 + 13), font_size=9, color=(150, 210, 255), bold=True)
        self._button(canvas, "jog:d", "X+ (D)", c2, r2, bw, bh, enabled=conn, accent=COL_ACCENT)
        self._button(canvas, "jog:j", "Z- (J)", cz, r2, 70, bh, enabled=conn)
        self._button(canvas, "jog:s", "Y- (S)", c_mid, r3, bw, bh, enabled=conn, accent=COL_ACCENT)
        # R 轴旋转
        ry = cy + 130
        self._button(canvas, "jog:q", "R+ 旋+ (Q)", c0, ry, 136, 34, enabled=conn)
        self._button(canvas, "jog:e", "R- 旋- (E)", x + 154, ry, 136, 34, enabled=conn)

        # 3. 关节角点动
        cy = self._panel(canvas, x, 362, w, 114, "关节角独立点动 (Joint Jog)")
        self._button(canvas, "jog:o", "O: 大臂θ+", x + 10, cy, 180, 36, enabled=conn)
        self._button(canvas, "jog:l", "L: 大臂θ-", x + 202, cy, 180, 36, enabled=conn)
        self._button(canvas, "jog:i", "I: 小臂ψ+", x + 10, cy + 42, 180, 36, enabled=conn)
        self._button(canvas, "jog:k", "K: 小臂ψ-", x + 202, cy + 42, 180, 36, enabled=conn)

        # 4. Z 轴快捷 + 夹爪
        cy = self._panel(canvas, x, 488, w, 162, "Z 轴快捷与夹爪舵机 (End-Effector)")
        self._button(canvas, "z_up", "Z升至 100", x + 10, cy, 118, 30, enabled=conn)
        self._button(canvas, "z_down", "Z降至 20", x + 134, cy, 118, 30, enabled=conn)
        z_label = "指定 Z ▲" if app.dd_z_open else "指定 Z ▼"
        self._button(canvas, "dd_open:z", z_label, x + 258, cy, 118, 30, enabled=conn)
        gy = cy + 40
        self._button(canvas, "grip_close", "双夹爪闭合", x + 10, gy, 118, 30, enabled=conn, accent=COL_WARN)
        self._button(canvas, "grip_open", "双夹爪打开", x + 134, gy, 118, 30, enabled=conn, accent=COL_OK)
        self._button(canvas, "grip1_close", "夹1闭", x + 258, gy, 57, 30, enabled=conn)
        self._button(canvas, "grip1_open", "夹1开", x + 319, gy, 57, 30, enabled=conn)
        self._button(canvas, "grip2_close", "夹2闭", x + 258, gy + 36, 57, 30, enabled=conn)
        self._button(canvas, "grip2_open", "夹2开", x + 319, gy + 36, 57, 30, enabled=conn)

    # ------------------------------------------------------------------
    # 右列: 直达 / 工位 / 宏 / 透传
    # ------------------------------------------------------------------
    def _render_right_column(self, canvas, app):
        x, w = 828, 440
        conn = app.robot.is_connected()

        # 1. 直达坐标 (G28/G92 已上移顶栏)
        cy = self._panel(canvas, x, 58, w, 66, "直达目标坐标 (G1 X Y Z R F)")
        self._button(canvas, "goto", "输入目标坐标 (如 X100 Y300 Z50 R0)...",
                     x + 10, cy, w - 20, 30, enabled=conn, accent=COL_GOLD)

        # 2. 工位跳转
        presets = app.presets.list_presets()
        n_show = min(len(presets), 5)
        cy = self._panel(canvas, x, 140, w, 76 + n_show * 34, "预设工位跳转 (Workstations)")
        keys = list(presets.keys())
        for i, name in enumerate(keys[:5]):
            pose = presets[name]
            label = f"{name[:12]} ({pose.x:.0f},{pose.y:.0f},{pose.z:.0f},{pose.r:.0f})"
            self._button(canvas, f"preset:{i}", label, x + 10, cy + i * 34, w - 96, 30, enabled=conn)
            self._button(canvas, f"preset_del:{i}", "删", x + w - 78, cy + i * 34, 28, 30,
                         enabled=True, accent=COL_ERR, hoverable=True)
        by = cy + n_show * 34 + 4
        self._button(canvas, "preset_save", "[存当前位置为预设]", x + 10, by, 200, 28)

        # 3. 搬运宏
        macro_top = 140 + 76 + n_show * 34 + 40
        cy = self._panel(canvas, x, macro_top, w, 100, "芦笋搬运节拍宏 (Pick & Place)")
        self._button(canvas, "macro_minus", "-", x + 10, cy, 36, 32, enabled=conn)
        draw_text(canvas, f"x{app.macro_cycles}", (x + 56, cy + 6), font_size=18,
                  color=COL_GOLD, bold=True)
        self._button(canvas, "macro_plus", "+", x + 108, cy, 36, 32, enabled=conn)
        self._button(canvas, "macro_run", f"[执行 {app.macro_cycles} 次搬运循环]",
                     x + 156, cy, 272, 32, enabled=conn, accent=COL_WARN)

        # 4. G-code 透传
        cy = self._panel(canvas, x, macro_top + 112, w, 66, "原生 G-code 透传 (Raw Terminal)")
        self._button(canvas, "gcode_input", "输入 G-code 指令透传至 Marlin (如 M119 / G28)...",
                     x + 10, cy, w - 20, 30, enabled=conn)

    # ------------------------------------------------------------------
    # 下拉框浮层 (最后绘制覆盖基础层; 后登记按钮使命中优先)
    # ------------------------------------------------------------------
    def _render_dropdowns(self, canvas, app):
        if app.dd_serial_open:
            px, pw = 496, 190
            ports = app.port_list
            h = 28 + 4 + len(ports) * 24 + 8
            cv2.rectangle(canvas, (px, 44), (px + pw, 44 + h), (24, 30, 38), -1)
            cv2.rectangle(canvas, (px, 44), (px + pw, 44 + h), COL_ACCENT, 1)
            cy = 50
            conn = app.robot.is_connected()
            self._button(canvas, "dd_serial:__refresh__", "↻ 刷新端口", px + 4, cy, pw - 8, 26)
            cy += 32
            cv2.line(canvas, (px + 4, cy - 4), (px + pw - 4, cy - 4), COL_BORDER, 1)
            if not ports:
                draw_text(canvas, "(无串口设备)", (px + 12, cy), font_size=12, color=COL_MUTED)
            for p in ports[:9]:
                mark = "✓ " if (conn and p == app.current_port) else "  "
                self._button(canvas, f"dd_serial:{p}", f"{mark}{p}", px + 4, cy, pw - 8, 22)
                cy += 24

        if app.dd_step_open:
            px, pw = 430, 376
            py, ph = 96, 3 * 32 + 8
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (24, 30, 38), -1)
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COL_ACCENT, 1)
            cy = py + 4
            for key, lin, rot in (("1", 1, 1), ("2", 10, 5), ("3", 50, 15)):
                active = (app.jog.step_linear_mm == float(lin))
                mark = "✓ " if active else "  "
                self._button(canvas, f"dd_step:{key}", f"{mark}[{key}]  {lin}mm / {rot}°",
                             px + 4, cy, pw - 8, 28, accent=COL_ACCENT if active else None)
                cy += 32

        if app.dd_z_open:
            px, pw = 678, 118
            n = 11
            ph = n * 24 + 8
            py = 514 - ph  # 自按钮底部向上展开
            cv2.rectangle(canvas, (px, py), (px + pw, 514), (24, 30, 38), -1)
            cv2.rectangle(canvas, (px, py), (px + pw, 514), COL_ACCENT, 1)
            cur_z = round(app.robot.current_pose.z / 10.0) * 10
            cy = py + 4
            for zv in range(0, 101, 10):
                mark = "✓ " if zv == cur_z else "  "
                self._button(canvas, f"dd_z:{zv}", f"{mark}{zv} mm", px + 4, cy, pw - 8, 22)
                cy += 24

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
