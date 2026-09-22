# -*- coding: utf-8 -*-
"""
Isolator WHEELS 调试视窗 (flux_isolate_wheels 分离轮 ESP32 MQTT 调试 GUI)
==========================================================================
协议依据: flux_isolate_wheels/doc/通讯协议.md (v1.0)
  - 连接: MQTT voicevon.vicp.io:1883 (von), keepalive 15s
  - 订阅: flux/loader/+/state | /done | /log (通配符多设备发现, 保留消息 + 遗嘱判活)
  - 下发: flux/loader/{devid}/cmd  {"cmd":"load","counts":[int x8]}  (1~8 号托架)
  - 稳妥模式: 仅 state=idle 受理命令, running 期间发送按钮置灰
纯 cv2 矢量 GUI (全鼠标化), ESC/红叉退出。
"""

import os
import sys
import json
import time
import threading
from collections import deque
from typing import Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402
from src.utils.gui_window_manager import GuiWindowManager  # noqa: E402
from src.utils.text_rendering import draw_text  # noqa: E402

log = get_logger(__name__)

# ==================== 协议常量 (flux_isolate_wheels/doc/通讯协议.md) ====================
BROKER_HOST = "voicevon.vicp.io"
BROKER_PORT = 1883
BROKER_USER = "von"
BROKER_PASS = "von123456"
TOPIC_PREFIX = "flux/loader"
DEFAULT_DEVID = "F8EC"
KEEPALIVE_S = 15

# ==================== 逻辑画布与几何常量 (渲染与命中测试单源) ====================
LOGIC_W, LOGIC_H = 1100, 720

BTN_CONNECT = (860, 18, 78, 34)
BTN_DISCONNECT = (946, 18, 78, 34)
BTN_QUIT = (1032, 18, 48, 34)

DEV_CHIP_X0, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H, DEV_CHIP_STEP = 110, 66, 100, 30, 106
DEV_CHIP_MAX = 8

STATUS_CARD = (20, 106, 500, 130)
STEPPER_X0, STEPPER_Y0, STEPPER_W, STEPPER_STEP = 24, 272, 56, 60
STEPPER_VAL_H = 46          # 数值显示框高
STEPPER_BTN_W, STEPPER_BTN_H = 27, 30
STEPPER_LABEL_H = 20

BTN_CLEAR = (24, 396, 130, 44)
BTN_SEND = (166, 396, 300, 44)

CMD_PREVIEW_CARD = (20, 456, 500, 74)
PROTO_INFO_CARD = (20, 542, 500, 140)

LOG_CARD = (536, 106, 544, 576)
LOG_HEADER_H = 30
LOG_LINE_H = 26

# 状态配色 (BGR): idle 绿 / running 琥珀 / offline 红 / 未上线 灰
STATE_COLORS = {
    "idle": (0, 200, 120),
    "running": (0, 170, 255),
    "offline": (80, 80, 255),
}
STATE_TEXTS = {
    "idle": "IDLE 空闲 (可受理命令)",
    "running": "RUNNING 节拍执行中",
    "offline": "OFFLINE 掉线 (遗嘱)",
}
COLOR_BG = (18, 20, 24)
COLOR_PANEL = (26, 30, 38)
COLOR_BORDER = (52, 60, 74)
COLOR_ACCENT = (0, 200, 240)
COLOR_GREEN = (0, 220, 140)
COLOR_TEXT = (230, 236, 244)
COLOR_SUB = (160, 172, 188)
COLOR_MUTED = (110, 120, 136)


class IsolateWheelsDebuggerApp:
    """分离轮 MQTT 调试主应用: 设备发现/状态监视/8 托架节拍命令下发/日志流"""

    def __init__(self, settings_file: Optional[str] = None):
        self.win_mgr = GuiWindowManager(
            app_id="isolate_wheels_debug",
            base_w=LOGIC_W, base_h=LOGIC_H,
            settings_file=settings_file,
            enable_keyboard_zoom=False  # 全鼠标化: 不启用键盘缩放热键
        )
        self.window_name = "flux_vision_3d | isolate_wheels"
        self.window_title = "flux_vision_3d | Isolator WHEELS 调试"
        self._running = True

        # MQTT 运行态 (回调来自网络线程, 共享读写加锁)
        self._lock = threading.Lock()
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._connecting = False
        self.devices: dict = {}                 # devid -> state (idle/running/offline)
        self.selected_devid: str = DEFAULT_DEVID
        self.done_count = 0
        self.last_done_time = ""
        self.last_cmd_json = ""
        self.last_publish_msg = ""
        self.log_lines: deque = deque(maxlen=300)
        self.counts: list = [0] * 8             # 1~8 号托架芦笋数量草稿

        # UI 态
        self.mouse_x, self.mouse_y = -1, -1
        self._toast = ""
        self._toast_until = 0.0

    # ==================== MQTT 层 ====================
    def connect_broker(self):
        """异步连接 Broker 并订阅通配主题 (loop_start 内部线程自动重连)"""
        if self._connected or self._connecting:
            return
        self._connecting = True
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"flux_vision_wheels_{os.getpid()}",
                protocol=mqtt.MQTTv311,
            )
            client.username_pw_set(BROKER_USER, BROKER_PASS)
            client.reconnect_delay_set(min_delay=2, max_delay=10)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message
            client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=KEEPALIVE_S)
            client.loop_start()
            self._client = client
            self._set_toast(f"正在连接 Broker {BROKER_HOST}:{BROKER_PORT} ...")
            log.info(f"[WHEELS] 连接 Broker: {BROKER_HOST}:{BROKER_PORT}")
        except Exception as e:
            self._connecting = False
            self._set_toast(f"MQTT 连接失败: {e}")
            log.warning(f"[WHEELS] MQTT 连接失败: {e}")

    def disconnect_broker(self):
        """主动断开并停掉网络线程"""
        if self._client is not None:
            try:
                self._client.disconnect()
                self._client.loop_stop()
            except Exception:
                pass
        self._client = None
        self._connected = False
        self._connecting = False
        self._set_toast("已断开 Broker 连接。")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        ok = not getattr(reason_code, "is_failure", int(reason_code) != 0)
        self._connected = ok
        self._connecting = False
        if ok:
            client.subscribe([
                (f"{TOPIC_PREFIX}/+/state", 0),
                (f"{TOPIC_PREFIX}/+/done", 0),
                (f"{TOPIC_PREFIX}/+/log", 0),
            ])
            self._set_toast("Broker 已连接, 订阅 flux/loader/+/* (等待设备保留消息...)")
            log.info("[WHEELS] Broker 已连接并完成订阅")
        else:
            self._set_toast(f"Broker 连接被拒: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected = False
        self._set_toast(f"Broker 连接断开: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """路由 flux/loader/{devid}/{state|done|log} 上行消息 (网络线程)"""
        parts = msg.topic.split("/")
        if len(parts) != 4 or parts[0] != "flux" or parts[1] != "loader":
            return
        devid, kind = parts[2], parts[3]
        payload = msg.payload.decode("utf-8", errors="replace")
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                data = {}
        except (ValueError, TypeError):
            data = {}
        with self._lock:
            if kind == "state":
                st = str(data.get("state", "")).strip()
                if st:
                    prev = self.devices.get(devid)
                    self.devices[devid] = st
                    if st == "offline" and prev not in (None, "offline"):
                        self.log_lines.append(f"[SYS] 设备 {devid} 遗嘱: 异常掉线")
            elif kind == "done":
                self.done_count += 1
                self.last_done_time = time.strftime("%H:%M:%S")
            elif kind == "log":
                self.log_lines.append(f"[{devid}] {payload}")

    def send_load(self) -> bool:
        """下发一个 load 节拍 (稳妥模式: 仅连接且设备 idle 时放行)"""
        if not self._connected:
            self._set_toast("尚未连接 Broker, 无法下发命令。")
            return False
        st = self.devices.get(self.selected_devid, "")
        if st != "idle":
            self._set_toast(f"设备 {self.selected_devid} 状态 {st or '未知'} 非 idle, 命令会被忽略 (稳妥模式)。")
            return False
        payload = json.dumps({"cmd": "load", "counts": list(self.counts)}, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{self.selected_devid}/cmd"
        try:
            self._client.publish(topic, payload, qos=0)
        except Exception as e:
            self._set_toast(f"发送失败: {e}")
            return False
        self.last_cmd_json = payload
        self.last_publish_msg = f"{time.strftime('%H:%M:%S')} 已发送 -> {topic}"
        self._set_toast(f"节拍命令已下发至 {self.selected_devid}: {payload}")
        log.info(f"[WHEELS] {self.last_publish_msg}: {payload}")
        return True

    # ==================== 交互逻辑 ====================
    def _set_toast(self, msg: str):
        self._toast = msg
        self._toast_until = time.time() + 4.0

    def _device_state(self, devid: str) -> str:
        with self._lock:
            return self.devices.get(devid, "")

    def _handle_click(self, x: int, y: int):
        """逻辑坐标左键点击分发 (纯逻辑, 可单元测试)"""
        # 1. 顶栏按钮
        if self._pt_in(x, y, BTN_CONNECT):
            self.connect_broker()
            return
        if self._pt_in(x, y, BTN_DISCONNECT):
            self.disconnect_broker()
            return
        if self._pt_in(x, y, BTN_QUIT):
            self._running = False
            return

        # 2. 设备芯片选择
        if DEV_CHIP_Y <= y <= DEV_CHIP_Y + DEV_CHIP_H:
            dev_ids = list(self.devices.keys()) or [self.selected_devid]
            for i in range(min(len(dev_ids), DEV_CHIP_MAX)):
                if self._pt_in(x, y, self._device_chip_rect(i)):
                    self.selected_devid = dev_ids[i]
                    return

        # 3. 托架步进器 [-] / [+]
        for idx in range(8):
            minus, plus = self._stepper_btn_rects(idx)
            if self._pt_in(x, y, minus):
                self.counts[idx] = max(0, self.counts[idx] - 1)
                return
            if self._pt_in(x, y, plus):
                self.counts[idx] = min(9, self.counts[idx] + 1)
                return

        # 4. 命令按钮
        if self._pt_in(x, y, BTN_CLEAR):
            self.counts = [0] * 8
            return
        if self._pt_in(x, y, BTN_SEND):
            self.send_load()
            return

    @staticmethod
    def _pt_in(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    @staticmethod
    def _device_chip_rect(idx: int) -> Tuple[int, int, int, int]:
        return (DEV_CHIP_X0 + idx * DEV_CHIP_STEP, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H)

    @staticmethod
    def _stepper_btn_rects(idx: int) -> Tuple[Tuple, Tuple]:
        x = STEPPER_X0 + idx * STEPPER_STEP
        minus = (x, STEPPER_Y0 + STEPPER_VAL_H + 4, STEPPER_BTN_W, STEPPER_BTN_H)
        plus = (x + STEPPER_BTN_W + 2, STEPPER_Y0 + STEPPER_VAL_H + 4, STEPPER_BTN_W, STEPPER_BTN_H)
        return minus, plus

    # ==================== 渲染 ====================
    def render(self) -> np.ndarray:
        """渲染 1100x720 逻辑画布"""
        canvas = np.full((LOGIC_H, LOGIC_W, 3), COLOR_BG, dtype=np.uint8)
        mpos = (self.mouse_x, self.mouse_y)
        self._draw_topbar(canvas, mpos)
        self._draw_device_chips(canvas, mpos)
        self._draw_status_card(canvas)
        self._draw_steppers(canvas, mpos)
        self._draw_cmd_buttons(canvas, mpos)
        self._draw_cmd_preview(canvas)
        self._draw_proto_info(canvas)
        self._draw_log_panel(canvas)
        self._draw_toast(canvas)
        return canvas

    def _draw_btn(self, canvas, rect, label, mpos, theme_color=None, enabled=True):
        """通用按钮: hover 提亮 + 禁用置灰 (几何与命中测试单源)"""
        x, y, w, h = rect
        hov = enabled and self._pt_in(mpos[0], mpos[1], rect)
        bg = (34, 42, 54) if hov else (28, 34, 44)
        border = theme_color or COLOR_BORDER
        text_col = (200, 210, 220) if enabled else COLOR_MUTED
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)
        est_w = 8 * len(label) if label.isascii() else 14 * len(label)
        draw_text(canvas, label, (x + max(4, (w - est_w) // 2), y + (h - 14) // 2),
                  font_size=13, color=text_col, bold=hov)

    def _draw_topbar(self, canvas, mpos):
        draw_text(canvas, "Isolator WHEELS 调试", (24, 24), font_size=17, color=COLOR_TEXT, bold=True)
        draw_text(canvas, f"MQTT {BROKER_HOST}:{BROKER_PORT}  (协议 v1.0, 设备默认 {DEFAULT_DEVID})",
                  (280, 28), font_size=12, color=COLOR_SUB)
        # 连接状态点
        if self._connected:
            dot_col, dot_text = COLOR_GREEN, "已连接"
        elif self._connecting:
            dot_col, dot_text = (0, 170, 255), "连接中"
        else:
            dot_col, dot_text = (90, 100, 115), "未连接"
        cv2.circle(canvas, (838, 35), 6, dot_col, -1)
        draw_text(canvas, dot_text, (850, 28), font_size=12, color=dot_col, bold=True)
        self._draw_btn(canvas, BTN_CONNECT, "连接", mpos, theme_color=(0, 160, 200), enabled=not self._connected)
        self._draw_btn(canvas, BTN_DISCONNECT, "断开", mpos, theme_color=(140, 90, 60), enabled=self._connected)
        self._draw_btn(canvas, BTN_QUIT, "退出", mpos, theme_color=(120, 60, 60))

    def _draw_device_chips(self, canvas, mpos):
        draw_text(canvas, "在线设备:", (24, 72), font_size=12, color=COLOR_SUB)
        with self._lock:
            dev_ids = sorted(self.devices.keys())
        if not dev_ids:
            dev_ids = [self.selected_devid]
        for i, devid in enumerate(dev_ids[:DEV_CHIP_MAX]):
            rx, ry, rw, rh = self._device_chip_rect(i)
            sel = (devid == self.selected_devid)
            st = self.devices.get(devid, "")
            hov = self._pt_in(mpos[0], mpos[1], (rx, ry, rw, rh))
            bg = (24, 44, 52) if sel else ((34, 42, 54) if hov else COLOR_PANEL)
            border = COLOR_ACCENT if sel else COLOR_BORDER
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border, 2 if sel else 1)
            st_col = STATE_COLORS.get(st, COLOR_MUTED)
            cv2.circle(canvas, (rx + 14, ry + rh // 2), 4, st_col, -1)
            draw_text(canvas, devid, (rx + 26, ry + 8), font_size=12,
                      color=COLOR_TEXT if sel else COLOR_SUB, bold=sel)

    def _draw_status_card(self, canvas):
        x, y, w, h = STATUS_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        cv2.rectangle(canvas, (x, y), (x + 4, y + h), COLOR_ACCENT, -1)
        draw_text(canvas, f"设备 {self.selected_devid}", (x + 18, y + 12), font_size=14, color=COLOR_TEXT, bold=True)
        st = self._device_state(self.selected_devid)
        st_col = STATE_COLORS.get(st, COLOR_MUTED)
        st_text = STATE_TEXTS.get(st, "未上线 (等待保留消息)")
        draw_text(canvas, st_text, (x + 18, y + 42), font_size=22, color=st_col, bold=True)
        draw_text(canvas, f"节拍完成 (done) {self.done_count} 次    最近节拍: {self.last_done_time or '--:--:--'}",
                  (x + 18, y + 84), font_size=12, color=COLOR_SUB)
        draw_text(canvas, f"最近下发: {self.last_publish_msg or '(无)'}",
                  (x + 18, y + 106), font_size=11, color=COLOR_MUTED)

    def _draw_steppers(self, canvas, mpos):
        draw_text(canvas, "8 托架数量编辑 (counts[0] = 1 号托架, 仅区分 0/1/>=2)",
                  (STEPPER_X0, STEPPER_Y0 - 22), font_size=12, color=COLOR_SUB)
        for idx in range(8):
            x = STEPPER_X0 + idx * STEPPER_STEP
            # 数值框
            vy = STEPPER_Y0
            cv2.rectangle(canvas, (x, vy), (x + STEPPER_W, vy + STEPPER_VAL_H), (20, 24, 31), -1)
            cv2.rectangle(canvas, (x, vy), (x + STEPPER_W, vy + STEPPER_VAL_H), COLOR_BORDER, 1)
            val = str(self.counts[idx])
            est_w = 8 * len(val)
            draw_text(canvas, val, (x + (STEPPER_W - est_w) // 2, vy + 13), font_size=17,
                      color=COLOR_TEXT if self.counts[idx] > 0 else COLOR_MUTED, bold=True)
            # [-] / [+]
            minus, plus = self._stepper_btn_rects(idx)
            self._draw_btn(canvas, minus, "-", mpos, enabled=self.counts[idx] > 0)
            self._draw_btn(canvas, plus, "+", mpos)
            # 托架编号
            draw_text(canvas, f"{idx + 1}号", (x + 16, vy + STEPPER_VAL_H + 38),
                      font_size=11, color=COLOR_SUB)

    def _draw_cmd_buttons(self, canvas, mpos):
        st = self._device_state(self.selected_devid)
        can_send = self._connected and st == "idle"
        self._draw_btn(canvas, BTN_CLEAR, "全部清零", mpos)
        self._draw_btn(canvas, BTN_SEND, "发送 load 节拍", mpos,
                       theme_color=(0, 170, 120) if can_send else COLOR_BORDER, enabled=can_send)
        if not self._connected:
            hint = "未连接 Broker - 请先点击 [连接]"
        elif st == "running":
            hint = "设备 running 中 - 收到 done 后再发下一帧 (稳妥模式)"
        elif st != "idle":
            hint = f"设备状态 {st or '未知'} - 命令仅 idle 受理"
        else:
            hint = "设备 idle - 可下发节拍命令 (单节拍约 0.2~0.5 秒)"
        draw_text(canvas, hint, (BTN_CLEAR[0], BTN_CLEAR[1] + BTN_CLEAR[3] + 8),
                  font_size=11, color=COLOR_MUTED)

    def _draw_cmd_preview(self, canvas):
        x, y, w, h = CMD_PREVIEW_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        draw_text(canvas, "命令预览 (cmd 主题载荷)", (x + 14, y + 8), font_size=12, color=COLOR_SUB)
        payload = json.dumps({"cmd": "load", "counts": list(self.counts)}, ensure_ascii=False)
        draw_text(canvas, payload, (x + 14, y + 32), font_size=13, color=COLOR_ACCENT, bold=True)

    def _draw_proto_info(self, canvas):
        x, y, w, h = PROTO_INFO_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        cv2.rectangle(canvas, (x, y), (x + 4, y + h), (90, 140, 195), -1)
        lines = [
            "协议: flux_isolate_wheels/doc/通讯协议.md (v1.0)",
            f"订阅: {TOPIC_PREFIX}/+/state, /done, /log (保留消息, 遗嘱判活)",
            f"下发: {TOPIC_PREFIX}/{{devid}}/cmd  QoS 0",
            "规则: idle 才受理; 一帧命令触发一个节拍; done 表示全部电机到位",
        ]
        for i, ln in enumerate(lines):
            draw_text(canvas, ln, (x + 14, y + 10 + i * 30), font_size=11, color=COLOR_SUB)

    def _draw_log_panel(self, canvas):
        x, y, w, h = LOG_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        draw_text(canvas, f"设备日志 ({TOPIC_PREFIX}/+/log)", (x + 14, y + 8), font_size=12, color=COLOR_SUB)
        area_y = y + LOG_HEADER_H + 6
        max_lines = (h - LOG_HEADER_H - 16) // LOG_LINE_H
        with self._lock:
            lines = list(self.log_lines)
        for i, ln in enumerate(lines[-max_lines:]):
            ly = area_y + i * LOG_LINE_H
            draw_text(canvas, ln[:64], (x + 14, ly), font_size=11, color=(185, 196, 208))

    def _draw_toast(self, canvas):
        if self._toast and time.time() < self._toast_until:
            draw_text(canvas, self._toast[:88], (24, LOGIC_H - 22), font_size=12, color=COLOR_GREEN)

    # ==================== 主循环 ====================
    def run(self):
        """主事件循环 (启动即自动连接 Broker)"""
        self.win_mgr.setup_window(self.window_name, self._on_mouse_event)
        self.win_mgr.set_unicode_title(self.window_title)
        try:
            cv2.resizeWindow(self.window_name, self.win_mgr.canvas_w, self.win_mgr.canvas_h)
        except Exception:
            pass  # GUI 可选功能: 初始窗口尺寸设置失败不影响主循环

        self.connect_broker()

        while self._running:
            poll_res = self.win_mgr.poll_events()
            if poll_res.should_quit:
                break
            if poll_res.toast_msg:
                self._set_toast(poll_res.toast_msg)

            raw = self.render()
            if self.win_mgr.canvas_w == LOGIC_W and self.win_mgr.canvas_h == LOGIC_H:
                present = raw
            else:
                present = self._present_scaled(raw)
            cv2.imshow(self.window_name, present)
            cv2.waitKeyEx(30)

        # 退出清理
        self.disconnect_broker()
        cv2.destroyAllWindows()

    def _present_scaled(self, raw: np.ndarray) -> np.ndarray:
        """按视窗物理分辨率严格上对齐呈现 (与 workspace_hub 同约定)"""
        present = np.full((self.win_mgr.canvas_h, self.win_mgr.canvas_w, 3), COLOR_BG, dtype=np.uint8)
        scale = min(self.win_mgr.canvas_w / float(LOGIC_W), self.win_mgr.canvas_h / float(LOGIC_H))
        target_w = int(round(LOGIC_W * scale))
        target_h = int(round(LOGIC_H * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(raw, (target_w, target_h), interpolation=interp)
        pad_x = (self.win_mgr.canvas_w - target_w) // 2
        present[0:target_h, pad_x:pad_x + target_w] = scaled
        return present

    def _on_mouse_event(self, event, x, y, flags, param):
        """鼠标事件: 物理坐标转逻辑坐标 (严格上对齐), 仅响应左键单击"""
        if self.win_mgr.canvas_w != LOGIC_W or self.win_mgr.canvas_h != LOGIC_H:
            scale = min(self.win_mgr.canvas_w / float(LOGIC_W), self.win_mgr.canvas_h / float(LOGIC_H))
            pad_x = (self.win_mgr.canvas_w - int(round(LOGIC_W * scale))) // 2
            x = max(0, min(LOGIC_W - 1, int((x - pad_x) / max(1e-6, scale))))
            y = max(0, min(LOGIC_H - 1, int(y / max(1e-6, scale))))
        else:
            x = max(0, min(LOGIC_W - 1, x))
            y = max(0, min(LOGIC_H - 1, y))

        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_click(x, y)


if __name__ == "__main__":
    app = IsolateWheelsDebuggerApp()
    app.run()
