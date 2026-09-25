# -*- coding: utf-8 -*-
"""
Isolator WHEELS 调试视窗 (flux_isolate_wheels 分离轮 ESP32 MQTT 调试 GUI)
==========================================================================
协议依据: flux_isolate_wheels/doc/通讯协议.md (v1.2, 2026-09-22)
  - 连接: MQTT voicevon.vicp.io:1883 (von), keepalive 15s
  - 订阅: flux/loader/+/state | /done | /log (通配符多设备发现, 保留消息 + 遗嘱判活)
  - 下发: flux/loader/{devid}/cmd
          {"cmd":"load","counts":[int x8]}               (1~8 号托架)
          {"cmd":"motor","motor":1-8,"dir":0/1,"angle":(0,360]} (v1.1 单电机)
          {"cmd":"multi","angles":[num x8]}              (v1.2 多电机, 0=不动, 负值=反转)
  - 应答: done 回带 cmd 类型 (load/motor/multi); 仅 state=idle 受理
基于轻量基类 BaseCvApp 构建，1280x760 宽屏“8 通道一体化工位机架”式设计 (从左到右: 8号轮 -> 1号轮)
纯 cv2 矢量 GUI (全鼠标化), ESC/红叉退出, 支持 Ctrl+滚轮等比缩放。
"""

import os
import sys
import json
import time
import threading
from collections import deque
from typing import Optional, Tuple, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402
from src.utils.base_cv_app import BaseCvApp  # noqa: E402
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

# ==================== 逻辑画布与几何常量 (单源标准 1280 × 760) ====================
LOGIC_W, LOGIC_H = 1280, 760

# 顶栏按钮
BTN_CONNECT = (1030, 14, 70, 30)
BTN_DISCONNECT = (1108, 14, 70, 30)
BTN_QUIT = (1186, 14, 66, 30)

# 在线设备芯片选择
DEV_CHIP_X0, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H, DEV_CHIP_STEP = 540, 14, 88, 30, 96
DEV_CHIP_MAX = 4

# 8 通道机架卡片整体尺寸
RACK_PANEL = (24, 58, 1232, 342)
TAB_MULTI = (988, 68, 126, 26)
TAB_SINGLE = (1120, 68, 122, 26)

# 8 个立式通道列 (物理实物排布: 屏幕从左到右显示 1号轮 -> 8号轮)
# 对应 internal idx: idx = col (col=0 为 1号轮, col=7 为 8号轮)
COL_W = 140
COL_GAP = 12
COL_X0 = 38
COL_Y0 = 100
COL_H = 224

# 机架底部操作工具栏
BTN_CLEAR_COUNTS = (38, 344, 128, 38)
BTN_RESET_ANGLES = (174, 344, 128, 38)
BTN_SEND_LOAD = (310, 344, 210, 38)

BTN_DIR_FWD = (854, 344, 68, 38)
BTN_DIR_REV = (928, 344, 68, 38)
BTN_SEND_MOTOR = (1004, 344, 238, 38)

# 下半区双栏
STATUS_CARD = (24, 406, 594, 316)
LOG_CARD = (634, 406, 622, 316)
BTN_CLEAR_LOG = (1182, 414, 64, 24)
LOG_HEADER_H = 34
LOG_LINE_H = 24

# 角度下拉快速预设列表
MULTI_ANGLE_OPTS = (0.0, 22.5, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0,
                    -22.5, -45.0, -90.0, -180.0, -360.0)
POPUP_ROW_H = 24

# 状态配色 (BGR)
STATE_COLORS = {
    "idle": (0, 210, 130),
    "running": (0, 175, 255),
    "offline": (80, 80, 255),
}
STATE_TEXTS = {
    "idle": "IDLE 空闲 (可受理命令)",
    "running": "RUNNING 节拍执行中 (等待完成)",
    "offline": "OFFLINE 掉线 (遗嘱)",
}

COLOR_CARD_SUB = (20, 23, 30)
COLOR_BORDER_HL = (0, 190, 235)


class IsolateWheelsDebuggerApp(BaseCvApp):
    """分离轮 MQTT 调试主应用: 继承自 BaseCvApp 轻量基类"""

    def __init__(self, settings_file: Optional[str] = None):
        super().__init__(
            app_id="isolate_wheels_debug",
            base_w=LOGIC_W,
            base_h=LOGIC_H,
            window_name="flux_vision_3d | isolate_wheels",
            window_title="flux_vision_3d | Isolator WHEELS 8 通道调试工作台",
            settings_file=settings_file,
            enable_keyboard_zoom=False,
        )

        # MQTT 运行态 (网络线程加锁)
        self._lock = threading.Lock()
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._connecting = False
        self.devices: dict = {}                 # devid -> state (idle/running/offline)
        self.selected_devid: str = DEFAULT_DEVID
        self.done_count = 0
        self.last_done_time = ""
        self.last_done_cmd = ""                 # load / motor / multi
        self.last_cmd_json = ""
        self.last_publish_msg = ""
        self.log_lines: deque = deque(maxlen=300)
        self.log_scroll: Optional[int] = None   # None=跟随底部; int=可视起始行号
        self.counts: List[int] = [0] * 8        # 1~8 号托架数量 (idx: 0=1号, 7=8号)

        # 电机调试态
        self.motor_mode = "multi"               # "multi"=多电机(默认) / "single"=单电机
        self.motor_sel = 1                      # 单电机模式下选中的电机号 1~8
        self.motor_dir = 1                      # 1=正转 0=反转
        self.motor_angle = 90.0                 # 单电机模式角度
        self.multi_angles: List[float] = [0.0] * 8  # 1~8 号电机角度 (idx: 0=1号, 7=8号)
        self.popup_col = -1                     # 正在展开下拉弹窗的列 index (0~7)

    # ==================== 生命周期钩子 ====================
    def setup(self):
        """应用初始化时连接 Broker"""
        self.connect_broker()

    def cleanup(self):
        """应用退出前清理 Broker 连接"""
        self.disconnect_broker()

    # ==================== MQTT 层 ====================
    def connect_broker(self):
        """异步连接 Broker 并订阅通配主题"""
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
            self.set_toast(f"正在连接 Broker {BROKER_HOST}:{BROKER_PORT} ...")
            log.info(f"[WHEELS] 连接 Broker: {BROKER_HOST}:{BROKER_PORT}")
        except Exception as e:
            self._connecting = False
            self.set_toast(f"MQTT 连接失败: {e}")
            log.warning(f"[WHEELS] MQTT 连接失败: {e}")

    def disconnect_broker(self):
        """主动断开并停止网络线程"""
        if self._client is not None:
            try:
                self._client.disconnect()
                self._client.loop_stop()
            except Exception:
                pass
        self._client = None
        self._connected = False
        self._connecting = False
        self.set_toast("已断开 Broker 连接。")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        ok = (not reason_code.is_failure) if hasattr(reason_code, "is_failure") else (int(reason_code) == 0)
        self._connected = ok
        self._connecting = False
        if ok:
            client.subscribe([
                (f"{TOPIC_PREFIX}/+/state", 0),
                (f"{TOPIC_PREFIX}/+/done", 0),
                (f"{TOPIC_PREFIX}/+/log", 0),
            ])
            self.set_toast("Broker 已连接，订阅 flux/loader/+/* (等待保留消息...)")
            log.info("[WHEELS] Broker 已连接并完成订阅")
        else:
            self.set_toast(f"Broker 连接被拒: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected = False
        self.set_toast(f"Broker 连接断开: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """路由 flux/loader/{devid}/{state|done|log} 上行消息"""
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
                self.last_done_cmd = str(data.get("cmd", ""))
            elif kind == "log":
                self.log_lines.append(f"[{devid}] {payload}")

    def send_load(self) -> bool:
        """下发 load 节拍命令 (稳妥模式: 仅连接且设备 idle 时放行)"""
        if not self._connected:
            self.set_toast("尚未连接 Broker，无法下发命令。")
            return False
        st = self.devices.get(self.selected_devid, "")
        if st != "idle":
            self.set_toast(f"设备 {self.selected_devid} 状态 {st or '未知'} 非 idle，命令已被阻止。")
            return False
        payload = json.dumps({"cmd": "load", "counts": list(self.counts)}, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{self.selected_devid}/cmd"
        try:
            self._client.publish(topic, payload, qos=0)
        except Exception as e:
            self.set_toast(f"发送失败: {e}")
            return False
        self.last_cmd_json = payload
        self.last_publish_msg = f"{time.strftime('%H:%M:%S')} 已下发 -> load"
        self.set_toast(f"节拍已下发至 {self.selected_devid}: {payload}")
        log.info(f"[WHEELS] {self.last_publish_msg}: {payload}")
        return True

    def send_motor(self) -> bool:
        """下发 v1.1 单电机调试命令"""
        if not self._connected:
            self.set_toast("尚未连接 Broker，无法下发命令。")
            return False
        st = self.devices.get(self.selected_devid, "")
        if st != "idle":
            self.set_toast(f"设备 {self.selected_devid} 状态 {st or '未知'} 非 idle，命令已被阻止。")
            return False
        ang = round(float(self.motor_angle), 1)
        if not (0 < ang <= 360):
            self.set_toast(f"角度 {ang}° 越界 (0, 360]，未下发。")
            return False
        ang_val = int(ang) if ang == int(ang) else ang
        payload = json.dumps({"cmd": "motor", "motor": int(self.motor_sel),
                              "dir": int(self.motor_dir), "angle": ang_val}, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{self.selected_devid}/cmd"
        try:
            self._client.publish(topic, payload, qos=0)
        except Exception as e:
            self.set_toast(f"发送失败: {e}")
            return False
        self.last_cmd_json = payload
        self.last_publish_msg = f"{time.strftime('%H:%M:%S')} 已下发 -> motor #{self.motor_sel}"
        self.set_toast(f"单电机调试命令已下发至 {self.selected_devid}: {payload}")
        log.info(f"[WHEELS] {self.last_publish_msg}: {payload}")
        return True

    def send_multi(self) -> bool:
        """下发 v1.2 多电机调试命令"""
        if not self._connected:
            self.set_toast("尚未连接 Broker，无法下发命令。")
            return False
        st = self.devices.get(self.selected_devid, "")
        if st != "idle":
            self.set_toast(f"设备 {self.selected_devid} 状态 {st or '未知'} 非 idle，命令已被阻止。")
            return False
        angles = self._multi_angles_payload()
        for r in angles:
            if not (-360.0 <= r <= 360.0):
                self.set_toast(f"角度 {r}° 越界 [-360, 360]，未下发。")
                return False
        payload = json.dumps({"cmd": "multi", "angles": angles}, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{self.selected_devid}/cmd"
        try:
            self._client.publish(topic, payload, qos=0)
        except Exception as e:
            self.set_toast(f"发送失败: {e}")
            return False
        self.last_cmd_json = payload
        self.last_publish_msg = f"{time.strftime('%H:%M:%S')} 已下发 -> multi 8轴"
        self.set_toast(f"多电机命令已下发至 {self.selected_devid}: {payload}")
        log.info(f"[WHEELS] {self.last_publish_msg}: {payload}")
        return True

    def send_motor_cmd(self) -> bool:
        """根据当前模式分发下发 motor 还是 multi"""
        if self.motor_mode == "single":
            return self.send_motor()
        else:
            return self.send_multi()

    # ==================== 几何与坐标计算 ====================
    @staticmethod
    def _device_chip_rect(i: int) -> Tuple[int, int, int, int]:
        return (DEV_CHIP_X0 + i * DEV_CHIP_STEP, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H)

    @staticmethod
    def _col_rect(col: int) -> Tuple[int, int, int, int]:
        """屏幕第 col 列 (0~7, 0 对应实物 8 号轮, 7 对应实物 1 号轮)"""
        return (COL_X0 + col * (COL_W + COL_GAP), COL_Y0, COL_W, COL_H)

    @classmethod
    def _col_header_rect(cls, col: int) -> Tuple[int, int, int, int]:
        x, y, w, _ = cls._col_rect(col)
        return (x, y, w, 30)

    @classmethod
    def _col_count_minus(cls, col: int) -> Tuple[int, int, int, int]:
        x, y, w, _ = cls._col_rect(col)
        return (x + 10, y + 84, 56, 26)

    @classmethod
    def _col_count_plus(cls, col: int) -> Tuple[int, int, int, int]:
        x, y, w, _ = cls._col_rect(col)
        return (x + w - 10 - 56, y + 84, 56, 26)

    @classmethod
    def _col_angle_box(cls, col: int) -> Tuple[int, int, int, int]:
        x, y, w, _ = cls._col_rect(col)
        return (x + 10, y + 146, w - 20, 30)

    @classmethod
    def _col_angle_minus(cls, col: int) -> Tuple[int, int, int, int]:
        x, y, w, _ = cls._col_rect(col)
        return (x + 10, y + 182, 56, 26)

    @classmethod
    def _col_angle_plus(cls, col: int) -> Tuple[int, int, int, int]:
        x, y, w, _ = cls._col_rect(col)
        return (x + w - 10 - 56, y + 182, 56, 26)

    def _popup_rect(self) -> Tuple[int, int, int, int]:
        """角度弹窗位置: 自被点击的列上方弹出"""
        if self.popup_col < 0:
            return (0, 0, 0, 0)
        bx, by, bw, _ = self._col_angle_box(self.popup_col)
        h = len(MULTI_ANGLE_OPTS) * POPUP_ROW_H + 8
        top = max(10, by - h - 4)
        return (bx - 8, top, bw + 16, by - 4 - top)

    @staticmethod
    def _fmt_angle(v: float) -> str:
        r = round(float(v), 1)
        return str(int(r)) if r == int(r) else str(r)

    def _multi_angles_payload(self) -> list:
        out = []
        for a in self.multi_angles:
            r = round(float(a), 1)
            out.append(int(r) if r == int(r) else r)
        return out

    def _device_state(self, devid: str) -> str:
        with self._lock:
            return self.devices.get(devid, "")

    # ==================== 点击事件响应 (覆盖基类 on_click) ====================
    def on_click(self, x: int, y: int):
        """逻辑坐标点击事件分发"""
        # 0. 弹层拦截
        if self.popup_col >= 0:
            px, py, pw, ph = self._popup_rect()
            if self.pt_in(x, y, (px, py, pw, ph)):
                row = (y - py - 4) // POPUP_ROW_H
                if 0 <= row < len(MULTI_ANGLE_OPTS):
                    chosen = float(MULTI_ANGLE_OPTS[row])
                    idx = self.popup_col
                    if self.motor_mode == "multi":
                        self.multi_angles[idx] = chosen
                    else:
                        self.motor_angle = abs(chosen) if chosen != 0 else 90.0
                self.popup_col = -1
                return
            self.popup_col = -1
            return

        # 1. 顶栏按钮
        if self.pt_in(x, y, BTN_CONNECT):
            self.connect_broker()
            return
        if self.pt_in(x, y, BTN_DISCONNECT):
            self.disconnect_broker()
            return
        if self.pt_in(x, y, BTN_QUIT):
            self._running = False
            return

        # 2. 在线设备切换
        if DEV_CHIP_Y <= y <= DEV_CHIP_Y + DEV_CHIP_H:
            dev_ids = list(self.devices.keys()) or [self.selected_devid]
            for i in range(min(len(dev_ids), DEV_CHIP_MAX)):
                if self.pt_in(x, y, self._device_chip_rect(i)):
                    self.selected_devid = dev_ids[i]
                    return

        # 3. 模式切换 Tab
        if self.pt_in(x, y, TAB_MULTI):
            self.motor_mode = "multi"
            return
        if self.pt_in(x, y, TAB_SINGLE):
            self.motor_mode = "single"
            return

        # 4. 8 通道交互 (col: 0~7 -> idx = col)
        for col in range(8):
            idx = col
            # 4.1 点击通道 Header -> 在单电机模式下选中该轮
            if self.pt_in(x, y, self._col_header_rect(col)):
                self.motor_sel = idx + 1
                return
            # 4.2 数量增减 [-] / [+]
            if self.pt_in(x, y, self._col_count_minus(col)):
                self.counts[idx] = max(0, self.counts[idx] - 1)
                return
            if self.pt_in(x, y, self._col_count_plus(col)):
                self.counts[idx] = min(9, self.counts[idx] + 1)
                return
            # 4.3 角度点击显示框 -> 弹出预设下拉
            if self.pt_in(x, y, self._col_angle_box(col)):
                self.popup_col = col
                if self.motor_mode == "single":
                    self.motor_sel = idx + 1
                return
            # 4.4 角度微调 [-] / [+]
            if self.pt_in(x, y, self._col_angle_minus(col)):
                if self.motor_mode == "multi":
                    self.multi_angles[idx] = max(-360.0, round(self.multi_angles[idx] - 22.5, 1))
                else:
                    self.motor_sel = idx + 1
                    self.motor_angle = max(22.5, round(self.motor_angle - 22.5, 1))
                return
            if self.pt_in(x, y, self._col_angle_plus(col)):
                if self.motor_mode == "multi":
                    self.multi_angles[idx] = min(360.0, round(self.multi_angles[idx] + 22.5, 1))
                else:
                    self.motor_sel = idx + 1
                    self.motor_angle = min(360.0, round(self.motor_angle + 22.5, 1))
                return

        # 5. 机架底部动作条
        if self.pt_in(x, y, BTN_CLEAR_COUNTS):
            self.counts = [0] * 8
            self.set_toast("8 托架数量已清零。")
            return
        if self.pt_in(x, y, BTN_RESET_ANGLES):
            self.multi_angles = [0.0] * 8
            self.motor_angle = 90.0
            self.set_toast("所有电机角度已归零/重置。")
            return
        if self.pt_in(x, y, BTN_SEND_LOAD):
            self.send_load()
            return
        if self.motor_mode == "single":
            if self.pt_in(x, y, BTN_DIR_FWD):
                self.motor_dir = 1
                return
            if self.pt_in(x, y, BTN_DIR_REV):
                self.motor_dir = 0
                return
        if self.pt_in(x, y, BTN_SEND_MOTOR):
            self.send_motor_cmd()
            return

        # 6. 日志清空按钮
        if self.pt_in(x, y, BTN_CLEAR_LOG):
            with self._lock:
                self.log_lines.clear()
            self.log_scroll = None
            self.set_toast("设备实时日志已清空。")
            return

    # ==================== 日志滚轮 (覆写基类 on_mouse_wheel) ====================
    def on_mouse_wheel(self, delta: int, flags: int):
        """鼠标在日志面板区域内滚动时, 滚动日志; 默认跟随底部, 上翻进入回看, 触底恢复跟随"""
        lx, ly = self.mouse_x, self.mouse_y
        x, y, w, h = LOG_CARD
        if not (x <= lx <= x + w and y <= ly <= y + h):
            return
        with self._lock:
            total = len(self.log_lines)
        vx_, vy_, vw_, vh_ = (x + 16, y + 46, w - 32, h - 58)
        max_lines = (vh_ - 12) // LOG_LINE_H
        if total <= max_lines:
            self.log_scroll = None
            return
        rows_delta = -3 if delta > 0 else 3  # 向上滚 = delta>0 = 往前看
        base = (total - max_lines) if self.log_scroll is None else self.log_scroll
        pos = max(0, min(total - max_lines, base + rows_delta))
        if pos >= total - max_lines:
            self.log_scroll = None   # 触底 → 恢复跟随
        else:
            self.log_scroll = pos

    # ==================== 渲染系统 (实现基类 render) ====================
    def render(self) -> np.ndarray:
        """渲染 1280 × 760 逻辑画布"""
        canvas = np.full((LOGIC_H, LOGIC_W, 3), self.COLOR_BG, dtype=np.uint8)
        mpos = (self.mouse_x, self.mouse_y)

        self._draw_topbar(canvas, mpos)
        self._draw_rack_panel(canvas, mpos)
        self._draw_status_panel(canvas)
        self._draw_log_panel(canvas, mpos)

        if self.popup_col >= 0:
            self._draw_popup(canvas, mpos)

        self.draw_toast(canvas)
        return canvas

    def _draw_topbar(self, canvas, mpos):
        """顶栏: 标题、Broker状态指示、在线设备、系统操作按钮"""
        draw_text(canvas, "Isolator WHEELS 调试", (24, 18), font_size=17, color=self.COLOR_TEXT, bold=True)
        draw_text(canvas, f"MQTT {BROKER_HOST}:{BROKER_PORT} (协议 v1.2)",
                  (260, 22), font_size=12, color=self.COLOR_SUB)

        # 在线设备标签
        draw_text(canvas, "设备:", (490, 22), font_size=12, color=self.COLOR_SUB)
        with self._lock:
            dev_ids = sorted(self.devices.keys())
        if not dev_ids:
            dev_ids = [self.selected_devid]
        for i, devid in enumerate(dev_ids[:DEV_CHIP_MAX]):
            rx, ry, rw, rh = self._device_chip_rect(i)
            sel = (devid == self.selected_devid)
            st = self.devices.get(devid, "")
            hov = self.pt_in(mpos[0], mpos[1], (rx, ry, rw, rh))
            bg = (26, 48, 56) if sel else ((34, 42, 54) if hov else self.COLOR_PANEL)
            border = self.COLOR_ACCENT if sel else self.COLOR_BORDER
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border, 2 if sel else 1)
            st_col = STATE_COLORS.get(st, self.COLOR_MUTED)
            cv2.circle(canvas, (rx + 12, ry + rh // 2), 4, st_col, -1)
            draw_text(canvas, devid, (rx + 22, ry + 7), font_size=12,
                      color=self.COLOR_TEXT if sel else self.COLOR_SUB, bold=sel)

        # 连接指示
        if self._connected:
            dot_col, dot_text = self.COLOR_GREEN, "已连接"
        elif self._connecting:
            dot_col, dot_text = self.COLOR_AMBER, "连接中"
        else:
            dot_col, dot_text = (90, 100, 115), "未连接"
        cv2.circle(canvas, (948, 29), 5, dot_col, -1)
        draw_text(canvas, dot_text, (960, 22), font_size=12, color=dot_col, bold=True)

        self.draw_btn(canvas, BTN_CONNECT, "连接", mpos, theme_color=(0, 160, 200), enabled=not self._connected)
        self.draw_btn(canvas, BTN_DISCONNECT, "断开", mpos, theme_color=(140, 90, 60), enabled=self._connected)
        self.draw_btn(canvas, BTN_QUIT, "退出", mpos, theme_color=(120, 60, 60))

    def _draw_rack_panel(self, canvas, mpos):
        """【8 轮集成控制机架】: 8 列立式通道条与底部工具条"""
        x, y, w, h = RACK_PANEL
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

        # 标头与模式页签
        draw_text(canvas, "8 通道集成控制台 (物理实物排布: 左 8 号轮 -> 右 1 号轮)",
                  (x + 16, y + 12), font_size=13, color=self.COLOR_TEXT, bold=True)

        # 模式切换 Tab
        for tab_rect, mode_name, label in ((TAB_MULTI, "multi", "多电机 (multi)"),
                                           (TAB_SINGLE, "single", "单电机 (motor)")):
            sel = (self.motor_mode == mode_name)
            tx, ty, tw, th = tab_rect
            hov = self.pt_in(mpos[0], mpos[1], tab_rect)
            bg = (24, 46, 54) if sel else ((32, 38, 48) if hov else (20, 24, 30))
            border = self.COLOR_ACCENT if sel else self.COLOR_BORDER
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), bg, -1)
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), border, 2 if sel else 1)
            est = 11 * len(label)
            draw_text(canvas, label, (tx + (tw - est) // 2, ty + 6), font_size=11,
                      color=self.COLOR_TEXT if sel else self.COLOR_SUB, bold=sel)

        # 8 个立式通道
        for col in range(8):
            idx = col  # 0=1号轮, 7=8号轮
            cx, cy, cw, ch = self._col_rect(col)
            is_single_sel = (self.motor_mode == "single" and self.motor_sel == idx + 1)

            # 通道容器
            bg_col = (28, 34, 44) if is_single_sel else (20, 24, 31)
            border_col = self.COLOR_ACCENT if is_single_sel else self.COLOR_BORDER
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), bg_col, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), border_col, 2 if is_single_sel else 1)

            # 1. 通道 Header
            hx, hy, hw, hh = self._col_header_rect(col)
            h_bg = (20, 48, 56) if is_single_sel else (26, 32, 42)
            cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), h_bg, -1)
            cv2.line(canvas, (hx, hy + hh), (hx + hw, hy + hh), border_col, 1)
            title = f"{idx + 1} 号轮" + (" (选)" if is_single_sel else "")
            est_t = 13 * len(title)
            draw_text(canvas, title, (hx + max(4, (hw - est_t) // 2), hy + 7), font_size=12,
                      color=self.COLOR_ACCENT if is_single_sel else self.COLOR_TEXT, bold=True)

            # 2. 数量编辑区
            draw_text(canvas, "托架数量", (cx + 12, cy + 36), font_size=10, color=self.COLOR_MUTED)
            val = self.counts[idx]
            vx, vy, vw, vh = (cx + 10, cy + 50, cw - 20, 28)
            # 数值框智能色彩: 0 灰暗, 1 绿色正常, >=2 琥珀色
            if val == 0:
                v_bg, v_border, v_color = (16, 20, 26), self.COLOR_BORDER, self.COLOR_MUTED
            elif val == 1:
                v_bg, v_border, v_color = (20, 44, 32), (0, 180, 100), self.COLOR_GREEN
            else:
                v_bg, v_border, v_color = (44, 36, 20), (0, 150, 220), self.COLOR_AMBER
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), v_bg, -1)
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), v_border, 1)
            draw_text(canvas, str(val), (vx + vw // 2 - 5, vy + 6), font_size=14, color=v_color, bold=True)

            self.draw_btn(canvas, self._col_count_minus(col), "-", mpos, enabled=val > 0)
            self.draw_btn(canvas, self._col_count_plus(col), "+", mpos, enabled=val < 9)

            # 分隔线
            cv2.line(canvas, (cx + 8, cy + 118), (cx + cw - 8, cy + 118), (36, 42, 54), 1)

            # 3. 电机角度区
            draw_text(canvas, "电机角度", (cx + 12, cy + 126), font_size=10, color=self.COLOR_MUTED)
            cur_ang = self.multi_angles[idx] if self.motor_mode == "multi" else (self.motor_angle if is_single_sel else 0.0)
            ax, ay, aw, ah = self._col_angle_box(col)
            ang_hov = self.pt_in(mpos[0], mpos[1], (ax, ay, aw, ah))
            ang_bg = (30, 42, 50) if ang_hov else (18, 22, 28)
            cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), ang_bg, -1)
            cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), self.COLOR_ACCENT if ang_hov else self.COLOR_BORDER, 1)

            ang_str = f"{self._fmt_angle(cur_ang)}°"
            ang_col = self.COLOR_ACCENT if cur_ang != 0 else self.COLOR_MUTED
            draw_text(canvas, ang_str, (ax + 16, ay + 7), font_size=12, color=ang_col, bold=cur_ang != 0)
            draw_text(canvas, "v", (ax + aw - 16, ay + 8), font_size=10, color=self.COLOR_SUB)

            self.draw_btn(canvas, self._col_angle_minus(col), "-", mpos)
            self.draw_btn(canvas, self._col_angle_plus(col), "+", mpos)

        # 4. 机架底部操作工具条
        st = self._device_state(self.selected_devid)
        can_send = self._connected and st == "idle"

        self.draw_btn(canvas, BTN_CLEAR_COUNTS, "全部清零 (数量)", mpos)
        self.draw_btn(canvas, BTN_RESET_ANGLES, "全部归零 (角度)", mpos)
        self.draw_btn(canvas, BTN_SEND_LOAD, "发送 load 节拍", mpos,
                      theme_color=(0, 180, 120) if can_send else None, enabled=can_send, bold=True)

        if self.motor_mode == "single":
            # 单电机正反转切换
            for b_rect, label, d in ((BTN_DIR_FWD, "正转", 1), (BTN_DIR_REV, "反转", 0)):
                sel_dir = (self.motor_dir == d)
                self.draw_btn(canvas, b_rect, label, mpos,
                              theme_color=self.COLOR_GREEN if sel_dir else None, bold=sel_dir)
            motor_btn_text = f"发送 motor (#{self.motor_sel}轮)"
        else:
            motor_btn_text = "发送 multi (8轴多电机)"

        self.draw_btn(canvas, BTN_SEND_MOTOR, motor_btn_text, mpos,
                      theme_color=(0, 160, 220) if can_send else None, enabled=can_send, bold=True)

    def _draw_status_panel(self, canvas):
        """下半区左侧: 设备状态、统计与命令预览"""
        x, y, w, h = STATUS_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)
        cv2.rectangle(canvas, (x, y), (x + 4, y + h), self.COLOR_ACCENT, -1)

        draw_text(canvas, f"设备监控与通讯 [{self.selected_devid}]",
                  (x + 16, y + 12), font_size=13, color=self.COLOR_TEXT, bold=True)

        st = self._device_state(self.selected_devid)
        st_col = STATE_COLORS.get(st, self.COLOR_MUTED)
        st_text = STATE_TEXTS.get(st, "未上线 (等待保留消息)")
        cv2.circle(canvas, (x + 24, y + 48), 6, st_col, -1)
        draw_text(canvas, st_text, (x + 38, y + 40), font_size=17, color=st_col, bold=True)

        # 统计数据条
        stat_line = f"节拍完成: {self.done_count} 次  |  最近 done: {self.last_done_cmd or '-'} @ {self.last_done_time or '--:--:--'}"
        draw_text(canvas, stat_line, (x + 16, y + 74), font_size=12, color=self.COLOR_SUB)
        draw_text(canvas, f"最近下发: {self.last_publish_msg or '(尚未下发)'}",
                  (x + 16, y + 96), font_size=11, color=self.COLOR_MUTED)

        # 命令 JSON 预览框
        cv2.line(canvas, (x + 16, y + 120), (x + w - 16, y + 120), (36, 42, 54), 1)
        draw_text(canvas, "命令载荷即时预览 (Payload):", (x + 16, y + 128), font_size=11, color=self.COLOR_SUB)

        px, py, pw, ph = (x + 16, y + 150, w - 32, 114)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_CARD_SUB, -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (34, 40, 52), 1)

        load_p = json.dumps({"cmd": "load", "counts": list(self.counts)}, ensure_ascii=False)
        ang = self.motor_angle
        ang = int(ang) if ang == int(ang) else ang
        motor_p = json.dumps({"cmd": "motor", "motor": int(self.motor_sel),
                              "dir": int(self.motor_dir), "angle": ang}, ensure_ascii=False)
        multi_p = json.dumps({"cmd": "multi", "angles": self._multi_angles_payload()}, ensure_ascii=False)

        draw_text(canvas, f"load : {load_p}", (px + 10, py + 12), font_size=11, color=self.COLOR_GREEN)
        draw_text(canvas, f"motor: {motor_p}", (px + 10, py + 42), font_size=11, color=self.COLOR_ACCENT)
        draw_text(canvas, f"multi: {multi_p}", (px + 10, py + 72), font_size=11, color=self.COLOR_ACCENT)

        # 底部协议提示
        draw_text(canvas, "协议规范: 仅 idle 状态受理命令; 稳妥模式待收到 done 应答后再下发下一帧。",
                  (x + 16, y + h - 22), font_size=11, color=self.COLOR_MUTED)

    def _draw_log_panel(self, canvas, mpos):
        """下半区右侧: 实时日志流与清空功能 (支持滚轮回看)"""
        x, y, w, h = LOG_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

        # 标题 (跟随底部时显示普通标题, 回看时显示提示)
        following = self.log_scroll is None
        title = f"设备实时日志 ({TOPIC_PREFIX}/+/log)"
        if not following:
            title += "  ↑ 回看模式 (滚至底部恢复跟随)"
        title_col = self.COLOR_TEXT if following else self.COLOR_AMBER
        draw_text(canvas, title, (x + 16, y + 14), font_size=13, color=title_col, bold=True)
        self.draw_btn(canvas, BTN_CLEAR_LOG, "清空日志", mpos)

        # 日志内容视窗
        vx, vy, vw, vh = (x + 16, y + 46, w - 32, h - 58)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (14, 16, 20), -1)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (34, 40, 52), 1)

        max_lines = (vh - 12) // LOG_LINE_H
        with self._lock:
            lines = list(self.log_lines)

        total = len(lines)
        if following or total <= max_lines:
            visible = lines[-max_lines:]
        else:
            start = max(0, min(self.log_scroll, total - max_lines))
            visible = lines[start:start + max_lines]

        for i, ln in enumerate(visible):
            ly = vy + 8 + i * LOG_LINE_H
            col = (195, 205, 218)
            if "SYS" in ln:
                col = self.COLOR_AMBER
            elif "err" in ln.lower():
                col = (100, 100, 255)
            draw_text(canvas, ln[:78], (vx + 10, ly), font_size=11, color=col)

        # 迷你滚动条指示器 (总行数超出可视行时绘制)
        if total > max_lines:
            sb_x = vx + vw - 6
            sb_h = vh - 8
            thumb_h = max(16, int(sb_h * max_lines / total))
            scroll_top = 0 if following else max(0, min(self.log_scroll, total - max_lines))
            if following:
                thumb_y = vy + 4 + sb_h - thumb_h
            else:
                ratio = scroll_top / max(1, total - max_lines)
                thumb_y = vy + 4 + int(ratio * (sb_h - thumb_h))
            # 滚动条轨道
            cv2.rectangle(canvas, (sb_x, vy + 4), (sb_x + 4, vy + 4 + sb_h), (24, 28, 36), -1)
            # 滚动条滑块
            thumb_col = self.COLOR_ACCENT if not following else (60, 70, 85)
            cv2.rectangle(canvas, (sb_x, thumb_y), (sb_x + 4, thumb_y + thumb_h), thumb_col, -1)

    def _draw_popup(self, canvas, mpos):
        """自点击列向上弹出的角度选择菜单"""
        px, py, pw, ph = self._popup_rect()
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (28, 34, 44), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_BORDER_HL, 2)

        idx = self.popup_col
        cur = self.multi_angles[idx] if self.motor_mode == "multi" else self.motor_angle

        for i, opt in enumerate(MULTI_ANGLE_OPTS):
            oy = py + 4 + i * POPUP_ROW_H
            hov = (px <= mpos[0] <= px + pw and oy <= mpos[1] < oy + POPUP_ROW_H)
            if hov:
                cv2.rectangle(canvas, (px + 2, oy), (px + pw - 2, oy + POPUP_ROW_H), (42, 52, 66), -1)
            label = "0° 不动作" if opt == 0 else f"{self._fmt_angle(opt)}°"
            is_cur = (cur == opt)
            draw_text(canvas, label, (px + 10, oy + 5), font_size=11,
                      color=self.COLOR_GREEN if is_cur else (self.COLOR_TEXT if hov else self.COLOR_SUB), bold=is_cur)


if __name__ == "__main__":
    app = IsolateWheelsDebuggerApp()
    app.run()
