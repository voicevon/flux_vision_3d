"""
Isolator WHEELS 调试应用单元测试
================================
无网络冒烟测试：MQTT 消息路由、节拍命令稳妥模式、托架步进器交互与渲染画布
"""

import os
import sys
import json
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.isolate_wheels_debug.app import (
    IsolateWheelsDebuggerApp, LOGIC_W, LOGIC_H,
    DEV_CHIP_X0, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H, DEV_CHIP_STEP,
    BTN_CLEAR, BTN_SEND,
)


class _FakeMsg:
    """模拟 paho-mqtt 上行消息对象"""

    def __init__(self, topic: str, payload: bytes):
        self.topic = topic
        self.payload = payload


class _FakeClient:
    """捕获 publish 调用的假 MQTT 客户端"""

    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, payload))
        return True


class TestIsolateWheelsDebug(unittest.TestCase):

    def setUp(self):
        self.app = IsolateWheelsDebuggerApp(settings_file=os.path.join(
            tempfile.mkdtemp(prefix="test_wheels_"), "gui_settings.json"))

    # ---------- MQTT 消息路由 ----------
    def test_on_message_state_routing(self):
        """state 保留消息初始化设备状态"""
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/state", b'{"state":"idle"}'))
        self.assertEqual(self.app._device_state("F8EC"), "idle")

    def test_on_message_offline_will_log(self):
        """offline 遗嘱记日志且状态更新"""
        self.app.devices["F8EC"] = "running"
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/state", b'{"state":"offline"}'))
        self.assertEqual(self.app._device_state("F8EC"), "offline")
        self.assertTrue(any("遗嘱" in line for line in self.app.log_lines))

    def test_on_message_done_and_log(self):
        """done 计数与 log 行追加"""
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/done", b'{"event":"done"}'))
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/log", "[INFO] 节拍运动步数".encode("utf-8")))
        self.assertEqual(self.app.done_count, 1)
        self.assertTrue(any("[F8EC]" in line for line in self.app.log_lines))

    def test_on_message_ignores_foreign_and_malformed(self):
        """非本系统主题与非法载荷安全忽略"""
        self.app._on_message(None, None, _FakeMsg("flux/other/F8EC/state", b'{"state":"idle"}'))
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/state", b"not json"))
        self.assertEqual(self.app.devices, {})
        self.assertEqual(self.app.done_count, 0)

    # ---------- 节拍命令下发 ----------
    def test_send_load_requires_connection(self):
        """未连接 Broker 时拒绝下发 (稳妥模式)"""
        self.assertFalse(self.app.send_load())

    def test_send_load_requires_idle(self):
        """设备非 idle 时拒绝下发"""
        self.app._connected = True
        self.app.devices["F8EC"] = "running"
        self.assertFalse(self.app.send_load())

    def test_send_load_publishes_json(self):
        """连接且 idle 时下发 JSON 载荷到 cmd 主题"""
        self.app._connected = True
        self.app.devices["F8EC"] = "idle"
        self.app.counts = [1, 0, 2, 0, 0, 0, 0, 3]
        fake = _FakeClient()
        self.app._client = fake
        self.assertTrue(self.app.send_load())
        self.assertEqual(len(fake.published), 1)
        topic, payload = fake.published[0]
        self.assertEqual(topic, "flux/loader/F8EC/cmd")
        self.assertEqual(json.loads(payload), {"cmd": "load", "counts": [1, 0, 2, 0, 0, 0, 0, 3]})
        self.assertEqual(self.app.last_cmd_json, payload)

    # ---------- 托架步进器交互 ----------
    def test_stepper_click_and_clamp(self):
        """[+]/[-] 点击改变托架数量且 0~9 边界钳制"""
        minus, plus = self.app._stepper_btn_rects(2)
        self.app._handle_click(plus[0] + plus[2] // 2, plus[1] + plus[3] // 2)
        self.assertEqual(self.app.counts[2], 1)
        self.app._handle_click(minus[0] + minus[2] // 2, minus[1] + minus[3] // 2)
        self.assertEqual(self.app.counts[2], 0)
        self.app._handle_click(minus[0] + minus[2] // 2, minus[1] + minus[3] // 2)  # 已为 0 再减
        self.assertEqual(self.app.counts[2], 0)

    def test_clear_button_zeroes_all(self):
        """全部清零按钮重置 8 托架"""
        self.app.counts = [3] * 8
        self.app._handle_click(BTN_CLEAR[0] + BTN_CLEAR[2] // 2, BTN_CLEAR[1] + BTN_CLEAR[3] // 2)
        self.assertEqual(self.app.counts, [0] * 8)

    def test_send_button_blocked_when_disconnected(self):
        """点击发送按钮在未连接时无副作用"""
        self.app._handle_click(BTN_SEND[0] + BTN_SEND[2] // 2, BTN_SEND[1] + BTN_SEND[3] // 2)
        self.assertEqual(self.app.last_cmd_json, "")

    # ---------- 设备芯片选择 ----------
    def test_device_chip_selection(self):
        """点击设备芯片切换目标设备"""
        self.app.devices = {"AAA1": "idle", "BBB2": "running"}
        cx = DEV_CHIP_X0 + DEV_CHIP_STEP + DEV_CHIP_W // 2   # 第 2 枚芯片中心
        cy = DEV_CHIP_Y + DEV_CHIP_H // 2
        self.app._handle_click(cx, cy)
        self.assertEqual(self.app.selected_devid, "BBB2")

    def test_quit_button(self):
        """顶栏退出按钮终止主循环"""
        self.app._handle_click(1032 + 24, 18 + 17)  # BTN_QUIT 中心
        self.assertFalse(self.app._running)

    # ---------- 渲染冒烟 ----------
    def test_render_canvas_shape(self):
        """渲染输出 1100x720 三通道逻辑画布"""
        canvas = self.app.render()
        self.assertEqual(canvas.shape, (LOGIC_H, LOGIC_W, 3))


if __name__ == "__main__":
    unittest.main()
