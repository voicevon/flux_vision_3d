#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：多视角采图向导 (纯预览 + 保存 瘦身版)
覆盖 渲染器工具栏命中表 / 画布合成冒烟 / 工具栏状态持久化 /
     相机开关状态机 / 保存快照
"""
import os
import sys
import json
import tempfile
import unittest

import numpy as np

# 添加工程根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tools.capture.capture_wizard as wizard_mod
from tools.capture.capture_wizard import CaptureWizard, APP_ID
from tools.capture.renderer import TOOLBAR_H


def make_wizard():
    """构造向导实例 (输出目录与 settings 文件均指向临时路径, 避免污染真实数据)。
    GUI_SETTINGS_FILE 重定向后保持生效, 使测试内的 _save_viewer_state 也写入临时文件。"""
    tmp_dir = tempfile.mkdtemp(prefix="wizard_test_")
    wizard_mod.GUI_SETTINGS_FILE = os.path.join(tmp_dir, "gui_settings.json")
    return CaptureWizard(output_dir=tmp_dir)


class TestCaptureWizard(unittest.TestCase):

    def test_toolbar_buttons_and_hit_test(self):
        """工具栏命中表: 常态 4 类按钮 id 存在且可命中; 下拉展开后出现选项按钮"""
        wiz = make_wizard()
        canvas = wiz.renderer.make_canvas()
        wiz.renderer.draw_toolbar(canvas)

        ids = [btn_id for btn_id, _, _ in wiz.renderer.buttons]
        for expected in ("TOGGLE_SCENE_DD", "TOGGLE_CAM_DD", "TOGGLE_RES_DD", "TOGGLE_CAMERA", "QUIT"):
            self.assertIn(expected, ids)

        # 命中检测: 场景下拉 (x=8 起)、相机类型下拉按钮 与退出按钮 (最右 90px) 均可命中
        hit_scene = wiz.renderer.hit_test(20, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_scene)
        self.assertEqual(hit_scene[0], "TOGGLE_SCENE_DD")

        hit_cam = wiz.renderer.hit_test(220, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_cam)
        self.assertEqual(hit_cam[0], "TOGGLE_CAM_DD")
        tw = canvas.shape[1]
        hit_quit = wiz.renderer.hit_test(tw - 40, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_quit)
        self.assertEqual(hit_quit[0], "QUIT")

        # 展开相机类型下拉后: 出现 DD_CAM_ 选项按钮, 点击动作可分发
        wiz.active_dropdown = "CAMERA_TYPE_DROPDOWN"
        wiz.renderer.draw_toolbar(canvas)
        dd_ids = [btn_id for btn_id, _, _ in wiz.renderer.buttons if btn_id.startswith("DD_CAM_")]
        self.assertEqual(len(dd_ids), len(wiz.camera_options))

        # 点击选项 -> 切换相机类型且下拉收起
        wiz._handle_action(dd_ids[1], wiz.camera_options[1][0])
        self.assertEqual(wiz.camera_type, wiz.camera_options[1][0])
        self.assertIsNone(wiz.active_dropdown)

    def test_canvas_compose_smoke(self):
        """画布合成冒烟: make_canvas / compose_canvas / draw_toast 不抛异常且尺寸正确"""
        wiz = make_wizard()
        canvas = wiz.renderer.make_canvas()
        cw, ch = wiz.win_mgr.canvas_w, wiz.win_mgr.canvas_h
        self.assertEqual(canvas.shape, (ch, cw, 3))

        frame = np.full((1080, 1920, 3), 80, dtype=np.uint8)  # 1080P 帧
        composed = wiz.renderer.compose_canvas(frame)
        self.assertEqual(composed.shape, (ch, cw, 3))

        wiz.set_toast("测试 Toast")
        wiz.renderer.draw_toolbar(composed)
        wiz.renderer.draw_toast(composed)  # 不抛异常即可

    def test_viewer_state_persistence(self):
        """工具栏状态持久化: _save_viewer_state -> _load_viewer_state 往返一致"""
        wiz = make_wizard()  # make_wizard 已将 GUI_SETTINGS_FILE 重定向至临时路径
        wiz.camera_type = "usb"
        wiz.resolution = "1280x720"
        wiz.frame_w, wiz.frame_h = 1280, 720
        wiz._save_viewer_state()

        settings_file = wizard_mod.GUI_SETTINGS_FILE
        self.assertTrue(os.path.exists(settings_file))
        with open(settings_file, "r", encoding="utf-8") as f:
            root = json.load(f)
        self.assertEqual(root[APP_ID]["viewer_state"]["camera_type"], "usb")

        # 模拟重启: 修改当前值后重新加载恢复
        wiz.camera_type = "realsense"
        wiz.resolution = "1920x1080"
        wiz._load_viewer_state()
        self.assertEqual(wiz.camera_type, "usb")
        self.assertEqual(wiz.resolution, "1280x720")
        self.assertEqual((wiz.frame_w, wiz.frame_h), (1280, 720))

    def test_camera_toggle_state_machine(self):
        """相机开关状态机: 未开启 get_frame 返回 None; toggle 翻转取流状态;
        开启失败时报 Toast 不静默, 开启成功后可正常关回"""
        wiz = make_wizard()
        self.assertFalse(wiz.pipeline_running)
        self.assertIsNone(wiz.get_frame(0))  # 未开启: 不取帧

        wiz._toggle_camera()  # 有物理相机则成功, 无则失败 Toast (与硬件环境无关)
        started = wiz.pipeline_running
        if started:
            self.assertIsNotNone(wiz.get_frame(0))
        else:
            self.assertNotEqual(wiz.status_toast, "")

        wiz._toggle_camera()  # 关闭
        self.assertFalse(wiz.pipeline_running)
        self.assertIsNone(wiz.get_frame(0))

    def test_change_resolution_and_type(self):
        """切换分辨率/相机类型: 未运行时仅更新状态并持久化"""
        wiz = make_wizard()

        wiz._change_resolution("1280x720")
        self.assertEqual(wiz.resolution, "1280x720")
        self.assertEqual((wiz.frame_w, wiz.frame_h), (1280, 720))

        wiz._select_camera_type("usb")
        self.assertEqual(wiz.camera_type, "usb")

        # 持久化已写入临时 settings 文件
        with open(wizard_mod.GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
            root = json.load(f)
        self.assertEqual(root[APP_ID]["viewer_state"]["resolution"], "1280x720")

    def test_save_image_snapshot(self):
        """保存快照: 纯原图保存 (无标注图/无观测清单), 计数递增"""
        wiz = make_wizard()
        frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
        path = wiz.save_image(frame)

        self.assertEqual(wiz.image_count, 1)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.basename(path).startswith("view_0001"))
        # 瘦身后不再产出标注子目录
        self.assertFalse(os.path.exists(os.path.join(wiz.output_dir, "visualized")))


if __name__ == "__main__":
    unittest.main()
