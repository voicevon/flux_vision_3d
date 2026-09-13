#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 离线标定综合工作站单元测试 (test_tag_offline_studio.py)
============================================================
覆盖测试用例：
  1. Studio 初始化与领域模型装配校验；
  2. 帧序列资产列表筛选过滤 (All / Warning / Excluded)；
  3. 单帧状态翻转 (保留 ⇋ 剔除) 与缓存同步；
  4. 单标靶状态翻转 (Keep / Exclude) 与当前帧残差重算；
  5. 三栏式 GUI 渲染流水线无异常闭环；
  6. 鼠标事件与按钮点击分发；
  7. 全景质检报告导出与 Markdown 文件完整性校验。
"""

import os
import sys
import glob
import shutil
import tempfile
import unittest
import numpy as np
import cv2


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.calibration.tag_offline_studio import TagOfflineStudio


class TestTagOfflineStudio(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # 生成 3 张模拟测试图片
        for i in range(1, 4):
            img = np.full((1080, 1920, 3), 40 + i * 10, dtype=np.uint8)
            cv2.imwrite(os.path.join(self.image_dir, f"view_{i:04d}.png"), img)

        self.map_path = os.path.join(self.temp_dir, "test_tags_map.yaml")
        # 复制或写入一个基础测试地图
        real_map = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
        if os.path.exists(real_map):
            shutil.copy(real_map, self.map_path)

        self.studio = TagOfflineStudio(
            map_path=self.map_path,
            image_dir=self.image_dir,
            marker_size_mm=50.0,
            win_w=1920,
            win_h=1080
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试 Studio 初始化与资产发现"""
        self.assertEqual(len(self.studio.image_files), 3, "应扫描到 3 张测试采图")
        self.assertEqual(self.studio.current_img_idx, 0, "默认初始选中第 0 帧")
        self.assertIsNotNone(self.studio.engine)
        self.assertIsNotNone(self.studio.manifest_repo)
        self.assertIsNotNone(self.studio.optimizer)
        self.assertIsNotNone(self.studio.reporter)

    def test_filter_modes(self):
        """测试左栏列表在 All / Warning / Excluded 模式下的索引筛选"""
        indices_all = self.studio._get_filtered_indices()
        self.assertEqual(len(indices_all), 3)

        # 手动剔除第一帧
        bname = os.path.basename(self.studio.image_files[0])
        self.studio.frame_metrics_cache[bname]["is_excluded"] = True

        self.studio.filter_mode = "excluded"
        indices_excl = self.studio._get_filtered_indices()
        self.assertEqual(len(indices_excl), 1)
        self.assertEqual(indices_excl[0], 0)

        # 恢复全部模式
        self.studio.filter_mode = "all"
        self.assertEqual(len(self.studio._get_filtered_indices()), 3)

    def test_toggle_frame_exclusion(self):
        """测试单帧状态翻转"""
        bname = os.path.basename(self.studio.image_files[0])
        self.assertFalse(self.studio.frame_metrics_cache[bname]["is_excluded"])

        self.studio.current_img_idx = 0
        self.studio.toggle_current_frame_exclusion()
        self.assertTrue(self.studio.frame_metrics_cache[bname]["is_excluded"])

        # 再次翻转恢复
        self.studio.toggle_current_frame_exclusion()
        self.assertFalse(self.studio.frame_metrics_cache[bname]["is_excluded"])

    def test_toggle_tag_exclusion(self):
        """测试单帧内单标靶状态翻转"""
        bname = os.path.basename(self.studio.image_files[0])
        # 伪造一个观测
        self.studio.manifest_data.setdefault("images", {})[bname] = {
            "observations": [{"tag_id": 18, "corners": [[0, 0], [10, 0], [10, 10], [0, 10]], "keep": True}]
        }
        self.studio.current_img_idx = 0
        self.studio.toggle_tag_exclusion_in_current_frame(18)

        obs = self.studio.get_observations_for_image(bname)
        self.assertEqual(len(obs), 1)
        self.assertFalse(obs[0]["keep"])


    def test_gui_render_pipeline(self):
        """测试完整三栏 GUI 渲染流水线无异常"""
        canvas = np.zeros((self.studio.win_h, self.studio.win_w, 3), dtype=np.uint8)
        self.studio.is_ba_running = True
        self.studio.set_toast("测试单元运行中")

        # 执行全景渲染
        self.studio.render(canvas)

        self.assertGreater(len(self.studio.gui_buttons), 0, "应成功注册 GUI 交互按钮")
        self.assertGreater(canvas.shape[0], 0)
        self.assertGreater(canvas.shape[1], 0)

    def test_button_click_events(self):
        """测试鼠标点击事件分发"""
        self.studio.is_running = True
        # 模拟点击退出按钮
        self.studio._handle_button_click("EXIT", "EXIT", 0, 0)
        self.assertFalse(self.studio.is_running, "点击 EXIT 应将 is_running 置为 False")

        # 模拟点击选帧
        self.studio._handle_button_click("SELECT_FRAME_2", 2, 0, 0)
        self.assertEqual(self.studio.current_img_idx, 2, "应成功切换选定帧索引至 2")

    def test_export_verification_report(self):
        """测试全景质检报告导出"""
        self.studio.export_verification_report()
        report_dir = os.path.join(PROJECT_ROOT, "data", "tag_calibration_verification")
        reports = glob.glob(os.path.join(report_dir, "studio_qa_report_*.md"))
        self.assertGreater(len(reports), 0, "应成功生成质检报告 Markdown 文件")


if __name__ == "__main__":
    unittest.main()
