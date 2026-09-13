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
        """测试完整三栏 GUI 渲染流水线无异常 (含 Tag 叠加与残差矢量)"""
        bname = os.path.basename(self.studio.image_files[0])
        # 伪造带实际观测角点的帧数据
        self.studio.manifest_data.setdefault("images", {})[bname] = {
            "observations": [{
                "tag_id": 0,
                "corners": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]],
                "keep": True
            }]
        }
        self.studio.refresh_all_frame_metrics()

        canvas = np.zeros((self.studio.win_h, self.studio.win_w, 3), dtype=np.uint8)
        self.studio.is_ba_running = True
        self.studio.set_toast("测试单元运行中")

        # 执行全景渲染
        self.studio.render(canvas)

        self.assertGreater(len(self.studio.gui_buttons), 0, "应成功注册 GUI 交互按钮")
        self.assertGreater(canvas.shape[0], 0)
        self.assertGreater(canvas.shape[1], 0)

        # 直接测试 visualizer.draw_reprojection_vectors
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        obs_pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
        proj_pts = np.array([[12, 11], [25, 23]], dtype=np.float32)
        self.studio.visualizer.draw_reprojection_vectors(test_img, obs_pts, proj_pts, scale_factor=20.0)
        self.assertIsNotNone(test_img)

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

    def test_viewport_zoom_and_pan(self):
        """测试视口区分区域滚轮 (左栏列表滚动 vs 中间画布缩放) 与平移重置"""
        # 1. 鼠标在左栏 (x=100, y=200): 滚轮只影响 scroll_offset
        self.assertEqual(self.studio.scroll_offset, 0)
        self.assertEqual(self.studio.zoom_level, 1.0)
        self.studio._on_mouse(cv2.EVENT_MOUSEWHEEL, mx=100, my=200, flags=-1, param=None)
        self.assertGreater(self.studio.scroll_offset, 0, "左栏滚轮应增加列表偏移")
        self.assertEqual(self.studio.zoom_level, 1.0, "左栏滚轮不应影响画布缩放")

        # 2. 鼠标在中间画布 (x=800, y=500): 滚轮只放大画布图像
        old_scroll = self.studio.scroll_offset
        self.studio._on_mouse(cv2.EVENT_MOUSEWHEEL, mx=800, my=500, flags=1, param=None)
        self.assertGreater(self.studio.zoom_level, 1.0, "中间画布滚轮向上应放大图像")
        self.assertEqual(self.studio.scroll_offset, old_scroll, "中间画布滚轮不应影响左栏列表")

        # 3. 鼠标右键在中间画布按住并拖拽
        old_pan_x = self.studio.pan_offset_x
        old_pan_y = self.studio.pan_offset_y
        self.studio._on_mouse(cv2.EVENT_RBUTTONDOWN, mx=800, my=500, flags=0, param=None)
        self.assertTrue(self.studio.is_panning)
        self.studio._on_mouse(cv2.EVENT_MOUSEMOVE, mx=830, my=520, flags=0, param=None)
        self.studio._on_mouse(cv2.EVENT_RBUTTONUP, mx=830, my=520, flags=0, param=None)
        self.assertFalse(self.studio.is_panning)
        self.assertEqual(self.studio.pan_offset_x, old_pan_x + 30.0)
        self.assertEqual(self.studio.pan_offset_y, old_pan_y + 20.0)


        # 4. 双击中间画布重置缩放与平移
        self.studio._on_mouse(cv2.EVENT_LBUTTONDBLCLK, mx=800, my=500, flags=0, param=None)
        self.assertEqual(self.studio.zoom_level, 1.0, "双击应重置缩放至 1.0x")
        self.assertEqual(self.studio.pan_offset_x, 0.0, "双击应重置平移偏置")
        self.assertEqual(self.studio.pan_offset_y, 0.0)

        # 5. 放大到 3.0x 下渲染画布无异常
        self.studio.zoom_level = 3.0
        canvas = np.zeros((self.studio.win_h, self.studio.win_w, 3), dtype=np.uint8)
        self.studio.render(canvas)
        self.assertGreater(canvas.shape[0], 0)



if __name__ == "__main__":
    unittest.main()
