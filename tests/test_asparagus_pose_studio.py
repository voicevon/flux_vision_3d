#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) 单元测试
===============================================
验证数据IO配对与扫描、G-code 导出、UI 渲染器度量与主控制器行为。
"""

import os
import sys
import time
import tempfile
import unittest

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.asparagus_pose_studio.data_io import (
    find_depth_pair,
    scan_samples,
    export_gcode_file,
    load_system_config,
)
from tools.asparagus_pose_studio.renderer import AsparagusPoseStudioRenderer
from tools.asparagus_pose_studio.app import AsparagusPoseStudioApp


def _create_dummy_samples(tmpdir):
    """构造测试样本集：一组成对快照、一张独立彩色照、一张派生可视化图"""
    color1 = os.path.join(tmpdir, "color_20260920_120000.png")
    depth1 = os.path.join(tmpdir, "depth_raw_20260920_120000.npy")
    photo2 = os.path.join(tmpdir, "sample_test.png")
    vis = os.path.join(tmpdir, "depth_vis_preview.png")

    cv2.imwrite(color1, np.zeros((48, 64, 3), dtype=np.uint8))
    cv2.imwrite(photo2, np.zeros((48, 64, 3), dtype=np.uint8))
    cv2.imwrite(vis, np.zeros((48, 64, 3), dtype=np.uint8))
    np.save(depth1, np.zeros((48, 64), dtype=np.uint16))

    old = time.time() - 50
    os.utime(color1, (old, old))
    os.utime(depth1, (old, old))
    return color1, depth1, photo2, vis


class TestDataIO(unittest.TestCase):

    def test_find_depth_pair_and_scan(self):
        """测试成对深度查找与样本过滤扫描"""
        with tempfile.TemporaryDirectory() as tmpdir:
            color1, depth1, photo2, vis = _create_dummy_samples(tmpdir)

            # 配对规则测试
            self.assertEqual(find_depth_pair(color1), depth1)
            self.assertIsNone(find_depth_pair(photo2))

            # 扫描测试 (跳过 depth_vis_，最新在前)
            samples = scan_samples(tmpdir)
            self.assertEqual(len(samples), 2)
            names = [s["name"] for s in samples]
            self.assertIn("sample_test.png", names)
            self.assertIn("color_20260920_120000.png", names)
            self.assertNotIn("depth_vis_preview.png", names)
            # photo2 较新，应排第一
            self.assertEqual(samples[0]["name"], "sample_test.png")

    def test_export_gcode_file(self):
        """测试 G-code 导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ok, path = export_gcode_file("G0 X100 Y100\nM3\n", report_dir=tmpdir)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("G0 X100 Y100", content)

            # 空内容失败
            ok_empty, _ = export_gcode_file("", report_dir=tmpdir)
            self.assertFalse(ok_empty)

    def test_load_system_config_fallback(self):
        """测试加载不存在配置文件的缺省降级"""
        cfg = load_system_config("non_existent_config.yaml")
        self.assertIn("safe_z", cfg)
        self.assertEqual(cfg["safe_z"], 80.0)


class TestRenderer(unittest.TestCase):

    def test_compute_metrics_and_panels(self):
        """测试度量计算与三栏面板物理分割"""
        m = AsparagusPoseStudioRenderer.compute_metrics(1280)
        self.assertEqual(m["s"], 1.0)
        self.assertGreater(m["list_w"], 0)
        self.assertGreater(m["right_w"], 0)

        list_p, img_p, right_p = AsparagusPoseStudioRenderer.compute_panels(1280, 800, m)
        self.assertEqual(list_p[0], m["L"])
        self.assertLess(list_p[2], img_p[0])
        self.assertLess(img_p[2], right_p[0])


class TestApp(unittest.TestCase):

    def test_app_lifecycle_and_render(self):
        """测试主应用初始化、样本切换与画布渲染"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_dummy_samples(tmpdir)
            app = AsparagusPoseStudioApp(sample_dir=tmpdir)
            self.assertEqual(len(app.samples), 2)
            self.assertEqual(app.sel_idx, 0)
            self.assertEqual(app.mode, "2d")

            # 首次渲染
            canvas = app.render()
            self.assertIsInstance(canvas, np.ndarray)
            self.assertEqual(canvas.shape[0], app.win_mgr.canvas_h)
            self.assertEqual(canvas.shape[1], app.win_mgr.canvas_w)

            # 切换到成对深度快照
            app._select_sample(1)
            self.assertEqual(app.sel_idx, 1)
            self.assertEqual(app.mode, "3d")

            # 切换算法流水线
            app.switch_pipeline("ridge_tracing")
            self.assertEqual(app.pipeline_key, "ridge_tracing")

            # 快捷键退出测试
            app._handle_key(ord("x"))
            self.assertFalse(app._running)


if __name__ == "__main__":
    unittest.main()
