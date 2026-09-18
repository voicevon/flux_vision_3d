"""
芦笋抓取位姿离线验证 GUI 单元测试 (tests/test_asparagus_offline.py)
====================================================================
覆盖: 快照配对规则 / 样本扫描 / 纯照片 2D 降级解算 / GUI 离线渲染与批量解算
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

import tools.asparagus_offline as mod
from tools.asparagus_offline import AsparagusOfflineApp, find_depth_pair, scan_samples
from src.vision.asparagus_analyzer import AsparagusAnalyzer


def _make_pair_dir(tmpdir):
    """构造样本目录: 一组成对快照 + 一张纯照片 + 一张应被跳过的可视化派生图"""
    color1 = os.path.join(tmpdir, "color_20260918_010101.png")
    depth1 = os.path.join(tmpdir, "depth_raw_20260918_010101.npy")
    photo2 = os.path.join(tmpdir, "photo_b.png")
    vis = os.path.join(tmpdir, "depth_vis_x.png")
    cv2.imwrite(color1, np.zeros((48, 64, 3), dtype=np.uint8))
    cv2.imwrite(photo2, np.zeros((48, 64, 3), dtype=np.uint8))
    cv2.imwrite(vis, np.zeros((48, 64, 3), dtype=np.uint8))
    np.save(depth1, np.zeros((48, 64), dtype=np.uint16))
    # photo_b 修改时间最新 → 排序应在最前
    old = time.time() - 100
    os.utime(color1, (old, old))
    os.utime(depth1, (old, old))
    return color1, depth1, photo2, vis


class TestPairingAndScan(unittest.TestCase):

    def test_find_depth_pair_two_rules(self):
        """深度配对双规则: 同茎同名 与 d435 抓拍 color_TS/depth_raw_TS"""
        with tempfile.TemporaryDirectory() as tmpdir:
            color1, depth1, photo2, _ = _make_pair_dir(tmpdir)
            self.assertEqual(find_depth_pair(color1), depth1)      # d435 规则
            self.assertIsNone(find_depth_pair(photo2))             # 无配对

            img3 = os.path.join(tmpdir, "img3.png")
            npy3 = os.path.join(tmpdir, "img3.npy")
            cv2.imwrite(img3, np.zeros((4, 4, 3), dtype=np.uint8))
            np.save(npy3, np.zeros((4, 4), dtype=np.uint16))
            self.assertEqual(find_depth_pair(img3), npy3)          # 同茎同名规则

    def test_scan_samples(self):
        """扫描: 跳过可视化派生图、标记成对深度、按修改时间倒序"""
        with tempfile.TemporaryDirectory() as tmpdir:
            color1, depth1, photo2, vis = _make_pair_dir(tmpdir)
            samples = scan_samples(tmpdir)
            self.assertEqual([s["name"] for s in samples],
                             ["photo_b.png", "color_20260918_010101.png"])
            self.assertIsNone(samples[0]["depth"])
            self.assertEqual(samples[1]["depth"], depth1)
            self.assertNotIn(vis, [s["png"] for s in samples])
            self.assertEqual(scan_samples(os.path.join(tmpdir, "no_such_dir")), [])


class TestAnalyze2D(unittest.TestCase):

    def test_analyze_2d_synthetic_bar(self):
        """纯照片 2D 降级: 深色底上绿色斜棒 → 检出 1 根, 角度/尺寸按标称距离估算"""
        img = np.full((480, 640, 3), 40, dtype=np.uint8)          # 深灰背景 (非植物色域)
        angle = np.deg2rad(15.0)
        p1 = (int(320 - 200 * np.cos(angle)), int(240 + 200 * np.sin(angle)))
        p2 = (int(320 + 200 * np.cos(angle)), int(240 - 200 * np.sin(angle)))
        cv2.line(img, p1, p2, (0, 200, 0), 18)                    # 400px 长 / 18px 宽绿棒
        analyzer = AsparagusAnalyzer()
        targets = analyzer.analyze(img, None)
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertFalse(t.is_topmost)                            # 无深度不做顶层判决
        self.assertEqual(t.calibration_source, "2d_preview")
        nominal_scale = 640.0 / analyzer.fx
        self.assertAlmostEqual(t.length_mm, 400 * nominal_scale, delta=25)
        self.assertAlmostEqual(t.diam_mm, 18 * nominal_scale, delta=5)
        self.assertAlmostEqual(abs(t.yaw_deg), 15.0, delta=3.0)

    def test_analyze_2d_rejects_gray_background(self):
        """纯灰底不应误检出目标"""
        img = np.full((480, 640, 3), 60, dtype=np.uint8)
        targets = AsparagusAnalyzer().analyze(img, None)
        self.assertEqual(len(targets), 0)


class TestAppOffline(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.color1, self.depth1, self.photo2, _ = _make_pair_dir(self.tmpdir)
        # 批量报表写入隔离目录, 不污染真实 reports/
        self._orig_report_dir = mod.REPORT_DIR
        mod.REPORT_DIR = self.tmpdir

    def tearDown(self):
        mod.REPORT_DIR = self._orig_report_dir
        self._tmp.cleanup()

    def test_app_load_and_render(self):
        """GUI 离线冒烟: 初始化扫描/选中渲染/样本切换均不依赖相机"""
        app = AsparagusOfflineApp(sample_dir=self.tmpdir)
        self.assertEqual(len(app.samples), 2)
        self.assertEqual(app.sel_idx, 0)               # 自动载入最新样本 (photo_b, 仅 2D)
        self.assertEqual(app.mode, "2d")
        canvas = app.render()
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.ndim, 3)

        app._select_sample(1)                          # 成对快照 → 3D 模式
        self.assertEqual(app.mode, "3d")
        canvas = app.render()
        self.assertEqual(canvas.shape[0], app.win_mgr.canvas_h)

    def test_batch_step_and_report(self):
        """批量解算逐帧推进并输出隔离目录下的 Markdown 汇总报表"""
        app = AsparagusOfflineApp(sample_dir=self.tmpdir)
        app.start_batch()
        self.assertEqual(len(app.batch_queue), 2)
        while app.batch_queue:
            app._batch_step()
        self.assertEqual(len(app.batch_results), 2)
        reports = [f for f in os.listdir(self.tmpdir) if f.startswith("asparagus_batch_report_")]
        self.assertEqual(len(reports), 1)
        with open(os.path.join(self.tmpdir, reports[0]), "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("photo_b.png", content)
        self.assertIn("color_20260918_010101.png", content)


if __name__ == "__main__":
    unittest.main()
