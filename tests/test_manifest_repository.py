#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ManifestRepository 单元测试 (test_manifest_repository.py)
=========================================================
覆盖核心特性：
  1. save_map 空间地图序列化保存与读取回验；
  2. load_manifest 剔除规则过滤与单标靶降级检查；
  3. enabled: false 整帧旁路；
  4. keep: false 劣质观测过滤。
"""

import os
import sys
import yaml
import tempfile
import shutil
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.manifest_repository import ManifestRepository


class TestManifestRepository(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_save_map(self):
        """测试标靶地图保存与持久化"""
        out_path = os.path.join(self.tmp_dir, "test_map.yaml")
        map_data = {
            "origin_tag_id": 0,
            "tags": {
                0: {"position_mm": [0.0, 0.0, 0.0]},
                1: {"position_mm": [100.0, 0.0, 0.0]}
            }
        }
        repo = ManifestRepository()
        repo.save_map(map_data, output_path=out_path)
        self.assertTrue(os.path.exists(out_path))

        with open(out_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        self.assertEqual(loaded["origin_tag_id"], 0)
        self.assertEqual(len(loaded["tags"]), 2)

    def test_load_manifest_filtering(self):
        """测试清单坏样本与整帧过滤机制"""
        manifest_path = os.path.join(self.tmp_dir, "manifest.yaml")
        raw_manifest = {
            "images": {
                "frame_ok.png": {
                    "enabled": True,
                    "image_path": "frame_ok.png",
                    "observations": [
                        {"tag_id": 0, "keep": True, "corners": [[10, 10], [50, 10], [50, 50], [10, 50]]},
                        {"tag_id": 1, "keep": True, "corners": [[100, 10], [150, 10], [150, 50], [100, 50]]},
                        {"tag_id": 2, "keep": False, "note": "模糊", "corners": [[200, 10], [250, 10], [250, 50], [200, 50]]}
                    ]
                },
                "frame_disabled.png": {
                    "enabled": False,
                    "image_path": "frame_disabled.png",
                    "observations": [
                        {"tag_id": 0, "keep": True, "corners": [[10, 10], [50, 10], [50, 50], [10, 50]]},
                        {"tag_id": 1, "keep": True, "corners": [[100, 10], [150, 10], [150, 50], [100, 50]]}
                    ]
                }
            }
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(raw_manifest, f)

        repo = ManifestRepository()
        f_det, v_frames, stats = repo.load_manifest(manifest_path)

        self.assertEqual(len(f_det), 1)
        self.assertEqual(v_frames, ["frame_ok.png"])
        self.assertEqual(stats["total_kept"], 2)
        self.assertEqual(stats["total_excluded_frames"], 1)
        # 验证 Tag 2 (keep: false) 被剔除
        self.assertNotIn(2, f_det[0])


if __name__ == "__main__":
    unittest.main()
