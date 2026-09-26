import os
import sys
import unittest
import numpy as np
import yaml
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.ba_optimizer import BundleAdjustmentOptimizer
from src.calibration.manifest_repository import ManifestRepository
from tools.spatial_mapping_studio.mapping_ba_runner import MappingBARunner
from tools.spatial_mapping_studio.mapping_state import MappingDataManager


class TestDecoupledBARunner(unittest.TestCase):
    """验证阶段一纯自由平差与阶段二独立世界系校准在 Runner 调度层的端到端作业流"""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.ws_dir = self.td.name
        self.map_path = os.path.join(self.ws_dir, "tags_map.yaml")
        self.raw_map_path = os.path.join(self.ws_dir, "tags_map_raw.yaml")
        self.manifest_path = os.path.join(self.ws_dir, "tag_observations.yaml")

        K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        self.optimizer = BundleAdjustmentOptimizer(camera_matrix=K, dist_coeffs=D, marker_size_mm=50.0)

        # 构造简单 manifest
        self._create_mock_manifest()

    def tearDown(self):
        self.td.cleanup()

    def _create_mock_manifest(self):
        import cv2
        T_w_t5 = np.eye(4)
        T_w_t6 = np.eye(4)
        T_w_t6[0, 3] = 348.0
        T_w_t7 = np.eye(4)
        T_w_t7[1, 3] = 470.0

        r1, t1 = np.zeros(3), np.array([0.0, 0.0, 1000.0])
        r2, t2 = np.array([0.0, 0.05, 0.0]), np.array([50.0, 20.0, 980.0])

        def proj(T_w_t, rv, tv):
            R, _ = cv2.Rodrigues(rv)
            T_c_w = np.eye(4)
            T_c_w[:3, :3] = R
            T_c_w[:3, 3] = tv
            T_c_t = T_c_w @ T_w_t
            rt, _ = cv2.Rodrigues(T_c_t[:3, :3])
            pts2d, _ = cv2.projectPoints(self.optimizer.obj_points, rt, T_c_t[:3, 3], self.optimizer.camera_matrix, self.optimizer.dist_coeffs)
            return pts2d.reshape(4, 2).tolist()

        manifest_data = {
            "images": {
                "v1.png": {
                    "file_name": "v1.png", "enabled": True, "excluded": False,
                    "observations": [
                        {"tag_id": 5, "keep": True, "corners": proj(T_w_t5, r1, t1)},
                        {"tag_id": 6, "keep": True, "corners": proj(T_w_t6, r1, t1)},
                        {"tag_id": 7, "keep": True, "corners": proj(T_w_t7, r1, t1)}
                    ]
                },
                "v2.png": {
                    "file_name": "v2.png", "enabled": True, "excluded": False,
                    "observations": [
                        {"tag_id": 5, "keep": True, "corners": proj(T_w_t5, r2, t2)},
                        {"tag_id": 6, "keep": True, "corners": proj(T_w_t6, r2, t2)},
                        {"tag_id": 7, "keep": True, "corners": proj(T_w_t7, r2, t2)}
                    ]
                }
            }
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest_data, f)

    def test_decoupled_two_stage_workflow(self):
        """测试阶段一自由平差 + 阶段二独立世界系校准端到端流转"""
        data_mgr = MappingDataManager(
            map_path=self.map_path,
            image_dir=self.ws_dir,
            manifest_path=self.manifest_path,
            engine=None,
            marker_size_mm=50.0
        )
        runner = MappingBARunner(
            data_mgr=data_mgr,
            optimizer=self.optimizer,
            manifest_repo=ManifestRepository(),
            map_path=self.map_path,
            manifest_path=self.manifest_path,
            marker_size_mm=50.0
        )

        # -------------------------------------------------------------
        # 1. 执行阶段一: 纯自由平差 (此时无锚点配置)
        # -------------------------------------------------------------
        succ, opt_res, msg = runner._execute_ba_solve()
        self.assertTrue(succ, f"自由平差应成功: {msg}")
        self.assertTrue(os.path.exists(self.raw_map_path), "应生成 tags_map_raw.yaml")
        raw_map = ManifestRepository.load_map(self.raw_map_path)
        self.assertEqual(raw_map["anchor_mode"], "unaligned")
        self.assertIn("raw_relative_poses", raw_map)
        self.assertLess(raw_map["final_rmse"], 0.2)

        # -------------------------------------------------------------
        # 2. 用户在白名单录入锚点真值
        # -------------------------------------------------------------
        anchors_cfg = {
            "anchor_tags": {
                5: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
                6: {"xyz_mm": [348.0, 0.0, 0.0], "known": [True, True, True]},
                7: {"xyz_mm": [0.0, 470.0, 0.0], "known": [True, True, True]}
            }
        }
        with open(os.path.join(self.ws_dir, "anchor_tags.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(anchors_cfg, f)

        # 模拟工作站工位引用
        class MockWS:
            workspace_dir = self.ws_dir
        runner.workspace = MockWS()

        # -------------------------------------------------------------
        # 3. 执行阶段二: 独立校准世界坐标系 (毫秒级)
        # -------------------------------------------------------------
        align_succ, align_msg, world_map = runner.execute_world_alignment()
        self.assertTrue(align_succ, f"世界系校准应成功: {align_msg}")
        self.assertTrue(os.path.exists(self.map_path), "应生成生产 tags_map.yaml")
        self.assertEqual(world_map["anchor_mode"], "full")
        self.assertEqual(world_map["world_anchor"]["solver_type"], "umeyama_3d")

        # 验证标靶世界绝对坐标落位
        p5 = world_map["tags"][5]["position_mm"]
        p6 = world_map["tags"][6]["position_mm"]
        p7 = world_map["tags"][7]["position_mm"]
        np.testing.assert_allclose(p5, [0.0, 0.0, 0.0], atol=0.05)
        np.testing.assert_allclose(p6, [348.0, 0.0, 0.0], atol=0.05)
        np.testing.assert_allclose(p7, [0.0, 470.0, 0.0], atol=0.05)


if __name__ == "__main__":
    unittest.main()
