"""
单元测试：标定采样场景管理与批次分组系统 (tests/test_scene_manager.py)
================================================================================
验证：
1. 场景创建、命名过滤与元数据生成
2. 工况场景默认定位与切换
3. 场景克隆与物理沙盒独立性
4. 场景发布至生产环境 (config/tags_map.yaml 与 config.yaml)
5. 场景自由删除与保护
6. 历史数据无损自动迁移 (Auto-migration)
"""

import os
import shutil
import tempfile
import unittest
import yaml

from src.calibration.scene_manager import CalibrationSceneManager


class TestCalibrationSceneManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scenes_dir = os.path.join(self.temp_dir, "scenes")
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        self.prod_map_path = os.path.join(self.temp_dir, "tags_map.yaml")

        self.legacy_dir = os.path.join(self.temp_dir, "legacy_images")
        os.makedirs(self.legacy_dir, exist_ok=True)

        # 写入初始空 config.yaml
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump({"calibration": {}}, f)

        self.mgr = CalibrationSceneManager(
            scenes_dir=self.scenes_dir,
            config_path=self.config_path,
            prod_map_path=self.prod_map_path,
            legacy_dir=self.legacy_dir
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_list_scene(self):
        """测试场景创建与列表检索"""
        scene1 = self.mgr.create_scene(alias="bench_a", description="工位A测试")
        self.assertTrue(os.path.isdir(scene1.scene_dir))
        self.assertTrue(os.path.isdir(scene1.raw_images_dir))
        self.assertTrue(os.path.exists(scene1.meta_path))
        self.assertIn("bench_a", scene1.scene_id)
        
        # 验证自动获取工况场景
        self.assertEqual(self.mgr.get_current_scene_id(), scene1.scene_id)

        # 创建第二个场景
        scene2 = self.mgr.create_scene(alias="bench_b", description="工位B测试")
        scenes = self.mgr.list_scenes()
        self.assertEqual(len(scenes), 2)
        scene_ids = [s.scene_id for s in scenes]
        self.assertIn(scene1.scene_id, scene_ids)
        self.assertIn(scene2.scene_id, scene_ids)

    def test_switch_current_scene(self):
        """测试工况场景切换"""
        scene1 = self.mgr.create_scene(alias="s1")
        scene2 = self.mgr.create_scene(alias="s2")

        self.assertEqual(self.mgr.get_current_scene_id(), scene2.scene_id)
        
        # 切换回 scene1
        ok = self.mgr.set_active_scene(scene1.scene_id)
        self.assertTrue(ok)
        self.assertEqual(self.mgr.get_current_scene_id(), scene1.scene_id)
        self.assertEqual(self.mgr.get_current_scene().scene_id, scene1.scene_id)

    def test_clone_scene_independence(self):
        """测试场景克隆与沙盒独立性"""
        scene1 = self.mgr.create_scene(alias="orig")
        # 写入一张模拟图片
        test_img_path = os.path.join(scene1.raw_images_dir, "view_0001.png")
        with open(test_img_path, "wb") as f:
            f.write(b"fake_png_data")

        # 克隆至新场景
        cloned = self.mgr.clone_scene(scene1.scene_id, new_alias="cloned")
        self.assertIsNotNone(cloned)
        self.assertTrue(os.path.exists(os.path.join(cloned.raw_images_dir, "view_0001.png")))

        # 修改原始场景的图片，不应影响克隆场景
        os.remove(test_img_path)
        self.assertFalse(os.path.exists(test_img_path))
        self.assertTrue(os.path.exists(os.path.join(cloned.raw_images_dir, "view_0001.png")))

    def test_publish_to_production(self):
        """测试将场景地图发布为生产运行地图"""
        scene = self.mgr.create_scene(alias="prod_candidate")
        
        # 尚未生成地图时发布应失败
        ok, msg = self.mgr.publish_to_production(scene.scene_id)
        self.assertFalse(ok)

        # 写入有效地图
        map_content = {
            "marker_size_mm": 50.0,
            "rmse_reprojection_px": 0.158,
            "tags": {0: {"position_mm": [0, 0, 0]}}
        }
        with open(scene.map_path, "w", encoding="utf-8") as f:
            yaml.dump(map_content, f)

        # 执行发布
        ok, msg = self.mgr.publish_to_production(scene.scene_id)
        self.assertTrue(ok, msg)
        self.assertTrue(os.path.exists(self.prod_map_path))

        # 验证生产地图内容
        with open(self.prod_map_path, "r", encoding="utf-8") as f:
            prod_m = yaml.safe_load(f)
        self.assertEqual(prod_m["rmse_reprojection_px"], 0.158)

        # 验证 config.yaml 同步记录了 active_scene
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["calibration"]["active_scene"], scene.scene_id)

    def test_delete_scene(self):
        """测试场景自由删除（已解除原禁止删除活动场景限制）"""
        scene1 = self.mgr.create_scene(alias="del_target")
        self.assertTrue(os.path.exists(scene1.scene_dir))

        ok, msg = self.mgr.delete_scene(scene1.scene_id)
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(scene1.scene_dir))

    def test_auto_migration(self):
        """测试旧版历史数据自动无损迁移功能"""
        # 准备一个独立的测试目录模拟历史环境
        mig_root = os.path.join(self.temp_dir, "mig_test")
        mig_scenes = os.path.join(mig_root, "scenes")
        mig_legacy = os.path.join(mig_root, "legacy")
        os.makedirs(mig_legacy, exist_ok=True)

        # 写入历史图像与审核清单
        with open(os.path.join(mig_legacy, "view_0001.png"), "wb") as f:
            f.write(b"png_sample")
        with open(os.path.join(mig_legacy, "tag_observations.yaml"), "w", encoding="utf-8") as f:
            yaml.dump({"frames": {"view_0001.png": {"excluded": False}}}, f)

        # 启动管理器触发自动迁移
        mgr = CalibrationSceneManager(
            scenes_dir=mig_scenes,
            config_path=os.path.join(mig_root, "config.yaml"),
            prod_map_path=os.path.join(mig_root, "tags_map.yaml"),
            legacy_dir=mig_legacy
        )

        scenes = mgr.list_scenes()
        self.assertEqual(len(scenes), 1)
        migrated_scene = scenes[0]
        self.assertIn("bench_default", migrated_scene.scene_id)
        self.assertTrue(os.path.exists(os.path.join(migrated_scene.raw_images_dir, "view_0001.png")))
        self.assertTrue(os.path.exists(migrated_scene.manifest_path))
        self.assertEqual(migrated_scene.image_count, 1)
        self.assertEqual(mgr.get_active_scene_id(), migrated_scene.scene_id)


if __name__ == "__main__":
    unittest.main()
