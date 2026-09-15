"""
Scene Hub 自动化单元测试
========================
验证 CameraStreamer 取流、HubState 场景状态机管理与原地连拍归档
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.calibration.scene_manager import CalibrationSceneManager
from src.calibration.camera_streamer import CameraStreamer
from tools.calibration.scene_hub.hub_state import HubState
from tools.calibration.scene_hub.hub_renderer import HubRenderer


class TestSceneHub(unittest.TestCase):

    def setUp(self):
        self.test_root = tempfile.mkdtemp(prefix="test_hub_")
        self.test_dir = os.path.join(self.test_root, "scenes")
        self.empty_legacy = os.path.join(self.test_root, "empty_legacy")
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.empty_legacy, exist_ok=True)
        self.scene_mgr = CalibrationSceneManager(scenes_dir=self.test_dir, legacy_dir=self.empty_legacy)
        # 创建两个测试场景
        self.sc1 = self.scene_mgr.create_scene(alias="site_a", description="测试工况A")
        self.sc2 = self.scene_mgr.create_scene(alias="site_b", description="测试工况B")
        self.scene_mgr.set_active_scene(self.sc1.scene_id)

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_camera_streamer_mock(self):
        """测试 CameraStreamer 仿真流读取与 FPS 统计"""
        streamer = CameraStreamer(force_mock=True)
        self.assertTrue(streamer.start())
        self.assertTrue(streamer.is_mock)
        ok, frame = streamer.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))
        streamer.stop()
        self.assertFalse(streamer.is_running)

    def test_hub_state_navigation(self):
        """测试 HubState 场景切换与活动场景设置"""
        state = HubState(self.scene_mgr, force_mock=True)
        self.assertEqual(len(state.scenes), 2)
        # 降序排序下，最新创建的 sc2 在 index 0，先创建的 sc1 在 index 1
        self.assertEqual(state.scenes[0].scene_id, self.sc2.scene_id)
        self.assertEqual(state.scenes[1].scene_id, self.sc1.scene_id)
        self.assertEqual(state.selected_scene_idx, 0)

        # 切换下一个场景 (index 0 -> 1)
        state.select_scene_by_offset(1)
        self.assertEqual(state.selected_scene_idx, 1)
        self.assertEqual(state.get_selected_scene().scene_id, self.sc1.scene_id)

        # 设为活动场景
        self.assertTrue(state.set_current_as_active())
        self.assertEqual(state.active_scene_id, self.sc1.scene_id)

    def test_hub_state_in_place_capture(self):
        """测试 HubState 原地连拍保存与归档"""
        state = HubState(self.scene_mgr, force_mock=True)
        cur_sc = state.get_selected_scene()
        self.assertEqual(cur_sc.image_count, 0)

        # 模拟生成并抓拍一帧
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        saved_file = state.save_capture_frame(test_frame)
        self.assertTrue(os.path.exists(saved_file))
        self.assertEqual(cur_sc.image_count, 1)
        self.assertEqual(len(state.current_images), 1)

        # 再次抓拍第二帧
        saved_file2 = state.save_capture_frame(test_frame)
        self.assertTrue(os.path.exists(saved_file2))
        self.assertEqual(cur_sc.image_count, 2)
        self.assertEqual(len(state.current_images), 2)

    def test_hub_renderer_canvas(self):
        """测试 HubRenderer 双缓冲画布渲染输出有效性"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()

        # 1. 渲染画廊模式
        state.mode = HubState.MODE_INSPECTOR
        canvas_inspector = renderer.render(state)
        self.assertEqual(canvas_inspector.shape, (720, 1280, 3))

        # 2. 渲染原地采图模式
        state.mode = HubState.MODE_CAPTURE
        state.camera_streamer.start()
        canvas_capture = renderer.render(state)
        self.assertEqual(canvas_capture.shape, (720, 1280, 3))
        state.camera_streamer.stop()

    def test_hub_toolbox_menu(self):
        """测试标定工具箱总菜单开关与浮层渲染"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()

        self.assertFalse(state.is_toolbox_open)
        # 打开工具箱
        state.toggle_toolbox()
        self.assertTrue(state.is_toolbox_open)

        # 渲染带有工具箱的画布
        canvas_with_toolbox = renderer.render(state)
        self.assertEqual(canvas_with_toolbox.shape, (720, 1280, 3))

        # 关闭工具箱
        state.toggle_toolbox()
        self.assertFalse(state.is_toolbox_open)

    def test_hub_help_modal(self):
        """测试【生效到生产系统】业务说明弹窗开启、互斥与渲染"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()

        self.assertFalse(state.is_help_modal_open)
        # 呼出 Help 弹窗
        state.toggle_help_modal()
        self.assertTrue(state.is_help_modal_open)
        self.assertFalse(state.is_toolbox_open)

        # 渲染带有 Help 弹窗的画布
        canvas_help = renderer.render(state)
        self.assertEqual(canvas_help.shape, (720, 1280, 3))

        # 关闭 Help 弹窗
        state.toggle_help_modal()
        self.assertFalse(state.is_help_modal_open)

    def test_hub_clone_and_immediate_refresh(self):
        """测试场景克隆、列表实时刷新与新场景自动定位"""
        state = HubState(self.scene_mgr, force_mock=True)
        initial_count = len(state.scenes)
        self.assertEqual(initial_count, 2)

        # 克隆场景
        cur_sc = state.get_selected_scene()
        cloned = self.scene_mgr.clone_scene(cur_sc.scene_id, new_alias="对照组_工况测试")
        self.assertIsNotNone(cloned)
        self.assertEqual(cloned.name, "对照组_工况测试")

        # 刷新并重新定位
        state.refresh_scenes()
        self.assertEqual(len(state.scenes), initial_count + 1)

        # 验证新场景在列表中且可被定位
        target_idx = -1
        for idx, sc in enumerate(state.scenes):
            if sc.scene_id == cloned.scene_id:
                target_idx = idx
                break
        self.assertNotEqual(target_idx, -1)
        state.selected_scene_idx = target_idx
        state.load_current_scene_images()
        self.assertEqual(state.get_selected_scene().name, "对照组_工况测试")

    def test_hub_rename_scene(self):
        """测试场景修改名称立即生效"""
        state = HubState(self.scene_mgr, force_mock=True)
        ok = state.rename_current_scene("全新车间工况A")
        self.assertTrue(ok)
        self.assertEqual(state.get_selected_scene().name, "全新车间工况A")

    def test_hub_expanded_preview_toggle(self):
        """测试 [F] 键单帧大图全宽自适应占满与三栏模式切换"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()

        self.assertFalse(state.expanded_preview_mode)
        state.toggle_expanded_preview()
        self.assertTrue(state.expanded_preview_mode)

        # 渲染全宽大图
        canvas_exp = renderer.render(state)
        self.assertEqual(canvas_exp.shape, (720, 1280, 3))

        state.toggle_expanded_preview()
        self.assertFalse(state.expanded_preview_mode)


if __name__ == "__main__":
    unittest.main()


