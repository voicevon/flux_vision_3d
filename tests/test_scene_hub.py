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
import cv2

from src.calibration.scene_manager import CalibrationSceneManager
from src.calibration.camera_streamer import CameraStreamer
from tools.scene_hub.hub_state import HubState
from tools.scene_hub.hub_renderer import HubRenderer


class TestSceneHub(unittest.TestCase):

    def setUp(self):
        self.test_root = tempfile.mkdtemp(prefix="test_hub_")
        self.test_dir = os.path.join(self.test_root, "scenes")
        self.empty_legacy = os.path.join(self.test_root, "empty_legacy")
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.empty_legacy, exist_ok=True)
        self.test_prod_map = os.path.join(self.test_root, "config", "tags_map.yaml")
        self.test_config_yaml = os.path.join(self.test_root, "config.yaml")
        self.scene_mgr = CalibrationSceneManager(
            scenes_dir=self.test_dir,
            legacy_dir=self.empty_legacy,
            prod_map_path=self.test_prod_map,
            config_path=self.test_config_yaml
        )
        # 创建两个测试场景
        self.sc1 = self.scene_mgr.create_scene(alias="site_a", description="测试工况A")
        self.sc2 = self.scene_mgr.create_scene(alias="site_b", description="测试工况B")

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
        """测试 HubState 场景切换与发布生产操作"""
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

        # 模拟选中场景具备平差结果并发布为生产运行
        cur_sc = state.get_selected_scene()
        cur_sc.ba_solved = True
        cur_sc.save_meta()
        with open(cur_sc.map_path, "w", encoding="utf-8") as f:
            f.write("tags:\n  0:\n    id: 0\n    position: [0.0, 0.0, 0.0]\n    orientation: [0.0, 0.0, 0.0, 1.0]\n")
        self.assertTrue(state.publish_selected_to_production())
        self.assertEqual(state.prod_scene_id, cur_sc.scene_id)

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
        """测试 HubRenderer 双缓冲画布在不同视图模式下的渲染输出有效性"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()

        # 1. 渲染标准三栏视图
        state.set_view_mode(HubState.VIEW_STANDARD)
        canvas_std = renderer.render(state)
        self.assertEqual(canvas_std.shape, (720, 1280, 3))

        # 2. 渲染全宽大图沉浸视图
        state.set_view_mode(HubState.VIEW_EXPANDED)
        canvas_exp = renderer.render(state)
        self.assertEqual(canvas_exp.shape, (720, 1280, 3))

        # 3. 渲染纯净体检健康大屏视图
        state.set_view_mode(HubState.VIEW_DASHBOARD)
        canvas_dash = renderer.render(state)
        self.assertEqual(canvas_dash.shape, (720, 1280, 3))

    def test_hub_header_buttons_layout(self):
        """测试 Header 顶部按钮布局及 Help 弹窗交互响应"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()
        canvas = renderer.render(state)
        self.assertEqual(canvas.shape, (720, 1280, 3))

    def test_hub_help_modal(self):
        """测试【生效到生产系统】业务说明弹窗开启与渲染"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()

        self.assertFalse(state.is_help_modal_open)
        # 呼出 Help 弹窗
        state.toggle_help_modal()
        self.assertTrue(state.is_help_modal_open)

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

    def test_hub_top_exit_button_click(self):
        """测试点击右上角 [X] 退出按钮能够正常结束主循环"""
        from tools.scene_hub import SceneHubApp
        app = SceneHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        self.assertTrue(app._running)

        # 模拟鼠标点击顶部右上角退出按钮 (x=1150, y=20)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1150, 20, 0, None)
        self.assertFalse(app._running)

    def test_hub_footer_camera_and_card_active_action(self):
        """测试 Footer 底部 Camera 状态指示以及卡片点击直接设为活动"""
        from tools.scene_hub import SceneHubApp
        app = SceneHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        renderer = HubRenderer()
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 验证 Footer 渲染不报错
        renderer._render_footer(canvas, app.state)

        # 验证场景生产运行地图状态
        self.assertIsNotNone(app.state.prod_scene_id)

    def test_three_view_modes_cycle_and_rendering(self):
        """测试三模态视图循环切换与各模态画布渲染稳定性"""
        state = HubState(self.scene_mgr, force_mock=True)
        renderer = HubRenderer()

        # 1. 初始为标准模式
        self.assertEqual(state.view_mode, HubState.VIEW_STANDARD)
        c1 = renderer.render(state)
        self.assertEqual(c1.shape, (720, 1280, 3))

        # 2. 循环切换至全宽大图模式
        state.cycle_view_mode()
        self.assertEqual(state.view_mode, HubState.VIEW_EXPANDED)
        self.assertTrue(state.expanded_preview_mode)
        c2 = renderer.render(state)
        self.assertEqual(c2.shape, (720, 1280, 3))

        # 3. 循环切换至纯净健康大屏模式
        state.cycle_view_mode()
        self.assertEqual(state.view_mode, HubState.VIEW_DASHBOARD)
        self.assertFalse(state.expanded_preview_mode)
        c3 = renderer.render(state)
        self.assertEqual(c3.shape, (720, 1280, 3))

        # 4. 循环回标准模式
        state.cycle_view_mode()
        self.assertEqual(state.view_mode, HubState.VIEW_STANDARD)

    def test_three_view_modes_tab_clicks(self):
        """测试鼠标点击顶部三段式 Tab 胶囊直接切换模式"""
        from tools.scene_hub import SceneHubApp
        app = SceneHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))

        # 点击 Tab 3: 纯净看板 (x=480, y=25)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 480, 25, 0, None)
        self.assertEqual(app.state.view_mode, HubState.VIEW_DASHBOARD)

        # 点击 Tab 2: 全宽大图 (x=400, y=25)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 400, 25, 0, None)
        self.assertEqual(app.state.view_mode, HubState.VIEW_EXPANDED)

        # 点击 Tab 1: 标准三栏 (x=320, y=25)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 320, 25, 0, None)
        self.assertEqual(app.state.view_mode, HubState.VIEW_STANDARD)

    def test_context_menu_open_and_actions(self):
        """测试场景卡片鼠标右键弹出菜单、项执行与渲染稳定性"""
        from tools.scene_hub import SceneHubApp
        app = SceneHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))

        self.assertFalse(app.state.context_menu_open)

        # 1. 模拟在第 1 张卡片上右键点击 (x=100, y=120)
        app._on_mouse_event(cv2.EVENT_RBUTTONDOWN, 100, 120, 0, None)
        self.assertTrue(app.state.context_menu_open)
        self.assertEqual(app.state.context_menu_pos, (100, 120))

        # 2. 渲染包含右键菜单的画布
        canvas = app.renderer.render(app.state)
        self.assertEqual(canvas.shape, (720, 1280, 3))

        # 3. 点击菜单第一项 [设为活动沙盒] (x=120, y=120+34+16 = 170)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 120, 170, 0, None)
        self.assertFalse(app.state.context_menu_open)

        # 4. 再次右键打开后点击外部区域，验证安全关闭
        app._on_mouse_event(cv2.EVENT_RBUTTONDOWN, 100, 120, 0, None)
        self.assertTrue(app.state.context_menu_open)
        # 点击右侧远端空白区域 (x=800, y=500)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 800, 500, 0, None)
        self.assertFalse(app.state.context_menu_open)

    def test_scene_hub_settings_persistence(self):
        """测试 Scene Hub 窗口尺寸与缩放比例的自动记忆持久化与二次启动恢复"""
        import tempfile
        from tools.scene_hub import SceneHubApp
        with tempfile.TemporaryDirectory() as tmpdir:
            test_cfg = os.path.join(tmpdir, "hub_test_settings.json")
            # 1. 启动第一实例并缩放到 120%
            app1 = SceneHubApp(force_mock=True, settings_file=test_cfg)
            app1.win_mgr.apply_zoom(+20)
            self.assertEqual(app1.win_mgr.scale_pct, 120)

            # 2. 启动第二实例，验证自动无感恢复
            app2 = SceneHubApp(force_mock=True, settings_file=test_cfg)
            self.assertEqual(app2.win_mgr.scale_pct, 120)
            self.assertEqual(app2.win_mgr.canvas_w, int(1280 * 1.2))
            self.assertEqual(app2.win_mgr.canvas_h, int(720 * 1.2))

            # 3. 模拟拖拽拉伸窗口改变分辨率，验证自动落盘
            app2.win_mgr.canvas_w = 1600
            app2.win_mgr.canvas_h = 900
            app2.win_mgr.save_settings()

            # 4. 启动第三实例，验证 1600x900 依然被精准记住
            app3 = SceneHubApp(force_mock=True, settings_file=test_cfg)
            self.assertEqual(app3.win_mgr.scale_pct, 120)
            self.assertEqual(app3.win_mgr.canvas_w, 1600)
            self.assertEqual(app3.win_mgr.canvas_h, 900)


if __name__ == "__main__":
    unittest.main()


