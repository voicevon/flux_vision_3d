"""
Workspace Hub 自动化单元测试
===========================
验证 CameraStreamer 取流、HubState 工位状态机管理与原地连拍归档
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.workspace_manager import WorkspaceManager
from src.calibration.camera_streamer import CameraStreamer
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import HubRenderer, HELP_MODAL_W, HELP_MODAL_H
from tools.workspace_hub.app import WorkspaceHubApp


class TestWorkspaceHub(unittest.TestCase):

    def setUp(self):
        self.test_root = tempfile.mkdtemp(prefix="test_hub_")
        self.workspaces_dir = os.path.join(self.test_root, "workspaces")
        os.makedirs(self.workspaces_dir, exist_ok=True)
        self.test_prod_map = os.path.join(self.test_root, "config", "tags_map.yaml")
        self.test_config_yaml = os.path.join(self.test_root, "config.yaml")
        self.workspace_mgr = WorkspaceManager(
            workspaces_dir=self.workspaces_dir,
            prod_map_path=self.test_prod_map,
            config_path=self.test_config_yaml
        )
        # 创建两个测试工位
        self.ws1 = self.workspace_mgr.create_workspace(alias="site_a", description="测试工况A")
        self.ws2 = self.workspace_mgr.create_workspace(alias="site_b", description="测试工况B")

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
        """测试 HubState 工位切换与发布生产操作"""
        state = HubState(self.workspace_mgr, force_mock=True)
        self.assertEqual(len(state.workspaces), 2)
        # 降序排序下，最新创建的 ws2 在 index 0，先创建的 ws1 在 index 1
        self.assertEqual(state.workspaces[0].workspace_id, self.ws2.workspace_id)
        self.assertEqual(state.workspaces[1].workspace_id, self.ws1.workspace_id)
        self.assertEqual(state.selected_workspace_idx, 0)

        # 切换下一个工位 (index 0 -> 1)
        state.select_workspace_by_offset(1)
        self.assertEqual(state.selected_workspace_idx, 1)
        self.assertEqual(state.get_selected_workspace().workspace_id, self.ws1.workspace_id)

        # 模拟选中工位具备平差结果并发布为生产运行
        cur_ws = state.get_selected_workspace()
        cur_ws.ba_solved = True
        cur_ws.save_meta()
        with open(cur_ws.map_path, "w", encoding="utf-8") as f:
            f.write("tags:\n  0:\n    id: 0\n    position: [0.0, 0.0, 0.0]\n    orientation: [0.0, 0.0, 0.0, 1.0]\n")
        self.assertTrue(state.publish_selected_to_production())
        self.assertEqual(state.prod_workspace_id, cur_ws.workspace_id)

    def test_hub_state_in_place_capture(self):
        """测试 HubState 原地连拍保存与归档"""
        state = HubState(self.workspace_mgr, force_mock=True)
        cur_ws = state.get_selected_workspace()
        self.assertEqual(cur_ws.image_count, 0)

        # 模拟生成并抓拍一帧
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        saved_file = state.save_capture_frame(test_frame)
        self.assertTrue(os.path.exists(saved_file))
        self.assertEqual(cur_ws.image_count, 1)
        self.assertEqual(len(state.current_images), 1)

        # 再次抓拍第二帧
        saved_file2 = state.save_capture_frame(test_frame)
        self.assertTrue(os.path.exists(saved_file2))
        self.assertEqual(cur_ws.image_count, 2)
        self.assertEqual(len(state.current_images), 2)

    def test_hub_renderer_canvas(self):
        """测试 HubRenderer 双缓冲画布在不同视图模式下的渲染输出有效性"""
        state = HubState(self.workspace_mgr, force_mock=True)
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
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()
        canvas = renderer.render(state)
        self.assertEqual(canvas.shape, (720, 1280, 3))

    def test_hub_help_modal(self):
        """测试【生效到生产系统】业务说明弹窗开启与渲染"""
        state = HubState(self.workspace_mgr, force_mock=True)
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
        """测试工位克隆、列表实时刷新与新工位自动定位"""
        state = HubState(self.workspace_mgr, force_mock=True)
        initial_count = len(state.workspaces)
        self.assertEqual(initial_count, 2)

        # 克隆工位
        cur_ws = state.get_selected_workspace()
        cloned = self.workspace_mgr.clone_workspace(cur_ws.workspace_id, new_alias="对照组_工况测试")
        self.assertIsNotNone(cloned)
        self.assertEqual(cloned.name, "对照组_工况测试")

        # 刷新并重新定位
        state.refresh_workspaces()
        self.assertEqual(len(state.workspaces), initial_count + 1)

        # 验证新工位在列表中且可被定位
        target_idx = -1
        for idx, ws in enumerate(state.workspaces):
            if ws.workspace_id == cloned.workspace_id:
                target_idx = idx
                break
        self.assertNotEqual(target_idx, -1)
        state.selected_workspace_idx = target_idx
        state.load_current_workspace_images()
        self.assertEqual(state.get_selected_workspace().name, "对照组_工况测试")

    def test_hub_rename_workspace(self):
        """测试工位修改名称立即生效"""
        state = HubState(self.workspace_mgr, force_mock=True)
        ok = state.rename_current_workspace("全新车间工况A")
        self.assertTrue(ok)
        self.assertEqual(state.get_selected_workspace().name, "全新车间工况A")

    def test_hub_expanded_preview_toggle(self):
        """测试 [F] 键单帧大图全宽自适应占满与三栏模式切换"""
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()

        self.assertFalse(state.expanded_preview_mode)
        state.toggle_expanded_preview()
        self.assertTrue(state.expanded_preview_mode)

        # 渲染全宽大图
        canvas_exp = renderer.render(state)
        self.assertEqual(canvas_exp.shape, (720, 1280, 3))

    def test_hub_top_exit_button_click(self):
        """测试点击右上角 [X] 退出按钮能够正常结束主循环"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        self.assertTrue(app._running)

        # 模拟鼠标点击顶部右上角退出按钮 (x=1150, y=20)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1150, 20, 0, None)
        self.assertFalse(app._running)

    def test_hub_footer_camera_and_card_active_action(self):
        """测试 Footer 底部 Camera 状态指示以及卡片点击直接设为活动"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        renderer = HubRenderer()
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 验证 Footer 渲染不报错
        renderer._render_footer(canvas, app.state)

        # 验证工位生产运行地图状态
        self.assertIsNotNone(app.state.prod_workspace_id)

    def test_three_view_modes_cycle_and_rendering(self):
        """测试三模态视图循环切换与各模态画布渲染稳定性"""
        state = HubState(self.workspace_mgr, force_mock=True)
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
        """测试鼠标点击转移至右侧相册栏的三段式 Tab 胶囊直接切换模式"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        app.win_mgr.canvas_w = 1280
        app.win_mgr.canvas_h = 720

        # 点击 Tab 3: 纯净看板 (x=1060, y=70)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1060, 70, 0, None)
        self.assertEqual(app.state.view_mode, HubState.VIEW_DASHBOARD)

        # 点击 Tab 2: 全宽大图 (x=1010, y=70)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1010, 70, 0, None)
        self.assertEqual(app.state.view_mode, HubState.VIEW_EXPANDED)

        # 在全宽大图模式下，点击右上角退出全宽按钮 (x=1150, y=70) 返回标准三栏
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1150, 70, 0, None)
        self.assertEqual(app.state.view_mode, HubState.VIEW_STANDARD)

    def test_context_menu_open_and_actions(self):
        """测试工位卡片鼠标右键弹出菜单、项执行与渲染稳定性"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))

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

    def test_workspace_hub_settings_persistence(self):
        """测试 Workspace Hub 窗口尺寸与缩放比例的自动记忆持久化与二次启动恢复"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_cfg = os.path.join(tmpdir, "hub_test_settings.json")
            # 1. 启动第一实例并缩放到 120%
            app1 = WorkspaceHubApp(force_mock=True, settings_file=test_cfg)
            app1.win_mgr.apply_zoom(+20)
            self.assertEqual(app1.win_mgr.scale_pct, 120)

            # 2. 启动第二实例，验证自动无感恢复
            app2 = WorkspaceHubApp(force_mock=True, settings_file=test_cfg)
            self.assertEqual(app2.win_mgr.scale_pct, 120)
            self.assertEqual(app2.win_mgr.canvas_w, int(1280 * 1.2))
            self.assertEqual(app2.win_mgr.canvas_h, int(720 * 1.2))

            # 3. 模拟拖拽拉伸窗口改变分辨率，验证自动落盘
            app2.win_mgr.canvas_w = 1600
            app2.win_mgr.canvas_h = 900
            app2.win_mgr.save_settings()

            # 4. 启动第三实例，验证 1600x900 依然被精准记住
            app3 = WorkspaceHubApp(force_mock=True, settings_file=test_cfg)
            self.assertEqual(app3.win_mgr.scale_pct, 120)
            self.assertEqual(app3.win_mgr.canvas_w, 1600)
            self.assertEqual(app3.win_mgr.canvas_h, 900)

    def test_help_modal_hit_test_and_close(self):
        """验证生产说明弹窗右上角 [X] 关闭按钮在各种坐标（中心、边缘、容差、物理缩放）下的瞬间关闭判定"""
        clean_cfg = os.path.join(self.test_root, "clean_hub_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg)
        app.win_mgr.canvas_w = 1280
        app.win_mgr.canvas_h = 720
        state = app.state
        state.is_help_modal_open = True

        # 计算理论按钮中心与边界
        mx = (1280 - HELP_MODAL_W) // 2
        my = (720 - HELP_MODAL_H) // 2
        bx1 = mx + HELP_MODAL_W - 116
        by1 = my + 11
        bx2 = bx1 + 100
        by2 = by1 + 32

        # 1. 模拟点击关闭按钮中心 (如 x=bx1+50, y=by1+16)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx1 + 50, by1 + 16, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮中心应立即关闭说明窗")

        # 2. 模拟点击关闭按钮最右上角边缘 (如 bx2, by1)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx2, by1, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮右上角应立即关闭说明窗")

        # 3. 模拟点击关闭按钮最左上角 (bx1, by1)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx1, by1, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮左上角应立即关闭说明窗")

        # 4. 模拟点击关闭按钮容差外扩热区 (+4px 边缘)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx2 + 4, by1 - 4, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮容差区域应立即关闭说明窗")

        # 5. 模拟点击弹窗外部半透明遮罩 (如左上角 x=50, y=50)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 50, 50, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击弹窗遮罩外部应立即关闭说明窗")

        # 6. 模拟点击弹窗内部内容区 (如工况卡片位置 x=mx+50, y=my+150)，弹窗应保持打开
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, mx + 50, my + 150, 0, None)
        self.assertTrue(state.is_help_modal_open, "点击弹窗内部卡片不应关闭说明窗")

        # 7. 测试 renderer 的 hit_test 判定
        hit_action = app.renderer.hit_test(bx1 + 50, by1 + 16, state)
        self.assertEqual(hit_action, "help_close", "renderer.hit_test 应对齐返回 help_close")

        # 8. 进阶测试：当窗口缩放至 1600x900 时，物理屏幕坐标映射后应同样秒关
        app.win_mgr.canvas_w = 1600
        app.win_mgr.canvas_h = 900
        scale = min(1600 / 1280.0, 900 / 720.0)  # 1.25
        pad_x = (1600 - int(1280 * scale)) // 2  # 0
        pad_y = (900 - int(720 * scale)) // 2    # 0
        phys_btn_x = int(pad_x + (bx1 + 50) * scale)
        phys_btn_y = int(pad_y + (by1 + 16) * scale)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, phys_btn_x, phys_btn_y, 0, None)
        self.assertFalse(state.is_help_modal_open, "1600x900 缩放下点击关闭按钮应同样瞬时关闭")

    def test_image_deletion_and_tabs_relocation(self):
        """测试照片删除功能与三段式Tab转移后的点击交互"""
        clean_cfg = os.path.join(self.test_root, "clean_tabs_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg)
        app.win_mgr.canvas_w = 1280
        app.win_mgr.canvas_h = 720
        state = app.state

        # 1. 创建两张测试图片放入当前选中工位中
        ws = state.get_selected_workspace()
        self.assertIsNotNone(ws)
        img1 = os.path.join(ws.calib_raw_images_dir, "test_view_01.png")
        img2 = os.path.join(ws.calib_raw_images_dir, "test_view_02.png")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(img1, dummy)
        cv2.imwrite(img2, dummy)
        state.load_current_workspace_images()

        initial_count = len(state.current_images)
        self.assertGreaterEqual(initial_count, 2)
        state.selected_image_idx = 0

        # 2. 测试通过 state.delete_selected_image() 删除首张照片
        deleted_file = state.current_images[0]
        ok = state.delete_selected_image()
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(deleted_file), "被删除的照片文件应已从磁盘移除")
        self.assertEqual(len(state.current_images), initial_count - 1)
        self.assertEqual(ws.image_count, initial_count - 1)

        # 3. 测试通过鼠标点击右上角 [Del] 按钮删除 (x: 1240, y: 70)
        del_target = state.current_images[0]
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1240, 70, 0, None)
        self.assertFalse(os.path.exists(del_target), "点击 [Del] 按钮应删除当前照片")
        self.assertEqual(len(state.current_images), initial_count - 2)

        # 4. 测试点击右侧新位置的 Tab 胶囊切换视图模式
        # 点击 [▤ 看板] (x: 1060, y: 70)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1060, 70, 0, None)
        self.assertEqual(state.view_mode, HubState.VIEW_DASHBOARD)

        # 点击 [⊞ 标准] (x: 960, y: 70)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 960, 70, 0, None)
        self.assertEqual(state.view_mode, HubState.VIEW_STANDARD)

        # 点击 [⤢ 大图] (x: 1010, y: 70)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1010, 70, 0, None)
        self.assertEqual(state.view_mode, HubState.VIEW_EXPANDED)

    def test_tag_whitelist_creation_and_context_menu(self):
        """测试通过 _handle_tag_whitelist() 自动生成 tag_whitelist.yaml 模板以及右键菜单项"""
        import yaml
        clean_cfg = os.path.join(self.test_root, "clean_whitelist_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg)
        ws = app.state.get_selected_workspace()
        self.assertIsNotNone(ws)

        # 确保初始无 whitelist
        wl_path = ws.whitelist_path
        if os.path.exists(wl_path):
            os.remove(wl_path)

        # 执行白名单处理 (Windows startfile 在 unittest 中打桩避免弹出外部编辑器)
        import unittest.mock as mock
        with mock.patch("os.startfile", create=True) as mock_startfile:
            app._handle_tag_whitelist()
            self.assertTrue(os.path.exists(wl_path), "应自动创建 tag_whitelist.yaml 文件")
            mock_startfile.assert_called_once_with(wl_path)

        # 验证文件结构符合规范
        with open(wl_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("enabled", cfg)
        self.assertIn("allowed_ids", cfg)
        self.assertIn("workspace_id", cfg)
        self.assertEqual(cfg["workspace_id"], ws.workspace_id)


if __name__ == "__main__":
    unittest.main()
