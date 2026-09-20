"""
工作空间综合管理中枢 (Workspace Hub)
==============================================
提供现代深色科技风格 GUI 界面：
- Workspace 画廊管理 (选择、切换、新建、重命名、克隆、删除)
- 历史采样照片卡片网格墙与全宽大图自适应视口 (双击卡片放大)
- 几何健康度与两阶段 BA 平差残差看板
- 数据工作空间与生命周期管理
- 一键直通标定离线 Studio 深度平差与原子发布至生产环境
- 完美支持中英文 Workspace 别名输入与显示
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.workspace_manager import WorkspaceManager
from src.utils.gui_window_manager import GuiWindowManager
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import HubRenderer, HELP_MODAL_W, HELP_MODAL_H, grid_hit_test
from src.utils.logger import get_logger

log = get_logger(__name__)


from src.utils.dialog_utils import prompt_confirm, prompt_input_text


class WorkspaceHubApp:
    """Workspace Hub 主应用"""

    def __init__(self, force_mock: bool = False, settings_file: str = None):
        self.force_mock = force_mock
        self.win_mgr = GuiWindowManager(
            app_id="workspace_hub",
            base_w=960,
            base_h=720,
            settings_file=settings_file,
            enable_keyboard_zoom=False  # 全鼠标化: 不启用 Ctrl/+/- 键盘缩放热键
        )
        self.workspace_mgr = WorkspaceManager()
        self.state = HubState(self.workspace_mgr, force_mock=force_mock)
        self.renderer = HubRenderer()
        # 窗口内部 key 必须纯 ASCII (namedWindow ANSI API), 中文标题走 set_unicode_title
        self.window_name = "flux_vision_3d | workspace"
        self.window_title = "flux_vision_3d | Workspace"
        self._running = True

        if self.win_mgr.scale_pct != 100 or self.win_mgr.canvas_w != 960 or self.win_mgr.canvas_h != 720:
            self.state.set_toast(f"已恢复偏好设置：放大镜 {self.win_mgr.scale_pct}%，视窗 {self.win_mgr.canvas_w}×{self.win_mgr.canvas_h}")

    def run(self):
        """主事件循环"""
        # 使用 GuiWindowManager 挂载原生窗口、记忆尺寸与 Unicode 标题
        self.win_mgr.setup_window(self.window_name, self._on_mouse_event)
        self.win_mgr.set_unicode_title(self.window_title)

        try:
            cv2.resizeWindow(self.window_name, self.win_mgr.canvas_w, self.win_mgr.canvas_h)
        except Exception:
            pass  # GUI 可选功能：初始窗口尺寸设置失败不影响主循环

        while self._running:
            # 1. 视窗管理器综合轮询 (红叉检测、硬件按键缩放、拖拽防抖持久化)
            poll_res = self.win_mgr.poll_events()
            if poll_res.should_quit:
                break
            if poll_res.toast_msg:
                self.state.set_toast(poll_res.toast_msg)

            # 2. 渲染画面并在当前窗口分辨率下严格上对齐呈现 (无多余顶部黑边)
            raw_canvas = self.renderer.render(self.state)
            if self.win_mgr.canvas_w == self.renderer.canvas_w and self.win_mgr.canvas_h == self.renderer.canvas_h:
                present_canvas = raw_canvas
            else:
                present_canvas = np.full((self.win_mgr.canvas_h, self.win_mgr.canvas_w, 3), (18, 20, 24), dtype=np.uint8)
                scale = min(self.win_mgr.canvas_w / float(self.renderer.canvas_w),
                            self.win_mgr.canvas_h / float(self.renderer.canvas_h))
                target_w = int(round(self.renderer.canvas_w * scale))
                target_h = int(round(self.renderer.canvas_h * scale))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                scaled = cv2.resize(raw_canvas, (target_w, target_h), interpolation=interp)
                pad_x = (self.win_mgr.canvas_w - target_w) // 2
                pad_y = 0  # 严格上对齐！
                present_canvas[0:target_h, pad_x:pad_x + target_w] = scaled

            cv2.imshow(self.window_name, present_canvas)

            # 3. waitKeyEx 仅用于驱动窗口消息泵刷新画面 (项目已全面鼠标化, 不响应任何键盘快捷键)
            cv2.waitKeyEx(15)

        # 退出清理
        cv2.destroyAllWindows()

    def _launch_capture_wizard(self):
        """启动多视角交互式采图向导 (tools/capture/capture_wizard.py)"""
        ws = self.state.get_selected_workspace()
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "tools", "capture", "capture_wizard.py")]
        if ws:
            cmd.extend(["--workspace", ws.workspace_id])
        self._run_subtool(cmd, "多视角交互采图向导")

    def _on_mouse_event(self, event, x, y, flags, param):
        """处理鼠标点击、悬浮 Hover 与滚轮切片交互 (支持 Ctrl+滚轮缩放与逻辑坐标映射)"""
        # 0. 优先拦截 Ctrl + 滚轮缩放 (委托通用视窗管理器)
        if event == 10:  # cv2.EVENT_MOUSEWHEEL
            handled, toast = self.win_mgr.handle_mouse_wheel(event, flags)
            if handled and toast:
                self.state.set_toast(toast)
                return

        # 0.1 物理坐标转换回 960x720 逻辑坐标 (严格上对齐)
        if self.win_mgr.canvas_w != self.renderer.canvas_w or self.win_mgr.canvas_h != self.renderer.canvas_h:
            scale = min(self.win_mgr.canvas_w / float(self.renderer.canvas_w),
                        self.win_mgr.canvas_h / float(self.renderer.canvas_h))
            pad_x = (self.win_mgr.canvas_w - int(round(self.renderer.canvas_w * scale))) // 2
            pad_y = 0  # 严格上对齐！
            logic_x = int((x - pad_x) / max(1e-6, scale))
            logic_y = int(y / max(1e-6, scale))
            x = max(0, min(self.renderer.canvas_w - 1, logic_x))
            y = max(0, min(self.renderer.canvas_h - 1, logic_y))

        # 1. 实时跟踪鼠标坐标，支持全部按钮平滑 Hover 高亮
        if event == cv2.EVENT_MOUSEMOVE:
            self.state.mouse_x = x
            self.state.mouse_y = y
            return

        # 2. 普通滚轮极速翻页/切换 Workspace (未按 Ctrl 时)
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = -1 if flags > 0 else 1
            if x <= 340 and 58 <= y <= 520:
                self.state.select_workspace_by_offset(delta)
            elif self.state.view_mode == HubState.VIEW_EXPANDED:
                # 全宽大图沉浸模式下滚轮切换大图
                self.state.select_image_by_offset(delta)
            else:
                # 卡片网格墙模式下滚轮翻页
                self._scroll_grid_for_active_tab(delta)
            return

        # 后续仅处理鼠标左键点击 (单击选中 / 双击放大)
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            return

        # =================== 3. 生产机制 Help 说明窗下的点击 ===================
        if self.state.is_help_modal_open:
            mx = (self.renderer.canvas_w - HELP_MODAL_W) // 2
            my = (self.renderer.canvas_h - HELP_MODAL_H) // 2
            bx1 = mx + HELP_MODAL_W - 116
            by1 = my + 11
            bx2 = bx1 + 100
            by2 = by1 + 32

            # 点击右上角 [X] 关闭按钮 (带 6px 宽容防抖热区)
            if (bx1 - 6) <= x <= (bx2 + 6) and (by1 - 6) <= y <= (by2 + 6):
                self.state.is_help_modal_open = False
                self.state.set_toast("已关闭说明窗。")
                return

            # 点击弹窗外部半透明遮罩：关闭
            if x < mx or x > mx + HELP_MODAL_W or y < my or y > my + HELP_MODAL_H:
                self.state.is_help_modal_open = False
                self.state.set_toast("已关闭说明窗。")
            return

        # =================== 4. 正常看板与采图模式下的鼠标点击 ===================
        # 4.0 顶部标题栏交互 (右侧动态区四页签 Tab + 紧邻生产相册的 [退出] 按钮)
        # 4.0.0 四页签 Tab 胶囊 (x: 348~816, y: 8~42, 每片 110px 宽、间距 8px)
        if 8 <= y <= 42 and 348 <= x <= 816:
            tab_idx = (x - 348) // 118
            if 0 <= tab_idx < 4:
                self.state.set_tab(HubState.TAB_ORDER[tab_idx])
            return

        # 4.0.2 [退出] 按钮紧贴生产相册右侧 (x: 824~948, y: 8~42)
        if 824 <= x <= 948 and 8 <= y <= 42:
            self._running = False
            return

        # 5.1 点击左侧 Workspace 列表卡片 (x: 10~330, y: 58~520, 支持 6 张卡片)
        if 10 <= x <= 330 and 58 <= y <= 520:
            card_h = 70
            gap = 8
            idx_in_view = (y - 58) // (card_h + gap)
            max_cards = 6
            scroll_start = max(0, self.state.selected_workspace_idx - max_cards + 1)
            target_idx = scroll_start + idx_in_view
            if 0 <= target_idx < len(self.state.workspaces):
                self.state.select_workspace_at_index(target_idx)
            return

        # 5.2 点击左侧通用全局 Workspace 管理按钮 (y: 614~654)
        if 10 <= x <= 330 and 614 <= y <= 654:
            self._handle_create_workspace()
            return

        # 5.4 全宽大图预览模式下的右上角按钮交互 (x: 340~960)
        if self.state.expanded_preview_mode:
            # 顶部按钮行 (y: 58~92)
            if 58 <= y <= 92:
                if 680 <= x <= 740:
                    self.state.select_image_by_offset(-1)
                    return
                if 746 <= x <= 806:
                    self.state.select_image_by_offset(1)
                    return
                if 812 <= x <= 880:
                    self.state.delete_selected_image()
                    return
                if 886 <= x <= 950:
                    self.state.toggle_expanded_preview()
                    return

            # 双击大图画面返回卡片网格墙
            if event == cv2.EVENT_LBUTTONDBLCLK and 340 <= x <= 960 and 96 <= y <= 670:
                self.state.toggle_expanded_preview()
                return

        # 5.5 右侧动态区四页签内容交互 (x: 340~960, y: 50~670)
        if not self.state.expanded_preview_mode:
            tab = self.state.active_tab

            # 5.5.1 Tag 白名单页签: [刷新] [编辑] (y: 58~88)
            if 58 <= y <= 88 and tab == HubState.TAB_WHITELIST:
                if 760 <= x <= 846:
                    self.state.refresh_whitelist_cache()
                    self.state.set_toast("已刷新 Tag 白名单状态。")
                    return
                if 854 <= x <= 944:
                    self._handle_tag_whitelist()
                    return

            # 5.5.1.5 体检报告页签：工位信息卡片内嵌按钮点击处理 (x: 864~938, 776~854)
            if tab == HubState.TAB_REPORT:
                # [重命名]
                if 864 <= x <= 938 and 68 <= y <= 94:
                    self._handle_rename_workspace()
                    return
                # [打开]
                if 864 <= x <= 938 and 96 <= y <= 122:
                    self._handle_open_directory()
                    return
                # [修改]
                if 864 <= x <= 938 and 152 <= y <= 178:
                    self._handle_edit_description()
                    return
                # [更新元数据] (移至卡片1底栏左侧, x: 372~504, y: 190~220)
                if 372 <= x <= 504 and 190 <= y <= 220:
                    self._handle_sync_data_consistency()
                    return
                # [克隆工位] (紧随其后 x: 512~592, y: 190~220)
                if (512 <= x <= 592 or 776 <= x <= 854) and 190 <= y <= 220:
                    self._handle_clone_workspace()
                    return
                # [删除] (x: 600~678, y: 190~220)
                if (600 <= x <= 678 or 864 <= x <= 938) and 190 <= y <= 220:
                    self._handle_delete_workspace()
                    return

            # 5.5.2 图片卡片网格墙点击: 单击选中卡片, 双击放大查看
            cell_idx = grid_hit_test(x, y)
            if cell_idx is not None:
                if tab == HubState.TAB_CALIB_IMAGES:
                    target = self.state.image_grid_offset + cell_idx
                    if 0 <= target < len(self.state.current_images):
                        self.state.select_image_at_index(target)
                        if event == cv2.EVENT_LBUTTONDBLCLK:
                            self.state.toggle_expanded_preview()
                elif tab == HubState.TAB_PROD_IMAGES:
                    target = self.state.prod_grid_offset + cell_idx
                    if 0 <= target < len(self.state.prod_images):
                        self.state.select_prod_image_at_index(target)
                return

    def _select_image_for_active_tab(self, delta: int):
        """按当前激活页签切换对应的相册照片 (标定相册/生产相册; 其余页签无相册则忽略)"""
        if self.state.view_mode == HubState.VIEW_EXPANDED or self.state.active_tab == HubState.TAB_CALIB_IMAGES:
            self.state.select_image_by_offset(delta)
        elif self.state.active_tab == HubState.TAB_PROD_IMAGES:
            self.state.select_prod_image_by_offset(delta)

    def _scroll_grid_for_active_tab(self, delta_rows: int):
        """按当前激活页签滚动卡片网格 (滚轮驱动)"""
        if self.state.active_tab == HubState.TAB_PROD_IMAGES:
            self.state.scroll_prod_grid(delta_rows)
        else:
            self.state.scroll_image_grid(delta_rows)

    def _handle_open_directory(self):
        """在系统资源管理器中打开工位目录"""
        ws = self.state.get_selected_workspace()
        if ws and os.path.exists(ws.workspace_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(ws.workspace_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", ws.workspace_dir])
                else:
                    subprocess.run(["xdg-open", ws.workspace_dir])
                self.state.set_toast(f"已在资源管理器中打开: {ws.name}")
            except Exception as e:
                self.state.set_toast(f"打开目录异常: {e}")

    def _handle_delete_workspace(self):
        """删除当前 Workspace"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        ok, msg = self.workspace_mgr.delete_workspace(ws.workspace_id)
        self.state.refresh_workspaces()
        self.state.set_toast(msg)

    def _run_subtool(self, cmd: list, desc: str):
        """统一子工具拉起执行器：销毁主窗、运行子工具、恢复环境与刷新状态"""
        self.state.set_toast(f"正在唤起 {desc}...")
        cv2.destroyAllWindows()

        try:
            subprocess.run(cmd)
        except Exception as e:
            log.warning(f"执行工具异常: {e}")

        # 重新创建主窗体并重新绑定鼠标事件
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.renderer.canvas_w, self.renderer.canvas_h)
        cv2.setMouseCallback(self.window_name, self._on_mouse_event)

        ws = self.state.get_selected_workspace()
        if ws:
            ws.refresh_stats()
            ws.save_meta()
        self.state.refresh_workspaces()
        self.state.load_current_workspace_images()
        self.state.set_toast(f"已完成 {desc} 并返回 Workspace 驾驶舱，数据已同步！")

    def _launch_spatial_mapping_studio(self):
        """启动 AprilTag 空间建图工作站 (Spatial Mapping Studio)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return
        cmd = [sys.executable, "-m", "tools.spatial_mapping_studio",
               "--workspace", ws.workspace_id,
               "--images", ws.calib_raw_images_dir,
               "--map", ws.map_path]
        self._run_subtool(cmd, "空间建图工作站 (Spatial Mapping Studio)")

    def _launch_image_diagnostics(self):
        """启动标靶单帧漏检病因深度切片与梯度诊断"""
        cmd = [sys.executable, "tools/calibration/diagnose_tag_frame.py"]
        self._run_subtool(cmd, "图像深度病因诊断切片系统")

    def _launch_tag_generator(self):
        """启动 AprilTag 标靶图纸生成与 1:1 A4 排版"""
        cmd = [sys.executable, "tools/calibration/generate_apriltags.py"]
        self._run_subtool(cmd, "标靶高清生成与排版工具")

    def _handle_tag_whitelist(self):
        """管理/编辑当前 Workspace 的 AprilTag ID 白名单 (tag_whitelist.yaml)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选择任何 Workspace")
            return

        whitelist_path = self.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        if not os.path.exists(whitelist_path):
            import yaml
            default_config = {
                "workspace_id": ws.workspace_id,
                "workspace_name": ws.name,
                "enabled": False,
                "allowed_ids": ws.valid_tag_ids if ws.valid_tag_ids else [],
                "description": f"Workspace {ws.name} 标靶白名单配置",
                "notes": "enabled 为 true 时仅放行 allowed_ids 中的标靶；为 false 或为空时放行所有检测到的有效标靶",
            }
            try:
                os.makedirs(os.path.dirname(whitelist_path), exist_ok=True)
                with open(whitelist_path, "w", encoding="utf-8") as f:
                    yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            except Exception as e:
                log.warning(f"创建默认 tag_whitelist.yaml 失败: {e}")

        # 使用操作系统关联程序打开文件供现场编辑
        try:
            if sys.platform == "win32":
                os.startfile(whitelist_path)
            else:
                subprocess.Popen(["xdg-open", whitelist_path])
            self.state.set_toast(f"已打开白名单: tag_whitelist.yaml")
        except Exception as e:
            self.state.set_toast(f"打开白名单失败: {e}")

    def _handle_rename_workspace(self):
        """修改 Workspace 显示名称 (支持中文)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        new_name = prompt_input_text(
            "修改 Workspace 名称",
            f"请输入 Workspace【{ws.name}】的新显示名称\n(支持中文、英文、数字，如: 1号机台主标定):",
            initial=ws.name
        )
        if new_name and new_name != ws.name:
            self.state.rename_current_workspace(new_name)

    def _handle_edit_description(self):
        """修改 Workspace 备注说明 (支持中文单行文本)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        cur_desc = getattr(ws, "description", "") or ""
        new_desc = prompt_input_text(
            "修改工位备注",
            f"请输入工位【{ws.name}】的备注信息 (单行文本):",
            initial=cur_desc
        )
        if new_desc is not None:
            self.state.update_current_workspace_description(new_desc.strip())

    def _handle_create_workspace(self):
        """新建 Workspace (支持中文名称弹窗)"""
        idx = len(self.state.workspaces) + 1
        default_alias = f"Workspace_{idx}"

        chosen_name = prompt_input_text(
            "新建 Workspace",
            "请输入新 Workspace 名称/别名 (支持中文、英文、数字，如: 2号机架高位):",
            initial=default_alias
        )
        if not chosen_name:
            self.state.set_toast("已取消新建 Workspace。")
            return

        new_ws = self.workspace_mgr.create_workspace(alias=chosen_name, description=f"Workspace {chosen_name}")
        self.state.refresh_workspaces()
        target_idx = 0
        for i, s in enumerate(self.state.workspaces):
            if s.workspace_id == new_ws.workspace_id:
                target_idx = i
                break
        self.state.select_workspace_at_index(target_idx)
        self.state.set_toast(f"已成功新建 Workspace: 【{new_ws.name}】({new_ws.workspace_id})，按 [C] 开始采图！")

    def _handle_clone_workspace(self):
        """克隆 Workspace (支持中文名称弹窗)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何 Workspace，无法克隆！")
            return

        default_clone_name = f"{ws.name}_对照组"
        chosen_name = prompt_input_text(
            "克隆 Workspace",
            f"请输入克隆后的新 Workspace 名称 (基于原 Workspace【{ws.name}】):",
            initial=default_clone_name
        )
        if not chosen_name:
            return

        cloned = self.workspace_mgr.clone_workspace(ws.workspace_id, new_alias=chosen_name)
        if cloned:
            # 立即刷新 Workspace 列表
            self.state.refresh_workspaces()
            target_idx = 0
            for i, s in enumerate(self.state.workspaces):
                if s.workspace_id == cloned.workspace_id:
                    target_idx = i
                    break
            self.state.select_workspace_at_index(target_idx)
            self.state.set_toast(f"已成功克隆 Workspace: 【{cloned.name}】并定位至新 Workspace！")
        else:
            self.state.set_toast("克隆 Workspace 失败，请检查源目录！")

    def _handle_delete_workspace(self):
        """删除当前选中的 Workspace"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何 Workspace，无法删除！")
            return

        if len(self.state.workspaces) <= 1:
            self.state.set_toast("至少需保留一个工位，禁止删除唯一工位！")
            return

        confirmed = prompt_confirm(
            "确认删除 Workspace",
            f"确定要永久删除工位【{ws.name}】吗？\n\n物理ID: {ws.workspace_id}\n此操作将删除该工位的所有图片和标定数据，不可恢复！"
        )
        if not confirmed:
            self.state.set_toast("已取消删除操作。")
            return

        ok, msg = self.workspace_mgr.delete_workspace(ws.workspace_id)
        if ok:
            self.state.refresh_workspaces()
            self.state.set_toast(f"已成功删除工位: 【{ws.name}】")
        else:
            self.state.set_toast(f"删除工位失败: {msg}")

    def _handle_sync_data_consistency(self):
        """核验物理磁盘与元数据一致性，重新扫描并自动自愈同步"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何工位，无法核验！")
            return

        old_calib = ws.image_count
        old_prod = ws.prod_image_count

        # 1. 强制重新扫描物理磁盘并更新元数据
        ws.refresh_stats()
        ws.save_meta()

        # 2. 刷新相册与列表数据
        self.state.load_current_workspace_images()
        self.state.load_prod_images()
        self.state.refresh_workspaces()

        diff_calib = ws.image_count - old_calib
        diff_prod = ws.prod_image_count - old_prod
        if diff_calib == 0 and diff_prod == 0:
            self.state.set_toast(f"一致性核验完成: 物理与元数据已是最新 (标定 {ws.image_count} 帧, 生产 {ws.prod_image_count} 帧)")
        else:
            self.state.set_toast(f"已同步数据一致性: 标定 {old_calib}→{ws.image_count} 帧, 生产 {old_prod}→{ws.prod_image_count} 帧")


def main():
    parser = argparse.ArgumentParser(description="工作空间综合管理中枢 (Workspace Hub)")
    parser.add_argument("--mock", action="store_true", help="强制以模拟仿真相机模式运行")
    args = parser.parse_args()

    app = WorkspaceHubApp(force_mock=args.mock)
    app.run()


if __name__ == "__main__":
    main()
