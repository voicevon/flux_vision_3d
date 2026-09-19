"""
工作空间综合管理中枢 (Workspace Hub)
==============================================
提供现代深色科技风格 GUI 界面：
- Workspace 画廊管理 (选择、切换、新建、重命名、克隆、删除)
- 历史采样照片缩略图流与单帧大图自适应视口 (支持 [F] 键全宽放大)
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
from tools.workspace_hub.hub_renderer import HubRenderer, HELP_MODAL_W, HELP_MODAL_H
from src.utils.logger import get_logger

log = get_logger(__name__)


def prompt_input_text(title: str, prompt_text: str, initial: str = "") -> str:
    """弹出轻量级原生 Windows 输入框，完美支持中文拼音/五笔输入法"""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        val = simpledialog.askstring(title, prompt_text, initialvalue=initial, parent=root)
        root.destroy()
        return val.strip() if val else ""
    except Exception:
        print(f"\n{title}: {prompt_text}")
        try:
            return input("请输入: ").strip()
        except Exception:
            return ""


class WorkspaceHubApp:
    """Workspace Hub 主应用"""

    def __init__(self, force_mock: bool = False, settings_file: str = None):
        self.force_mock = force_mock
        self.win_mgr = GuiWindowManager(
            app_id="workspace_hub",
            base_w=1280,
            base_h=720,
            settings_file=settings_file
        )
        self.workspace_mgr = WorkspaceManager()
        self.state = HubState(self.workspace_mgr, force_mock=force_mock)
        self.renderer = HubRenderer()
        # 窗口内部 key 必须纯 ASCII (namedWindow ANSI API), 中文标题走 set_unicode_title
        self.window_name = "flux_vision_3d | workspace"
        self.window_title = "flux_vision_3d | Workspace"
        self._running = True

        if self.win_mgr.scale_pct != 100 or self.win_mgr.canvas_w != 1280 or self.win_mgr.canvas_h != 720:
            self.state.set_toast(f"已恢复偏好设置：放大镜 {self.win_mgr.scale_pct}%，视窗 {self.win_mgr.canvas_w}×{self.win_mgr.canvas_h} (Ctrl+0 复位)")

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

            # 2. 渲染画面并在当前窗口分辨率下自适应居中呈现
            raw_canvas = self.renderer.render(self.state)
            if self.win_mgr.canvas_w == 1280 and self.win_mgr.canvas_h == 720:
                present_canvas = raw_canvas
            else:
                present_canvas = np.full((self.win_mgr.canvas_h, self.win_mgr.canvas_w, 3), (18, 20, 24), dtype=np.uint8)
                scale = min(self.win_mgr.canvas_w / 1280.0, self.win_mgr.canvas_h / 720.0)
                target_w = int(round(1280 * scale))
                target_h = int(round(720 * scale))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                scaled = cv2.resize(raw_canvas, (target_w, target_h), interpolation=interp)
                pad_x = (self.win_mgr.canvas_w - target_w) // 2
                pad_y = (self.win_mgr.canvas_h - target_h) // 2
                present_canvas[pad_y:pad_y + target_h, pad_x:pad_x + target_w] = scaled

            cv2.imshow(self.window_name, present_canvas)

            # 3. 使用 waitKeyEx 兼容 Windows 扩展方向键与业务按键
            raw_key = cv2.waitKeyEx(15)
            if raw_key == -1:
                continue

            # 处理后备键盘缩放 (若未在物理级截获)
            fb_changed, fb_toast = self.win_mgr.handle_keyboard_fallback(raw_key)
            if fb_changed and fb_toast:
                self.state.set_toast(fb_toast)
                continue

            key = raw_key & 0xFF

            # =================== 场景右键上下文菜单打开时的按键处理 ===================
            if self.state.context_menu_open:
                if raw_key in (27, ord('q'), ord('Q')):
                    self.state.close_context_menu()
                    self.state.set_toast("已关闭右键菜单。")
                    continue

            # =================== 生产系统机制 Help 弹窗模式事件 ===================
            if self.state.is_help_modal_open:
                # [ESC] / [H] / [Q] / [M]: 关闭 Help 弹窗
                if raw_key in (27, ord('q'), ord('Q'), ord('h'), ord('H'), ord('m'), ord('M')):
                    self.state.is_help_modal_open = False
                    self.state.set_toast("已关闭生产系统机制说明窗。")
                    continue
                elif key in (ord('p'), ord('P')):
                    self.state.is_help_modal_open = False
                    self._handle_publish_to_production()
                    continue
                continue

            # [ESC] 或 [Q] 退出逻辑 (非说明模式)
            if raw_key in (27, ord('q'), ord('Q')):
                break

            # [H] 切换生产机制解析说明窗
            if key in (ord('h'), ord('H')):
                self.state.toggle_help_modal()
                continue

            # [C] 启动外部专属多视角交互采图向导
            if key in (ord('c'), ord('C')):
                self._launch_capture_wizard()
                continue

            # [W] 管理当前 Workspace 标靶 ID 白名单 (tag_whitelist.yaml)
            if key in (ord('w'), ord('W')):
                self._handle_tag_whitelist()
                continue

            # =================== 标准看板模式下的事件 (多键位全覆盖) ===================
            # [↑] 上方向键: Windows waitKeyEx 2490368 / 65362 或 小键盘 8
            if raw_key in (2490368, 65362, ord('8')) or key in (ord('8'),):
                self.state.select_workspace_by_offset(-1)

            # [↓] 下方向键: Windows waitKeyEx 2621440 / 65364 或 小键盘 2
            elif raw_key in (2621440, 65364, ord('2')) or key in (ord('2'),):
                self.state.select_workspace_by_offset(1)

            # [←] 左方向键: Windows waitKeyEx 2424832 / 65361 或 'a' / 小键盘 4 / 'j'
            elif raw_key in (2424832, 65361, ord('a'), ord('A'), ord('4'), ord('j'), ord('J')) or key in (ord('a'), ord('A')):
                self.state.select_image_by_offset(-1)

            # [→] 右方向键: Windows waitKeyEx 2555904 / 65363 或 'd' / 小键盘 6 / 'l'
            elif raw_key in (2555904, 65363, ord('d'), ord('D'), ord('6'), ord('l'), ord('L')) or key in (ord('d'), ord('D')):
                self.state.select_image_by_offset(1)

            # [Enter] (回车键: 13, 10): 启动离线 Studio 深度平差
            elif raw_key in (13, 10):
                self._launch_offline_studio()

            # [F] 顺次循环切换三模态视图: 标准三栏 -> 全宽大图 -> 纯净健康大屏
            elif key in (ord('f'), ord('F')):
                self.state.cycle_view_mode()

            # [Del] / [Delete] 删除当前选中的照片帧
            elif raw_key in (3014656, 65535, 127, 8) or key in (127, 8):
                self.state.delete_selected_image()

            # [R] 重命名当前 Workspace 显示名称 (支持中文)
            elif key in (ord('r'), ord('R')):
                self._handle_rename_workspace()

            # [N] 新建 Workspace (支持弹窗输入中文别名)
            elif key in (ord('n'), ord('N')):
                self._handle_create_workspace()

            # [O] 或 [S] 启动离线 Studio 深度平差
            elif key in (ord('o'), ord('O'), ord('s'), ord('S')):
                self._launch_offline_studio()

            # [P] 生效为生产运行基准 (原子覆盖生产基准)
            elif key in (ord('p'), ord('P')):
                self._handle_publish_to_production()

            # [K] 克隆 Workspace 副本
            elif key in (ord('k'), ord('K')):
                self._handle_clone_workspace()

            # [V] 打开本地物理目录
            elif key in (ord('v'), ord('V')):
                self._handle_open_directory()

            # [X] 或 [Delete] 删除 Workspace
            elif raw_key in (ord('x'), ord('X'), 3014656):
                self._handle_delete_workspace()

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

        # 0.1 物理坐标转换回 1280x720 逻辑坐标
        if self.win_mgr.canvas_w != 1280 or self.win_mgr.canvas_h != 720:
            scale = min(self.win_mgr.canvas_w / 1280.0, self.win_mgr.canvas_h / 720.0)
            pad_x = (self.win_mgr.canvas_w - int(1280 * scale)) // 2
            pad_y = (self.win_mgr.canvas_h - int(720 * scale)) // 2
            logic_x = int((x - pad_x) / max(1e-6, scale))
            logic_y = int((y - pad_y) / max(1e-6, scale))
            x = max(0, min(1279, logic_x))
            y = max(0, min(719, logic_y))

        # 1. 实时跟踪鼠标坐标，支持全部按钮平滑 Hover 高亮
        if event == cv2.EVENT_MOUSEMOVE:
            self.state.mouse_x = x
            self.state.mouse_y = y
            return

        # 1.1 鼠标右键点击卡片：弹出 Workspace 专属上下文菜单 (Context Menu)
        if event == cv2.EVENT_RBUTTONDOWN:
            if 10 <= x <= 330 and 90 <= y <= 480:
                card_h = 70
                gap = 8
                idx_in_view = (y - 90) // (card_h + gap)
                max_cards = 5
                scroll_start = max(0, self.state.selected_workspace_idx - max_cards + 1)
                target_idx = scroll_start + idx_in_view
                if 0 <= target_idx < len(self.state.workspaces):
                    self.state.open_context_menu(x, y, target_idx)
                    return
            return

        # 2. 普通滚轮极速翻页/切换 Workspace (未按 Ctrl 时)
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = -1 if flags > 0 else 1
            if x <= 340 and 80 <= y <= 480:
                self.state.select_workspace_by_offset(delta)
            else:
                self.state.select_image_by_offset(delta)
            return

        # 后续仅处理鼠标左键点击
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # =================== 2.5 Workspace 右键上下文菜单处于激活状态下的点击 ===================
        if self.state.context_menu_open:
            menu_w = 216
            item_h = 32
            menu_items_count = 6
            menu_h = 34 + menu_items_count * item_h + 6
            mx, my = self.state.context_menu_pos

            # 自适应防超出屏幕边界 (与 hub_renderer 保持一致)
            if mx + menu_w > 1280 - 10:
                mx = 1280 - menu_w - 10
            if my + menu_h > 665:
                my = 665 - menu_h
            if mx < 10:
                mx = 10
            if my < 50:
                my = 50

            # 判定是否点击在具体菜单项上
            if mx <= x <= mx + menu_w and (my + 32) <= y <= (my + 32 + menu_items_count * item_h):
                item_idx = (y - (my + 32)) // item_h
                self.state.close_context_menu()
                if item_idx == 0:
                    self._handle_publish_to_production()
                elif item_idx == 1:
                    self._handle_tag_whitelist()
                elif item_idx == 2:
                    self._handle_rename_workspace()
                elif item_idx == 3:
                    self._handle_clone_workspace()
                elif item_idx == 4:
                    self._handle_open_directory()
                elif item_idx == 5:
                    self._handle_delete_workspace()
                return

            # 点击菜单外部任意区域：安全关闭菜单
            self.state.close_context_menu()
            return

        # =================== 3. 生产机制 Help 说明窗下的点击 ===================
        if self.state.is_help_modal_open:
            mx = (1280 - HELP_MODAL_W) // 2
            my = (720 - HELP_MODAL_H) // 2
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
        # 4.0 顶部标题栏交互 (仅 [H] 与 [X])
        # 4.0.1 [H] 生产机制说明按钮 (x: 940~1070, y: 8~42)
        if 940 <= x <= 1070 and 8 <= y <= 42:
            self.state.toggle_help_modal()
            return

        # 4.0.2 [X] 退出按钮 (x: 1085~1265, y: 8~42)
        if 1085 <= x <= 1265 and 8 <= y <= 42:
            self._running = False
            return

        # 5.1 点击左侧 Workspace 列表卡片 (x: 10~330, y: 90~560, 支持 6 张卡片)
        if 10 <= x <= 330 and 90 <= y <= 560:
            card_h = 70
            gap = 8
            idx_in_view = (y - 90) // (card_h + gap)
            max_cards = 6
            scroll_start = max(0, self.state.selected_workspace_idx - max_cards + 1)
            target_idx = scroll_start + idx_in_view
            if 0 <= target_idx < len(self.state.workspaces):
                target_ws = self.state.workspaces[target_idx]
                card_cy = 90 + idx_in_view * (card_h + gap)

                # 检查是否直接点击了右侧操作胶囊 (x: 226~326, y: card_cy + 18 ~ card_cy + 54)
                if 226 <= x <= 326 and card_cy + 18 <= y <= card_cy + 54:
                    self.state.selected_workspace_idx = target_idx
                    if not target_ws.is_published and target_ws.ba_solved:
                        self._handle_publish_to_production()
                    elif target_ws.is_published:
                        self.state.toggle_help_modal()
                    return

                # 点击卡片其余区域：选中该 Workspace 并载入图像
                self.state.selected_workspace_idx = target_idx
                self.state.load_current_workspace_images()
            return

        # 5.2 点击左侧通用全局 Workspace 管理按钮 (y: 614~654)
        # 按钮 1: 新建 Workspace [N] (x: 10~165, y: 614~654)
        if 10 <= x <= 165 and 614 <= y <= 654:
            self._handle_create_workspace()
            return
        # 按钮 2: 打开 Workspace 物理总目录 [V] (x: 175~330, y: 614~654)
        if 175 <= x <= 330 and 614 <= y <= 654:
            self._handle_open_directory()
            return

        # 5.4 全宽大图预览模式下的右上角按钮交互 (x: 340~1280)
        if self.state.expanded_preview_mode:
            # 顶部按钮行 (y: 58~92)
            if 58 <= y <= 92:
                # [◀] 上张按钮 (x: 810~890)
                if 810 <= x <= 890:
                    self.state.select_image_by_offset(-1)
                    return
                # [▶] 下张按钮 (x: 896~976)
                if 896 <= x <= 976:
                    self.state.select_image_by_offset(1)
                    return
                # [Del] 删帧按钮 (x: 982~1082)
                if 982 <= x <= 1082:
                    self.state.delete_selected_image()
                    return
                # [F] 退出全宽放大按钮 (x: 1088~1260)
                if 1088 <= x <= 1260:
                    self.state.toggle_expanded_preview()
                    return

            # 点击大图画面本身也可以切换回标准看板
            if 340 <= x <= 1280 and 96 <= y <= 670:
                self.state.toggle_expanded_preview()
                return

        # 5.5 标准三栏看板模式下的右侧相册控制栏交互 (x: 800~1280, y: 56~90)
        if not self.state.expanded_preview_mode:
            if 56 <= y <= 90:
                # 5.5.1 三段式视图模式切换 Tab 胶囊 (x: 940~1084)
                if 940 <= x <= 988:
                    self.state.set_view_mode(HubState.VIEW_STANDARD)
                    return
                elif 989 <= x <= 1036:
                    self.state.set_view_mode(HubState.VIEW_EXPANDED)
                    return
                elif 1037 <= x <= 1084:
                    self.state.set_view_mode(HubState.VIEW_DASHBOARD)
                    return

                # 5.5.2 相册控制实体按钮组: [<] [>] [F] [Del]
                # [◀] 按钮
                if 1092 <= x <= 1124:
                    self.state.select_image_by_offset(-1)
                    return
                # [▶] 按钮
                if 1128 <= x <= 1160:
                    self.state.select_image_by_offset(1)
                    return
                # [F] 全宽放大按钮
                if 1164 <= x <= 1214:
                    self.state.toggle_expanded_preview()
                    return
                # [Del] 删除选中照片
                if 1218 <= x <= 1270:
                    self.state.delete_selected_image()
                    return

            # 点击缩略图水平滚动带 (x: 816~1260, y: 96~158)
            if 816 <= x <= 1260 and 96 <= y <= 158:
                tw = 98
                pad = 8
                thumb_idx = (x - 816) // (tw + pad)
                offset = self.state.image_strip_offset
                target_img_idx = offset + thumb_idx
                if 0 <= target_img_idx < len(self.state.current_images):
                    self.state.selected_image_idx = target_img_idx
                return

            # 点击单帧大图视口区域：进入全宽大图模式
            if 816 <= x <= 1260 and 168 <= y <= 660:
                self.state.toggle_expanded_preview()
                return

    def _handle_publish_to_production(self):
        """生效为生产运行地图 (覆盖全局 config/tags_map.yaml)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何工位，无法生效！")
            return

        ok, msg = self.workspace_mgr.publish_to_production(ws.workspace_id)
        self.state.refresh_workspaces()
        if ok:
            toast = f"★ 生产生效成功！已将【{ws.name}】高精度地图覆盖发布至: config/tags_map.yaml"
            self.state.set_toast(toast)
        else:
            self.state.set_toast(f"生效失败: {msg}")

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
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._on_mouse_event)

        ws = self.state.get_selected_workspace()
        if ws:
            ws.refresh_stats()
            ws.save_meta()
        self.state.refresh_workspaces()
        self.state.load_current_workspace_images()
        self.state.set_toast(f"已完成 {desc} 并返回 Workspace 驾驶舱，数据已同步！")

    def _launch_offline_studio(self):
        """启动 AprilTag 离线 Studio 深度平差"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return
        cmd = [sys.executable, "tools/studio/app.py",
               "--workspace", ws.workspace_id,
               "--images", ws.calib_raw_images_dir,
               "--map", ws.map_path]
        self._run_subtool(cmd, "Offline Studio 深度平差工作站")

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
        self.state.selected_workspace_idx = target_idx
        self.state.selected_image_idx = 0
        self.state.image_strip_offset = 0
        self.state.load_current_workspace_images()
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
            self.state.selected_workspace_idx = target_idx
            self.state.selected_image_idx = 0
            self.state.image_strip_offset = 0
            self.state.load_current_workspace_images()
            self.state.set_toast(f"已成功克隆 Workspace: 【{cloned.name}】并定位至新 Workspace！")
        else:
            self.state.set_toast("克隆 Workspace 失败，请检查源目录！")


def main():
    parser = argparse.ArgumentParser(description="工作空间综合管理中枢 (Workspace Hub)")
    parser.add_argument("--mock", action="store_true", help="强制以模拟仿真相机模式运行")
    args = parser.parse_args()

    app = WorkspaceHubApp(force_mock=args.mock)
    app.run()


if __name__ == "__main__":
    main()
