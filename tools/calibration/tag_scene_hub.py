"""
AprilTag 标定采样场景综合管理驾驶舱 (Scene Hub)
==============================================
提供现代深色科技风格 GUI 界面：
- 场景画廊管理 (选择、切换、新建、重命名、克隆、删除)
- 历史采样照片缩略图流与单帧大图自适应视口 (支持 [F] 键全宽放大)
- 场景几何健康度与两阶段 BA 平差残差看板
- 原地无缝 1080P/720P 相机取流与空格连拍自动归档
- 一键直通离线 Studio 深度平差与原子发布至生产环境
- 完美支持中英文场景别名输入与显示
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

from src.calibration.scene_manager import CalibrationSceneManager
from tools.calibration.scene_hub.hub_state import HubState
from tools.calibration.scene_hub.hub_renderer import HubRenderer


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


class TagSceneHubApp:
    """Scene Hub 主应用"""

    def __init__(self, force_mock: bool = False):
        self.scene_mgr = CalibrationSceneManager()
        self.state = HubState(self.scene_mgr, force_mock=force_mock)
        self.renderer = HubRenderer()
        self.window_name = "flux_vision_3d | AprilTag Scene Hub"

    def run(self):
        """主事件循环"""
        # 使用 WINDOW_NORMAL 支持自由拖动缩放与最大化占满屏幕
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._on_mouse_event)

        while True:
            # 渲染画面
            canvas = self.renderer.render(self.state)
            cv2.imshow(self.window_name, canvas)

            # 使用 waitKeyEx 兼容 Windows 扩展方向键
            raw_key = cv2.waitKeyEx(15)
            if raw_key == -1:
                continue

            key = raw_key & 0xFF

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

            # =================== 标定工具箱总菜单模式事件 ===================
            if self.state.is_toolbox_open:
                # [ESC] 或 [M] 或 [Q]: 关闭工具箱返回主看板
                if raw_key in (27, ord('q'), ord('Q'), ord('m'), ord('M')):
                    self.state.is_toolbox_open = False
                    self.state.set_toast("已关闭标定工具箱。")
                    continue

                # 工具快捷键直达
                if key in (ord('s'), ord('S')):
                    self.state.is_toolbox_open = False
                    self._launch_offline_studio()
                    continue
                elif key in (ord('a'), ord('A')):
                    self.state.is_toolbox_open = False
                    self._launch_ar_verifier()
                    continue
                elif key in (ord('l'), ord('L')):
                    self.state.is_toolbox_open = False
                    self._launch_offline_verifier()
                    continue
                elif key in (ord('d'), ord('D')):
                    self.state.is_toolbox_open = False
                    self._launch_image_diagnostics()
                    continue
                elif key in (ord('t'), ord('T')):
                    self.state.is_toolbox_open = False
                    self._launch_tag_generator()
                    continue
                elif key in (ord('w'), ord('W')):
                    self.state.is_toolbox_open = False
                    self._handle_tag_whitelist()
                    continue

                continue

            # [ESC] 或 [Q] 退出逻辑 (非工具箱与说明模式)
            if raw_key in (27, ord('q'), ord('Q')):
                if self.state.mode == HubState.MODE_CAPTURE:
                    # 仅退出采图视口，返回三栏看板
                    self.state.mode = HubState.MODE_INSPECTOR
                    self.state.camera_streamer.stop()
                    self.state.set_toast("已退出采图向导，返回三栏看板。")
                    continue
                else:
                    # 退出整个 Scene Hub
                    break

            # [M] 切换标定综合工具箱菜单
            if key in (ord('m'), ord('M')):
                self.state.toggle_toolbox()
                continue

            # [H] 切换生产机制解析说明窗
            if key in (ord('h'), ord('H')):
                self.state.toggle_help_modal()
                continue

            # [C] 切换采图模式
            if key in (ord('c'), ord('C')):
                self._toggle_capture_mode()
                continue

            # =================== 采图模式下的事件 ===================
            if self.state.mode == HubState.MODE_CAPTURE:
                # [Space] 空格抓拍
                if raw_key == 32:
                    ok, frame = self.state.camera_streamer.read()
                    if ok and frame is not None:
                        saved_path = self.state.save_capture_frame(frame)
                    continue

                # [S] 拍完直接平差
                if key in (ord('s'), ord('S')):
                    self.state.camera_streamer.stop()
                    self.state.mode = HubState.MODE_INSPECTOR
                    self._launch_offline_studio()
                    continue

                continue

            # =================== 标准看板模式下的事件 (多键位全覆盖) ===================
            # [↑] 上方向键: Windows waitKeyEx 2490368 / 65362 或 'w' / 小键盘 8
            if raw_key in (2490368, 65362, ord('w'), ord('W'), ord('8')) or key in (ord('w'), ord('W')):
                self.state.select_scene_by_offset(-1)

            # [↓] 下方向键: Windows waitKeyEx 2621440 / 65364 或 's' / 小键盘 2
            elif raw_key in (2621440, 65364, ord('s'), ord('S'), ord('2')) or key in (ord('s'), ord('S')):
                self.state.select_scene_by_offset(1)

            # [←] 左方向键: Windows waitKeyEx 2424832 / 65361 或 'a' / 小键盘 4 / 'j'
            elif raw_key in (2424832, 65361, ord('a'), ord('A'), ord('4'), ord('j'), ord('J')) or key in (ord('a'), ord('A')):
                self.state.select_image_by_offset(-1)

            # [→] 右方向键: Windows waitKeyEx 2555904 / 65363 或 'd' / 小键盘 6 / 'l'
            elif raw_key in (2555904, 65363, ord('d'), ord('D'), ord('6'), ord('l'), ord('L')) or key in (ord('d'), ord('D')):
                self.state.select_image_by_offset(1)

            # [Enter] (回车键: 13, 10): 设为全局活动场景
            elif raw_key in (13, 10):
                self.state.set_current_as_active()

            # [F] 切换单帧大图全宽自适应占满 / 标准三栏看板模式
            elif key in (ord('f'), ord('F')):
                self.state.toggle_expanded_preview()

            # [R] 重命名当前场景显示名称 (支持中文)
            elif key in (ord('r'), ord('R')):
                self._handle_rename_scene()

            # [N] 新建工况场景 (支持弹窗输入中文别名)
            elif key in (ord('n'), ord('N')):
                self._handle_create_scene()

            # [O] 或 [S] 启动离线 Studio 深度平差
            elif key in (ord('o'), ord('O'), ord('s'), ord('S')):
                self._launch_offline_studio()

            # [P] 生效为生产运行地图 (覆盖全局 config/tags_map.yaml)
            elif key in (ord('p'), ord('P')):
                self._handle_publish_to_production()

            # [K] 克隆场景副本
            elif key in (ord('k'), ord('K')):
                self._handle_clone_scene()

            # [V] 打开本地场景目录
            elif key in (ord('v'), ord('V')):
                self._handle_open_directory()

            # [X] 或 [Delete] 删除场景
            elif raw_key in (ord('x'), ord('X'), 3014656):
                self._handle_delete_scene()

        # 退出清理
        self.state.camera_streamer.stop()
        cv2.destroyAllWindows()

    def _toggle_capture_mode(self):
        """切入或退出相机实时连拍向导"""
        if self.state.mode == HubState.MODE_INSPECTOR:
            self.state.mode = HubState.MODE_CAPTURE
            self.state.camera_streamer.start()
            sc = self.state.get_selected_scene()
            sid = sc.name if sc else ""
            self.state.set_toast(f"已切入相机连拍向导 (按空格抓拍，保存至场景【{sid}】)")
        else:
            self.state.mode = HubState.MODE_INSPECTOR
            self.state.camera_streamer.stop()
            self.state.set_toast("已返回三栏看板。")

    def _on_mouse_event(self, event, x, y, flags, param):
        """处理鼠标点击、悬浮 Hover 与滚轮切片交互"""
        # 1. 实时跟踪鼠标坐标，支持全部按钮平滑 Hover 高亮
        if event == cv2.EVENT_MOUSEMOVE:
            self.state.mouse_x = x
            self.state.mouse_y = y
            return

        # 2. 鼠标滚轮极速翻页/切换场景
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = -1 if flags > 0 else 1
            if x <= 340 and 80 <= y <= 380:
                self.state.select_scene_by_offset(delta)
            else:
                self.state.select_image_by_offset(delta)
            return

        # 后续仅处理鼠标左键点击
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # =================== 3. 生产机制 Help 说明窗下的点击 ===================
        if self.state.is_help_modal_open:
            modal_w, modal_h = 860, 490
            mx = (1280 - modal_w) // 2
            my = (720 - modal_h) // 2

            # 点击右上角 [X] 关闭按钮
            if (mx + modal_w - 116) <= x <= (mx + modal_w - 16) and (my + 11) <= y <= (my + 43):
                self.state.is_help_modal_open = False
                self.state.set_toast("已关闭说明窗。")
                return

            # 点击弹窗外部半透明遮罩：关闭
            if x < mx or x > mx + modal_w or y < my or y > my + modal_h:
                self.state.is_help_modal_open = False
                self.state.set_toast("已关闭说明窗。")
            return

        # =================== 4. 工具箱模式下的鼠标点击 ===================
        if self.state.is_toolbox_open:
            modal_w, modal_h = 880, 520
            mx = (1280 - modal_w) // 2  # 200
            my = (720 - modal_h) // 2   # 100

            # 点击右上角 [X] 关闭按钮
            if (mx + modal_w - 120) <= x <= (mx + modal_w - 16) and (my + 10) <= y <= (my + 44):
                self.state.is_toolbox_open = False
                self.state.set_toast("已关闭标定工具箱。")
                return

            # 点击 6 个工具卡片
            cw, ch = 398, 86
            col_xs = [mx + 28, mx + 454]
            row_ys = [my + 96, my + 196, my + 296]

            tool_actions = [
                self._launch_offline_studio,
                self._launch_ar_verifier,
                self._launch_offline_verifier,
                self._launch_image_diagnostics,
                self._launch_tag_generator,
                self._handle_tag_whitelist,
            ]

            for idx, action in enumerate(tool_actions):
                col_i = idx % 2
                row_i = idx // 2
                bx = col_xs[col_i]
                by = row_ys[row_i]
                if bx <= x <= bx + cw and by <= y <= by + ch:
                    self.state.is_toolbox_open = False
                    action()
                    return

            # 点击弹窗外部阴影区域：关闭工具箱
            if x < mx or x > mx + modal_w or y < my or y > my + modal_h:
                self.state.is_toolbox_open = False
                self.state.set_toast("已关闭标定工具箱。")
            return

        # =================== 5. 正常看板与采图模式下的鼠标点击 ===================
        # 点击顶部标题栏 [M] 工具箱菜单按钮 (x: 1040~1180, y: 8~42)
        if 1040 <= x <= 1180 and 8 <= y <= 42:
            self.state.toggle_toolbox()
            return

        # 点击顶部标题栏【生产运行场景】标签区域：秒级唤出业务机制说明窗 (x: 600~1030, y: 8~42)
        if 600 <= x <= 1030 and 8 <= y <= 42:
            self.state.toggle_help_modal()
            return

        # 如果在相机采图全屏模式，点击画面抓拍
        if self.state.mode == HubState.MODE_CAPTURE:
            if 50 < y < 670:
                ok, frame = self.state.camera_streamer.read()
                if ok and frame is not None:
                    self.state.save_capture_frame(frame)
            return

        # 5.1 点击左侧场景列表卡片 (x: 10~330, y: 88~370)
        if 10 <= x <= 330 and 88 <= y <= 370:
            card_h = 66
            gap = 6
            idx_in_view = (y - 88) // (card_h + gap)
            max_cards = 4
            scroll_start = max(0, self.state.selected_scene_idx - max_cards + 1)
            target_idx = scroll_start + idx_in_view
            if 0 <= target_idx < len(self.state.scenes):
                card_cy = 88 + idx_in_view * (card_h + gap)
                # 检查是否直接点击了右侧的 [活动] 或 ★生产/草稿 交互徽章 (x: 248~326, y: card_cy+4 ~ card_cy+62)
                if 248 <= x <= 326 and (card_cy + 4 <= y <= card_cy + 62):
                    self.state.toggle_help_modal()
                    return

                self.state.selected_scene_idx = target_idx
                self.state.load_current_scene_images()
            return

        # 5.2 点击左侧场景管理按钮 (y: 412~490)
        # 按钮 1: 新建场景 [N] (x: 10~165, y: 412~450)
        if 10 <= x <= 165 and 412 <= y <= 450:
            self._handle_create_scene()
            return
        # 按钮 2: 修改名称 [R] (x: 175~330, y: 412~450)
        if 175 <= x <= 330 and 412 <= y <= 450:
            self._handle_rename_scene()
            return
        # 按钮 3: 克隆场景 [K] (x: 10~165, y: 452~490)
        if 10 <= x <= 165 and 452 <= y <= 490:
            self._handle_clone_scene()
            return
        # 按钮 4: 打开目录 [V] (x: 175~330, y: 452~490)
        if 175 <= x <= 330 and 452 <= y <= 490:
            self._handle_open_directory()
            return

        # 5.3 点击左侧核心工作流通道 (x: 10~330, y: 536~680)
        if 10 <= x <= 330:
            if 536 <= y <= 580:
                self._toggle_capture_mode()
                return
            elif 584 <= y <= 628:
                self._launch_offline_studio()
                return
            elif 632 <= y <= 678:
                # 判断是否点击右侧的专属 [? Help] 按钮 (x: 262~326, y: 638~672)
                if 260 <= x <= 326 and 636 <= y <= 672:
                    self.state.toggle_help_modal()
                else:
                    self._handle_publish_to_production()
                return

        # 5.4 全宽大图预览模式下的右上角按钮交互 (x: 340~1280)
        if self.state.expanded_preview_mode:
            # [◀] 上张按钮 (x: 1280-364 ~ 1280-280, y: 60~92)
            if (1280 - 364) <= x <= (1280 - 280) and 60 <= y <= 92:
                self.state.select_image_by_offset(-1)
                return
            # [▶] 下张按钮 (x: 1280-274 ~ 1280-190, y: 60~92)
            if (1280 - 274) <= x <= (1280 - 190) and 60 <= y <= 92:
                self.state.select_image_by_offset(1)
                return
            # [F] 退出全宽放大按钮 (x: 1280-184 ~ 1280-20, y: 60~92)
            if (1280 - 184) <= x <= (1280 - 20) and 60 <= y <= 92:
                self.state.toggle_expanded_preview()
                return

            # 点击大图画面本身也可以切换回标准看板
            if 340 <= x <= 1280 and 50 <= y <= 670:
                self.state.toggle_expanded_preview()
                return

        # 5.5 标准三栏看板模式下的右侧相册交互 (x: 800~1280)
        if not self.state.expanded_preview_mode:
            # 顶部实体按钮组:
            # [◀] 按钮 (x: 1280 - 224 ~ 1280 - 184, y: 58~90)
            if (1280 - 224) <= x <= (1280 - 184) and 58 <= y <= 90:
                self.state.select_image_by_offset(-1)
                return
            # [▶] 按钮 (x: 1280 - 178 ~ 1280 - 138, y: 58~90)
            if (1280 - 178) <= x <= (1280 - 138) and 58 <= y <= 90:
                self.state.select_image_by_offset(1)
                return
            # [F] 全宽放大按钮 (x: 1280 - 132 ~ 1280 - 14, y: 58~90)
            if (1280 - 132) <= x <= (1280 - 14) and 58 <= y <= 90:
                self.state.toggle_expanded_preview()
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
        sc = self.state.get_selected_scene()
        if not sc:
            self.state.set_toast("未选中任何场景，无法生效！")
            return

        ok, msg = self.scene_mgr.publish_to_production(sc.scene_id)
        self.state.refresh_scenes()
        if ok:
            toast = f"★ 生产生效成功！已将【{sc.name}】高精度地图覆盖发布至: config/tags_map.yaml"
            self.state.set_toast(toast)
        else:
            self.state.set_toast(f"生效失败: {msg}")

    def _handle_open_directory(self):
        """在系统资源管理器中打开场景目录"""
        sc = self.state.get_selected_scene()
        if sc and os.path.exists(sc.scene_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(sc.scene_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", sc.scene_dir])
                else:
                    subprocess.run(["xdg-open", sc.scene_dir])
                self.state.set_toast(f"已在资源管理器中打开: {sc.name}")
            except Exception as e:
                self.state.set_toast(f"打开目录异常: {e}")

    def _handle_delete_scene(self):
        """删除当前场景"""
        sc = self.state.get_selected_scene()
        if not sc:
            return
        if sc.scene_id == self.state.active_scene_id:
            self.state.set_toast("【安全保护】严禁删除当前活动场景！请先切换活动场景。")
            return

        ok, msg = self.scene_mgr.delete_scene(sc.scene_id)
        self.state.refresh_scenes()
        self.state.set_toast(msg)

    def _run_subtool(self, cmd: list, desc: str):
        """统一子工具拉起执行器：停止当前相机流、销毁主窗、运行子工具、恢复环境与刷新状态"""
        self.state.set_toast(f"正在唤起 {desc}...")
        self.state.camera_streamer.stop()
        cv2.destroyAllWindows()

        try:
            subprocess.run(cmd)
        except Exception as e:
            print(f"[ERROR] 执行工具异常: {e}")

        # 重新创建主窗体并重新绑定鼠标事件
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._on_mouse_event)

        sc = self.state.get_selected_scene()
        if sc:
            sc.refresh_stats()
            sc.save_meta()
        self.state.refresh_scenes()
        self.state.load_current_scene_images()
        self.state.set_toast(f"已完成 {desc} 并返回 Scene Hub，数据已同步！")

    def _launch_offline_studio(self):
        """启动 AprilTag 离线 Studio 深度平差"""
        sc = self.state.get_selected_scene()
        if not sc:
            return
        cmd = [sys.executable, "tools/calibration/tag_offline_studio.py",
               "--images", sc.raw_images_dir,
               "--map", sc.map_path]
        self._run_subtool(cmd, "Offline Studio 深度平差工作站")

    def _launch_ar_verifier(self):
        """启动在线 AR 精度体验与 3D 虚实融合系统"""
        sc = self.state.get_selected_scene()
        map_p = sc.map_path if sc and os.path.exists(sc.map_path) else "config/tags_map.yaml"
        cmd = [sys.executable, "tools/calibration/tag_calibration_verifier.py",
               "--map", map_p]
        if self.state.camera_streamer.is_mock:
            cmd.append("--mock")
        self._run_subtool(cmd, "在线 AR 综合验证系统")

    def _launch_offline_verifier(self):
        """启动离线留一交叉验证 (LOO) 盲测工作台"""
        sc = self.state.get_selected_scene()
        if not sc:
            return
        cmd = [sys.executable, "tools/calibration/tag_offline_verifier.py",
               "--map", sc.map_path,
               "--image_dir", sc.raw_images_dir]
        self._run_subtool(cmd, "离线精度体检与留一盲测工作台")

    def _launch_image_diagnostics(self):
        """启动标靶单帧漏检病因深度切片与梯度诊断"""
        cmd = [sys.executable, "tools/calibration/diagnose_tag_frame.py"]
        self._run_subtool(cmd, "图像深度病因诊断切片系统")

    def _launch_tag_generator(self):
        """启动 AprilTag 标靶图纸生成与 1:1 A4 排版"""
        cmd = [sys.executable, "tools/calibration/generate_apriltags.py"]
        self._run_subtool(cmd, "标靶高清生成与排版工具")

    def _handle_tag_whitelist(self):
        """管理当前场景标靶 ID 白名单"""
        sc = self.state.get_selected_scene()
        sname = sc.name if sc else "默认场景"
        self.state.set_toast(f"标靶白名单: 当前场景【{sname}】默认放行所有有效 16h5 标靶")

    def _handle_rename_scene(self):
        """修改场景显示名称 (支持中文)"""
        sc = self.state.get_selected_scene()
        if not sc:
            return

        new_name = prompt_input_text(
            "修改场景名称",
            f"请输入场景【{sc.name}】的新显示名称\n(支持中文、英文、数字，如: 1号机台主标定):",
            initial=sc.name
        )
        if new_name and new_name != sc.name:
            self.state.rename_current_scene(new_name)

    def _handle_create_scene(self):
        """新建工况场景 (支持中文名称弹窗)"""
        idx = len(self.state.scenes) + 1
        default_alias = f"标定工况_{idx}"

        chosen_name = prompt_input_text(
            "新建采样工况场景",
            "请输入新场景名称/别名 (支持中文、英文、数字，如: 2号机架高位):",
            initial=default_alias
        )
        if not chosen_name:
            self.state.set_toast("已取消新建场景。")
            return

        new_sc = self.scene_mgr.create_scene(alias=chosen_name, description=f"工况场景 {chosen_name}")
        self.state.refresh_scenes()
        target_idx = 0
        for i, s in enumerate(self.state.scenes):
            if s.scene_id == new_sc.scene_id:
                target_idx = i
                break
        self.state.selected_scene_idx = target_idx
        self.state.selected_image_idx = 0
        self.state.image_strip_offset = 0
        self.state.load_current_scene_images()
        self.state.set_toast(f"已成功新建场景: 【{new_sc.name}】({new_sc.scene_id})，按 [C] 可立即开始采图！")

    def _handle_clone_scene(self):
        """克隆场景 (支持中文名称弹窗)"""
        sc = self.state.get_selected_scene()
        if not sc:
            self.state.set_toast("未选中任何场景，无法克隆！")
            return

        default_clone_name = f"{sc.name}_对照组"
        chosen_name = prompt_input_text(
            "克隆场景",
            f"请输入克隆后的新场景名称 (基于原场景【{sc.name}】):",
            initial=default_clone_name
        )
        if not chosen_name:
            return

        cloned = self.scene_mgr.clone_scene(sc.scene_id, new_alias=chosen_name)
        if cloned:
            # 立即刷新场景列表
            self.state.refresh_scenes()
            target_idx = 0
            for i, s in enumerate(self.state.scenes):
                if s.scene_id == cloned.scene_id:
                    target_idx = i
                    break
            self.state.selected_scene_idx = target_idx
            self.state.selected_image_idx = 0
            self.state.image_strip_offset = 0
            self.state.load_current_scene_images()
            self.state.set_toast(f"已成功克隆场景: 【{cloned.name}】并定位至新场景！")
        else:
            self.state.set_toast("克隆场景失败，请检查源场景目录！")


def main():
    parser = argparse.ArgumentParser(description="AprilTag 标定采样场景综合管理驾驶舱 (Scene Hub)")
    parser.add_argument("--mock", action="store_true", help="强制以模拟仿真相机模式运行")
    args = parser.parse_args()

    app = TagSceneHubApp(force_mock=args.mock)
    app.run()


if __name__ == "__main__":
    main()
