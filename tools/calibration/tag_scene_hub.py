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

            # [ESC] 或 [Q] 退出逻辑
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

            # =================== 标准看板模式下的事件 ===================
            # [↑] 上方向键: Windows waitKeyEx code 2490368 或 'w'
            if raw_key in (2490368, ord('w'), ord('W')):
                self.state.select_scene_by_offset(-1)

            # [↓] 下方向键: Windows waitKeyEx code 2621440 或 's'
            elif raw_key in (2621440, ord('s'), ord('S')):
                self.state.select_scene_by_offset(1)

            # [←] 左方向键: Windows waitKeyEx code 2424832 或 'a'
            elif raw_key in (2424832, ord('a'), ord('A')):
                self.state.select_image_by_offset(-1)

            # [→] 右方向键: Windows waitKeyEx code 2555904 或 'd'
            elif raw_key in (2555904, ord('d'), ord('D')):
                self.state.select_image_by_offset(1)

            # [Enter] (回车键: 13): 设为全局活动场景
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
        """处理鼠标点击交互：卡片点击、按钮点击、相册选图与大图切换"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 如果在相机采图全屏模式，点击画面抓拍
        if self.state.mode == HubState.MODE_CAPTURE:
            if y > 50 and y < 670:
                ok, frame = self.state.camera_streamer.read()
                if ok and frame is not None:
                    self.state.save_capture_frame(frame)
            return

        # 1. 点击左侧场景列表卡片 (x: 10~330, y: 88~370)
        if 10 <= x <= 330 and 88 <= y <= 370:
            card_h = 66
            gap = 6
            idx_in_view = (y - 88) // (card_h + gap)
            max_cards = 4
            scroll_start = max(0, self.state.selected_scene_idx - max_cards + 1)
            target_idx = scroll_start + idx_in_view
            if 0 <= target_idx < len(self.state.scenes):
                self.state.selected_scene_idx = target_idx
                self.state.load_current_scene_images()
            return

        # 2. 点击左侧场景管理按钮 (y: 414~490)
        # 按钮 1: 新建场景 [N] (x: 10~165, y: 414~450)
        if 10 <= x <= 165 and 414 <= y <= 450:
            self._handle_create_scene()
            return
        # 按钮 2: 修改名称 [R] (x: 175~330, y: 414~450)
        if 175 <= x <= 330 and 414 <= y <= 450:
            self._handle_rename_scene()
            return
        # 按钮 3: 克隆场景 [K] (x: 10~165, y: 456~490)
        if 10 <= x <= 165 and 456 <= y <= 490:
            self._handle_clone_scene()
            return
        # 按钮 4: 打开目录 [V] (x: 175~330, y: 456~490)
        if 175 <= x <= 330 and 456 <= y <= 490:
            self._handle_open_directory()
            return

        # 3. 点击左侧核心工作流通道 (x: 10~330, y: 536~680)
        if 10 <= x <= 330:
            if 536 <= y <= 580:
                self._toggle_capture_mode()
                return
            elif 586 <= y <= 630:
                self._launch_offline_studio()
                return
            elif 636 <= y <= 680:
                self._handle_publish_to_production()
                return

        # 4. 点击最右侧相册缩略图 (x: 816~1260, y: 94~156)
        if not self.state.expanded_preview_mode and 816 <= x <= 1260 and 94 <= y <= 156:
            tw = 98
            pad = 8
            thumb_idx = (x - 816) // (tw + pad)
            offset = self.state.image_strip_offset
            target_img_idx = offset + thumb_idx
            if 0 <= target_img_idx < len(self.state.current_images):
                self.state.selected_image_idx = target_img_idx
            return

        # 5. 点击大图预览视口：触发 [F] 模式切换
        if (not self.state.expanded_preview_mode and 816 <= x <= 1260 and 168 <= y <= 660) or \
           (self.state.expanded_preview_mode and 340 <= x <= 1280 and 50 <= y <= 670):
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

    def _launch_offline_studio(self):
        """直通离线 Studio 深度平差并在退出后刷新状态"""
        sc = self.state.get_selected_scene()
        if not sc:
            return

        self.state.set_toast(f"正在唤起 Offline Studio 深度平差工作站...")
        cv2.destroyAllWindows()

        cmd = [sys.executable, "tools/calibration/tag_offline_studio.py",
               "--images", sc.raw_images_dir,
               "--map", sc.map_path]
        subprocess.run(cmd)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        sc.refresh_stats()
        sc.save_meta()
        self.state.refresh_scenes()
        self.state.load_current_scene_images()
        self.state.set_toast(f"已完成 Studio 平差并返回 Scene Hub，数据已最新！")

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
        existing_names = [s.name for s in self.state.scenes]
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
        for i, s in enumerate(self.state.scenes):
            if s.scene_id == new_sc.scene_id:
                self.state.selected_scene_idx = i
                break
        self.state.load_current_scene_images()
        self.state.set_toast(f"已成功新建场景: 【{new_sc.name}】({new_sc.scene_id})，按 [C] 可立即开始采图！")

    def _handle_clone_scene(self):
        """克隆场景 (支持中文名称弹窗)"""
        sc = self.state.get_selected_scene()
        if not sc:
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
            self.state.refresh_scenes()
            for i, s in enumerate(self.state.scenes):
                if s.scene_id == cloned.scene_id:
                    self.state.selected_scene_idx = i
                    break
            self.state.load_current_scene_images()
            self.state.set_toast(f"已成功克隆场景: 【{cloned.name}】")


def main():
    parser = argparse.ArgumentParser(description="AprilTag 标定采样场景综合管理驾驶舱 (Scene Hub)")
    parser.add_argument("--mock", action="store_true", help="强制以模拟仿真相机模式运行")
    args = parser.parse_args()

    app = TagSceneHubApp(force_mock=args.mock)
    app.run()


if __name__ == "__main__":
    main()
