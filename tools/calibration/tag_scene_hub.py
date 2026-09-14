"""
AprilTag 标定采样场景综合管理驾驶舱 (Scene Hub)
==============================================
提供现代深色科技风格 GUI 界面：
- 场景画廊管理 (选择、切换、新建、克隆、删除)
- 历史采样照片缩略图瀑布流与单帧大图高清视口
- 场景几何健康度与两阶段 BA 平差残差看板
- 原地无缝 1080P/720P 相机取流与空格连拍自动归档
- 一键直通离线 Studio 深度平差与原子发布至生产环境
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


class TagSceneHubApp:
    """Scene Hub 主应用"""

    def __init__(self, force_mock: bool = False):
        self.scene_mgr = CalibrationSceneManager()
        self.state = HubState(self.scene_mgr, force_mock=force_mock)
        self.renderer = HubRenderer()
        self.window_name = "flux_vision_3d | AprilTag Scene Hub"

    def run(self):
        """主事件循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

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
                    # 仅退出采图视口，返回画廊看板
                    self.state.mode = HubState.MODE_INSPECTOR
                    self.state.camera_streamer.stop()
                    self.state.set_toast("已退出采图向导，返回场景看板。")
                    continue
                else:
                    # 退出整个 Scene Hub
                    break

            # [C] 切换采图模式
            if key in (ord('c'), ord('C')):
                if self.state.mode == HubState.MODE_INSPECTOR:
                    self.state.mode = HubState.MODE_CAPTURE
                    self.state.camera_streamer.start()
                    sc = self.state.get_selected_scene()
                    sid = sc.scene_id if sc else ""
                    self.state.set_toast(f"已激活实时采图视口 (按空格抓拍，保存至 {sid})")
                else:
                    self.state.mode = HubState.MODE_INSPECTOR
                    self.state.camera_streamer.stop()
                    self.state.set_toast("已返回场景看板。")
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

            # =================== 画廊模式下的事件 ===================
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

            # [O] 启动 Studio
            elif key in (ord('o'), ord('O')):
                self._launch_offline_studio()

            # [P] 一键发布到生产
            elif key in (ord('p'), ord('P')):
                sc = self.state.get_selected_scene()
                if sc:
                    ok, msg = self.scene_mgr.publish_to_production(sc.scene_id)
                    self.state.refresh_scenes()
                    self.state.set_toast(msg)

            # [N] 新建工况场景
            elif key in (ord('n'), ord('N')):
                self._handle_create_scene()

            # [K] 克隆场景
            elif key in (ord('k'), ord('K')):
                sc = self.state.get_selected_scene()
                if sc:
                    cloned = self.scene_mgr.clone_scene(sc.scene_id)
                    if cloned:
                        self.state.refresh_scenes()
                        # 选中刚克隆的场景
                        for i, s in enumerate(self.state.scenes):
                            if s.scene_id == cloned.scene_id:
                                self.state.selected_scene_idx = i
                                break
                        self.state.load_current_scene_images()
                        self.state.set_toast(f"已成功克隆场景: {cloned.scene_id}")

            # [V] 打开本地目录
            elif key in (ord('v'), ord('V')):
                sc = self.state.get_selected_scene()
                if sc and os.path.exists(sc.scene_dir):
                    try:
                        if sys.platform == "win32":
                            os.startfile(sc.scene_dir)
                        elif sys.platform == "darwin":
                            subprocess.run(["open", sc.scene_dir])
                        else:
                            subprocess.run(["xdg-open", sc.scene_dir])
                        self.state.set_toast(f"已在资源管理器中打开: {sc.scene_id}")
                    except Exception as e:
                        self.state.set_toast(f"打开目录异常: {e}")

            # [X] 或 [Delete] 删除场景
            elif raw_key in (ord('x'), ord('X'), 3014656):
                sc = self.state.get_selected_scene()
                if sc:
                    if sc.scene_id == self.state.active_scene_id:
                        self.state.set_toast("【安全保护】严禁删除当前活动场景！请先切换活动场景。")
                    else:
                        ok, msg = self.scene_mgr.delete_scene(sc.scene_id)
                        self.state.refresh_scenes()
                        self.state.set_toast(msg)

        # 退出清理
        self.state.camera_streamer.stop()
        cv2.destroyAllWindows()

    def _launch_offline_studio(self):
        """直通离线 Studio 深度平差并在退出后刷新状态"""
        sc = self.state.get_selected_scene()
        if not sc:
            return

        self.state.set_toast(f"正在唤起 Offline Studio 深度平差工作站...")
        # 隐藏当前窗口以防焦点冲突
        cv2.destroyAllWindows()

        cmd = [sys.executable, "tools/calibration/tag_offline_studio.py",
               "--images", sc.raw_images_dir,
               "--map", sc.map_path]
        subprocess.run(cmd)

        # 重新初始化窗口与刷新场景状态
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        sc.refresh_stats()
        sc.save_meta()
        self.state.refresh_scenes()
        self.state.load_current_scene_images()
        self.state.set_toast(f"已完成 Studio 平差并返回 Scene Hub，数据已最新！")

    def _handle_create_scene(self):
        """新建场景快速创建"""
        # 自动生成递增序号别名，如 bench_a, bench_b ...
        existing_aliases = [s.name for s in self.state.scenes]
        idx = 1
        while True:
            candidate = f"bench_site{idx}"
            if candidate not in existing_aliases:
                break
            idx += 1

        new_sc = self.scene_mgr.create_scene(alias=candidate, description=f"工况采样子集 {candidate}")
        self.state.refresh_scenes()
        for i, s in enumerate(self.state.scenes):
            if s.scene_id == new_sc.scene_id:
                self.state.selected_scene_idx = i
                break
        self.state.load_current_scene_images()
        self.state.set_toast(f"已新建场景: {new_sc.scene_id}，按 [C] 可立即开始采图！")


def main():
    parser = argparse.ArgumentParser(description="AprilTag 标定采样场景综合管理驾驶舱 (Scene Hub)")
    parser.add_argument("--mock", action="store_true", help="强制以模拟仿真相机模式运行")
    args = parser.parse_args()

    app = TagSceneHubApp(force_mock=args.mock)
    app.run()


if __name__ == "__main__":
    main()
