#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视角采图向导 (Capture Wizard)
===================================================
单一职责：多视角采集高质量图像 (纯预览 + 保存)。
用途：
  1. GUI 先行启动 (不自动开相机)：顶部工具栏选相机类型 (RealSense D435 / USB 摄像头) →
     分辨率 → [开启] 乒乓开关 (布局与 Robot 在线跟踪第一排左半部分同款)，点击 [开启] 后进入预览；
  2. 按 [空格] 键一键拍摄保存无标注的高清原始帧至活动场景图像目录；
  3. 提供拍照快门白闪视觉反馈与采样计数，采图完毕后衔接离线空间建图 (tag_map_builder)；
  4. 曝光调节 [ ] 与自动曝光切换 [E] (RealSense 物理感光控制)。
注：Tag 识别/观测解算统一由离线建图管线完成，向导不做任何检测 (先采后验)。
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import cv2

# Windows 终端中文色彩
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass  # 编码重配置失败无伤大雅，终端仍可正常运行

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.calibration.scene_manager import CalibrationSceneManager
    DEFAULT_IMAGE_DIR = CalibrationSceneManager().get_active_scene().raw_images_dir
except (ImportError, RuntimeError):
    DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")

from src.calibration.camera_service import CameraService
from src.utils.text_rendering import draw_text
from src.utils.logger import get_logger
from src.utils.gui_window_manager import GuiWindowManager
from tools.capture.renderer import (
    CaptureRenderer, COLOR_ACCENT, COLOR_TEXT_SUB, COL_YELLOW)

GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "capture_wizard"

try:
    from tools.window_helper import force_window_focus
except ImportError:
    force_window_focus = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

log = get_logger(__name__)


class CaptureWizard:
    def __init__(self, output_dir: str = None, scene_id: str = None):
        from src.calibration.scene_manager import CalibrationSceneManager
        self.scene_mgr = CalibrationSceneManager()
        self.scenes = self.scene_mgr.list_scenes()

        # 确定初始归档场景
        is_custom_output = bool(output_dir and output_dir != DEFAULT_IMAGE_DIR)
        sc = None
        if scene_id:
            sc = self.scene_mgr.get_scene_by_id(scene_id)
        elif is_custom_output:
            norm_target = os.path.normpath(output_dir)
            for s in self.scenes:
                if os.path.normpath(s.raw_images_dir) == norm_target or os.path.normpath(s.scene_dir) == norm_target:
                    sc = s
                    break

        if not sc and not is_custom_output:
            sc = self.scene_mgr.get_active_scene()

        self.current_scene = sc
        self.current_scene_id = sc.scene_id if sc else ""
        self.output_dir = sc.raw_images_dir if sc else (output_dir or DEFAULT_IMAGE_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

        # 硬件与运行时状态 (必须先声明，严禁在后续被覆盖为 None)
        self.is_running = False
        self.flash_timer = 0.0
        self.actual_stream_desc = "相机未开启"
        # 统一取流服务: 硬件启停/帧读取全部委托 CameraService
        self._cam_srv = CameraService()

        # GUI 状态: 启动只加载界面不开相机, 用户选择相机/分辨率后点击 [开启] 才进入预览
        self.win_mgr = GuiWindowManager(app_id=APP_ID)
        self.renderer = CaptureRenderer(self)
        self.active_dropdown = None        # None / CAMERA_TYPE_DROPDOWN / RES_DROPDOWN
        self.pipeline_running = False
        self.camera_type = "realsense"
        self.camera_options = [
            ("realsense", "RealSense D435"),
            ("usb",       "USB 普通摄像头"),
        ]
        self.resolution = "1920x1080"      # 延续现状 1080P 优先 (高像素利于离线建图解算)
        self.resolution_options = [
            ("1920x1080", "1920 × 1080  (推荐)"),
            ("1280x720",  "1280 × 720"),
            ("848x480",   "848 × 480"),
            ("640x480",   "640 × 480"),
        ]
        _w, _h = self.resolution.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)
        self._load_viewer_state()

        # 运行时状态
        self.status_toast = ""
        self.status_toast_time = 0.0
        self.color_sensor = None

        # 统计已有图片数
        existing = glob.glob(os.path.join(self.output_dir, "view_*.png"))
        self.image_count = len(existing)

    @property
    def scene_options(self):
        """动态读取所有可用场景供下拉菜单展示"""
        self.scenes = self.scene_mgr.list_scenes()
        opts = []
        for s in self.scenes:
            tag = "★ " if s.is_published else ""
            opts.append((s.scene_id, f"{tag}{s.name} ({s.image_count}帧)"))
        return opts

    @property
    def current_scene_name(self):
        return self.current_scene.name if self.current_scene else "默认工位"

    def switch_scene(self, scene_id: str):
        """实时切换采图目标场景 (照片自动路由至该场景的 raw_images)"""
        sc = self.scene_mgr.get_scene_by_id(scene_id)
        if not sc:
            return
        self.current_scene = sc
        self.current_scene_id = sc.scene_id
        self.output_dir = sc.raw_images_dir
        os.makedirs(self.output_dir, exist_ok=True)
        existing = glob.glob(os.path.join(self.output_dir, "view_*.png"))
        self.image_count = len(existing)
        self.set_toast(f"已切换归档场景: 【{sc.name}】(当前 {self.image_count} 帧)")
        log.info(f"采图向导已切换归档场景: {sc.name} ({sc.scene_id}) -> {self.output_dir}")

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()

    # ------------------------------ 相机开关 (GUI 先行, 借鉴 Robot 在线跟踪) ------------------------------
    def _load_viewer_state(self):
        """从 config/gui_settings.json 恢复上次退出时的下拉选择 (相机类型/分辨率)"""
        try:
            if not os.path.exists(GUI_SETTINGS_FILE):
                return
            with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            state = (root.get(APP_ID) or {}).get("viewer_state") or {}
            if state.get("camera_type") in ("realsense", "usb"):
                self.camera_type = state["camera_type"]
            if any(k == state.get("resolution") for k, _ in self.resolution_options):
                self.resolution = state["resolution"]
                _w, _h = self.resolution.split("x")
                self.frame_w, self.frame_h = int(_w), int(_h)
        except Exception as e:
            log.warning(f"恢复采图向导状态失败，使用默认配置: {e}")

    def _save_viewer_state(self):
        """保存下拉选择 (相机类型/分辨率) 到 config/gui_settings.json"""
        try:
            root = {}
            if os.path.exists(GUI_SETTINGS_FILE):
                try:
                    with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                        root = json.load(f)
                    if not isinstance(root, dict):
                        root = {}
                except Exception:
                    root = {}
            node = root.setdefault(APP_ID, {})
            node["viewer_state"] = {
                "camera_type": self.camera_type,
                "resolution": self.resolution,
            }
            node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(GUI_SETTINGS_FILE), exist_ok=True)
            with open(GUI_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存采图向导状态失败: {e}")

    def _select_camera_type(self, cam_key):
        """切换相机类型：如果取流已运行则先停再切换"""
        if cam_key == self.camera_type:
            return
        if self.pipeline_running:
            self._stop_camera()
        self.camera_type = cam_key
        self._save_viewer_state()
        log.info(f"相机类型已切换为: {dict(self.camera_options).get(cam_key, cam_key)}")

    def _change_resolution(self, res_key):
        """切换分辨率：运行中则先停再按新分辨率重启"""
        if res_key == self.resolution:
            return
        was_running = self.pipeline_running
        if was_running:
            self._stop_camera()
        self.resolution = res_key
        _w, _h = res_key.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)
        self._save_viewer_state()
        if was_running:
            self._start_camera()
        log.info(f"分辨率已切换: {res_key}")

    def _toggle_camera(self):
        """开启或关闭相机取流 (乒乓)"""
        if self.pipeline_running:
            self._stop_camera()
            self.set_toast("相机已关闭")
            log.info("相机已关闭")
        else:
            self._start_camera()

    def _start_camera(self):
        """按当前类型/分辨率启动取流; 失败 Toast 报错 (现场采图不能误采仿真帧, 无 Mock 降级)"""
        w, h = self.frame_w, self.frame_h
        try:
            if self.camera_type == "realsense":
                # 8fps 高像素模式取流更稳 (延续原 1080P@8fps 现状), 同档内回退 8fps
                self._cam_srv.start_realsense(
                    w, h, fps=8 if w > 1280 else 15,
                    fallbacks=((w, h, 8),),
                    mock_fallback=False)
            else:
                self._cam_srv.start_usb(w, h)
        except Exception as e:
            log.warning(f"相机开启失败: {e}")
            self.set_toast(f"相机开启失败: {e}")
            return
        self.pipeline_running = True
        self.color_sensor = self._cam_srv.color_sensor
        self.actual_stream_desc = self._cam_srv.stream_desc
        cam_desc = dict(self.camera_options).get(self.camera_type, self.camera_type)
        self.set_toast(f"相机已开启: {cam_desc} @ {self.resolution}")
        log.info(f"[OK] 相机已开启: {self.camera_type} @ {self.resolution} ({self.actual_stream_desc})")

    def _stop_camera(self):
        """幂等关闭取流"""
        self._cam_srv.stop()
        self.pipeline_running = False
        self.color_sensor = None

    def get_frame(self, frame_idx: int):
        """获取当前视频帧 (BGR): 仅取流运行时读取 (Mock 帧由服务内部生成); 未开启返回 None"""
        if self.pipeline_running:
            return self._cam_srv.read_frame(timeout_ms=2500)
        return None

    def save_image(self, raw_frame: np.ndarray) -> str:
        """
        保存采图快照：无标注高清原始帧存至活动场景图像目录 view_XXXX.png
        (Tag 观测解算统一由离线建图管线 tag_map_builder 完成, 先采后验)
        """
        self.image_count += 1
        raw_filename = f"view_{self.image_count:04d}.png"
        raw_filepath = os.path.join(self.output_dir, raw_filename)
        cv2.imwrite(raw_filepath, raw_frame)
        log.info(f"[CAPTURE] 快照 #{self.image_count} 拍摄成功: {raw_filepath}")

        # 同步更新归档场景元数据
        try:
            if self.current_scene:
                self.current_scene.refresh_stats()
                self.current_scene.save_meta()
        except Exception as e:
            log.warning(f"场景元数据刷新失败 (非致命): {e}")

        self.flash_timer = time.time()
        return raw_filepath

    def adjust_hardware_exposure(self, delta_us: float):
        """微调 RealSense 物理感光曝光时间 (微秒，步进 2000us = 2ms)"""
        if self.color_sensor is None:
            self.set_toast("当前未检测到 RealSense 物理彩色传感器")
            return
        try:
            # 若处于自动曝光，先切为手动曝光
            if self.color_sensor.supports(rs.option.enable_auto_exposure):
                is_auto = self.color_sensor.get_option(rs.option.enable_auto_exposure)
                if is_auto > 0.5:
                    self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            
            if self.color_sensor.supports(rs.option.exposure):
                cur_exp = self.color_sensor.get_option(rs.option.exposure)
                # D435 彩色相机 exposure 单位为 100微秒或毫秒，安全范围通常在 10 ~ 1000
                new_exp = max(10.0, min(1000.0, cur_exp + delta_us))
                self.color_sensor.set_option(rs.option.exposure, new_exp)
                self.set_toast(f"硬件手动曝光: {int(new_exp)} (按 [ 压暗 / ] 提亮)")
        except Exception as e:
            self.set_toast(f"调曝光失败: {e}")

    def toggle_auto_exposure(self):
        """一键切换 RealSense 自动曝光与手动曝光"""
        if self.color_sensor is None:
            self.set_toast("当前非物理相机")
            return
        try:
            if self.color_sensor.supports(rs.option.enable_auto_exposure):
                cur = self.color_sensor.get_option(rs.option.enable_auto_exposure)
                new_state = 0 if cur > 0.5 else 1
                self.color_sensor.set_option(rs.option.enable_auto_exposure, new_state)
                desc = "已开启【自动曝光 Auto】" if new_state == 1 else "已关闭【手动曝光模式】"
                self.set_toast(desc)
        except Exception as e:
            self.set_toast(f"切换自动曝光失败: {e}")

    # ------------------------------ 鼠标交互 ------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        # 画布按窗口尺寸真矢量重绘, imshow 1:1 呈现, 窗口坐标即画布坐标 (零偏移)
        if event == cv2.EVENT_MOUSEMOVE:
            self.renderer.on_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            hit = self.renderer.hit_test(x, y)
            if hit is not None:
                self._handle_action(*hit)
                return
            # 点击空白处收起下拉
            if self.active_dropdown is not None:
                self.active_dropdown = None

    def _handle_action(self, btn_id, payload):
        """工具栏按钮动作分发 (与 Robot 在线跟踪同名同义)"""
        if btn_id == "TOGGLE_SCENE_DD":
            self.active_dropdown = None if self.active_dropdown == "SCENE_DROPDOWN" \
                else "SCENE_DROPDOWN"
        elif btn_id.startswith("DD_SCENE_"):
            self.active_dropdown = None
            self.switch_scene(payload)
        elif btn_id == "TOGGLE_CAM_DD":
            self.active_dropdown = None if self.active_dropdown == "CAMERA_TYPE_DROPDOWN" \
                else "CAMERA_TYPE_DROPDOWN"
        elif btn_id == "TOGGLE_RES_DD":
            self.active_dropdown = None if self.active_dropdown == "RES_DROPDOWN" \
                else "RES_DROPDOWN"
        elif btn_id.startswith("DD_CAM_"):
            self.active_dropdown = None
            self._select_camera_type(payload)
        elif btn_id.startswith("DD_RES_"):
            self.active_dropdown = None
            self._change_resolution(payload)
        elif btn_id == "TOGGLE_CAMERA":
            self.active_dropdown = None
            self._toggle_camera()
        elif btn_id == "QUIT":
            self.is_running = False

    def run(self):
        """运行采图主循环 (GUI 先行: 启动只加载界面, 相机等用户点击 [开启])"""
        self.is_running = True
        win_key = "capture_wizard"  # 窗口 key 纯 ASCII (namedWindow ANSI API)
        self.win_mgr.setup_window(win_key, mouse_callback=self._on_mouse)
        self.win_mgr.set_unicode_title("采图向导 | flux_vision_3d")

        print("\n" + "=" * 70)
        print("          多视角采图向导启动 (GUI 已启动, 相机未开启)")
        print("=" * 70)
        print(" 单一职责: 纯预览 + 保存 (Tag 识别/解算由离线建图管线完成, 先采后验)")
        print(" 顶部工具栏 (与 Robot 在线跟踪同款): [相机类型 ▼] [分辨率 ▼] [开启/关闭] ... [退出 X]")
        print(f" [存储目录]   : {self.output_dir}")
        print(f" [已存图像]   : {self.image_count} 张")
        print(" [快捷键指南] (开启相机后生效):")
        print("   - [Space] (空格键) : 拍摄并保存当前视角高清原图；")
        print("   - [ [ / ] ]        : RealSense 硬件手动曝光 (压暗 / 提亮)；")
        print("   - [E]              : 切换自动曝光 / 手动曝光；")
        print("   - [X] / [Q] / [ESC] : 退出向导。")
        print(" 窗口: 拖拽边框自由缩放 (自动记忆) | Ctrl+滚轮/Ctrl+加减 矢量缩放 | Ctrl+0 复位")
        print("=" * 70 + "\n")

        frame_idx = 0
        frames_shown = 0

        try:
            while self.is_running:
                raw_frame = None
                if self.pipeline_running:
                    raw_frame = self.get_frame(frame_idx)
                    frame_idx += 1

                if raw_frame is not None:
                    # 纯预览: 无任何检测叠加; 仅快门白闪反馈
                    if time.time() - self.flash_timer < 0.12:
                        disp = cv2.addWeighted(raw_frame, 0.4, np.full_like(raw_frame, 255), 0.6, 0)
                    else:
                        disp = raw_frame
                    canvas = self.renderer.compose_canvas(disp)
                elif self.pipeline_running:
                    # 取流已启动但帧未就绪
                    canvas = self.renderer.make_canvas()
                    cw, ch = self.win_mgr.canvas_w, self.win_mgr.canvas_h
                    draw_text(canvas, "取流中...", (cw // 2 - 60, ch // 2), 22, COL_YELLOW, True)
                else:
                    # 相机未开启: 窗口尺寸占位画布 (开启前后工具栏位置严格一致)
                    canvas = self.renderer.make_canvas()
                    cw, ch = self.win_mgr.canvas_w, self.win_mgr.canvas_h
                    draw_text(canvas, "相机未开启",
                              (cw // 2 - 120, ch // 2 - 50), 32, COLOR_ACCENT, True)
                    draw_text(canvas, "请先选择相机类型和分辨率，然后点击 [开启] 按钮",
                              (cw // 2 - 250, ch // 2 + 10), 18, COLOR_TEXT_SUB)

                self.renderer.draw_toolbar(canvas)
                self.renderer.draw_toast(canvas)
                cv2.imshow(win_key, canvas)
                frames_shown += 1
                if frames_shown <= 3 and force_window_focus:
                    force_window_focus(win_key)

                key = cv2.waitKeyEx(30)
                poll = self.win_mgr.poll_events(key)
                if poll.should_quit:
                    log.info(f"\n采图向导结束。当前数据集共计 {self.image_count} 帧。")
                    break
                if key == -1:
                    continue
                k = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""

                if k in ("q", "x"):  # Q / X 退出 (ESC 由 poll_events 处理)
                    log.info(f"\n采图向导结束。当前数据集共计 {self.image_count} 帧。")
                    break
                elif k == " ":  # Space 拍摄保存 (需相机已开启)
                    if raw_frame is not None:
                        self.save_image(raw_frame)
                elif k == "[":  # 压暗曝光
                    self.adjust_hardware_exposure(-50.0)
                elif k == "]":  # 提亮曝光
                    self.adjust_hardware_exposure(50.0)
                elif k == "e":  # 切换自动曝光
                    self.toggle_auto_exposure()

        finally:
            self._stop_camera()
            cv2.destroyAllWindows()
            log.info(f"[OK] 采图向导已退出 (当前数据集共计 {self.image_count} 帧)")


def main():
    parser = argparse.ArgumentParser(description="多视角采图向导 (纯预览 + 保存)")
    parser.add_argument("--scene", type=str, default="", help="指定初始归档场景 ID")
    parser.add_argument("--dir", "--output-dir", "--output_dir", dest="dir", type=str, default=None, help="保存采集图像的目录路径")
    args = parser.parse_args()

    wizard = CaptureWizard(output_dir=args.dir, scene_id=args.scene)
    wizard.run()


if __name__ == "__main__":
    main()
