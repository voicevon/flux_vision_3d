#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) - 主应用控制器
=====================================================
集成工位感知、流水线算法插拔、视口交互器与主事件循环。
"""

import os
import sys
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.gui_window_manager import GuiWindowManager
from src.utils.logger import get_logger
from src.calibration.workspace_manager import WorkspaceManager
from src.vision.asparagus_analyzer import AsparagusAnalyzer
from src.vision.pipelines import PipelineRegistry, BaseAsparagusPipeline, PipelineResult

from tools.spatial_mapping_studio.mapping_viewport_interactor import MappingViewportInteractor
from tools.asparagus_pose_studio.data_io import (
    APP_ID, BASE_H, BASE_W, DEFAULT_DIR, REPORT_DIR, WINDOW_KEY,
    load_system_config, scan_samples, export_gcode_file,
    load_studio_settings, save_studio_settings
)
from tools.asparagus_pose_studio.renderer import AsparagusPoseStudioRenderer

log = get_logger(__name__)


class AsparagusPoseStudioApp:
    """芦笋位姿工作室 GUI 主应用"""

    def __init__(self, sample_dir: Optional[str] = None, settings_file: Optional[str] = None):
        self.settings_file = settings_file
        self.sys_cfg = load_system_config()
        self.win_mgr = GuiWindowManager(
            app_id=APP_ID, base_w=BASE_W, base_h=BASE_H, min_w=900, min_h=600,
            settings_file=settings_file
        )

        # 工位管理器感知
        self.workspace_mgr = WorkspaceManager()
        cur_ws = self.workspace_mgr.get_current_workspace()
        self.current_workspace_id = cur_ws.workspace_id if cur_ws else ""

        # 样本目录：优先取当前工位 production 采图；若未指定且无工位，回退 DEFAULT_DIR
        if sample_dir:
            self.sample_dir = sample_dir
        else:
            self.sample_dir = cur_ws.prod_raw_images_dir if cur_ws else DEFAULT_DIR

        self.active_dropdown = None
        self._dd_items = []
        self._workspace_rect = None
        self._pipeline_rect = None

        # 标定链: AprilTag 建图定位器 + 手工外参回退
        self.tag_localizer = None
        self._init_localizer()

        # 状态持久化加载 (算法路线、上次选中的样本)
        self._persisted_state = load_studio_settings(self.settings_file)
        persisted_pipe = self._persisted_state.get("pipeline_key")
        valid_pipes = dict(PipelineRegistry.list_options())
        if persisted_pipe and persisted_pipe in valid_pipes:
            self.pipeline_key = persisted_pipe
        else:
            self.pipeline_key = "ridge_tracing"

        self._persisted_sample_name = self._persisted_state.get("selected_sample_name", "")

        # 样本与解算状态
        self.samples = []
        self.sel_idx = -1
        self.scroll_off = 0
        self.targets = []
        self.sel_target = 0
        self.vis_img = None
        self.mode = "3d"             # "3d" | "2d"
        self.error = ""
        self.gcode_text = ""

        # 多技术路线算法流水线
        self.pipeline: Optional[BaseAsparagusPipeline] = None
        self.pipeline_result: Optional[PipelineResult] = None
        self.active_step_key = "stage3_poses"
        self._init_pipeline()

        # 视口平移与无级缩放交互控制器 (自适应双排工具栏顶部高度 86)
        self.viewport = MappingViewportInteractor(top_bar_h=86, bottom_bar_h=46, win_w=BASE_W, win_h=BASE_H)

        # 交互与事件映射
        self.mouse_pos = (-1, -1)
        self._buttons = []
        self._sample_rows = []
        self._result_rows = []
        self._toast_msg = None
        self._toast_until = 0.0
        self._running = True

        self.rescan(auto_load=True)

    def _save_persisted_state(self):
        """持久化保存当前的算法路线与选中的样本名"""
        sel_name = ""
        if 0 <= self.sel_idx < len(self.samples):
            sel_name = self.samples[self.sel_idx]["name"]
        elif self._persisted_sample_name:
            sel_name = self._persisted_sample_name

        save_studio_settings({
            "pipeline_key": self.pipeline_key,
            "selected_sample_name": sel_name
        }, settings_file=self.settings_file)

    # ------------------------------ 数据与标定 ------------------------------
    def _init_localizer(self):
        """装载当前工位的 AprilTag 空间标靶地图"""
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        tags_path = ws.map_path if ws else ""
        if tags_path and os.path.exists(tags_path) and os.path.getsize(tags_path) > 50:
            try:
                from src.vision.tag_localizer import TagLocalizer
                self.tag_localizer = TagLocalizer(tags_map_path=tags_path)
            except Exception as exc:
                log.warning("AprilTag 定位器加载失败: %s", exc)
                self.tag_localizer = None
        else:
            self.tag_localizer = None

    @property
    def workspace_options(self) -> List[Tuple[str, str]]:
        """各工位立体地图下拉选项"""
        opts = []
        for s in self.workspace_mgr.list_workspaces():
            status = f"RMSE: {s.global_rmse_px:.2f}px" if s.ba_solved else "未平差"
            opts.append((s.workspace_id, f"{s.name} ({s.image_count}帧, {status})"))
        return opts

    @property
    def current_workspace_name(self) -> str:
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        return ws.name if ws else "默认工位"

    @property
    def pipeline_options(self) -> List[Tuple[str, str]]:
        """算法流水线选项"""
        return PipelineRegistry.list_options()

    @property
    def current_pipeline_name(self) -> str:
        return dict(self.pipeline_options).get(self.pipeline_key, self.pipeline_key)

    def switch_workspace(self, workspace_key: str):
        """动态切换工位并同步更新样本源与标靶地图"""
        self.current_workspace_id = workspace_key
        self.workspace_mgr.set_active_workspace(workspace_key)

        ws = self.workspace_mgr.get_workspace_by_id(workspace_key)
        self.sample_dir = ws.prod_raw_images_dir if ws else ""

        self._init_localizer()
        self.rescan(auto_load=True)

        if self.tag_localizer:
            tag_cnt = len(getattr(self.tag_localizer, "tag_poses", {}))
            self.set_toast(f"已装载【{self.current_workspace_name}】地图 ({tag_cnt}个标靶)")
        else:
            self.set_toast(f"【{self.current_workspace_name}】未平差 tags_map.yaml，降级估算！")

    def rescan(self, auto_load: bool = False):
        """重新扫描样本目录 (选中样本自动立即触发识别定位)"""
        self.samples = scan_samples(self.sample_dir)
        self.sel_idx = -1
        self.targets, self.vis_img, self.gcode_text = [], None, ""
        self.sel_target, self.error = 0, ""
        if auto_load and self.samples:
            target_idx = 0
            if self._persisted_sample_name:
                for i, smp in enumerate(self.samples):
                    if smp["name"] == self._persisted_sample_name:
                        target_idx = i
                        break
            self._select_sample(target_idx, analyze_now=True)

    def _keep_selection_visible(self, idx: int):
        """键盘切换样本时保持选中项在列表中可见"""
        m = AsparagusPoseStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        _, _, _, bottom = AsparagusPoseStudioRenderer.compute_panels(
            self.win_mgr.canvas_w, self.win_mgr.canvas_h, m
        )[0]
        content_h = bottom - m["header_h"] - int(30 * m["s"])
        visible = max(1, content_h // (m["row_h"] + int(4 * m["s"])))
        if idx < self.scroll_off:
            self.scroll_off = idx
        elif idx >= self.scroll_off + visible:
            self.scroll_off = idx - visible + 1

    def _get_scaled_intrinsics(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """根据实际图像分辨率按比例调整相机内参"""
        intr = self.sys_cfg["intrinsics"]
        if intr:
            fx, fy, cx, cy, cfg_w, cfg_h = intr
            if cfg_w > 0 and cfg_h > 0 and (cfg_w != img_w or cfg_h != img_h):
                fx, cx = fx * img_w / cfg_w, cx * img_w / cfg_w
                fy, cy = fy * img_h / cfg_h, cy * img_h / cfg_h
        else:
            fx, fy, cx, cy = 909.12, 907.46, 647.46, 377.51
        return fx, fy, cx, cy

    def _init_pipeline(self):
        """初始化选中的算法流水线"""
        fx, fy, cx, cy = self._get_scaled_intrinsics(1920, 1080)
        self.pipeline = PipelineRegistry.create(self.pipeline_key, fx=fx, fy=fy, cx=cx, cy=cy)
        if self.pipeline:
            steps = self.pipeline.get_steps()
            if steps:
                self.active_step_key = steps[-1].key

    def switch_pipeline(self, pipeline_key: str):
        """切换算法路线并重新触发分析"""
        if pipeline_key == self.pipeline_key and self.pipeline is not None:
            return
        self.pipeline_key = pipeline_key
        self._init_pipeline()
        self._save_persisted_state()
        self.set_toast(f"已切换算法路线: 【{self.current_pipeline_name}】")
        if 0 <= self.sel_idx < len(self.samples):
            self.run_analyze()

    def _build_analyzer(self, img_w: int, img_h: int) -> AsparagusAnalyzer:
        """构建向下兼容的 AsparagusAnalyzer 实例"""
        fx, fy, cx, cy = self._get_scaled_intrinsics(img_w, img_h)
        analyzer = AsparagusAnalyzer(fx=fx, fy=fy, cx=cx, cy=cy)
        analyzer.set_tag_localizer(self.tag_localizer)
        analyzer.set_hand_eye_matrix(self.sys_cfg["t_cam_to_scara"])
        return analyzer

    def _select_sample(self, idx: int, analyze_now: bool = False):
        """选中样本照片并重置视口"""
        if not (0 <= idx < len(self.samples)):
            return
        self.sel_idx = idx
        self._keep_selection_visible(idx)
        self.targets, self.vis_img, self.gcode_text = [], None, ""
        self.sel_target, self.error = 0, ""
        self.viewport.reset()
        sample = self.samples[idx]

        self._persisted_sample_name = sample["name"]
        self._save_persisted_state()

        color = cv2.imread(sample["png"])
        if color is None:
            self.error, self.mode = "彩色图读取失败", "2d"
            return
        self.mode = "3d" if sample.get("depth") else "2d"
        self.vis_img = color.copy()

        if analyze_now:
            self.run_analyze()

    def run_analyze(self):
        """对当前样本执行完整的识别定位解算"""
        if not (0 <= self.sel_idx < len(self.samples)):
            self.set_toast("请先在左侧列表中选择一张样本照片")
            return

        sample = self.samples[self.sel_idx]
        color = cv2.imread(sample["png"])
        if color is None:
            self.error = "彩色图读取失败"
            self.set_toast(self.error, True)
            return

        depth = None
        if sample.get("depth"):
            try:
                depth = np.load(sample["depth"], allow_pickle=False)
                if depth.shape[:2] != color.shape[:2]:
                    depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            except Exception as exc:
                log.warning("深度加载失败 (%s): %s", sample["depth"], exc)
                depth = None
        self.mode = "3d" if depth is not None else "2d"

        try:
            fx, fy, cx, cy = self._get_scaled_intrinsics(color.shape[1], color.shape[0])
            if self.pipeline:
                self.pipeline.update_intrinsics(fx, fy, cx, cy)
            else:
                self._init_pipeline()

            dummy_analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            frame_transform, frame_calib_source = dummy_analyzer._resolve_calibration(color)
            plane_coeff = dummy_analyzer.fit_table_plane(depth) if depth is not None else None

            self.pipeline_result = self.pipeline.run(
                color_bgr=color,
                depth_mm=depth,
                plane_coeff=plane_coeff,
                frame_transform=frame_transform,
                frame_calib_source=frame_calib_source,
                nominal_z_mm=float(plane_coeff[2]) if (plane_coeff is not None and abs(plane_coeff[2]) > 300) else 640.0
            )
            self.targets = self.pipeline_result.targets
            self.sel_target = 0

            steps = self.pipeline.get_steps()
            step_keys = [s.key for s in steps]
            if self.active_step_key not in step_keys and step_keys:
                self.active_step_key = step_keys[-1]

            self._apply_active_step()
        except Exception as exc:
            log.exception("流水线执行异常")
            self.error = f"执行异常: {exc}"
            self.set_toast(self.error, True)
            return

        if self.targets:
            top = self.targets[self.sel_target]
            if self.mode == "3d":
                self.gcode_text = top.generate_gcode(
                    safe_z=self.sys_cfg["safe_z"],
                    drop_x=self.sys_cfg["drop_x"],
                    drop_y=self.sys_cfg["drop_y"]
                )
            self.set_toast(f"解算完成: 检出 {len(self.targets)} 个目标 ({self.pipeline_result.elapsed_ms:.0f}ms)", duration=2.2)
        else:
            self.set_toast(f"未检出符合规格目标 ({self.pipeline_result.elapsed_ms:.0f}ms)", duration=2.2)

    def _select_step(self, step_key: str):
        """切换算法流水线的中间步骤视图"""
        self.active_step_key = step_key
        if self.pipeline_result is None and (0 <= self.sel_idx < len(self.samples)):
            self.run_analyze()
            return
        self._apply_active_step()

    def _apply_active_step(self):
        """根据当前激活步骤更新视口显示的特征图"""
        if self.pipeline_result and self.active_step_key in self.pipeline_result.step_snapshots:
            img = self.pipeline_result.step_snapshots[self.active_step_key]
            if img is not None:
                self.vis_img = img
                step_obj = next((s for s in self.pipeline.get_steps() if s.key == self.active_step_key), None)
                if step_obj:
                    self.set_toast(f"视口显示: {step_obj.name} ({step_obj.description})", duration=1.8)
                return
        if 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                self.vis_img = color.copy()

    def _select_target(self, t_idx: int):
        """选择目标芦笋并联动 G-code"""
        if not (0 <= t_idx < len(self.targets)):
            return
        self.sel_target = t_idx
        if 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                analyzer = self._build_analyzer(color.shape[1], color.shape[0])
                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                if self.pipeline_result:
                    self.pipeline_result.step_snapshots[self.active_step_key] = self.vis_img

        t = self.targets[t_idx]
        if self.mode == "3d":
            self.gcode_text = t.generate_gcode(
                safe_z=self.sys_cfg["safe_z"],
                drop_x=self.sys_cfg["drop_x"],
                drop_y=self.sys_cfg["drop_y"],
            )

    def export_gcode(self):
        """导出当前选中目标的 G-code"""
        if not self.gcode_text:
            self.set_toast("当前无可导出的 G-code (2D 预览或未检出目标)")
            return
        ok, res = export_gcode_file(self.gcode_text, report_dir=REPORT_DIR)
        if ok:
            self.set_toast(f"G-code 已导出: {os.path.basename(res)}")
        else:
            self.set_toast(f"G-code 导出失败: {res}", True)

    # ------------------------------ 交互与事件 ------------------------------
    def set_toast(self, msg: str, sticky: bool = False, duration: float = 2.2):
        self._toast_msg = msg
        self._toast_until = time.time() + (3600 if sticky else duration)

    def hit_test(self, x: int, y: int) -> Optional[Tuple[str, Any]]:
        for rect, action in reversed(self._buttons):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return action
        return None

    def _on_button(self, label: str):
        if label.startswith("识别定位"):
            self.run_analyze()
        elif label.startswith("导出"):
            self.export_gcode()
        elif label.startswith("退出"):
            self._running = False

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        m = AsparagusPoseStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        list_p, img_p, _ = AsparagusPoseStudioRenderer.compute_panels(
            self.win_mgr.canvas_w, self.win_mgr.canvas_h, m
        )

        # 1. 鼠标滚轮事件
        if event == cv2.EVENT_MOUSEWHEEL:
            wheel_up = (flags > 0)
            if list_p[0] <= x <= list_p[2]:
                if wheel_up:
                    self.scroll_off = max(0, self.scroll_off - 2)
                else:
                    self.scroll_off += 2
                return
            elif img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
                vx, vy, vw, vh = img_p[0] + 2, img_p[1] + 2, img_p[2] - img_p[0] - 4, img_p[3] - img_p[1] - 4
                self.viewport.zoom_at(x, y, wheel_up, (vx, vy, vw, vh))
                return
            handled, toast = self.win_mgr.handle_mouse_wheel(event, flags)
            if handled and toast:
                self.set_toast(toast)
            return

        # 2. 拖拽平移事件 (右键或中键)
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            if img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
                self.viewport.start_pan(x, y)
                return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.viewport.update_pan(x, y):
                return
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            if self.viewport.is_panning:
                self.viewport.end_pan()
                return

        # 3. 双击复位事件
        if event in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDBLCLK):
            if img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
                self.viewport.reset()
                self.set_toast("视口已重置为适应窗口 (1.0x)")
                return

        # 4. 常规左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.active_dropdown and self._dd_items:
                for rect, key in self._dd_items:
                    if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                        dd_type = self.active_dropdown
                        self.active_dropdown = None
                        if dd_type == "WORKSPACE_DROPDOWN":
                            self.switch_workspace(key)
                        elif dd_type == "PIPELINE_DROPDOWN":
                            self.switch_pipeline(key)
                        return
                self.active_dropdown = None

            hit = self.hit_test(x, y)
            if hit:
                act_type, act_val = hit
                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                elif act_type == "set_step":
                    self._select_step(act_val)
                return

            for rect, idx in self._sample_rows:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    self._select_sample(idx, analyze_now=True)
                    return

            for rect, t_idx in self._result_rows:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    self._select_target(t_idx)
                    return

    def _handle_key(self, raw_key: int):
        handled, toast = self.win_mgr.handle_keyboard_fallback(raw_key)
        if toast:
            self.set_toast(toast)
        if handled:
            return
        key = chr(raw_key & 0xFF).lower() if (raw_key & 0xFF) < 128 else ""
        if key in ("x", "\x1b"):
            self._running = False
        elif key == " " or raw_key == 32 or raw_key in (10, 13):
            self.run_analyze()
        elif key == "e":
            self.export_gcode()
        elif raw_key in (2490368, 65362, 38):      # 上方向键
            if self.sel_idx > 0:
                self._select_sample(self.sel_idx - 1, analyze_now=True)
        elif raw_key in (2621440, 65364, 40):      # 下方向键
            if self.sel_idx < len(self.samples) - 1:
                self._select_sample(self.sel_idx + 1, analyze_now=True)

    # ------------------------------ 渲染与主循环 ------------------------------
    def render(self) -> np.ndarray:
        """重绘整个画布并更新交互命中区域"""
        W, H = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        canvas = np.full((H, W, 3), (16, 18, 22), dtype=np.uint8)
        self._buttons, self._sample_rows, self._result_rows, self._dd_items = (
            AsparagusPoseStudioRenderer.render_scene(canvas, self)
        )
        return canvas

    def run(self):
        """GUI 主循环"""
        self.win_mgr.setup_window(WINDOW_KEY, self._on_mouse)
        self.win_mgr.set_unicode_title("芦笋位姿工作室 - Asparagus Pose Studio")
        log.info("芦笋位姿工作室 GUI 已启动: %s (%d 个样本)", self.sample_dir, len(self.samples))

        while self._running:
            key = cv2.waitKey(30)
            poll = self.win_mgr.poll_events(key & 0xFFFF if key > 0 else -1)
            if poll.should_quit:
                break
            if key > 0:
                self._handle_key(key)
            if poll.toast_msg:
                self.set_toast(poll.toast_msg)

            cv2.imshow(WINDOW_KEY, self.render())

        try:
            cv2.destroyWindow(WINDOW_KEY)
        except Exception:
            pass
        log.info("芦笋位姿工作室 GUI 已安全退出")


def main():
    parser = argparse.ArgumentParser(description="芦笋位姿工作室 GUI (Asparagus Pose Studio)")
    parser.add_argument("--dir", type=str, default=None,
                        help="样本目录 (默认当前工位 production/raw_images/, 彩色 png + 可选对齐深度 npy)")
    args = parser.parse_args()
    app = AsparagusPoseStudioApp(sample_dir=args.dir)
    app.run()


if __name__ == "__main__":
    main()
