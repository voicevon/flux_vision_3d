#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋抓取位姿离线验证 GUI (Asparagus Offline) — Dashboard 6 号卡片
=================================================================
标定流水线收尾验证工具 (完全离线, 不依赖相机实时取流):
  ① 数据源为文件照片: 默认扫描 data/snapshots/ (d435_viewer 抓拍), --dir 可指定目录;
  ② 彩色 png + 对齐深度 npy 成对 → 完整 3D 链路: 平面拟合/实例切分/顶层判决/SCARA 位姿/G-code 预览;
  ③ 纯照片 (无深度) → 2D 检测预览降级: 轴线倾角与标称距离估算尺寸, 不输出抓取 G-code;
  ④ 批量解算: 一键跑目录全样本, 输出 reports/asparagus_batch_report_*.md 汇总报表。
窗口偏好 (缩放/尺寸) 经 GuiWindowManager 归档 config/gui_settings.json。
"""

import os
import sys
import glob
import time
import argparse

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.utils.gui_theme import GuiTheme
from src.utils.gui_components import draw_dropdown_button, render_dropdown_popup
from src.utils.gui_window_manager import GuiWindowManager
from src.utils.text_rendering import draw_text, measure_text
from src.utils.logger import get_logger
from src.vision.asparagus_analyzer import AsparagusAnalyzer
from src.calibration.workspace_manager import WorkspaceManager
from tools.spatial_mapping_studio.mapping_viewport_interactor import MappingViewportInteractor

WINDOW_KEY = "AsparagusOffline"   # cv2 窗口内部 key (纯 ASCII, 中文标题经 SetWindowTextW 注入)
APP_ID = "asparagus_offline"
BASE_W, BASE_H = 1280, 800        # 基准逻辑画布 (真矢量模式按窗口物理尺寸重绘)
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "snapshots")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

log = get_logger(__name__)


# ============================== 数据与配置 ==============================

def load_system_config():
    """读取 config.yaml: 内参/标定矩阵/机械臂参数 (与生产链路同源)"""
    cfg = {
        "intrinsics": None,          # (fx, fy, cx, cy, width, height)
        "t_cam_to_scara": None,
        "tags_map_path": "",
        "safe_z": 80.0,
        "drop_x": 220.0,
        "drop_y": 0.0,
    }
    if not os.path.exists(CONFIG_PATH):
        return cfg
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cam = raw.get("camera", {}).get("color", {})
        if all(k in cam for k in ("fx", "fy", "cx", "cy")):
            cfg["intrinsics"] = (float(cam["fx"]), float(cam["fy"]),
                                 float(cam["cx"]), float(cam["cy"]),
                                 int(cam.get("width", 0)), int(cam.get("height", 0)))
        calib = raw.get("calibration", {})
        if calib.get("t_cam_to_scara"):
            cfg["t_cam_to_scara"] = np.array(calib["t_cam_to_scara"], dtype=float)
        cfg["tags_map_path"] = calib.get("tags_map_path", "") or ""
        robot = raw.get("robot", {})
        cfg["safe_z"] = float(robot.get("safe_z_mm", 80.0))
        cfg["drop_x"] = float(robot.get("drop_x_mm", 220.0))
        cfg["drop_y"] = float(robot.get("drop_y_mm", 0.0))
    except Exception as exc:
        log.warning("config.yaml 加载异常 (使用缺省值): %s", exc)
    return cfg


def find_depth_pair(png_path: str):
    """彩色 png 配对原始深度 npy: 支持同茎同名与 d435_viewer 抓拍 (color_TS ↔ depth_raw_TS) 两种规则"""
    folder = os.path.dirname(png_path)
    stem = os.path.splitext(os.path.basename(png_path))[0]
    candidates = [stem + ".npy"]
    if stem.startswith("color_"):
        candidates.append("depth_raw_" + stem[len("color_"):] + ".npy")
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def scan_samples(sample_dir: str):
    """扫描样本目录: 返回 [{png, depth, name}] (按修改时间倒序, 最新在前)"""
    samples = []
    if not os.path.isdir(sample_dir):
        return samples
    for png in glob.glob(os.path.join(sample_dir, "*.png")) + glob.glob(os.path.join(sample_dir, "*.jpg")):
        base = os.path.basename(png)
        if base.startswith(("depth_vis_", "height_vis_")):
            continue  # 跳过可视化派生图, 只留原始输入
        samples.append({"png": png, "depth": find_depth_pair(png), "name": base})
    samples.sort(key=lambda s: os.path.getmtime(s["png"]), reverse=True)
    return samples


CALIB_LABELS = {
    "tag_online": "AprilTag 在线定位",
    "tag_cached": "AprilTag 缓存外参",
    "hand_eye": "手工 SVD 标定",
    "2d_preview": "2D 预览 (无深度)",
    "uncalibrated": "未标定 - 防撞保护",
}


# ============================== 主应用 ==============================

class AsparagusOfflineApp:
    """芦笋离线验证 GUI 主应用: 样本列表/标注大图/检测结果/G-code 预览/批量解算"""

    def __init__(self, sample_dir: str = None):
        self.sys_cfg = load_system_config()
        self.win_mgr = GuiWindowManager(app_id=APP_ID, base_w=BASE_W, base_h=BASE_H,
                                        min_w=900, min_h=600)

        # 工位管理器感知
        self.workspace_mgr = WorkspaceManager()
        cur_ws = self.workspace_mgr.get_current_workspace()
        self.current_workspace_id = cur_ws.workspace_id if cur_ws else ""

        # 默认样本目录: 严格取当前工位 production 采图；如果没有指定且无工位，才回退 DEFAULT_DIR
        if sample_dir:
            self.sample_dir = sample_dir
        else:
            if cur_ws:
                self.sample_dir = cur_ws.prod_raw_images_dir
            else:
                self.sample_dir = DEFAULT_DIR
        self.active_dropdown = None
        self._dd_items = []
        self._workspace_rect = None

        # 标定链: AprilTag 建图定位器 (一次装载) + 手工标定矩阵回退
        self.tag_localizer = None
        self._init_localizer()

        # 样本与结果状态
        self.samples = []
        self.sel_idx = -1            # 当前样本
        self.scroll_off = 0          # 样本列表滚动偏移
        self.targets = []
        self.sel_target = 0          # 结果列表选中目标 (前3位中选中的那一个)
        self.vis_img = None          # 标注可视化 (原始分辨率)
        self.mode = "3d"             # "3d" | "2d"
        self.error = ""              # 当前样本加载/解算错误
        self.gcode_text = ""

        # 视口显示图层切换 (0: 1.前景, 1: 距离场, 2: 峰脊线, 3: 2.骨架, 4: 3.位姿) - 纯显示切换，一次性全量解算
        self.active_view_mode = 4
        self.stage_vis = [None, None, None, None, None]

        # 视口交互控制器: 参照 Spatial Mapping Studio 实现滚轮放大缩小与平移
        self.viewport = MappingViewportInteractor(top_bar_h=52, bottom_bar_h=46, win_w=BASE_W, win_h=BASE_H)

        # 交互状态
        self.mouse_pos = (-1, -1)
        self._buttons = []           # [(rect, action)]
        self._sample_rows = []       # [(rect, sample_idx)]
        self._result_rows = []       # [(rect, target_idx)]
        self._toast_msg = None
        self._toast_until = 0.0
        self._running = True

        self.rescan(auto_load=True)

    # ------------------------------ 数据流程 ------------------------------
    def _init_localizer(self):
        """装载标靶立体地图"""
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
    def workspace_options(self):
        """动态列出各工位专属地图选项"""
        opts = []
        for s in self.workspace_mgr.list_workspaces():
            status = f"RMSE: {s.global_rmse_px:.2f}px" if s.ba_solved else "未平差"
            opts.append((s.workspace_id, f"{s.name} ({s.image_count}帧, {status})"))
        return opts

    @property
    def current_workspace_name(self):
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        return ws.name if ws else "默认工位"

    def switch_workspace(self, workspace_key: str):
        """动态切换标靶立体地图并同步刷新样本列表 (严格只检索目标工位 production/raw_images)"""
        self.current_workspace_id = workspace_key
        self.workspace_mgr.set_active_workspace(workspace_key)

        ws = self.workspace_mgr.get_workspace_by_id(workspace_key)
        if ws:
            # 仅检索该工位 production/raw_images，如果没有就是没有
            self.sample_dir = ws.prod_raw_images_dir
        else:
            self.sample_dir = ""

        self._init_localizer()

        # 强制重新扫描刷新列表，清空旧状态
        self.rescan(auto_load=True)

        if self.tag_localizer:
            tag_cnt = len(getattr(self.tag_localizer, "tag_poses", {}))
            self.set_toast(f"已装载【{self.current_workspace_name}】地图 (包含 {tag_cnt} 个标靶)")
        else:
            self.set_toast(f"【{self.current_workspace_name}】尚未平差生成 tags_map.yaml，降级估算！")

    def rescan(self, auto_load=False):
        """重新扫描样本目录, 可选自动载入首个样本原图"""
        self.samples = scan_samples(self.sample_dir)
        self.sel_idx = -1
        self.targets, self.vis_img, self.gcode_text = [], None, ""
        self.sel_target, self.error = 0, ""
        if auto_load and self.samples:
            self._select_sample(0, analyze_now=False)

    def _keep_selection_visible(self, idx: int):
        """键盘切换样本时, 滚动偏移跟随选中行保持可见"""
        m = self._metrics()
        _, _, _, bottom = self._panels(m)[0]
        content_h = bottom - self._metrics()["header_h"] - int(30 * m["s"])
        visible = max(1, content_h // (m["row_h"] + int(4 * m["s"])))
        if idx < self.scroll_off:
            self.scroll_off = idx
        elif idx >= self.scroll_off + visible:
            self.scroll_off = idx - visible + 1

    def _build_analyzer(self, img_w: int, img_h: int) -> AsparagusAnalyzer:
        """构建分析器: 内参取 config.yaml 并按快照实际分辨率等比缩放"""
        intr = self.sys_cfg["intrinsics"]
        if intr:
            fx, fy, cx, cy, cfg_w, cfg_h = intr
            if cfg_w > 0 and cfg_h > 0 and (cfg_w != img_w or cfg_h != img_h):
                fx, cx = fx * img_w / cfg_w, cx * img_w / cfg_w
                fy, cy = fy * img_h / cfg_h, cy * img_h / cfg_h
        else:
            fx, fy, cx, cy = 909.12, 907.46, 647.46, 377.51   # 640x480 缺省内参
        analyzer = AsparagusAnalyzer(fx=fx, fy=fy, cx=cx, cy=cy)
        analyzer.set_tag_localizer(self.tag_localizer)
        analyzer.set_hand_eye_matrix(self.sys_cfg["t_cam_to_scara"])
        return analyzer

    def _select_sample(self, idx: int, analyze_now: bool = False):
        """选中样本：载入原图显示；若 analyze_now=True 则立即触发识别定位"""
        if not (0 <= idx < len(self.samples)):
            return
        self.sel_idx = idx
        self._keep_selection_visible(idx)
        self.targets, self.vis_img, self.gcode_text = [], None, ""
        self.stage_vis = [None, None, None, None, None]
        self.sel_target, self.error = 0, ""
        self.viewport.reset()
        sample = self.samples[idx]

        color = cv2.imread(sample["png"])
        if color is None:
            self.error, self.mode = "彩色图读取失败", "2d"
            return
        self.mode = "3d" if sample["depth"] else "2d"
        self.vis_img = color.copy()

        if analyze_now:
            self.run_analyze()

    def run_analyze(self):
        """对当前选中的样本执行【识别定位】(一次性完成前景、骨架、位姿全部计算，结果数据全展示)"""
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
        if sample["depth"]:
            try:
                depth = np.load(sample["depth"], allow_pickle=False)
                if depth.shape[:2] != color.shape[:2]:   # 深度与彩色分辨率不一致 → 对齐彩色尺寸
                    depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            except Exception as exc:
                log.warning("深度加载失败 (%s): %s", sample["depth"], exc)
                depth = None
        self.mode = "3d" if depth is not None else "2d"

        try:
            analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            # 无论当前在看哪个视图，后台一次性全量解算完所有算法步骤！
            self.targets = analyzer.analyze(color, depth, stages=(True, True, True))
            # 缓存 5 步视觉过程：[1.前景, 距离场, 峰脊线, 2.骨架, 3.位姿]
            self.stage_vis = [
                analyzer.vis_stage1,
                getattr(analyzer, "vis_dist", None),
                getattr(analyzer, "vis_peaks", None),
                analyzer.vis_stage2,
                analyzer.vis_stage3
            ]
            self.sel_target = 0
            self._apply_view_mode()
        except Exception as exc:
            log.exception("样本解算异常")
            self.error = f"解算异常: {exc}"
            self.set_toast(self.error, True)
            return

        if self.targets:
            top = self.targets[self.sel_target]
            if self.mode == "3d":
                self.gcode_text = top.generate_gcode(safe_z=self.sys_cfg["safe_z"],
                                                     drop_x=self.sys_cfg["drop_x"],
                                                     drop_y=self.sys_cfg["drop_y"])
            self.set_toast(f"识别定位完成: 提取前 {len(self.targets)} 位优选目标", duration=2.2)
        else:
            self.set_toast("未检出符合规格的芦笋目标", duration=2.2)

    def _select_view_mode(self, mode_idx: int):
        """
        独立切换视口显示的算法图层 (0: 1.前景, 1: 距离场, 2: 峰脊线, 3: 2.骨架, 4: 3.位姿)
        纯粹切换视口显示内容，完全独立 (individual)；右侧所有识别结果和文本数据始终保持展示！
        """
        self.active_view_mode = mode_idx
        if self.stage_vis[mode_idx] is None and (0 <= self.sel_idx < len(self.samples)):
            self.run_analyze()
            return
        self._apply_view_mode()

    def _apply_view_mode(self):
        """刷新视口显示的图像内容"""
        labels = [
            "1.前景物料 (ExG + 传送带 ROI)",
            "CV算法: 欧氏距离变换场 (Distance Transform 半径能量)",
            "CV算法: 垂向极大值峰脊线 (Transverse NMS Ridge Peaks)",
            "2.中轴骨架与单体验证 (Spine & Ridge Tracing)",
            "3.位姿定位与顶层锁定 (Top 3 抓取目标)"
        ]
        idx = max(0, min(self.active_view_mode, len(labels) - 1))
        if self.stage_vis[idx] is not None:
            self.vis_img = self.stage_vis[idx]
            self.set_toast(f"视口显示: {labels[idx]}", duration=1.5)
        elif 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                self.vis_img = color.copy()

    def _select_target(self, t_idx: int):
        """在结果列表中切换选中的芦笋目标 (同步画布高亮与 G-code)"""
        if not (0 <= t_idx < len(self.targets)):
            return
        self.sel_target = t_idx
        if 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                analyzer = self._build_analyzer(color.shape[1], color.shape[0])
                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                self.stage_vis[4] = self.vis_img
                self.active_view_mode = 4

        t = self.targets[t_idx]
        if self.mode == "3d":
            self.gcode_text = t.generate_gcode(
                safe_z=self.sys_cfg["safe_z"],
                drop_x=self.sys_cfg["drop_x"],
                drop_y=self.sys_cfg["drop_y"],
            )

    def export_gcode(self):
        """导出当前选中目标的 G-code 到 reports/ 目录"""
        if not self.gcode_text:
            self.set_toast("当前无可导出的 G-code (2D 预览或未检出目标)")
            return
        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            path = os.path.join(REPORT_DIR, f"asparagus_gcode_{time.strftime('%Y%m%d_%H%M%S')}.gcode")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.gcode_text + "\n")
            self.set_toast(f"G-code 已导出: {os.path.basename(path)}")
        except Exception as exc:
            log.error("G-code 导出失败: %s", exc)
            self.set_toast("G-code 导出失败, 详见日志", True)

    # ------------------------------ 布局与渲染 ------------------------------
    def _metrics(self):
        s = self.win_mgr.canvas_w / BASE_W
        return {
            "s": s,
            "L": int(12 * s),            # 全局左边距
            "list_w": int(140 * s),      # 左侧样本列表宽 (按用户要求缩减至一半)
            "right_w": int(236 * s),     # 右侧结果面板宽 (按用户要求缩减至 2/3)
            "header_h": int(52 * s),
            "bottom_h": int(46 * s),
            "row_h": int(32 * s),        # 样本行高 (更紧凑一屏浏览更多)
            "btn_h": int(30 * s),
            "fs_title": max(14, int(20 * s)),
            "fs_sub": max(10, int(12 * s)),
            "fs_body": max(11, int(13 * s)),
            "fs_small": max(9, int(11 * s)),
            "fs_gcode": max(9, int(11 * s)),
        }

    def _panels(self, m):
        """三栏面板矩形: (样本列表, 图像区, 右侧结果)"""
        W, H = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        top = m["header_h"]
        bottom = H - m["bottom_h"]
        x0 = m["L"]
        x1 = x0 + m["list_w"]
        x4 = W - m["L"]
        x3 = x4 - m["right_w"]
        return (x0, top, x1, bottom), (x1 + int(10 * m["s"]), top, x3 - int(10 * m["s"]), bottom), (x3, top, x4, bottom)

    def _draw_button(self, canvas, rect, label, enabled=True, active=False):
        """标准按钮: 悬停提亮 + 冷青描边 + 加粗 (GuiTheme.BTN_BEHAVIOR 单源)"""
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        hover = enabled and x1 <= mx <= x2 and y1 <= my <= y2
        if not enabled:
            bg, border, col = GuiTheme.BTN_DISABLED_BG, GuiTheme.BTN_DISABLED_BORDER, GuiTheme.TEXT_DISABLED
        elif active:
            bg, border, col = GuiTheme.CARD_SEL, GuiTheme.BORDER_SEL, GuiTheme.WHITE
        elif hover:
            if "退出" in label:
                bg, border, col = (45, 38, 75), (80, 80, 220), (230, 230, 255)
            else:
                bg, border, col = GuiTheme.BTN_HOVER, GuiTheme.BORDER_HOVER, GuiTheme.BTN_TEXT_HOVER
        else:
            if "退出" in label:
                bg, border, col = GuiTheme.BTN, (60, 60, 110), GuiTheme.BTN_TEXT
            else:
                bg, border, col = GuiTheme.BTN, GuiTheme.BTN_BORDER, GuiTheme.BTN_TEXT
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if hover else 1)
        m = self._metrics()
        (tw, th), _ = measure_text(label, font_size=m["fs_sub"])
        draw_text(canvas, label, (x1 + ((x2 - x1) - tw) // 2, y1 + ((y2 - y1) - th) // 2),
                  m["fs_sub"], col, bold=(hover and GuiTheme.BTN_BEHAVIOR["HOVER_BOLD"]) or active)
        if enabled:
            self._buttons.append((rect, ("btn", label)))

    def _draw_view_pill(self, canvas, rect, label, is_active: bool):
        """舒适高质感分段视图切换药丸 (纯显示层切换，互斥单选，视觉反馈鲜明)"""
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        hover = x1 <= mx <= x2 and y1 <= my <= y2
        m = self._metrics()

        if is_active:
            bg = (32, 68, 48)            # 沉稳翡翠绿底色
            border = (0, 235, 120)        # 鲜亮高光绿边框
            text_col = (255, 255, 255)    # 纯白加粗文字
        elif hover:
            bg = (34, 38, 46)
            border = (110, 130, 155)
            text_col = (235, 240, 245)
        else:
            bg = (24, 28, 34)
            border = (52, 58, 68)
            text_col = (165, 175, 185)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if is_active else (1 if hover else 1))

        # 激活项左侧绘制一个发光小圆点
        (tw, th), _ = measure_text(label, font_size=m["fs_sub"])
        if is_active:
            dot_x = x1 + int(10 * m["s"])
            dot_y = y1 + (y2 - y1) // 2
            cv2.circle(canvas, (dot_x, dot_y), int(3.5 * m["s"]), (0, 235, 120), -1)
            cv2.circle(canvas, (dot_x, dot_y), int(5 * m["s"]), (0, 235, 120), 1)
            tx = dot_x + int(8 * m["s"])
        else:
            tx = x1 + ((x2 - x1) - tw) // 2

        ty = y1 + ((y2 - y1) - th) // 2
        draw_text(canvas, label, (tx, ty), m["fs_sub"], text_col, bold=is_active)

    def _draw_dropdown_button(self, canvas, rect, label, is_open=False):
        """扁平化下拉框按钮 (统一调用 gui_components)"""
        m = self._metrics()
        draw_dropdown_button(canvas, rect, label, is_open, self.mouse_pos, font_size=m["fs_sub"])

    def _render_dropdown_popup(self, canvas, rect, options, active_key):
        """置顶悬浮下拉菜单浮层 (统一调用 gui_components)"""
        m = self._metrics()
        item_h = int(28 * m["s"])
        btns = render_dropdown_popup(
            canvas,
            anchor_rect=rect,
            options=options,
            active_key=active_key,
            btn_prefix="DD_MAP_",
            item_h=item_h,
            min_width=int(260 * m["s"]),
        )
        self._dd_items = [(item_rect, key) for _, item_rect, key in btns]

    def render(self):
        """真矢量渲染: 画布按窗口物理尺寸 1:1 重绘 (imshow 零缩放, 鼠标坐标零偏移)"""
        m = self._metrics()
        W, H = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        canvas = np.full((H, W, 3), GuiTheme.BG, dtype=np.uint8)
        self._buttons, self._sample_rows, self._result_rows = [], [], []

        # 标题栏: 左上角显示“识别芦笋位姿”，不显示冗余目录路径
        draw_text(canvas, "识别芦笋位姿", (m["L"], int(16 * m["s"])), m["fs_title"],
                  GuiTheme.TEXT, bold=True)

        list_p, img_p, right_p = self._panels(m)
        self._draw_sample_list(canvas, m, list_p)
        self._draw_image_area(canvas, m, img_p)
        self._draw_result_panel(canvas, m, right_p)

        # 标题栏右侧按钮组 (从右向左布局)
        btn_w = int(112 * m["s"])
        bx = W - m["L"]
        buttons = [
            ("退出 [X]", "exit", True),
            ("导出G-code [E]", "export", bool(self.gcode_text)),
            ("识别定位 [空格]", "analyze", bool(self.samples and self.sel_idx >= 0)),
        ]
        for label, _act, enabled in buttons:
            bx -= btn_w + int(8 * m["s"])
            self._draw_button(canvas, (bx, int(14 * m["s"]), bx + btn_w, int(14 * m["s"]) + m["btn_h"]),
                              label, enabled=enabled)

        # 视口算法流程全透明分段选择器：[ 1.前景 | 距离场 | 峰脊线 | 2.骨架 | 3.位姿 ]
        view_names = ["1.前景", "距离场", "峰脊线", "2.骨架", "3.位姿"]
        pill_w = int(60 * m["s"])
        for v_i in (4, 3, 2, 1, 0):
            bx -= pill_w + int(4 * m["s"])
            pill_rect = (bx, int(14 * m["s"]), bx + pill_w, int(14 * m["s"]) + m["btn_h"])
            self._draw_view_pill(canvas, pill_rect, view_names[v_i], is_active=(self.active_view_mode == v_i))
            self._buttons.append((pill_rect, ("set_view_mode", v_i)))

        # 分段选择器左侧提示标签：“显示:”
        bx -= int(38 * m["s"])
        draw_text(canvas, "显示:", (bx, int(22 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)

        # 最左侧：工位地图选择下拉按钮
        sc_w = int(175 * m["s"])
        bx -= sc_w + int(10 * m["s"])
        self._workspace_rect = (bx, int(14 * m["s"]), bx + sc_w, int(14 * m["s"]) + m["btn_h"])
        is_sc_open = (self.active_dropdown == "WORKSPACE_DROPDOWN")
        self._draw_dropdown_button(canvas, self._workspace_rect, f"地图: {self.current_workspace_name}", is_open=is_sc_open)
        self._buttons.append((self._workspace_rect, ("toggle_dd", "WORKSPACE_DROPDOWN")))

        # 底部状态栏
        yb = H - m["bottom_h"] + int(8 * m["s"])
        calib = CALIB_LABELS.get(
            self.targets[0].calibration_source if self.targets else "uncalibrated", "-")
        n3d = sum(1 for smp in self.samples if smp["depth"])
        status = f"工位【{self.current_workspace_name}】 · 生产样本 {len(self.samples)} 个 (3D成对 {n3d} / 2D {len(self.samples) - n3d}) · 标定状态: {calib}"
        draw_text(canvas, status, (m["L"], yb), m["fs_sub"], GuiTheme.TEXT_SUB)
        draw_text(canvas, "[↑↓] 样本  ·  [空格] 识别定位  ·  右键拖拽  ·  滚轮无级缩放  ·  双击复位",
                  (W - int(460 * m["s"]), yb), m["fs_sub"], GuiTheme.TEXT_MUTED)

        # Toast (底部居中)
        if self._toast_msg and time.time() < self._toast_until:
            (tw, th), _ = measure_text(self._toast_msg, font_size=m["fs_body"])
            tx, ty = (W - tw) // 2, H - int(76 * m["s"])
            pad = int(10 * m["s"])
            cv2.rectangle(canvas, (tx - pad, ty - int(6 * m["s"])), (tx + tw + pad, ty + th + int(8 * m["s"])),
                          (40, 34, 26), -1)
            cv2.rectangle(canvas, (tx - pad, ty - int(6 * m["s"])), (tx + tw + pad, ty + th + int(8 * m["s"])),
                          GuiTheme.WARN, 1)
            draw_text(canvas, self._toast_msg, (tx, ty), m["fs_body"], GuiTheme.WARN, bold=True)

        # 置顶渲染下拉弹出菜单 (覆盖在所有内容最上层)
        if self.active_dropdown == "WORKSPACE_DROPDOWN" and self._workspace_rect:
            self._render_dropdown_popup(canvas, self._workspace_rect, self.workspace_options, self.current_workspace_id)

        return canvas

    def _panel_bg(self, canvas, rect, title):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.CARD_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)
        m = self._metrics()
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + int(26 * m["s"])), (26, 30, 38), -1)
        draw_text(canvas, title, (x1 + int(8 * m["s"]), y1 + int(6 * m["s"])), m["fs_body"],
                  GuiTheme.ACCENT, bold=True)
        return y1 + int(30 * m["s"])   # 内容起始 y

    def _draw_sample_list(self, canvas, m, rect):
        y = self._panel_bg(canvas, rect, f"样本列表 ({len(self.samples)})")
        if not self.samples:
            draw_text(canvas, "当前工位无生产样本", (rect[0] + int(10 * m["s"]), y + int(10 * m["s"])),
                      m["fs_body"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "请在采集向导中拍摄生产样本,",
                      (rect[0] + int(10 * m["s"]), y + int(32 * m["s"])), m["fs_small"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "或通过上方地图下拉框切换工位",
                      (rect[0] + int(10 * m["s"]), y + int(50 * m["s"])), m["fs_small"], GuiTheme.TEXT_MUTED)
            return

        x1, _, x2, y2 = rect
        row_h = m["row_h"]
        visible = max(1, (y2 - y - int(6 * m["s"])) // (row_h + int(4 * m["s"])))
        self.scroll_off = max(0, min(self.scroll_off, len(self.samples) - visible))
        for row_i in range(visible):
            idx = self.scroll_off + row_i
            if idx >= len(self.samples):
                break
            smp = self.samples[idx]
            ry1 = y + row_i * (row_h + int(4 * m["s"]))
            ry2 = ry1 + row_h
            is_sel = (idx == self.sel_idx)
            if is_sel:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_SEL, -1)
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.BORDER_SEL, 1)
            elif x1 < self.mouse_pos[0] < x2 and ry1 <= self.mouse_pos[1] <= ry2:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_HOVER, -1)

            raw_name = smp["name"]
            short_name = raw_name.replace(".png", "").replace("view_", "")
            if len(short_name) > 10:
                short_name = short_name[-10:]
            
            draw_text(canvas, short_name, (x1 + int(8 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB, bold=is_sel)
            
            tag = "3D" if smp["depth"] else "2D"
            tag_col = GuiTheme.OK if smp["depth"] else GuiTheme.TEXT_MUTED
            draw_text(canvas, tag, (x2 - int(24 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], tag_col, bold=is_sel)
            self._sample_rows.append(((x1 + 2, ry1, x2 - 2, ry2), idx))

    def _draw_image_area(self, canvas, m, rect):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 12, 16), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)

        if self.vis_img is None:
            msg = self.error if self.error else ("请在左侧选择样本" if self.samples else "当前工位无样本")
            draw_text(canvas, msg, (x1 + int(16 * m["s"]), y1 + int(16 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if self.error else GuiTheme.TEXT_MUTED)
            return

        # 视口矩形与图像切片计算 (基于 MappingViewportInteractor)
        vx, vy, vw, vh = x1 + 2, y1 + 2, x2 - x1 - 4, y2 - y1 - 4
        ih, iw = self.vis_img.shape[:2]
        rois = self.viewport.compute_viewport_render_rois((vx, vy, vw, vh), iw, ih)

        if rois:
            (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = rois
            src_crop = self.vis_img[src_y1:src_y2, src_x1:src_x2]
            dw = dst_x2 - dst_x1
            dh = dst_y2 - dst_y1
            if dw > 0 and dh > 0 and src_crop.size > 0:
                interp = cv2.INTER_LINEAR if self.viewport.zoom_level > 1.0 else cv2.INTER_AREA
                disp = cv2.resize(src_crop, (dw, dh), interpolation=interp)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = disp

        # 视口底部状态提示
        if self.targets:
            mode_txt = f"{'3D 完整链路' if self.mode == '3d' else '2D 预览'} — 提取前 {len(self.targets)} 位目标 (已高亮目标 #{self.targets[self.sel_target].id})"
            mode_col = GuiTheme.OK if self.mode == "3d" else GuiTheme.WARN
        else:
            mode_txt = "原图已载入 — 点击上方【识别定位】(或按空格键) 开始位姿解算"
            mode_col = GuiTheme.GOLD
        draw_text(canvas, mode_txt, (x1 + int(8 * m["s"]), y2 - int(20 * m["s"])),
                  m["fs_small"], mode_col, bold=True)

        # 视口右上角缩放比例悬浮指示 (滚轮缩放时即时反馈)
        if abs(self.viewport.zoom_level - 1.0) > 0.01 or self.viewport.is_panning:
            zoom_badge = f"缩放: {self.viewport.zoom_level:.1f}x [右键拖拽/双击复位]"
            (zw, zh), _ = measure_text(zoom_badge, font_size=m["fs_small"])
            cv2.rectangle(canvas, (x2 - zw - int(16 * m["s"]), y1 + int(8 * m["s"])),
                          (x2 - int(6 * m["s"]), y1 + zh + int(14 * m["s"])), (20, 24, 30), -1)
            cv2.rectangle(canvas, (x2 - zw - int(16 * m["s"]), y1 + int(8 * m["s"])),
                          (x2 - int(6 * m["s"]), y1 + zh + int(14 * m["s"])), GuiTheme.BORDER, 1)
            draw_text(canvas, zoom_badge, (x2 - zw - int(11 * m["s"]), y1 + int(11 * m["s"])),
                      m["fs_small"], GuiTheme.ACCENT)

    def _draw_result_panel(self, canvas, m, rect):
        y = self._panel_bg(canvas, rect, f"识别定位结果 (前3位: {len(self.targets)})")
        x1, _, x2, y2 = rect
        if not self.targets:
            msg = self.error if self.error else ("点击【识别定位】开始分析" if (self.samples and self.sel_idx >= 0) else "未检出目标")
            draw_text(canvas, msg, (x1 + int(10 * m["s"]), y + int(14 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if self.error else GuiTheme.TEXT_MUTED)
            self._draw_gcode_box(canvas, m, rect)
            return

        # 结果列表: 遍历排名前 3 位目标，以卡片形式展示核心指标
        card_h = int(60 * m["s"])
        for list_idx, t in enumerate(self.targets):
            ry1 = y + list_idx * (card_h + int(8 * m["s"]))
            if ry1 + card_h > y2 - int(190 * m["s"]):
                break   # 预留 G-code 区
            ry2 = ry1 + card_h
            is_sel = (list_idx == self.sel_target)
            is_top = t.is_topmost

            # 卡片背景与高亮边框
            bg_col = (28, 38, 32) if is_sel else (20, 24, 30)
            border_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else (55, 65, 80))
            cv2.rectangle(canvas, (x1 + 4, ry1), (x2 - 4, ry2), bg_col, -1)
            cv2.rectangle(canvas, (x1 + 4, ry1), (x2 - 4, ry2), border_col, 2 if is_sel else 1)

            badge = "#1最优" if is_top else f"#{t.id}候选"
            badge_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else GuiTheme.TEXT_SUB)
            draw_text(canvas, f"{badge} D:{t.diam_mm} L:{int(t.length_mm)}mm",
                      (x1 + int(8 * m["s"]), ry1 + int(5 * m["s"])), m["fs_small"], badge_col, bold=True)

            h_str = f"+{t.rel_height_mm}mm" if t.rel_height_mm > 0 else (f"Z:{int(t.grip_z)}" if t.grip_z > 0 else "--")
            draw_text(canvas, f"方向:{t.yaw_deg}° 凸起:{h_str}",
                      (x1 + int(8 * m["s"]), ry1 + int(23 * m["s"])), m["fs_small"],
                      GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB)

            draw_text(canvas, f"S:({int(t.robot_x)},{int(t.robot_y)},{int(t.robot_z)}) R:{int(t.robot_r)}°",
                      (x1 + int(8 * m["s"]), ry1 + int(41 * m["s"])), m["fs_small"],
                      GuiTheme.ACCENT if is_sel else GuiTheme.TEXT_MUTED)

            self._result_rows.append(((x1 + 4, ry1, x2 - 4, ry2), list_idx))

        self._draw_gcode_box(canvas, m, rect)

    def _draw_gcode_box(self, canvas, m, rect):
        x1, _, x2, y2 = rect
        gh = int(185 * m["s"])
        gy1 = y2 - gh - int(6 * m["s"])
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      (16, 19, 24), -1)
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      GuiTheme.BORDER, 1)
        if self.gcode_text:
            title = f"G-code 预览 (目标 #{self.targets[self.sel_target].id if self.targets else 1}) [E] 导出"
            draw_text(canvas, title, (x1 + int(12 * m["s"]), gy1 + int(5 * m["s"])),
                      m["fs_small"], GuiTheme.GOLD, bold=True)
            yy = gy1 + int(24 * m["s"])
            line_h = int(13 * m["s"])
            for line in self.gcode_text.splitlines():
                if yy + line_h > y2 - int(8 * m["s"]):
                    draw_text(canvas, "... (完整内容见 [E] 导出文件)",
                              (x1 + int(12 * m["s"]), yy), m["fs_small"], GuiTheme.TEXT_MUTED)
                    break
                draw_text(canvas, line, (x1 + int(12 * m["s"]), yy), m["fs_gcode"], GuiTheme.TEXT_SUB)
                yy += line_h
        else:
            note = "2D 预览无深度, 不生成抓取 G-code" if self.mode == "2d" else "未检出可抓取目标"
            draw_text(canvas, note, (x1 + int(12 * m["s"]), gy1 + int(8 * m["s"])),
                      m["fs_small"], GuiTheme.TEXT_MUTED)

    # ------------------------------ 交互 ------------------------------
    def set_toast(self, msg, sticky=False, duration=2.2):
        self._toast_msg = msg
        self._toast_until = time.time() + (3600 if sticky else duration)

    def hit_test(self, x, y):
        for rect, action in reversed(self._buttons):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return action
        return None

    def _on_button(self, label):
        if label.startswith("识别定位"):
            self.run_analyze()
        elif label.startswith("导出"):
            self.export_gcode()
        elif label.startswith("退出"):
            self._running = False

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        m = self._metrics()
        list_p, img_p, _ = self._panels(m)

        # 1. 鼠标滚轮事件 (严格区分左栏滚动 vs 中间视口以鼠标为中心缩放)
        if event == cv2.EVENT_MOUSEWHEEL:
            wheel_up = (flags > 0)

            # A. 鼠标位于左栏样本列表：上下翻滚列表
            if list_p[0] <= x <= list_p[2]:
                if wheel_up:
                    self.scroll_off = max(0, self.scroll_off - 2)
                else:
                    self.scroll_off += 2
                return

            # B. 鼠标位于中间图像视口：以光标为中心自适应无级缩放
            elif img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
                vx, vy, vw, vh = img_p[0] + 2, img_p[1] + 2, img_p[2] - img_p[0] - 4, img_p[3] - img_p[1] - 4
                self.viewport.zoom_at(x, y, wheel_up, (vx, vy, vw, vh))
                return

            # C. 其他区域交给 WindowManager
            handled, toast = self.win_mgr.handle_mouse_wheel(event, flags)
            if handled and toast:
                self.set_toast(toast)
            return

        # 2. 拖拽平移事件 (支持鼠标右键或中键按住拖拽，参照 Mapping Studio)
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

        # 3. 双击事件 (视口内双击左键或右键一键重置缩放与平移)
        if event in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDBLCLK):
            if img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
                self.viewport.reset()
                self.set_toast("视口已重置为适应窗口 (1.0x)")
                return

        # 4. 常规左键点击事件
        if event == cv2.EVENT_LBUTTONDOWN:
            # 优先判定置顶下拉浮层点击
            if self.active_dropdown and self._dd_items:
                for rect, key in self._dd_items:
                    if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                        self.active_dropdown = None
                        self.switch_workspace(key)
                        return
                self.active_dropdown = None

            # 按钮点击
            hit = self.hit_test(x, y)
            if hit:
                act_type, act_val = hit
                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                elif act_type == "set_view_mode":
                    self._select_view_mode(act_val)
                return
            for rect, idx in self._sample_rows:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    self._select_sample(idx, analyze_now=False)
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
        elif raw_key in (2490368, 65362, 38):      # 上
            if self.sel_idx > 0:
                self._select_sample(self.sel_idx - 1, analyze_now=False)
        elif raw_key in (2621440, 65364, 40):      # 下
            if self.sel_idx < len(self.samples) - 1:
                self._select_sample(self.sel_idx + 1, analyze_now=False)

    # ------------------------------ 主循环 ------------------------------
    def run(self):
        self.win_mgr.setup_window(WINDOW_KEY, self._on_mouse)
        self.win_mgr.set_unicode_title("识别芦笋位姿 - Asparagus Offline")
        log.info("芦笋位姿识别 GUI 已启动: %s (%d 个样本)", self.sample_dir, len(self.samples))

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
        log.info("芦笋位姿识别 GUI 已退出")


def main():
    parser = argparse.ArgumentParser(description="识别芦笋位姿 GUI (文件照片输入)")
    parser.add_argument("--dir", type=str, default=None,
                        help="样本目录 (默认当前工位 production/raw_images/, 彩色 png + 可选对齐深度 npy)")
    args = parser.parse_args()
    app = AsparagusOfflineApp(sample_dir=args.dir)
    app.run()


if __name__ == "__main__":
    main()

