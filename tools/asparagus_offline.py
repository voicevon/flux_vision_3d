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
        self.sample_dir = sample_dir or DEFAULT_DIR
        self.sys_cfg = load_system_config()
        self.win_mgr = GuiWindowManager(app_id=APP_ID, base_w=BASE_W, base_h=BASE_H,
                                        min_w=900, min_h=600)

        # 场景管理器感知
        from src.calibration.scene_manager import CalibrationSceneManager
        self.scene_mgr = CalibrationSceneManager()
        self.current_scene_id = "__prod__"  # 默认使用全局生产地图
        self.active_dropdown = None
        self._dd_items = []
        self._scene_rect = None

        # 标定链: AprilTag 建图定位器 (一次装载) + 手工标定矩阵回退
        self.tag_localizer = None
        self._init_localizer()

        # 样本与结果状态
        self.samples = []
        self.sel_idx = -1            # 当前样本
        self.scroll_off = 0          # 样本列表滚动偏移
        self.targets = []
        self.sel_target = 0          # 结果列表选中目标 (查看 G-code)
        self.vis_img = None          # 标注可视化 (原始分辨率)
        self.mode = "3d"             # "3d" | "2d"
        self.error = ""              # 当前样本加载/解算错误
        self.gcode_text = ""

        # 批量状态 (每帧推进一个样本, 不阻塞 UI)
        self.batch_queue = []
        self.batch_results = []

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
        if self.current_scene_id == "__prod__":
            tags_path = self.sys_cfg.get("tags_map_path", "")
            if tags_path and not os.path.isabs(tags_path):
                tags_path = os.path.join(PROJECT_ROOT, tags_path)
        else:
            sc = self.scene_mgr.get_scene_by_id(self.current_scene_id)
            tags_path = sc.map_path if sc else ""

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
    def scene_options(self):
        """动态列出可选地图：首项为生产全局地图，后续为各标定场景地图"""
        opts = [("__prod__", "★ 当前生产地图 (config/tags_map.yaml)")]
        for s in self.scene_mgr.list_scenes():
            tag = "★ " if s.is_published else ""
            status = f"{s.global_rmse_px:.2f}px" if s.ba_solved else "未平差"
            opts.append((s.scene_id, f"{tag}{s.name} ({s.image_count}帧, {status})"))
        return opts

    @property
    def current_scene_name(self):
        if self.current_scene_id == "__prod__":
            return "生产地图"
        sc = self.scene_mgr.get_scene_by_id(self.current_scene_id)
        return sc.name if sc else "默认"

    def switch_scene(self, scene_key: str):
        """动态切换标靶立体地图并重新解算当前样本"""
        self.current_scene_id = scene_key
        self._init_localizer()
        if self.tag_localizer:
            tag_cnt = len(getattr(self.tag_localizer, "tag_poses", {}))
            self.set_toast(f"已装载【{self.current_scene_name}】地图 (包含 {tag_cnt} 个标靶)")
        else:
            self.set_toast(f"【{self.current_scene_name}】尚未平差生成 tags_map.yaml，降级估算！")
        
        # 立即重新解算当前样本
        if 0 <= self.sel_idx < len(self.samples):
            self._select_sample(self.sel_idx)

    def rescan(self, auto_load=False):
        """重新扫描样本目录, 可选自动载入最新样本"""
        self.samples = scan_samples(self.sample_dir)
        if auto_load and self.samples:
            self._select_sample(0)

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

    def _select_sample(self, idx: int):
        """载入样本并执行解算 (同步, 单帧解算为亚秒级)"""
        if not (0 <= idx < len(self.samples)):
            return
        self.sel_idx = idx
        self._keep_selection_visible(idx)
        self.targets, self.vis_img, self.gcode_text = [], None, ""
        self.sel_target, self.error = 0, ""
        sample = self.samples[idx]

        color = cv2.imread(sample["png"])
        if color is None:
            self.error, self.mode = "彩色图读取失败", "2d"
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
            self.targets = analyzer.analyze(color, depth)
            self.vis_img = analyzer.draw_detections(color, self.targets)
        except Exception as exc:
            log.exception("样本解算异常")
            self.error = f"解算异常: {exc}"
            return

        if self.mode == "3d" and self.targets:
            top = next((t for t in self.targets if t.is_topmost), self.targets[0])
            self.sel_target = self.targets.index(top)
            self.gcode_text = top.generate_gcode(safe_z=self.sys_cfg["safe_z"],
                                                 drop_x=self.sys_cfg["drop_x"],
                                                 drop_y=self.sys_cfg["drop_y"])

    def start_batch(self):
        """启动批量解算 (逐帧推进, 保持界面响应)"""
        if not self.samples:
            self.set_toast("样本目录为空, 无可批量解算")
            return
        self.batch_queue = list(range(len(self.samples)))
        self.batch_results = []
        self.set_toast(f"批量解算启动: 共 {len(self.batch_queue)} 个样本", duration=1.6)

    def _batch_step(self):
        """批量解算推进: 每次调用处理一个样本"""
        if not self.batch_queue:
            return
        idx = self.batch_queue.pop(0)
        sample = self.samples[idx]
        entry = {"name": sample["name"], "mode": "-", "count": 0,
                 "top": None, "error": ""}
        try:
            color = cv2.imread(sample["png"])
            depth = None
            if sample["depth"]:
                depth = np.load(sample["depth"], allow_pickle=False)
                if depth.shape[:2] != color.shape[:2]:
                    depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            targets = analyzer.analyze(color, depth)
            entry["mode"] = "3d" if depth is not None else "2d"
            entry["count"] = len(targets)
            if targets:
                top = next((t for t in targets if t.is_topmost), targets[0])
                entry["top"] = (top.length_mm, top.diam_mm, top.robot_r,
                                top.robot_x, top.robot_y, top.robot_z)
        except Exception as exc:
            entry["error"] = str(exc)
        self.batch_results.append(entry)

        if not self.batch_queue:   # 收尾: 写汇总报表
            report = self._write_batch_report()
            if report:
                self.set_toast(f"批量完成: 报告已保存 {os.path.basename(report)}", duration=4.0)

    def _write_batch_report(self):
        """批量结果落盘 Markdown 汇总报表"""
        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            path = os.path.join(REPORT_DIR, f"asparagus_batch_report_{time.strftime('%Y%m%d_%H%M%S')}.md")
            ok3d = sum(1 for r in self.batch_results if r["mode"] == "3d" and r["count"] > 0)
            ok2d = sum(1 for r in self.batch_results if r["mode"] == "2d" and r["count"] > 0)
            fails = sum(1 for r in self.batch_results if r["error"] or r["count"] == 0)
            lines = [
                "# 芦笋离线批量解算报告",
                "",
                f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"- 目录: {self.sample_dir}",
                f"- 样本: {len(self.batch_results)} | 3D 成功: {ok3d} | 2D 预览: {ok2d} | 无检出/异常: {fails}",
                "",
                "| 样本 | 模式 | 检出 | 顶层 L/D (mm) | R (deg) | SCARA (X, Y, Z) |",
                "|---|---|---|---|---|---|",
            ]
            for r in self.batch_results:
                if r["error"]:
                    lines.append(f"| {r['name']} | - | - | - | - | 异常: {r['error']} |")
                elif r["top"]:
                    l_mm, d_mm, rr, rx, ry, rz = r["top"]
                    lines.append(f"| {r['name']} | {r['mode']} | {r['count']} "
                                 f"| {l_mm}/{d_mm} | {rr} | ({rx}, {ry}, {rz}) |")
                else:
                    lines.append(f"| {r['name']} | {r['mode']} | 0 | - | - | 未检出目标 |")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            log.info("批量解算报告已保存: %s", path)
            return path
        except Exception as exc:
            log.error("批量报告保存失败: %s", exc)
            self.set_toast("批量报告保存失败, 详见日志", True)
            return None

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
            "list_w": int(280 * s),      # 左侧样本列表宽
            "right_w": int(352 * s),     # 右侧结果面板宽
            "header_h": int(52 * s),
            "bottom_h": int(46 * s),
            "row_h": int(44 * s),        # 样本/结果行高
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

        # 标题栏
        draw_text(canvas, "芦笋抓取位姿离线验证", (m["L"], int(16 * m["s"])), m["fs_title"],
                  GuiTheme.TEXT, bold=True)
        draw_text(canvas, f"目录: {self.sample_dir}  ·  样本 {len(self.samples)} 个",
                  (m["L"], int(34 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)

        list_p, img_p, right_p = self._panels(m)
        self._draw_sample_list(canvas, m, list_p)
        self._draw_image_area(canvas, m, img_p)
        self._draw_result_panel(canvas, m, right_p)

        # 标题栏右侧按钮组 (从右向左布局)
        bw = int(88 * m["s"])
        bx = W - m["L"]
        batching = bool(self.batch_queue)
        buttons = [
            ("批量解算 [B]", "batch", not batching),
            ("停止 [S]", "stop", batching),
            ("重新扫描 [R]", "rescan", True),
            ("导出G-code [E]", "export", bool(self.gcode_text)),
            ("退出 [X]", "exit", True),
        ]
        for label, _act, enabled in reversed(buttons):
            bx -= bw + int(8 * m["s"])
            self._draw_button(canvas, (bx, int(14 * m["s"]), bx + bw, int(14 * m["s"]) + m["btn_h"]),
                              label, enabled=enabled)

        # 按钮组最左侧：地图场景选择下拉按钮
        sc_w = int(165 * m["s"])
        bx -= sc_w + int(8 * m["s"])
        self._scene_rect = (bx, int(14 * m["s"]), bx + sc_w, int(14 * m["s"]) + m["btn_h"])
        is_sc_open = (self.active_dropdown == "SCENE_DROPDOWN")
        self._draw_dropdown_button(canvas, self._scene_rect, f"地图: {self.current_scene_name}", is_open=is_sc_open)
        self._buttons.append((self._scene_rect, ("toggle_dd", "SCENE_DROPDOWN")))

        # 底部状态栏
        yb = H - m["bottom_h"] + int(8 * m["s"])
        if self.batch_queue:
            done = len(self.batch_results)
            total = done + len(self.batch_queue)
            status = f"批量解算中... {done}/{total}  ·  3D 成功 " \
                     f"{sum(1 for r in self.batch_results if r['mode'] == '3d' and r['count'])}"
        elif self.batch_results:
            ok = sum(1 for r in self.batch_results if r["count"])
            status = f"上次批量: {len(self.batch_results)} 样本, 检出 {ok}, 报告见 reports/"
        else:
            calib = CALIB_LABELS.get(
                self.targets[0].calibration_source if self.targets else "uncalibrated", "-")
            n3d = sum(1 for smp in self.samples if smp["depth"])
            status = f"样本 {len(self.samples)} 个 (3D 成对 {n3d} / 仅 2D {len(self.samples) - n3d})" \
                     f"  ·  当前标定状态: {calib}"
        draw_text(canvas, status, (m["L"], yb), m["fs_sub"], GuiTheme.TEXT_SUB)
        draw_text(canvas, "[↑↓] 切换样本  ·  [ESC]/[X] 退出  ·  Ctrl+滚轮/± 缩放",
                  (W - int(360 * m["s"]), yb), m["fs_sub"], GuiTheme.TEXT_MUTED)

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
        if self.active_dropdown == "SCENE_DROPDOWN" and self._scene_rect:
            self._render_dropdown_popup(canvas, self._scene_rect, self.scene_options, self.current_scene_id)

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
            draw_text(canvas, "目录无样本", (rect[0] + int(10 * m["s"]), y + int(10 * m["s"])),
                      m["fs_body"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "请用 d435_viewer [S] 抓拍,",
                      (rect[0] + int(10 * m["s"]), y + int(32 * m["s"])), m["fs_small"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "或 --dir 指定照片目录",
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

            name = smp["name"]
            if len(name) > 26:
                name = name[:12] + "..." + name[-11:]
            draw_text(canvas, name, (x1 + int(10 * m["s"]), ry1 + int(5 * m["s"])),
                      m["fs_small"], GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB)
            tag = "3D 成对" if smp["depth"] else "仅 2D"
            tag_col = GuiTheme.OK if smp["depth"] else GuiTheme.WARN
            draw_text(canvas, tag, (x1 + int(10 * m["s"]), ry1 + row_h - int(18 * m["s"])),
                      m["fs_small"], tag_col)
            self._sample_rows.append(((x1 + 2, ry1, x2 - 2, ry2), idx))

    def _draw_image_area(self, canvas, m, rect):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 12, 16), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)

        if self.vis_img is None:
            msg = self.error if self.error else ("选择左侧样本以载入解算" if self.samples else "无样本")
            draw_text(canvas, msg, (x1 + int(16 * m["s"]), y1 + int(16 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if self.error else GuiTheme.TEXT_MUTED)
            return

        # 等比适配面板 (留 8px 内边距)
        avail_w, avail_h = x2 - x1 - int(16 * m["s"]), y2 - y1 - int(16 * m["s"])
        ih, iw = self.vis_img.shape[:2]
        scale = min(avail_w / iw, avail_h / ih)
        disp = cv2.resize(self.vis_img, (max(1, int(iw * scale)), max(1, int(ih * scale))),
                          interpolation=cv2.INTER_AREA)
        ox, oy = x1 + (x2 - x1 - disp.shape[1]) // 2, y1 + (y2 - y1 - disp.shape[0]) // 2
        canvas[oy:oy + disp.shape[0], ox:ox + disp.shape[1]] = disp

        mode_txt = "3D 完整链路" if self.mode == "3d" else "2D 预览 (无深度) — 尺寸按 640mm 标称距离估算"
        mode_col = GuiTheme.OK if self.mode == "3d" else GuiTheme.WARN
        draw_text(canvas, mode_txt, (x1 + int(8 * m["s"]), y2 - int(20 * m["s"])),
                  m["fs_small"], mode_col, bold=True)

    def _draw_result_panel(self, canvas, m, rect):
        y = self._panel_bg(canvas, rect, f"检测结果 ({len(self.targets)})")
        x1, _, x2, y2 = rect
        if not self.targets:
            msg = self.error if self.error else "未检出符合规格的芦笋目标"
            draw_text(canvas, msg, (x1 + int(10 * m["s"]), y + int(10 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if self.error else GuiTheme.TEXT_MUTED)
            self._draw_gcode_box(canvas, m, rect)
            return

        # 结果列表 (顶层目标排序在前, 每行双行文本)
        for list_idx, t in enumerate(self.targets):
            ry1 = y + list_idx * (m["row_h"] + int(4 * m["s"]))
            if ry1 + m["row_h"] > y2 - int(200 * m["s"]):
                break   # 预留 G-code 区
            ry2 = ry1 + m["row_h"]
            is_sel = (list_idx == self.sel_target)
            is_top = t.is_topmost
            if is_sel:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_SEL, -1)
            elif x1 < self.mouse_pos[0] < x2 and ry1 <= self.mouse_pos[1] <= ry2:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_HOVER, -1)
            col = (GuiTheme.OK if is_top else GuiTheme.TEXT_SUB) if not is_sel else GuiTheme.WHITE
            badge = "[TOP] " if is_top else ""
            draw_text(canvas, f"{badge}#{t.id}  L:{t.length_mm}  D:{t.diam_mm}  R:{t.robot_r}",
                      (x1 + int(10 * m["s"]), ry1 + int(4 * m["s"])), m["fs_small"], col, bold=is_top)
            draw_text(canvas, f"SCARA ({t.robot_x}, {t.robot_y}, {t.robot_z})",
                      (x1 + int(10 * m["s"]), ry1 + m["row_h"] - int(18 * m["s"])),
                      m["fs_small"], GuiTheme.ACCENT if is_sel else GuiTheme.TEXT_MUTED)
            self._result_rows.append(((x1 + 2, ry1, x2 - 2, ry2), list_idx))

        self._draw_gcode_box(canvas, m, rect)

    def _draw_gcode_box(self, canvas, m, rect):
        x1, _, x2, y2 = rect
        gh = int(190 * m["s"])
        gy1 = y2 - gh - int(6 * m["s"])
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      (16, 19, 24), -1)
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      GuiTheme.BORDER, 1)
        if self.gcode_text:
            title = f"G-code 预览 (目标 #{self.targets[self.sel_target].id if self.targets else 1}) [E] 导出"
            draw_text(canvas, title, (x1 + int(12 * m["s"]), gy1 + int(5 * m["s"])),
                      m["fs_small"], GuiTheme.GOLD, bold=True)
            yy = gy1 + int(26 * m["s"])
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
        if label.startswith("批量"):
            self.start_batch()
        elif label.startswith("停止"):
            self.batch_queue = []
            self.set_toast("批量解算已停止")
        elif label.startswith("重新扫描"):
            self.rescan()
            self.set_toast(f"已重新扫描: {len(self.samples)} 个样本")
        elif label.startswith("导出"):
            self.export_gcode()
        elif label.startswith("退出"):
            self._running = False

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
            return

        handled, toast = self.win_mgr.handle_mouse_wheel(event, flags)
        if handled:
            self.set_toast(toast)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = (x, y)
            # 1. 优先判定置顶下拉浮层点击
            if self.active_dropdown and self._dd_items:
                for rect, key in self._dd_items:
                    if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                        self.active_dropdown = None
                        self.switch_scene(key)
                        return
                self.active_dropdown = None

            # 2. 常规按钮点击
            hit = self.hit_test(x, y)
            if hit:
                act_type, act_val = hit
                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                return
            for rect, idx in self._sample_rows:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    self._select_sample(idx)
                    return
            for rect, t_idx in self._result_rows:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    self.sel_target = t_idx
                    if self.mode == "3d" and self.targets:
                        self.gcode_text = self.targets[t_idx].generate_gcode(
                            safe_z=self.sys_cfg["safe_z"],
                            drop_x=self.sys_cfg["drop_x"], drop_y=self.sys_cfg["drop_y"])
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
        elif key == "b":
            self.start_batch()
        elif key == "s" and self.batch_queue:
            self.batch_queue = []
            self.set_toast("批量解算已停止")
        elif key == "r":
            self.rescan()
            self.set_toast(f"已重新扫描: {len(self.samples)} 个样本")
        elif key == "e":
            self.export_gcode()
        elif raw_key in (2490368, 65362, 38):      # 上
            if self.sel_idx > 0:
                self._select_sample(self.sel_idx - 1)
        elif raw_key in (2621440, 65364, 40):      # 下
            if self.sel_idx < len(self.samples) - 1:
                self._select_sample(self.sel_idx + 1)

    # ------------------------------ 主循环 ------------------------------
    def run(self):
        self.win_mgr.setup_window(WINDOW_KEY, self._on_mouse)
        self.win_mgr.set_unicode_title("芦笋抓取位姿离线验证 - Asparagus Offline")
        log.info("芦笋离线验证 GUI 已启动: %s (%d 个样本)", self.sample_dir, len(self.samples))

        while self._running:
            key = cv2.waitKey(30)
            poll = self.win_mgr.poll_events(key & 0xFFFF if key > 0 else -1)
            if poll.should_quit:
                break
            if key > 0:
                self._handle_key(key)
            if poll.toast_msg:
                self.set_toast(poll.toast_msg)

            if self.batch_queue:
                self._batch_step()   # 每帧推进一个样本, UI 保持响应

            cv2.imshow(WINDOW_KEY, self.render())

        try:
            cv2.destroyWindow(WINDOW_KEY)
        except Exception:
            pass
        log.info("芦笋离线验证 GUI 已退出")


def main():
    parser = argparse.ArgumentParser(description="芦笋抓取位姿离线验证 GUI (文件照片输入)")
    parser.add_argument("--dir", type=str, default=None,
                        help="样本目录 (默认 data/snapshots/, 彩色 png + 可选对齐深度 npy)")
    args = parser.parse_args()
    app = AsparagusOfflineApp(sample_dir=args.dir)
    app.run()


if __name__ == "__main__":
    main()
