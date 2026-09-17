#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 多视角交互式采图向导 (Tag Capture Wizard)
===================================================
用途：
  1. 实时预览 RealSense D435 彩色画面，毫秒级检测并高亮 AprilTag 16h5；
  2. 实时监测画面中的标靶数量与共视条件（>= 2 个 Tag 为有效建图视角）；
  3. 提示 Tag 0（SCARA 基座原点）的捕获状态；
  4. 按 [空格] 键一键拍摄保存无标注的高清原始帧至 data/tag_calibration_images/；
  5. 提供拍照快门白闪视觉反馈与采样计数，采图完毕后可直接衔接空间建图；
  6. 支持 --mock 模式，无物理相机时亦可进行交互演示。
"""

import os
import sys
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

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.calibration.scene_manager import CalibrationSceneManager
    DEFAULT_IMAGE_DIR = CalibrationSceneManager().get_active_scene().raw_images_dir
except (ImportError, RuntimeError):
    DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")

from src.calibration.tag_detector import TagDetector
from src.calibration.camera_service import CameraService
from src.calibration.prism_renderer import draw_prism, COLORS_MAPPING
from src.utils.config_guard import load_raw_config
from src.utils.logger import get_logger

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

try:
    from tools.window_helper import force_window_focus
except ImportError:
    force_window_focus = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

log = get_logger(__name__)


class TagCaptureWizard:
    def __init__(self, output_dir: str = DEFAULT_IMAGE_DIR, mock_mode: bool = False, config_path: str = CONFIG_PATH):
        self.output_dir = output_dir
        self.mock_mode = mock_mode
        self.config_path = config_path
        os.makedirs(self.output_dir, exist_ok=True)

        # 硬件与运行时状态 (必须先声明，严禁在后续被覆盖为 None)
        self.is_running = False
        self.flash_timer = 0.0
        self.actual_stream_desc = "1080P Full HD"
        # 统一取流服务: 硬件启停/帧读取/Mock 仿真全部委托 CameraService
        self._cam_srv = CameraService()

        # 默认反差与环境光配置 (优先读取 config.yaml)
        self.contrast_boost = 1.8
        self.enable_auto_stretch = True
        self.clahe_clip_limit = 4.0
        self.min_otsu_std_dev = 1.5
        self.adaptive_thresh_constant = 3.0
        self.min_perimeter_rate = 0.006
        self.active_preset_name = "超强低反差模式"
        self.status_toast = ""
        self.status_toast_time = 0.0

        self.valid_tag_ids = []
        self.color_sensor = None
        self.show_3d_axes = False          # 采图向导默认关闭繁重 3D 棱柱，专注极速跟手与轻量取景
        self.load_config()

        # 初始化 AprilTag 16h5 统一检测器 (dictionary 供 generateImageMarker 复用)
        self.tag_detector = TagDetector(
            valid_tag_ids=self.valid_tag_ids,
            min_perimeter_rate=self.min_perimeter_rate,
            enable_auto_stretch=self.enable_auto_stretch
        )
        self.dictionary = self.tag_detector.dictionary
        self.rebuild_detector()

        # 加载相机内参与 3D 空间坐标投影模型 (支持实心加粗 5mm Z轴与 XYZ 空间坐标系渲染)
        self.marker_size_mm = 40.0
        try:
            from src.utils.config_guard import resolve_camera_intrinsics
            K, dist, _ = resolve_camera_intrinsics(self.config_path)
            self.camera_matrix = K
            self.dist_coeffs = dist
        except (ImportError, FileNotFoundError, KeyError, yaml.YAMLError):
            self.camera_matrix = np.array([
                [1363.68, 0.0, 971.19],
                [0.0, 1361.19, 566.26],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        s = self.marker_size_mm / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

        # 统计已有图片数
        existing = glob.glob(os.path.join(self.output_dir, "view_*.png"))
        self.image_count = len(existing)

        # 最终启动物理相机流 (唯一启动入口; 无 SDK/无设备时服务内部优雅切 Mock)
        if not self.mock_mode:
            self._init_realsense()
        else:
            self._cam_srv.enter_mock()
            log.info("处于仿真模式 (--mock)，将使用模拟视觉流。")

    def load_config(self):
        """读取 config.yaml 中的反差与光线配置 (统一走 config_guard)"""
        cfg = load_raw_config(self.config_path)
        if cfg:
            try:
                det_cfg = cfg.get("calibration", {}).get("tag_detection", {})
                self.valid_tag_ids = [int(x) for x in cfg.get("calibration", {}).get("valid_tag_ids", [])]
                self.contrast_boost = float(det_cfg.get("contrast_boost", self.contrast_boost))
                self.enable_auto_stretch = bool(det_cfg.get("enable_auto_stretch", self.enable_auto_stretch))
                self.clahe_clip_limit = float(det_cfg.get("clahe_clip_limit", self.clahe_clip_limit))
                self.min_otsu_std_dev = float(det_cfg.get("min_otsu_std_dev", self.min_otsu_std_dev))
                self.adaptive_thresh_constant = float(det_cfg.get("adaptive_thresh_constant", self.adaptive_thresh_constant))
                self.min_perimeter_rate = float(det_cfg.get("min_perimeter_rate", self.min_perimeter_rate))
                whitelist_desc = f", 物理白名单={self.valid_tag_ids}" if self.valid_tag_ids else ""
                log.info(f"[OK] 已成功加载反差配置: 对比度x{self.contrast_boost:.1f}, CLAHE={self.clahe_clip_limit:.1f}{whitelist_desc}")
            except Exception as e:
                log.warning(f"加载 config.yaml 异常: {e}")

    def save_config(self):
        """将当前调整满意的反差参数写回 config.yaml"""
        try:
            cfg = load_raw_config(self.config_path)

            if "calibration" not in cfg:
                cfg["calibration"] = {}

            cfg["calibration"]["tag_detection"] = {
                "contrast_boost": round(float(self.contrast_boost), 2),
                "enable_auto_stretch": bool(self.enable_auto_stretch),
                "clahe_clip_limit": round(float(self.clahe_clip_limit), 1),
                "min_otsu_std_dev": round(float(self.min_otsu_std_dev), 2),
                "adaptive_thresh_constant": round(float(self.adaptive_thresh_constant), 1),
                "min_perimeter_rate": round(float(self.min_perimeter_rate), 4),
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            self.set_toast(f"参数已持久化保存至 config.yaml！")
            log.info(f"\n[OK] 当前反差与光线配置已成功写入: {self.config_path}")
        except Exception as e:
            self.set_toast(f"保存失败: {e}")
            log.warning(f"保存 config.yaml 失败: {e}")

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()

    def rebuild_detector(self):
        """根据当前参数重构检测器 (仅重构检测器算法参数，绝不动硬件连接)"""
        self.tag_detector.set_params(
            valid_tag_ids=self.valid_tag_ids,
            min_perimeter_rate=self.min_perimeter_rate,
            enable_auto_stretch=self.enable_auto_stretch
        )

    def _init_realsense(self):
        """启动 RealSense 彩色流 (委托统一 CameraService)：
        1080P 超清优先 (8fps 高像素模式提升小标靶识别率)，逐级回退 720P/640x480，
        全部失败时服务内部优雅切入 Mock 仿真模式。"""
        self._cam_srv.start_realsense(
            1920, 1080, fps=8,
            fallbacks=((1280, 720, 15), (640, 480, 30)),
            mock_fallback=True)
        self.mock_mode = self._cam_srv.is_mock
        self.color_sensor = self._cam_srv.color_sensor
        if not self.mock_mode:
            self.actual_stream_desc = self._cam_srv.stream_desc

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """获取当前视频帧 (BGR): 硬件帧优先, 瞬时失败回退最近有效帧, Mock 生成仿真帧"""
        if not self.mock_mode:
            frame = self._cam_srv.read_frame(timeout_ms=2500)
            if frame is not None:
                return frame
        return self._cam_srv.make_mock_frame(frame_idx)

    def save_image(self, raw_frame: np.ndarray, annotated_frame: np.ndarray = None) -> str:
        """
        保存采图快照：
        - 原始无标注高清原图保存至: data/tag_calibration_images/view_XXXX.png (供几何建图高精度求解)
        - 带标靶多边形、角点、ID标注的图示化图片集中保存至: data/tag_calibration_images/visualized/view_XXXX_annotated.png (供人工直观检查复核)
        """
        self.image_count += 1
        raw_filename = f"view_{self.image_count:04d}.png"
        raw_filepath = os.path.join(self.output_dir, raw_filename)
        cv2.imwrite(raw_filepath, raw_frame)

        # 集中保存图示化文件至子目录
        vis_dir = os.path.join(self.output_dir, "visualized")
        os.makedirs(vis_dir, exist_ok=True)
        if annotated_frame is not None:
            vis_filename = f"view_{self.image_count:04d}_annotated.png"
            vis_filepath = os.path.join(vis_dir, vis_filename)
            cv2.imwrite(vis_filepath, annotated_frame)
            log.info(f"[CAPTURE] 快照 #{self.image_count} 拍摄成功: 原图存入 {raw_filename} | 图示化标注存入 visualized/{vis_filename}")
        else:
            vis_filepath = ""
            log.info(f"[CAPTURE] 成功拍摄并保存快照 #{self.image_count}: {raw_filepath}")

        # 若已有 tag_observations.yaml 存在，自动将该帧增量追加进观测清单
        manifest_path = os.path.join(self.output_dir, "tag_observations.yaml")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = yaml.safe_load(f) or {}
            images_dict = manifest_data.setdefault("images", {})
            if raw_filename not in images_dict:
                found_tags = self.detect_tags_robust(raw_frame) or {}
                obs_list = []
                for tid in sorted(found_tags.keys()):
                    corners = found_tags[tid]
                    pts = corners.reshape((4, 2)).astype(np.float64)
                    l01 = float(np.linalg.norm(pts[1] - pts[0]))
                    l12 = float(np.linalg.norm(pts[2] - pts[1]))
                    l23 = float(np.linalg.norm(pts[3] - pts[2]))
                    l30 = float(np.linalg.norm(pts[0] - pts[3]))
                    cell_w = int(round((l01 + l23) / 12.0))
                    cell_h = int(round((l30 + l12) / 12.0))
                    center_x = float(np.mean(pts[:, 0]))
                    center_y = float(np.mean(pts[:, 1]))
                    area = float(cv2.contourArea(pts.astype(np.float32)))
                    obs_list.append({
                        "tag_id": int(tid),
                        "keep": True,
                        "cell_size_px": [cell_w, cell_h],
                        "center_px": [round(center_x, 1), round(center_y, 1)],
                        "area_px": round(area, 1),
                        "note": "采图向导现场拍摄录入",
                        "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in pts]
                    })
                rel_img_path = os.path.relpath(raw_filepath, PROJECT_ROOT).replace("\\", "/")
                rel_vis_path = os.path.relpath(vis_filepath, PROJECT_ROOT).replace("\\", "/") if vis_filepath else ""
                images_dict[raw_filename] = {
                    "file_name": raw_filename,
                    "image_path": rel_img_path,
                    "annotated_path": rel_vis_path,
                    "detected_count": len(obs_list),
                    "observations": obs_list
                }
                total_obs = sum(len(img["observations"]) for img in images_dict.values())
                total_kept = sum(sum(1 for obs in img["observations"] if obs.get("keep", True)) for img in images_dict.values())
                manifest_data.setdefault("summary", {})
                manifest_data["summary"]["total_images"] = len(images_dict)
                manifest_data["summary"]["total_observations"] = total_obs
                manifest_data["summary"]["total_kept"] = total_kept
                with open(manifest_path, "w", encoding="utf-8") as f:
                    yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                log.info(f"  [AUTO-SYNC] 已将快照 #{self.image_count} 自动同步录入清单 {manifest_path} (检出 {len(obs_list)} 个标靶)")

        # 同步更新活动场景元数据
        try:
            from src.calibration.scene_manager import CalibrationSceneManager
            active_sc = CalibrationSceneManager().get_active_scene()
            if os.path.normpath(active_sc.raw_images_dir) == os.path.normpath(self.output_dir):
                active_sc.refresh_stats()
                active_sc.save_meta()
        except Exception as e:
            log.warning(f"场景元数据刷新失败 (非致命): {e}")

        self.flash_timer = time.time()
        return raw_filepath

    def preprocess_image(self, bgr_frame: np.ndarray) -> np.ndarray:
        """根据对比度与环境光参数预处理灰度图像 (快速采样分位数)"""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        # 仅在明确启用时进行动态拉伸，采用 4x4 步长快速采样计算分位数，避免全图 200万像素排序带来的毫秒级延迟
        if self.enable_auto_stretch:
            p_low, p_high = np.percentile(gray[::4, ::4], (2, 98))
            if p_high > p_low + 10:
                gray = np.clip((gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)

        # 温和的对比度调节 (避免过激截断)
        if abs(self.contrast_boost - 1.0) > 0.05:
            gray = np.clip(128.0 + (gray.astype(np.float32) - 128.0) * self.contrast_boost, 0, 255).astype(np.uint8)

        return gray

    def apply_preset(self, preset_idx: int):
        """应用三种典型工业场景反差预设"""
        if preset_idx == 0:
            self.active_preset_name = "均衡标准模式"
            self.contrast_boost = 1.0
            self.enable_auto_stretch = False
            self.clahe_clip_limit = 2.5
            self.min_otsu_std_dev = 3.5
            self.adaptive_thresh_constant = 5.0
        elif preset_idx == 1:
            self.active_preset_name = "超强低反差模式 (针对当前现场发灰/暗处)"
            self.contrast_boost = 1.8
            self.enable_auto_stretch = True
            self.clahe_clip_limit = 4.5
            self.min_otsu_std_dev = 1.2
            self.adaptive_thresh_constant = 2.5
        elif preset_idx == 2:
            self.active_preset_name = "抗强光过曝反光模式"
            self.contrast_boost = 1.2
            self.enable_auto_stretch = False
            self.clahe_clip_limit = 3.0
            self.min_otsu_std_dev = 3.0
            self.adaptive_thresh_constant = 6.0

        self.rebuild_detector()
        self.set_toast(f"已切换至: {self.active_preset_name}")

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

    def detect_tags_robust(self, raw_frame: np.ndarray):
        """
        极速自适应双路检测 (委托统一 TagDetector fast 路)：
        高光路检出 >= 2 个白名单标靶且未强制拉伸时短路返回，消除拖影；
        否则自动执行暗部动态拉伸路补齐低反差/黑度不纯标靶 (实时取流不做精修)。
        """
        return self.tag_detector.detect_tags(raw_frame, fast=True, refine=False)

    def render_tag_3d_axes(self, img: np.ndarray, corners: np.ndarray, tag_id: int):
        """
        绘制标靶 3D 空间坐标系与实心正四棱柱：
        - X 轴 (红色, 25mm), Y 轴 (绿色, 25mm)
        - Z 轴指示: 边长 20.0mm x 20.0mm (原始标靶 1/2)、长 120.0mm 的实心正四棱柱 (半透明实心柱体 + 12条高亮棱线 + 顶盖透视截面)
        """
        corners_2d = corners.reshape((4, 2)).astype(np.float64)
        retval, rvecs, tvecs, reprojErrors = cv2.solvePnPGeneric(
            self.obj_points, corners_2d, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not retval or len(rvecs) == 0:
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_points, corners_2d, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return
        elif len(rvecs) == 1:
            rvec, tvec = rvecs[0], tvecs[0]
        else:
            err0 = reprojErrors[0][0] if reprojErrors is not None else 0.0
            err1 = reprojErrors[1][0] if reprojErrors is not None else 0.0
            best_idx = 0
            if abs(err0 - err1) < 2.5:
                R0, _ = cv2.Rodrigues(rvecs[0])
                R1, _ = cv2.Rodrigues(rvecs[1])
                if R0[1, 2] >= 0 and R1[1, 2] < 0:
                    best_idx = 1
                elif R1[1, 2] >= 0 and R0[1, 2] < 0:
                    best_idx = 0
                else:
                    best_idx = 0 if err0 <= err1 else 1
            else:
                best_idx = 0 if err0 <= err1 else 1
            rvec, tvec = rvecs[best_idx], tvecs[best_idx]

        # 统一 PrismRenderer: 实心正四棱柱 (半透明 + 12 棱线 + 顶盖) + XYZ 坐标轴
        draw_prism(img, self.camera_matrix, self.dist_coeffs, rvec, tvec,
                   half_w=15.0, height=80.0, colors=COLORS_MAPPING,
                   alpha=0.42, draw_axes=True, axis_len=25.0)

    def run(self):
        """运行交互式采图主循环"""
        self.is_running = True
        window_name = "AprilTag Multi-View Capture Wizard (Space: Capture | Q: Exit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        print("\n" + "=" * 70)
        print("          AprilTag 多视角交互式采图向导启动")
        print("=" * 70)
        print(f" [画面流规格] : {self.actual_stream_desc}")
        print(f" [检测模式]   : {self.active_preset_name} (对比度x{self.contrast_boost:.1f}, CLAHE={self.clahe_clip_limit:.1f})")
        print(f" [存储目录]   : {self.output_dir}")
        print(f" [已存图像]   : {self.image_count} 张")
        print(" [快捷键指南] :")
        print("   - [Space] (空格键) : 拍摄并保存当前视角高清原图；")
        print("   - [Tab]            : 循环切换 3 种反差预设 (标准 / 超强低反差 / 抗过曝)；")
        print("   - [W]              : 一键切换【白名单过滤】/【全量探索模式 (放行所有Tag 0~29)】；")
        print("   - [I] / [K]        : 实时增减对比度拉伸 (I 增加 / K 减少)；")
        print("   - [S]              : 将当前微调参数持久化写入 config.yaml；")
        print("   - [C]              : 清空当前采图目录；")
        print("   - [Q] 或 [ESC]     : 退出向导。")
        print("=" * 70 + "\n")

        preset_index = 1 if self.contrast_boost > 1.3 else 0
        frame_idx = 0
        fps_display = 0.0
        fps_calc_time = time.time()
        fps_frames = 0
        cached_valid_ids = list(self.valid_tag_ids)

        try:
            while self.is_running:
                t_frame_start = time.time()
                raw_frame = self.get_frame(frame_idx)
                frame_idx += 1
                disp_frame = raw_frame.copy()

                # 超高灵敏度融合检测 AprilTag
                found_tags = self.detect_tags_robust(raw_frame) or {}

                detected_tags = list(found_tags.keys())
                has_origin_tag = 0 in found_tags

                for tag_id, corner_arr in found_tags.items():
                    pts = corner_arr.reshape((4, 2)).astype(int)
                    # Tag 0 采用高亮金黄 (0, 215, 255)，普通已知标靶采用鲜明绿色 (0, 255, 0)
                    is_origin = (tag_id == 0)
                    tag_color = (0, 215, 255) if is_origin else (0, 255, 0)

                    # 绘制 2D 轻量高反差多边形双层边框 (外黑内亮，不吃 CPU)
                    cv2.polylines(disp_frame, [pts], isClosed=True, color=(10, 10, 10), thickness=4)
                    cv2.polylines(disp_frame, [pts], isClosed=True, color=tag_color, thickness=2)

                    # 绘制角点序号微圆点 (0:红, 1:绿, 2:蓝, 3:黄，清晰辨识方向)
                    dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                    for pt_idx, pt in enumerate(pts):
                        cv2.circle(disp_frame, tuple(pt), 4, dot_colors[pt_idx], -1)

                    # 计算机械标靶单元方格像素尺寸 (AprilTag 16h5 为 6x6 网格)
                    l01 = np.linalg.norm(pts[1] - pts[0])
                    l12 = np.linalg.norm(pts[2] - pts[1])
                    l23 = np.linalg.norm(pts[3] - pts[2])
                    l30 = np.linalg.norm(pts[0] - pts[3])
                    cell_w = int(round((l01 + l23) / 12.0))
                    cell_h = int(round((l30 + l12) / 12.0))

                    # 绘制极速轻量标牌
                    cx = int(np.mean(pts[:, 0]))
                    min_y = int(np.min(pts[:, 1]))
                    tag_text = f"Tag {tag_id}" + (" [ORIGIN 原点]" if is_origin else "")
                    cell_text = f"Cell: {cell_w}x{cell_h}px"
                    
                    badge_w = 140 if is_origin else 115
                    badge_x = max(10, min(disp_frame.shape[1] - badge_w - 10, cx - badge_w // 2))
                    badge_y = max(42, min_y - 12)
                    cv2.rectangle(disp_frame, (badge_x - 6, badge_y - 30), (badge_x + badge_w, badge_y + 8), (20, 20, 20), -1)
                    cv2.rectangle(disp_frame, (badge_x - 6, badge_y - 30), (badge_x + badge_w, badge_y + 8), tag_color, 1)
                    cv2.putText(disp_frame, tag_text, (badge_x, badge_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(disp_frame, cell_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

                    # 仅在用户显式开启时才做 3D 棱柱投影 (默认关闭，释放全部算力供 8fps 流畅取景)
                    if self.show_3d_axes:
                        self.render_tag_3d_axes(disp_frame, corner_arr, tag_id)

                num_tags = len(detected_tags)
                is_covisible = num_tags >= 2

                # 顶部状态条渲染 (半透明黑底)
                overlay = disp_frame.copy()
                cv2.rectangle(overlay, (0, 0), (disp_frame.shape[1], 65), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.8, disp_frame, 0.2, 0, disp_frame)

                # 状态文字 (展示已检出全部 ID 列表)
                status_color = (0, 255, 0) if is_covisible else (0, 165, 255)
                tag_list_str = str(sorted(detected_tags)) if detected_tags else "None"
                status_text = f"Tags [{num_tags}]: {tag_list_str} " + ("[CO-VISIBILITY OK]" if is_covisible else "[NEED >= 2]")
                if has_origin_tag:
                    status_text += " | [Tag 0 ORIGIN OK]"

                cv2.putText(disp_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # 第二行显示相机模式、实时FPS与白名单模式
                if not self.valid_tag_ids:
                    wl_str = " | Whitelist: ALL [Exploring]"
                    wl_color = (0, 255, 255)
                else:
                    wl_str = f" | Whitelist: {len(self.valid_tag_ids)} IDs"
                    wl_color = (190, 190, 190)

                fps_frames += 1
                now = time.time()
                if now - fps_calc_time >= 0.5:
                    fps_display = fps_frames / (now - fps_calc_time)
                    fps_calc_time = now
                    fps_frames = 0

                info_text = f"FPS: {fps_display:.1f} | Stream: {self.actual_stream_desc} | Contrast: x{self.contrast_boost:.1f}{wl_str}"
                cv2.putText(disp_frame, info_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, wl_color, 1)

                # 右侧计数与保存提示
                tip_text = f"Saved: {self.image_count} frames | [Space]: Save"
                (rw, _), _ = cv2.getTextSize(tip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.putText(disp_frame, tip_text, (disp_frame.shape[1] - rw - 15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

                # 底部控制提示条 (半透明)
                h_img, w_img = disp_frame.shape[:2]
                cv2.rectangle(disp_frame, (0, h_img - 35), (w_img, h_img), (15, 15, 15), -1)
                ctrl_tip = "[Space]: Pic | [Tab]: Preset | [[ / ]]: Exp | [E]: AutoExp | [I/K]: Contrast | [A]: 3D | [W]: WhiteList | [Q]: Exit"
                cv2.putText(disp_frame, ctrl_tip, (15, h_img - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

                # Toast 临时通知提示
                if time.time() - self.status_toast_time < 2.5 and self.status_toast:
                    (tw, _), _ = cv2.getTextSize(self.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    toast_x = (w_img - tw) // 2
                    cv2.rectangle(disp_frame, (toast_x - 12, h_img - 80), (toast_x + tw + 12, h_img - 45), (0, 120, 0), -1)
                    cv2.putText(disp_frame, self.status_toast, (toast_x, h_img - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                # 快门白闪反馈
                if time.time() - self.flash_timer < 0.12:
                    disp_frame = cv2.addWeighted(disp_frame, 0.4, np.full_like(disp_frame, 255), 0.6, 0)

                cv2.imshow(window_name, disp_frame)
                if frame_idx <= 3 and force_window_focus:
                    force_window_focus(window_name)

                key = cv2.waitKey(10) & 0xFF

                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    log.info(f"\n采图向导结束。当前数据集共计 {self.image_count} 帧。")
                    break
                elif key == 32:  # Space
                    self.save_image(raw_frame, disp_frame)
                elif key == 9:   # Tab (切换反差预设)
                    preset_index = (preset_index + 1) % 3
                    self.apply_preset(preset_index)
                elif key == ord('['):  # 压暗曝光
                    self.adjust_hardware_exposure(-50.0)
                elif key == ord(']'):  # 提亮曝光
                    self.adjust_hardware_exposure(50.0)
                elif key in (ord('e'), ord('E')):  # 切换自动曝光
                    self.toggle_auto_exposure()
                elif key in (ord('a'), ord('A')):  # 切换 3D 棱柱显示
                    self.show_3d_axes = not self.show_3d_axes
                    self.set_toast(f"3D 棱柱空间轴: {'开启' if self.show_3d_axes else '关闭 (极速2D)'}")
                elif key in (ord('w'), ord('W')):  # 一键切换白名单探索模式 / 限制模式
                    if self.valid_tag_ids:
                        cached_valid_ids = list(self.valid_tag_ids)
                        self.valid_tag_ids = []
                        self.set_toast("已开启【全量探索模式】：放行所有识别到的 AprilTag (0~29)")
                    else:
                        self.valid_tag_ids = list(cached_valid_ids) if cached_valid_ids else []
                        if self.valid_tag_ids:
                            self.set_toast(f"已恢复【白名单过滤模式】：仅放行 {self.valid_tag_ids}")
                        else:
                            self.set_toast("当前未配置固定白名单，仍处于全量探索模式")
                elif key in (ord('i'), ord('I'), ord('+'), ord('=')):  # 增大对比度增益 (I 键超便捷)
                    self.contrast_boost = min(3.5, self.contrast_boost + 0.2)
                    self.rebuild_detector()
                    self.set_toast(f"对比度增益已调至: x{self.contrast_boost:.1f} [按 I 增大 / K 减小]")
                elif key in (ord('k'), ord('K'), ord('-'), ord('_')):  # 降低对比度增益 (K 键超便捷)
                    self.contrast_boost = max(0.6, self.contrast_boost - 0.2)
                    self.rebuild_detector()
                    self.set_toast(f"对比度增益已调至: x{self.contrast_boost:.1f} [按 I 增大 / K 减小]")
                elif key in (ord('s'), ord('S')):  # 保存参数至 config.yaml
                    self.save_config()
                elif key in (ord('c'), ord('C')):  # Clear
                    print("\n[?] 是否确认清空当前采图目录所有原图与图示化文件？按 [Y] 确认，其他键取消: ", end="", flush=True)
                    cv2.waitKey(0)  # 让出短暂交互
                    confirm = input() if sys.stdin.isatty() else "n"
                    if confirm.strip().lower() == 'y':
                        all_del = glob.glob(os.path.join(self.output_dir, "*.png")) + glob.glob(os.path.join(self.output_dir, "visualized", "*.png"))
                        for f in all_del:
                            try:
                                os.remove(f)
                            except OSError:
                                pass  # 文件被占用/已删除/权限不足 — 合法窄异常
                        self.image_count = 0
                        log.info("[OK] 原图与图示化文件目录已全部清空。")

        finally:
            self._cam_srv.stop()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AprilTag 多视角交互式采图向导")
    parser.add_argument("--dir", "--output_dir", dest="dir", type=str, default=DEFAULT_IMAGE_DIR, help="保存采集图像的目录路径")
    parser.add_argument("--mock", action="store_true", help="仿真模式：无需物理相机演示采图交互")
    args = parser.parse_args()

    wizard = TagCaptureWizard(output_dir=args.dir, mock_mode=args.mock)
    wizard.run()


if __name__ == "__main__":
    main()
