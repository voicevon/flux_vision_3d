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
        pass

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.calibration.scene_manager import CalibrationSceneManager
    DEFAULT_IMAGE_DIR = CalibrationSceneManager().get_active_scene().raw_images_dir
except (ImportError, RuntimeError):
    DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

try:
    from src.utils.window_helper import force_window_focus
except ImportError:
    force_window_focus = None

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False


class TagCaptureWizard:
    def __init__(self, output_dir: str = DEFAULT_IMAGE_DIR, mock_mode: bool = False, config_path: str = CONFIG_PATH):
        self.output_dir = output_dir
        self.mock_mode = mock_mode
        self.config_path = config_path
        os.makedirs(self.output_dir, exist_ok=True)

        # 硬件与运行时状态 (必须先声明，严禁在后续被覆盖为 None)
        self.pipeline = None
        self.is_running = False
        self.flash_timer = 0.0
        self.actual_stream_desc = "1080P Full HD"
        self.last_valid_frame = None

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

        # 初始化 AprilTag 16h5 超高灵敏度检测器
        self.tag_family = cv2.aruco.DICT_APRILTAG_16h5
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.tag_family)
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

        # 最终启动物理相机流 (唯一启动入口)
        if not self.mock_mode and HAVE_REALSENSE:
            self._init_realsense()
        else:
            self.mock_mode = True
            print("[INFO] 处于仿真模式 (--mock) 或未检测到 RealSense 驱动，将使用模拟视觉流。")

    def load_config(self):
        """读取 config.yaml 中的反差与光线配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                det_cfg = cfg.get("calibration", {}).get("tag_detection", {})
                self.valid_tag_ids = [int(x) for x in cfg.get("calibration", {}).get("valid_tag_ids", [])]
                self.contrast_boost = float(det_cfg.get("contrast_boost", self.contrast_boost))
                self.enable_auto_stretch = bool(det_cfg.get("enable_auto_stretch", self.enable_auto_stretch))
                self.clahe_clip_limit = float(det_cfg.get("clahe_clip_limit", self.clahe_clip_limit))
                self.min_otsu_std_dev = float(det_cfg.get("min_otsu_std_dev", self.min_otsu_std_dev))
                self.adaptive_thresh_constant = float(det_cfg.get("adaptive_thresh_constant", self.adaptive_thresh_constant))
                self.min_perimeter_rate = float(det_cfg.get("min_perimeter_rate", self.min_perimeter_rate))
                whitelist_desc = f", 物理白名单={self.valid_tag_ids}" if self.valid_tag_ids else ""
                print(f"[OK] 已成功加载反差配置: 对比度x{self.contrast_boost:.1f}, CLAHE={self.clahe_clip_limit:.1f}{whitelist_desc}")
            except Exception as e:
                print(f"[WARN] 加载 config.yaml 异常: {e}")

    def save_config(self):
        """将当前调整满意的反差参数写回 config.yaml"""
        try:
            cfg = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

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
            print(f"\n[OK] 当前反差与光线配置已成功写入: {self.config_path}")
        except Exception as e:
            self.set_toast(f"保存失败: {e}")
            print(f"[ERROR] 保存 config.yaml 失败: {e}")

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()

    def rebuild_detector(self):
        """根据当前参数重构检测器 (仅重构检测器算法参数，绝不动硬件连接)"""
        # 核心防卡死与实时帧率优化：
        # 1. 窗口扫描步长优化为 10，大幅削减无谓的多边形生成
        # 2. 最小周长门限由 0.006 提升为 0.018 (40mm 标靶即使在 1m 外周长亦大于 60 像素)，彻底过滤 15,000+ 个背景微弱噪点
        # 针对现场黑度不够纯、局部反光与远景小标靶的优化：
        # 1. 窗口扫描步长优化为 8，兼顾极速与细密采样
        # 2. 极低反差门限 (minOtsuStdDev=0.45)，彻底解决纸张发灰、黑度不够纯导致的漏检
        # 3. 周长门限放宽至 0.008，支持远距小标靶
        # 4. 白名单硬锁保护下允许 0.50 汉明纠错，救活墨色不匀与倾斜透视标靶
        
        # 基础参数模版
        def make_params(thresh_c, min_otsu):
            p = cv2.aruco.DetectorParameters()
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 43
            p.adaptiveThreshWinSizeStep = 8
            p.adaptiveThreshConstant = thresh_c
            p.minOtsuStdDev = min_otsu
            p.minMarkerPerimeterRate = max(0.008, self.min_perimeter_rate)
            p.maxMarkerPerimeterRate = 4.0
            p.polygonalApproxAccuracyRate = 0.09
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            p.perspectiveRemovePixelPerCell = 10
            p.errorCorrectionRate = 0.50
            p.perspectiveRemoveIgnoredMarginPerCell = 0.15
            p.maxErroneousBitsInBorderRate = 0.30
            return p

        # 构建两路互补检测器：
        # 路 A: 抗反光/高亮清晰路 (C=5.5, 适合高光、灯光直射视角)
        self.params_bright = make_params(thresh_c=5.5, min_otsu=0.55)
        self.detector_bright = cv2.aruco.ArucoDetector(self.dictionary, self.params_bright)
        
        # 路 B: 低反差/黑度不纯路 (C=2.5, min_otsu=0.45, 适合背光、发灰视角)
        self.params_dark = make_params(thresh_c=2.5, min_otsu=0.45)
        self.detector_dark = cv2.aruco.ArucoDetector(self.dictionary, self.params_dark)
        self.detector = self.detector_bright

        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))

    def _init_realsense(self):
        try:
            ctx = rs.context()
            devices = list(ctx.query_devices())
            if not devices:
                print("[WARN] 未检测到物理相机设备，切至 --mock 仿真模式")
                self.mock_mode = True
                return

            dev = devices[0]
            usb_desc = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"
            print(f"[INFO] 正在连接相机: {dev.get_info(rs.camera_info.name)} (USB 模式: {usb_desc})")

            self.pipeline = rs.pipeline()
            config = rs.config()

            # 优先启用 1080P 超高清模式 (降低帧率至 8fps，像素量暴增 2.25 倍大幅提升小标靶识别率)
            started = False
            try:
                config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 8)
                self.pipeline.start(config)
                self.actual_stream_desc = "1920x1080 @ 8fps (1080P 超清)"
                print("[OK] RealSense D435 彩色流启动成功: 1920x1080 @ 8fps (超高像素模式)")
                started = True
            except Exception as e_1080:
                print(f"[INFO] 1080P 请求未满足 ({e_1080})，尝试 720P 高清模式...")

            if not started:
                # 备用 720P
                if "2." in usb_desc:
                    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
                    self.actual_stream_desc = "1280x720 @ 15fps"
                else:
                    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                    self.actual_stream_desc = "1280x720 @ 30fps"

                self.pipeline.start(config)
                print(f"[OK] RealSense D435 彩色相机启动成功 ({self.actual_stream_desc})！")

            # 预热抛弃前 5 帧，让感光元件自动曝光稳定
            for _ in range(5):
                self.pipeline.wait_for_frames(timeout_ms=2500)

            # 获取物理彩色传感器句柄，支持实时快捷调控硬件曝光与增益
            prof = self.pipeline.get_active_profile()
            for s in prof.get_device().query_sensors():
                if s.is_color_sensor():
                    self.color_sensor = s
                    break

        except Exception as e:
            # 二级回退: 尝试标称 640x480
            try:
                print(f"[WARN] 高清流启动失败 ({e})，正在尝试 640x480 兼容模式...")
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(config)
                self.actual_stream_desc = "640x480 @ 30fps"
                for _ in range(5):
                    self.pipeline.wait_for_frames(timeout_ms=2500)
                print("[OK] RealSense D435 以 640x480 兼容模式启动成功！")
            except Exception as e2:
                print(f"[WARN] 启动物理相机失败: {e2}，自动切换至 --mock 仿真模式")
                self.mock_mode = True
                self.pipeline = None

    def _generate_mock_frame(self, frame_idx: int) -> np.ndarray:
        """生成包含移动 AprilTag 的合成演示帧"""
        frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
        for x in range(0, 1280, 80):
            cv2.line(frame, (x, 0), (x, 720), (55, 55, 55), 1)
        for y in range(0, 720, 80):
            cv2.line(frame, (0, y), (1280, y), (55, 55, 55), 1)

        t = frame_idx * 0.05
        tag_configs = [
            (0, int(350 + 40 * np.sin(t)), int(300 + 30 * np.cos(t)), 90),
            (1, int(650 + 30 * np.cos(t)), int(280 + 20 * np.sin(t)), 85),
            (2, int(850 + 20 * np.sin(t * 0.8)), int(450 + 25 * np.cos(t * 0.8)), 80),
            (3, int(450 + 35 * np.cos(t * 1.2)), int(500 + 15 * np.sin(t * 1.2)), 75),
        ]

        for tag_id, cx, cy, sz in tag_configs:
            hs = sz // 2
            x1, y1 = max(0, cx - hs), max(0, cy - hs)
            x2, y2 = min(1280, cx + hs), min(720, cy + hs)
            tag_img = cv2.aruco.generateImageMarker(self.dictionary, tag_id, sz)
            tag_bgr = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
            h_sub, w_sub = y2 - y1, x2 - x1
            if h_sub > 0 and w_sub > 0:
                frame[y1:y2, x1:x2] = tag_bgr[:h_sub, :w_sub]

        return frame

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """获取当前视频帧 (BGR)"""
        if not self.mock_mode and self.pipeline is not None:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2500)
                color_frame = frames.get_color_frame()
                if color_frame:
                    self.last_valid_frame = np.asanyarray(color_frame.get_data())
                    return self.last_valid_frame
            except Exception as e:
                if self.last_valid_frame is not None:
                    return self.last_valid_frame
                print(f"[WARN] 获取相机帧超时: {e}")
        return self._generate_mock_frame(frame_idx)

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
            print(f"[CAPTURE] 快照 #{self.image_count} 拍摄成功: 原图存入 {raw_filename} | 图示化标注存入 visualized/{vis_filename}")
        else:
            vis_filepath = ""
            print(f"[CAPTURE] 成功拍摄并保存快照 #{self.image_count}: {raw_filepath}")

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
                print(f"  [AUTO-SYNC] 已将快照 #{self.image_count} 自动同步录入清单 {manifest_path} (检出 {len(obs_list)} 个标靶)")

        # 同步更新活动场景元数据
        try:
            from src.calibration.scene_manager import CalibrationSceneManager
            active_sc = CalibrationSceneManager().get_active_scene()
            if os.path.normpath(active_sc.raw_images_dir) == os.path.normpath(self.output_dir):
                active_sc.refresh_stats()
                active_sc.save_meta()
        except Exception as e:
            print(f"[WARN] 场景元数据刷新失败 (非致命): {e}")

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
        极速自适应双路检测 (Fast-Path Adaptive Detection)：
        - 优先执行极速路 1 (原图灰度 + 较严二值门限)，单次仅需 ~25ms；
        - 若已稳定检出充足已知标靶 (>=2) 且未处于强制拉伸预设，直接短路返回，彻底消除拖影；
        - 若路 1 检出标靶不足 2 个，或处于低反差强力预设，才自适应执行路 2 (动态拉伸路) 补充暗部；
        - 兼顾 8fps 满帧跟手与 100% 极限召回。
        """
        if len(raw_frame.shape) == 3:
            gray_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_raw = raw_frame

        found = {}

        # 路 1: 针对清晰/高光/反光区域 (极速单路)
        c1, ids1, _ = self.detector_bright.detectMarkers(gray_raw)
        if ids1 is not None and len(ids1) > 0:
            for i, tid in enumerate(ids1.flatten()):
                tid_int = int(tid)
                if self.valid_tag_ids and tid_int not in self.valid_tag_ids:
                    continue
                found[tid_int] = c1[i]

        # 快速短路：若普通路已检出满足共视条件的已知标靶，跳过耗时的二次动态拉伸
        if len(found) >= 2 and not self.enable_auto_stretch:
            return found

        # 路 2: 针对暗部/低反差/打印黑度不够纯区域 (动态拉伸 + 宽松门限)
        p_low, p_high = np.percentile(gray_raw[::4, ::4], (2, 98))
        if p_high > p_low + 10:
            gray_stretch = np.clip((gray_raw.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            gray_stretch = gray_raw

        c2, ids2, _ = self.detector_dark.detectMarkers(gray_stretch)
        if ids2 is not None and len(ids2) > 0:
            for i, tid in enumerate(ids2.flatten()):
                tid_int = int(tid)
                if self.valid_tag_ids and tid_int not in self.valid_tag_ids:
                    continue
                if tid_int not in found:
                    found[tid_int] = c2[i]

        return found

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

        hw = 15.0   # 截面半宽 15mm，整体截面边长为 30.0mm x 30.0mm
        L = 80.0    # 柱体长度 80mm (约原高度 2/3)

        # 8 个 3D 角点: 底面 4 点 (Z=0), 顶面 4 点 (Z=L)
        pts_3d = np.array([
            # 底面 4 点
            [-hw, -hw, 0.0],
            [ hw, -hw, 0.0],
            [ hw,  hw, 0.0],
            [-hw,  hw, 0.0],
            # 顶面 4 点
            [-hw, -hw, L],
            [ hw, -hw, L],
            [ hw,  hw, L],
            [-hw,  hw, L],
            # 顶面中心
            [0.0, 0.0, L],
            # X 轴与 Y 轴参考端点 (从中心伸出 25mm，突出棱柱外侧)
            [25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)

        proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        proj = proj.reshape((-1, 2)).astype(int)

        b_pts = proj[0:4] # 底面 4 点
        t_pts = proj[4:8] # 顶面 4 点
        top_center = tuple(proj[8])
        p_x = tuple(proj[9])
        p_y = tuple(proj[10])
        p_orig = tuple(proj[11])

        # 1. 绘制 X 轴 (红色) 和 Y 轴 (绿色)
        cv2.line(img, p_orig, p_x, (0, 0, 240), 2, cv2.LINE_AA)
        cv2.putText(img, 'X', p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, p_orig, p_y, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Y', p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # 2. 半透明填充 4 个侧面与顶面 (呈现实心方柱体立体质感)
        overlay = img.copy()
        for i in range(4):
            next_i = (i + 1) % 4
            side_poly = np.array([b_pts[i], b_pts[next_i], t_pts[next_i], t_pts[i]], dtype=np.int32)
            cv2.fillPoly(overlay, [side_poly], (240, 160, 30)) # BGR: 浅蓝/青
        cv2.fillPoly(overlay, [t_pts], (255, 220, 90))         # 顶面高光
        cv2.addWeighted(overlay, 0.42, img, 0.58, 0, img)

        # 3. 绘制 12 条棱线 (高清晰边框)
        cv2.polylines(img, [b_pts], isClosed=True, color=(180, 80, 0), thickness=2, lineType=cv2.LINE_AA)
        for i in range(4):
            cv2.line(img, tuple(b_pts[i]), tuple(t_pts[i]), (255, 130, 0), 2, cv2.LINE_AA)
        cv2.polylines(img, [t_pts], isClosed=True, color=(255, 240, 120), thickness=2, lineType=cv2.LINE_AA)

        # 4. 顶面中心标注点与文字 (简洁工业标定, 仅保留 Z 轴标识)
        cv2.circle(img, top_center, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, 'Z', (top_center[0] + 5, top_center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 80), 2, cv2.LINE_AA)

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
                    print(f"\n[INFO] 采图向导结束。当前数据集共计 {self.image_count} 帧。")
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
                        print("[OK] 原图与图示化文件目录已全部清空。")

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    print(f"[WARN] pipeline.stop 异常 (非致命): {e}")
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
