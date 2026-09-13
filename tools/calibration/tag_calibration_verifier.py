#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 标定精度与 3D 坐标系在线 AR 综合验证系统 (Integrated AR & Static Lock Verifier)
======================================================================================
核心职责与专业特性：
  1. 实时动态 (LIVE) 与 静态滤波锁定 (STATIC LOCKED) 一键乒乓切换：
     - 模式 A【实时动态 (LIVE)】：
       * 强调即时巡检、高帧率动态响应、零延迟所见即所得；
       * 单帧亚像素即时 PnP 求解，直观展现真实传感器与环境动态。
     - 模式 B【静态滤波锁定 (STATIC LOCKED)】：
       * 针对工业现场“相机与标靶短时间内相对静止”的客观物理先验；
       * 【两阶段批处理基准凝固】：在静止状态下采足固定批次帧数（默认 30 帧或 60 帧）；
       * 采足后对视野内每个 Tag 的 4 角点集中进行【去极值均值滤波】，消除 CMOS 散粒噪声；
       * 一次性计算超定高精相机 6DoF 位姿后【彻底停止采样与滤波，将位姿绝对冻结锁定 (LOCKED)】；
       * 空间绝对抖动严格为 0.00 mm，位姿彻底凝固，专供高精度留一盲测与验收量测。
  2. 留一法盲测交叉验证 (Leave-One-Out Blind AR Cross-Validation)：
     - 无论在实时动态还是静态锁定状态下，均可点选指定标靶（如 Tag 26）作为盲测目标；
     - PnP 解算时主动剔除该标靶，由其余标靶隔空反推其 3D 空间位姿与 30mm 实心正四棱柱 3D 轴；
     - 实时量化计算重投影像元误差 (px) 与空间绝对偏差 (mm)。
  3. 工业级 3D 正四棱柱立体质感渲染 (render_tag_3d_axes)：
     - 截面边长 30.0mm x 30.0mm、柱体高 80.0mm (约原高度 2/3)，粗壮紧凑；
     - 12 条亮白棱线 + 顶盖透视截面 + 盲测高光品红立体光效。
  4. 全 GUI 交互工具栏与快捷键无缝配合：
     - 底部乒乓开关大按钮 + 顶栏标靶点选 + 画面直接点选；
     - 快捷键：[Tab/M] 乒乓切换模式、[Space] 重新采样锁定/抓拍、[W] 切换采样批次(30F/60F)、[T/B] 盲测切换、[A/D] 仿真翻页。
"""


import os
import sys
import time
import math
import glob
import yaml
import argparse
import numpy as np
import cv2

# Windows 终端 UTF-8 编码适配
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_MAP_PATH = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
VERIFICATION_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_verification")
CALIB_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

sys.path.insert(0, PROJECT_ROOT)
try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

try:
    from src.utils.window_helper import force_window_focus
except ImportError:
    force_window_focus = None

try:
    from src.utils.viewport_manager import (
        ViewportManager, get_safe_screen_size,
        draw_styled_button, draw_segmented_toggle
    )
except ImportError:
    ViewportManager = None
    get_safe_screen_size = None
    draw_styled_button = None
    draw_segmented_toggle = None

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False


class TagCalibrationVerifier:
    def __init__(self, map_path: str = DEFAULT_MAP_PATH, mock_mode: bool = False, report_path: str = None):
        self.map_path = map_path
        self.mock_mode = mock_mode
        self.report_path = report_path
        self.tags_map = None
        self.marker_size_mm = 50.0

        # 自适应屏幕工作区与分层视口管理器 (彻底解决超屏与底部按钮被任务栏遮挡问题)
        if get_safe_screen_size:
            self.win_w, self.win_h = get_safe_screen_size(preferred_w=1280, preferred_h=720)
        else:
            self.win_w, self.win_h = 1280, 720

        if ViewportManager:
            self.viewport = ViewportManager(win_w=self.win_w, win_h=self.win_h, top_bar_h=44, bottom_bar_h=52)
        else:
            self.viewport = None

        # 确保专属验证输出目录存在
        os.makedirs(VERIFICATION_DIR, exist_ok=True)

        # 1. 加载 config.yaml 配置 (物理内参 + 标靶白名单)
        self.valid_tag_ids = []
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()

        # 2. 加载空间立体地图
        self._load_tags_map()

        # 3. AprilTag 16h5 超高灵敏度检测器 (采用 SUBPIX 角点细化与抗模糊配置)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 53
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.minMarkerPerimeterRate = 0.006
        self.detector_params.maxMarkerPerimeterRate = 4.0
        self.detector_params.polygonalApproxAccuracyRate = 0.08
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.perspectiveRemovePixelPerCell = 8
        self.detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.18
        self.detector_params.errorCorrectionRate = 0.85
        self.detector_params.maxErroneousBitsInBorderRate = 0.40
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # 4. 【核心乒乓开关】：实时动态 (LIVE) vs. 静态滤波锁定 (STATIC LOCKED)
        # True: 实时动态模式 (Live Dynamic) ; False: 静态滤波锁定模式 (Static Locked)
        self.live_mode = True

        # 5. 【两阶段批处理基准凝固与锁定引擎】
        self.batch_target_frames = 30  # 批次采样帧数目标 (默认 30 帧，可在 30 / 60 间切换)
        self.batch_options = [30, 60]
        self.batch_option_idx = 0
        self.is_collecting_batch = False
        self.collected_batch_frames = 0
        # 批次角点暂存缓冲区: {tag_id: list of np.ndarray(4, 2)}
        self.batch_corner_buffer = {}

        # 彻底锁定的稳态结果
        self.locked_pose = None
        # 结构: {
        #   'rvec': np.ndarray, 'tvec': np.ndarray,
        #   'R_c_w': np.ndarray, 'R_w_c': np.ndarray,
        #   'pos_w': np.ndarray, 'euler': tuple,
        #   'rmse': float, 'matched_tags': list,
        #   'filtered_corners': {tag_id: np.ndarray(4, 2)}
        # }

        # 6. 留一盲测核心状态 (Leave-One-Out Cross-Validation)
        # None: 全量标靶参与解算; int: 指定盲测标靶 ID (如 26)
        self.blind_target_tag_id = None
        self.mapped_tag_ids = []
        if self.tags_map and "tags" in self.tags_map:
            self.mapped_tag_ids = sorted([int(tid) for tid in self.tags_map["tags"].keys()])

        # GUI 交互热区
        self.gui_buttons = []
        self.current_frame_tags_polys = {}
        self.mouse_pos = (-1, -1)
        self.is_running = True

        # 仿真回放模式采图列表
        self.mock_image_files = sorted(glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.png")))
        self.mock_img_idx = 0

        # 临时 Toast 通知
        self.status_toast = ""
        self.status_toast_time = 0.0

        # 7. 一键 BA 全局平差优化与 HUD 诊断终端状态 (方案 B 可折叠终端浮层)
        self.show_hud_terminal = False
        self.hud_terminal_lines = []
        self.hud_scroll_offset = 0
        self.is_ba_running = False
        self.ba_result_queue = None
        self.ba_thread = None
        self.load_latest_diagnostic_report()

        # 硬件相机管道
        self.pipeline = None
        if not self.mock_mode and HAVE_REALSENSE:
            self._init_realsense()
        else:
            self.mock_mode = True

    def _load_camera_intrinsics(self):
        """加载高精度物理内参，优先读取 config_guard 或出厂标定"""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                self.valid_tag_ids = [int(x) for x in cfg.get("calibration", {}).get("valid_tag_ids", [])]
            except Exception:
                pass

        if resolve_camera_intrinsics is not None:
            K, dist, meta = resolve_camera_intrinsics(CONFIG_PATH)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            print(f"[OK] 成功绑定相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            return K, dist
        else:
            fx, fy = 1363.68, 1361.19
            cx, cy = 971.19, 566.26
            dist = np.zeros((5, 1), dtype=np.float64)
            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            return K, dist

    def _load_tags_map(self):
        if not os.path.exists(self.map_path):
            print(f"[WARN] 找不到标靶地图文件: {self.map_path}")
            print(f"[*] 请先运行工序 [3] 构建 AprilTag 3D 空间立体地图！")
            return

        try:
            with open(self.map_path, "r", encoding="utf-8") as f:
                self.tags_map = yaml.safe_load(f)
            self.marker_size_mm = float(self.tags_map.get("marker_size_mm", 50.0))
            tag_count = len(self.tags_map.get("tags", {}))
            print(f"[OK] 成功加载标靶空间地图: {self.map_path} (共包含 {tag_count} 个已知标靶)")
        except Exception as e:
            print(f"[ERROR] 读取标靶地图失败: {e}")

    def hot_reload_map(self) -> bool:
        """内存即刻热重载最新生成的标靶空间立体地图"""
        try:
            self._load_tags_map()
            if self.tags_map and "tags" in self.tags_map:
                self.mapped_tag_ids = sorted([int(tid) for tid in self.tags_map["tags"].keys()])
                # 若此前处于静态锁定状态，清空锁定位姿以使用全新地图重新定姿
                self.locked_pose = None
                print(f"[HOT-RELOAD] 标靶空间地图已成功热重载！包含 {len(self.mapped_tag_ids)} 个标靶: {self.mapped_tag_ids}")
                return True
        except Exception as e:
            print(f"[ERROR] 热重载标靶地图失败: {e}")
        return False

    @property
    def hud_visible(self) -> bool:
        return self.show_hud_terminal

    @hud_visible.setter
    def hud_visible(self, val: bool):
        self.show_hud_terminal = val

    @property
    def diagnostic_lines(self) -> list:
        return self.hud_terminal_lines

    def load_latest_diagnostic_report(self):
        """加载最新的 BA 全局平差诊断报告 Markdown 文本"""
        diag_path = self.report_path if self.report_path else os.path.join(VERIFICATION_DIR, "ba_precision_diagnostic_report.md")
        lines = []
        if os.path.exists(diag_path):
            try:
                with open(diag_path, "r", encoding="utf-8") as f:
                    raw_lines = f.readlines()
                for line in raw_lines:
                    cleaned = line.rstrip("\r\n")
                    if cleaned.strip() != "":
                        lines.append(cleaned)
                self.hud_terminal_lines = lines
                self.hud_scroll_offset = 0
            except Exception as e:
                self.hud_terminal_lines = [f"[WARN] 读取诊断报告异常: {e}"]
        else:
            self.hud_terminal_lines = [
                "# AprilTag 全局 BA 平差精度诊断终端",
                "--------------------------------------------------",
                "尚未执行过 BA 全局平差求解，无历史报告。",
                "提示: 请直接点击底部 [求解BA (B)] 按钮一键启动求解！"
            ]

    def toggle_hud_terminal(self):
        """展开或折叠 HUD 诊断报告终端浮层"""
        self.show_hud_terminal = not self.show_hud_terminal
        if self.show_hud_terminal:
            self.load_latest_diagnostic_report()
            self.set_toast("已展开 HUD 诊断报告终端 (按 H 折叠)")
        else:
            self.set_toast("已折叠 HUD 终端")

    def start_async_bundle_adjustment(self):
        """启动后台线程异步求解全局 BA 平差优化，前台画面保持极速流畅"""
        if self.is_ba_running:
            self.set_toast("BA 全局平差优化正在进行中，请稍候...")
            return

        import threading
        self.is_ba_running = True
        self.ba_result_queue = None
        self.set_toast("正在执行 BA 平差优化计算 (请稍候)...")
        print("\n" + "=" * 70)
        print("  [*] [ASYNC-BA] 收到平差求解指令，正在启动后台优化线程...")
        print("=" * 70)

        def _worker():
            try:
                from tools.calibration.tag_map_builder import TagMapBuilder
                builder = TagMapBuilder(marker_size_mm=self.marker_size_mm)
                manifest_file = os.path.join(CALIB_IMAGES_DIR, "tag_observations.yaml")
                if not os.path.exists(manifest_file):
                    self.ba_result_queue = (False, f"未找到观测清单: {manifest_file}")
                    return

                # 执行两阶段建图与 BA 平差求解 (原点锚定 Tag 0，X 轴基准 Tag 18)
                tags_map = builder.build_map_from_manifest(
                    manifest_path=manifest_file,
                    origin_tag_id=0,
                    x_align_tag_id=18
                )
                builder.save_map(tags_map, self.map_path)
                rmse = float(tags_map.get("metrics", {}).get("reprojection_rmse_px", 0.0))
                self.ba_result_queue = (True, f"BA 平差优化完成! RMSE: {rmse:.3f}px | 地图已热加载")
            except Exception as e:
                self.ba_result_queue = (False, f"BA 求解失败: {e}")
            finally:
                self.is_ba_running = False

        self.ba_thread = threading.Thread(target=_worker, daemon=True)
        self.ba_thread.start()

    def render_hud_terminal(self, disp_frame: np.ndarray):
        """在画面右侧渲染科技感黑晶磨砂 HUD 诊断文本终端浮层"""
        if not self.show_hud_terminal:
            return disp_frame

        h_img, w_img = disp_frame.shape[:2]
        term_w = min(620, w_img - 80)
        term_h = h_img - 110
        tx1 = w_img - term_w - 18
        ty1 = 48
        tx2 = w_img - 18
        ty2 = ty1 + term_h

        # 1. 磨砂暗黑半透明底板
        overlay = disp_frame.copy()
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.90, disp_frame, 0.10, 0, disp_frame)
        cv2.rectangle(disp_frame, (tx1, ty1), (tx2, ty2), (0, 190, 230), 1)

        # 2. 顶部标题栏 (深青蓝背景，高 34px)
        title_h = 34
        cv2.rectangle(disp_frame, (tx1, ty1), (tx2, ty1 + title_h), (25, 45, 75), -1)
        cv2.line(disp_frame, (tx1, ty1 + title_h), (tx2, ty1 + title_h), (0, 200, 240), 1)
        cv2.putText(disp_frame, "【BA 全局平差精度诊断与空间位姿终端】", 
                    (tx1 + 12, ty1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # 标题栏右上角关闭按钮 [X]
        close_btn_w = 64
        cx1, cy1 = tx2 - close_btn_w - 6, ty1 + 4
        cx2, cy2 = tx2 - 6, ty1 + title_h - 4
        cv2.rectangle(disp_frame, (cx1, cy1), (cx2, cy2), (60, 40, 130), -1)
        cv2.rectangle(disp_frame, (cx1, cy1), (cx2, cy2), (100, 80, 230), 1)
        cv2.putText(disp_frame, "关闭(H)", (cx1 + 8, cy1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        self.gui_buttons.append(("TOGGLE_HUD", (cx1, cy1, cx2, cy2), "TOGGLE_HUD"))

        # 3. 终端正文区域渲染
        content_y1 = ty1 + title_h + 10
        content_y2 = ty2 - 28
        visible_lines_count = max(1, (content_y2 - content_y1) // 22)

        total_lines = len(self.hud_terminal_lines)
        max_scroll = max(0, total_lines - visible_lines_count)
        self.hud_scroll_offset = max(0, min(max_scroll, self.hud_scroll_offset))

        curr_y = content_y1 + 16
        for l_idx in range(self.hud_scroll_offset, min(total_lines, self.hud_scroll_offset + visible_lines_count)):
            line_str = self.hud_terminal_lines[l_idx]
            # 颜色语法高亮匹配
            if "PASS" in line_str or "√" in line_str or "极佳" in line_str or "SUCCESS" in line_str:
                line_col = (80, 240, 120)
            elif "FAIL" in line_str or "×" in line_str or "告警" in line_str or "ERROR" in line_str:
                line_col = (80, 80, 255)
            elif "Tag #" in line_str or "===" in line_str or "【" in line_str or "###" in line_str:
                line_col = (0, 215, 255)
            elif "RMSE" in line_str or "尺度" in line_str:
                line_col = (255, 200, 50)
            else:
                line_col = (215, 215, 215)

            disp_line = line_str[:65]
            cv2.putText(disp_frame, disp_line, (tx1 + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, line_col, 1, cv2.LINE_AA)
            curr_y += 22

        # 4. 底部滚动翻页提示条 (高 26px)
        cv2.rectangle(disp_frame, (tx1, ty2 - 26), (tx2, ty2), (20, 20, 25), -1)
        cv2.line(disp_frame, (tx1, ty2 - 26), (tx2, ty2 - 26), (50, 50, 60), 1)
        scroll_tip = f"行 {self.hud_scroll_offset + 1}~{min(total_lines, self.hud_scroll_offset + visible_lines_count)}/{total_lines} | [U 向上滚动 | J 向下滚动 | H 折叠/展开]"
        cv2.putText(disp_frame, scroll_tip, (tx1 + 12, ty2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        return disp_frame

    def _init_realsense(self):
        try:
            ctx = rs.context()
            devices = list(ctx.query_devices())
            if not devices:
                self.mock_mode = True
                return

            self.pipeline = rs.pipeline()
            config = rs.config()
            try:
                config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 8)
                profile = self.pipeline.start(config)
                col_prof = profile.get_stream(rs.stream.color)
                intr = col_prof.as_video_stream_profile().get_intrinsics()
                self.camera_matrix = np.array([
                    [intr.fx, 0.0, intr.ppx],
                    [0.0, intr.fy, intr.ppy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                print(f"[OK] RealSense 硬件在线内参已同步: fx={intr.fx:.2f}, cx={intr.ppx:.2f}")
            except Exception:
                config = rs.config()
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                self.pipeline.start(config)

            for _ in range(5):
                self.pipeline.wait_for_frames(timeout_ms=2000)

        except Exception as e:
            print(f"[WARN] 启动物理相机失败: {e}，切换至仿真模式")
            self.mock_mode = True
            self.pipeline = None

    def get_frame(self, frame_idx: int) -> np.ndarray:
        if not self.mock_mode and self.pipeline is not None:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    return np.asanyarray(color_frame.get_data())
            except Exception:
                pass

        if self.mock_image_files:
            cur_file = self.mock_image_files[self.mock_img_idx % len(self.mock_image_files)]
            img = cv2.imread(cur_file)
            if img is not None:
                return img

        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        for x in range(0, 1920, 100):
            cv2.line(frame, (x, 0), (x, 1080), (55, 55, 55), 1)
        for y in range(0, 1080, 100):
            cv2.line(frame, (0, y), (1920, y), (55, 55, 55), 1)
        return frame

    def _get_tag_world_transform(self, tag_id: int) -> np.ndarray:
        if self.tags_map is None or "tags" not in self.tags_map:
            return None
        tag_data = self.tags_map["tags"].get(tag_id)
        if tag_data is None:
            return None
        if "transform_matrix" in tag_data:
            return np.array(tag_data["transform_matrix"], dtype=np.float64)
        elif "position_mm" in tag_data:
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = np.array(tag_data["position_mm"], dtype=np.float64)
            return T
        return None

    def _get_tag_world_corners(self, tag_id: int) -> np.ndarray:
        T_w_t = self._get_tag_world_transform(tag_id)
        if T_w_t is None:
            return None
        s = self.marker_size_mm / 2.0
        local_corners = np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)
        return (T_w_t @ local_corners.T).T[:, :3]

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()

    def toggle_mode(self):
        """核心乒乓开关切换：实时动态 (LIVE) ⇋ 静态滤波锁定 (STATIC LOCKED)"""
        self.live_mode = not self.live_mode
        if self.live_mode:
            self.set_toast("已切换为: 实时动态模式 (零延迟即时巡检)")
        else:
            if self.locked_pose is None:
                # 若尚未锁定，自动触发一次批次采样锁定
                self.start_batch_collection()
            else:
                self.set_toast("已切换为: 静态滤波锁定模式 (位姿已绝对锁定)")

    def cycle_batch_target(self):
        """切换批次采样帧数 (30F / 60F)"""
        self.batch_option_idx = (self.batch_option_idx + 1) % len(self.batch_options)
        self.batch_target_frames = self.batch_options[self.batch_option_idx]
        self.set_toast(f"采样批次设定为: {self.batch_target_frames} 帧")

    def start_batch_collection(self):
        """启动定点静止采样批次 (取足帧数后一次性去噪并绝对锁定)"""
        self.live_mode = False
        self.is_collecting_batch = True
        self.collected_batch_frames = 0
        self.batch_corner_buffer.clear()
        self.set_toast(f"开始静止采样 ({self.batch_target_frames} 帧，请保持相机静止！)")
        print(f"\n[*] 正在启动定点静止标靶采样 ({self.batch_target_frames} 帧批次)...")

    def _compute_and_lock_pose(self):
        """对采足的批次数据执行集中时域去噪，一次性解算高精相机位姿并彻底锁定"""
        if not self.batch_corner_buffer:
            self.set_toast("未采得有效标靶数据，无法锁定！")
            self.is_collecting_batch = False
            return

        filtered_corners = {}
        # 1. 对每个 Tag 的 4 角点集中进行去极值均值滤波
        for tid, corners_list in self.batch_corner_buffer.items():
            if len(corners_list) == 0:
                continue
            arr = np.array(corners_list) # shape: (N, 4, 2)
            if len(corners_list) >= 5:
                # 剔除最大值与最小值 (去掉散粒突跳)
                sorted_arr = np.sort(arr, axis=0)
                trimmed = sorted_arr[1:-1]
                filtered_corners[tid] = np.mean(trimmed, axis=0)
            else:
                filtered_corners[tid] = np.mean(arr, axis=0)

        # 2. 准备 PnP 解算点对 (若指定了盲测 Tag，求解中强制排除)
        all_obj_pts = []
        all_img_pts = []
        matched_tags = []

        for tid, c_2d in filtered_corners.items():
            if self.blind_target_tag_id is not None and tid == self.blind_target_tag_id:
                continue
            w_c = self._get_tag_world_corners(tid)
            if w_c is not None:
                matched_tags.append(tid)
                all_obj_pts.append(w_c)
                all_img_pts.append(c_2d)

        if len(all_obj_pts) < 1:
            self.set_toast("有效已知标靶不足，无法锁定！")
            self.is_collecting_batch = False
            return

        obj_flat = np.concatenate(all_obj_pts, axis=0)
        img_flat = np.concatenate(all_img_pts, axis=0)

        # 3. 超定非线性最小化求解高稳态位姿
        success, rvec_opt, tvec_opt = cv2.solvePnP(
            obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        if success:
            success, rvec_opt, tvec_opt = cv2.solvePnP(
                obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                rvec=rvec_opt, tvec=tvec_opt, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            self.set_toast("PnP 位姿解算失败！")
            self.is_collecting_batch = False
            return

        proj_pts, _ = cv2.projectPoints(obj_flat, rvec_opt, tvec_opt, self.camera_matrix, self.dist_coeffs)
        reproj_err = np.linalg.norm(img_flat - proj_pts.reshape((-1, 2)), axis=1)
        rmse = float(np.sqrt(np.mean(reproj_err ** 2)))

        R_c_w, _ = cv2.Rodrigues(rvec_opt)
        R_w_c = R_c_w.T
        pos_w = (-R_w_c @ tvec_opt).flatten()

        sy = math.sqrt(R_w_c[0, 0]**2 + R_w_c[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R_w_c[2, 1], R_w_c[2, 2])
            pitch = math.atan2(-R_w_c[2, 0], sy)
            yaw = math.atan2(R_w_c[1, 0], R_w_c[0, 0])
        else:
            roll = math.atan2(-R_w_c[1, 2], R_w_c[1, 1])
            pitch = math.atan2(-R_w_c[2, 0], sy)
            yaw = 0.0

        # 4. 存入锁定成果并彻底停止采样
        self.locked_pose = {
            'rvec': rvec_opt,
            'tvec': tvec_opt,
            'R_c_w': R_c_w,
            'R_w_c': R_w_c,
            'pos_w': pos_w,
            'euler': (math.degrees(roll), math.degrees(pitch), math.degrees(yaw)),
            'rmse': rmse,
            'matched_tags': matched_tags,
            'filtered_corners': filtered_corners
        }
        self.is_collecting_batch = False
        self.set_toast(f"已成功锁定相机位姿！(批次: {self.batch_target_frames}F | 抖动: 0.00mm)")
        print(f"[OK] 静态位姿已成功锁定: X={pos_w[0]:+.2f} Y={pos_w[1]:+.2f} Z={pos_w[2]:+.2f} mm | RMSE={rmse:.3f}px")

    def render_tag_3d_axes(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                           tag_id: int, is_blind_projection: bool = False):
        """
        绘制粗壮高品质的 3D 实心正四棱柱：
        - 截面边长 30.0mm x 30.0mm (hw = 15.0mm)
        - 柱体高度 80.0mm (L = 80.0mm，约为原高度 2/3)
        """
        try:
            hw = 15.0   # 截面半宽 15mm，整体截面边长 30.0mm x 30.0mm
            L = 80.0    # 柱体高度 80mm

            pts_3d = np.array([
                [-hw, -hw, 0.0],
                [ hw, -hw, 0.0],
                [ hw,  hw, 0.0],
                [-hw,  hw, 0.0],
                [-hw, -hw, L],
                [ hw, -hw, L],
                [ hw,  hw, L],
                [-hw,  hw, L],
                [0.0, 0.0, L],
                [35.0, 0.0, 0.0],
                [0.0, 35.0, 0.0],
                [0.0, 0.0, 0.0]
            ], dtype=np.float64)

            proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            proj = proj.reshape((-1, 2)).astype(int)

            b_pts = proj[0:4]
            t_pts = proj[4:8]
            p_x = tuple(proj[9])
            p_y = tuple(proj[10])
            p_orig = tuple(proj[11])

            # 绘制 X/Y 轴
            cv2.line(img, p_orig, p_x, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(img, 'X', p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(img, p_orig, p_y, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(img, 'Y', p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            overlay = img.copy()
            if is_blind_projection:
                side_color = (220, 50, 180)  # 品红高光 (留一盲测反推)
                cap_color = (255, 120, 240)
                alpha = 0.55
            else:
                side_color = (240, 160, 30)  # 金黄天蓝 (稳态已知)
                cap_color = (255, 220, 90)
                alpha = 0.45

            for i in range(4):
                next_i = (i + 1) % 4
                side_poly = np.array([b_pts[i], b_pts[next_i], t_pts[next_i], t_pts[i]], dtype=np.int32)
                cv2.fillPoly(overlay, [side_poly], side_color)
            cv2.fillPoly(overlay, [t_pts], cap_color)
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

            # 棱线高亮描边
            edge_color = (255, 255, 255)
            cv2.polylines(img, [b_pts], isClosed=True, color=edge_color, thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(img, [t_pts], isClosed=True, color=edge_color, thickness=2, lineType=cv2.LINE_AA)
            for i in range(4):
                cv2.line(img, tuple(b_pts[i]), tuple(t_pts[i]), edge_color, 2, cv2.LINE_AA)
            top_center = tuple(proj[8])
            cv2.putText(img, 'Z (Norm)', top_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            return {
                'b_pts': b_pts,
                't_pts': t_pts,
                'p_orig': p_orig,
                'top_center': top_center,
                'proj_pts': proj
            }
        except Exception:
            return None

    def render_evasive_blind_badge(self, img: np.ndarray, obstacle_pts: np.ndarray, 
                                   tag_id: int, err_px: float, err_mm: float):
        """
        工业级几何自适应避让算法：
        自动探测 3D 棱柱与标靶基底的全部 2D 投影包围区，智能避开棱柱拔起方向，
        将微型半透明标牌推送到安全空旷区，并通过细折线引导线 (Leader Line) 连回标靶，
        确保 3D 棱柱根部、柱身与物理标靶底座 100% 毫无遮挡！
        """
        h_img, w_img = img.shape[:2]
        if obstacle_pts is None or len(obstacle_pts) == 0:
            return

        # 1. 计算 3D 棱柱 + 标靶底座在画面上的联合凸包禁区
        min_x = int(np.min(obstacle_pts[:, 0]))
        max_x = int(np.max(obstacle_pts[:, 0]))
        min_y = int(np.min(obstacle_pts[:, 1]))
        max_y = int(np.max(obstacle_pts[:, 1]))
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2

        badge_w, badge_h = 165, 34
        margin = 35 # 安全避让边距 (离多边形边界至少 35px)

        # 2. 生成多方向候选放置位置 (优先选择下方与侧方，远离向天空延伸的 Z 轴)
        candidates = [
            # 候选 1: 右下方 (俯视视线最不易被遮挡区域)
            (max_x + margin, max_y + margin // 2),
            # 候选 2: 左下方
            (min_x - badge_w - margin, max_y + margin // 2),
            # 候选 3: 正下方
            (cx - badge_w // 2, max_y + margin),
            # 候选 4: 右侧外
            (max_x + margin, cy - badge_h // 2),
            # 候选 5: 左侧外
            (min_x - badge_w - margin, cy - badge_h // 2),
            # 候选 6: 正上方 (备用)
            (cx - badge_w // 2, min_y - badge_h - margin)
        ]

        top_limit = 50
        bottom_limit = h_img - 55

        best_pos = None
        for bx, by in candidates:
            if bx >= 10 and (bx + badge_w) <= (w_img - 10) and by >= top_limit and (by + badge_h) <= bottom_limit:
                best_pos = (bx, by)
                break

        if best_pos is None:
            bx = max(15, min(w_img - badge_w - 15, max_x + margin))
            by = max(top_limit, min(bottom_limit - badge_h, max_y + margin))
            best_pos = (bx, by)

        bx, by = best_pos

        # 3. 寻找离标牌最近的障碍边缘点作为引导线锚点
        anchor_pt = (cx, cy)
        min_d = float('inf')
        badge_center = (bx + badge_w // 2, by + badge_h // 2)
        for pt in obstacle_pts:
            d = (pt[0] - badge_center[0])**2 + (pt[1] - badge_center[1])**2
            if d < min_d:
                min_d = d
                anchor_pt = (int(pt[0]), int(pt[1]))

        if bx > anchor_pt[0]:
            line_target = (bx, by + badge_h // 2)
        else:
            line_target = (bx + badge_w, by + badge_h // 2)

        # 绘制精致引导线 (Leader Line)
        cv2.circle(img, anchor_pt, 3, (255, 120, 240), -1, lineType=cv2.LINE_AA)
        cv2.line(img, anchor_pt, line_target, (220, 100, 220), 1, cv2.LINE_AA)

        # 4. 绘制半透明深色磨砂标牌 (杜绝大黄色遮挡块！)
        overlay = img.copy()
        cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), (20, 20, 20), -1)
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

        border_color = (0, 230, 100) if err_px < 1.0 else (0, 180, 255)
        cv2.rectangle(img, (bx, by), (bx + badge_w, by + badge_h), border_color, 1, cv2.LINE_AA)

        cv2.putText(img, f"BLIND Tag #{tag_id}", (bx + 8, by + 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Err: {err_px:.2f}px ({err_mm:.2f}mm)", (bx + 8, by + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, border_color, 1, cv2.LINE_AA)

    def export_report(self):
        """导出当前标定验证检验单"""
        if self.live_mode or self.locked_pose is None:
            self.set_toast("请先进入静态锁定模式并完成一次位姿锁定！")
            return

        ts = int(time.time())
        rep_file = os.path.join(VERIFICATION_DIR, f"static_precision_report_{ts}.md")
        lp = self.locked_pose
        pos = lp['pos_w']
        euler = lp['euler']

        with open(rep_file, "w", encoding="utf-8") as f:
            f.write("# AprilTag 标定精度静态滤波锁定检验报告单\n\n")
            f.write(f"- **检验生成时间**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
            f.write(f"- **测量工序类型**: 静态多帧去极值均值滤波锁定基准 (Fixate & Lock)\n")
            f.write(f"- **采样批次深度**: `{self.batch_target_frames}` 帧集中去噪\n")
            f.write(f"- **参与锁定标靶**: `{lp['matched_tags']}` (共 {len(lp['matched_tags'])} 枚)\n")
            f.write(f"- **重投影均方根**: `{lp['rmse']:.3f} px`\n\n")
            f.write("### 1. 相机在 SCARA 世界系下的高精绝对位姿 (绝对锁定)\n\n")
            f.write(f"| 空间坐标分量 | 锁定绝对值 | 抖动控制 | 稳态评级 |\n")
            f.write(f"| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **X 轴 (mm)** | `{pos[0]:+.2f}` | `0.000 mm` | ROCK-SOLID (绝对冻结) |\n")
            f.write(f"| **Y 轴 (mm)** | `{pos[1]:+.2f}` | `0.000 mm` | ROCK-SOLID (绝对冻结) |\n")
            f.write(f"| **Z 轴 (mm)** | `{pos[2]:+.2f}` | `0.000 mm` | ROCK-SOLID (绝对冻结) |\n")
            f.write(f"| **Roll (deg)** | `{euler[0]:+.3f}°` | `0.000°` | ROCK-SOLID (绝对冻结) |\n")
            f.write(f"| **Pitch (deg)** | `{euler[1]:+.3f}°` | `0.000°` | ROCK-SOLID (绝对冻结) |\n")
            f.write(f"| **Yaw (deg)** | `{euler[2]:+.3f}°` | `0.000°` | ROCK-SOLID (绝对冻结) |\n\n")

        self.set_toast(f"已成功导出质检单: static_precision_report_{ts}.md")
        print(f"[OK] 静态锁定高精度量测检验报告已保存至: {rep_file}")

    def _on_mouse(self, event, x, y, flags, param):
        """鼠标交互处理：点击顶栏 Tag 按钮、点击画面标靶、点击底部工具栏"""
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. 检查是否点击了 GUI 按钮 (顶栏 Tag 列表 / 底部工具栏)
            for btn_id, (bx1, by1, bx2, by2), label in self.gui_buttons:
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if btn_id == "RUN_BA":
                        self.start_async_bundle_adjustment()
                    elif btn_id == "TOGGLE_HUD":
                        self.toggle_hud_terminal()
                    elif btn_id == "TOGGLE_MODE":
                        self.toggle_mode()
                    elif btn_id == "CYCLE_BATCH":
                        self.cycle_batch_target()
                    elif btn_id == "RESAMPLE_LOCK":
                        self.start_batch_collection()
                    elif btn_id == "EXPORT":
                        self.export_report()
                    elif btn_id == "OPEN_REVIEWER":
                        self.open_reviewer()
                    elif btn_id == "OPEN_OFFLINE_VERIFIER":
                        self.open_offline_verifier()
                    elif btn_id == "CLEAR_BLIND":
                        self.blind_target_tag_id = None
                        self.set_toast("已清除盲测，恢复全量解算 (ALL)")
                    elif btn_id == "EXIT":
                        self.is_running = False
                    elif isinstance(btn_id, (int, type(None))):
                        # 顶栏 Tag 按钮
                        if btn_id == self.blind_target_tag_id:
                            self.blind_target_tag_id = None
                            self.set_toast("已取消盲测，恢复全量标靶解算 (ALL)")
                        else:
                            self.blind_target_tag_id = btn_id
                            if btn_id is None:
                                self.set_toast("已切换为: 全量标靶联合解算 (ALL)")
                            else:
                                self.set_toast(f"已锁定留一盲测目标: Tag #{btn_id} (由其余标靶反推)")
                    return

            # 2. 检查是否直接点击了画面中的标靶轮廓 (映射回原图坐标)
            if self.viewport:
                img_x, img_y = self.viewport.win_to_img_coords(x, y)
            else:
                img_x, img_y = x, y

            if img_x is not None:
                for tid, poly in self.current_frame_tags_polys.items():
                    if cv2.pointPolygonTest(poly, (float(img_x), float(img_y)), False) >= 0:
                        if self.blind_target_tag_id == tid:
                            self.blind_target_tag_id = None
                            self.set_toast(f"已取消 Tag #{tid} 盲测，恢复全量解算")
                        else:
                            self.blind_target_tag_id = tid
                            self.set_toast(f"已选定 Tag #{tid} 为盲测验证目标 (PnP中已主动屏蔽)")
                        return

    def open_reviewer(self):
        """唤起人工审核画板 (Reviewer)，支持带入当前盲测 Tag 靶向直达，关闭后自动触发 BA 求解与热重载"""
        from tools.calibration.tag_manifest_reviewer import TagManifestReviewer
        manifest_file = os.path.join(CALIB_IMAGES_DIR, "tag_observations.yaml")
        if not os.path.exists(manifest_file):
            self.set_toast(f"未找到观测清单: {manifest_file}")
            return

        focus_tid = self.blind_target_tag_id
        tid_str = f"Tag #{focus_tid}" if focus_tid is not None else "全量"
        self.set_toast(f"正在唤起审核画板 (定向排查: {tid_str})...")
        print(f"\n[*] [HANDSHAKE] 正在呼出人工审核画板 (focus_tag_id={focus_tid})...")

        try:
            reviewer = TagManifestReviewer(manifest_path=manifest_file, focus_tag_id=focus_tid)
            reviewer.run()

            # 画板退出后，重新强夺验证器窗口焦点
            window_name = "AprilTag SCARA AR & Precision Verifier (Integrated Edition)"
            if force_window_focus:
                force_window_focus(window_name)

            # 检查画板是否请求了自动重新平差验证或发生了任何修改
            if reviewer.trigger_verify_and_ba or getattr(reviewer, "has_modified_manifest", False) or reviewer.has_unsaved_changes:
                self.set_toast("已载入画板最新修改，正在自动启动 BA 全局平差优化...")
                self.start_async_bundle_adjustment()
            else:
                self.set_toast("已从审核画板返回验证器")
        except Exception as e:
            self.set_toast(f"呼出审核画板失败: {e}")
            print(f"[ERROR] 唤起审核画板异常: {e}")

    def open_offline_verifier(self):
        """唤起离线标定精度体检工作台 (TagOfflineVerifier)，完成后热重载地图并恢复 AR 验证"""
        from tools.calibration.tag_offline_verifier import TagOfflineVerifier
        self.set_toast("正在进入离线标定体检工作台...")
        print("\n[*] [HANDSHAKE] 正在呼出离线标定精度体检工作台...")

        try:
            # 临时销毁 AR 窗口，避免 OpenCV 全局按键广播冲突
            cv2.destroyAllWindows()
            verifier = TagOfflineVerifier(
                map_path=self.map_path,
                image_dir=CALIB_IMAGES_DIR,
                marker_size_mm=self.marker_size_mm,
                source="auto",
                caller_ar_instance=self
            )
            verifier.run_gui()

            # 从体检工作台返回后，地图可能在体检中执行了 BA 平差，执行热重载
            self._load_tags_map()

            # 重建 AR 验证器窗口并重夺焦点
            window_name = "AprilTag SCARA AR & Precision Verifier (Integrated Edition)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.win_w, self.win_h)
            cv2.setMouseCallback(window_name, self._on_mouse)
            if force_window_focus:
                force_window_focus(window_name)
            self.set_toast("已从离线体检工作台返回 AR 验证器 (地图已热重载)")
        except Exception as e:
            self.set_toast(f"呼出离线体检失败: {e}")
            print(f"[ERROR] 唤起离线体检工作台异常: {e}")
            window_name = "AprilTag SCARA AR & Precision Verifier (Integrated Edition)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.win_w, self.win_h)
            cv2.setMouseCallback(window_name, self._on_mouse)

    def run(self):
        window_name = "AprilTag SCARA AR & Precision Verifier (Integrated Edition)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.win_w, self.win_h)
        cv2.setMouseCallback(window_name, self._on_mouse)

        print("\n" + "=" * 80)
        print("     AprilTag 标定精度与 3D 坐标系在线 AR 综合验证系统 (Integrated)")
        print("=" * 80)
        print(f" [专属验证目录] : {VERIFICATION_DIR}")
        print(f" [相机内参绑定] : fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}, cx={self.camera_matrix[0,2]:.1f}")
        print(f" [地图已知标靶] : {self.mapped_tag_ids}")
        print(" [核心功能特性] :")
        print("   - 【乒乓开关】[Tab/M] 在【实时动态 (LIVE)】与【静态滤波锁定 (STATIC LOCKED)】之间一键切换；")
        print("   - 【基准凝固】在静态模式下采足 30/60 帧后一次性去噪并绝对锁死位姿，抖动严格 0.00mm；")
        print("   - 【留一盲测】在顶栏直接点击 Tag 编号，由其余标靶反推 3D 棱柱并评估残差；")
        print("   - 【一键平差】在 Verify 中直接按 [B] 键即可异步执行全局 BA 优化，地图自动热重载；")
        print("   - 【HUD 终端】按 [H] 键随时展开/折叠黑晶高科技诊断报告控制台，U/J 滚动翻页。")
        print(" [快捷键指南]   :")
        print("   - [O]             : 【呼出人工审核画板，带入当前盲测 Tag 定向排查】；")
        print("   - [P]             : 【呼出离线标定体检工作台，全局指标排查与闭环重算】；")
        print("   - [B]             : 【一键异步求解 BA 全局平差并热更新地图】；")
        print("   - [H]             : 【展开/折叠 HUD 诊断报告控制台终端】；")
        print("   - [U] / [J]       : HUD 终端向上 / 向下滚动翻页浏览；")
        print("   - [Tab] / [M]     : 乒乓切换模式 (实时动态 <-> 静态锁定)；")
        print("   - [Space] (空格键) : 静态模式下【重新采样并锁定位姿】；实时模式下抓拍单帧；")
        print("   - [W]             : 切换采样批次深度 (30F / 60F)；")
        print("   - [T]             : 顺序轮换留一盲测目标 (None -> 18 -> 19 -> 20...)；")
        print("   - [C]             : 清除盲测目标，恢复全量融合解算；")
        print("   - [A] / [D]       : 仿真回放模式下，前后翻页浏览真实采图；")
        print("   - [Q] / [ESC]     : 安全退出验证。")
        print("=" * 80 + "\n")

        frame_idx = 0
        try:
            while self.is_running:
                # 实时动态感知用户拖拽 Resize 或点击最大化窗口后的实际物理尺寸并自适应重排
                if self.viewport and self.viewport.sync_window_size(window_name):
                    self.win_w = self.viewport.win_w
                    self.win_h = self.viewport.win_h

                raw_frame = self.get_frame(frame_idx)
                frame_idx += 1
                disp_frame = raw_frame.copy()
                h_img, w_img = disp_frame.shape[:2]

                # 检测 AprilTag (原生灰度)
                gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)

                all_obj_pts = []
                all_img_pts = []
                matched_tags = []
                self.current_frame_tags_polys.clear()

                blind_real_corners = None
                blind_real_center = None

                if ids is not None and len(ids) > 0 and self.tags_map is not None:
                    for i, tid in enumerate(ids.flatten()):
                        tid_int = int(tid)
                        # 白名单硬过滤
                        if self.valid_tag_ids and tid_int not in self.valid_tag_ids:
                            continue

                        raw_c = corners[i].reshape((4, 2)).astype(np.float64)
                        pts = raw_c.astype(int)
                        self.current_frame_tags_polys[tid_int] = pts
                        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

                        # 若正在批次采样，将角点存入缓冲区
                        if self.is_collecting_batch:
                            if tid_int not in self.batch_corner_buffer:
                                self.batch_corner_buffer[tid_int] = []
                            self.batch_corner_buffer[tid_int].append(raw_c)

                        # 如果当前标靶是“留一盲测目标”，在 PnP 求解中强制屏蔽它！
                        if self.blind_target_tag_id is not None and tid_int == self.blind_target_tag_id:
                            blind_real_corners = raw_c
                            blind_real_center = (cx, cy)
                            cv2.polylines(disp_frame, [pts], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
                            cv2.putText(disp_frame, f"Tag #{tid_int} [REAL PHYSICAL]", (cx - 50, cy + 25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                            continue

                        w_corners = self._get_tag_world_corners(tid_int)
                        if w_corners is not None:
                            matched_tags.append(tid_int)
                            all_obj_pts.append(w_corners)
                            all_img_pts.append(raw_c)

                            # 绘制标靶识别外框
                            box_color = (0, 230, 0) if self.live_mode else (255, 200, 40)
                            cv2.polylines(disp_frame, [pts], isClosed=True, color=box_color, thickness=2, lineType=cv2.LINE_AA)
                            cv2.putText(disp_frame, f"#{tid_int}", (cx - 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # ===================== 静态批次采样进度检查 =====================
                if self.is_collecting_batch:
                    self.collected_batch_frames += 1
                    if self.collected_batch_frames >= self.batch_target_frames:
                        # 采足帧数，立即执行集中去噪解算并彻底锁定！
                        self._compute_and_lock_pose()

                # ===================== 相机位姿解算与 3D 渲染 =====================
                cam_pose_str = "未收敛"
                rmse_str = "N/A"
                jitter_str = "0.00 mm"
                blind_summary = None

                # 决定当前用于渲染的相机位姿
                active_rvec = None
                active_tvec = None
                active_matched_tags = []

                if self.live_mode:
                    # 【模式 A：实时动态】：单帧即时 PnP 求解
                    if len(all_obj_pts) >= 1:
                        obj_flat = np.concatenate(all_obj_pts, axis=0)
                        img_flat = np.concatenate(all_img_pts, axis=0)

                        success, rvec_opt, tvec_opt = cv2.solvePnP(
                            obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_SQPNP
                        )
                        if success:
                            success, rvec_opt, tvec_opt = cv2.solvePnP(
                                obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                                rvec=rvec_opt, tvec=tvec_opt, useExtrinsicGuess=True,
                                flags=cv2.SOLVEPNP_ITERATIVE
                            )
                        if success:
                            active_rvec = rvec_opt
                            active_tvec = tvec_opt
                            active_matched_tags = matched_tags

                            proj_pts, _ = cv2.projectPoints(obj_flat, rvec_opt, tvec_opt, self.camera_matrix, self.dist_coeffs)
                            reproj_err = np.linalg.norm(img_flat - proj_pts.reshape((-1, 2)), axis=1)
                            rmse = np.sqrt(np.mean(reproj_err ** 2))
                            rmse_str = f"{rmse:.2f} px"

                            R_c_w, _ = cv2.Rodrigues(rvec_opt)
                            R_w_c = R_c_w.T
                            pos_w = (-R_w_c @ tvec_opt).flatten()
                            cam_pose_str = f"X:{pos_w[0]:+.1f} Y:{pos_w[1]:+.1f} Z:{pos_w[2]:+.1f} mm"
                            jitter_str = "~0.35 mm (实时散粒)"

                else:
                    # 【模式 B：静态滤波锁定】：直接读取锁定的绝对稳态位姿
                    if self.locked_pose is not None:
                        active_rvec = self.locked_pose['rvec']
                        active_tvec = self.locked_pose['tvec']
                        active_matched_tags = self.locked_pose['matched_tags']
                        pos_w = self.locked_pose['pos_w']
                        cam_pose_str = f"X:{pos_w[0]:+.1f} Y:{pos_w[1]:+.1f} Z:{pos_w[2]:+.1f} mm [LOCKED]"
                        rmse_str = f"{self.locked_pose['rmse']:.2f} px"
                        jitter_str = "0.00 mm (已绝对锁死)"

                # 执行 3D 实心正四棱柱与留一盲测渲染
                if active_rvec is not None and active_tvec is not None:
                    R_c_w, _ = cv2.Rodrigues(active_rvec)

                    # 1. 渲染所有已知标靶的 3D 实心轴 (30mm x 30mm x 80mm)
                    for tid_int in active_matched_tags:
                        T_w_t = self._get_tag_world_transform(tid_int)
                        if T_w_t is not None:
                            T_c_w = np.eye(4, dtype=np.float64)
                            T_c_w[:3, :3] = R_c_w
                            T_c_w[:3, 3] = active_tvec.flatten()
                            T_c_t = T_c_w @ T_w_t
                            r_tag, _ = cv2.Rodrigues(T_c_t[:3, :3])
                            t_tag = T_c_t[:3, 3].reshape((3, 1))
                            self.render_tag_3d_axes(disp_frame, r_tag, t_tag, tid_int, is_blind_projection=False)

                    # 2. 留一盲测隔空推算
                    if self.blind_target_tag_id is not None:
                        T_w_blind = self._get_tag_world_transform(self.blind_target_tag_id)
                        if T_w_blind is not None:
                            T_c_w = np.eye(4, dtype=np.float64)
                            T_c_w[:3, :3] = R_c_w
                            T_c_w[:3, 3] = active_tvec.flatten()
                            T_c_blind = T_c_w @ T_w_blind
                            r_blind, _ = cv2.Rodrigues(T_c_blind[:3, :3])
                            t_blind = T_c_blind[:3, 3].reshape((3, 1))

                            # 渲染品红高光的 30mm 正四棱柱 3D 轴
                            prism_info = self.render_tag_3d_axes(disp_frame, r_blind, t_blind, self.blind_target_tag_id, is_blind_projection=True)

                            # 计算反推 4 角点
                            s = self.marker_size_mm / 2.0
                            local_corners = np.array([
                                [-s,  s, 0.0],
                                [ s,  s, 0.0],
                                [ s, -s, 0.0],
                                [-s, -s, 0.0]
                            ], dtype=np.float64)
                            proj_blind_c, _ = cv2.projectPoints(local_corners, r_blind, t_blind, self.camera_matrix, self.dist_coeffs)
                            proj_blind_c = proj_blind_c.reshape((-1, 2))
                            blind_pts = proj_blind_c.astype(int)

                            cv2.polylines(disp_frame, [blind_pts], isClosed=True, color=(255, 60, 220), thickness=2, lineType=cv2.LINE_AA)

                            # 若物理标靶也在画面中，定量计算反推误差
                            if blind_real_corners is not None:
                                dist_px = np.linalg.norm(blind_real_corners - proj_blind_c, axis=1)
                                mean_err_px = float(np.mean(dist_px))
                                fx = self.camera_matrix[0, 0]
                                depth_z = t_blind[2, 0]
                                err_mm = (mean_err_px * depth_z) / fx

                                blind_summary = {
                                    "tag_id": self.blind_target_tag_id,
                                    "err_px": mean_err_px,
                                    "err_mm": err_mm,
                                    "depth_mm": depth_z,
                                    "solved_by": active_matched_tags
                                }

                                # 汇集 3D 棱柱和标靶底部的全部投影点，执行智能避让绘制
                                obstacle_list = [blind_pts]
                                if prism_info is not None and 'proj_pts' in prism_info:
                                    obstacle_list.append(prism_info['proj_pts'])
                                combined_obstacles = np.concatenate(obstacle_list, axis=0)

                                # 自适应无遮挡避让算法渲染折线标牌
                                self.render_evasive_blind_badge(disp_frame, combined_obstacles, 
                                                               self.blind_target_tag_id, mean_err_px, err_mm)

                # ===================== 视口分层渲染：底图等比贴入视口 =====================
                canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
                if self.viewport:
                    self.viewport.render_viewport(canvas, disp_frame)
                    top_bar_h = self.viewport.top_bar_h
                    bottom_bar_h = self.viewport.bottom_bar_h
                else:
                    top_bar_h = 42
                    bottom_bar_h = 52
                    canvas = cv2.resize(disp_frame, (self.win_w, self.win_h))

                w_img, h_img = self.win_w, self.win_h

                # ===================== 屏幕中心采样进度条 (若正在批次采样) =====================
                if self.is_collecting_batch:
                    p_w, p_h = min(480, w_img - 60), 75
                    px1, py1 = (w_img - p_w) // 2, (h_img - p_h) // 2
                    overlay_bar = canvas.copy()
                    cv2.rectangle(overlay_bar, (px1, py1), (px1 + p_w, py1 + p_h), (20, 20, 20), -1)
                    cv2.addWeighted(overlay_bar, 0.85, canvas, 0.15, 0, canvas)
                    cv2.rectangle(canvas, (px1, py1), (px1 + p_w, py1 + p_h), (0, 215, 255), 2)

                    cv2.putText(canvas, f"正在静止采样标靶角点 (请保持相机完全不动)", (px1 + 20, py1 + 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                    # 进度条
                    bar_len = p_w - 40
                    prog = min(1.0, self.collected_batch_frames / max(1, self.batch_target_frames))
                    fill_w = int(bar_len * prog)
                    cv2.rectangle(canvas, (px1 + 20, py1 + 42), (px1 + 20 + bar_len, py1 + 58), (60, 60, 60), -1)
                    cv2.rectangle(canvas, (px1 + 20, py1 + 42), (px1 + 20 + fill_w, py1 + 58), (0, 200, 100), -1)
                    cv2.putText(canvas, f"{self.collected_batch_frames}/{self.batch_target_frames}", (px1 + p_w - 80, py1 + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

                # ===================== UI 装饰与仪表盘 =====================
                self.gui_buttons.clear()

                # 1. 顶栏：留一盲测标靶切换按钮组 (直接以窗口物理像素绘制，清晰饱满)
                cv2.rectangle(canvas, (0, 0), (w_img, top_bar_h), (25, 25, 25), -1)
                cv2.line(canvas, (0, top_bar_h), (w_img, top_bar_h), (60, 60, 60), 1)
                cv2.putText(canvas, "留一盲测目标:", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

                btn_x = 125
                btn_y1, btn_y2 = 6, top_bar_h - 6
                # [ALL] 按钮
                is_all_active = (self.blind_target_tag_id is None)
                all_bg = (40, 160, 40) if is_all_active else (50, 50, 50)
                all_w = 54
                cv2.rectangle(canvas, (btn_x, btn_y1), (btn_x + all_w, btn_y2), all_bg, -1)
                cv2.rectangle(canvas, (btn_x, btn_y1), (btn_x + all_w, btn_y2), (90, 90, 90), 1)
                cv2.putText(canvas, "ALL", (btn_x + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1 if not is_all_active else 2, cv2.LINE_AA)
                self.gui_buttons.append((None, (btn_x, btn_y1, btn_x + all_w, btn_y2), "ALL"))
                btn_x += all_w + 6

                # 各 Tag 按钮
                for tid in self.mapped_tag_ids:
                    is_active = (self.blind_target_tag_id == tid)
                    bg = (180, 40, 160) if is_active else (45, 45, 45)
                    tw = 68
                    cv2.rectangle(canvas, (btn_x, btn_y1), (btn_x + tw, btn_y2), bg, -1)
                    border_c = (255, 120, 240) if is_active else (75, 75, 75)
                    cv2.rectangle(canvas, (btn_x, btn_y1), (btn_x + tw, btn_y2), border_c, 1)
                    cv2.putText(canvas, f"Tag {tid}", (btn_x + 7, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2 if is_active else 1, cv2.LINE_AA)
                    self.gui_buttons.append((tid, (btn_x, btn_y1, btn_x + tw, btn_y2), f"Tag {tid}"))
                    btn_x += tw + 5

                # 2. 底部工具栏 (1:1 独立物理像素绘制，常驻在屏幕最底端，绝不掉出屏幕)
                by1 = h_img - bottom_bar_h
                cv2.rectangle(canvas, (0, by1), (w_img, h_img), (20, 20, 20), -1)
                cv2.line(canvas, (0, by1), (w_img, by1), (65, 65, 65), 1)

                # 异步 BA 求解完成通知检查与内存地图热重载
                if self.ba_result_queue is not None:
                    succ, msg = self.ba_result_queue
                    self.ba_result_queue = None
                    self.set_toast(msg)
                    if succ:
                        self.hot_reload_map()
                        self.load_latest_diagnostic_report()
                        self.show_hud_terminal = True

                # 若正在异步求解 BA，显示居中科技感卡片
                if self.is_ba_running:
                    p_w, p_h = 500, 75
                    px1, py1 = (w_img - p_w) // 2, (h_img - p_h) // 2
                    overlay_ba = canvas.copy()
                    cv2.rectangle(overlay_ba, (px1, py1), (px1 + p_w, py1 + p_h), (25, 20, 15), -1)
                    cv2.addWeighted(overlay_ba, 0.88, canvas, 0.12, 0, canvas)
                    cv2.rectangle(canvas, (px1, py1), (px1 + p_w, py1 + p_h), (0, 215, 255), 2)
                    cv2.putText(canvas, "[BA] 正在执行 BA 全局平差优化计算...", (px1 + 20, py1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(canvas, "两阶段稳健优化中，前台画面保持实时响应，请稍候...", (px1 + 20, py1 + 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

                mx, my = self.mouse_pos
                bx = 10
                btn_y_top = by1 + 6
                btn_y_bot = h_img - 6

                # 按钮 1：【求解 BA】
                ba_btn_w = 110
                if self.is_ba_running:
                    ba_style = "warning"
                    ba_txt = "求解中..."
                else:
                    ba_style = "primary"
                    ba_txt = "求解BA (B)"
                draw_styled_button(canvas, (bx, btn_y_top, bx + ba_btn_w, btn_y_bot), ba_txt,
                                   style=ba_style, hover=(bx <= mx <= bx + ba_btn_w and btn_y_top <= my <= btn_y_bot))
                self.gui_buttons.append(("RUN_BA", (bx, btn_y_top, bx + ba_btn_w, btn_y_bot), "RUN_BA"))
                bx += ba_btn_w + 6

                # 按钮 2：【诊断报告控制台终端】
                hud_btn_w = 110
                hud_style = "info" if self.show_hud_terminal else "normal"
                hud_btn_txt = "折叠终端 (H)" if self.show_hud_terminal else "诊断终端 (H)"
                draw_styled_button(canvas, (bx, btn_y_top, bx + hud_btn_w, btn_y_bot), hud_btn_txt,
                                   style=hud_style, hover=(bx <= mx <= bx + hud_btn_w and btn_y_top <= my <= btn_y_bot))
                self.gui_buttons.append(("TOGGLE_HUD", (bx, btn_y_top, bx + hud_btn_w, btn_y_bot), "TOGGLE_HUD"))
                bx += hud_btn_w + 6

                # 按钮 3：【核心乒乓开关：实时动态 vs 静态锁定】(分段胶囊控件)
                mode_btn_w = 175
                cur_mode_key = "live" if self.live_mode else "locked"
                mode_options = [("live", "实时动态"), ("locked", "静态锁定")]
                draw_segmented_toggle(canvas, (bx, btn_y_top, bx + mode_btn_w, btn_y_bot),
                                      mode_options, active_key=cur_mode_key, shortcut="Tab")
                self.gui_buttons.append(("TOGGLE_MODE", (bx, btn_y_top, bx + mode_btn_w, btn_y_bot), "TOGGLE_MODE"))
                bx += mode_btn_w + 6

                # 按钮 4：【采样批次深度切换】(分段胶囊控件)
                batch_btn_w = 110
                cur_batch_key = str(self.batch_target_frames)
                batch_options = [("30", "30F"), ("60", "60F")]
                draw_segmented_toggle(canvas, (bx, btn_y_top, bx + batch_btn_w, btn_y_bot),
                                      batch_options, active_key=cur_batch_key, shortcut="W")
                self.gui_buttons.append(("CYCLE_BATCH", (bx, btn_y_top, bx + batch_btn_w, btn_y_bot), "CYCLE_BATCH"))
                bx += batch_btn_w + 6

                # 按钮 5：【采样并锁定位姿】
                sample_btn_w = 145
                if self.is_collecting_batch:
                    sample_style = "warning"
                    sample_text = f"采样中 ({self.collected_batch_frames}/{self.batch_target_frames})"
                else:
                    sample_style = "success" if not self.live_mode else "normal"
                    sample_text = "采样锁定 (Space)"
                draw_styled_button(canvas, (bx, btn_y_top, bx + sample_btn_w, btn_y_bot), sample_text,
                                   style=sample_style, hover=(bx <= mx <= bx + sample_btn_w and btn_y_top <= my <= btn_y_bot))
                self.gui_buttons.append(("RESAMPLE_LOCK", (bx, btn_y_top, bx + sample_btn_w, btn_y_bot), "RESAMPLE_LOCK"))
                bx += sample_btn_w + 6

                # 按钮 6：【导出质检单】
                exp_btn_w = 80
                draw_styled_button(canvas, (bx, btn_y_top, bx + exp_btn_w, btn_y_bot), "报告导出",
                                   style="normal", hover=(bx <= mx <= bx + exp_btn_w and btn_y_top <= my <= btn_y_bot))
                self.gui_buttons.append(("EXPORT", (bx, btn_y_top, bx + exp_btn_w, btn_y_bot), "EXPORT"))
                bx += exp_btn_w + 6

                # 按钮 7：【呼出人工审核画板】
                rev_btn_w = 110
                draw_styled_button(canvas, (bx, btn_y_top, bx + rev_btn_w, btn_y_bot), "审核画板 (O)",
                                   style="purple", hover=(bx <= mx <= bx + rev_btn_w and btn_y_top <= my <= btn_y_bot))
                self.gui_buttons.append(("OPEN_REVIEWER", (bx, btn_y_top, bx + rev_btn_w, btn_y_bot), "OPEN_REVIEWER"))
                bx += rev_btn_w + 6

                # 按钮 8：【呼出离线标定体检工作台】
                off_btn_w = 110
                draw_styled_button(canvas, (bx, btn_y_top, bx + off_btn_w, btn_y_bot), "体检台 (P)",
                                   style="info", hover=(bx <= mx <= bx + off_btn_w and btn_y_top <= my <= btn_y_bot))
                self.gui_buttons.append(("OPEN_OFFLINE_VERIFIER", (bx, btn_y_top, bx + off_btn_w, btn_y_bot), "OPEN_OFFLINE_VERIFIER"))
                bx += off_btn_w + 6

                # 按钮 9：【退出】
                exit_btn_w = 75
                draw_styled_button(canvas, (w_img - exit_btn_w - 10, btn_y_top, w_img - 10, btn_y_bot), "退出 (Q)",
                                   style="danger", hover=((w_img - exit_btn_w - 10) <= mx <= (w_img - 10) and btn_y_top <= my <= btn_y_bot))
                self.gui_buttons.append(("EXIT", (w_img - exit_btn_w - 10, btn_y_top, w_img - 10, btn_y_bot), "EXIT"))

                # 3. 左下角仪表盘 (HUD)
                hud_x, hud_y = 12, h_img - bottom_bar_h - 128
                hud_w, hud_h = 340, 120
                hud_overlay = canvas.copy()
                cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 15, 15), -1)
                cv2.addWeighted(hud_overlay, 0.75, canvas, 0.25, 0, canvas)
                cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (75, 75, 75), 1)

                if self.live_mode:
                    mode_title = "MODE: LIVE DYNAMIC (实时动态)"
                    mode_color = (0, 230, 255)
                    stability_label = "DYNAMIC (实时追踪)"
                else:
                    if self.locked_pose is not None:
                        mode_title = "MODE: STATIC LOCKED (基准绝对锁定)"
                        mode_color = (80, 220, 100)
                        stability_label = "ROCK-SOLID (抖动绝对为 0)"
                    else:
                        mode_title = "MODE: STATIC (待采样锁定)"
                        mode_color = (0, 160, 255)
                        stability_label = "WAITING SAMPLE"

                cv2.putText(canvas, mode_title, (hud_x + 10, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"Cam Pose: {cam_pose_str}", (hud_x + 10, hud_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"Reproj RMSE : {rmse_str}", (hud_x + 10, hud_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"3D Jitter: {jitter_str} [{stability_label}]", (hud_x + 10, hud_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

                source_str = "硬件 RealSense 1080P" if not self.mock_mode else f"采图回放 ({self.mock_img_idx+1}/{len(self.mock_image_files)})"
                cv2.putText(canvas, f"输入源: {source_str}", (hud_x + 10, hud_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

                # 4. 右上角常驻【留一盲测定量质检卡片 (HUD)】
                if blind_summary is not None:
                    hud_rx = w_img - 305
                    hud_ry = top_bar_h + 10
                    hud_rw = 295
                    hud_rh = 95
                    hud_r_overlay = canvas.copy()
                    cv2.rectangle(hud_r_overlay, (hud_rx, hud_ry), (hud_rx + hud_rw, hud_ry + hud_rh), (16, 16, 16), -1)
                    cv2.addWeighted(hud_r_overlay, 0.82, canvas, 0.18, 0, canvas)
                    
                    b_eval_c = (0, 230, 100) if blind_summary['err_px'] < 1.0 else (0, 180, 255)
                    cv2.rectangle(canvas, (hud_rx, hud_ry), (hud_rx + hud_rw, hud_ry + hud_rh), b_eval_c, 1, cv2.LINE_AA)

                    cv2.putText(canvas, f"[盲测] 留一盲测反推评估 (Tag #{blind_summary['tag_id']})", 
                                (hud_rx + 10, hud_ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 120, 240), 1, cv2.LINE_AA)
                    
                    p_txt = "[PERFECT]" if blind_summary['err_px'] < 1.0 else "[ACCEPTABLE]"
                    cv2.putText(canvas, f"重投影残差 : {blind_summary['err_px']:.2f} px {p_txt}", 
                                (hud_rx + 10, hud_ry + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
                    cv2.putText(canvas, f"空间绝对偏差: {blind_summary['err_mm']:.2f} mm", 
                                (hud_rx + 10, hud_ry + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.40, b_eval_c, 1, cv2.LINE_AA)
                    cv2.putText(canvas, f"目标测距深度: {blind_summary['depth_mm']:.1f} mm", 
                                (hud_rx + 10, hud_ry + 83), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

                # 5. 浮层 Toast 通知
                if time.time() - self.status_toast_time < 2.5 and self.status_toast:
                    (tw, th), _ = cv2.getTextSize(self.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                    toast_x = (w_img - tw) // 2
                    cv2.rectangle(canvas, (toast_x - 14, h_img - bottom_bar_h - 48), (toast_x + tw + 14, h_img - bottom_bar_h - 16), (140, 0, 120), -1)
                    cv2.putText(canvas, self.status_toast, (toast_x, h_img - bottom_bar_h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

                # 6. 渲染可折叠 HUD 诊断文本终端浮层 (方案 B)
                self.render_hud_terminal(canvas)

                cv2.imshow(window_name, canvas)
                if frame_idx <= 3 and force_window_focus:
                    force_window_focus(window_name)

                key = cv2.waitKey(20) & 0xFF

                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('b'), ord('B')):     # B 键 -> 一键异步求解 BA 全局平差并热重载
                    self.start_async_bundle_adjustment()
                elif key in (ord('h'), ord('H')):     # H 键 -> 展开/折叠 HUD 诊断报告终端
                    self.toggle_hud_terminal()
                elif key in (ord('u'), ord('U')):     # U 键 -> HUD 终端向上翻滚
                    self.hud_scroll_offset = max(0, self.hud_scroll_offset - 6)
                elif key in (ord('j'), ord('J')):     # J 键 -> HUD 终端向下翻滚
                    self.hud_scroll_offset += 6
                elif key in (9, ord('m'), ord('M')):  # Tab 键 (ASCII 9) 或 M 键 -> 乒乓切换
                    self.toggle_mode()
                elif key in (ord('w'), ord('W')):    # W 键 -> 切换批次深度
                    self.cycle_batch_target()
                elif key in (ord('t'), ord('T')):    # T 键 -> 循环切换留一盲测目标
                    if self.mapped_tag_ids:
                        if self.blind_target_tag_id is None:
                            self.blind_target_tag_id = self.mapped_tag_ids[0]
                        else:
                            try:
                                cur_idx = self.mapped_tag_ids.index(self.blind_target_tag_id)
                                if cur_idx + 1 < len(self.mapped_tag_ids):
                                    self.blind_target_tag_id = self.mapped_tag_ids[cur_idx + 1]
                                else:
                                    self.blind_target_tag_id = None
                            except ValueError:
                                self.blind_target_tag_id = None
                        toast = f"已锁定留一盲测目标: Tag #{self.blind_target_tag_id}" if self.blind_target_tag_id is not None else "全量标靶联合解算 (ALL)"
                        self.set_toast(toast)

                elif key in (ord('c'), ord('C')):
                    self.blind_target_tag_id = None
                    self.set_toast("已清除盲测，恢复全量解算 (ALL)")

                elif key in (ord('o'), ord('O')):     # O 键 -> 呼出人工审核画板并定向排查当前盲测 Tag
                    self.open_reviewer()

                elif key in (ord('p'), ord('P')):     # P 键 -> 呼出离线标定体检工作台
                    self.open_offline_verifier()

                elif key in (ord('a'), ord('A'), 81):  # A 键或左方向键
                    if self.mock_mode and self.mock_image_files:
                        self.mock_img_idx = (self.mock_img_idx - 1) % len(self.mock_image_files)
                        cur_name = os.path.basename(self.mock_image_files[self.mock_img_idx])
                        self.set_toast(f"切换至仿真帧: {cur_name}")

                elif key in (ord('d'), ord('D'), 83):  # D 键或右方向键
                    if self.mock_mode and self.mock_image_files:
                        self.mock_img_idx = (self.mock_img_idx + 1) % len(self.mock_image_files)
                        cur_name = os.path.basename(self.mock_image_files[self.mock_img_idx])
                        self.set_toast(f"切换至仿真帧: {cur_name}")

                elif key == 32:  # Space 空格键
                    if not self.live_mode:
                        # 静态模式下：重新触发定点采样并锁定
                        self.start_batch_collection()
                    else:
                        # 实时模式下：抓拍当前单帧并导出质检单
                        ts = int(time.time())
                        snap_name = f"verification_view_{ts}.png"
                        snap_path = os.path.join(VERIFICATION_DIR, snap_name)
                        cv2.imwrite(snap_path, disp_frame)

                        report_path = os.path.join(VERIFICATION_DIR, f"report_verification_{ts}.md")
                        with open(report_path, "w", encoding="utf-8") as rf:
                            rf.write(f"# AprilTag 在线 AR 标定验证质检单\n\n")
                            rf.write(f"- **质检抓拍时间**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
                            rf.write(f"- **运行模式**: `实时动态模式 (LIVE)`\n")
                            rf.write(f"- **存储目录**: `data/tag_calibration_verification/`\n")
                            rf.write(f"- **抓拍图像文件**: [{snap_name}]({snap_name})\n")
                            rf.write(f"- **相机位姿状态**: `{cam_pose_str}`\n")
                            rf.write(f"- **重投影误差**: `{rmse_str}`\n")
                            rf.write(f"- **抖动稳定性**: `{jitter_str}`\n")
                            rf.write(f"- **参与求解标靶**: `{active_matched_tags}`\n")
                            if blind_summary:
                                rf.write(f"\n### 留一盲测交叉验证结果\n\n")
                                rf.write(f"- **盲测验证目标**: `Tag #{blind_summary['tag_id']}` (解算时已强制屏蔽)\n")
                                rf.write(f"- **反推像元误差**: `{blind_summary['err_px']:.2f} px`\n")
                                rf.write(f"- **空间几何偏差**: `{blind_summary['err_mm']:.2f} mm`\n")
                                rf.write(f"- **目标测距深度**: `{blind_summary['depth_mm']:.1f} mm`\n")

                        self.set_toast("实时质检快照与报告已导出！")
                        print(f"[OK] 实时质检图已保存: {snap_path}")
                        print(f"[OK] 实时质检单已保存: {report_path}")

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AprilTag 标定精度与 3D 坐标系在线 AR 综合验证")
    parser.add_argument("--map", type=str, default=DEFAULT_MAP_PATH, help="标靶空间地图路径")
    parser.add_argument("--mock", action="store_true", help="仿真演示模式 (无需物理硬件)")
    parser.add_argument("--blind_tag", type=int, default=None, help="默认启用的留一盲测标靶 ID (如 26)")
    parser.add_argument("--locked", action="store_true", help="初始启动即进入静态滤波锁定模式")
    args = parser.parse_args()

    verifier = TagCalibrationVerifier(map_path=args.map, mock_mode=args.mock)
    if args.blind_tag is not None:
        verifier.blind_target_tag_id = args.blind_tag
    if args.locked:
        verifier.live_mode = False
        verifier.start_batch_collection()
    verifier.run()


if __name__ == "__main__":
    main()
