#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 标定精度与 3D 坐标系在线实时 AR 验证系统 (Tag Calibration Verifier)
==========================================================================
核心职责与专业特性：
  1. 留一法盲测交叉验证 (Leave-One-Out Blind AR Verification)：
     - 允许用户在 GUI 中点选指定标靶 (如 Tag 26) 作为“盲测验证目标”；
     - 算法在 PnP 解算相机位姿时强制屏蔽该标靶，仅依靠视野中其余标靶反推该标靶的 3D 空间位姿；
     - 将反推出来的虚拟标靶角点与“很粗很壮的 20mm 正四棱柱 3D 轴 (Z轴)”投影叠加在画面上；
     - 肉眼直观检验虚实重合度，并实时量化计算像元级重投影残差 (px) 与毫米级空间绝对误差 (mm)。
  2. 极简轻量级 OpenCV 原生 GUI 按钮交互：
     - 顶栏直接渲染可点击的标靶列表按钮：[ALL] [Tag 24] [Tag 25] [Tag 26]...
     - 支持鼠标在视频画面中直接单击任意标靶将其设为盲测目标；
     - 支持键盘快捷键 [T/B] 循环切换盲测目标、[C] 清除恢复全量解算、[A/D] 切换仿真帧。
  3. 工业级 3D 正四棱柱立体质感渲染 (render_tag_3d_axes)：
     - 边长 20.0mm x 20.0mm、长 120.0mm 的实心正四棱柱立体填充；
     - 12 条亮白棱线 + 顶盖透视截面 + 盲测高亮品红/金黄光效。
  4. 真实相机流与 15 帧真实采图回放仿真无缝兼容：
     - 有物理 RealSense 时直接对接 1080P 活体视频流；
     - 无物理硬件或指定 --mock 时，自动加载 data/tag_calibration_images/ 真实采图进行离线交叉验证。
  5. 专属隔离质检目录：
     - 按 [Space] 一键保存 AR 质检帧与详细 Markdown 质检单至 data/tag_calibration_verification/。
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False


class TagCalibrationVerifier:
    def __init__(self, map_path: str = DEFAULT_MAP_PATH, mock_mode: bool = False):
        self.map_path = map_path
        self.mock_mode = mock_mode
        self.tags_map = None
        self.marker_size_mm = 50.0

        # 确保专属验证目录存在
        os.makedirs(VERIFICATION_DIR, exist_ok=True)

        # 1. 加载 config.yaml 配置 (内参 + 标靶白名单)
        self.valid_tag_ids = []
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()

        # 2. 加载地图
        self._load_tags_map()

        # 3. AprilTag 16h5 超高灵敏度检测器 (采用 SUBPIX 角点细化与高抗模糊配置)
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

        # 4. 时域抗抖平滑滤波器 (EMA Filter)
        self.filter_mode = 1
        self.filter_alphas = [1.0, 0.35, 0.15]
        self.filter_names = ["无滤波 (RAW裸解)", "动态平滑 (SMOOTH)", "超稳固防抖 (ROCK-SOLID)"]
        self.smoothed_rvec = None
        self.smoothed_tvec = None

        # 5. 实时抖动监控探针 (过去 30 帧位姿历史)
        self.pose_history_xyz = []
        self.max_history_len = 30

        # 6. 留一盲测核心状态 (Leave-One-Out Cross-Validation)
        # None: 全量参与解算; int: 指定盲测标靶 ID (例如 26)
        self.blind_target_tag_id = None
        self.mapped_tag_ids = []
        if self.tags_map and "tags" in self.tags_map:
            self.mapped_tag_ids = sorted([int(tid) for tid in self.tags_map["tags"].keys()])

        # GUI 交互热区 [(tag_id, (x1, y1, x2, y2), label)] (None 代表 ALL 按钮)
        self.gui_buttons = []
        self.current_frame_tags_polys = {} # {tag_id: np.ndarray(4, 2)}
        self.mouse_pos = (-1, -1)

        # 仿真模式采图列表
        self.mock_image_files = sorted(glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.png")))
        self.mock_img_idx = 0

        # 临时 Toast 通知
        self.status_toast = ""
        self.status_toast_time = 0.0

        # 硬件相机
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
            print(f"[*] 请先运行标定向导 [3] 构建 AprilTag 3D 空间立体地图！")
            return

        try:
            with open(self.map_path, "r", encoding="utf-8") as f:
                self.tags_map = yaml.safe_load(f)
            self.marker_size_mm = float(self.tags_map.get("marker_size_mm", 50.0))
            tag_count = len(self.tags_map.get("tags", {}))
            print(f"[OK] 成功加载标靶空间地图: {self.map_path} (共包含 {tag_count} 个已知标靶)")
        except Exception as e:
            print(f"[ERROR] 读取标靶地图失败: {e}")

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
                print(f"[OK] RealSense 硬件实时在线内参已同步: fx={intr.fx:.2f}, cx={intr.ppx:.2f}")
            except Exception:
                config = rs.config()
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                self.pipeline.start(config)

            for _ in range(5):
                self.pipeline.wait_for_frames(timeout_ms=2000)

        except Exception as e:
            print(f"[WARN] 启动物理相机失败: {e}，切换至仿真回放模式")
            self.mock_mode = True
            self.pipeline = None

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """获取图像帧：优先实时相机流；若仿真模式则回放采图数据集"""
        if not self.mock_mode and self.pipeline is not None:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    return np.asanyarray(color_frame.get_data())
            except Exception:
                pass

        # 仿真/回放模式：读取真实采图
        if self.mock_image_files:
            cur_file = self.mock_image_files[self.mock_img_idx % len(self.mock_image_files)]
            img = cv2.imread(cur_file)
            if img is not None:
                return img

        # 纯虚拟背景回退
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        for x in range(0, 1920, 100):
            cv2.line(frame, (x, 0), (x, 1080), (55, 55, 55), 1)
        for y in range(0, 1080, 100):
            cv2.line(frame, (0, y), (1920, y), (55, 55, 55), 1)
        return frame

    def _get_tag_world_transform(self, tag_id: int) -> np.ndarray:
        """获取标靶在世界坐标系下的 4x4 变换矩阵 T_w_tag"""
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
        """获取标靶 4 个角点在世界坐标系下的绝对 3D 坐标"""
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

    def render_tag_3d_axes(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                           tag_id: int, is_blind_projection: bool = False):
        """
        在图像上绘制标靶 3D 空间坐标系与很粗很壮的实心正四棱柱立体轴：
        - 边长 20.0mm x 20.0mm，长度 120.0mm 的实心正四棱柱
        - 半透明立体渐变侧面 + 12 条高亮棱线 + 顶盖透视截面
        - 留一盲测模式下采用高光品红色/亮金黄色光效，形成强烈的“幽灵虚实结合”质感
        """
        try:
            hw = 10.0   # 截面半宽 10mm (整体截面 20.0mm x 20.0mm)
            L = 120.0   # 棱柱长度 120mm

            # 3D 端点定义
            pts_3d = np.array([
                # 底面 4 点 (Z=0)
                [-hw, -hw, 0.0],
                [ hw, -hw, 0.0],
                [ hw,  hw, 0.0],
                [-hw,  hw, 0.0],
                # 顶面 4 点 (Z=L)
                [-hw, -hw, L],
                [ hw, -hw, L],
                [ hw,  hw, L],
                [-hw,  hw, L],
                # 顶面中心
                [0.0, 0.0, L],
                # X 轴与 Y 轴参考端点 (从中心伸出 30mm)
                [30.0, 0.0, 0.0],
                [0.0, 30.0, 0.0],
                [0.0, 0.0, 0.0]
            ], dtype=np.float64)

            proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            proj = proj.reshape((-1, 2)).astype(int)

            b_pts = proj[0:4] # 底面 4 点
            t_pts = proj[4:8] # 顶面 4 点
            p_x = tuple(proj[9])
            p_y = tuple(proj[10])
            p_orig = tuple(proj[11])

            # 1. 绘制 X 轴 (红) 和 Y 轴 (绿)
            cv2.line(img, p_orig, p_x, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(img, 'X', p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(img, p_orig, p_y, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(img, 'Y', p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            # 2. 立体柱体半透明填充
            overlay = img.copy()
            if is_blind_projection:
                # 盲测反推高光配色：品红/紫罗兰渐变，极具辨识度
                side_color = (220, 50, 180)   # BGR
                cap_color = (255, 120, 240)
                edge_color = (255, 200, 255)
                alpha = 0.55
            else:
                # 正常模式：经典工业天蓝/亮青色
                side_color = (240, 160, 30)
                cap_color = (255, 220, 90)
                edge_color = (255, 255, 255)
                alpha = 0.42

            for i in range(4):
                next_i = (i + 1) % 4
                side_poly = np.array([b_pts[i], b_pts[next_i], t_pts[next_i], t_pts[i]], dtype=np.int32)
                cv2.fillPoly(overlay, [side_poly], side_color)
            cv2.fillPoly(overlay, [t_pts], cap_color) # 顶面高光
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

            # 3. 绘制 12 条高亮棱线
            cv2.polylines(img, [b_pts], isClosed=True, color=edge_color, thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(img, [t_pts], isClosed=True, color=edge_color, thickness=2, lineType=cv2.LINE_AA)
            for i in range(4):
                cv2.line(img, tuple(b_pts[i]), tuple(t_pts[i]), edge_color, 2, cv2.LINE_AA)

            # 4. Z 轴标识
            top_center = tuple(proj[8])
            cv2.putText(img, 'Z (Norm)', top_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception:
            pass

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()

    def _on_mouse(self, event, x, y, flags, param):
        """鼠标交互处理：点击顶栏按钮或直接点击画面中的标靶切换盲测目标"""
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. 检查是否点击了顶部 GUI 按钮
            for btn_id, (bx1, by1, bx2, by2), label in self.gui_buttons:
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if btn_id == self.blind_target_tag_id:
                        # 再次点击已选中的按钮 -> 取消盲测
                        self.blind_target_tag_id = None
                        self.set_toast("已取消盲测，恢复全量标靶解算 (ALL)")
                    else:
                        self.blind_target_tag_id = btn_id
                        if btn_id is None:
                            self.set_toast("已切换为: 全量标靶联合解算 (ALL)")
                        else:
                            self.set_toast(f"已锁定留一盲测目标: Tag #{btn_id} (由其余标靶反推验证)")
                    return

            # 2. 检查是否点击了画面中的标靶轮廓
            for tid, poly in self.current_frame_tags_polys.items():
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    if self.blind_target_tag_id == tid:
                        self.blind_target_tag_id = None
                        self.set_toast(f"已取消 Tag #{tid} 盲测，恢复全量解算")
                    else:
                        self.blind_target_tag_id = tid
                        self.set_toast(f"已选定 Tag #{tid} 为盲测验证目标 (PnP中已主动屏蔽)")
                    return

    def run(self):
        window_name = "AprilTag SCARA Hand-Eye AR Verifier (Interactive Blind Cross-Validation)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self._on_mouse)

        print("\n" + "=" * 80)
        print("     AprilTag 标定精度与 3D 坐标系在线实时 AR 验证系统 (Blind Verification Edition)")
        print("=" * 80)
        print(f" [专属验证目录] : {VERIFICATION_DIR}")
        print(f" [相机内参绑定] : fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}, cx={self.camera_matrix[0,2]:.1f}")
        print(f" [地图标靶列表] : {self.mapped_tag_ids}")
        print(f" [时域抗抖滤波] : 默认启用 -> {self.filter_names[self.filter_mode]}")
        print(" [留一盲测说明] : 在顶栏直接鼠标点击 Tag 编号 (或画面中标靶)，系统将在 PnP 中屏蔽它，")
        print("                 并纯由其余标靶隔空反推其 3D 位置与立体 Z 轴，实现虚实重合验证！")
        print(" [快捷键指南]   :")
        print("   - [鼠标左键]      : 点击顶栏 Tag 按钮或画面中标靶，直接切换盲测目标；")
        print("   - [T] / [B]       : 顺序轮换留一盲测目标 (None -> 24 -> 25 -> 26...)；")
        print("   - [C]             : 清除盲测目标，恢复全量融合解算；")
        print("   - [A] / [D]       : 仿真回放模式下，前后翻页浏览 15 张真实采图；")
        print("   - [Space] (空格键) : 抓取当前 AR 质检帧，并导出至 data/tag_calibration_verification/；")
        print("   - [F]             : 循环切换时域抗抖滤波模式 (无滤波 / 动态平滑 / 超稳固)；")
        print("   - [Q] / [ESC]     : 退出在线验证。")
        print("=" * 80 + "\n")

        frame_idx = 0
        try:
            while True:
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

                        pts = corners[i].reshape((4, 2)).astype(int)
                        self.current_frame_tags_polys[tid_int] = pts
                        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

                        # 如果当前标靶是“留一盲测目标”，在 PnP 求解中强制屏蔽它！
                        if self.blind_target_tag_id is not None and tid_int == self.blind_target_tag_id:
                            blind_real_corners = corners[i].reshape((4, 2)).astype(np.float64)
                            blind_real_center = (cx, cy)
                            # 仅绘制微弱的灰色真实边框，提示物理标靶存在
                            cv2.polylines(disp_frame, [pts], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
                            cv2.putText(disp_frame, f"Tag #{tid_int} [REAL PHYSICAL]", (cx - 50, cy + 25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                            continue

                        w_corners = self._get_tag_world_corners(tid_int)
                        if w_corners is not None:
                            matched_tags.append(tid_int)
                            all_obj_pts.append(w_corners)
                            all_img_pts.append(corners[i].reshape((4, 2)))

                            # 绘制正常参与解算的标靶边框与标牌
                            box_color = (0, 255, 255) if tid_int == 0 else (0, 255, 0)
                            cv2.polylines(disp_frame, [pts], isClosed=True, color=box_color, thickness=2, lineType=cv2.LINE_AA)
                            label = f"Tag #{tid_int} [ORIGIN]" if tid_int == 0 else f"Tag #{tid_int}"
                            cv2.putText(disp_frame, label, (cx - 35, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.polylines(disp_frame, [pts], isClosed=True, color=(0, 165, 255), thickness=1)
                            cv2.putText(disp_frame, f"Tag #{tid_int} [UNMAP]", (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

                # 解算相机在 SCARA 世界系下的 6DoF 位姿
                cam_pose_str = "Camera 6DoF: Waiting for >= 2 mapped tags..."
                rmse_str = "Reproj RMSE: N/A"
                jitter_str = "Jitter: N/A"
                is_valid_pose = False
                blind_summary = None

                if len(matched_tags) >= 2 and len(all_obj_pts) > 0:
                    obj_pts_arr = np.vstack(all_obj_pts).astype(np.float64)
                    img_pts_arr = np.vstack(all_img_pts).astype(np.float64)

                    # 采用高精度 SQPnP / ITERATIVE PnP 求解
                    success, rvec_raw, tvec_raw = cv2.solvePnP(
                        obj_pts_arr,
                        img_pts_arr,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        is_valid_pose = True

                        # --- 时域低通平滑滤波 (EMA Anti-Jitter Filter) ---
                        alpha = self.filter_alphas[self.filter_mode]
                        if alpha < 1.0:
                            if self.smoothed_rvec is None:
                                self.smoothed_rvec = rvec_raw.copy()
                                self.smoothed_tvec = tvec_raw.copy()
                            else:
                                self.smoothed_rvec = (1.0 - alpha) * self.smoothed_rvec + alpha * rvec_raw
                                self.smoothed_tvec = (1.0 - alpha) * self.smoothed_tvec + alpha * tvec_raw
                            rvec_cam = self.smoothed_rvec.copy()
                            tvec_cam = self.smoothed_tvec.copy()
                        else:
                            rvec_cam = rvec_raw.copy()
                            tvec_cam = tvec_raw.copy()

                        # 重投影均方根误差 (RMSE)
                        proj_pts, _ = cv2.projectPoints(obj_pts_arr, rvec_cam, tvec_cam, self.camera_matrix, self.dist_coeffs)
                        errors = np.linalg.norm(img_pts_arr - proj_pts.reshape(-1, 2), axis=1)
                        rmse_px = float(np.sqrt(np.mean(errors ** 2)))

                        # 相机在世界坐标系下的绝对位置与姿态
                        R_w2c, _ = cv2.Rodrigues(rvec_cam)
                        t_w2c = tvec_cam.reshape((3, 1))
                        R_c2w = R_w2c.T
                        t_c2w = -R_c2w @ t_w2c
                        cx_mm, cy_mm, cz_mm = t_c2w.flatten()

                        # 记录位姿历史，计算实时抖动标准差
                        self.pose_history_xyz.append((cx_mm, cy_mm, cz_mm))
                        if len(self.pose_history_xyz) > self.max_history_len:
                            self.pose_history_xyz.pop(0)

                        if len(self.pose_history_xyz) >= 10:
                            arr = np.array(self.pose_history_xyz)
                            std_xyz = np.std(arr, axis=0)
                            sigma_max = float(np.max(std_xyz))
                            jitter_status = "STABLE" if sigma_max < 0.8 else ("MODERATE" if sigma_max < 2.0 else "JITTER")
                            jitter_color_str = f"sigma=+/-{sigma_max:.2f}mm [{jitter_status}]"
                        else:
                            jitter_color_str = "Measuring..."

                        # 欧拉角 (RPY deg)
                        sy = math.sqrt(R_c2w[0, 0] ** 2 + R_c2w[1, 0] ** 2)
                        if sy > 1e-6:
                            roll = math.atan2(R_c2w[2, 1], R_c2w[2, 2])
                            pitch = math.atan2(-R_c2w[2, 0], sy)
                            yaw = math.atan2(R_c2w[1, 0], R_c2w[0, 0])
                        else:
                            roll = math.atan2(-R_c2w[1, 2], R_c2w[1, 1])
                            pitch = math.atan2(-R_c2w[2, 0], sy)
                            yaw = 0.0

                        cam_pose_str = f"Cam: X={cx_mm:+.1f}mm, Y={cy_mm:+.1f}mm, Z={cz_mm:+.1f}mm | R={math.degrees(roll):.1f} P={math.degrees(pitch):.1f} Y={math.degrees(yaw):.1f}"
                        quality = "EXCELLENT (<0.5px)" if rmse_px < 0.5 else ("GOOD (<1.0px)" if rmse_px < 1.0 else "FAIR")
                        rmse_str = f"RMSE: {rmse_px:.2f}px [{quality}] | Solved Tags: {matched_tags}"
                        jitter_str = f"Stability: {jitter_color_str}"

                        # =========================================================================
                        # 【核心创新】留一法盲测反推投影 (Leave-One-Out Blind Projection)
                        # =========================================================================
                        if self.blind_target_tag_id is not None:
                            T_w_target = self._get_tag_world_transform(self.blind_target_tag_id)
                            if T_w_target is not None:
                                # 相机系变换矩阵 T_c_w
                                T_c_w = np.eye(4, dtype=np.float64)
                                T_c_w[:3, :3] = R_w2c
                                T_c_w[:3, 3] = tvec_cam.flatten()

                                # 隔空反推盲测标靶在当前相机系下的位姿: T_c_target = T_c_w @ T_w_target
                                T_c_target = T_c_w @ T_w_target
                                R_c_target = T_c_target[:3, :3]
                                t_c_target = T_c_target[:3, 3].reshape((3, 1))
                                rvec_target, _ = cv2.Rodrigues(R_c_target)

                                # 1. 投影虚拟标靶 4 个角点
                                s = self.marker_size_mm / 2.0
                                local_corners_3d = np.array([
                                    [-s,  s, 0.0],
                                    [ s,  s, 0.0],
                                    [ s, -s, 0.0],
                                    [-s, -s, 0.0]
                                ], dtype=np.float64)

                                pred_pts_2d, _ = cv2.projectPoints(local_corners_3d, rvec_target, t_c_target, 
                                                                   self.camera_matrix, self.dist_coeffs)
                                pred_pts_2d = pred_pts_2d.reshape((4, 2))
                                pred_poly = pred_pts_2d.astype(int)
                                pred_center = (int(np.mean(pred_pts_2d[:, 0])), int(np.mean(pred_pts_2d[:, 1])))

                                # 2. 渲染很粗很壮的 20mm 正四棱柱 3D 轴 (Z轴高光挺立)
                                self.render_tag_3d_axes(disp_frame, rvec_target, t_c_target, 
                                                        tag_id=self.blind_target_tag_id, is_blind_projection=True)

                                # 3. 渲染虚拟预测标靶轮廓 (亮品红发光边框)
                                cv2.polylines(disp_frame, [pred_poly], isClosed=True, color=(255, 0, 255), thickness=3, lineType=cv2.LINE_AA)
                                cv2.drawMarker(disp_frame, pred_center, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                                cv2.putText(disp_frame, f"[BLIND PREDICTED] Tag #{self.blind_target_tag_id}", 
                                            (pred_poly[0][0], pred_poly[0][1] - 12), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2, cv2.LINE_AA)

                                # 4. 若物理图像中真实检测到了该标靶，计算定量重投影吻合误差
                                if blind_real_corners is not None:
                                    err_px = float(np.sqrt(np.mean(np.sum((blind_real_corners - pred_pts_2d) ** 2, axis=1))))
                                    target_depth_mm = float(t_c_target[2, 0])
                                    fx = float(self.camera_matrix[0, 0])
                                    err_mm = (err_px * target_depth_mm) / fx if fx > 0 else 0.0

                                    # 绘制真假角点之间的连线残差指示 (黄色虚线/细线)
                                    for cp, cr in zip(pred_pts_2d, blind_real_corners):
                                        cv2.line(disp_frame, tuple(cp.astype(int)), tuple(cr.astype(int)), (0, 255, 255), 2, cv2.LINE_AA)

                                    blind_summary = {
                                        "tag_id": self.blind_target_tag_id,
                                        "err_px": err_px,
                                        "err_mm": err_mm,
                                        "depth_mm": target_depth_mm,
                                        "solved_by": matched_tags
                                    }

                                    # 绘制悬浮盲测残差卡片
                                    card_x = max(15, pred_center[0] - 150)
                                    card_y = max(120, pred_center[1] + 60)
                                    card_w = 340
                                    card_h = 75
                                    cv2.rectangle(disp_frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 0, 40), -1)
                                    cv2.rectangle(disp_frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 255), 2)
                                    
                                    eval_str = "PERFECT FIT (<1px)" if err_px < 1.0 else ("GOOD (<2px)" if err_px < 2.0 else "FAIR")
                                    cv2.putText(disp_frame, f"BLIND VERIFICATION: Tag #{self.blind_target_tag_id}", 
                                                (card_x + 10, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 180, 255), 2)
                                    cv2.putText(disp_frame, f"Reproj Error : {err_px:.2f} px [{eval_str}]", 
                                                (card_x + 10, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
                                    cv2.putText(disp_frame, f"3D Distortion: {err_mm:.2f} mm | Solved by {len(matched_tags)} tags", 
                                                (card_x + 10, card_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1)

                        else:
                            # 未开启盲测时，在世界原点 (Tag 0) 处绘制全局参考 3D 轴
                            T_w_0 = self._get_tag_world_transform(0)
                            if T_w_0 is not None:
                                T_c_w = np.eye(4, dtype=np.float64)
                                T_c_w[:3, :3] = R_w2c
                                T_c_w[:3, 3] = tvec_cam.flatten()
                                T_c_0 = T_c_w @ T_w_0
                                rvec_0, _ = cv2.Rodrigues(T_c_0[:3, :3])
                                tvec_0 = T_c_0[:3, 3].reshape((3, 1))
                                self.render_tag_3d_axes(disp_frame, rvec_0, tvec_0, tag_id=0, is_blind_projection=False)

                # =========================================================================
                # 顶部半透明 HUD 仪表盘与 GUI 按钮栏
                # =========================================================================
                hud_h = 100
                overlay = disp_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w_img, hud_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.82, disp_frame, 0.18, 0, disp_frame)

                status_color = (0, 255, 0) if is_valid_pose else (0, 180, 255)
                cv2.putText(disp_frame, cam_pose_str, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)
                cv2.putText(disp_frame, rmse_str, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
                cv2.putText(disp_frame, jitter_str, (15, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

                # --- 渲染可点击的 GUI Tag 选择按钮 ---
                btn_y1 = 68
                btn_y2 = 94
                btn_x = 15
                self.gui_buttons.clear()

                # 1. [ALL 全量解算] 按钮
                all_w = 95
                is_all_active = (self.blind_target_tag_id is None)
                all_bg = (0, 140, 0) if is_all_active else (50, 50, 50)
                cv2.rectangle(disp_frame, (btn_x, btn_y1), (btn_x + all_w, btn_y2), all_bg, -1)
                cv2.rectangle(disp_frame, (btn_x, btn_y1), (btn_x + all_w, btn_y2), (200, 200, 200), 1)
                cv2.putText(disp_frame, "ALL (全量)", (btn_x + 8, btn_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                self.gui_buttons.append((None, (btn_x, btn_y1, btn_x + all_w, btn_y2), "ALL"))
                btn_x += all_w + 10

                # 2. 地图中的各个 Tag 按钮
                for tid in self.mapped_tag_ids:
                    btn_w = 78
                    is_btn_active = (self.blind_target_tag_id == tid)
                    if is_btn_active:
                        btn_bg = (180, 0, 160)     # 激活品红
                        border_col = (255, 100, 255)
                        txt_col = (255, 255, 255)
                        btn_label = f"Tag #{tid} *"
                    else:
                        btn_bg = (55, 55, 55)
                        border_col = (150, 150, 150)
                        txt_col = (210, 210, 210)
                        btn_label = f"Tag #{tid}"

                    cv2.rectangle(disp_frame, (btn_x, btn_y1), (btn_x + btn_w, btn_y2), btn_bg, -1)
                    cv2.rectangle(disp_frame, (btn_x, btn_y1), (btn_x + btn_w, btn_y2), border_col, 1)
                    cv2.putText(disp_frame, btn_label, (btn_x + 8, btn_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_col, 1)
                    self.gui_buttons.append((tid, (btn_x, btn_y1, btn_x + btn_w, btn_y2), f"Tag #{tid}"))
                    btn_x += btn_w + 8

                # 盲测引导提示
                if self.blind_target_tag_id is not None:
                    guide_msg = f"[留一盲测中] 已主动屏蔽 Tag #{self.blind_target_tag_id}，正在由其余标靶反推 3D 棱柱！点击 [ALL] 取消"
                    cv2.putText(disp_frame, guide_msg, (btn_x + 15, btn_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 120, 255), 1)
                else:
                    guide_msg = "提示: 鼠标点击上方任意 Tag 按钮或直接点击画面中标靶，即可启动【留一盲测反推验证】"
                    cv2.putText(disp_frame, guide_msg, (btn_x + 15, btn_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

                # 底部控制提示条
                cv2.rectangle(disp_frame, (0, h_img - 35), (w_img, h_img), (15, 15, 15), -1)
                mode_str = f"MOCK 回放 ({self.mock_img_idx+1}/{len(self.mock_image_files)}) [A/D切图]" if self.mock_mode else "LIVE 真实硬件流"
                ctrl_tip = f"[{mode_str}] | [鼠标点选]: 切换盲测 | [T/B]: 轮换盲测 | [C]: 恢复全量 | [Space]: 存质检单 | [F]: 滤波 | [Q]: 退出"
                cv2.putText(disp_frame, ctrl_tip, (15, h_img - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

                # Toast 临时通知
                if time.time() - self.status_toast_time < 2.5 and self.status_toast:
                    (tw, _), _ = cv2.getTextSize(self.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    toast_x = (w_img - tw) // 2
                    cv2.rectangle(disp_frame, (toast_x - 12, h_img - 80), (toast_x + tw + 12, h_img - 45), (140, 0, 120), -1)
                    cv2.putText(disp_frame, self.status_toast, (toast_x, h_img - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                cv2.imshow(window_name, disp_frame)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('t'), ord('T'), ord('b'), ord('B')):
                    # T 或 B 键循环切换盲测目标
                    if not self.mapped_tag_ids:
                        continue
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
                    if self.blind_target_tag_id is None:
                        self.set_toast("已清除盲测，恢复全量解算 (ALL)")
                    else:
                        self.set_toast(f"已锁定留一盲测目标: Tag #{self.blind_target_tag_id}")

                elif key in (ord('c'), ord('C')):
                    # C 键清除盲测
                    self.blind_target_tag_id = None
                    self.set_toast("已清除盲测，恢复全量解算 (ALL)")

                elif key in (ord('a'), ord('A'), 81): # A 键或左方向键
                    if self.mock_mode and self.mock_image_files:
                        self.mock_img_idx = (self.mock_img_idx - 1) % len(self.mock_image_files)
                        cur_name = os.path.basename(self.mock_image_files[self.mock_img_idx])
                        self.set_toast(f"切换至仿真帧: {cur_name}")

                elif key in (ord('d'), ord('D'), 83): # D 键或右方向键
                    if self.mock_mode and self.mock_image_files:
                        self.mock_img_idx = (self.mock_img_idx + 1) % len(self.mock_image_files)
                        cur_name = os.path.basename(self.mock_image_files[self.mock_img_idx])
                        self.set_toast(f"切换至仿真帧: {cur_name}")

                elif key in (ord('f'), ord('F')): # F 键切换滤波模式
                    self.filter_mode = (self.filter_mode + 1) % len(self.filter_alphas)
                    self.smoothed_rvec = None
                    self.smoothed_tvec = None
                    self.pose_history_xyz.clear()
                    self.set_toast(f"滤波模式: {self.filter_names[self.filter_mode]}")

                elif key == 32:  # Space 存验证快照与报告
                    ts = int(time.time())
                    snap_name = f"verification_view_{ts}.png"
                    snap_path = os.path.join(VERIFICATION_DIR, snap_name)
                    cv2.imwrite(snap_path, disp_frame)

                    report_path = os.path.join(VERIFICATION_DIR, f"report_verification_{ts}.md")
                    with open(report_path, "w", encoding="utf-8") as rf:
                        rf.write(f"# AprilTag 在线 AR 标定验证质检单\n\n")
                        rf.write(f"- **质检抓拍时间**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
                        rf.write(f"- **存储目录**: `data/tag_calibration_verification/` (专属隔离目录)\n")
                        rf.write(f"- **抓拍图像文件**: [{snap_name}]({snap_name})\n")
                        rf.write(f"- **相机位姿状态**: `{cam_pose_str}`\n")
                        rf.write(f"- **重投影误差**: `{rmse_str}`\n")
                        rf.write(f"- **抖动稳定性**: `{jitter_str}`\n")
                        rf.write(f"- **参与求解标靶**: `{matched_tags}`\n")
                        if blind_summary:
                            rf.write(f"\n### 留一盲测交叉验证结果 (Leave-One-Out Result)\n\n")
                            rf.write(f"- **盲测验证目标**: `Tag #{blind_summary['tag_id']}` (解算时已强制屏蔽)\n")
                            rf.write(f"- **反推像元误差**: `{blind_summary['err_px']:.2f} px`\n")
                            rf.write(f"- **空间几何偏差**: `{blind_summary['err_mm']:.2f} mm`\n")
                            rf.write(f"- **目标测距深度**: `{blind_summary['depth_mm']:.1f} mm`\n")
                            rf.write(f"- **支撑反推标靶**: `{blind_summary['solved_by']}`\n")

                    self.set_toast(f"质检快照与报告已保存至 tag_calibration_verification/")
                    print(f"[OK] 验证质检图已保存: {snap_path}")
                    print(f"[OK] 验证质检单已保存: {report_path}")

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AprilTag 标定精度与 3D 坐标系在线实时 AR 验证")
    parser.add_argument("--map", type=str, default=DEFAULT_MAP_PATH, help="标靶空间地图路径")
    parser.add_argument("--mock", action="store_true", help="仿真演示模式 (无需物理硬件)")
    parser.add_argument("--blind_tag", type=int, default=None, help="默认启用的留一盲测标靶 ID (如 26)")
    args = parser.parse_args()

    verifier = TagCalibrationVerifier(map_path=args.map, mock_mode=args.mock)
    if args.blind_tag is not None:
        verifier.blind_target_tag_id = args.blind_tag
    verifier.run()


if __name__ == "__main__":
    main()
