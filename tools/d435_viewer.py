"""
D435 实时彩色/对齐深度流可视化与深度探针交互工具
=====================================================
用途：
  1. 实时预览 RealSense D435 的 RGB 画面与对齐深度热力图
  2. 支持多种排版布局切换：[V] 键在【上下排列】/【单图放大(仅RGB)】/【左右排列】/【单图(仅深度)】之间切换
  3. 支持 [Space] 空格键一键定格/暂停画面，彻底解决高频刷新花屏闪烁，便于细致观察
  4. 实时芦笋特征提取与顶层抓取点标注，按 [G] 打印 SCARA G-code
  5. 叠加有效作业 ROI 区域边框，清晰展现感知分析过程
  6. 鼠标悬停/点击任意点查看毫米级深度与 (X, Y, Z) 空间坐标
  7. 支持 --mock 模式：无需物理相机即可模拟仿真场景
"""

import os
import sys
import time
from datetime import datetime
import argparse
import yaml
import numpy as np
import cv2

# 解决 Windows 控制台中文编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False

# 导入芦笋特征分析与最顶层判决核心模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vision.asparagus_analyzer import AsparagusAnalyzer, AsparagusTarget


class D435Viewer:
    def __init__(self, config_path: str = "config.yaml", mock_mode: bool = False):
        self.config_path = config_path
        self.mock_mode = mock_mode
        self.load_config()

        self.pipeline = None
        self.rs_config = None
        self.align = None

        if HAVE_REALSENSE and not self.mock_mode:
            self.pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.align = rs.align(rs.stream.color)

        self.filters_enabled = True
        self.laser_enabled = True
        self.init_filters()

        # 芦笋视觉特征分析器
        self.detection_enabled = True
        self.analyzer = AsparagusAnalyzer()
        if self.t_cam_to_scara is not None:
            self.analyzer.set_hand_eye_matrix(self.t_cam_to_scara)
        if self.tag_localizer is not None:
            self.analyzer.set_tag_localizer(self.tag_localizer)
        self.latest_targets = []

        # 鼠标交互状态
        self.hover_x = -1
        self.hover_y = -1
        self.selected_point = None

        self.color_intrinsics = None

        # 深度色彩映射范围 (米) - 默认 0.48m ~ 0.66m (适配当前 640mm 工作台)
        self.cmap_min = 0.48
        self.cmap_max = 0.66
        self.auto_range = False

        # --- 视图与交互升级特性 ---
        # 深度图呈现模式: "height_map" (传送带纠偏相对高度图, 台面Z=0) 或 "raw_depth" (相机原生镜头绝对深度)
        self.depth_display_mode = "height_map"
        self.max_height_mm = 80.0       # 纠偏高度图满量程刻度 (mm)

        # 排版模式: "split_v" (上下排列), "rgb_only" (单图放大仅看RGB), "split_h" (左右并排), "depth_only" (仅深度图)
        self.view_mode = "split_v"
        self.is_paused = False          # 空格键暂停/定格模式，消除花屏闪烁
        self.show_roi = True            # 显示有效作业 ROI 区域框
        self.paused_color_frame = None
        self.paused_depth_frame = None

    def load_config(self):
        """加载 config.yaml 中的相机与滤波配置"""
        default_config = {
            "camera": {
                "color": {"width": 1280, "height": 720, "fps": 30},
                "depth": {"width": 848, "height": 480, "fps": 30},
                "visual_preset": 3,
                "laser_power": 120,
                "filters": {
                    "spatial": {"enabled": True, "smooth_alpha": 0.5, "smooth_delta": 20, "magnitude": 2, "hole_fill": 1},
                    "temporal": {"enabled": True, "smooth_alpha": 0.4, "smooth_delta": 20, "persistence_control": 3},
                    "threshold": {"enabled": True, "min_distance": 0.40, "max_distance": 0.70},
                },
                "colormap": {
                    "min_distance": 0.48,
                    "max_distance": 0.66,
                    "auto_range": False
                }
            }
        }
        self.t_cam_to_scara = None
        self.tag_localizer = None
        self.safe_z = 80.0
        self.drop_x = 220.0
        self.drop_y = 0.0

        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    if "camera" in loaded:
                        default_config["camera"].update(loaded["camera"])
                    if "calibration" in loaded:
                        self.t_cam_to_scara = loaded["calibration"].get("t_cam_to_scara", None)
                        # AprilTag 多标靶地图定位
                        tags_map_path = loaded["calibration"].get("tags_map_path", "")
                        if tags_map_path and os.path.exists(tags_map_path):
                            try:
                                from src.vision.tag_localizer import TagLocalizer
                                self.tag_localizer = TagLocalizer(tags_map_path=tags_map_path)
                                print(f"[D435Viewer] AprilTag 地图已加载: {tags_map_path}")
                            except Exception as e:
                                print(f"[D435Viewer] AprilTag 定位器加载失败: {e}")
                    if "robot" in loaded:
                        self.safe_z = float(loaded["robot"].get("safe_z_mm", 80.0))
                        self.drop_x = float(loaded["robot"].get("drop_x_mm", 220.0))
                        self.drop_y = float(loaded["robot"].get("drop_y_mm", 0.0))

        self.cfg = default_config["camera"]
        self.cmap_min = self.cfg["colormap"].get("min_distance", 0.48)
        self.cmap_max = self.cfg["colormap"].get("max_distance", 0.66)
        self.auto_range = self.cfg["colormap"].get("auto_range", False)

        if hasattr(self, 'analyzer') and self.analyzer:
            self.analyzer.set_hand_eye_matrix(self.t_cam_to_scara)
            if self.tag_localizer is not None:
                self.analyzer.set_tag_localizer(self.tag_localizer)

    def init_filters(self):
        """初始化 RealSense 后处理滤波模块"""
        if not HAVE_REALSENSE or self.mock_mode:
            return

        f_cfg = self.cfg["filters"]
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, f_cfg["spatial"]["smooth_alpha"])
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, f_cfg["spatial"]["smooth_delta"])
        self.spatial_filter.set_option(rs.option.filter_magnitude, f_cfg["spatial"]["magnitude"])
        self.spatial_filter.set_option(rs.option.holes_fill, f_cfg["spatial"]["hole_fill"])

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, f_cfg["temporal"]["smooth_alpha"])
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, f_cfg["temporal"]["smooth_delta"])
        self.temporal_filter.set_option(rs.option.holes_fill, f_cfg["temporal"]["persistence_control"])

        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, f_cfg["threshold"]["min_distance"])
        self.threshold_filter.set_option(rs.option.max_distance, f_cfg["threshold"]["max_distance"])

    def apply_filters(self, depth_frame):
        """对深度帧执行 SDK 级硬件滤波链"""
        if not self.filters_enabled or not HAVE_REALSENSE or self.mock_mode:
            return depth_frame
        try:
            filtered = self.threshold_filter.process(depth_frame)
            filtered = self.spatial_filter.process(filtered)
            filtered = self.temporal_filter.process(filtered)
            return filtered
        except Exception:
            return depth_frame

    def start(self):
        """启动设备数据流"""
        if self.mock_mode or not HAVE_REALSENSE:
            print("[INFO] 正在以仿真模拟模式 (--mock) 启动虚拟相机流...")
            self.actual_w, self.actual_h = 1280, 720
            return

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未检测到任何物理连接的 RealSense 设备，请插入 USB 接口或使用 --mock 模式运行。")

        dev = devices[0]
        dev_name = dev.get_info(rs.camera_info.name)
        dev_sn = dev.get_info(rs.camera_info.serial_number)
        usb_desc = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"

        print(f"[INFO] 成功连接设备: {dev_name} (S/N: {dev_sn}, USB 模式: {usb_desc})")

        c_w, c_h, fps = self.cfg["color"]["width"], self.cfg["color"]["height"], self.cfg["color"]["fps"]
        d_w, d_h = self.cfg["depth"]["width"], self.cfg["depth"]["height"]

        if "2." in usb_desc:
            print("[WARN] [!] 检测到当前相机工作在 USB 2.1 带宽下（建议连接电脑蓝色 USB 3.0 端口）。")
            print("[INFO] [*] 正在自动启用 USB 2.1 自适应高清/流畅流配置...")
            self.rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        else:
            self.rs_config.enable_stream(rs.stream.color, c_w, c_h, rs.format.bgr8, fps)
            self.rs_config.enable_stream(rs.stream.depth, d_w, d_h, rs.format.z16, fps)

        profile = self.pipeline.start(self.rs_config)
        self.apply_device_settings(profile)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_stream.get_intrinsics()
        self.actual_w = self.color_intrinsics.width
        self.actual_h = self.color_intrinsics.height
        self.analyzer.update_intrinsics(
            self.color_intrinsics.fx, self.color_intrinsics.fy,
            self.color_intrinsics.ppx, self.color_intrinsics.ppy
        )

    def apply_device_settings(self, profile):
        """配置预设与红外发射功率"""
        try:
            adv_mode = rs.rs400_advanced_mode(profile.get_device())
            if adv_mode.is_enabled():
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_sensor.set_option(rs.option.visual_preset, float(self.cfg.get("visual_preset", 3)))
        except Exception:
            pass

        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, float(self.cfg.get("laser_power", 120)))
        except Exception:
            pass

    def generate_mock_frame(self):
        """生成仿真三层芦笋堆叠数据 (匹配真实工况)"""
        h, w = 720, 1280
        # 工作台背景 (黑色传送带, 深度 640mm)
        color_img = np.full((h, w, 3), (25, 25, 25), dtype=np.uint8)
        depth_img = np.full((h, w), 640, dtype=np.uint16)

        # 构造三根相互叠压的芦笋 (底面 -> 中层 -> 顶层)
        # 芦笋 1 (底层): 深度 580mm (凸起 60mm)
        cv2.ellipse(color_img, (640, 420), (160, 16), -8, 0, 360, (30, 110, 35), -1)
        mask1 = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask1, (640, 420), (160, 16), -8, 0, 360, 255, -1)
        depth_img[mask1 > 0] = 580

        # 芦笋 2 (中层): 深度 555mm (凸起 85mm)
        cv2.ellipse(color_img, (620, 370), (150, 14), 5, 0, 360, (35, 120, 40), -1)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask2, (620, 370), (150, 14), 5, 0, 360, 255, -1)
        depth_img[mask2 > 0] = 555

        # 芦笋 3 (最顶层): 深度 525mm (凸起 115mm)
        cv2.ellipse(color_img, (600, 320), (140, 13), -2, 0, 360, (45, 140, 50), -1)
        mask3 = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask3, (600, 320), (140, 13), -2, 0, 360, 255, -1)
        depth_img[mask3 > 0] = 525

        return color_img, depth_img

    def on_mouse(self, event, x, y, flags, param):
        """鼠标移动/点击事件处理 (自适应多种布局模式坐标映射)"""
        view_mode = getattr(self, "view_mode", "split_v")
        orig_w = getattr(self, "actual_w", 1280)
        orig_h = getattr(self, "actual_h", 720)

        # 获取当前窗口尺寸
        cw = param.get("current_w", 960)
        ch = param.get("current_h", 1080)

        u, v = -1, -1

        if view_mode == "rgb_only":
            # 单图放大模式：整个窗口对应 RGB 画面
            u = int(x * orig_w / cw)
            v = int(y * orig_h / ch)
        elif view_mode == "split_v":
            # 上下排列模式：上半屏 (0 ~ ch/2) 为 RGB 画面，下半屏为深度图
            half_h = ch // 2
            if y < half_h:
                u = int(x * orig_w / cw)
                v = int(y * orig_h / half_h)
            else:
                u = int(x * orig_w / cw)
                v = int((y - half_h) * orig_h / half_h)
        elif view_mode == "split_h":
            # 左右排列模式：左半屏为 RGB，右半屏为深度图
            half_w = cw // 2
            local_x = x % half_w
            u = int(local_x * orig_w / half_w)
            v = int(y * orig_h / ch)
        elif view_mode == "depth_only":
            u = int(x * orig_w / cw)
            v = int(y * orig_h / ch)

        if 0 <= u < orig_w and 0 <= v < orig_h:
            self.hover_x = u
            self.hover_y = v
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selected_point = (u, v)

    def draw_roi_bounds(self, img):
        """在画面上绘制中心有效分析 ROI 区域框"""
        h, w = img.shape[:2]
        r_x1, r_y1 = int(w * 0.04), int(h * 0.04)
        r_x2, r_y2 = int(w * 0.96), int(h * 0.96)
        # 绘制亮黄色半透明作业有效线
        cv2.rectangle(img, (r_x1, r_y1), (r_x2, r_y2), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "WORK_ROI (有效分析作业区)", (r_x1 + 8, r_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    def run(self):
        """主可视化与交互事件循环"""
        self.start()

        window_name = "RealSense D435 芦笋 3D 视觉智能查看器"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        mouse_params = {"current_w": 960, "current_h": 1080}
        cv2.setMouseCallback(window_name, self.on_mouse, mouse_params)

        print("\n" + "=" * 68)
        print(" RealSense D435 实时交互指南 (支持暂停定格与上下/单图放大):")
        print("   - [Space] 空格键: 【定格/暂停画面】(静止当前帧，解决花屏闪烁，从容查看)")
        print("   - [V] 键: 切换排版视图 (上下排列 -> 单图全屏RGB -> 左右排列 -> 仅深度图)")
        print("   - [O] 键: 切换作业 ROI 边框显示 (开/关)")
        print("   - [H] 键: 切换【传送带纠偏相对高度图 (Z=0拉平)】与【相机原生深度图】")
        print("   - [D] 键: 切换芦笋识别与顶层尺寸/位姿解算 (开/关)")
        print("   - [G] 键: 打印当前最顶层芦笋 SCARA G-code 抓取指令")
        print("   - [S] 键: 抓拍并保存当前帧数据至 data/snapshots/")
        print("   - [A] 键: 切换【固定深度范围】与【动态自适应色阶 Auto-Range】")
        print("   - [ [ ] / [ ] ] 键: 微调近端色阶下限 (+/- 1cm)")
        print("   - [ - ] / [ = ] 键: 微调远端色阶上限 (+/- 1cm)")
        print("   - [R] 键: 重置色彩区间为 0.48m ~ 0.66m")
        print("   - [F] 键: 切换硬件后处理滤波器 (开/关)")
        print("   - [L] 键: 切换激光散斑发射器 (开/关)")
        print("   - [Q] 或 [ESC]: 退出查看器")
        print("=" * 68 + "\n")

        fps_counter = 0
        fps_time = time.time()
        current_fps = 30.0

        try:
            while True:
                # 1. 采集或保持定格数据帧
                if not self.is_paused:
                    if self.mock_mode:
                        color_image, depth_image = self.generate_mock_frame()
                        time.sleep(0.03)
                    else:
                        frames = self.pipeline.wait_for_frames()
                        aligned_frames = self.align.process(frames)
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()

                        if not color_frame or not depth_frame:
                            continue

                        filtered_depth = self.apply_filters(depth_frame)
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(filtered_depth.get_data())

                    self.paused_color_frame = color_image.copy()
                    self.paused_depth_frame = depth_image.copy()
                else:
                    # 暂停定格状态：直接复用定格帧，不发生刷新跳变
                    color_image = self.paused_color_frame.copy()
                    depth_image = self.paused_depth_frame.copy()
                    time.sleep(0.03)

                # 2. 传送带基准平面拟合与纠偏相对高度图渲染 (Z=0 平面基准)
                current_plane_coeff = self.analyzer.fit_table_plane(depth_image)
                pitch_deg, roll_deg, tilt_deg = self.analyzer.get_table_tilt_angles(current_plane_coeff)

                # 生成 A: 传送带纠偏相对高度图 (Height Map / Elevation Map, 黑色皮带拉平为 Z=0)
                height_colormap = self.analyzer.render_height_map(depth_image, current_plane_coeff, max_h_mm=self.max_height_mm)
                self.current_height_colormap = height_colormap.copy()

                # 生成 B: 相机原生镜头绝对深度图 (Raw Depth Colormap)
                depth_meters = depth_image.astype(float) * 0.001
                if self.auto_range:
                    valid_mask = (depth_image > 200) & (depth_image < 1500)
                    if np.count_nonzero(valid_mask) > 100:
                        act_min = float(np.percentile(depth_meters[valid_mask], 2))
                        act_max = float(np.percentile(depth_meters[valid_mask], 98))
                        act_max = max(act_max, act_min + 0.03)
                    else:
                        act_min, act_max = self.cmap_min, self.cmap_max
                else:
                    act_min, act_max = self.cmap_min, self.cmap_max

                norm = np.clip((act_max - depth_meters) / max(0.005, (act_max - act_min)) * 255.0, 0, 255).astype(np.uint8)
                raw_depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                raw_depth_colormap[depth_image == 0] = (25, 25, 25)

                # 根据当前显示模式选择底图并绘制模式标识
                if self.depth_display_mode == "height_map":
                    depth_colormap = height_colormap
                    mode_tag = f"[纠偏高度图 / Z=0皮带基准] 倾角 P:{pitch_deg:+.1f} R:{roll_deg:+.1f} | 满程: 0~{int(self.max_height_mm)}mm (按[H]切深度)"
                    cv2.rectangle(depth_colormap, (8, 6), (560, 30), (15, 15, 15), -1)
                    cv2.putText(depth_colormap, mode_tag, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)
                else:
                    depth_colormap = raw_depth_colormap
                    mode_tag = f"[相机原生镜头深度] 绝对深度: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m (按[H]切高度图)"
                    cv2.rectangle(depth_colormap, (8, 6), (560, 30), (15, 15, 15), -1)
                    cv2.putText(depth_colormap, mode_tag, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)

                # 3. 实时芦笋特征提取与顶层判决
                raw_color_for_save = color_image.copy()
                if self.detection_enabled:
                    try:
                        self.latest_targets = self.analyzer.analyze(color_image, depth_image)
                        display_color_image = self.analyzer.draw_detections(color_image, self.latest_targets)
                    except Exception as e:
                        # 避免刷屏，每隔 3 秒打印一次异常报告
                        curr_t = time.time()
                        if getattr(self, '_last_err_print_t', 0) + 3.0 < curr_t:
                            self._last_err_print_t = curr_t
                            print(f"\n[ERROR in analyze] 感知解算发生异常: {e}")
                        display_color_image = color_image.copy()
                else:
                    display_color_image = color_image.copy()

                # 叠加 ROI 边框显示
                if self.show_roi:
                    self.draw_roi_bounds(display_color_image)

                # 4. 鼠标探针测距与 3D 坐标反投影 (融合传送带相对净高与空间坐标)
                target_pt = self.selected_point if self.selected_point else (self.hover_x, self.hover_y)
                probe_text = "Probe: N/A (移动鼠标探测台面相对高度与 3D 坐标)"
                if target_pt[0] >= 0 and target_pt[1] >= 0:
                    px, py = target_pt
                    if py < depth_image.shape[0] and px < depth_image.shape[1]:
                        depth_mm = depth_image[py, px]
                        if current_plane_coeff is not None and depth_mm > 0:
                            table_z_local = current_plane_coeff[0] * px + current_plane_coeff[1] * py + current_plane_coeff[2]
                            rel_h_val = float(table_z_local - depth_mm)
                            rel_str = f"台面净高:+{rel_h_val:.1f}mm" if rel_h_val >= 0 else f"台面净高:{rel_h_val:.1f}mm"
                        else:
                            rel_str = f"台面净高:+{max(0.0, 640.0 - depth_mm):.1f}mm" if depth_mm > 0 else "台面净高:0mm"

                        if depth_mm > 0 and self.color_intrinsics:
                            pt_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [px, py], depth_mm * 0.001)
                            probe_text = f"点:({px},{py}) | 【{rel_str}】 | 镜头深度:{depth_mm}mm | 空间3D:(X:{pt_3d[0]*1000:+.1f}, Y:{pt_3d[1]*1000:+.1f}, Z:{pt_3d[2]*1000:+.1f})mm"
                        elif depth_mm > 0:
                            probe_text = f"点:({px},{py}) | 【{rel_str}】 | 镜头深度:{depth_mm}mm"
                        else:
                            probe_text = f"点:({px},{py}) | 【无效盲区】(0mm)"

                    cv2.drawMarker(display_color_image, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.drawMarker(depth_colormap, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()

                # 5. 根据当前视图模式合成最终画布
                # 模式 A: 单图大屏高清模式 (仅看 RGB 彩色图，画幅最大)
                if self.view_mode == "rgb_only":
                    # 采用 1024x576 或全屏比例
                    disp_w, disp_h = 1024, 576
                    final_canvas = cv2.resize(display_color_image, (disp_w, disp_h))
                    mouse_params["current_w"] = disp_w
                    mouse_params["current_h"] = disp_h

                # 模式 B: 上下排列模式 (上图为大画幅 RGB，下图为深度热力图)
                elif self.view_mode == "split_v":
                    disp_w, disp_h = 960, 480
                    c_up = cv2.resize(display_color_image, (disp_w, disp_h))
                    d_dn = cv2.resize(depth_colormap, (disp_w, disp_h))
                    # 绘制分割线
                    cv2.line(d_dn, (0, 0), (disp_w, 0), (80, 80, 80), 2)
                    final_canvas = np.vstack((c_up, d_dn))
                    mouse_params["current_w"] = disp_w
                    mouse_params["current_h"] = disp_h * 2

                # 模式 C: 左右并排模式
                elif self.view_mode == "split_h":
                    disp_w, disp_h = 640, 360
                    c_left = cv2.resize(display_color_image, (disp_w, disp_h))
                    d_right = cv2.resize(depth_colormap, (disp_w, disp_h))
                    final_canvas = np.hstack((c_left, d_right))
                    mouse_params["current_w"] = disp_w * 2
                    mouse_params["current_h"] = disp_h

                # 模式 D: 仅看深度图
                else:
                    disp_w, disp_h = 1024, 576
                    final_canvas = cv2.resize(depth_colormap, (disp_w, disp_h))
                    mouse_params["current_w"] = disp_w
                    mouse_params["current_h"] = disp_h

                canvas_w = final_canvas.shape[1]
                canvas_h = final_canvas.shape[0]

                # 6. 状态顶条与定格提醒
                topmost = next((t for t in self.latest_targets if t.is_topmost), None) if self.detection_enabled else None
                view_name_map = {"split_v": "上下排列", "rgb_only": "单图RGB高清", "split_h": "左右并排", "depth_only": "仅深度图"}

                # 计算标定状态 OSD 标识
                tag_info = getattr(self.analyzer, 'last_tag_info', {})
                if tag_info and tag_info.get("static_tags_count", 0) >= 2:
                    cal_osd_str = f"[TAG] {tag_info['static_tags_count']}静止标靶|重投影{tag_info.get('reprojection_error_px', 0.0):.2f}px"
                elif self.analyzer and self.analyzer.last_valid_tag_transform is not None:
                    cal_osd_str = "[TAG-CACHE 历史锁定]"
                elif getattr(self.analyzer, 'is_hand_eye_calibrated', False):
                    cal_osd_str = "[HAND-EYE 手工标定]"
                else:
                    cal_osd_str = "[未标定防撞]"

                if self.is_paused:
                    # 暂停定格状态：醒目黄色条提醒
                    cv2.rectangle(final_canvas, (0, 0), (canvas_w, 32), (0, 140, 255), -1)
                    pause_msg = f"{cal_osd_str} [PAUSED / 画面已定格] 按 [Space] 恢复播放 | 检出芦笋: {len(self.latest_targets)} 根 | 视图: {view_name_map.get(self.view_mode)}"
                    cv2.putText(final_canvas, pause_msg, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
                else:
                    if topmost:
                        top_banner = f"{cal_osd_str} [TOPMOST] L:{topmost.length_mm:.1f}mm D:{topmost.diam_mm:.1f}mm (+{topmost.rel_height_mm:.1f}mm) | Grip:(X:{topmost.grip_x:.1f}, Y:{topmost.grip_y:.1f}, Z:{topmost.grip_z:.1f}) R:{topmost.yaw_deg:.1f}°"
                        cv2.rectangle(final_canvas, (0, 0), (canvas_w, 30), (15, 55, 15), -1)
                        cv2.putText(final_canvas, top_banner, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
                    else:
                        live_banner = f"{cal_osd_str} | LIVE [{view_name_map.get(self.view_mode)}] | 检出: {len(self.latest_targets)} 根 | FPS: {current_fps:.1f} | 按[Space]可定格画面"
                        cv2.rectangle(final_canvas, (0, 0), (canvas_w, 28), (35, 35, 35), -1)
                        cv2.putText(final_canvas, live_banner, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # 底部探针信息栏
                cv2.rectangle(final_canvas, (0, canvas_h - 32), (canvas_w, canvas_h), (20, 20, 20), -1)
                cv2.putText(final_canvas, probe_text, (12, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

                cv2.imshow(window_name, final_canvas)

                # 7. 键盘响应
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break

                # [Space] 一键暂停/定格画面
                elif key == 32:  # 空格键
                    self.is_paused = not self.is_paused
                    state_str = "【已定格暂停】(画面已静止，可从容观察分析)" if self.is_paused else "【已恢复实时流】"
                    print(f"\n[ACTION] 画面状态: {state_str}")

                # [V] 切换排版布局视图
                elif key in [ord('v'), ord('V')]:
                    modes = ["split_v", "rgb_only", "split_h", "depth_only"]
                    curr_idx = modes.index(self.view_mode)
                    self.view_mode = modes[(curr_idx + 1) % len(modes)]
                    print(f"\n[ACTION] 视图模式切换为: 【{view_name_map.get(self.view_mode)}】")

                # [O] 切换 ROI 边框显示
                elif key in [ord('o'), ord('O')]:
                    self.show_roi = not self.show_roi
                    print(f"[ACTION] 作业 ROI 边框显示: {'开启' if self.show_roi else '关闭'}")

                # [H] 切换传送带纠偏相对高度图 / 相机原生绝对深度图
                elif key in [ord('h'), ord('H')]:
                    if self.depth_display_mode == "height_map":
                        self.depth_display_mode = "raw_depth"
                        print("\n[ACTION] 视图切换为: 【相机原生镜头绝对深度图 (Raw Depth)】")
                    else:
                        self.depth_display_mode = "height_map"
                        print("\n[ACTION] 视图切换为: 【传送带纠偏相对高度图 (Height Map)】(黑色传送带拉平置零 Z=0)")

                # [S] 抓拍保存当前帧
                elif key == ord('s'):
                    self.save_snapshot(raw_color_for_save, depth_image, depth_colormap, getattr(self, 'current_height_colormap', None))

                # [D] 芦笋识别与顶层解算开关
                elif key == ord('d'):
                    self.detection_enabled = not self.detection_enabled
                    print(f"[ACTION] 芦笋特征识别与顶层解算: {'开启' if self.detection_enabled else '关闭'}")

                # [G] 打印安全规范的 SCARA G-code
                elif key == ord('g'):
                    topmost = next((t for t in self.latest_targets if t.is_topmost), None)
                    if topmost:
                        print("\n" + topmost.generate_gcode(safe_z=self.safe_z, drop_x=self.drop_x, drop_y=self.drop_y) + "\n")
                    else:
                        # 检查是否有鼠标左键点击的物理点
                        if self.selected_point is not None:
                            px, py = self.selected_point
                            if py < depth_image.shape[0] and px < depth_image.shape[1]:
                                d_val = float(depth_image[py, px])
                                if d_val > 0:
                                    z_cam = d_val
                                    x_cam = (px - self.analyzer.cx) * z_cam / self.analyzer.fx
                                    y_cam = (py - self.analyzer.cy) * z_cam / self.analyzer.fy
                                    plane_coeff = self.analyzer.fit_table_plane(depth_image)
                                    if plane_coeff is not None:
                                        table_z = plane_coeff[0] * px + plane_coeff[1] * py + plane_coeff[2]
                                        rel_h = max(10.0, float(table_z - z_cam))
                                    else:
                                        rel_h = 25.0

                                    manual_t = AsparagusTarget(
                                        id=999,
                                        center_px=(float(px), float(py)),
                                        length_px=200.0,
                                        diam_px=20.0,
                                        yaw_deg=0.0,
                                        axis_vector=(1.0, 0.0),
                                        box_corners=np.zeros((4, 2), dtype=np.int32),
                                        contour=np.zeros((1, 1, 2), dtype=np.int32),
                                        length_mm=180.0,
                                        diam_mm=15.0,
                                        grip_x=round(x_cam, 1),
                                        grip_y=round(y_cam, 1),
                                        grip_z=round(z_cam, 1),
                                        z_top=round(z_cam, 1),
                                        rel_height_mm=round(rel_h, 1),
                                        robot_x=round(x_cam, 1),
                                        robot_y=round(y_cam, 1),
                                        robot_z=round(rel_h, 1),
                                        robot_r=0.0,
                                        is_topmost=True,
                                        calibration_source="uncalibrated"
                                    )
                                    print("\n[INFO] 自动检测未锁定顶层，但已根据【鼠标点选示教点】一键生成防撞 G-code:")
                                    print(manual_t.generate_gcode(safe_z=self.safe_z, drop_x=self.drop_x, drop_y=self.drop_y) + "\n")
                                    continue

                        print("\n" + "=" * 70)
                        print("[!] 视野内当前未自动检测到符合规范的芦笋目标，正在执行现场感知即时诊断...")
                        print("-" * 70)
                        diag_info = self.analyzer.diagnose(color_image, depth_image)
                        print(diag_info)
                        print("=" * 70 + "\n")

                # [A] 动态自适应色阶
                elif key == ord('a'):
                    self.auto_range = not self.auto_range
                    status = "动态自适应 (Auto-Range)" if self.auto_range else f"固定区间 [{self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m]"
                    print(f"[ACTION] 色彩映射模式切换: {status}")

                # [ [ ] / [ ] ] 微调近端下限
                elif key == ord('['):
                    self.cmap_min = max(0.10, round(self.cmap_min - 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整近端色阶 min: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")
                elif key == ord(']'):
                    self.cmap_min = min(self.cmap_max - 0.02, round(self.cmap_min + 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整近端色阶 min: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")

                # [ - ] / [ = ] 微调远端上限
                elif key == ord('-'):
                    self.cmap_max = max(self.cmap_min + 0.02, round(self.cmap_max - 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整远端色阶 max: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")
                elif key == ord('='):
                    self.cmap_max = min(2.0, round(self.cmap_max + 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整远端色阶 max: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")

                # [R] 重置色阶
                elif key == ord('r'):
                    self.cmap_min = 0.48
                    self.cmap_max = 0.66
                    self.auto_range = False
                    print(f"[ACTION] 色彩区间已重置为现场标准: 0.48m ~ 0.66m")

                # [F] 硬件后处理滤波开关
                elif key == ord('f'):
                    self.filters_enabled = not self.filters_enabled
                    print(f"[ACTION] 深度后处理滤波器: {'开启' if self.filters_enabled else '关闭'}")

                # [L] 红外散斑发射器开关
                elif key == ord('l') and not self.mock_mode:
                    depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        self.laser_enabled = not self.laser_enabled
                        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if self.laser_enabled else 0.0)
                        print(f"[ACTION] 红外散斑发射器: {'开启' if self.laser_enabled else '关闭'}")

        finally:
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            print("[INFO] 采集与可视化窗口已安全退出。")

    def save_snapshot(self, color_img, depth_raw, depth_color, height_color=None):
        """抓拍并保存当前帧数据 (含彩色图、原生深度图、纠偏高度图、原始点云矩阵)"""
        save_dir = os.path.join("data", "snapshots")
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        rgb_path = os.path.join(save_dir, f"color_{ts}.png")
        depth_vis_path = os.path.join(save_dir, f"depth_vis_{ts}.png")
        height_vis_path = os.path.join(save_dir, f"height_vis_{ts}.png")
        depth_raw_path = os.path.join(save_dir, f"depth_raw_{ts}.npy")

        cv2.imwrite(rgb_path, color_img)
        cv2.imwrite(depth_vis_path, depth_color)
        if height_color is not None:
            cv2.imwrite(height_vis_path, height_color)
        np.save(depth_raw_path, depth_raw)

        print(f"\n[SNAPSHOT] 数据已抓拍保存:")
        print(f"  -> 彩色图:   {rgb_path}")
        print(f"  -> 深度热力: {depth_vis_path}")
        if height_color is not None:
            print(f"  -> 纠偏高度: {height_vis_path}")
        print(f"  -> 原始深度: {depth_raw_path} (uint16 mm)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense D435 实时深度探针与对齐查看工具")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--mock", action="store_true", help="强制以仿真模式运行")
    args = parser.parse_args()

    try:
        viewer = D435Viewer(config_path=args.config, mock_mode=args.mock)
        viewer.run()
    except Exception as e:
        print(f"\n[ERROR] 运行中断: {e}")
