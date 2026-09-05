"""
D435 实时彩色/对齐深度流可视化与深度探针工具
=====================================================
用途：
  1. 实时预览 RealSense D435 的 RGB 画面与对齐深度热力图
  2. 验证 70cm 架设高度下的视野覆盖与反光情况
  3. 鼠标悬停/点击查看任意像素点的毫米级深度与 (X, Y, Z) 3D 相机坐标
  4. 按 's' 键快速抓拍当前对齐数据帧供离线算法调试
  5. 按 'f' 键动态切换后处理滤波器的开启与关闭，对比抗噪效果
  6. 支持 --mock 模式：无需物理相机即可模拟 70cm 俯视下的 3 层堆叠芦笋场景
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
        self.latest_targets = []

        # 鼠标交互状态
        self.hover_x = -1
        self.hover_y = -1
        self.selected_point = None

        self.color_intrinsics = None

        # 深度色彩映射范围 (米) - 默认 0.40m ~ 0.55m
        self.cmap_min = 0.40
        self.cmap_max = 0.55
        self.auto_range = False

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
                    "threshold": {"enabled": True, "min_distance": 0.35, "max_distance": 0.65},
                },
                "colormap": {
                    "min_distance": 0.40,
                    "max_distance": 0.55,
                    "auto_range": False
                }
            }
        }
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and "camera" in loaded:
                    default_config["camera"].update(loaded["camera"])
        self.cfg = default_config["camera"]
        
        cmap_cfg = self.cfg.get("colormap", {})
        self.cmap_min = float(cmap_cfg.get("min_distance", 0.40))
        self.cmap_max = float(cmap_cfg.get("max_distance", 0.55))
        self.auto_range = bool(cmap_cfg.get("auto_range", False))

    def init_filters(self):
        """初始化 SDK 硬件后处理滤波器"""
        if not HAVE_REALSENSE or self.mock_mode:
            return

        self.spatial_filter = rs.spatial_filter()
        f_cfg = self.cfg.get("filters", {}).get("spatial", {})
        if f_cfg:
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, f_cfg.get("smooth_alpha", 0.5))
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, f_cfg.get("smooth_delta", 20))
            self.spatial_filter.set_option(rs.option.filter_magnitude, f_cfg.get("magnitude", 2))
            self.spatial_filter.set_option(rs.option.holes_fill, f_cfg.get("hole_fill", 1))

        self.temporal_filter = rs.temporal_filter()
        t_cfg = self.cfg.get("filters", {}).get("temporal", {})
        if t_cfg:
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, t_cfg.get("smooth_alpha", 0.4))
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, t_cfg.get("smooth_delta", 20))
            self.temporal_filter.set_option(rs.option.holes_fill, t_cfg.get("persistence_control", 3))

        self.threshold_filter = rs.threshold_filter()
        th_cfg = self.cfg.get("filters", {}).get("threshold", {})
        if th_cfg:
            self.threshold_filter.set_option(rs.option.min_distance, th_cfg.get("min_distance", 0.35))
            self.threshold_filter.set_option(rs.option.max_distance, th_cfg.get("max_distance", 0.65))

    def apply_filters(self, depth_frame):
        """按链条顺序应用后处理滤波"""
        if self.mock_mode or not self.filters_enabled:
            return depth_frame

        f = depth_frame
        f_cfg = self.cfg.get("filters", {})
        if f_cfg.get("threshold", {}).get("enabled", True):
            f = self.threshold_filter.process(f)
        if f_cfg.get("spatial", {}).get("enabled", True):
            f = self.spatial_filter.process(f)
        if f_cfg.get("temporal", {}).get("enabled", True):
            f = self.temporal_filter.process(f)
        return f

    def start(self):
        """启动 D435 采集流或进入 Mock 模式（智能自适应 USB 2.1 / USB 3.0）"""
        if self.mock_mode:
            print("[INFO] 正在运行于 MOCK 仿真模式（生成 70cm 俯视 3 层叠压芦笋场景）")
            self.actual_w = 1280
            self.actual_h = 720
            if HAVE_REALSENSE:
                self.color_intrinsics = rs.intrinsics()
                self.color_intrinsics.width = 1280
                self.color_intrinsics.height = 720
                self.color_intrinsics.ppx = 640.0
                self.color_intrinsics.ppy = 360.0
                self.color_intrinsics.fx = 920.0
                self.color_intrinsics.fy = 920.0
                self.color_intrinsics.model = rs.distortion.none
            return

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("[WARN] 未检测到物理 RealSense 设备，自动回退到 --mock 仿真模式！")
            self.mock_mode = True
            self.start()
            return

        dev = devices[0]
        dev_name = dev.get_info(rs.camera_info.name)
        dev_sn = dev.get_info(rs.camera_info.serial_number)
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"
        print(f"[INFO] 成功连接设备: {dev_name} (S/N: {dev_sn}, USB 模式: {usb_type})")

        is_usb2 = "2." in str(usb_type)
        if is_usb2:
            print("[WARN] [!] 检测到当前相机工作在 USB 2.1 带宽下（若需 1280x720@30fps 满带宽，建议连接电脑蓝色 USB 3.0 端口）。")
            print("[INFO] [*] 正在自动启用 USB 2.1 自适应高清/流畅流配置...")

        candidate_configs = []
        if not is_usb2:
            candidate_configs.append({
                "c_w": self.cfg["color"]["width"], "c_h": self.cfg["color"]["height"], "c_fps": self.cfg["color"]["fps"],
                "d_w": self.cfg["depth"]["width"], "d_h": self.cfg["depth"]["height"], "d_fps": self.cfg["depth"]["fps"],
                "desc": f"USB3.0 首选 ({self.cfg['color']['width']}x{self.cfg['color']['height']}@{self.cfg['color']['fps']})"
            })

        candidate_configs.extend([
            {"c_w": 1280, "c_h": 720, "c_fps": 15, "d_w": 640, "d_h": 480, "d_fps": 15, "desc": "USB 2.1 高清模式: 彩色1280x720@15fps + 深度640x480@15fps"},
            {"c_w": 640, "c_h": 480, "c_fps": 30, "d_w": 640, "d_h": 480, "d_fps": 30, "desc": "USB 2.1 流畅模式: 彩色640x480@30fps + 深度640x480@30fps"},
            {"c_w": 640, "c_h": 480, "c_fps": 15, "d_w": 640, "d_h": 480, "d_fps": 15, "desc": "USB 2.1 兜底模式: 彩色640x480@15fps + 深度640x480@15fps"},
        ])

        profile = None
        for cand in candidate_configs:
            cfg_try = rs.config()
            cfg_try.enable_stream(rs.stream.color, cand["c_w"], cand["c_h"], rs.format.bgr8, cand["c_fps"])
            cfg_try.enable_stream(rs.stream.depth, cand["d_w"], cand["d_h"], rs.format.z16, cand["d_fps"])
            try:
                profile = self.pipeline.start(cfg_try)
                self.rs_config = cfg_try
                print(f"[INFO] [OK] 成功协商启动流配置: {cand['desc']}")
                break
            except Exception:
                continue

        if profile is None:
            raise RuntimeError("无法与 RealSense 协商出可用流配置，请检查连接状态或更换 USB 端口。")

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            preset_val = self.cfg.get("visual_preset", 3)
            try:
                depth_sensor.set_option(rs.option.visual_preset, preset_val)
                print(f"[INFO] 载入视觉预设: High Accuracy (Mode {preset_val})")
            except Exception:
                pass

        if depth_sensor.supports(rs.option.laser_power):
            pwr = self.cfg.get("laser_power", 120)
            try:
                depth_sensor.set_option(rs.option.laser_power, pwr)
                print(f"[INFO] 调优红外激光功率: {pwr} mW (防反光黑洞)")
            except Exception:
                pass

        color_stream = profile.get_stream(rs.stream.color)
        video_prof = color_stream.as_video_stream_profile()
        self.actual_w = video_prof.width()
        self.actual_h = video_prof.height()
        self.color_intrinsics = video_prof.get_intrinsics()
        print(f"[INFO] 彩色镜头实际画幅: {self.actual_w}x{self.actual_h}, 内参: fx={self.color_intrinsics.fx:.2f}, "
              f"fy={self.color_intrinsics.fy:.2f}, ppx={self.color_intrinsics.ppx:.2f}, ppy={self.color_intrinsics.ppy:.2f}")

        # 同步更新芦笋几何分析器的内参矩阵
        self.analyzer.update_intrinsics(
            fx=self.color_intrinsics.fx,
            fy=self.color_intrinsics.fy,
            cx=self.color_intrinsics.ppx,
            cy=self.color_intrinsics.ppy
        )

    def generate_mock_frame(self, w=1280, h=720):
        """生成带几何圆柱深度的 3 层叠压芦笋场景 (0.40m ~ 0.55m 视野)"""
        color_img = np.full((h, w, 3), (40, 42, 45), dtype=np.uint8)  # 工作台底色
        # 网格参考线
        for y in range(0, h, 80):
            cv2.line(color_img, (0, y), (w, y), (50, 52, 55), 1)
        for x in range(0, w, 80):
            cv2.line(color_img, (x, 0), (x, h), (50, 52, 55), 1)

        depth_img = np.full((h, w), 530, dtype=np.uint16)  # 工作台平面 Z=530mm (0.53m)

        # 3根堆叠芦笋规范: (起点, 终点, 半径mm, 颜色, 中心Z_mm, 描述)
        # 顺序：先画底层，再画中层，最后画顶层
        asparagus_list = [
            # 1. 底层芦笋 (Layer 0, Z_top=495mm, 水平偏下)
            {"p1": (320, 460), "p2": (960, 420), "r_px": 22, "r_mm": 8.0, "z_center": 503, "name": "Asparagus_Bottom"},
            # 2. 中层芦笋 (Layer 1, Z_top=465mm, 斜穿底层)
            {"p1": (420, 240), "p2": (820, 560), "r_px": 24, "r_mm": 9.0, "z_center": 474, "name": "Asparagus_Middle"},
            # 3. 顶层芦笋 (Layer 2, Z_top=430mm, 压在底层和中层上面, 最靠近相机)
            {"p1": (500, 540), "p2": (780, 180), "r_px": 26, "r_mm": 10.0, "z_center": 440, "name": "Asparagus_TOPMOST"},
        ]

        # 像素坐标网格
        y_coords, x_coords = np.indices((h, w))

        for asp in asparagus_list:
            p1 = np.array(asp["p1"], dtype=float)
            p2 = np.array(asp["p2"], dtype=float)
            vec = p2 - p1
            vec_len = np.linalg.norm(vec)
            u_vec = vec / vec_len

            # 点到线段距离计算
            px_diff = np.stack([x_coords - p1[0], y_coords - p1[1]], axis=-1)
            proj = np.sum(px_diff * u_vec, axis=-1)
            proj_clamped = np.clip(proj, 0, vec_len)
            closest = p1 + proj_clamped[..., None] * u_vec
            dist = np.linalg.norm(np.stack([x_coords, y_coords], axis=-1) - closest, axis=-1)

            mask = dist <= asp["r_px"]
            cylinder_h_ratio = np.sqrt(np.maximum(0, 1.0 - (dist / asp["r_px"]) ** 2))
            asp_z = asp["z_center"] - (cylinder_h_ratio * asp["r_mm"])

            update_mask = mask & (asp_z < depth_img)
            depth_img[update_mask] = asp_z[update_mask].astype(np.uint16)

            green_val = np.clip(160 * cylinder_h_ratio + 40, 30, 210).astype(np.uint8)
            color_img[update_mask, 0] = (25 * cylinder_h_ratio[update_mask]).astype(np.uint8)
            color_img[update_mask, 1] = green_val[update_mask]
            color_img[update_mask, 2] = (30 * cylinder_h_ratio[update_mask]).astype(np.uint8)

        # 添加少许高斯噪声模拟真实深度
        noise = np.random.normal(0, 0.8, (h, w)).astype(np.int16)
        depth_img = np.clip(depth_img.astype(np.int16) + noise, 0, 65535).astype(np.uint16)

        return color_img, depth_img

    def on_mouse(self, event, x, y, flags, param):
        """鼠标移动/点击事件处理"""
        display_w = param.get("display_w", 640)
        display_h = param.get("display_h", 360)
        orig_w = param.get("orig_w", 1280)
        orig_h = param.get("orig_h", 720)

        local_x = x % display_w
        local_y = y

        u = int(local_x * orig_w / display_w)
        v = int(local_y * orig_h / display_h)

        if 0 <= u < orig_w and 0 <= v < orig_h:
            self.hover_x = u
            self.hover_y = v
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selected_point = (u, v)

    def run(self):
        """主可视化循环"""
        self.start()

        window_name = "RealSense D435 RGB-D Aligned Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        disp_w = 640
        disp_h = 360
        cv2.setMouseCallback(window_name, self.on_mouse, {"display_w": disp_w, "display_h": disp_h, "orig_w": self.actual_w, "orig_h": self.actual_h})

        print("\n" + "=" * 64)
        print(" RealSense D435 实时深度探针交互说明:")
        print("   - 鼠标悬停/左键点击: 实时探测物料点 (X, Y, Z) 3D 空间坐标")
        print("   - 色彩分布规律: 越在上层(越近) -> 红色/暖色; 越在下层(底台) -> 蓝色/冷色")
        print("   - [A] 键: 切换【固定范围 0.40~0.55m】与【动态自适应 Auto-Range】")
        print("   - [ [ ] / [ ] ] 键: 微调近端下限 min (+/- 1cm)")
        print("   - [ - ] / [ = ] 键: 微调远端上限 max (+/- 1cm)")
        print("   - [R] 键: 重置色彩区间为 0.40m ~ 0.55m")
        print("   - [D] 键: 切换芦笋识别与顶层尺寸/坐标解算 (开/关)")
        print("   - [G] 键: 打印当前最顶层芦笋 SCARA G-code 抓取指令")
        print("   - [S] 键: 抓拍并保存当前对齐帧 (RGB图 + 深度图 + 原始数值)")
        print("   - [F] 键: 切换硬件后处理滤波器 (开/关)")
        print("   - [L] 键: 切换激光散斑发射器 (开/关)")
        print("   - [Q] 或 [ESC]: 退出程序")
        print("=" * 64 + "\n")

        fps_counter = 0
        fps_time = time.time()
        current_fps = 30.0

        try:
            while True:
                if self.mock_mode:
                    color_image, depth_image = self.generate_mock_frame()
                    time.sleep(0.03)  # ~30fps
                else:
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)

                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    filtered_depth_frame = self.apply_filters(depth_frame)
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(filtered_depth_frame.get_data())

                # 动态自适应与色彩范围映射
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

                # 色彩映射：最顶层(近距，如0.40m)对应暖红(255)，底面(远距，如0.55m)对应深蓝(0)
                norm = np.clip((act_max - depth_meters) / max(0.005, (act_max - act_min)) * 255.0, 0, 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

                # 将无效深度 (0mm / 反光黑洞) 涂抹为深黑灰，与真实台面深度鲜明区分
                depth_colormap[depth_image == 0] = (25, 25, 25)

                # 实时芦笋特征分析与最顶层判决
                raw_color_for_save = color_image.copy()
                if self.detection_enabled:
                    try:
                        self.latest_targets = self.analyzer.analyze(color_image, depth_image)
                        display_color_image = self.analyzer.draw_detections(color_image, self.latest_targets)
                    except Exception:
                        display_color_image = color_image.copy()
                else:
                    display_color_image = color_image.copy()

                # 鼠标点测距与反投影
                target_pt = self.selected_point if self.selected_point else (self.hover_x, self.hover_y)
                probe_text = "Probe: N/A"
                if target_pt[0] >= 0 and target_pt[1] >= 0:
                    px, py = target_pt
                    if py < depth_image.shape[0] and px < depth_image.shape[1]:
                        depth_mm = depth_image[py, px]
                        if depth_mm > 0 and self.color_intrinsics:
                            depth_m = depth_mm * 0.001
                            pt_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [px, py], depth_m)
                            probe_text = f"Pixel:({px},{py}) | Depth:{depth_mm}mm | 3D:({pt_3d[0]*1000:.1f}, {pt_3d[1]*1000:.1f}, {pt_3d[2]*1000:.1f})mm"
                        elif depth_mm > 0:
                            probe_text = f"Pixel:({px},{py}) | Depth:{depth_mm}mm (Z-Height)"
                        else:
                            probe_text = f"Pixel:({px},{py}) | Depth: INVALID (0mm / Hole)"

                    cv2.drawMarker(display_color_image, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.drawMarker(depth_colormap, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()

                color_resized = cv2.resize(display_color_image, (disp_w, disp_h))
                depth_resized = cv2.resize(depth_colormap, (disp_w, disp_h))
                combined = np.hstack((color_resized, depth_resized))

                mode_str = "MOCK" if self.mock_mode else "D435"
                filter_status = "ON" if self.filters_enabled else "OFF"
                detect_status = f"DETECT:{len(self.latest_targets)}" if self.detection_enabled else "DETECT:OFF"
                range_str = f"AUTO:{act_min:.2f}~{act_max:.2f}m" if self.auto_range else f"{act_min:.2f}m(Red)~{act_max:.2f}m(Blue)"

                # 顶部状态条
                topmost = next((t for t in self.latest_targets if t.is_topmost), None) if self.detection_enabled else None
                if topmost:
                    top_banner = f"[TOPMOST 1] L:{topmost.length_mm}mm D:{topmost.diam_mm}mm | Grip:({topmost.grip_x}, {topmost.grip_y}, {topmost.grip_z})mm R:{topmost.yaw_deg}deg"
                    cv2.rectangle(combined, (0, 0), (disp_w * 2, 28), (15, 50, 15), -1)
                    cv2.putText(combined, top_banner, (12, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2)
                else:
                    cv2.putText(combined, f"RGB [{mode_str}] | {detect_status} | FPS: {current_fps:.1f}", (15, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

                cv2.putText(combined, f"Depth [{range_str}] (Filt:{filter_status})", (disp_w + 10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

                cv2.rectangle(combined, (0, disp_h - 35), (disp_w * 2, disp_h), (25, 25, 25), -1)
                cv2.putText(combined, probe_text, (15, disp_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                cv2.imshow(window_name, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif key == ord('s'):
                    self.save_snapshot(raw_color_for_save, depth_image, depth_colormap)
                elif key == ord('d'):
                    self.detection_enabled = not self.detection_enabled
                    print(f"[ACTION] 芦笋特征识别与顶层解算: {'开启' if self.detection_enabled else '关闭'}")
                elif key == ord('g'):
                    topmost = next((t for t in self.latest_targets if t.is_topmost), None)
                    if topmost:
                        print("\n" + "=" * 65)
                        print(f"[SCARA G-CODE] 最上层芦笋抓取指令序列 (L={topmost.length_mm}mm, D={topmost.diam_mm}mm, 凸起={topmost.rel_height_mm}mm):")
                        print(f"G90                     ; 绝对坐标模式")
                        print(f"G0 Z50.0 F3000          ; 提升末端到安全高度")
                        print(f"M280 P1 S10             ; 预张开夹板")
                        print(f"G0 X{topmost.grip_x:.1f} Y{topmost.grip_y:.1f} R{topmost.yaw_deg:.1f} F4000 ; 平移对准并旋转夹爪轴线")
                        print(f"G1 Z{topmost.grip_z:.1f} F1500          ; 垂直下探至夹持面")
                        print(f"M280 P1 S90             ; 夹板夹紧物料")
                        print(f"G4 P200                 ; 稳固延时 200ms")
                        print(f"G0 Z50.0 F3000          ; 提起最上层芦笋")
                        print("=" * 65 + "\n")
                    else:
                        print("\n[!] 当前未检测到可抓取的最顶层芦笋，无法生成 G-code。")
                elif key == ord('a'):
                    self.auto_range = not self.auto_range
                    status = "动态自适应 (Auto-Range)" if self.auto_range else f"固定区间 [{self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m]"
                    print(f"[ACTION] 色彩映射模式切换: {status}")
                elif key == ord('['):
                    self.cmap_min = max(0.10, round(self.cmap_min - 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整近端色阶 min: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")
                elif key == ord(']'):
                    self.cmap_min = min(self.cmap_max - 0.02, round(self.cmap_min + 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整近端色阶 min: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")
                elif key == ord('-'):
                    self.cmap_max = max(self.cmap_min + 0.02, round(self.cmap_max - 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整远端色阶 max: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")
                elif key == ord('='):
                    self.cmap_max = min(2.0, round(self.cmap_max + 0.01, 3))
                    self.auto_range = False
                    print(f"[ACTION] 调整远端色阶 max: {self.cmap_min:.2f}m ~ {self.cmap_max:.2f}m")
                elif key == ord('r'):
                    self.cmap_min = 0.40
                    self.cmap_max = 0.55
                    self.auto_range = False
                    print(f"[ACTION] 色彩区间已重置为默认: 0.40m ~ 0.55m")
                elif key == ord('f'):
                    self.filters_enabled = not self.filters_enabled
                    print(f"[ACTION] 深度后处理滤波器: {'开启' if self.filters_enabled else '关闭'}")
                elif key == ord('l') and not self.mock_mode:
                    depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        self.laser_enabled = not self.laser_enabled
                        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if self.laser_enabled else 0.0)
                        print(f"[ACTION] 红外散斑发射器: {'开启' if self.laser_enabled else '关闭'}")

        finally:
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()
            print("[INFO] 采集与可视化窗口已安全退出。")

    def save_snapshot(self, color_img, depth_raw, depth_color):
        """抓拍并保存当前帧数据"""
        save_dir = os.path.join("data", "snapshots")
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        rgb_path = os.path.join(save_dir, f"color_{ts}.png")
        depth_vis_path = os.path.join(save_dir, f"depth_vis_{ts}.png")
        depth_raw_path = os.path.join(save_dir, f"depth_raw_{ts}.npy")

        cv2.imwrite(rgb_path, color_img)
        cv2.imwrite(depth_vis_path, depth_color)
        np.save(depth_raw_path, depth_raw)

        print(f"\n[SNAPSHOT] 数据已抓拍保存:")
        print(f"  -> 彩色图:   {rgb_path}")
        print(f"  -> 深度热力: {depth_vis_path}")
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
