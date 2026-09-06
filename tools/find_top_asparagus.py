"""
芦笋最顶层目标解算与 SCARA 抓取位姿输出工具 (CLI)
=====================================================
功能：
  1. 支持从物理 RealSense D435 实时抓取单帧，或载入离线抓拍文件 (.png + .npy)
  2. 自动拟合台面倾斜方程，过滤复杂木纹背景与平行并拢棒体
  3. 计算每根芦笋的【物理长度】与【物理直径】(mm)
  4. 识别并锁定【最上面一根芦笋】(Topmost Pickable Target)
  5. 计算夹爪夹持中心点的三维空间坐标 (X, Y, Z) mm 与夹爪旋转角度 Yaw (R deg)
  6. 格式化输出终端报表、结构化 JSON、以及 SCARA 机械臂直接执行的 G-code 指令
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 导入 RealSense (若可用)
try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vision.asparagus_analyzer import AsparagusAnalyzer, AsparagusTarget


def parse_args():
    parser = argparse.ArgumentParser(description="解算最顶层芦笋物理尺寸与机械臂抓取位姿")
    parser.add_argument("--image", type=str, default=None, help="彩色图像路径 (.png/.jpg)，若留空则从相机实时抓取")
    parser.add_argument("--depth", type=str, default=None, help="原始深度矩阵路径 (.npy)，若留空则从相机实时抓取")
    parser.add_argument("--save-vis", type=str, default="data/snapshots/latest_top_asparagus.png", help="可视化标注结果保存路径")
    parser.add_argument("--json", action="store_true", help="仅在终端输出纯 JSON 结果")
    return parser.parse_args()


def capture_from_d435():
    """从 RealSense D435 捕获一帧高质量对齐 RGB-D 数据"""
    if not HAVE_REALSENSE:
        raise RuntimeError("系统未安装 pyrealsense2 库，无法调用物理相机。")

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("未检测到连接的 Intel RealSense D435 相机！")

    pipeline = rs.pipeline()
    config = rs.config()
    align = rs.align(rs.stream.color)

    # 兼容 USB 2.1 与 USB 3.0
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    profile = pipeline.start(config)

    # 硬件滤波器链条
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 1)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    try:
        # 预热曝光与自动白平衡
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        c_frame = aligned.get_color_frame()
        d_frame = aligned.get_depth_frame()

        d_filtered = spatial.process(temporal.process(d_frame))

        color_img = np.asanyarray(c_frame.get_data())
        depth_img = np.asanyarray(d_filtered.get_data())
        return color_img, depth_img, intrinsics
    finally:
        pipeline.stop()


import yaml

def load_system_config():
    """读取 config.yaml 中的标定与机器人参数"""
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    t_cam_to_scara = None
    safe_z = 80.0
    drop_x = 220.0
    drop_y = 0.0
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    calib = cfg.get("calibration", {})
                    t_cam_to_scara = calib.get("t_cam_to_scara", None)
                    r_cfg = cfg.get("robot", {})
                    safe_z = float(r_cfg.get("safe_z_mm", 80.0))
                    drop_x = float(r_cfg.get("drop_x_mm", 220.0))
                    drop_y = float(r_cfg.get("drop_y_mm", 0.0))
        except Exception as e:
            print(f"[WARN] 加载 config.yaml 异常: {e}")
    return t_cam_to_scara, safe_z, drop_x, drop_y


def main():
    args = parse_args()
    t_cam_to_scara, safe_z, drop_x, drop_y = load_system_config()

    # 1. 获取彩色与深度数据
    if args.image and args.depth:
        if not os.path.exists(args.image) or not os.path.exists(args.depth):
            print(f"[ERROR] 指定的文件不存在: {args.image} 或 {args.depth}")
            sys.exit(1)
        color_img = cv2.imread(args.image)
        depth_img = np.load(args.depth)
        analyzer = AsparagusAnalyzer(fx=909.12, fy=907.46, cx=647.46, cy=377.51)
    else:
        print("[INFO] 正在从 Intel RealSense D435 捕获当前对齐数据帧...")
        color_img, depth_img, intrinsics = capture_from_d435()
        analyzer = AsparagusAnalyzer(fx=intrinsics.fx, fy=intrinsics.fy, cx=intrinsics.ppx, cy=intrinsics.ppy)

    # 加载手眼标定矩阵
    if t_cam_to_scara is not None:
        analyzer.set_hand_eye_matrix(np.array(t_cam_to_scara, dtype=float))

    # 2. 执行芦笋特征分析与最顶层判决
    targets = analyzer.analyze(color_img, depth_img)

    if len(targets) == 0:
        if args.json:
            print(json.dumps({"success": False, "error": "未检测到符合物理尺寸规范的芦笋目标"}))
        else:
            print("\n[!] 当前视野内未检测到符合芦笋物理规格的目标。请检查物料摆放或相机对焦。")
        sys.exit(0)

    # 3. 提取最顶层芦笋
    topmost = next((t for t in targets if t.is_topmost), targets[0])

    # 4. 保存可视化标注图
    if args.save_vis:
        vis_img = analyzer.draw_detections(color_img, targets)
        os.makedirs(os.path.dirname(args.save_vis), exist_ok=True)
        cv2.imwrite(args.save_vis, vis_img)

    # 5. 输出结果
    result_dict = {
        "success": True,
        "total_detected": len(targets),
        "topmost_target": {
            "id": topmost.id,
            "length_mm": topmost.length_mm,
            "diameter_mm": topmost.diam_mm,
            "relative_height_mm": topmost.rel_height_mm,
            "yaw_deg": topmost.yaw_deg,
            "grasp_point_camera": {
                "x_mm": topmost.grip_x,
                "y_mm": topmost.grip_y,
                "z_mm": topmost.grip_z
            },
            "robot_grasp_pose": {
                "x_mm": topmost.robot_x,
                "y_mm": topmost.robot_y,
                "z_mm": topmost.robot_z,
                "r_deg": topmost.robot_r,
                "is_calibrated": topmost.is_calibrated
            },
            "pixel_center": [round(topmost.center_px[0], 1), round(topmost.center_px[1], 1)]
        },
        "all_targets": [
            {
                "id": t.id,
                "is_topmost": t.is_topmost,
                "length_mm": t.length_mm,
                "diameter_mm": t.diam_mm,
                "yaw_deg": t.yaw_deg,
                "grasp_point_camera": {"x_mm": t.grip_x, "y_mm": t.grip_y, "z_mm": t.grip_z},
                "robot_grasp_pose": {"x_mm": t.robot_x, "y_mm": t.robot_y, "z_mm": t.robot_z, "r_deg": t.robot_r},
                "relative_height_mm": t.rel_height_mm
            } for t in targets
        ]
    }

    if args.json:
        print(json.dumps(result_dict, indent=2))
        return

    print("\n" + "=" * 76)
    print("        Intel RealSense D435 芦笋视觉特征分析与 SCARA 抓取解算报告")
    print("=" * 76)
    print(f"检测物料总数: {len(targets)} 根 | 手眼标定状态: {'[已标定 Eye-to-Hand]' if analyzer.is_hand_eye_calibrated else '[未标定 - 防撞保护模式]'}")
    print("-" * 76)
    print(f"{'序号':<6}{'状态':<10}{'长度(mm)':<12}{'直径(mm)':<12}{'偏航角R(°)':<12}{'台面凸起':<10}{'SCARA目标(X,Y,Z) mm'}")
    print("-" * 76)
    for t in targets:
        status = "[TOP 最顶层]" if t.is_topmost else f"第{t.id}层(下压)"
        scara_str = f"({t.robot_x}, {t.robot_y}, {t.robot_z})"
        print(f"#{t.id:<5}{status:<10}{t.length_mm:<12}{t.diam_mm:<12}{t.robot_r:<12}+{t.rel_height_mm:<9}{scara_str}")
    print("=" * 76)

    print("\n>>> 最上层可抓取芦笋 (Topmost Pickable Target) 核心参数:")
    print(f"  * 物理长度 (Length):     {topmost.length_mm} mm")
    print(f"  * 物理直径 (Diameter):   {topmost.diam_mm} mm")
    print(f"  * 夹持旋转角 (Yaw / R):  {topmost.robot_r} deg (夹爪末端旋转角度)")
    print(f"  * 相机坐标测量 (Camera): X={topmost.grip_x} mm, Y={topmost.grip_y} mm, Z={topmost.grip_z} mm")
    print(f"  * 机械臂抓取目标 (SCARA): X={topmost.robot_x} mm, Y={topmost.robot_y} mm, Z={topmost.robot_z} mm")
    print(f"  * 相对工作台凸起净高:    +{topmost.rel_height_mm} mm")

    print("\n>>> 生成的 SCARA 防撞抓取执行指令 (G-code):")
    print(topmost.generate_gcode(safe_z=safe_z, drop_x=drop_x, drop_y=drop_y))
    print("=" * 76)

    if args.save_vis:
        print(f"[OK] 视觉检测标注图已保存至: {args.save_vis}\n")


if __name__ == "__main__":
    main()
