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


def generate_scara_gcode(target: AsparagusTarget, safe_z: float = 50.0) -> str:
    """生成 SCARA 机械臂直接执行的 G-code 抓取序列"""
    gcode = []
    gcode.append(f"; === SCARA 机械臂抓取最上层芦笋指令 ===")
    gcode.append(f"; 目标物料: 长度={target.length_mm}mm, 直径={target.diam_mm}mm, 相对台面高={target.rel_height_mm}mm")
    gcode.append(f"; 相机系夹持中心: X={target.grip_x}mm, Y={target.grip_y}mm, Z={target.grip_z}mm, 偏航角={target.yaw_deg}度")
    gcode.append(f"G90                     ; 绝对坐标模式")
    gcode.append(f"G0 Z{safe_z:.1f} F3000          ; 提升末端执行器至安全离地高度")
    gcode.append(f"M280 P1 S10             ; 预先张开气动/伺服平行夹板")
    gcode.append(f"; 注意: 实际执行时需将 (grip_x, grip_y) 经手眼矩阵变换至机器人基座 (X_base, Y_base)")
    gcode.append(f"G0 X{target.grip_x:.1f} Y{target.grip_y:.1f} R{target.yaw_deg:.1f} F4000 ; 平移对准目标上方，同时旋转夹爪对齐芦笋轴线")
    gcode.append(f"G1 Z{target.grip_z:.1f} F1500          ; 垂直下探至目标物料夹持高度")
    gcode.append(f"M280 P1 S90             ; 闭合夹板夹紧芦笋")
    gcode.append(f"G4 P200                 ; 驻留延时 200ms 确保夹持稳固")
    gcode.append(f"G0 Z{safe_z:.1f} F3000          ; 垂直平稳提起最上层芦笋")
    return "\n".join(gcode)


def main():
    args = parse_args()

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
                "relative_height_mm": t.rel_height_mm
            } for t in targets
        ]
    }

    if args.json:
        print(json.dumps(result_dict, indent=2))
        return

    print("\n" + "=" * 70)
    print("      Intel RealSense D435 芦笋视觉特征分析与最顶层判决报告")
    print("=" * 70)
    print(f"检测物料总数: {len(targets)} 根")
    print("-" * 70)
    print(f"{'序号':<6}{'状态':<10}{'长度(mm)':<12}{'直径(mm)':<12}{'偏航角R(°)':<12}{'台面凸起(mm)':<14}{'抓取位姿(X, Y, Z) mm'}")
    print("-" * 70)
    for t in targets:
        status = "[TOP 最顶层]" if t.is_topmost else f"第{t.id}层(下压)"
        grip_str = f"({t.grip_x}, {t.grip_y}, {t.grip_z})"
        print(f"#{t.id:<5}{status:<10}{t.length_mm:<12}{t.diam_mm:<12}{t.yaw_deg:<12}{t.rel_height_mm:<14}{grip_str}")
    print("=" * 70)

    print("\n>>> 最上层可抓取芦笋 (Topmost Pickable Target) 核心参数:")
    print(f"  * 物理长度 (Length):     {topmost.length_mm} mm")
    print(f"  * 物理直径 (Diameter):   {topmost.diam_mm} mm")
    print(f"  * 夹持旋转角 (Yaw / R):  {topmost.yaw_deg} deg (夹板旋转角度)")
    print(f"  * 夹爪目标坐标 (Camera): X={topmost.grip_x} mm, Y={topmost.grip_y} mm, Z={topmost.grip_z} mm")
    print(f"  * 相对工作台凸起净高:    +{topmost.rel_height_mm} mm")

    print("\n>>> 生成的 SCARA 抓取执行指令 (G-code):")
    print(generate_scara_gcode(topmost))
    print("=" * 70)

    if args.save_vis:
        print(f"[OK] 视觉检测标注图已保存至: {args.save_vis}\n")


if __name__ == "__main__":
    main()
