"""
flux_vision_3d 交互式 CLI 控制台与工具导航菜单
=================================================
系统架构：
  - 主菜单：日常生产监视、可视化与位姿解算
  - 标定建图专区：1-制靶 -> 2-采图 -> 3-建图 -> 4-验证 的闭环向导
  - 自动化测试专区：单元测试与算法回归
"""

import os
import sys
import glob
import time
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from tools.env_utils import check_env_status  # noqa: E402

# Windows 终端色彩支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    os.system("")

# 终端 ANSI 色彩定义
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_GRAY = "\033[90m"
C_RESET = "\033[0m"




def print_main_banner(status):
    """打印控制台主横幅与状态"""
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    print(f"{C_CYAN}{C_BOLD}             flux_vision_3d 芦笋 3D 视觉与抓取位姿估计系统 - 控制终端              {C_RESET}")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    
    cv_str = f"{C_GREEN}{status['opencv'][1]}{C_RESET}" if status['opencv'][0] else f"{C_RED}未安装{C_RESET}"
    np_str = f"{C_GREEN}{status['numpy'][1]}{C_RESET}" if status['numpy'][0] else f"{C_RED}未安装{C_RESET}"
    rs_str = f"{C_GREEN}{status['realsense'][1]}{C_RESET}" if status['realsense'][0] else f"{C_YELLOW}{status['realsense'][1]}{C_RESET}"
    snap_str = f"{C_GREEN}{status['snapshot_count']} 帧可用{C_RESET}" if status['snapshot_count'] > 0 else f"{C_GRAY}暂无快照{C_RESET}"
    
    print(f" 环境状态: Python {C_GREEN}{sys.version.split()[0]}{C_RESET} | OpenCV: {cv_str} | NumPy: {np_str} | D435驱动: {rs_str}")
    print(f" 本地数据: snapshots 快照 ({snap_str}) | 建图采图 ({status['calib_image_count']} 帧)")
    print(f"{C_CYAN}-------------------------------------------------------------------------------{C_RESET}")
    print(f"{C_BOLD} [ 视觉控制中枢与顶级入口 (Master Dashboard) ]{C_RESET}")
    print(f"   {C_CYAN}{C_BOLD}[G]{C_RESET} {C_GREEN}{C_BOLD}启动 3D 视觉综合控制中心 (GUI Launcher)  ★ 推荐！1280x720 工业科技大屏{C_RESET}")
    print(f"   {C_GREEN}{C_BOLD}[2]{C_RESET} {C_CYAN}{C_BOLD}工况与场景综合管理中枢 (Scene Hub)      ★ 核心一级入口！(工况切换/沙盒体检/生产发布){C_RESET}")
    print("")
    print(f"{C_BOLD} [ 核心生产与工况管理 (Core & Workspace) ]{C_RESET}")
    print(f"   {C_GREEN}[1]{C_RESET} 启动 D435 实时相机查看器与深度探针     (物理硬件模式)")
    print(f"   {C_GREEN}[3]{C_RESET} 解算最顶层芦笋抓取位姿 (实时相机)      (find_top_asparagus.py 单帧采集解算)")
    print(f"   {C_GREEN}[4]{C_RESET} 解算最顶层芦笋抓取位姿 (离线快照)      (自动读取最新本地快照快速验证)")
    print(f"   {C_GREEN}[5]{C_RESET} 快速生成一帧模拟快照至 snapshots       (方便无相机时进行算法验证)")
    print(f"   {C_GREEN}[6]{C_RESET} 启动 D435 仿真模拟可视化查看器         ({C_YELLOW}--mock{C_RESET} 模式，无需物理相机)")
    print("")
    print(f"{C_BOLD} [ 视觉标定与自动化专区 (Specialized Suites) ]{C_RESET}")
    tag_map_info = f"{C_GREEN}(已建图){C_RESET}" if status['has_tag_map'] else f"{C_YELLOW}(未建图){C_RESET}"
    print(f"   {C_GREEN}{C_BOLD}[H]{C_RESET} 进入「SCARA 手眼标定与 AprilTag 空间建图」专区 {tag_map_info}")
    print(f"   {C_GREEN}{C_BOLD}[T]{C_RESET} 进入「自动化测试与算法回归」专区")
    print("")
    print(f"{C_BOLD} [ 依赖与系统维护 ]{C_RESET}")
    print(f"   {C_GREEN}[8]{C_RESET} 安装 / 更新项目依赖包                  (pip install -r requirements.txt)")
    print(f"   {C_GREEN}[9]{C_RESET} 详细环境与驱动诊断")
    print(f"   {C_GREEN}[C]{C_RESET} 进入项目根目录命令行 (CMD)")
    print(f"   {C_RED}[0]{C_RESET} 退出控制终端")
    print(f"{C_CYAN}==============================================================================={C_RESET}")


def print_calibration_banner(status):
    """打印标定与空间建图专区横幅"""
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    print(f"{C_CYAN}{C_BOLD}       【手眼标定与 AprilTag 空间建图专区】(Calibration & Tag Mapping)         {C_RESET}")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    print(f" 标准流水线: {C_YELLOW}[1 制靶]{C_RESET} -> {C_YELLOW}[2 采图向导]{C_RESET} -> {C_YELLOW}[3 超精提取]{C_RESET} -> {C_YELLOW}[S 离线Studio]{C_RESET} -> {C_YELLOW}[6 在线AR验证]{C_RESET}")

    scene = status.get('active_scene')
    if scene:
        pub_tag = f"{C_GREEN}★已发布生产{C_RESET}" if scene.is_published else f"{C_YELLOW}草稿待发布{C_RESET}"
        rmse_tag = f"RMSE: {C_GREEN}{scene.global_rmse_px:.3f}px{C_RESET}" if scene.ba_solved else f"{C_GRAY}未平差{C_RESET}"
        print(f" 【当前活动场景】: {C_BOLD}{C_GREEN}{scene.scene_id}{C_RESET} (别名: {scene.name} | 采图: {scene.image_count}帧 | 有效: {scene.active_image_count}帧 | {rmse_tag} | {pub_tag})")
        print(f" 场景物理沙盒  : {C_GRAY}{scene.scene_dir}{C_RESET}")
    else:
        print(f" 状态一览: 采图集: {C_GREEN}{status['calib_image_count']}{C_RESET} 帧 | 审核清单: {status.get('has_manifest', False)}")

    print(f"{C_CYAN}-------------------------------------------------------------------------------{C_RESET}")
    print(f"{C_BOLD} [ 一、 标靶准备 (Target Preparation) ]{C_RESET}")
    print(f"   {C_GREEN}[1]{C_RESET} AprilTag 标靶图纸生成                  (生成 0~29 号高清标靶与 1:1 A4 排版 PDF)")
    print("")
    print(f"{C_BOLD} [ 二、 图像采集与外部向导 (Image Acquisition) ]{C_RESET}")
    print(f"   {C_GREEN}[2]{C_RESET} AprilTag 多视角交互式采图向导          (自动存入当前场景 raw_images/，空格一键连拍) {C_GRAY}[快捷键: C]{C_RESET}")
    print("")
    print(f"{C_BOLD} [ 三、 离线解算与质量闭环 (Offline Pipeline & QA) ]{C_RESET}")
    print(f"   {C_GREEN}{C_BOLD}[S]{C_RESET} {C_CYAN}{C_BOLD}进入 AprilTag 离线标定综合工作站 (Offline Studio)  ★ 自动装载当前活动场景{C_RESET}")
    print(f"       {C_GRAY}(整合样本交互审核、高精BA平差解算、热力覆盖率分析与全局体检闭环){C_RESET}")
    print(f"   {C_GREEN}[3]{C_RESET} 离线图像诊断调优与超精重提取           (当前场景: 16级阈值网格+双尺度CLAHE+0.01px精修) {C_GRAY}[快捷键: 4]{C_RESET}")
    print(f"   {C_GREEN}[4]{C_RESET} 纯计算空间立体建图与两阶段 BA 平差     (tag_map_builder.py，命令行静默求解) {C_GRAY}[快捷键: 5/M]{C_RESET}")
    print(f"   {C_GREEN}[5]{C_RESET} 离线标定精度体检工作台 (LOO盲测体检)   (当前场景全量留一盲测，残差矢量评估) {C_GRAY}[快捷键: 6/P]{C_RESET}")
    print("")
    print(f"{C_BOLD} [ 四、 在线验收与生产部署 (Online AR Verification & Deployment) ]{C_RESET}")
    print(f"   {C_GREEN}[6]{C_RESET} 标定精度在线 AR 综合实时验证系统      (相机实时取流，3D轴/棱柱虚实融合，静态位姿锁定) {C_GRAY}[快捷键: 7]{C_RESET}")
    print("")
    print(f"{C_BOLD} [ 五、 辅助工具与维护通道 (Auxiliary Tools & Maintenance) ]{C_RESET}")
    print(f"   {C_GREEN}[H]{C_RESET} 跳转打开工况与场景管理中枢 (Scene Hub)  (工况沙盒画廊、数据体检与生产生效)")
    print(f"   {C_GREEN}[W]{C_RESET} AprilTag 标靶 ID 白名单管理            (查看当前/一键放行探索/指定有效 ID 列表)")
    print(f"   {C_GREEN}[V]{C_RESET} 浏览当前场景标注与分析目录            (在系统资源管理器中打开当前场景 visualized/)")
    print(f"   {C_GREEN}[D]{C_RESET} 单帧标靶漏检病因深度诊断与切片分析    (分析真图淘汰候选框/尺寸/反差/模糊原因)")
    print(f"   {C_GREEN}[C]{C_RESET} 一键清空当前活动场景采图数据集        (重置当前场景采图集从 0 开始)")
    print(f"   {C_GREEN}[8]{C_RESET} 备用通道: SCARA 经典接触式物理标定    (SVD 点对刚体配准，极端无Tag场景备用)")
    print("")
    print(f"   {C_YELLOW}[B]{C_RESET} 返回主菜单")
    print(f"{C_CYAN}==============================================================================={C_RESET}")



def print_test_banner():
    """打印自动化测试专区横幅"""
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    print(f"{C_CYAN}{C_BOLD}             【自动化测试与算法验证专区】(Automated Tests & CI)                 {C_RESET}")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    print(f" 说明: 执行各层级自动化单元与回归测试，确保算法数学正确性与系统稳定性。")
    print(f"{C_CYAN}-------------------------------------------------------------------------------{C_RESET}")
    print(f"   {C_GREEN}[1]{C_RESET} 运行端到端视觉管线仿真测试             (tests/test_mock_pipeline.py)")
    print(f"   {C_GREEN}[2]{C_RESET} 运行真实快照芦笋算法测试               (tests/test_real_snapshot.py)")
    print(f"   {C_GREEN}[3]{C_RESET} 运行 AprilTag 空间建图与平差单元测试    (tests/test_tag_map_builder.py)")
    print(f"   {C_GREEN}[4]{C_RESET} 运行 AprilTag 在线 AR 综合验证单元测试  (tests/test_tag_calibration_verifier.py)")
    print(f"   {C_GREEN}[5]{C_RESET} 运行 AprilTag 离线精度体检单元测试      (tests/test_tag_offline_verifier.py)")
    print(f"   {C_GREEN}[6]{C_RESET} 运行 AprilTag 场景管理与取流单元测试    (tests/test_scene_hub.py)")
    print(f"   {C_GREEN}[A]{C_RESET} 一键运行全部自动化测试")
    print("")
    print(f"   {C_YELLOW}[B]{C_RESET} 返回主菜单")
    print(f"{C_CYAN}==============================================================================={C_RESET}")


def pause_prompt():
    print(f"\n{C_GRAY}按回车键继续...{C_RESET}", end="", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def ensure_camera_connected():
    """检查物理相机是否已连接，若未连接给出排查引导并询问是否使用仿真模式"""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        if len(ctx.query_devices()) > 0:
            return True, None
    except Exception as e:
        return False, f"RealSense SDK 异常: {e}"

    print(f"\n{C_RED}{C_BOLD}[警告] 未检测到连接的 Intel RealSense 物理相机设备！{C_RESET}")
    print("可能的原因与排查步骤：")
    print("  1. 相机 Type-C 线缆未连接或接触不良，建议插在电脑背板蓝色 USB 3.0 接口；")
    print("  2. 请拔下相机线缆重新插入，等待 3 秒使系统识别设备；")
    print("  3. 相机可能已被其他程序独占（如 RealSense Viewer）。")
    print("-" * 65)
    choice = input("请选择: [R]重试连接 / [M]改用仿真模式(--mock) / [任意键]取消返回: ").strip().lower()
    if choice == 'r':
        return ensure_camera_connected()
    elif choice == 'm':
        return True, "mock"
    else:
        return False, "cancel"


# ===================== 主菜单调用功能 =====================

def run_tool_d435_real():
    print(f"\n{C_CYAN}[启动]{C_RESET} 准备启动 RealSense D435 物理相机查看器...")
    ok, mode = ensure_camera_connected()
    if not ok:
        print(f"{C_YELLOW}[已取消]{C_RESET} 未启动物理相机查看器。")
        pause_prompt()
        return

    if mode == "mock":
        run_tool_d435_mock()
        return

    print(f"{C_GRAY}操作提示: [Space]定格/暂停画面 | [V]切换视图 | [G]打印G-code | [D]芦笋检测 | [S]抓拍 | [Q]退出{C_RESET}")
    subprocess.run([sys.executable, "tools/d435_viewer.py"])


def run_tool_d435_mock():
    print(f"\n{C_CYAN}[启动]{C_RESET} 正在以仿真模拟模式启动 D435 可视化查看器 (--mock)...")
    subprocess.run([sys.executable, "tools/d435_viewer.py", "--mock"])


def run_tool_top_real():
    print(f"\n{C_CYAN}[启动]{C_RESET} 准备从 RealSense D435 物理相机单帧捕获并解算最顶层芦笋...")
    ok, mode = ensure_camera_connected()
    if not ok or mode == "mock":
        if mode == "mock":
            print(f"{C_YELLOW}[提示]{C_RESET} 仿真模式请使用主菜单选项 [4] 加载快照数据解算。")
        pause_prompt()
        return

    subprocess.run([sys.executable, "tools/find_top_asparagus.py"])
    pause_prompt()


def run_tool_top_offline():
    print(f"\n{C_CYAN}[启动]{C_RESET} 正在检索最新快照并解算最顶层芦笋...")
    snapshots = sorted(glob.glob("data/snapshots/color_*.png"))
    if not snapshots:
        print(f"{C_YELLOW}[提示]{C_RESET} 尚未在 data/snapshots/ 目录下找到抓拍的 color_*.png 文件！")
        print(f"建议先在主菜单中选择 {C_GREEN}[5]{C_RESET} 生成一帧模拟数据。")
        pause_prompt()
        return

    latest_color = snapshots[-1]
    ts = os.path.basename(latest_color).replace("color_", "").replace(".png", "")
    latest_depth = f"data/snapshots/depth_raw_{ts}.npy"

    if not os.path.exists(latest_depth):
        print(f"{C_RED}[错误]{C_RESET} 未找到对应的深度数据文件: {latest_depth}")
        pause_prompt()
        return

    print(f"{C_GREEN}[INFO]{C_RESET} 加载快照数据: {latest_color}")
    subprocess.run([sys.executable, "tools/find_top_asparagus.py", "--image", latest_color, "--depth", latest_depth])
    pause_prompt()


def run_gen_mock_snapshot():
    print(f"\n{C_CYAN}[操作]{C_RESET} 正在生成仿真 3D 芦笋堆叠数据帧...")
    try:
        from tools.d435_viewer import D435Viewer
        viewer = D435Viewer(mock_mode=True)
        viewer.start()
        color, depth = viewer.generate_mock_frame()
        viewer.save_snapshot(color, depth, color)
        print(f"{C_GREEN}[成功]{C_RESET} 模拟数据已成功生成至 data/snapshots/！")
    except Exception as e:
        print(f"{C_RED}[失败]{C_RESET} 生成模拟数据异常: {e}")
    pause_prompt()


# ===================== 标定与建图专区功能 =====================

def run_generate_tags():
    print(f"\n{C_CYAN}[制靶]{C_RESET} 正在生成 AprilTag 16h5 高清标靶与排版图...")
    subprocess.run([sys.executable, "tools/calibration/generate_apriltags.py"])
    pause_prompt()


def run_tag_capture_wizard(status=None):
    active_scene = status.get('active_scene') if status else None
    scene_name = active_scene.scene_id if active_scene else "默认场景"
    print(f"\n{C_CYAN}[采图]{C_RESET} 正在启动 AprilTag 交互式多视角采图向导 (当前场景: {C_GREEN}{scene_name}{C_RESET})...")
    # 检查硬件
    ok, mode = ensure_camera_connected()
    cmd = [sys.executable, "tools/calibration/tag_capture_wizard.py"]
    if active_scene:
        cmd.extend(["--output_dir", active_scene.raw_images_dir])
    if mode == "mock" or not ok:
        print(f"{C_YELLOW}[提示]{C_RESET} 正在以 --mock 仿真模式启动采图向导...")
        cmd.append("--mock")
    
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n{C_RED}[异常退出] 采图向导异常退出 (退出码: {res.returncode})，详细错误堆栈如上所示。{C_RESET}")
        pause_prompt()
    elif active_scene:
        active_scene.refresh_stats()
        active_scene.save_meta()


def run_tag_super_extractor(status=None):
    active_scene = status.get('active_scene') if status else None
    image_dir = active_scene.raw_images_dir if active_scene else "data/tag_calibration_images"
    manifest_path = active_scene.manifest_path if active_scene else "data/tag_calibration_images/tag_observations.yaml"
    scene_name = active_scene.scene_id if active_scene else "默认目录"

    print(f"\n{C_CYAN}[工序 4: 超精提取]{C_RESET} 正在启动 AprilTag 图像质量诊断与超精重提取 (当前场景: {C_GREEN}{scene_name}{C_RESET})...")
    images = glob.glob(os.path.join(image_dir, "*.png"))
    if not images:
        print(f"{C_YELLOW}[提示]{C_RESET} 当前场景目录 ({image_dir}) 下没有图像！")
        print(f"请先运行工序 {C_GREEN}[3]{C_RESET} 采图向导，拍摄约 10~20 张多视角标靶照片后再运行重提取。")
        pause_prompt()
        return

    print(f"{C_GREEN}[性能解耦说明]{C_RESET} 离线引擎针对静态磁盘原图批处理，彻底解除 CPU 与耗时限制。")
    print(f"执行多尺度 CLAHE 增强、16级自适应阈值网格、微靶超分重判与 0.01px 亚像素精修。")
    cmd = [sys.executable, "tools/calibration/tag_super_extractor.py", "--image_dir", image_dir, "--manifest", manifest_path]
    subprocess.run(cmd)
    if active_scene:
        active_scene.refresh_stats()
        active_scene.save_meta()
    pause_prompt()


def run_build_tag_map(status=None):
    active_scene = status.get('active_scene') if status else None
    image_dir = active_scene.raw_images_dir if active_scene else "data/tag_calibration_images"
    manifest_path = active_scene.manifest_path if active_scene else "data/tag_calibration_images/tag_observations.yaml"
    map_path = active_scene.map_path if active_scene else "config/tags_map.yaml"
    scene_name = active_scene.scene_id if active_scene else "默认目录"

    print(f"\n{C_CYAN}[工序 5: BA平差建图]{C_RESET} 正在启动 AprilTag 3D 空间立体建图与两阶段 BA 平差求解 (当前场景: {C_GREEN}{scene_name}{C_RESET})...")
    images = glob.glob(os.path.join(image_dir, "*.png"))
    if not images:
        print(f"{C_YELLOW}[提示]{C_RESET} 当前场景目录 ({image_dir}) 下没有图像！")
        print(f"请先运行工序 {C_GREEN}[3]{C_RESET} 采图向导，拍摄约 10~20 张多视角标靶照片后再运行建图。")
        pause_prompt()
        return

    cmd = [sys.executable, "tools/calibration/tag_map_builder.py", "--image_dir", image_dir, "--manifest", manifest_path, "--output", map_path]
    subprocess.run(cmd)
    if active_scene:
        active_scene.refresh_stats()
        active_scene.save_meta()
    pause_prompt()


def run_tag_calibration_verifier():
    print(f"\n{C_CYAN}[工序 7: 在线AR验证]{C_RESET} 正在启动标定精度与 3D 坐标系在线 AR 综合验证系统 (tag_calibration_verifier.py)...")
    map_path = "config/tags_map.yaml"
    if not os.path.exists(map_path):
        print(f"{C_YELLOW}[提示]{C_RESET} 尚未检测到生产标靶地图文件: {map_path}！")
        print(f"请先在场景管理器 {C_GREEN}[2]{C_RESET} 或离线工作站中将平差完毕的场景地图【[P]/[U] 一键发布至生产环境】后再运行在线 AR 验证。")
        pause_prompt()
        return

    ok, mode = ensure_camera_connected()
    cmd = [sys.executable, "tools/calibration/tag_calibration_verifier.py"]
    if mode == "mock" or not ok:
        print(f"{C_YELLOW}[提示]{C_RESET} 正在以 --mock 仿真模式启动 AR 综合验证...")
        cmd.append("--mock")

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n{C_RED}[异常退出] 在线 AR 验证系统异常退出 (退出码: {res.returncode})，详细错误堆栈如上所示。{C_RESET}")
        pause_prompt()


def run_hand_eye_calibration():
    print(f"\n{C_CYAN}[标定]{C_RESET} 正在启动 SCARA 经典接触式物理标定向导 (hand_eye_calibration.py)...")
    subprocess.run([sys.executable, "tools/calibration/hand_eye_calibration.py"])
    pause_prompt()


def run_diagnose_tag_frame():
    print(f"\n{C_CYAN}[诊断]{C_RESET} 正在启动 AprilTag 真实图像深度病因诊断 (diagnose_tag_frame.py)...")
    subprocess.run([sys.executable, "tools/calibration/diagnose_tag_frame.py"])
    pause_prompt()


def run_clear_calib_dataset(status=None):
    active_scene = status.get('active_scene') if status else None
    image_dir = active_scene.raw_images_dir if active_scene else "data/tag_calibration_images"
    scene_name = active_scene.scene_id if active_scene else "默认目录"

    print(f"\n{C_CYAN}[清理]{C_RESET} 准备清空当前活动场景采图数据集 (当前场景: {C_GREEN}{scene_name}{C_RESET} | {image_dir})...")
    files = glob.glob(os.path.join(image_dir, "*.png"))
    if not files:
        print(f"{C_GREEN}[提示]{C_RESET} 当前场景采图目录已为空，无需清理。")
        pause_prompt()
        return

    print(f"当前场景共有 {len(files)} 张旧采图照片。")
    ans = input(f"是否确认清空该场景的所有照片从 0 开始重新采集？(其他场景不受影响) [Y/N]: ").strip().lower()
    if ans == 'y':
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
        if active_scene:
            active_scene.refresh_stats()
            active_scene.save_meta()
        print(f"{C_GREEN}[OK] 已成功清空场景 [{scene_name}] 的标定图像！下次采图将重新从 view_0001.png 开始。{C_RESET}")
    else:
        print(f"{C_YELLOW}[已取消]{C_RESET} 操作已终止。")
    pause_prompt()


def run_tag_whitelist_manager():
    """AprilTag 标靶白名单查看与管理子菜单"""
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
        print(f"{C_CYAN}{C_BOLD}            【AprilTag 标靶 ID 白名单管理】(Tag Whitelist Config)              {C_RESET}")
        print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
        
        # 实时读取配置文件
        current_ids = []
        raw_cfg = {}
        if os.path.exists(cfg_path):
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw_cfg = yaml.safe_load(f) or {}
                current_ids = raw_cfg.get("calibration", {}).get("valid_tag_ids", [])
            except Exception as e:
                print(f"{C_RED}[WARN] 读取 config.yaml 异常: {e}{C_RESET}")

        if not current_ids:
            print(f" 当前运行模式: {C_CYAN}{C_BOLD}【全量探索模式】(放行所有检测到的 16h5 标靶 ID 0~29){C_RESET}")
            print(f" 白名单配置  : valid_tag_ids: [] (未受限)")
        else:
            print(f" 当前运行模式: {C_GREEN}{C_BOLD}【物理白名单过滤模式】(仅放行预设物理标靶，杜绝噪点误识){C_RESET}")
            print(f" 当前已设 ID : {C_GREEN}{sorted(current_ids)}{C_RESET} (共 {len(current_ids)} 个)")

        print(f"\n{C_BOLD}模式说明：{C_RESET}")
        print(f"  * {C_YELLOW}全量探索模式{C_RESET}：当您重新打印/张贴了新标靶，标靶 ID 尚未确定时使用，采图向导将放行所有检测到的标靶；")
        print(f"  * {C_GREEN}白名单模式{C_RESET}  ：当物理标靶张贴固定后，锁定已知标靶 ID 列表，可彻底杜绝复杂反光背景下的伪造虚警。")
        print(f"{C_CYAN}-------------------------------------------------------------------------------{C_RESET}")
        print(f"   {C_GREEN}[1]{C_RESET} 刷新查看当前配置")
        print(f"   {C_GREEN}[2]{C_RESET} {C_CYAN}一键清空白名单 -> 切换至【全量探索模式】{C_RESET} (放行所有标靶)")
        print(f"   {C_GREEN}[3]{C_RESET} 手动设定白名单 ID 列表                  (如: 0,1,2,4,11-14)")
        print("")
        print(f"   {C_YELLOW}[B]{C_RESET} 返回标定专区")
        print(f"{C_CYAN}==============================================================================={C_RESET}")

        sub_choice = input(f"请输入操作编号 [1-3, B]: ").strip().upper()
        if sub_choice == '1':
            continue
        elif sub_choice == '2':
            try:
                import yaml
                if "calibration" not in raw_cfg:
                    raw_cfg["calibration"] = {}
                raw_cfg["calibration"]["valid_tag_ids"] = []
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(raw_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"\n{C_GREEN}[OK] 成功切换为【全量探索模式】！config.yaml 中 valid_tag_ids 已清空。{C_RESET}")
            except Exception as e:
                print(f"\n{C_RED}[ERROR] 保存失败: {e}{C_RESET}")
            pause_prompt()
        elif sub_choice == '3':
            print(f"\n请输入新的标靶 ID 列表，支持逗号分隔和连续范围，例如: {C_GREEN}0, 1, 2, 4, 11-14{C_RESET}")
            input_val = input("请输入: ").strip()
            if not input_val:
                print(f"{C_YELLOW}[提示] 未输入内容，操作已取消。{C_RESET}")
                pause_prompt()
                continue
            
            parsed_ids = set()
            try:
                parts = [p.strip() for p in input_val.replace("，", ",").split(",") if p.strip()]
                for part in parts:
                    if "-" in part:
                        start_s, end_s = part.split("-", 1)
                        for x in range(int(start_s.strip()), int(end_s.strip()) + 1):
                            parsed_ids.add(x)
                    else:
                        parsed_ids.add(int(part))
                
                final_list = sorted(list(parsed_ids))
                import yaml
                if "calibration" not in raw_cfg:
                    raw_cfg["calibration"] = {}
                raw_cfg["calibration"]["valid_tag_ids"] = final_list
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(raw_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"\n{C_GREEN}[OK] 白名单已成功更新为: {final_list} (已保存至 config.yaml){C_RESET}")
            except Exception as e:
                print(f"\n{C_RED}[ERROR] 输入解析或保存失败: {e}，请检查输入格式 (如 0, 1, 2){C_RESET}")
            pause_prompt()
        elif sub_choice in ('B', '0'):
            break


def run_open_visualized_dir(status=None):
    """在操作系统文件资源管理器中直接打开当前场景图示化分析文件目录"""
    active_scene = status.get('active_scene') if status else None
    vis_dir = active_scene.visualized_dir if active_scene else os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "visualized")
    os.makedirs(vis_dir, exist_ok=True)
    scene_name = active_scene.scene_id if active_scene else "默认目录"
    print(f"\n{C_CYAN}[浏览]{C_RESET} 正在打开场景 [{scene_name}] 的图示化分析图像目录: {vis_dir}...")
    try:
        if sys.platform == "win32":
            os.startfile(vis_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", vis_dir])
        else:
            subprocess.run(["xdg-open", vis_dir])
        print(f"{C_GREEN}[OK] 已在操作系统资源管理器中弹出该目录窗口，您可以直接双击观察带标注的图示化分析图像。{C_RESET}")
    except Exception as e:
        print(f"{C_RED}[WARN] 无法自动弹出窗口，请手动访问: {vis_dir} ({e}){C_RESET}")


def run_open_observations_manifest():
    """启动标靶交互审核画板 (tag_manifest_reviewer.py)"""
    manifest_path = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "tag_observations.yaml")
    if not os.path.exists(manifest_path):
        print(f"\n{C_YELLOW}[提示]{C_RESET} 尚未检测到审核清单: {manifest_path}")
        print(f"请先运行工序 {C_GREEN}[4]{C_RESET}，系统将自动扫描当前采图并生成观测清单。")
        pause_prompt()
        return

    print(f"\n{C_CYAN}[启动]{C_RESET} 正在启动 AprilTag 观测样本轻量级交互审核画板...")
    res = subprocess.run([sys.executable, "tools/calibration/tag_manifest_reviewer.py"])
    if res.returncode != 0:
        print(f"{C_YELLOW}[回退]{C_RESET} 无法正常启动图形画板，正在尝试在系统默认文本编辑器中打开 YAML 清单...")
        try:
            if sys.platform == "win32":
                os.startfile(manifest_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", manifest_path])
            else:
                subprocess.run(["xdg-open", manifest_path])
        except Exception as e:
            print(f"{C_RED}[WARN] 无法自动打开编辑器: {e}，请手动编辑该文件。{C_RESET}")
        pause_prompt()


def run_offline_verifier(status=None):
    """运行离线标定精度体检与 LOO 盲测批量验证"""
    active_scene = status.get('active_scene') if status else None
    map_path = active_scene.map_path if active_scene else os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
    image_dir = active_scene.raw_images_dir if active_scene else os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")
    scene_name = active_scene.scene_id if active_scene else "默认场景"

    if not os.path.exists(map_path) or os.path.getsize(map_path) < 50:
        print(f"\n{C_YELLOW}[提示]{C_RESET} 尚未检测到场景 [{scene_name}] 的有效标靶地图文件: {map_path}！")
        print(f"请先在离线 Studio {C_GREEN}[S]{C_RESET} 或工序 {C_GREEN}[5]{C_RESET} 中完成 BA 平差求解生成地图后再进行精度体检。")
        pause_prompt()
        return

    print(f"\n{C_CYAN}[工序 6: 离线体检]{C_RESET} 正在启动场景 [{scene_name}] 的标定精度体检与 Leave-One-Out 盲测批量验证...")
    cmd = [sys.executable, "tools/calibration/tag_offline_verifier.py", "--images", image_dir, "--map", map_path]
    subprocess.run(cmd)
    pause_prompt()


def run_offline_studio(status=None):
    """启动 AprilTag 离线标定综合工作站 (Tag Offline Studio)"""
    active_scene = status.get('active_scene') if status else None
    cmd = [sys.executable, "tools/calibration/tag_offline_studio.py"]
    scene_name = active_scene.scene_id if active_scene else "默认场景"
    if active_scene:
        cmd.extend(["--images", active_scene.raw_images_dir, "--map", active_scene.map_path])
    print(f"\n{C_CYAN}[旗舰工作站]{C_RESET} 正在启动 AprilTag 离线标定综合工作站 (当前沙盒: {C_GREEN}{scene_name}{C_RESET})...")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n{C_RED}[异常退出] 离线综合工作站异常退出 (退出码: {res.returncode}){C_RESET}")
        pause_prompt()
    elif active_scene:
        active_scene.refresh_stats()
        active_scene.save_meta()


def run_gui_launcher():
    """启动 3D 视觉综合控制中心 (GUI Launcher)"""
    print(f"\n{C_CYAN}[控制中心]{C_RESET} 正在启动 3D 视觉综合控制中心 (GUI Launcher)...")
    cmd = [sys.executable, "tools/gui_launcher.py"]
    try:
        res = subprocess.run(cmd)
        return res.returncode == 0
    except Exception as e:
        print(f"\n{C_YELLOW}[提示]{C_RESET} 无法启动 GUI 控制中心 ({e})，返回终端控制台...")
        return False


def run_scene_hub(status=None):
    """优先启动工况与场景综合管理 GUI 中枢 (Scene Hub)，异常时优雅回退至命令行菜单"""
    print(f"\n{C_CYAN}[中枢]{C_RESET} 正在启动工况与场景综合管理 GUI 中枢 (Scene Hub)...")
    cmd = [sys.executable, "-m", "tools.scene_hub"]
    try:
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"\n{C_YELLOW}[提示]{C_RESET} GUI 驾驶舱异常退出 (退出码: {res.returncode})，切入文本式命令行场景管理器...")
            submenu_scene_manager(status.get('scene_mgr') if status else None)
    except Exception as e:
        print(f"\n{C_YELLOW}[回退]{C_RESET} 无法启动 GUI 驾驶舱 ({e})，切入文本式命令行场景管理器...")
        submenu_scene_manager(status.get('scene_mgr') if status else None)


def submenu_scene_manager(scene_mgr):
    """标定采样场景与批次分组管理专属子菜单"""
    import time
    if not scene_mgr:
        try:
            from src.calibration.scene_manager import CalibrationSceneManager
            scene_mgr = CalibrationSceneManager()
        except Exception as e:
            print(f"{C_RED}[错误] 场景管理器未能正常加载: {e}{C_RESET}")
            pause_prompt()
            return

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
        print(f"{C_CYAN}{C_BOLD}       【标定采样场景与批次分组管理系统】(Calibration Scene Management)         {C_RESET}")
        print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
        active_id = scene_mgr.get_active_scene_id()
        scenes = scene_mgr.list_scenes()
        print(f" 场景总库目录: {C_GRAY}{scene_mgr.scenes_dir}{C_RESET}")
        print(f" 当前活动场景: {C_BOLD}{C_GREEN}{active_id}{C_RESET}")
        print(f"{C_CYAN}-------------------------------------------------------------------------------{C_RESET}")
        print(f" {C_BOLD}{'序号':<4} {'场景唯一标识 (ID)':<28} {'别名':<14} {'采图':<6} {'平差RMSE':<12} {'生产发布'}{C_RESET}")
        print(f"{C_GRAY}" + "-" * 79 + f"{C_RESET}")
        for idx, sc in enumerate(scenes, 1):
            is_active = (sc.scene_id == active_id)
            active_marker = f"{C_GREEN}★ [当前活动]{C_RESET}" if is_active else "            "
            pub_marker = f"{C_GREEN}★已发布{C_RESET}" if sc.is_published else f"{C_GRAY}草稿{C_RESET}"
            rmse_str = f"{sc.global_rmse_px:.3f} px" if sc.ba_solved else f"{C_GRAY}未求解{C_RESET}"
            color = C_GREEN if is_active else C_RESET
            print(f" {color}[{idx:02d}]{C_RESET} {color}{sc.scene_id:<28}{C_RESET} {sc.name:<14} {sc.image_count:<6} {rmse_str:<12} {pub_marker} {active_marker}")
        print(f"   {C_GREEN}[S]{C_RESET} 切换活动场景                    {C_GREEN}[N]{C_RESET} 新建采样工况场景 (支持中文)")
        print(f"   {C_GREEN}[R]{C_RESET} 修改当前场景名称 (支持中文)    {C_GREEN}[P]{C_RESET} 生效到生产系统 (覆盖全局 config/tags_map.yaml)")
        print(f"   {C_GREEN}[C]{C_RESET} 克隆当前场景作为对比实验        {C_RED}[D]{C_RESET} 安全删除指定废弃场景")
        print(f"   {C_YELLOW}[B]{C_RESET} 返回标定专区")
        print(f"{C_CYAN}==============================================================================={C_RESET}")
        sub_ch = input(f"请输入操作指令 [S, N, R, C, P, D, B 或 场景序号 1-{len(scenes)}]: ").strip().upper()

        if sub_ch in ('B', 'Q', ''):
            break
        elif sub_ch.isdigit():
            idx = int(sub_ch)
            if 1 <= idx <= len(scenes):
                target_sc = scenes[idx - 1]
                scene_mgr.set_active_scene(target_sc.scene_id)
                print(f"\n{C_GREEN}[成功] 活动场景已切换为: {target_sc.scene_id}{C_RESET}")
                time.sleep(0.6)
            else:
                print(f"{C_RED}[!] 序号超出范围{C_RESET}")
                time.sleep(0.8)
        elif sub_ch == 'S':
            sid = input(f"请输入要切换的目标场景 ID (或序号): ").strip()
            if sid.isdigit() and 1 <= int(sid) <= len(scenes):
                target_sc = scenes[int(sid) - 1]
                scene_mgr.set_active_scene(target_sc.scene_id)
                print(f"\n{C_GREEN}[成功] 活动场景已切换为: {target_sc.scene_id}{C_RESET}")
            elif any(s.scene_id == sid for s in scenes):
                scene_mgr.set_active_scene(sid)
                print(f"\n{C_GREEN}[成功] 活动场景已切换为: {sid}{C_RESET}")
            else:
                print(f"{C_RED}[!] 目标场景不存在{C_RESET}")
            time.sleep(1)
        elif sub_ch == 'R':
            curr = scene_mgr.get_active_scene()
            print(f"\n当前选中场景: {curr.name} ({curr.scene_id})")
            new_name = input(f"请输入新的显示名称/别名 (支持中文，如: 1号机台主标定): ").strip()
            if new_name:
                ok = scene_mgr.rename_scene(curr.scene_id, new_name)
                if ok:
                    print(f"\n{C_GREEN}[成功] 场景名称已修改为: {new_name}{C_RESET}")
                else:
                    print(f"\n{C_RED}[失败] 修改场景名称失败{C_RESET}")
            pause_prompt()
        elif sub_ch == 'N':
            alias = input(f"请输入新场景别名 (支持中文、英文、数字，如: 2号机架高位): ").strip()
            if not alias:
                print(f"{C_YELLOW}[提示] 别名不能为空，操作已取消。{C_RESET}")
                time.sleep(0.8)
                continue
            desc = input(f"请输入场景说明备注 (可选): ").strip()
            new_sc = scene_mgr.create_scene(alias=alias, description=desc)
            print(f"\n{C_GREEN}[成功] 已创建并激活新场景: {new_sc.name} ({new_sc.scene_id}){C_RESET}")
            pause_prompt()
        elif sub_ch == 'C':
            curr = scene_mgr.get_active_scene()
            new_alias = input(f"请输入克隆场景新别名 (基于当前 {curr.name}): ").strip()
            if not new_alias:
                print(f"{C_YELLOW}[提示] 别名不能为空，操作已取消。{C_RESET}")
                time.sleep(0.8)
                continue
            cloned = scene_mgr.clone_scene(curr.scene_id, new_alias=new_alias)
            if cloned:
                print(f"\n{C_GREEN}[成功] 已成功克隆并激活新场景: {cloned.scene_id}{C_RESET}")
            else:
                print(f"{C_RED}[失败] 克隆场景失败！{C_RESET}")
            pause_prompt()
        elif sub_ch == 'P':
            curr = scene_mgr.get_active_scene()
            print(f"\n{C_YELLOW}[确认] 准备将场景 [{curr.scene_id}] 的三维地图发布覆盖至全局生产环境 (config/tags_map.yaml)...{C_RESET}")
            confirm = input(f"确认发布？(Y/N): ").strip().upper()
            if confirm == 'Y':
                ok, msg = scene_mgr.publish_to_production(curr.scene_id)
                if ok:
                    print(f"{C_GREEN}[成功] {msg}{C_RESET}")
                else:
                    print(f"{C_RED}[失败] {msg}{C_RESET}")
                pause_prompt()
        elif sub_ch == 'D':
            del_id = input(f"请输入要删除的废弃场景 ID (严禁删除当前活动场景): ").strip()
            if del_id.isdigit() and 1 <= int(del_id) <= len(scenes):
                del_id = scenes[int(del_id) - 1].scene_id
            if del_id:
                confirm = input(f"警告：该操作将永久物理删除场景 [{del_id}] 及其所有图片！确认删除？(Y/N): ").strip().upper()
                if confirm == 'Y':
                    ok, msg = scene_mgr.delete_scene(del_id)
                    if ok:
                        print(f"{C_GREEN}[成功] {msg}{C_RESET}")
                    else:
                        print(f"{C_RED}[失败] {msg}{C_RESET}")
                    pause_prompt()


def submenu_calibration_suite(cached_status=None):
    """二级子菜单：手眼标定与 AprilTag 空间建图专区 (支持传入预检状态秒开进入)"""
    first_run = True
    while True:
        if first_run and cached_status is not None:
            status = cached_status
            first_run = False
        else:
            status = check_env_status()
        print_calibration_banner(status)
        choice = input(f"请输入工序编号 [S, 1-6, H, W, V, D, C, 8, B]: ").strip().upper()
        
        if choice in ('S', 'STUDIO', 'O'):
            run_offline_studio(status)
        elif choice == '1':
            run_generate_tags()
        elif choice in ('2', 'C', 'CAP'):
            run_tag_capture_wizard(status)
        elif choice in ('3', '4'):
            run_tag_super_extractor(status)
        elif choice in ('4', '5', 'M'):
            run_build_tag_map(status)
        elif choice in ('5', '6', 'P'):
            run_offline_verifier(status)
        elif choice in ('6', '7'):
            run_tag_calibration_verifier()
        elif choice in ('H', 'HUB', 'SCENE', '0'):
            run_scene_hub(status)
        elif choice in ('CLI', 'TXT'):
            submenu_scene_manager(status.get('scene_mgr'))
        elif choice == '8':
            run_hand_eye_calibration()
        elif choice == 'W':
            run_tag_whitelist_manager()
        elif choice == 'V':
            run_open_visualized_dir(status)
        elif choice == 'D':
            run_diagnose_tag_frame()
        elif choice == 'C':
            run_clear_calib_dataset(status)
        elif choice in ('B', 'Q'):
            break
        else:
            print(f"{C_RED}[!] 无效选项，请重新输入{C_RESET}")



# ===================== 自动化测试专区功能 =====================

def run_test_mock():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行仿真管线测试 (test_mock_pipeline.py)...")
    subprocess.run([sys.executable, "tests/test_mock_pipeline.py"])
    pause_prompt()


def run_test_real():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行真实快照算法测试 (test_real_snapshot.py)...")
    subprocess.run([sys.executable, "tests/test_real_snapshot.py"])
    pause_prompt()


def run_test_tag_builder():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行 AprilTag 空间平差建图数学单元测试 (test_tag_map_builder.py)...")
    subprocess.run([sys.executable, "tests/test_tag_map_builder.py"])
    pause_prompt()


def run_test_tag_verifier():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行 AprilTag 在线 AR 综合验证单元测试 (test_tag_calibration_verifier.py)...")
    subprocess.run([sys.executable, "tests/test_tag_calibration_verifier.py"])
    pause_prompt()


def run_test_tag_offline_verifier():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行 AprilTag 离线精度体检单元测试 (test_tag_offline_verifier.py)...")
    subprocess.run([sys.executable, "tests/test_tag_offline_verifier.py"])
    pause_prompt()


def run_test_scene_hub():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行 AprilTag 场景管理与取流单元测试 (tests/test_scene_hub.py)...")
    subprocess.run([sys.executable, "-m", "unittest", "tests/test_scene_hub.py"])
    pause_prompt()


def run_test_all():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在一键执行全部自动化测试...")
    print(f"{C_BOLD}--- 1. 运行仿真管线测试 ---{C_RESET}")
    res1 = subprocess.run([sys.executable, "tests/test_mock_pipeline.py"]).returncode
    print(f"\n{C_BOLD}--- 2. 运行真实快照测试 ---{C_RESET}")
    res2 = subprocess.run([sys.executable, "tests/test_real_snapshot.py"]).returncode
    print(f"\n{C_BOLD}--- 3. 运行 AprilTag 空间建图单元测试 ---{C_RESET}")
    res3 = subprocess.run([sys.executable, "tests/test_tag_map_builder.py"]).returncode
    print(f"\n{C_BOLD}--- 4. 运行 AprilTag 在线 AR 综合验证单元测试 ---{C_RESET}")
    res4 = subprocess.run([sys.executable, "tests/test_tag_calibration_verifier.py"]).returncode
    print(f"\n{C_BOLD}--- 5. 运行 AprilTag 离线精度体检单元测试 ---{C_RESET}")
    res5 = subprocess.run([sys.executable, "tests/test_tag_offline_verifier.py"]).returncode
    print(f"\n{C_BOLD}--- 6. 运行 AprilTag 场景管理与取流单元测试 ---{C_RESET}")
    res6 = subprocess.run([sys.executable, "-m", "unittest", "tests/test_scene_hub.py"]).returncode

    print(f"\n{C_CYAN}================ 测试汇总结果 ================{C_RESET}")
    print(f" 1. 仿真管线: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res1 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 2. 真实快照: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res2 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 3. 空间建图: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res3 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 4. 综合验证: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res4 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 5. 离线体检: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res5 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 6. 场景管理: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res6 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f"{C_CYAN}=============================================={C_RESET}")
    pause_prompt()


def submenu_test_suite():
    """二级子菜单：自动化测试与算法验证专区"""
    while True:
        print_test_banner()
        choice = input(f"请输入测试选项 [1-6, A, B]: ").strip().upper()
        
        if choice == '1':
            run_test_mock()
        elif choice == '2':
            run_test_real()
        elif choice == '3':
            run_test_tag_builder()
        elif choice == '4':
            run_test_tag_verifier()
        elif choice == '5':
            run_test_tag_offline_verifier()
        elif choice == '6':
            run_test_scene_hub()
        elif choice == 'A':
            run_test_all()
        elif choice == 'B' or choice == '0':
            break
        else:
            print(f"{C_RED}[!] 无效选项，请重新输入{C_RESET}")
            pause_prompt()


# ===================== 系统维护功能 =====================

def run_pip_install():
    print(f"\n{C_CYAN}[系统]{C_RESET} 正在执行 pip 安装依赖 (requirements.txt)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    pause_prompt()


def run_diagnostics():
    print(f"\n{C_CYAN}[诊断]{C_RESET} 正在执行系统与硬件环境诊断...")
    status = check_env_status()
    print("-" * 50)
    print(f"Python 解释器 : {sys.executable}")
    print(f"工作区根目录  : {PROJECT_ROOT}")
    print(f"OpenCV 状态   : {status['opencv'][1]}")
    print(f"NumPy 状态    : {status['numpy'][1]}")
    print(f"D435 相机驱动 : {status['realsense'][1]}")
    print(f"离线快照帧数  : {status['snapshot_count']} 帧")
    print(f"标定采图帧数  : {status['calib_image_count']} 帧")
    print(f"Tag 3D 地图   : {'存在 (config/tags_map.yaml)' if status['has_tag_map'] else '未创建'}")
    print("-" * 50)
    pause_prompt()


def run_open_cmd():
    print(f"\n{C_CYAN}[系统]{C_RESET} 正在启动工作区命令行 (输入 exit 返回)...")
    subprocess.run(["cmd.exe", "/K", "title flux_vision_3d 调试终端"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="flux_vision_3d 工业视觉控制终端")
    parser.add_argument("--cli", action="store_true", help="强制以纯文本字符控制台菜单模式运行")
    parser.add_argument("--diagnose", action="store_true", help="直接运行系统与驱动环境诊断后退出")
    args = parser.parse_args()

    if args.diagnose:
        run_diagnostics()
        return

    # 默认优先尝试启动 1280x720 工业科技大屏 GUI 控制中心 (如用户未指定 --cli)
    if not args.cli:
        # 检测是否可拉起 GUI
        try:
            if run_gui_launcher():
                return
        except Exception:
            pass

    # 终端文本交互模式
    while True:
        status = check_env_status()
        print_main_banner(status)
        choice = input(f"请输入选项编号并按回车 [G, 1-6, H, T, 8, 9, C, 0]: ").strip().upper()

        if choice in ('G', 'GUI'):
            run_gui_launcher()
        elif choice == '1':
            run_tool_d435_real()
        elif choice in ('2', 'HUB', 'SCENE'):
            run_scene_hub(status)
        elif choice == '3':
            run_tool_top_real()
        elif choice == '4':
            run_tool_top_offline()
        elif choice == '5':
            run_gen_mock_snapshot()
        elif choice == '6':
            run_tool_d435_mock()
        elif choice in ('H', 'CAL'):
            submenu_calibration_suite(cached_status=status)
        elif choice == 'T':
            submenu_test_suite()
        elif choice == '8':
            run_pip_install()
        elif choice == '9':
            run_diagnostics()
        elif choice == 'C':
            run_open_cmd()
        elif choice == '0':
            print(f"\n{C_GREEN}感谢使用 flux_vision_3d 控制终端，再见！{C_RESET}")
            sys.exit(0)
        else:
            print(f"{C_RED}[!] 无效选项，请重新输入{C_RESET}")
            pause_prompt()


if __name__ == "__main__":
    main()
