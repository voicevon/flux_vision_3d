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
import subprocess
from datetime import datetime

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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)


def check_env_status():
    """检查系统关键库及硬件、快照信息"""
    status = {}
    # RealSense 驱动与物理硬件检测
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if len(devices) > 0:
            dev_name = devices[0].get_info(rs.camera_info.name)
            status['realsense'] = (True, f"已连接 ({dev_name})", True)
        else:
            status['realsense'] = (False, "驱动已装，但未检测到物理相机(USB未连接)", False)
    except ImportError:
        status['realsense'] = (False, "未安装驱动库 pyrealsense2", False)
    except Exception as e:
        status['realsense'] = (False, f"相机状态异常: {e}", False)

    # OpenCV
    try:
        import cv2
        status['opencv'] = (True, f"v{cv2.__version__}")
    except ImportError:
        status['opencv'] = (False, "未安装")

    # NumPy
    try:
        import numpy as np
        status['numpy'] = (True, f"v{np.__version__}")
    except ImportError:
        status['numpy'] = (False, "未安装")

    # 快照统计
    snapshots = glob.glob(os.path.join(PROJECT_ROOT, "data", "snapshots", "color_*.png"))
    status['snapshot_count'] = len(snapshots)

    # 采图数据集统计
    calib_images = glob.glob(os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "*.png"))
    status['calib_image_count'] = len(calib_images)

    # 标靶空间地图状态
    map_path = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
    status['has_tag_map'] = os.path.exists(map_path)

    # 标靶观测数据审核清单状态
    manifest_path = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "tag_observations.yaml")
    status['has_manifest'] = os.path.exists(manifest_path)
    manifest_excluded = 0
    if status['has_manifest']:
        try:
            import yaml
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = yaml.safe_load(f) or {}
            for img in m.get("images", {}).values():
                for obs in img.get("observations", []):
                    if not obs.get("keep", True):
                        manifest_excluded += 1
        except Exception:
            pass
    status['manifest_excluded'] = manifest_excluded

    # 标靶 ID 白名单状态
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    valid_tag_ids = []
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            valid_tag_ids = cfg.get("calibration", {}).get("valid_tag_ids", [])
        except Exception:
            pass
    status['valid_tag_ids'] = valid_tag_ids

    return status


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
    print(f"{C_BOLD} [ 视觉预览与日常解算 (Vision Tools) ]{C_RESET}")
    print(f"   {C_GREEN}[1]{C_RESET} 启动 D435 实时相机查看器与深度探针     (物理硬件模式)")
    print(f"   {C_GREEN}[2]{C_RESET} 启动 D435 仿真模拟可视化查看器         ({C_YELLOW}--mock{C_RESET} 模式，无需物理相机)")
    print(f"   {C_GREEN}[3]{C_RESET} 解算最顶层芦笋抓取位姿 (实时相机)      (find_top_asparagus.py 单帧采集解算)")
    print(f"   {C_GREEN}[4]{C_RESET} 解算最顶层芦笋抓取位姿 (离线快照)      (自动读取最新本地快照)")
    print(f"   {C_GREEN}[5]{C_RESET} 快速生成一帧模拟快照至 snapshots       (方便无相机时进行算法验证)")
    print("")
    print(f"{C_BOLD} [ 核心向导与自动化专区 (Specialized Suites) ]{C_RESET}")
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
    print(f" 标准工序: {C_YELLOW}[1 制靶]{C_RESET} -> {C_YELLOW}[2 采图]{C_RESET} -> {C_YELLOW}[3 超精提取]{C_RESET} -> {C_YELLOW}[4 交互审核]{C_RESET} -> {C_YELLOW}[5 BA平差]{C_RESET} -> {C_YELLOW}[6 离线体检]{C_RESET} -> {C_YELLOW}[7 在线AR验证]{C_RESET}")
    map_status_str = f"{C_GREEN}已生成 (config/tags_map.yaml){C_RESET}" if status['has_tag_map'] else f"{C_YELLOW}未生成 (请执行 1~5 依次解算){C_RESET}"
    if not status.get('valid_tag_ids'):
        wl_status_str = f"{C_CYAN}全量探索 (放行所有标靶 0~29){C_RESET}"
    else:
        wl_status_str = f"{C_GREEN}白名单过滤 ({len(status['valid_tag_ids'])}个已知ID){C_RESET}"

    if status.get('has_manifest'):
        exc_cnt = status.get('manifest_excluded', 0)
        obs_str = f"{C_GREEN}已生成 (已剔除 {exc_cnt} 项){C_RESET}" if exc_cnt > 0 else f"{C_GREEN}已生成 (全量保留){C_RESET}"
    else:
        obs_str = f"{C_YELLOW}未生成 (进入工序3自动创建){C_RESET}"

    print(f" 状态一览: 采图集: {C_GREEN}{status['calib_image_count']}{C_RESET} 帧 | 审核清单: {obs_str} | 空间地图: {map_status_str}")
    print(f"{C_CYAN}-------------------------------------------------------------------------------{C_RESET}")
    print(f"{C_BOLD} [ 一、 标靶准备 (Target Preparation) ]{C_RESET}")
    print(f"   {C_GREEN}[1]{C_RESET} AprilTag 标靶图纸生成                  (生成 0~29 号高清标靶与 1:1 A4 排版 PDF)")
    print("")
    print(f"{C_BOLD} [ 二、 图像采集 (Image Acquisition) ]{C_RESET}")
    print(f"   {C_GREEN}[2]{C_RESET} AprilTag 多视角交互式采图向导          (1080P @ 8fps 丝滑轻量采图，空格一键连拍)")
    print("")
    print(f"{C_BOLD} [ 三、 离线解算与质量闭环 (Offline Pipeline & QA) ]{C_RESET}")
    print(f"   {C_GREEN}{C_BOLD}[S]{C_RESET} {C_CYAN}{C_BOLD}进入 AprilTag 离线标定综合工作站 (Offline Studio)  ★ 旗舰一站式集成工作台{C_RESET}")
    print(f"   {C_GREEN}[3]{C_RESET} 离线图像诊断调优与超精重提取           (16级阈值网格+双尺度CLAHE+0.01px亚像素精修)")
    print(f"   {C_GREEN}[4]{C_RESET} 标靶观测样本交互审核画板              (单靶/整帧剔除，拓扑连通把关) {C_GRAY}[快捷键: O]{C_RESET}")
    print(f"   {C_GREEN}[5]{C_RESET} 纯计算空间立体建图与两阶段 BA 平差     (tag_map_builder.py，全局误差优化) {C_GRAY}[快捷键: M]{C_RESET}")
    print(f"   {C_GREEN}[6]{C_RESET} 离线标定精度体检工作台 (LOO盲测体检)   (全量留一盲测批处理，残差矢量，门限放行) {C_GRAY}[快捷键: P]{C_RESET}")

    print("")
    print(f"{C_BOLD} [ 四、 在线验收与生产部署 (Online AR Verification & Deployment) ]{C_RESET}")
    print(f"   {C_GREEN}[7]{C_RESET} 标定精度在线 AR 综合实时验证系统      (相机实时取流，3D轴/棱柱虚实融合，静态位姿锁定)")
    print("")
    print(f"{C_BOLD} [ 五、 辅助工具与维护通道 (Auxiliary Tools & Maintenance) ]{C_RESET}")
    print(f"   {C_GREEN}[W]{C_RESET} AprilTag 标靶 ID 白名单管理            (查看当前/一键放行探索/指定有效 ID 列表)")
    print(f"   {C_GREEN}[V]{C_RESET} 浏览图示化分析与检测标注目录          (在系统资源管理器中打开 visualized/)")
    print(f"   {C_GREEN}[D]{C_RESET} 单帧标靶漏检病因深度诊断与切片分析    (分析真图淘汰候选框/尺寸/反差/模糊原因)")
    print(f"   {C_GREEN}[C]{C_RESET} 一键清空标定采图数据集                (重置采图集从 0 开始重新编号)")
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


def run_tag_capture_wizard():
    print(f"\n{C_CYAN}[采图]{C_RESET} 正在启动 AprilTag 交互式多视角采图向导 (tag_capture_wizard.py)...")
    # 检查硬件
    ok, mode = ensure_camera_connected()
    cmd = [sys.executable, "tools/calibration/tag_capture_wizard.py"]
    if mode == "mock" or not ok:
        print(f"{C_YELLOW}[提示]{C_RESET} 正在以 --mock 仿真模式启动采图向导...")
        cmd.append("--mock")
    
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n{C_RED}[异常退出] 采图向导异常退出 (退出码: {res.returncode})，详细错误堆栈如上所示。{C_RESET}")
        pause_prompt()


def run_tag_super_extractor():
    print(f"\n{C_CYAN}[工序 3: 超精提取]{C_RESET} 正在启动 AprilTag 离线图像质量诊断与超精重提取 (tag_super_extractor.py)...")
    images = glob.glob("data/tag_calibration_images/*.png")
    if not images:
        print(f"{C_YELLOW}[提示]{C_RESET} 当前 data/tag_calibration_images/ 目录下没有图像！")
        print(f"请先运行工序 {C_GREEN}[2]{C_RESET} 采图向导，拍摄约 10~20 张多视角标靶照片后再运行重提取。")
        pause_prompt()
        return

    print(f"{C_GREEN}[性能解耦说明]{C_RESET} 离线引擎针对静态磁盘原图批处理，彻底解除 CPU 与耗时限制。")
    print(f"执行多尺度 CLAHE 增强、16级自适应阈值网格、微靶超分重判与 0.01px 亚像素精修。")
    subprocess.run([sys.executable, "tools/calibration/tag_super_extractor.py"])
    pause_prompt()


def run_build_tag_map():
    print(f"\n{C_CYAN}[工序 5: BA平差建图]{C_RESET} 正在启动 AprilTag 3D 空间立体地图建图与两阶段 BA 平差求解 (tag_map_builder.py)...")
    images = glob.glob("data/tag_calibration_images/*.png")
    if not images:
        print(f"{C_YELLOW}[提示]{C_RESET} 当前 data/tag_calibration_images/ 目录下没有图像！")
        print(f"请先运行工序 {C_GREEN}[2]{C_RESET} 采图向导，拍摄约 10~20 张多视角标靶照片后再运行建图。")
        pause_prompt()
        return

    subprocess.run([sys.executable, "tools/calibration/tag_map_builder.py"])
    pause_prompt()


def run_tag_calibration_verifier():
    print(f"\n{C_CYAN}[工序 7: 在线AR验证]{C_RESET} 正在启动标定精度与 3D 坐标系在线 AR 综合验证系统 (tag_calibration_verifier.py)...")
    map_path = "config/tags_map.yaml"
    if not os.path.exists(map_path):
        print(f"{C_YELLOW}[提示]{C_RESET} 尚未检测到标靶地图文件: {map_path}！")
        print(f"请先执行工序 {C_GREEN}[5]{C_RESET} 空间建图与 BA 平差求解，或在审核画板 {C_GREEN}[4]{C_RESET} 按 [V] 键一键求解。")
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


def run_clear_calib_dataset():
    print(f"\n{C_CYAN}[清理]{C_RESET} 准备清空标定采图数据集 (data/tag_calibration_images/)...")
    files = glob.glob("data/tag_calibration_images/*.png")
    if not files:
        print(f"{C_GREEN}[提示]{C_RESET} 当前采图目录已为空，无需清理。")
        pause_prompt()
        return

    print(f"当前目录共有 {len(files)} 张旧采图照片。")
    ans = input("是否确认全部删除从 0 开始重新采集？[Y/N]: ").strip().lower()
    if ans == 'y':
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
        print(f"{C_GREEN}[OK] 已成功清空全部旧标定图像！下次采图将从 view_0001.png 重新编号。{C_RESET}")
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


def run_open_visualized_dir():
    """在操作系统文件资源管理器中直接打开图示化分析文件目录"""
    vis_dir = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "visualized")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"\n{C_CYAN}[浏览]{C_RESET} 正在打开图示化分析图像目录: {vis_dir}...")
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
        print(f"请先运行工序 {C_GREEN}[3]{C_RESET}，系统将自动扫描当前采图并生成观测清单。")
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


def run_offline_verifier():
    """运行离线标定精度体检与 LOO 盲测批量验证"""
    map_path = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
    if not os.path.exists(map_path):
        print(f"\n{C_YELLOW}[提示]{C_RESET} 尚未检测到标靶地图文件: {map_path}！")
        print(f"请先执行工序 {C_GREEN}[5]{C_RESET} 空间建图与 BA 平差求解生成地图后再进行精度体检。")
        pause_prompt()
        return

    print(f"\n{C_CYAN}[工序 6: 离线体检]{C_RESET} 正在启动离线标定精度体检与 Leave-One-Out 盲测批量验证...")
    subprocess.run([sys.executable, "tools/calibration/tag_offline_verifier.py"])
    pause_prompt()


def run_offline_studio():
    """启动 AprilTag 离线标定综合工作站 (Tag Offline Studio)"""
    print(f"\n{C_CYAN}[旗舰工作站]{C_RESET} 正在启动 AprilTag 离线标定综合工作站 (tag_offline_studio.py)...")
    res = subprocess.run([sys.executable, "tools/calibration/tag_offline_studio.py"])
    if res.returncode != 0:
        print(f"\n{C_RED}[异常退出] 离线综合工作站异常退出 (退出码: {res.returncode}){C_RESET}")
        pause_prompt()


def submenu_calibration_suite():
    """二级子菜单：手眼标定与 AprilTag 空间建图专区"""
    while True:
        status = check_env_status()
        print_calibration_banner(status)
        choice = input(f"请输入工序编号 [S, 1-8, W, V, D, C, M, O, P, B]: ").strip().upper()
        
        if choice in ('S', 'STUDIO'):
            run_offline_studio()
        elif choice == '1':
            run_generate_tags()

        elif choice == '2':
            run_tag_capture_wizard()
        elif choice == '3':
            run_tag_super_extractor()
        elif choice in ('4', 'O'):
            run_open_observations_manifest()
        elif choice in ('5', 'M'):
            run_build_tag_map()
        elif choice in ('6', 'P'):
            run_offline_verifier()
        elif choice == '7':
            run_tag_calibration_verifier()
        elif choice == '8':
            run_hand_eye_calibration()
        elif choice == 'W':
            run_tag_whitelist_manager()
        elif choice == 'V':
            run_open_visualized_dir()
        elif choice == 'D':
            run_diagnose_tag_frame()
        elif choice == 'C':
            run_clear_calib_dataset()
        elif choice in ('B', '0'):
            break
        else:
            print(f"{C_RED}[!] 无效选项，请重新输入{C_RESET}")
            pause_prompt()


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

    print(f"\n{C_CYAN}================ 测试汇总结果 ================{C_RESET}")
    print(f" 1. 仿真管线: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res1 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 2. 真实快照: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res2 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 3. 空间建图: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res3 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 4. 综合验证: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res4 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f" 5. 离线体检: {'[ ' + C_GREEN + 'PASS' + C_RESET + ' ]' if res5 == 0 else '[ ' + C_RED + 'FAIL' + C_RESET + ' ]'}")
    print(f"{C_CYAN}=============================================={C_RESET}")
    pause_prompt()


def submenu_test_suite():
    """二级子菜单：自动化测试与算法验证专区"""
    while True:
        print_test_banner()
        choice = input(f"请输入测试选项 [1-5, A, B]: ").strip().upper()
        
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
    while True:
        status = check_env_status()
        print_main_banner(status)
        choice = input(f"请输入选项编号并按回车 [1-5, H, T, 8, 9, C, 0]: ").strip().upper()
        
        if choice == '1':
            run_tool_d435_real()
        elif choice == '2':
            run_tool_d435_mock()
        elif choice == '3':
            run_tool_top_real()
        elif choice == '4':
            run_tool_top_offline()
        elif choice == '5':
            run_gen_mock_snapshot()
        elif choice in ('H', 'CAL'):
            submenu_calibration_suite()
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
