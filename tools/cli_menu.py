"""
flux_vision_3d 交互式 CLI 控制台与工具导航菜单
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

    return status


def print_banner(status):
    """打印控制台横幅与状态"""
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    print(f"{C_CYAN}{C_BOLD}             flux_vision_3d 芦笋 3D 视觉与抓取位姿估计系统 - 控制终端              {C_RESET}")
    print(f"{C_CYAN}{C_BOLD}==============================================================================={C_RESET}")
    
    cv_str = f"{C_GREEN}{status['opencv'][1]}{C_RESET}" if status['opencv'][0] else f"{C_RED}未安装{C_RESET}"
    np_str = f"{C_GREEN}{status['numpy'][1]}{C_RESET}" if status['numpy'][0] else f"{C_RED}未安装{C_RESET}"
    rs_str = f"{C_GREEN}{status['realsense'][1]}{C_RESET}" if status['realsense'][0] else f"{C_YELLOW}{status['realsense'][1]}{C_RESET}"
    snap_str = f"{C_GREEN}{status['snapshot_count']} 帧可用{C_RESET}" if status['snapshot_count'] > 0 else f"{C_GRAY}暂无快照{C_RESET}"
    
    print(f" 环境状态: Python {C_GREEN}{sys.version.split()[0]}{C_RESET} | OpenCV: {cv_str} | NumPy: {np_str} | D435驱动: {rs_str}")
    print(f" 本地数据: snapshots 快照目录 ({snap_str})")
    print(f"{C_CYAN}-------------------------------------------------------------------------------{C_RESET}")
    print(f"{C_BOLD} [ 视觉预览与交互工具 (Tools) ]{C_RESET}")
    print(f"   {C_GREEN}[1]{C_RESET} 启动 D435 实时相机查看器与深度探针     (物理硬件模式)")
    print(f"   {C_GREEN}[2]{C_RESET} 启动 D435 仿真模拟可视化查看器         ({C_YELLOW}--mock{C_RESET} 模式，无需物理相机)")
    print(f"   {C_GREEN}[3]{C_RESET} 解算最顶层芦笋抓取位姿 (实时相机)      (find_top_asparagus.py 单帧采集解算)")
    print(f"   {C_GREEN}[4]{C_RESET} 解算最顶层芦笋抓取位姿 (离线快照)      (自动读取最新本地快照)")
    print(f"   {C_GREEN}[5]{C_RESET} 快速生成一帧模拟快照至 snapshots       (方便无相机时进行算法验证)")
    print(f"   {C_GREEN}[H]{C_RESET} 运行 SCARA 手眼标定向导               (Eye-to-Hand 物理接触配准/求解矩阵)")
    print(f"   {C_GREEN}[T]{C_RESET} 生成 AprilTag 16h5 标靶高清图与排版   (生成 ID 0~19 标靶，含 Tag 0 原点对齐刻度)")
    print(f"   {C_GREEN}[M]{C_RESET} 构建 AprilTag 多标靶 3D 空间立体地图   (读取多视角图片运行 BA 全局平差建图)")
    print("")
    print(f"{C_BOLD} [ 自动化测试 (Tests) ]{C_RESET}")
    print(f"   {C_GREEN}[6]{C_RESET} 运行仿真管线测试                       (tests/test_mock_pipeline.py)")
    print(f"   {C_GREEN}[7]{C_RESET} 运行真实快照算法测试                   (tests/test_real_snapshot.py)")
    print(f"   {C_GREEN}[A]{C_RESET} 一键运行全部自动化测试")
    print("")
    print(f"{C_BOLD} [ 依赖与系统维护 ]{C_RESET}")
    print(f"   {C_GREEN}[8]{C_RESET} 安装 / 更新项目依赖包                  (pip install -r requirements.txt)")
    print(f"   {C_GREEN}[9]{C_RESET} 详细环境与驱动诊断")
    print(f"   {C_GREEN}[C]{C_RESET} 进入项目根目录命令行 (CMD)")
    print(f"   {C_RED}[0]{C_RESET} 退出控制终端")
    print(f"{C_CYAN}==============================================================================={C_RESET}")


def pause_prompt():
    print(f"\n{C_GRAY}按回车键返回主菜单...{C_RESET}", end="", flush=True)
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
    print("  1. 相机 Type-C 线缆未连接或接触不良，建议插在电脑背板或蓝色 USB 3.0 接口；")
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

    print(f"{C_GRAY}操作提示: [Space]定格/暂停画面 | [V]切换上下/单图放大视图 | [G]打印G-code | [D]芦笋检测 | [S]抓拍 | [Q]退出{C_RESET}")
    subprocess.run([sys.executable, "tools/d435_viewer.py"])
    pause_prompt()


def run_tool_d435_mock():
    print(f"\n{C_CYAN}[启动]{C_RESET} 正在以仿真模拟模式启动 D435 可视化查看器 (--mock)...")
    print(f"{C_GRAY}说明: 模拟 70cm 垂直俯视工况下的 3 层堆叠芦笋与工作台场景。{C_RESET}")
    subprocess.run([sys.executable, "tools/d435_viewer.py", "--mock"])
    pause_prompt()


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
        print(f"建议先在主菜单中选择 {C_GREEN}[5]{C_RESET} 生成一帧模拟数据，或通过 {C_GREEN}[1]/[2]{C_RESET} 查看器按 's' 抓拍。")
        pause_prompt()
        return

    latest_color = snapshots[-1]
    ts = os.path.basename(latest_color).replace("color_", "").replace(".png", "")
    latest_depth = f"data/snapshots/depth_raw_{ts}.npy"

    if not os.path.exists(latest_depth):
        print(f"{C_RED}[错误]{C_RESET} 未找到对应的深度数据文件: {latest_depth}")
        pause_prompt()
        return

    print(f"{C_GREEN}[INFO]{C_RESET} 加载快照数据:")
    print(f"  -> 彩色图像: {latest_color}")
    print(f"  -> 深度矩阵: {latest_depth}")
    
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
        print(f"现在可以返回主菜单选择 [4] 进行离线目标解算测试。")
    except Exception as e:
        print(f"{C_RED}[失败]{C_RESET} 生成模拟数据异常: {e}")
    pause_prompt()


def run_hand_eye_calibration():
    print(f"\n{C_CYAN}[标定]{C_RESET} 正在启动 SCARA 手眼标定向导 (hand_eye_calibration.py)...")
    subprocess.run([sys.executable, "tools/hand_eye_calibration.py"])
    pause_prompt()


def run_generate_tags():
    print(f"\n{C_CYAN}[标靶]{C_RESET} 正在生成 AprilTag 16h5 高清标靶与排版图...")
    subprocess.run([sys.executable, "tools/generate_apriltags.py"])
    pause_prompt()


def run_build_tag_map():
    print(f"\n{C_CYAN}[建图]{C_RESET} 正在启动多标靶空间立体地图建图工具 (tag_map_builder.py)...")
    subprocess.run([sys.executable, "tools/tag_map_builder.py"])
    pause_prompt()


def run_test_mock():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行仿真管线测试 (test_mock_pipeline.py)...")
    subprocess.run([sys.executable, "tests/test_mock_pipeline.py"])
    pause_prompt()


def run_test_real():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在执行真实快照算法测试 (test_real_snapshot.py)...")
    subprocess.run([sys.executable, "tests/test_real_snapshot.py"])
    pause_prompt()


def run_test_all():
    print(f"\n{C_CYAN}[测试]{C_RESET} 正在依次运行全部自动化测试套件...")
    print(f"\n--- [1/2] 运行 test_mock_pipeline.py ---")
    ret1 = subprocess.run([sys.executable, "tests/test_mock_pipeline.py"])
    print(f"\n--- [2/2] 运行 test_real_snapshot.py ---")
    ret2 = subprocess.run([sys.executable, "tests/test_real_snapshot.py"])
    print(f"\n{C_BOLD}==============================================================================={C_RESET}")
    if ret1.returncode == 0 and ret2.returncode == 0:
        print(f"{C_GREEN}[ALL PASSED] 所有测试均成功通过！{C_RESET}")
    else:
        print(f"{C_RED}[WARNING] 部分测试未通过或跳过，请检查上述日志。{C_RESET}")
    print(f"{C_BOLD}==============================================================================={C_RESET}")
    pause_prompt()


def run_install_reqs():
    print(f"\n{C_CYAN}[安装]{C_RESET} 正在根据 requirements.txt 安装/更新依赖包...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    pause_prompt()


def run_env_diagnostics():
    print(f"\n{C_CYAN}[诊断]{C_RESET} 正在检查当前系统环境与依赖库详情:")
    print("-" * 60)
    print(f"Python 路径 : {sys.executable}")
    print(f"Python 版本 : {sys.version.splitlines()[0]}")
    
    modules = [
        ("OpenCV", "cv2", "__version__"),
        ("NumPy", "numpy", "__version__"),
        ("PyYAML", "yaml", "__version__"),
        ("SciPy", "scipy", "__version__"),
        ("RealSense SDK", "pyrealsense2", "__version__"),
        ("PySerial (串口)", "serial", "__version__"),
        ("Bleak (BLE蓝牙)", "bleak", "__version__"),
        ("Matplotlib", "matplotlib", "__version__"),
    ]
    
    for name, mod_name, ver_attr in modules:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, ver_attr, "已就绪")
            print(f" [OK] {name:<16} : {ver}")
        except ImportError:
            print(f" {C_YELLOW}[未安装]{C_RESET} {name:<12} : 缺失 (可使用选项 8 安装)")

    print("-" * 60)
    pause_prompt()


def run_cmd_prompt():
    print(f"\n{C_CYAN}[终端]{C_RESET} 已进入项目工作目录: {PROJECT_ROOT}")
    print(f"{C_GRAY}输入 'exit' 并按回车即可返回主菜单。{C_RESET}\n")
    if sys.platform == "win32":
        subprocess.run(["cmd.exe", "/k"])
    else:
        subprocess.run([os.environ.get("SHELL", "/bin/bash")])


def main():
    while True:
        status = check_env_status()
        print_banner(status)
        try:
            choice = input(f"{C_BOLD}请输入选项编号并按回车 [0-9, A, C]: {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_YELLOW}用户中断，退出控制台。{C_RESET}")
            break

        if choice == "1":
            run_tool_d435_real()
        elif choice == "2":
            run_tool_d435_mock()
        elif choice == "3":
            run_tool_top_real()
        elif choice == "4":
            run_tool_top_offline()
        elif choice == "5":
            run_gen_mock_snapshot()
        elif choice.upper() == "H":
            run_hand_eye_calibration()
        elif choice.upper() == "T":
            run_generate_tags()
        elif choice.upper() == "M":
            run_build_tag_map()
        elif choice == "6":
            run_test_mock()
        elif choice == "7":
            run_test_real()
        elif choice.upper() == "A":
            run_test_all()
        elif choice == "8":
            run_install_reqs()
        elif choice == "9":
            run_env_diagnostics()
        elif choice.upper() == "C":
            run_cmd_prompt()
        elif choice == "0":
            print(f"\n{C_GREEN}已退出 flux_vision_3d 控制终端。祝研发顺利！{C_RESET}\n")
            break
        else:
            print(f"\n{C_RED}[提示] 无效的输入 '{choice}'，请重新输入！{C_RESET}")
            import time
            time.sleep(1)


if __name__ == "__main__":
    main()
