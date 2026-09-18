#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统环境深度诊断脚本 (Diagnose Env)
====================================
Dashboard [T] 卡片 (TERM 模式) 的承载脚本：
  - 检查 Python / OpenCV / NumPy / RealSense 驱动与本地数据就绪状态并输出诊断报告。
  - 全量测试执行命令：python -m unittest discover -s tests -p "test_*.py"
原为 cli_menu.py --diagnose 模式；cli_menu 交互菜单退役后独立成脚本。
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from tools.env_utils import check_env_status  # noqa: E402


def main():
    status = check_env_status()
    print("=" * 65)
    print(" flux_vision_3d 系统环境深度诊断报告")
    print("=" * 65)
    print(f"Python 解释器 : {sys.executable}")
    print(f"工作区根目录  : {PROJECT_ROOT}")
    print(f"OpenCV 状态   : {status['opencv'][1]}")
    print(f"NumPy 状态    : {status['numpy'][1]}")
    print(f"D435 相机驱动 : {status['realsense'][1]}")
    print(f"离线快照帧数  : {status['snapshot_count']} 帧")
    print(f"标定采图帧数  : {status['calib_image_count']} 帧")
    print(f"Tag 3D 地图   : {'存在 (config/tags_map.yaml)' if status['has_tag_map'] else '未创建'}")
    print("-" * 65)
    print("全量自动化测试: python -m unittest discover -s tests -p \"test_*.py\"")
    print("=" * 65)


if __name__ == "__main__":
    main()
