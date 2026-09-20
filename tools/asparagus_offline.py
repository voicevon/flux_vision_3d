#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋抓取位姿离线验证 GUI (Asparagus Offline) — 向前兼容门面
=========================================================
此模块已重构为模块化包 tools.asparagus_pose_studio。
本文件作为兼容层保留，完全透明转发至新包中的实现，确保历史测试用例与启动脚本无缝兼容。
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.asparagus_pose_studio import (
    APP_ID,
    BASE_H,
    BASE_W,
    CALIB_LABELS,
    CONFIG_PATH,
    DEFAULT_DIR,
    REPORT_DIR,
    WINDOW_KEY,
    AsparagusOfflineApp,
    AsparagusPoseStudioApp,
    WorkspaceManager,
    export_gcode_file,
    find_depth_pair,
    load_system_config,
    main,
    scan_samples,
)

__all__ = [
    "AsparagusOfflineApp",
    "AsparagusPoseStudioApp",
    "find_depth_pair",
    "scan_samples",
    "load_system_config",
    "export_gcode_file",
    "WorkspaceManager",
    "main",
    "APP_ID",
    "WINDOW_KEY",
    "BASE_W",
    "BASE_H",
    "DEFAULT_DIR",
    "REPORT_DIR",
    "CONFIG_PATH",
    "CALIB_LABELS",
]

if __name__ == "__main__":
    main()
