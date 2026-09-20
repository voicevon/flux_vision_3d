#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) 模块入口
支持使用 python -m tools.asparagus_pose_studio 启动
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.asparagus_pose_studio.app import main

if __name__ == "__main__":
    main()
