#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[向后兼容转发桩 / Deprecation Forwarder]
该模块已整理归类迁移至: tools/calibration/diagnose_tag_frame.py
此处保留轻量级转发接口，确保外部自动化脚本与开发者历史命令 100% 无缝兼容。
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.calibration.diagnose_tag_frame import *
from tools.calibration.diagnose_tag_frame import main

if __name__ == "__main__":
    main()
