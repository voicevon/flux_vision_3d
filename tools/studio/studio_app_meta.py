#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Studio 共享常量模块 (studio_app_meta.py)
========================================
集中存放 app.py 与各 Mixin 模块共同引用的模块级常量，
避免 Mixin 反向导入 app.py 造成循环依赖。仅存放常量，不定义类。
"""

import os

# 项目根目录 (tools/studio 的上两级目录)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
