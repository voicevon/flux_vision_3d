#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标准化日志工具 (Logger)
================================
项目统一日志入口，替代分散的 print 输出：
- 统一命名空间 `flux_vision.*`，控制台分级输出 (INFO/WARNING/ERROR)
- 幂等配置：任意模块首次调用即完成全局 handler 装配
- 用法: `log = get_logger(__name__)` 后使用 log.info / log.warning / log.error
"""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """获取项目统一配置的 logger (幂等配置，线程安全由 logging 保证)"""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        root = logging.getLogger("flux_vision")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    if name.startswith("flux_vision"):
        return logging.getLogger(name)
    return logging.getLogger(f"flux_vision.{name}")
