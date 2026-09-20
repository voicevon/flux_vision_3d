"""
芦笋感知算法流水线模块初始化
"""

from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry

# 显式导入各技术路线以触发自动注册
import src.vision.pipelines.ridge_tracing_pipeline
import src.vision.pipelines.polarity_scanline_pipeline
import src.vision.pipelines.edge_centerline_pipeline

__all__ = [
    "BaseAsparagusPipeline",
    "PipelineResult",
    "PipelineStep",
    "PipelineRegistry",
]
