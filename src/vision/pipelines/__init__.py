"""
芦笋感知算法流水线模块初始化
"""

from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry

# 显式导入各算法流水线以触发自动注册
import src.vision.pipelines.ridge_tracing_pipeline
import src.vision.pipelines.polarity_scanline_pipeline
import src.vision.pipelines.edge_centerline_pipeline
import src.vision.pipelines.skeleton_thinning_pipeline
import src.vision.pipelines.frangi_vesselness_pipeline
import src.vision.pipelines.stub_pipelines

# H 系列：非机器学习经典算法流水线
import src.vision.pipelines.h1_sato_pipeline
import src.vision.pipelines.h2_skeleton_pipeline
import src.vision.pipelines.h3_fast_marching_pipeline
import src.vision.pipelines.h4_graph_tracing_pipeline
import src.vision.pipelines.h5_phase_congruency_pipeline
import src.vision.pipelines.h6_steerable_pipeline

__all__ = [
    "BaseAsparagusPipeline",
    "PipelineResult",
    "PipelineStep",
    "PipelineRegistry",
]
