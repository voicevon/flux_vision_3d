"""
芦笋识别感知算法流水线统一抽象基类与通用数据契约
======================================================
规范所有研发技术路线 (Pipeline) 的生命周期、中间过程暴露与标准化输出。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PipelineStep:
    """单个算法流水线内部步骤的声明"""
    key: str                    # 内部唯一键，如 "stage1_prep"
    name: str                   # UI 药丸控件标签，如 "1.双边滤波"
    description: str            # 简要简介 (第 1 行)
    details: str = ""           # 深度原理解析 (为什么要有此步，解决什么物理/视觉问题)
    parameters: str = ""        # 核心调节参数与工程建议值 (参数名、默认值、调优方向)
    pros_cons: str = ""         # 优缺点与现场边界条件 (优势 / 局限)


@dataclass
class PipelineResult:
    """算法流水线执行输出的标准统一契约"""
    targets: List[Any]                               # 标准化识别目标列表 (Top 3 AsparagusTarget)
    elapsed_ms: float                                # 单帧全流程执行耗时 (毫秒)
    step_snapshots: Dict[str, Optional[np.ndarray]]  # 各步骤的可视化图像字典 {step_key: bgr_image}
    extra_metrics: Dict[str, Any] = field(default_factory=dict)  # 路线专有研发指标 (如轮廓数、平均置信度)


class BaseAsparagusPipeline(ABC):
    """所有芦笋识别研发算法路线的通用抽象基类"""

    name: str = "Base Pipeline"
    description: str = "通用感知流水线抽象基类"

    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """动态更新内参"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @abstractmethod
    def get_steps(self) -> List[PipelineStep]:
        """
        获取该技术路线支持的全部中间步骤列表
        GUI 会根据此声明动态渲染步骤切换药丸，零侵入适配新路线！
        """
        pass

    @abstractmethod
    def run(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        plane_coeff: Optional[np.ndarray] = None,
        frame_transform: Optional[np.ndarray] = None,
        frame_calib_source: str = "uncalibrated",
        nominal_z_mm: float = 640.0
    ) -> PipelineResult:
        """
        执行完整的端到端识别流水线
        :return: PipelineResult 标准化结果对象
        """
        pass
