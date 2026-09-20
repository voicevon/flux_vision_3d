"""
芦笋感知算法研发路线注册中心 (Pipeline Registry)
======================================================
负责管理、发现与实例化所有可用的感知算法技术路线，支持运行时无缝插拔。
"""

from typing import Dict, List, Optional, Tuple, Type
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline


class PipelineRegistry:
    """感知算法路线注册表 (单例模式)"""

    _pipelines: Dict[str, Tuple[str, Type[BaseAsparagusPipeline]]] = {}

    @classmethod
    def register(cls, key: str, display_name: str):
        """装饰器：注册一个新的算法技术路线"""
        def decorator(subclass: Type[BaseAsparagusPipeline]):
            cls._pipelines[key] = (display_name, subclass)
            return subclass
        return decorator

    @classmethod
    def list_options(cls) -> List[Tuple[str, str]]:
        """列出所有已注册的算法路线选项 [(key, display_name), ...]"""
        return [(k, name) for k, (name, _) in cls._pipelines.items()]

    @classmethod
    def get_pipeline_class(cls, key: str) -> Optional[Type[BaseAsparagusPipeline]]:
        """根据 key 获取对应的算法类"""
        if key in cls._pipelines:
            return cls._pipelines[key][1]
        return None

    @classmethod
    def create(cls, key: str, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51) -> Optional[BaseAsparagusPipeline]:
        """实例化指定的算法路线"""
        klass = cls.get_pipeline_class(key)
        if klass is not None:
            return klass(fx=fx, fy=fy, cx=cx, cy=cy)
        return None
