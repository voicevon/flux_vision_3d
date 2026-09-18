"""
env_utils.py — 共享环境检测工具
=================================
提供系统关键库、硬件及场景状态的检测函数，供 gui_launcher 与 diagnose_env 诊断脚本共同调用，
避免模块之间产生循环或单向依赖。
"""

import os
import sys
import glob
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from src.calibration.scene_manager import CalibrationSceneManager
except Exception:
    CalibrationSceneManager = None  # type: ignore[assignment,misc]

# 环境检测缓存 (TTL 30 秒，避免反复阻塞枚举 USB 总线)
_ENV_STATUS_CACHE = None
_ENV_CACHE_TIME = 0.0


def check_env_status(force_refresh: bool = False) -> dict:
    """检查系统关键库及硬件、快照信息 (带轻量 TTL 缓存，消除反复枚举 USB 的 1.5s 延迟)"""
    global _ENV_STATUS_CACHE, _ENV_CACHE_TIME
    now = time.time()
    if not force_refresh and _ENV_STATUS_CACHE is not None and (now - _ENV_CACHE_TIME < 30.0):
        return _ENV_STATUS_CACHE

    status: dict = {}

    # RealSense 驱动与物理硬件检测
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if len(devices) > 0:
            dev_name = devices[0].get_info(rs.camera_info.name)
            status['realsense'] = (True, f"已连接 ({dev_name})", True)
        else:
            status['realsense'] = (False, "驱动已装，但未检测到物理相机(USB未连接)", False)
    except ImportError:
        status['realsense'] = (False, "未安装驱动库 pyrealsense2", False)
    except Exception as e:
        status['realsense'] = (False, f"相机状态异常: {e}", False)

    # OpenCV
    try:
        import cv2
        status['opencv'] = (True, f"v{cv2.__version__}")
    except ImportError:
        status['opencv'] = (False, "未安装")

    # NumPy
    try:
        import numpy as np
        status['numpy'] = (True, f"v{np.__version__}")
    except ImportError:
        status['numpy'] = (False, "未安装")

    # 快照统计
    snapshots = glob.glob(os.path.join(PROJECT_ROOT, "data", "snapshots", "color_*.png"))
    status['snapshot_count'] = len(snapshots)

    # 标定场景管理器与当前工况场景
    scene_mgr = CalibrationSceneManager() if CalibrationSceneManager else None
    current_scene = scene_mgr.get_current_scene() if scene_mgr else None
    status['scene_mgr'] = scene_mgr
    status['current_scene'] = current_scene
    status['active_scene'] = current_scene  # 兼容旧代码键

    # 采图数据集统计 (以当前工况场景为主，兼容旧路径)
    if current_scene:
        status['calib_image_count'] = current_scene.image_count
        status['has_tag_map'] = current_scene.ba_solved
        status['has_manifest'] = os.path.exists(current_scene.manifest_path)
        status['manifest_path'] = current_scene.manifest_path
        status['image_dir'] = current_scene.raw_images_dir
        status['map_path'] = current_scene.map_path
        status['is_published'] = current_scene.is_published
    else:
        calib_images = glob.glob(os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "*.png"))
        status['calib_image_count'] = len(calib_images)
        map_path = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
        status['has_tag_map'] = os.path.exists(map_path)
        manifest_path = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images", "tag_observations.yaml")
        status['has_manifest'] = os.path.exists(manifest_path)
        status['manifest_path'] = manifest_path
        status['image_dir'] = os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")
        status['map_path'] = map_path
        status['is_published'] = os.path.exists(map_path)

    # 生产环境全局地图状态
    prod_map_path = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
    status['prod_has_map'] = os.path.exists(prod_map_path)

    manifest_excluded = 0
    if status['has_manifest']:
        try:
            import yaml
            with open(status['manifest_path'], "r", encoding="utf-8") as f:
                m = yaml.safe_load(f) or {}
            frames = m.get("images", m.get("frames", {}))
            for img in frames.values():
                if isinstance(img, dict) and (img.get("excluded", False) or not img.get("enabled", True)):
                    manifest_excluded += 1
                elif isinstance(img, dict):
                    for obs in img.get("observations", []):
                        if not obs.get("keep", True):
                            manifest_excluded += 1
                            break
        except Exception as e:
            log.warning(f"读取标定清单统计排除帧失败: {e}")
    status['manifest_excluded'] = manifest_excluded

    # 标靶 ID 白名单状态
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    valid_tag_ids = []
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            valid_tag_ids = cfg.get("calibration", {}).get("valid_tag_ids", [])
        except Exception as e:
            log.warning(f"读取 config.yaml 标靶白名单失败: {e}")
    status['valid_tag_ids'] = valid_tag_ids

    _ENV_STATUS_CACHE = status
    _ENV_CACHE_TIME = time.time()
    return status
