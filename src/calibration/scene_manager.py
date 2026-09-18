"""
AprilTag 标定采样场景与批次分组管理核心模块 (CalibrationSceneManager)
================================================================================
负责标定工程中各场景数据包的自包含生命周期管理：
1. 物理沙盒隔离：每个场景拥有自包含的 raw_images、tag_observations.yaml、tags_map.yaml、reports
2. 命名与检索：YYYYMMDD_<alias> 规范化目录生成与场景元数据 (scene_meta.yaml) 维护
3. 状态与追溯：记录采图数、平差状态、RMSE 指标与发布状态
4. 生产环境发布：将经过平差验证的场景地图安全原子同步至 config/tags_map.yaml 与 config.yaml
5. 历史向前兼容：首次启动时自动无损迁移旧版 data/tag_calibration_images/
"""

import os
import shutil
import time
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)

# 项目根目录常量
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SCENES_DIR = os.path.join(PROJECT_ROOT, "data", "calibration_scenes")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_PROD_MAP_PATH = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")


@dataclass
class CalibrationScene:
    """单个标定采样场景自包含数据包"""
    scene_id: str                          # 场景唯一标识 (目录名，如 20260915_bench_a)
    name: str                              # 自定义别名 (如 bench_a)
    scene_dir: str                         # 场景绝对物理根目录
    description: str = ""                  # 场景说明备注
    created_at: str = ""                   # 创建时间
    updated_at: str = ""                   # 最后更新时间
    camera_serial: str = ""                # 采集相机硬件序列号
    valid_tag_ids: List[int] = field(default_factory=list)
    origin_tag_id: int = 0
    x_axis_tag_id: int = 28
    
    # 状态指标
    image_count: int = 0
    active_image_count: int = 0
    ba_solved: bool = False
    global_rmse_px: float = 0.0
    is_published: bool = False

    @property
    def raw_images_dir(self) -> str:
        """原始采图存储目录"""
        return os.path.join(self.scene_dir, "raw_images")

    @property
    def manifest_path(self) -> str:
        """审核清单 (tag_observations.yaml) 路径"""
        return os.path.join(self.scene_dir, "tag_observations.yaml")

    @property
    def map_path(self) -> str:
        """本场景专属空间立体几何地图路径"""
        return os.path.join(self.scene_dir, "tags_map.yaml")

    @property
    def backup_map_path(self) -> str:
        """本场景地图备份路径"""
        return os.path.join(self.scene_dir, "tags_map.yaml.bak")

    @property
    def visualized_dir(self) -> str:
        """图示化分析与残差场标注目录"""
        return os.path.join(self.scene_dir, "visualized")

    @property
    def reports_dir(self) -> str:
        """质检单与盲测体检报告目录"""
        return os.path.join(self.scene_dir, "reports")

    @property
    def meta_path(self) -> str:
        """场景自描述元数据路径"""
        return os.path.join(self.scene_dir, "scene_meta.yaml")

    def ensure_directories(self):
        """确保场景所需的所有子目录结构完整存在"""
        for d in [self.scene_dir, self.raw_images_dir, self.visualized_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)

    def refresh_stats(self):
        """自动扫描磁盘刷新该场景的统计指标"""
        self.ensure_directories()
        
        # 扫描原始图片数
        pngs = glob.glob(os.path.join(self.raw_images_dir, "*.png"))
        self.image_count = len(pngs)

        # 扫描审核清单
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = yaml.safe_load(f) or {}
                frames = manifest.get("images", manifest.get("frames", {}))
                excluded_cnt = sum(1 for v in frames.values() if isinstance(v, dict) and (v.get("excluded", False) or not v.get("enabled", True)))
                self.active_image_count = max(0, len(frames) - excluded_cnt)
            except Exception:
                self.active_image_count = self.image_count
        else:
            self.active_image_count = self.image_count

        # 扫描地图状态
        if os.path.exists(self.map_path) and os.path.getsize(self.map_path) > 50:
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    m = yaml.safe_load(f) or {}
                self.ba_solved = bool(m.get("tags"))
                self.global_rmse_px = float(m.get("rmse_reprojection_px", m.get("rmse_px", 0.0)))
            except Exception:
                self.ba_solved = False
        else:
            self.ba_solved = False
            self.global_rmse_px = 0.0

    def save_meta(self):
        """将场景元数据持久化落盘至 scene_meta.yaml"""
        self.ensure_directories()
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "scene_id": self.scene_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "camera_serial": self.camera_serial,
            "valid_tag_ids": self.valid_tag_ids,
            "origin_tag_id": self.origin_tag_id,
            "x_axis_tag_id": self.x_axis_tag_id,
            "status": {
                "image_count": self.image_count,
                "active_image_count": self.active_image_count,
                "ba_solved": self.ba_solved,
                "global_rmse_px": round(self.global_rmse_px, 4),
                "is_published": self.is_published
            }
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, scene_dir: str, force_refresh: bool = False) -> Optional["CalibrationScene"]:
        """从已有场景物理目录构建加载 CalibrationScene 对象"""
        if not os.path.isdir(scene_dir):
            return None
        scene_id = os.path.basename(os.path.normpath(scene_dir))
        meta_path = os.path.join(scene_dir, "scene_meta.yaml")
        
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
            except Exception:
                meta = {}

        name = meta.get("name") or (scene_id.split("_", 1)[1] if "_" in scene_id else scene_id)
        desc = meta.get("description", "")
        created_at = meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        updated_at = meta.get("updated_at", created_at)
        camera_serial = meta.get("camera_serial", "")
        valid_tag_ids = meta.get("valid_tag_ids", [])
        origin_tag_id = int(meta.get("origin_tag_id", 0))
        x_axis_tag_id = int(meta.get("x_axis_tag_id", 28))

        status = meta.get("status", {})
        is_published = bool(status.get("is_published", False))

        image_count = int(status.get("image_count", 0))
        active_image_count = int(status.get("active_image_count", 0))
        ba_solved = bool(status.get("ba_solved", False))
        global_rmse_px = float(status.get("global_rmse_px", 0.0))

        scene = cls(
            scene_id=scene_id,
            name=name,
            scene_dir=scene_dir,
            description=desc,
            created_at=created_at,
            updated_at=updated_at,
            camera_serial=camera_serial,
            valid_tag_ids=valid_tag_ids,
            origin_tag_id=origin_tag_id,
            x_axis_tag_id=x_axis_tag_id,
            image_count=image_count,
            active_image_count=active_image_count,
            ba_solved=ba_solved,
            global_rmse_px=global_rmse_px,
            is_published=is_published
        )
        if force_refresh or not status:
            scene.refresh_stats()
        return scene


class CalibrationSceneManager:
    """标定采样场景总库管理器"""

    def __init__(
        self,
        scenes_dir: str = DEFAULT_SCENES_DIR,
        config_path: str = DEFAULT_CONFIG_PATH,
        prod_map_path: str = DEFAULT_PROD_MAP_PATH,
        legacy_dir: Optional[str] = None
    ):
        self.scenes_dir = os.path.abspath(scenes_dir)
        self.config_path = os.path.abspath(config_path)
        self.prod_map_path = os.path.abspath(prod_map_path)
        self.legacy_dir = os.path.abspath(legacy_dir) if legacy_dir else os.path.join(PROJECT_ROOT, "data", "tag_calibration_images")
        self.active_marker_file = os.path.join(self.scenes_dir, ".active_scene")
        self._cached_active_scene: Optional[CalibrationScene] = None
        self._cached_scenes: Dict[str, CalibrationScene] = {}
        
        # 确保根目录存在并执行历史数据向前兼容自愈
        os.makedirs(self.scenes_dir, exist_ok=True)
        self.auto_migrate_legacy_data()

    def auto_migrate_legacy_data(self):
        """检测并无损迁移旧 data/tag_calibration_images/ 历史数据"""
        if not self.legacy_dir or not os.path.exists(self.legacy_dir):
            return

        # 检查是否已经存在迁移或新建场景
        existing_scenes = self.list_scenes()
        if existing_scenes:
            return

        legacy_images = glob.glob(os.path.join(self.legacy_dir, "*.png"))
        legacy_manifest = os.path.join(self.legacy_dir, "tag_observations.yaml")
        if not legacy_images and not os.path.exists(legacy_manifest):
            return

        log.info(f"[SCENE] 检测到历史采图数据 ({len(legacy_images)} 帧)，执行平滑自动迁移...")
        default_scene_id = f"{time.strftime('%Y%m%d')}_bench_default"
        default_scene_dir = os.path.join(self.scenes_dir, default_scene_id)
        scene = CalibrationScene(
            scene_id=default_scene_id,
            name="bench_default",
            scene_dir=default_scene_dir,
            description="历史采图数据自动迁移默认场景",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        scene.ensure_directories()

        # 拷贝原始图像
        for img in legacy_images:
            shutil.copy2(img, os.path.join(scene.raw_images_dir, os.path.basename(img)))

        # 拷贝审核清单
        if os.path.exists(legacy_manifest):
            shutil.copy2(legacy_manifest, scene.manifest_path)

        # 拷贝图示化目录
        legacy_vis = os.path.join(self.legacy_dir, "visualized")
        if os.path.exists(legacy_vis):
            for vis_f in glob.glob(os.path.join(legacy_vis, "*.*")):
                shutil.copy2(vis_f, os.path.join(scene.visualized_dir, os.path.basename(vis_f)))

        # 拷贝生产地图为本场景初始地图
        if os.path.exists(self.prod_map_path):
            shutil.copy2(self.prod_map_path, scene.map_path)
            scene.is_published = True

        scene.refresh_stats()
        scene.save_meta()
        self.set_active_scene(default_scene_id)
        log.info(f"[SCENE] 成功构建默认沙盒场景: {default_scene_id}")

    def _migrate_legacy_non_ascii_dir(self, item: str) -> Optional[str]:
        """将物理路径中包含非 ASCII/中文的历史遗留场景目录，安全原子重命名为纯 ASCII 目录"""
        old_path = os.path.join(self.scenes_dir, item)
        if not os.path.isdir(old_path):
            return None

        # 尝试读取内部元数据或保留友好中文名
        display_name = item
        meta_file = os.path.join(old_path, "scene_meta.yaml")
        meta = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                if meta.get("name"):
                    display_name = meta["name"]
            except Exception:
                pass

        # 提取已有 ASCII 前缀 (如 20260915)
        ascii_parts = "".join(c for c in item if c.isascii() and (c.isalnum() or c in ("_", "-"))).strip("_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if ascii_parts:
            new_id = f"{ascii_parts}_migrated"
        else:
            new_id = f"{timestamp}_scene"

        counter = 1
        base_id = new_id
        while os.path.exists(os.path.join(self.scenes_dir, new_id)):
            new_id = f"{base_id}_{counter}"
            counter += 1

        new_path = os.path.join(self.scenes_dir, new_id)
        try:
            os.rename(old_path, new_path)
            # 更新内部元数据中的 scene_id 与友好名称
            meta["scene_id"] = new_id
            meta["name"] = display_name
            with open(os.path.join(new_path, "scene_meta.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(meta, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            # 若 .active_scene 指向旧目录，同步更新
            if os.path.exists(self.active_marker_file):
                try:
                    with open(self.active_marker_file, "r", encoding="utf-8") as f:
                        cur_active = f.read().strip()
                    if cur_active == item:
                        with open(self.active_marker_file, "w", encoding="utf-8") as f:
                            f.write(new_id)
                except Exception:
                    pass

            log.info(f"[SCENE] 成功将历史非 ASCII 目录【{item}】安全迁移为【{new_id}】(友好名称仍为: {display_name})")
            return new_id
        except Exception as e:
            log.warning(f"[WARN] 迁移历史目录【{item}】失败: {e}")
            return None

    def list_scenes(self) -> List[CalibrationScene]:
        """枚举所有有效场景，按创建时间降序排序"""
        scenes = []
        if not os.path.isdir(self.scenes_dir):
            return []
        
        for item in os.listdir(self.scenes_dir):
            if item.startswith("."):
                continue
            item_path = os.path.join(self.scenes_dir, item)
            if os.path.isdir(item_path):
                # 检查是否存在非 ASCII 字符 (如遗留中文文件夹)，自动安全迁移
                if any(ord(c) > 127 for c in item):
                    migrated_id = self._migrate_legacy_non_ascii_dir(item)
                    if migrated_id:
                        item = migrated_id
                        item_path = os.path.join(self.scenes_dir, item)

                try:
                    scene = CalibrationScene.load(item_path)
                    if scene:
                        scenes.append(scene)
                except Exception as e:
                    log.warning(f"[WARN] 加载场景异常 {item}: {e}")

        # 排序：创建时间降序，次要以 scene_id 降序
        scenes.sort(key=lambda s: (s.created_at, s.scene_id), reverse=True)
        return scenes

    def get_production_scene_id(self) -> str:
        """获取当前发布为生产运行的场景 ID (优先读 config.yaml 生产记录，次选 is_published 标记)"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                calib = cfg.get("calibration", {})
                prod_id = calib.get("prod_scene_id") or calib.get("active_scene", "")
                if prod_id and os.path.isdir(os.path.join(self.scenes_dir, prod_id)):
                    return prod_id
            except Exception:
                pass

        for sc in self.list_scenes():
            if sc.is_published:
                return sc.scene_id
        return ""

    def get_production_scene(self) -> Optional[CalibrationScene]:
        """获取当前发布为生产运行的场景对象"""
        pid = self.get_production_scene_id()
        return self.get_scene_by_id(pid) if pid else None

    def get_scene_by_id(self, scene_id: str, force_refresh: bool = False) -> Optional[CalibrationScene]:
        """根据场景 ID 检索场景对象 (带缓存支持)"""
        if not scene_id:
            return None
        target_dir = os.path.join(self.scenes_dir, scene_id)
        if not os.path.isdir(target_dir):
            return None
        if not force_refresh and scene_id in self._cached_scenes:
            return self._cached_scenes[scene_id]
        scene = CalibrationScene.load(target_dir, force_refresh=force_refresh)
        if scene:
            self._cached_scenes[scene_id] = scene
        return scene

    def get_current_scene_id(self) -> str:
        """获取当前默认工况场景 ID (优先生产地图对应场景，兜底最新场景)"""
        prod_id = self.get_production_scene_id()
        if prod_id:
            return prod_id

        if os.path.exists(self.active_marker_file):
            try:
                with open(self.active_marker_file, "r", encoding="utf-8") as f:
                    sid = f.read().strip()
                if sid and os.path.isdir(os.path.join(self.scenes_dir, sid)):
                    return sid
            except Exception:
                pass
        
        scenes = self.list_scenes()
        if scenes:
            return scenes[0].scene_id
        return ""

    def get_active_scene_id(self) -> str:
        """获取默认场景 ID (兼容向后调用别名)"""
        return self.get_current_scene_id()

    def get_current_scene(self, force_refresh: bool = False) -> CalibrationScene:
        """获取当前默认工况场景对象 (优先生产地图对应场景，次选最新场景，无场景则自动初始化)"""
        cur_id = self.get_current_scene_id()
        if cur_id:
            scene = self.get_scene_by_id(cur_id, force_refresh=force_refresh)
            if scene:
                return scene

        new_scene = self.create_scene(alias="默认工位", description="系统自动初始化默认工况场景")
        return new_scene

    def get_active_scene(self, force_refresh: bool = False) -> CalibrationScene:
        """获取场景对象 (兼容向后调用别名)"""
        return self.get_current_scene(force_refresh=force_refresh)

    def set_active_scene(self, scene_id: str) -> bool:
        """切换默认场景 (向前兼容保留接口)"""
        target_dir = os.path.join(self.scenes_dir, scene_id)
        if not os.path.isdir(target_dir):
            return False
        try:
            with open(self.active_marker_file, "w", encoding="utf-8") as f:
                f.write(scene_id.strip())
            self._cached_scenes[scene_id] = CalibrationScene.load(target_dir)
            return True
        except Exception as e:
            log.warning(f"[SCENE] 写入默认场景标记失败: {e}")
            return False

    def invalidate_cache(self):
        """显式使场景列表与对象缓存失效"""
        self._cached_active_scene = None
        self._cached_scenes.clear()

    def create_scene(self, alias: str, description: str = "") -> CalibrationScene:
        """根据操作员自定义别名创建新场景 (安全 ASCII 时间戳目录 + 完整友好中文别名)"""
        display_name = alias.strip() if alias and alias.strip() else "新建工况"
        
        # 物理 scene_id 目录名强制使用纯 ASCII 安全时间戳 + 序列，杜绝底层编码隐患
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ascii_suffix = "".join(c for c in alias if c.isascii() and (c.isalnum() or c in ("_", "-"))).strip("_")
        if ascii_suffix:
            scene_id = f"{timestamp}_{ascii_suffix}"
        else:
            scene_id = f"{timestamp}_scene"

        counter = 1
        original_id = scene_id
        while os.path.exists(os.path.join(self.scenes_dir, scene_id)):
            scene_id = f"{original_id}_{counter}"
            counter += 1

        scene_dir = os.path.join(self.scenes_dir, scene_id)
        scene = CalibrationScene(
            scene_id=scene_id,
            name=display_name,
            scene_dir=scene_dir,
            description=description,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        scene.ensure_directories()
        scene.save_meta()
        self._cached_scenes[scene_id] = scene
        try:
            with open(self.active_marker_file, "w", encoding="utf-8") as f:
                f.write(scene_id.strip())
        except Exception:
            pass
        return scene

    def clone_scene(self, src_scene_id: str, new_alias: str, description: str = "") -> Optional[CalibrationScene]:
        """克隆已有场景所有数据至新场景 (便于开展平差参数对比实验)"""
        src_dir = os.path.join(self.scenes_dir, src_scene_id)
        if not os.path.isdir(src_dir):
            return None

        src_scene = CalibrationScene.load(src_dir)
        if not src_scene:
            return None

        new_scene = self.create_scene(alias=new_alias, description=description or f"克隆自 {src_scene_id}")
        
        # 拷贝原始图像
        if os.path.exists(src_scene.raw_images_dir):
            for f in glob.glob(os.path.join(src_scene.raw_images_dir, "*.png")):
                shutil.copy2(f, os.path.join(new_scene.raw_images_dir, os.path.basename(f)))

        # 拷贝审核清单
        if os.path.exists(src_scene.manifest_path):
            shutil.copy2(src_scene.manifest_path, new_scene.manifest_path)

        # 拷贝地图
        if os.path.exists(src_scene.map_path):
            shutil.copy2(src_scene.map_path, new_scene.map_path)

        new_scene.refresh_stats()
        new_scene.save_meta()
        self._cached_scenes[new_scene.scene_id] = new_scene
        return new_scene

    def rename_scene(self, scene_id: str, new_name: str, new_description: Optional[str] = None) -> bool:
        """修改场景友好显示别名 (支持中文、英文、数字)，不破坏底层物理目录与历史引用"""
        clean_name = new_name.strip()
        if not clean_name:
            return False

        target_dir = os.path.join(self.scenes_dir, scene_id)
        scene = CalibrationScene.load(target_dir)
        if not scene:
            return False

        scene.name = clean_name
        if new_description is not None:
            scene.description = new_description
        scene.save_meta()
        self._cached_scenes[scene_id] = scene
        return True

    def publish_to_production(self, scene_id: Optional[str] = None) -> Tuple[bool, str]:
        """将指定场景的 tags_map.yaml 安全原子发布覆盖至 config/tags_map.yaml 并记录至 config.yaml"""
        target_id = scene_id or self.get_active_scene_id()
        if not target_id:
            return False, "无可发布的场景"

        scene_dir = os.path.join(self.scenes_dir, target_id)
        scene = CalibrationScene.load(scene_dir)
        if not scene:
            return False, f"场景不存在: {target_id}"

        if not os.path.exists(scene.map_path) or os.path.getsize(scene.map_path) < 50:
            return False, f"该场景尚未平差生成有效地图 (tags_map.yaml 缺失或为空)"

        try:
            # 1. 安全备份生产旧地图
            os.makedirs(os.path.dirname(self.prod_map_path), exist_ok=True)
            if os.path.exists(self.prod_map_path):
                shutil.copy2(self.prod_map_path, f"{self.prod_map_path}.bak")

            # 2. 原子拷贝
            shutil.copy2(scene.map_path, self.prod_map_path)

            # 3. 更新 config.yaml 中的 prod_scene_id 与 active_scene 字段
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if "calibration" not in cfg:
                    cfg["calibration"] = {}
                cfg["calibration"]["prod_scene_id"] = target_id
                cfg["calibration"]["active_scene"] = target_id  # 保持旧逻辑兼容
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            # 4. 同步各场景的发布状态 (唯有被发布场景 is_published=True)
            for other_sc in self.list_scenes():
                if other_sc.scene_id == target_id:
                    other_sc.is_published = True
                    other_sc.save_meta()
                elif other_sc.is_published:
                    other_sc.is_published = False
                    other_sc.save_meta()

            self.invalidate_cache()
            return True, f"成功将场景 [{target_id}] 发布为全局生产运行地图 (RMSE: {scene.global_rmse_px:.3f}px)"
        except Exception as e:
            return False, f"发布至生产环境发生异常: {e}"

    def delete_scene(self, scene_id: str) -> Tuple[bool, str]:
        """安全物理删除指定场景 (允许删除任意非空场景)"""
        scene_dir = os.path.join(self.scenes_dir, scene_id)
        if not os.path.isdir(scene_dir):
            return False, f"场景目录不存在: {scene_id}"

        try:
            shutil.rmtree(scene_dir)
            if scene_id in self._cached_scenes:
                del self._cached_scenes[scene_id]
            self.invalidate_cache()
            return True, f"已成功删除场景: {scene_id}"
        except Exception as e:
            return False, f"删除场景发生异常: {e}"
