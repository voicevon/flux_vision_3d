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
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import yaml

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
    def load(cls, scene_dir: str) -> Optional["CalibrationScene"]:
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
            is_published=is_published
        )
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

        print(f"[SCENE] 检测到历史采图数据 ({len(legacy_images)} 帧)，执行平滑自动迁移...")
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
        print(f"[SCENE] 成功构建默认沙盒场景: {default_scene_id}")

    def list_scenes(self) -> List[CalibrationScene]:
        """枚举所有有效场景，按创建时间降序排序"""
        scenes = []
        if not os.path.isdir(self.scenes_dir):
            return []
        
        active_id = self.get_active_scene_id()
        for item in os.listdir(self.scenes_dir):
            if item.startswith("."):
                continue
            item_path = os.path.join(self.scenes_dir, item)
            if os.path.isdir(item_path):
                scene = CalibrationScene.load(item_path)
                if scene:
                    scenes.append(scene)

        # 排序：创建时间降序
        scenes.sort(key=lambda s: s.created_at, reverse=True)
        return scenes

    def get_active_scene_id(self) -> str:
        """获取当前激活场景的 ID"""
        if os.path.exists(self.active_marker_file):
            try:
                with open(self.active_marker_file, "r", encoding="utf-8") as f:
                    sid = f.read().strip()
                if sid and os.path.isdir(os.path.join(self.scenes_dir, sid)):
                    return sid
            except Exception:
                pass
        
        # 回退逻辑：如果标记文件损坏或指向空，选首个场景或创建默认场景
        existing = [d for d in os.listdir(self.scenes_dir) if not d.startswith(".") and os.path.isdir(os.path.join(self.scenes_dir, d))]
        if existing:
            return sorted(existing)[-1]
        return ""

    def get_active_scene(self) -> CalibrationScene:
        """获取当前激活场景对象，若无场景则自愈创建默认场景"""
        active_id = self.get_active_scene_id()
        if active_id:
            scene = CalibrationScene.load(os.path.join(self.scenes_dir, active_id))
            if scene:
                return scene

        # 若没有任何场景，初始化一个标准默认场景
        default_alias = "bench_default"
        return self.create_scene(alias=default_alias, description="默认工位标定场景")

    def set_active_scene(self, scene_id: str) -> bool:
        """切换当前活动场景"""
        target_dir = os.path.join(self.scenes_dir, scene_id)
        if not os.path.isdir(target_dir):
            return False
        
        try:
            with open(self.active_marker_file, "w", encoding="utf-8") as f:
                f.write(scene_id.strip())
            return True
        except Exception as e:
            print(f"[SCENE] 切换活动场景失败: {e}")
            return False

    def create_scene(self, alias: str, description: str = "") -> CalibrationScene:
        """根据操作员自定义别名创建新场景 (自动添加 YYYYMMDD 前缀)"""
        clean_alias = "".join(c for c in alias if c.isalnum() or c in ("_", "-")).strip("_")
        if not clean_alias:
            clean_alias = "scene"

        date_prefix = time.strftime("%Y%m%d")
        scene_id = f"{date_prefix}_{clean_alias}"
        
        # 避免同名覆盖，自增序列
        counter = 1
        original_id = scene_id
        while os.path.exists(os.path.join(self.scenes_dir, scene_id)):
            scene_id = f"{original_id}_{counter}"
            counter += 1

        scene_dir = os.path.join(self.scenes_dir, scene_id)
        scene = CalibrationScene(
            scene_id=scene_id,
            name=clean_alias,
            scene_dir=scene_dir,
            description=description,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        scene.ensure_directories()
        scene.save_meta()
        
        # 自动设为当前活动场景
        self.set_active_scene(scene_id)
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
        return new_scene

    def publish_to_production(self, scene_id: Optional[str] = None) -> Tuple[bool, str]:
        """将指定场景的 tags_map.yaml 安全原子发布覆盖至 config/tags_map.yaml 并记录至 config.yaml"""
        target_id = scene_id or self.get_active_scene_id()
        if not target_id:
            return False, "无可发布的活动场景"

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

            # 3. 更新 config.yaml 中的 active_scene 字段
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if "calibration" not in cfg:
                    cfg["calibration"] = {}
                cfg["calibration"]["active_scene"] = target_id
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            # 4. 更新场景的元数据标志
            scene.is_published = True
            scene.save_meta()
            return True, f"成功将场景 [{target_id}] 发布为全局生产运行地图 (RMSE: {scene.global_rmse_px:.3f}px)"
        except Exception as e:
            return False, f"发布至生产环境发生异常: {e}"

    def delete_scene(self, scene_id: str) -> Tuple[bool, str]:
        """安全物理删除指定场景 (禁止删除当前正在激活的场景)"""
        active_id = self.get_active_scene_id()
        if scene_id == active_id:
            return False, "禁止删除当前正在激活的活动场景！请先切换至其他场景。"

        scene_dir = os.path.join(self.scenes_dir, scene_id)
        if not os.path.isdir(scene_dir):
            return False, f"场景目录不存在: {scene_id}"

        try:
            shutil.rmtree(scene_dir)
            return True, f"已成功删除场景: {scene_id}"
        except Exception as e:
            return False, f"删除场景发生异常: {e}"
