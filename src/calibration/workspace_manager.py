"""
工位工作空间 (Workspace) 与批次沙盒管理核心模块
=================================================
负责工业工位 (Workspace) 全局数据生命周期管理：
1. 物理沙盒隔离：每个工位自包含顶层资产 (tags_map.yaml, tag_whitelist.yaml, workspace_meta.yaml)
2. 双业务分支：
   - calibration/ (标定专区: raw_images/, tag_observations.yaml, reports/, visualized/)
   - production/  (生产与模拟生产专区: raw_images/, reports/, results/)
3. 命名与检索：YYYYMMDD_<alias> 规范化目录生成与元数据持久化
4. 生产环境发布：将经过平差验证的工位顶层地图安全原子发布至 config/tags_map.yaml 与 config.yaml
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
DEFAULT_WORKSPACES_DIR = os.path.join(PROJECT_ROOT, "data", "workspaces")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_PROD_MAP_PATH = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")


@dataclass
class Workspace:
    """单个工位工作空间自包含数据模型"""
    workspace_id: str                      # 工位唯一标识 (目录名，如 20260915_bench_default)
    name: str                              # 友好别名 (如 bench_default, 现场工位A)
    workspace_dir: str                     # 工位物理根目录绝对路径
    description: str = ""                  # 工位说明备注
    created_at: str = ""                   # 创建时间
    updated_at: str = ""                   # 最后更新时间
    camera_serial: str = ""                # 采集相机硬件序列号
    valid_tag_ids: List[int] = field(default_factory=list)
    origin_tag_id: int = 0
    x_axis_tag_id: int = 28

    # 状态与指标
    image_count: int = 0                   # 标定图片数
    prod_image_count: int = 0              # 生产采图数
    active_image_count: int = 0            # 标定有效帧数
    ba_solved: bool = False
    global_rmse_px: float = 0.0
    is_published: bool = False

    # ------------------------------ 顶层核心资产路径 ------------------------------
    @property
    def map_path(self) -> str:
        """【工位核心资产】空间立体几何地图路径 (工位顶层，供生产/标定消费)"""
        return os.path.join(self.workspace_dir, "tags_map.yaml")

    @property
    def backup_map_path(self) -> str:
        """工位立体几何地图备份路径"""
        return os.path.join(self.workspace_dir, "tags_map.yaml.bak")

    @property
    def whitelist_path(self) -> str:
        """【工位核心资产】本工位合法 AprilTag 白名单文件"""
        return os.path.join(self.workspace_dir, "tag_whitelist.yaml")

    @property
    def meta_path(self) -> str:
        """工位自描述元数据路径"""
        return os.path.join(self.workspace_dir, "workspace_meta.yaml")

    # ------------------------------ 标定业务专区 (calibration/) ------------------------------
    @property
    def calibration_dir(self) -> str:
        """标定业务专区根目录"""
        return os.path.join(self.workspace_dir, "calibration")

    @property
    def calib_raw_images_dir(self) -> str:
        """标定采集原始图像存储目录"""
        return os.path.join(self.calibration_dir, "raw_images")

    @property
    def calib_manifest_path(self) -> str:
        """标定观测清单 (tag_observations.yaml) 路径"""
        return os.path.join(self.calibration_dir, "tag_observations.yaml")

    @property
    def calib_reports_dir(self) -> str:
        """标定质检单与盲测体检报告目录"""
        return os.path.join(self.calibration_dir, "reports")

    @property
    def calib_visualized_dir(self) -> str:
        """标定残差场与特征图示化标注目录"""
        return os.path.join(self.calibration_dir, "visualized")

    # ------------------------------ 生产业务专区 (production/) ------------------------------
    @property
    def production_dir(self) -> str:
        """生产与模拟生产业务专区根目录"""
        return os.path.join(self.workspace_dir, "production")

    @property
    def prod_raw_images_dir(self) -> str:
        """生产与模拟生产采样图像存储目录"""
        return os.path.join(self.production_dir, "raw_images")

    @property
    def prod_reports_dir(self) -> str:
        """生产评测报告与离线质检结果目录"""
        return os.path.join(self.production_dir, "reports")

    @property
    def prod_results_dir(self) -> str:
        """生产结果与产物导出目录"""
        return os.path.join(self.production_dir, "results")

    # ------------------------------ 业务方法 ------------------------------
    def get_raw_images_dir(self, purpose: str = "calibration") -> str:
        """按用途返回采图目录: 'calibration' -> calib_raw_images_dir, 'production' -> prod_raw_images_dir"""
        if purpose == "production":
            return self.prod_raw_images_dir
        return self.calib_raw_images_dir

    def get_image_count(self, purpose: str = "calibration") -> int:
        """根据用途返回当前磁盘物理图片数量"""
        folder = self.get_raw_images_dir(purpose)
        if not os.path.isdir(folder):
            return 0
        calib_exts = ("*.png", "*.jpg", "*.jpeg")
        count = 0
        for ext in calib_exts:
            count += len(glob.glob(os.path.join(folder, ext)))
        return count

    def ensure_directories(self):
        """确保工位完整的顶层及双分支子目录就绪"""
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.calib_raw_images_dir, exist_ok=True)
        os.makedirs(self.calib_reports_dir, exist_ok=True)
        os.makedirs(self.calib_visualized_dir, exist_ok=True)
        os.makedirs(self.prod_raw_images_dir, exist_ok=True)
        os.makedirs(self.prod_reports_dir, exist_ok=True)
        os.makedirs(self.prod_results_dir, exist_ok=True)

    def refresh_stats(self):
        """快速刷新物理磁盘状态并持久化至元数据缓存"""
        self.ensure_directories()

        # 1. 标定原始图片数
        calib_exts = ("*.png", "*.jpg", "*.jpeg")
        calib_files = []
        for ext in calib_exts:
            calib_files.extend(glob.glob(os.path.join(self.calib_raw_images_dir, ext)))
        self.image_count = len(calib_files)

        # 2. 生产采样图片数
        prod_files = []
        for ext in calib_exts:
            prod_files.extend(glob.glob(os.path.join(self.prod_raw_images_dir, ext)))
        self.prod_image_count = len(prod_files)

        # 3. 标定观测清单解析
        if os.path.exists(self.calib_manifest_path):
            try:
                with open(self.calib_manifest_path, "r", encoding="utf-8") as f:
                    obs_data = yaml.safe_load(f) or {}
                images_dict = obs_data.get("images", {})
                self.active_image_count = len([img for img in images_dict.values() if img.get("tags")])
                seen_tags = set()
                for img in images_dict.values():
                    for t in img.get("tags", []):
                        seen_tags.add(t.get("tag_id"))
                self.valid_tag_ids = sorted(list(seen_tags))
            except Exception as e:
                log.debug(f"[WS] 解析标定清单失败 {self.workspace_id}: {e}")
                self.active_image_count = 0
        else:
            self.active_image_count = 0
            self.valid_tag_ids = []

        # 4. BA 平差解算状态及指标
        if os.path.exists(self.map_path) and os.path.getsize(self.map_path) > 50:
            self.ba_solved = True
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    map_data = yaml.safe_load(f) or {}
                meta = map_data.get("meta", {})
                self.global_rmse_px = float(meta.get("global_rmse_px", 0.0))
            except Exception:
                self.global_rmse_px = 0.0
        else:
            self.ba_solved = False
            self.global_rmse_px = 0.0

        # 5. 检查是否为生产发布地图
        if os.path.exists(DEFAULT_CONFIG_PATH):
            try:
                with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                prod_ws_id = cfg.get("calibration", {}).get("prod_workspace_id", "")
                self.is_published = (prod_ws_id == self.workspace_id)
            except Exception:
                self.is_published = False
        else:
            self.is_published = False

        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def save_meta(self):
        """持久化元数据至 workspace_meta.yaml"""
        self.ensure_directories()
        data = {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": self.updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_serial": self.camera_serial,
            "valid_tag_ids": self.valid_tag_ids,
            "origin_tag_id": self.origin_tag_id,
            "x_axis_tag_id": self.x_axis_tag_id,
            "status": {
                "image_count": self.image_count,
                "prod_image_count": self.prod_image_count,
                "active_image_count": self.active_image_count,
                "ba_solved": self.ba_solved,
                "global_rmse_px": self.global_rmse_px,
                "is_published": self.is_published,
            }
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, workspace_dir: str, force_refresh: bool = False) -> Optional["Workspace"]:
        """从已有物理目录加载 Workspace 对象 (唯一信任 workspace_meta.yaml)"""
        if not os.path.isdir(workspace_dir):
            return None
        ws_id = os.path.basename(os.path.normpath(workspace_dir))
        meta_path = os.path.join(workspace_dir, "workspace_meta.yaml")

        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
            except Exception:
                meta = {}

        name = meta.get("name") or (ws_id.split("_", 1)[1] if "_" in ws_id else ws_id)
        desc = meta.get("description", "")
        created_at = meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        updated_at = meta.get("updated_at", "")
        camera_serial = meta.get("camera_serial", "")
        valid_tag_ids = meta.get("valid_tag_ids", [])
        origin_tag_id = meta.get("origin_tag_id", 0)
        x_axis_tag_id = meta.get("x_axis_tag_id", 28)

        status = meta.get("status", {})
        image_count = status.get("image_count", 0)
        prod_image_count = status.get("prod_image_count", 0)
        active_image_count = status.get("active_image_count", 0)
        ba_solved = status.get("ba_solved", False)
        global_rmse_px = status.get("global_rmse_px", 0.0)
        is_published = status.get("is_published", False)

        ws = cls(
            workspace_id=ws_id,
            name=name,
            workspace_dir=os.path.abspath(workspace_dir),
            description=desc,
            created_at=created_at,
            updated_at=updated_at,
            camera_serial=camera_serial,
            valid_tag_ids=valid_tag_ids,
            origin_tag_id=origin_tag_id,
            x_axis_tag_id=x_axis_tag_id,
            image_count=image_count,
            prod_image_count=prod_image_count,
            active_image_count=active_image_count,
            ba_solved=ba_solved,
            global_rmse_px=global_rmse_px,
            is_published=is_published
        )
        if force_refresh or not status:
            ws.refresh_stats()
        return ws


class WorkspaceManager:
    """工位工作空间总库管理器"""

    def __init__(
        self,
        workspaces_dir: str = DEFAULT_WORKSPACES_DIR,
        config_path: str = DEFAULT_CONFIG_PATH,
        prod_map_path: str = DEFAULT_PROD_MAP_PATH
    ):
        self.workspaces_dir = os.path.abspath(workspaces_dir)
        self.config_path = os.path.abspath(config_path)
        self.prod_map_path = os.path.abspath(prod_map_path)
        self.active_marker_file = os.path.join(self.workspaces_dir, ".active_workspace")
        self._cached_active_ws: Optional[Workspace] = None
        self._cached_workspaces: Dict[str, Workspace] = {}

        os.makedirs(self.workspaces_dir, exist_ok=True)

    def list_workspaces(self) -> List[Workspace]:
        """枚举所有有效工位，按创建时间降序"""
        workspaces = []
        if not os.path.isdir(self.workspaces_dir):
            return []

        for item in os.listdir(self.workspaces_dir):
            if item.startswith("."):
                continue
            item_path = os.path.join(self.workspaces_dir, item)
            if os.path.isdir(item_path):
                try:
                    ws = Workspace.load(item_path)
                    if ws:
                        workspaces.append(ws)
                except Exception as e:
                    log.warning(f"[WARN] 加载工位异常 {item}: {e}")

        workspaces.sort(key=lambda s: (s.created_at, s.workspace_id), reverse=True)
        return workspaces

    def get_production_workspace_id(self) -> str:
        """获取当前发布为生产运行的工位 ID"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                prod_id = cfg.get("calibration", {}).get("prod_workspace_id", "")
                if prod_id and os.path.isdir(os.path.join(self.workspaces_dir, prod_id)):
                    return prod_id
            except Exception:
                pass

        for ws in self.list_workspaces():
            if ws.is_published:
                return ws.workspace_id
        return ""

    def get_production_workspace(self) -> Optional[Workspace]:
        """获取当前发布为生产运行的工位对象"""
        pid = self.get_production_workspace_id()
        return self.get_workspace_by_id(pid) if pid else None

    def get_workspace_by_id(self, ws_id: str, force_refresh: bool = False) -> Optional[Workspace]:
        """根据工位 ID 检索 Workspace 对象"""
        if not ws_id:
            return None
        target_dir = os.path.join(self.workspaces_dir, ws_id)
        if not os.path.isdir(target_dir):
            return None
        if not force_refresh and ws_id in self._cached_workspaces:
            return self._cached_workspaces[ws_id]
        ws = Workspace.load(target_dir, force_refresh=force_refresh)
        if ws:
            self._cached_workspaces[ws_id] = ws
        return ws

    def get_current_workspace_id(self) -> str:
        """获取当前默认工位 ID (优先生产工位，次优活动标记，兜底最新工位)"""
        prod_id = self.get_production_workspace_id()
        if prod_id:
            return prod_id

        # 检查 .active_workspace 标记
        if os.path.exists(self.active_marker_file):
            try:
                with open(self.active_marker_file, "r", encoding="utf-8") as f:
                    sid = f.read().strip()
                if sid and os.path.isdir(os.path.join(self.workspaces_dir, sid)):
                    return sid
            except Exception:
                pass

        workspaces = self.list_workspaces()
        if workspaces:
            return workspaces[0].workspace_id
        return ""

    def get_current_workspace(self, force_refresh: bool = False) -> Workspace:
        """获取当前默认工位对象"""
        cur_id = self.get_current_workspace_id()
        if cur_id:
            ws = self.get_workspace_by_id(cur_id, force_refresh=force_refresh)
            if ws:
                return ws

        new_ws = self.create_workspace(alias="默认工位", description="系统自动初始化默认工位")
        return new_ws

    def set_active_workspace(self, ws_id: str) -> bool:
        """切换默认工位"""
        target_dir = os.path.join(self.workspaces_dir, ws_id)
        if not os.path.isdir(target_dir):
            return False
        try:
            with open(self.active_marker_file, "w", encoding="utf-8") as f:
                f.write(ws_id.strip())
            self._cached_workspaces[ws_id] = Workspace.load(target_dir)
            return True
        except Exception as e:
            log.warning(f"[WS] 写入默认工位标记失败: {e}")
            return False

    def invalidate_cache(self):
        """显式使缓存失效"""
        self._cached_active_ws = None
        self._cached_workspaces.clear()

    def create_workspace(self, alias: str, description: str = "") -> Workspace:
        """创建新工位"""
        display_name = alias.strip() if alias and alias.strip() else "新建工位"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ascii_suffix = "".join(c for c in alias if c.isascii() and (c.isalnum() or c in ("_", "-"))).strip("_")
        if ascii_suffix:
            ws_id = f"{timestamp}_{ascii_suffix}"
        else:
            ws_id = f"{timestamp}_workspace"

        counter = 1
        original_id = ws_id
        while os.path.exists(os.path.join(self.workspaces_dir, ws_id)):
            ws_id = f"{original_id}_{counter}"
            counter += 1

        ws_dir = os.path.join(self.workspaces_dir, ws_id)
        ws = Workspace(
            workspace_id=ws_id,
            name=display_name,
            workspace_dir=ws_dir,
            description=description,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        ws.ensure_directories()
        ws.save_meta()

        # 生成初始白名单
        with open(ws.whitelist_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "workspace_id": ws_id,
                "whitelist_tag_ids": [],
                "description": f"{display_name} Tag 白名单"
            }, f, allow_unicode=True, default_flow_style=False)

        self._cached_workspaces[ws_id] = ws
        try:
            with open(self.active_marker_file, "w", encoding="utf-8") as f:
                f.write(ws_id.strip())
        except Exception:
            pass
        return ws

    def clone_workspace(self, src_ws_id: str, new_alias: str, description: str = "") -> Optional[Workspace]:
        """克隆已有工位数据至新工位"""
        src_dir = os.path.join(self.workspaces_dir, src_ws_id)
        if not os.path.isdir(src_dir):
            return None

        src_ws = Workspace.load(src_dir)
        if not src_ws:
            return None

        new_ws = self.create_workspace(alias=new_alias, description=description or f"克隆自 {src_ws_id}")

        # 1. 拷贝顶层地图与白名单
        if os.path.exists(src_ws.map_path):
            shutil.copy2(src_ws.map_path, new_ws.map_path)
        if os.path.exists(src_ws.whitelist_path):
            shutil.copy2(src_ws.whitelist_path, new_ws.whitelist_path)

        # 2. 拷贝标定图片与清单
        if os.path.exists(src_ws.calib_raw_images_dir):
            for f in glob.glob(os.path.join(src_ws.calib_raw_images_dir, "*.png")):
                shutil.copy2(f, os.path.join(new_ws.calib_raw_images_dir, os.path.basename(f)))
        if os.path.exists(src_ws.calib_manifest_path):
            shutil.copy2(src_ws.calib_manifest_path, new_ws.calib_manifest_path)

        # 3. 拷贝生产图片
        if os.path.exists(src_ws.prod_raw_images_dir):
            for f in glob.glob(os.path.join(src_ws.prod_raw_images_dir, "*.png")):
                shutil.copy2(f, os.path.join(new_ws.prod_raw_images_dir, os.path.basename(f)))

        new_ws.refresh_stats()
        new_ws.save_meta()
        self._cached_workspaces[new_ws.workspace_id] = new_ws
        return new_ws

    def rename_workspace(self, ws_id: str, new_name: str, new_description: Optional[str] = None) -> bool:
        """修改工位别名与描述"""
        clean_name = new_name.strip()
        if not clean_name:
            return False

        target_dir = os.path.join(self.workspaces_dir, ws_id)
        ws = Workspace.load(target_dir)
        if not ws:
            return False

        ws.name = clean_name
        if new_description is not None:
            ws.description = new_description
        ws.save_meta()
        self._cached_workspaces[ws_id] = ws
        return True

    def publish_to_production(self, ws_id: Optional[str] = None) -> Tuple[bool, str]:
        """将指定工位的 tags_map.yaml 安全原子发布覆盖至 config/tags_map.yaml 并记录至 config.yaml"""
        target_id = ws_id or self.get_current_workspace_id()
        if not target_id:
            return False, "无可发布的工位"

        ws_dir = os.path.join(self.workspaces_dir, target_id)
        ws = Workspace.load(ws_dir)
        if not ws:
            return False, f"工位不存在: {target_id}"

        if not os.path.exists(ws.map_path) or os.path.getsize(ws.map_path) < 50:
            return False, "该工位尚未平差生成有效地图 (tags_map.yaml 缺失或为空)"

        try:
            os.makedirs(os.path.dirname(self.prod_map_path), exist_ok=True)
            if os.path.exists(self.prod_map_path):
                shutil.copy2(self.prod_map_path, f"{self.prod_map_path}.bak")

            shutil.copy2(ws.map_path, self.prod_map_path)

            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if "calibration" not in cfg:
                    cfg["calibration"] = {}
                cfg["calibration"]["prod_workspace_id"] = target_id
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            for other_ws in self.list_workspaces():
                if other_ws.workspace_id == target_id:
                    other_ws.is_published = True
                    other_ws.save_meta()
                elif other_ws.is_published:
                    other_ws.is_published = False
                    other_ws.save_meta()

            self.invalidate_cache()
            return True, f"成功将工位 [{target_id}] 发布为全局生产运行地图 (RMSE: {ws.global_rmse_px:.3f}px)"
        except Exception as e:
            return False, f"发布至生产环境发生异常: {e}"

    def delete_workspace(self, ws_id: str) -> Tuple[bool, str]:
        """物理删除指定工位"""
        ws_dir = os.path.join(self.workspaces_dir, ws_id)
        if not os.path.isdir(ws_dir):
            return False, f"工位目录不存在: {ws_id}"

        try:
            shutil.rmtree(ws_dir)
            if ws_id in self._cached_workspaces:
                del self._cached_workspaces[ws_id]
            self.invalidate_cache()
            return True, f"已成功删除工位: {ws_id}"
        except Exception as e:
            return False, f"删除工位发生异常: {e}"

