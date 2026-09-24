"""
工位多坐标系拓扑树管理模块 (Coordinate Tree Manager)
=====================================================
负责管理工位内多坐标系 (世界坐标系 + 多个相对坐标系) 的树状拓扑关系与矩阵级联变换：
1. 支持绝对世界坐标系 (world) 作为唯一根节点；
2. 支持固定刚体变换类型 (fixed_transform: 平移 + 欧拉角 RPY)；
3. 支持 AprilTag 动标绑定类型 (tag_bound: 绑定 Tag ID + 局部微调偏移)；
4. 提供多级坐标系递归解析、循环依赖防护、动标丢失降级处理；
5. 提供高效的点云与位姿变换接口。
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from src.utils.logger import get_logger

log = get_logger(__name__)


def rpy_deg_to_rot_mat(rpy_deg: List[float]) -> np.ndarray:
    """欧拉角 (Roll, Pitch, Yaw, 单位: 度, 顺序: xyz 外部旋转) 转换为 3x3 旋转矩阵"""
    if not rpy_deg or len(rpy_deg) < 3:
        return np.eye(3, dtype=np.float64)
    # 使用 ext-xyz (roll حول X, pitch حول Y, yaw حول Z)
    rot = R.from_euler("xyz", rpy_deg, degrees=True)
    return rot.as_matrix()


def rot_mat_to_rpy_deg(rot_mat: np.ndarray) -> List[float]:
    """3x3 旋转矩阵转换为 (Roll, Pitch, Yaw, 单位: 度)"""
    rot = R.from_matrix(rot_mat)
    angles = rot.as_euler("xyz", degrees=True)
    return [float(a) for a in angles]


def make_transform_matrix(rot_mat: np.ndarray, t_xyz: List[float]) -> np.ndarray:
    """构建 4x4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_mat
    if t_xyz and len(t_xyz) >= 3:
        T[:3, 3] = np.array(t_xyz[:3], dtype=np.float64)
    return T


@dataclass
class FrameDefinition:
    """单个坐标系数据模型"""
    frame_id: str                                  # 唯一标识，如 world, frame_conveyor, frame_belt
    name: str                                      # 友好别名，如 绝对世界坐标系, 主输送机坐标系
    parent_frame_id: Optional[str] = None          # 父坐标系 ID (world 为根节点, 值为 None)
    type: str = "fixed_transform"                  # 类型: "world" | "fixed_transform" | "tag_bound"
    description: str = ""                          # 说明备注

    # 针对 fixed_transform 类型的参数
    translation_xyz_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 针对 tag_bound 类型的参数 (支持多动标列表)
    tag_id: Optional[int] = None
    tag_ids: List[int] = field(default_factory=list)
    offset_xyz_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    offset_rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def get_tag_ids(self) -> List[int]:
        """获取动标绑定的 Tag ID 列表 (兼容单动标与多动标冗余组)"""
        res: List[int] = []
        if getattr(self, "tag_ids", None):
            for t in self.tag_ids:
                try:
                    res.append(int(t))
                except (ValueError, TypeError):
                    pass
        if not res and self.tag_id is not None:
            try:
                res.append(int(self.tag_id))
            except (ValueError, TypeError):
                pass
        return res

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典供 YAML 存储"""
        d: Dict[str, Any] = {
            "frame_id": self.frame_id,
            "name": self.name,
            "parent_frame_id": self.parent_frame_id,
            "type": self.type,
            "description": self.description,
        }
        if self.type == "fixed_transform":
            d["transform"] = {
                "translation_xyz_mm": [float(x) for x in self.translation_xyz_mm],
                "rotation_rpy_deg": [float(x) for x in self.rotation_rpy_deg],
            }
        elif self.type == "tag_bound":
            t_ids = self.get_tag_ids()
            d["tag_binding"] = {
                "tag_ids": t_ids,
                "tag_id": t_ids[0] if t_ids else 0,
                "offset_xyz_mm": [float(x) for x in self.offset_xyz_mm],
                "offset_rpy_deg": [float(x) for x in self.offset_rpy_deg],
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameDefinition":
        """从字典反序列化"""
        fid = data.get("frame_id", "")
        name = data.get("name", fid)
        parent_fid = data.get("parent_frame_id")
        ftype = data.get("type", "fixed_transform")
        desc = data.get("description", "")

        t_xyz = [0.0, 0.0, 0.0]
        r_rpy = [0.0, 0.0, 0.0]
        tag_id = None
        tag_ids: List[int] = []
        off_xyz = [0.0, 0.0, 0.0]
        off_rpy = [0.0, 0.0, 0.0]

        if ftype == "fixed_transform":
            trans_block = data.get("transform", {})
            t_xyz = trans_block.get("translation_xyz_mm", [0.0, 0.0, 0.0])
            r_rpy = trans_block.get("rotation_rpy_deg", [0.0, 0.0, 0.0])
        elif ftype == "tag_bound":
            bind_block = data.get("tag_binding", {})
            tag_id = bind_block.get("tag_id")
            raw_ids = bind_block.get("tag_ids")
            if isinstance(raw_ids, list):
                for x in raw_ids:
                    try:
                        tag_ids.append(int(x))
                    except (ValueError, TypeError):
                        pass
            elif tag_id is not None:
                tag_ids = [int(tag_id)]
            if tag_id is None and tag_ids:
                tag_id = tag_ids[0]
            off_xyz = bind_block.get("offset_xyz_mm", [0.0, 0.0, 0.0])
            off_rpy = bind_block.get("offset_rpy_deg", [0.0, 0.0, 0.0])

        return cls(
            frame_id=fid,
            name=name,
            parent_frame_id=parent_fid,
            type=ftype,
            description=desc,
            translation_xyz_mm=t_xyz,
            rotation_rpy_deg=r_rpy,
            tag_id=tag_id,
            tag_ids=tag_ids,
            offset_xyz_mm=off_xyz,
            offset_rpy_deg=off_rpy,
        )


class CoordinateTreeManager:
    """工位坐标系树管理与几何变换求解器"""

    def __init__(self, workspace_id: str = "", frames_yaml_path: Optional[str] = None):
        self.workspace_id = workspace_id
        self.frames_yaml_path = frames_yaml_path
        self.active_frame_id: str = "world"
        self._frames: Dict[str, FrameDefinition] = {}
        self._tags_map: Dict[int, np.ndarray] = {}  # {tag_id: 4x4 T_world_from_tag}

        # 默认加载或提供初始世界坐标系
        if frames_yaml_path and os.path.exists(frames_yaml_path):
            self.load()
        else:
            self._init_default_world()

    def _init_default_world(self):
        """初始化最简世界坐标系"""
        self._frames = {
            "world": FrameDefinition(
                frame_id="world",
                name="绝对世界坐标系",
                parent_frame_id=None,
                type="world",
                description="工位绝对基准坐标系，与 tags_map.yaml 和 anchor_tags.yaml 对齐",
            )
        }
        self.active_frame_id = "world"

    def set_tags_map(self, tags_map: Dict[int, np.ndarray]):
        """更新当前工位解算出的世界系 AprilTag 位姿地图"""
        self._tags_map = {int(k): np.array(v, dtype=np.float64) for k, v in tags_map.items()}

    def add_frame(self, frame: FrameDefinition) -> bool:
        """添加或替换坐标系定义 (含拓扑防环检查)"""
        # 预先检查父坐标系是否存在 (除 world 外)
        if frame.frame_id != "world":
            if not frame.parent_frame_id:
                frame.parent_frame_id = "world"
            if frame.parent_frame_id not in self._frames:
                log.error(f"[FrameTree] 父坐标系 '{frame.parent_frame_id}' 不存在，添加 '{frame.frame_id}' 失败！")
                return False

        temp_frames = dict(self._frames)
        temp_frames[frame.frame_id] = frame
        if self._has_cycle(temp_frames):
            log.error(f"[FrameTree] 添加坐标系 {frame.frame_id} 会导致拓扑循环依赖，已被拒绝！")
            return False

        self._frames[frame.frame_id] = frame
        return True

    def remove_frame(self, frame_id: str) -> bool:
        """删除指定坐标系 (根坐标系 world 禁止删除)"""
        if frame_id == "world":
            log.warning("[FrameTree] 根坐标系 world 为绝对基准，禁止删除！")
            return False
        if frame_id not in self._frames:
            return False

        # 将以此为父的子坐标系重定向到 world
        for f in self._frames.values():
            if f.parent_frame_id == frame_id:
                f.parent_frame_id = "world"

        del self._frames[frame_id]
        if self.active_frame_id == frame_id:
            self.active_frame_id = "world"
        return True

    def rename_frame(self, old_frame_id: str, new_frame_id: str) -> bool:
        """重命名指定坐标系，并级联更新所有子坐标系的 parent_frame_id"""
        if old_frame_id == "world":
            log.warning("[FrameTree] 根坐标系 world 为绝对基准，禁止重命名！")
            return False
        if not new_frame_id or not new_frame_id.strip():
            log.warning("[FrameTree] 新坐标系 ID 不能为空！")
            return False
        new_frame_id = new_frame_id.strip()
        if old_frame_id == new_frame_id:
            return True
        if old_frame_id not in self._frames:
            log.warning(f"[FrameTree] 原坐标系 {old_frame_id} 不存在！")
            return False
        if new_frame_id in self._frames:
            log.warning(f"[FrameTree] 目标坐标系 ID {new_frame_id} 已存在，不可重复！")
            return False

        frame = self._frames.pop(old_frame_id)
        frame.frame_id = new_frame_id
        self._frames[new_frame_id] = frame

        # 级联更新以此为父的子坐标系
        for f in self._frames.values():
            if f.parent_frame_id == old_frame_id:
                f.parent_frame_id = new_frame_id

        if self.active_frame_id == old_frame_id:
            self.active_frame_id = new_frame_id
        return True

    def get_frame(self, frame_id: str) -> Optional[FrameDefinition]:
        return self._frames.get(frame_id)

    def list_frames(self) -> List[FrameDefinition]:
        """返回按层级拓扑排序的坐标系列表"""
        # world 始终第一
        res = []
        if "world" in self._frames:
            res.append(self._frames["world"])
        for fid, f in self._frames.items():
            if fid != "world":
                res.append(f)
        return res

    def _has_cycle(self, frames_dict: Dict[str, FrameDefinition]) -> bool:
        """检测坐标系拓扑树是否存在环路"""
        visited = set()
        stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            parent = frames_dict[node].parent_frame_id if node in frames_dict else None
            if parent and parent in frames_dict:
                if parent not in visited:
                    if dfs(parent):
                        return True
                elif parent in stack:
                    return True
            stack.remove(node)
            return False

        for node in frames_dict:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def get_relative_transform_to_parent(self, frame_id: str) -> Tuple[np.ndarray, bool]:
        """
        计算坐标系相对于其直接父级坐标系的 4x4 变换矩阵 T_parent_from_frame。
        返回值: (T, is_valid)
        """
        if frame_id not in self._frames:
            return np.eye(4, dtype=np.float64), False

        frame = self._frames[frame_id]
        if frame.type == "world" or not frame.parent_frame_id:
            return np.eye(4, dtype=np.float64), True

        if frame.type == "fixed_transform":
            R_mat = rpy_deg_to_rot_mat(frame.rotation_rpy_deg)
            T = make_transform_matrix(R_mat, frame.translation_xyz_mm)
            return T, True

        if frame.type == "tag_bound":
            # 动标绑定类型 (支持多动标冗余跟踪与回退)
            cand_tags = frame.get_tag_ids()
            resolved_tag = next((tid for tid in cand_tags if tid in self._tags_map), None)
            R_off = rpy_deg_to_rot_mat(frame.offset_rpy_deg)
            T_tag_from_frame = make_transform_matrix(R_off, frame.offset_xyz_mm)

            if resolved_tag is None:
                # 所有候选动标均丢失或未标定，降级输出局部偏移，并标记有效性为 False
                return T_tag_from_frame, False

            # T_world_from_tag
            T_world_from_tag = self._tags_map[resolved_tag]
            # 若 parent 是 world，则 T_parent_from_frame = T_world_from_tag * T_tag_from_frame
            # 若 parent 不是 world，需要按 T_parent_from_world * T_world_from_tag * T_tag_from_frame 计算
            if frame.parent_frame_id == "world":
                return T_world_from_tag @ T_tag_from_frame, True
            else:
                T_world_from_parent, valid_p = self.get_frame_to_world(frame.parent_frame_id)
                if not valid_p:
                    return T_tag_from_frame, False
                T_parent_from_world = np.linalg.inv(T_world_from_parent)
                return T_parent_from_world @ T_world_from_tag @ T_tag_from_frame, True

        return np.eye(4, dtype=np.float64), True

    def get_frame_to_world(self, frame_id: str) -> Tuple[np.ndarray, bool]:
        """
        递归求解任意坐标系到世界坐标系的变换矩阵 T_world_from_frame。
        返回值: (T_4x4, is_resolved)
        """
        if frame_id == "world":
            return np.eye(4, dtype=np.float64), True
        if frame_id not in self._frames:
            return np.eye(4, dtype=np.float64), False

        frame = self._frames[frame_id]
        if frame.type == "tag_bound":
            cand_tags = frame.get_tag_ids()
            resolved_tag = next((tid for tid in cand_tags if tid in self._tags_map), None)
            R_off = rpy_deg_to_rot_mat(frame.offset_rpy_deg)
            T_tag_from_frame = make_transform_matrix(R_off, frame.offset_xyz_mm)

            if resolved_tag is not None:
                T_world_from_tag = self._tags_map[resolved_tag]
                return T_world_from_tag @ T_tag_from_frame, True
            else:
                # Tag 丢失，未解算
                return T_tag_from_frame, False

        # fixed_transform 沿树递归向上乘
        parent_id = frame.parent_frame_id or "world"
        T_parent_world, parent_valid = self.get_frame_to_world(parent_id)
        T_parent_from_frame, rel_valid = self.get_relative_transform_to_parent(frame_id)

        return T_parent_world @ T_parent_from_frame, (parent_valid and rel_valid)

    def get_transform(self, src_frame: str, dst_frame: str) -> Tuple[np.ndarray, bool]:
        """
        计算从 src_frame 到 dst_frame 的 4x4 齐次变换矩阵 T_dst_from_src。
        任意点 P_dst = T_dst_from_src @ P_src
        """
        if src_frame == dst_frame:
            return np.eye(4, dtype=np.float64), True

        T_world_from_src, valid_src = self.get_frame_to_world(src_frame)
        T_world_from_dst, valid_dst = self.get_frame_to_world(dst_frame)

        if not valid_src or not valid_dst:
            # 降级尝试
            pass

        try:
            T_dst_from_world = np.linalg.inv(T_world_from_dst)
            return T_dst_from_world @ T_world_from_src, (valid_src and valid_dst)
        except Exception as e:
            log.error(f"[FrameTree] 矩阵求逆失败 ({dst_frame}): {e}")
            return np.eye(4, dtype=np.float64), False

    def transform_points(self, points: np.ndarray, src_frame: str, dst_frame: str) -> np.ndarray:
        """
        批量变换 3D 点云: points 为 (N, 3) 数组。
        输出变换到 dst_frame 后的 (N, 3) 数组。
        """
        if points is None or len(points) == 0:
            return points
        if src_frame == dst_frame:
            return points

        T_dst_from_src, _ = self.get_transform(src_frame, dst_frame)
        N = points.shape[0]
        homo = np.hstack([points[:, :3], np.ones((N, 1), dtype=points.dtype)])
        transformed = (T_dst_from_src @ homo.T).T
        return transformed[:, :3]

    def load(self, filepath: Optional[str] = None) -> bool:
        """从 frames.yaml 文件加载坐标系配置"""
        path = filepath or self.frames_yaml_path
        if not path or not os.path.exists(path):
            self._init_default_world()
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.workspace_id = data.get("workspace_id", self.workspace_id)
            self.active_frame_id = data.get("active_frame_id", "world")
            raw_frames = data.get("frames", [])

            loaded = {}
            for item in raw_frames:
                frame = FrameDefinition.from_dict(item)
                loaded[frame.frame_id] = frame

            if "world" not in loaded:
                loaded["world"] = FrameDefinition(
                    frame_id="world",
                    name="绝对世界坐标系",
                    parent_frame_id=None,
                    type="world",
                    description="工位绝对基准坐标系",
                )

            self._frames = loaded
            if self.active_frame_id not in self._frames:
                self.active_frame_id = "world"

            log.debug(f"[FrameTree] 成功加载 {len(self._frames)} 个坐标系: {path}")
            return True
        except Exception as e:
            log.error(f"[FrameTree] 加载坐标系配置失败 ({path}): {e}")
            self._init_default_world()
            return False

    def save(self, filepath: Optional[str] = None) -> bool:
        """持久化保存至 frames.yaml 文件"""
        path = filepath or self.frames_yaml_path
        if not path:
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            data = {
                "version": "1.0",
                "workspace_id": self.workspace_id,
                "active_frame_id": self.active_frame_id,
                "frames": [f.to_dict() for f in self.list_frames()],
            }
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)
            log.info(f"[FrameTree] 成功写穿坐标系配置 ({len(self._frames)} 个): {path}")
            return True
        except Exception as e:
            log.error(f"[FrameTree] 保存坐标系配置失败 ({path}): {e}")
            return False
