#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StudioDataManager - 离线标定工作站数据管理与指标缓存层
======================================================
单一职责：
1. 采图资产扫描与当前选定帧维护；
2. 观测清单 (Manifest YAML) 与标靶立体几何地图 (tags_map.yaml) 的持久化与同步；
3. 全景/单帧重投影误差 (RMSE) 精度体检缓存与实时计算；
4. 帧/标靶保留与剔除状态翻转；
5. 左栏序列过滤筛选 (All / Warning / Excluded) 与四种排序规则。
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import yaml

from src.calibration.manifest_repository import ManifestRepository
from src.calibration.offline_engine import OfflineVerificationEngine


class StudioDataManager:
    """Offline Studio 领域模型与数据状态管理器"""

    def __init__(
        self,
        map_path: str,
        image_dir: str,
        manifest_path: str,
        engine: OfflineVerificationEngine,
        marker_size_mm: float = 50.0
    ):
        self.map_path = map_path
        self.image_dir = image_dir
        self.manifest_path = manifest_path
        self.engine = engine
        self.marker_size_mm = marker_size_mm

        # 1. 资产与清单
        self.manifest_repo = ManifestRepository()
        self.manifest_data: Dict[str, Any] = {}
        self._load_manifest()

        self.tags_map_data: Dict[str, Any] = {}
        self._load_tags_map()

        self.image_files: List[str] = []
        self._scan_images()
        self.current_img_idx: int = 0
        self.scroll_offset: int = 0

        # 2. 筛选与排序模式
        self.filter_mode: str = "all"
        self.sort_mode: str = "name_asc"

        # 3. 体检指标缓存
        self.frame_metrics_cache: Dict[str, Dict[str, Any]] = {}
        self.global_rmse: float = 0.0

        # 首次加载全集残差指标
        self.refresh_all_frame_metrics()

    def _scan_images(self):
        """扫描采图资产目录"""
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir, exist_ok=True)
            self.image_files = []
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        self.image_files = sorted(files)

    def _load_manifest(self):
        """加载观测清单文件"""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest_data = yaml.safe_load(f) or {}
            except Exception:
                self.manifest_data = {"images": {}}
        else:
            self.manifest_data = {"images": {}}

    def _save_manifest(self):
        """保存观测清单文件"""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(self.manifest_data, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"[WARN] 保存 Manifest 异常: {e}")

    def _load_tags_map(self):
        """加载空间立体地图数据"""
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.tags_map_data = yaml.safe_load(f) or {}
                if self.tags_map_data and "tags" in self.tags_map_data:
                    print(f"[OK] Studio 成功装载地图: {self.map_path} (共 {len(self.tags_map_data['tags'])} 个标靶)")
            except Exception as e:
                print(f"[WARN] 无法读取地图: {e}")
                self.tags_map_data = {}
        else:
            self.tags_map_data = {}
        if self.engine:
            self.engine.tags_map = self.tags_map_data

    def get_tag_transform(self, tag_id: int) -> Optional[np.ndarray]:
        """获取已知标靶在世界系下的 4x4 位姿变换矩阵"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return None
        t_info = self.tags_map_data["tags"].get(tag_id)
        if not t_info or "transform_matrix" not in t_info:
            return None
        return np.array(t_info["transform_matrix"], dtype=np.float64)

    def get_tag_world_corners(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶 4 角点在世界坐标系下的 3D 物理坐标 (4, 3)"""
        T = self.get_tag_transform(tag_id)
        if T is None:
            return None
        s = self.marker_size_mm / 2.0
        local_corners = np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)
        world_corners = (T @ local_corners.T).T
        return world_corners[:, :3]

    def get_observations_for_image(self, base_name: str) -> List[Dict[str, Any]]:
        """获取某张采图下的所有标靶观测记录"""
        images_dict = self.manifest_data.get("images", {})
        info = images_dict.get(base_name, {})
        return info.get("observations", [])

    def is_image_excluded(self, base_name: str) -> bool:
        """判断某帧是否被整帧标记为剔除"""
        images_dict = self.manifest_data.get("images", {})
        info = images_dict.get(base_name, {})
        return info.get("excluded", False)

    def toggle_image_exclusion(self, base_name: str) -> bool:
        """翻转单张图像的保留/剔除状态并持久化"""
        self.manifest_data.setdefault("images", {}).setdefault(base_name, {})
        curr = self.manifest_data["images"][base_name].get("excluded", False)
        new_status = not curr
        self.manifest_data["images"][base_name]["excluded"] = new_status
        self._save_manifest()
        return new_status

    def toggle_observation_keep(self, base_name: str, tag_id: int) -> bool:
        """翻转某帧中特定标靶的保留/剔除状态并持久化"""
        obs_list = self.get_observations_for_image(base_name)
        new_keep = False
        for obs in obs_list:
            if obs.get("tag_id") == tag_id:
                curr_keep = obs.get("keep", True)
                new_keep = not curr_keep
                obs["keep"] = new_keep
                break
        self._save_manifest()
        return new_keep

    def toggle_current_frame_exclusion(self) -> Tuple[str, bool]:
        """翻转当前选中帧的保留/剔除状态"""
        if not self.image_files or self.current_img_idx >= len(self.image_files):
            return "", False
        cur_file = self.image_files[self.current_img_idx]
        base_name = os.path.basename(cur_file)

        is_now_excluded = self.toggle_image_exclusion(base_name)
        if base_name in self.frame_metrics_cache:
            self.frame_metrics_cache[base_name]["is_excluded"] = is_now_excluded
        return base_name, is_now_excluded

    def toggle_tag_exclusion_in_current_frame(self, target_tag_id: int) -> Tuple[str, bool]:
        """翻转当前帧中特定标靶的保留/剔除状态并重算当前帧指标"""
        if not self.image_files or self.current_img_idx >= len(self.image_files):
            return "", False
        cur_file = self.image_files[self.current_img_idx]
        base_name = os.path.basename(cur_file)

        is_kept = self.toggle_observation_keep(base_name, target_tag_id)
        meta = self.frame_metrics_cache.get(base_name, {})
        obs_list = self.get_observations_for_image(base_name)
        mean_err, max_err, errors_dict = self._evaluate_frame_reprojection(obs_list)
        meta["observations"] = obs_list
        meta["mean_err"] = mean_err
        meta["max_err"] = max_err
        meta["tag_errors"] = errors_dict
        return base_name, is_kept

    def _evaluate_frame_reprojection(self, observations: List[Dict[str, Any]]) -> Tuple[float, float, Dict[int, float]]:
        """计算单帧中所有有效标靶的重投影残差"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return 0.0, 0.0, {}

        obj_pts = []
        img_pts = []
        valid_tids = []

        for obs in observations:
            if not obs.get("keep", True):
                continue
            tid = obs["tag_id"]
            w_corners = self.get_tag_world_corners(tid)
            if w_corners is not None:
                c_arr = np.array(obs["corners"], dtype=np.float64).reshape((4, 2))
                obj_pts.append(w_corners)
                img_pts.append(c_arr)
                valid_tids.append(tid)

        if not obj_pts:
            return 0.0, 0.0, {}

        obj_flat = np.concatenate(obj_pts, axis=0)
        img_flat = np.concatenate(img_pts, axis=0)

        rvec, tvec, success = self.engine.solve_pnp(obj_flat, img_flat)
        if not success:
            return 0.0, 0.0, {}

        proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, self.engine.camera_matrix, self.engine.dist_coeffs)
        dists = np.linalg.norm(img_flat - proj_pts.reshape((-1, 2)), axis=1)

        errors_dict = {}
        for idx, tid in enumerate(valid_tids):
            errors_dict[tid] = float(np.mean(dists[idx * 4:(idx + 1) * 4]))

        mean_val = float(np.mean(dists))
        max_val = float(np.max(dists))
        return mean_val, max_val, errors_dict

    def refresh_all_frame_metrics(self):
        """全量预热并刷新所有采图帧的精度体检残差指标"""
        self.frame_metrics_cache.clear()
        all_reproj_errors = []

        for p in self.image_files:
            base_name = os.path.basename(p)
            obs_list = self.get_observations_for_image(base_name)
            is_excl = self.is_image_excluded(base_name)

            mean_err, max_err, errors_dict = self._evaluate_frame_reprojection(obs_list)
            if not is_excl and obs_list:
                all_reproj_errors.extend(list(errors_dict.values()))

            self.frame_metrics_cache[base_name] = {
                "tag_count": len(obs_list),
                "mean_err": mean_err,
                "max_err": max_err,
                "is_excluded": is_excl,
                "observations": obs_list,
                "tag_errors": errors_dict
            }

        if all_reproj_errors:
            self.global_rmse = float(np.sqrt(np.mean(np.array(all_reproj_errors) ** 2)))
        else:
            self.global_rmse = 0.0

    def get_filtered_indices(self) -> List[int]:
        """依据当前的过滤模式与排序模式获取最终展示的帧索引列表"""
        matched = []
        for idx, p in enumerate(self.image_files):
            base_name = os.path.basename(p)
            meta = self.frame_metrics_cache.get(base_name, {})
            if self.filter_mode == "all":
                matched.append(idx)
            elif self.filter_mode == "warning":
                if meta.get("mean_err", 0.0) > 0.5 and not meta.get("is_excluded", False):
                    matched.append(idx)
            elif self.filter_mode == "excluded":
                if meta.get("is_excluded", False):
                    matched.append(idx)

        # 排序规则处理
        if self.sort_mode == "name_asc":
            matched.sort(key=lambda i: os.path.basename(self.image_files[i]))
        elif self.sort_mode == "err_desc":
            matched.sort(
                key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("mean_err", 0.0),
                reverse=True
            )
        elif self.sort_mode == "err_asc":
            matched.sort(
                key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("mean_err", 0.0)
            )
        elif self.sort_mode == "tags_desc":
            matched.sort(
                key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("tag_count", 0),
                reverse=True
            )

        return matched
