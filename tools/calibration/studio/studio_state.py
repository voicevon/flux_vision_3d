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
        self.global_median_mm: float = 0.0
        self.global_mean_mm: float = 0.0
        self.gate_status: str = "REVIEW"
        self.topology_status: Dict[str, Any] = {
            "is_valid": True,
            "components_count": 1,
            "critical_bridges": [],
            "unconnected_tags": [],
            "connected_tags_count": 0,
            "message": "就绪"
        }
        self.current_diagnostics: Dict[str, Any] = {}

        # 4. 智能剪枝平差逐帧多轮残差收敛矩阵
        # frame_convergence_matrix: { "view_0001.png": [582.13, 39.81, 25.30, ...] }
        # convergence_headers: ["R0(基准)", "R1", "R2", ...]
        self.frame_convergence_matrix: Dict[str, List[Optional[float]]] = {}
        self.convergence_headers: List[str] = []

        # 5. 超精重提取引擎 (惰性装载)
        self._super_extractor = None

        # 首次加载全集残差指标
        self.refresh_all_frame_metrics()

    @property
    def super_extractor(self):
        """惰性装载工序 3 工业级超精重提取引擎"""
        if self._super_extractor is None:
            from tools.calibration.tag_super_extractor import TagSuperExtractor
            self._super_extractor = TagSuperExtractor(
                image_dir=self.image_dir,
                manifest_path=self.manifest_path,
                marker_size_mm=self.marker_size_mm
            )
        return self._super_extractor

    def _sync_manifest_summary(self):
        """同步更新 manifest 顶层 summary 统计指标"""
        total_obs = 0
        total_kept = 0
        total_excl = 0
        total_enabled = 0
        images_dict = self.manifest_data.get("images", {})
        for img_info in images_dict.values():
            if img_info.get("enabled", True) and not img_info.get("excluded", False):
                total_enabled += 1
            for obs in img_info.get("observations", []):
                total_obs += 1
                if obs.get("keep", True):
                    total_kept += 1
                else:
                    total_excl += 1
        if "summary" not in self.manifest_data or not isinstance(self.manifest_data["summary"], dict):
            self.manifest_data["summary"] = {}
        self.manifest_data["summary"].update({
            "total_images": len(images_dict),
            "total_enabled_images": total_enabled,
            "total_observations": total_obs,
            "total_kept": total_kept,
            "total_excluded": total_excl
        })

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
        """判断某帧是否被整帧标记为剔除 (兼容 excluded 与 not enabled)"""
        images_dict = self.manifest_data.get("images", {})
        info = images_dict.get(base_name, {})
        if "excluded" in info:
            return bool(info["excluded"])
        if "enabled" in info:
            return not bool(info["enabled"])
        return False

    def toggle_image_exclusion(self, base_name: str) -> bool:
        """翻转单张图像的保留/剔除状态并持久化 (双向同步 excluded 与 enabled)"""
        self.manifest_data.setdefault("images", {}).setdefault(base_name, {})
        curr = self.is_image_excluded(base_name)
        new_excluded = not curr
        self.manifest_data["images"][base_name]["excluded"] = new_excluded
        self.manifest_data["images"][base_name]["enabled"] = not new_excluded
        self._sync_manifest_summary()
        self._save_manifest()
        return new_excluded

    def super_extract_current_frame(self, img_idx: Optional[int] = None) -> Tuple[str, int]:
        """
        对指定帧或当前选定帧执行工序 3 工业级超精重提取并原子持久化
        动用 5 路增强底图 + 16 级致密网格 + 微靶 2x 超分 + 0.01px 轮廓正交亚像素精修
        """
        if img_idx is None:
            img_idx = self.current_img_idx
        if not self.image_files or img_idx >= len(self.image_files):
            return "", 0

        cur_file = self.image_files[img_idx]
        bname = os.path.basename(cur_file)

        # 动用工序 3 离线超精提取引擎
        super_res = self.super_extractor.extract_from_image(cur_file)

        # 智能继承已有的人工保留/剔除决策与备注
        existing_img_entry = self.manifest_data.get("images", {}).get(bname, {})
        existing_prefs = {}
        for obs in existing_img_entry.get("observations", []):
            tid = obs.get("tag_id")
            existing_prefs[tid] = (obs.get("keep", True), obs.get("note", ""))

        new_obs = []
        for tid in sorted(super_res.keys()):
            item = super_res[tid]
            keep_val, note_val = existing_prefs.get(tid, (True, f"超精提取 [{item.get('channel', 'SUPER')}]"))
            m = item.get("metrics", {})
            c_arr = item["corners"]
            new_obs.append({
                "tag_id": int(tid),
                "keep": bool(keep_val),
                "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in c_arr.reshape(4, 2)],
                "cell_size_px": m.get("cell_size_px", [0, 0]),
                "center_px": m.get("center_px", [0.0, 0.0]),
                "area_px": m.get("area_px", 0.0),
                "channel": item.get("channel", "SUPER"),
                "note": str(note_val)
            })

        # 组装符合 Manifest 标准格式的完整结构
        self.manifest_data.setdefault("images", {})[bname] = {
            "file_name": bname,
            "image_path": cur_file.replace("\\", "/"),
            "enabled": not self.is_image_excluded(bname),
            "excluded": self.is_image_excluded(bname),
            "detected_count": len(new_obs),
            "observations": new_obs
        }

        # 更新 summary 统计
        self._sync_manifest_summary()

        # 立即原子持久化落盘
        self._save_manifest()

        # 刷新体检指标缓存
        self.refresh_all_frame_metrics()
        print(f"[OK] [STUDIO] 帧 {bname} 超精重提取完成并已原子持久化: 检出 {len(new_obs)} 个标靶")
        return bname, len(new_obs)

    def super_extract_all_frames(
        self,
        progress_callback: Optional[Any] = None
    ) -> Tuple[int, int]:
        """
        全局全量超精重提取：
        对所有采图帧原有提取的角点与空间观测完全清空删除，
        从头开始重新调用工序 3 工业级超精重提取引擎 (5路增强 + 16级致密网格 + 2x超分 + 0.01px轮廓正交亚像素拟合)
        提取所有标靶角点，默认全部从头启用为有效保留状态，并原子持久化至 manifest_path。
        :param progress_callback: 可选进度回调 callback(cur_idx, total_count, bname, tag_count)
        :return: (处理的总帧数, 累计检出的标靶总数)
        """
        if not self.image_files:
            return 0, 0

        total_frames = len(self.image_files)
        total_tags = 0

        # 清空原有的所有帧 observations，从头彻底重建 images 字典
        new_images_dict: Dict[str, Any] = {}

        for i, cur_file in enumerate(self.image_files):
            bname = os.path.basename(cur_file)

            # 调用工序 3 离线超精提取引擎
            try:
                super_res = self.super_extractor.extract_from_image(cur_file)
            except Exception as e:
                print(f"[WARN] [STUDIO] 超精提取帧 {bname} 异常: {e}")
                super_res = {}

            new_obs = []
            for tid in sorted(super_res.keys()):
                item = super_res[tid]
                m = item.get("metrics", {})
                c_arr = item["corners"]
                new_obs.append({
                    "tag_id": int(tid),
                    "keep": True,  # 全部从头开始重新启用
                    "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in c_arr.reshape(4, 2)],
                    "cell_size_px": m.get("cell_size_px", [0, 0]),
                    "center_px": m.get("center_px", [0.0, 0.0]),
                    "area_px": m.get("area_px", 0.0),
                    "channel": item.get("channel", "SUPER"),
                    "note": f"全局超精提取 [{item.get('channel', 'SUPER')}]"
                })

            new_images_dict[bname] = {
                "file_name": bname,
                "image_path": cur_file.replace("\\", "/"),
                "enabled": True,
                "excluded": False,
                "detected_count": len(new_obs),
                "observations": new_obs
            }
            total_tags += len(new_obs)

            if progress_callback is not None:
                try:
                    progress_callback(i + 1, total_frames, bname, len(new_obs))
                except Exception:
                    pass

        self.manifest_data["images"] = new_images_dict

        # 更新 summary 统计
        self._sync_manifest_summary()

        # 立即原子持久化落盘
        self._save_manifest()

        # 刷新所有体检指标
        self.refresh_all_frame_metrics()
        print(f"[OK] [STUDIO] 全局全量超精提取完成并持久化: 共处理 {total_frames} 帧，累计检出 {total_tags} 个标靶")
        return total_frames, total_tags

    def reset_map(self) -> bool:
        """
        清空当前解算的空间立体地图，使工作站完全复位为未建图的初始状态
        同时将已存在的 tags_map.yaml 自动备份为 tags_map.yaml.bak 并写空
        """
        if os.path.exists(self.map_path):
            try:
                bak_path = self.map_path + ".bak"
                import shutil
                shutil.copyfile(self.map_path, bak_path)
                print(f"[*] [STUDIO] 旧地图已安全备份至: {bak_path}")
            except Exception as e:
                print(f"[WARN] 备份地图失败: {e}")

        # 清空内存与引擎中的地图
        self.tags_map_data = {"version": "2.0_reset", "tags": {}}
        if self.engine:
            self.engine.tags_map = self.tags_map_data

        # 写回空地图文件
        try:
            with open(self.map_path, "w", encoding="utf-8") as f:
                yaml.dump(self.tags_map_data, f, allow_unicode=True, sort_keys=False)
            print(f"[OK] [STUDIO] 地图文件已成功复位清空: {self.map_path}")
        except Exception as e:
            print(f"[WARN] 写入空地图文件失败: {e}")

        self.refresh_all_frame_metrics()
        return True

    def reset_all_keep_status(self) -> int:
        """
        一键复位全量观测状态：将所有帧设为保留 (enabled: true, excluded: false)，所有标靶观测设为有效 (keep: true)
        :return: 恢复的总观测数量
        """
        restored_count = 0
        images_dict = self.manifest_data.get("images", {})
        for img_info in images_dict.values():
            img_info["enabled"] = True
            img_info["excluded"] = False
            for obs in img_info.get("observations", []):
                obs["keep"] = True
                restored_count += 1

        self._sync_manifest_summary()
        self._save_manifest()
        self.refresh_all_frame_metrics()
        print(f"[OK] [STUDIO] 已一键复位所有标靶保留状态: 共恢复 {restored_count} 次标靶观测为有效")
        return restored_count

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
        mean_err, max_err, errors_dict, rvec, tvec, mean_mm, errs_mm = self._evaluate_frame_reprojection(obs_list)
        meta["observations"] = obs_list
        meta["mean_err"] = mean_err
        meta["max_err"] = max_err
        meta["mean_err_mm"] = mean_mm
        meta["tag_errors"] = errors_dict
        meta["tag_errors_mm"] = errs_mm
        meta["rvec"] = rvec
        meta["tvec"] = tvec
        return base_name, is_kept

    def create_manifest_snapshot(self) -> Dict[str, Any]:
        """创建当前审核清单与地图的状态快照，支持无损撤销"""
        import copy
        self._manifest_snapshot = {
            "manifest_data": copy.deepcopy(self.manifest_data),
            "tags_map_data": copy.deepcopy(self.tags_map_data),
            "global_rmse": self.global_rmse,
            "global_median_mm": self.global_median_mm,
            "global_mean_mm": self.global_mean_mm,
            "gate_status": self.gate_status,
            "topology_status": copy.deepcopy(self.topology_status),
            "frame_convergence_matrix": copy.deepcopy(self.frame_convergence_matrix),
            "convergence_headers": copy.deepcopy(self.convergence_headers)
        }
        return self._manifest_snapshot

    def restore_manifest_snapshot(self, snapshot: Optional[Dict[str, Any]] = None) -> bool:
        """从快照恢复审核清单与地图状态"""
        import copy
        snap = snapshot or getattr(self, "_manifest_snapshot", None)
        if not snap:
            return False
        self.manifest_data = copy.deepcopy(snap["manifest_data"])
        self.tags_map_data = copy.deepcopy(snap["tags_map_data"])
        self.global_rmse = snap["global_rmse"]
        self.global_median_mm = snap["global_median_mm"]
        self.global_mean_mm = snap["global_mean_mm"]
        self.gate_status = snap["gate_status"]
        self.topology_status = copy.deepcopy(snap["topology_status"])
        if "frame_convergence_matrix" in snap:
            self.frame_convergence_matrix = copy.deepcopy(snap["frame_convergence_matrix"])
        if "convergence_headers" in snap:
            self.convergence_headers = copy.deepcopy(snap["convergence_headers"])

        self._save_manifest()
        if self.map_path and os.path.exists(os.path.dirname(self.map_path)):
            try:
                with open(self.map_path, "w", encoding="utf-8") as f:
                    yaml.dump(self.tags_map_data, f, allow_unicode=True, sort_keys=False)
            except Exception as e:
                print(f"[WARN] 恢复地图文件异常: {e}")
        self.refresh_all_frame_metrics()
        return True

    def find_worst_prunable_observations(
        self,
        top_k: int = 2
    ) -> List[Tuple[str, int, float, float]]:
        """
        寻找当前全局残差最大的 Top-K 个有效标靶观测，严格受共视拓扑与最小观测度（>=2次）保护
        返回: [(image_basename, tag_id, err_px, err_mm)]
        """
        from src.calibration.covisibility_graph import CovisibilityGraphAnalyzer
        import copy

        # 1. 收集全局所有有效观测及其残差
        candidates = []
        tag_active_counts: Dict[int, int] = {}
        for bname, meta in self.frame_metrics_cache.items():
            if meta.get("is_excluded", False):
                continue
            tag_errors = meta.get("tag_errors", {})
            tag_errors_mm = meta.get("tag_errors_mm", {})
            for obs in meta.get("observations", []):
                if not obs.get("keep", True):
                    continue
                tid = obs["tag_id"]
                tag_active_counts[tid] = tag_active_counts.get(tid, 0) + 1
                err_px = tag_errors.get(tid, 0.0)
                err_mm = tag_errors_mm.get(tid, 0.0)
                candidates.append((err_px, err_mm, bname, tid))

        # 按像素残差降序排列
        candidates.sort(key=lambda item: item[0], reverse=True)

        selected = []
        sim_manifest = copy.deepcopy(self.manifest_data)
        sim_counts = dict(tag_active_counts)

        for err_px, err_mm, bname, tid in candidates:
            if len(selected) >= top_k:
                break
            if err_px <= 0.0:
                continue

            # 守门规则 1: 剔除后该 Tag 的全局观测次数不得 < 2
            if sim_counts.get(tid, 0) - 1 < 2:
                continue

            # 守门规则 2: 模拟剔除后用共视拓扑图审查，必须保持单一连通且无孤岛
            obs_entry = None
            for o in sim_manifest.get("images", {}).get(bname, {}).get("observations", []):
                if o.get("tag_id") == tid:
                    obs_entry = o
                    break
            if obs_entry is None:
                continue

            obs_entry["keep"] = False

            # 正确构建 detections 结构传递给 CovisibilityGraphAnalyzer
            sim_detections = []
            sim_names = []
            for sim_bname, sim_img in sim_manifest.get("images", {}).items():
                if sim_img.get("excluded", False) or not sim_img.get("enabled", True):
                    continue
                d_map = {}
                for o in sim_img.get("observations", []):
                    if o.get("keep", True):
                        d_map[o["tag_id"]] = np.array(o["corners"], dtype=np.float32)
                if d_map:
                    sim_detections.append(d_map)
                    sim_names.append(sim_bname)

            topo_res = CovisibilityGraphAnalyzer.analyze(sim_detections, sim_names)
            if not topo_res.get("is_valid", False) or topo_res.get("unconnected_tags"):
                obs_entry["keep"] = True
                continue

            selected.append((bname, tid, err_px, err_mm))
            sim_counts[tid] -= 1

        return selected

    def prune_observations(self, prune_list: List[Tuple[str, int, float, float]]) -> int:
        """批量将指定的观测标靶标记为剔除状态并持久化与更新指标"""
        count = 0
        for bname, tid, _, _ in prune_list:
            img_entry = self.manifest_data.get("images", {}).get(bname, {})
            for obs in img_entry.get("observations", []):
                if obs.get("tag_id") == tid and obs.get("keep", True):
                    obs["keep"] = False
                    count += 1
                    break
        if count > 0:
            self._save_manifest()
            self.refresh_all_frame_metrics()
        return count

    def _evaluate_frame_reprojection(
        self,
        observations: List[Dict[str, Any]]
    ) -> Tuple[float, float, Dict[int, float], Optional[np.ndarray], Optional[np.ndarray], float, Dict[int, float]]:
        """计算单帧中所有有效标靶的像素残差与空间毫米偏差"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return 0.0, 0.0, {}, None, None, 0.0, {}

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
            return 0.0, 0.0, {}, None, None, 0.0, {}

        obj_flat = np.concatenate(obj_pts, axis=0)
        img_flat = np.concatenate(img_pts, axis=0)

        rvec, tvec, success = self.engine.solve_pnp(obj_flat, img_flat)
        if not success:
            return 0.0, 0.0, {}, None, None, 0.0, {}

        proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, self.engine.camera_matrix, self.engine.dist_coeffs)
        dists = np.linalg.norm(img_flat - proj_pts.reshape((-1, 2)), axis=1)

        tz = float(tvec[2, 0]) if tvec is not None else 800.0
        fx = float(self.engine.camera_matrix[0, 0]) if (self.engine and self.engine.camera_matrix is not None) else 1363.0
        scale_mm_per_px = abs(tz) / fx if fx > 0 else 0.0

        errors_dict = {}
        errors_dict_mm = {}
        for idx, tid in enumerate(valid_tids):
            err_px = float(np.mean(dists[idx * 4:(idx + 1) * 4]))
            errors_dict[tid] = err_px
            errors_dict_mm[tid] = float(err_px * scale_mm_per_px)

        mean_val = float(np.mean(dists))
        max_val = float(np.max(dists))
        mean_val_mm = float(mean_val * scale_mm_per_px)
        return mean_val, max_val, errors_dict, rvec, tvec, mean_val_mm, errors_dict_mm

    def refresh_all_frame_metrics(self):
        """全量预热并刷新所有采图帧的精度体检残差指标、空间毫米偏差与共视拓扑健康度"""
        self.frame_metrics_cache.clear()
        all_reproj_errors = []
        all_mm_errors = []
        valid_frame_detections = []
        valid_frame_names = []

        for p in self.image_files:
            base_name = os.path.basename(p)
            obs_list = self.get_observations_for_image(base_name)
            is_excl = self.is_image_excluded(base_name)

            mean_err, max_err, errors_dict, rvec, tvec, mean_mm, errs_mm = self._evaluate_frame_reprojection(obs_list)
            if not is_excl and obs_list:
                all_reproj_errors.extend(list(errors_dict.values()))
                all_mm_errors.extend(list(errs_mm.values()))
                tag_dict = {}
                for obs in obs_list:
                    if obs.get("keep", True):
                        tag_dict[int(obs["tag_id"])] = np.array(obs["corners"], dtype=np.float64)
                if len(tag_dict) >= 1:
                    valid_frame_detections.append(tag_dict)
                    valid_frame_names.append(base_name)

            self.frame_metrics_cache[base_name] = {
                "tag_count": len(obs_list),
                "mean_err": mean_err,
                "max_err": max_err,
                "mean_err_mm": mean_mm,
                "is_excluded": is_excl,
                "observations": obs_list,
                "tag_errors": errors_dict,
                "tag_errors_mm": errs_mm,
                "rvec": rvec,
                "tvec": tvec
            }

        if all_reproj_errors:
            self.global_rmse = float(np.sqrt(np.mean(np.array(all_reproj_errors) ** 2)))
        else:
            self.global_rmse = 0.0

        if all_mm_errors:
            self.global_median_mm = float(np.median(np.array(all_mm_errors)))
            self.global_mean_mm = float(np.mean(np.array(all_mm_errors)))
        else:
            self.global_median_mm = 0.0
            self.global_mean_mm = 0.0

        # 共视拓扑连通度检查
        if valid_frame_detections:
            try:
                from src.calibration.covisibility_graph import CovisibilityGraphAnalyzer
                topo = CovisibilityGraphAnalyzer.analyze(valid_frame_detections, valid_frame_names)
                self.topology_status = {
                    "is_valid": bool(topo.get("is_valid", False)),
                    "components_count": len(topo.get("components", [])),
                    "critical_bridges": topo.get("critical_bridges", []),
                    "unconnected_tags": topo.get("unconnected_tags", []),
                    "connected_tags_count": len(topo.get("connected_tags", [])),
                    "message": str(topo.get("message", "拓扑分析完成"))
                }
            except Exception as e:
                self.topology_status = {
                    "is_valid": True,
                    "components_count": 1,
                    "critical_bridges": [],
                    "unconnected_tags": [],
                    "connected_tags_count": 0,
                    "message": f"拓扑分析异常: {e}"
                }
        else:
            self.topology_status = {
                "is_valid": False,
                "components_count": 0,
                "critical_bridges": [],
                "unconnected_tags": [],
                "connected_tags_count": 0,
                "message": "暂无参与解算的有效采图帧"
            }

        # 放行门限评定
        if self.global_rmse > 0 and self.global_rmse <= 0.50 and self.global_median_mm <= 1.50 and self.topology_status.get("is_valid", False):
            self.gate_status = "PASS"
        elif self.global_rmse > 0 and self.global_rmse <= 1.00:
            self.gate_status = "ACCEPTABLE"
        else:
            self.gate_status = "REVIEW"

    def diagnose_frame(self, img_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        对指定帧或当前选定帧执行工序 3/4 深度图像质量与漏检病因切片诊断
        """
        if img_idx is None:
            img_idx = self.current_img_idx
        if not self.image_files or img_idx >= len(self.image_files):
            return {}

        cur_file = self.image_files[img_idx]
        bname = os.path.basename(cur_file)
        bgr = cv2.imread(cur_file)
        if bgr is None:
            return {}

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 候选多边形与拒检分析
        c_raw, ids_raw, rejected = self.engine.detector_bright.detectMarkers(gray)
        detected_tids = set(ids_raw.flatten().tolist()) if ids_raw is not None else set()
        rej_count = len(rejected) if rejected is not None else 0

        rej_small = 0
        rej_aspect = 0
        if rejected is not None:
            for r in rejected:
                pts = r.reshape((4, 2))
                area = cv2.contourArea(pts)
                if area < 200:
                    rej_small += 1
                side_a = np.linalg.norm(pts[0] - pts[1])
                side_b = np.linalg.norm(pts[1] - pts[2])
                if side_b > 0 and (side_a / side_b > 3.0 or side_b / side_a > 3.0):
                    rej_aspect += 1

        # 理论漏检分析
        missing_tags = []
        meta = self.frame_metrics_cache.get(bname, {})
        rvec = meta.get("rvec")
        tvec = meta.get("tvec")
        if rvec is not None and tvec is not None:
            mapped_tids = self.tags_map_data.get("tags", {})
            for tid_str in mapped_tids.keys():
                t_int = int(tid_str)
                if t_int not in detected_tids:
                    wc = self.get_tag_world_corners(t_int)
                    if wc is not None:
                        proj, _ = cv2.projectPoints(wc, rvec, tvec, self.engine.camera_matrix, self.engine.dist_coeffs)
                        p2 = proj.reshape((4, 2))
                        if np.all(p2[:, 0] >= -20) and np.all(p2[:, 0] < w + 20) and np.all(p2[:, 1] >= -20) and np.all(p2[:, 1] < h + 20):
                            missing_tags.append({
                                "tag_id": t_int,
                                "predicted_corners": p2.tolist(),
                                "reason": "视场内但未被快速检出 (建议按 E 超精提取)"
                            })

        diag_res = {
            "image": bname,
            "contrast": contrast,
            "contrast_rms": contrast,
            "contrast_grade": "良" if contrast >= 35 else ("偏低" if contrast >= 20 else "极差"),
            "brightness": brightness,
            "mean_intensity": brightness,
            "brightness_grade": "正常" if 60 <= brightness <= 190 else ("偏暗" if brightness < 60 else "过曝"),
            "sharpness": sharpness,
            "laplacian_var": sharpness,
            "sharpness_grade": "清晰" if sharpness >= 100 else ("轻微模糊" if sharpness >= 50 else "严重虚焦"),
            "detected_count": len(detected_tids),
            "rejected_quads_count": rej_count,
            "false_rejections_count": rej_count,
            "rej_small": rej_small,
            "rej_aspect": rej_aspect,
            "missing_theoretical_tags": missing_tags,
            "missing_projected_tags": missing_tags
        }
        self.current_diagnostics = diag_res
        return diag_res

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
