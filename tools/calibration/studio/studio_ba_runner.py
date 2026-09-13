"""
AprilTag 离线标定工作站 - 异步 BA 平差调度器 (StudioBARunner)
================================================================================
负责工作站中高耗时计算任务的生命周期调度与状态通知：
1. 后台异步线程执行两阶段 Cauchy 稳健核平差优化 (Bundle Adjustment)
2. 细粒度双进度条推进 (大阶段全局进度 ba_progress + 求解器迭代子进度 ba_sub_progress)
3. 实时迭代收敛指标监听与文本反馈 (轮次、迭代 RMSE)
4. 平差后空间立体地图就地热更新与持久化保存
"""

import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from src.calibration.ba_optimizer import BundleAdjustmentOptimizer
from src.calibration.manifest_repository import ManifestRepository
from tools.calibration.studio.studio_state import StudioDataManager


class StudioBARunner:
    """异步 BA 平差执行器与进度调度器"""

    def __init__(
        self,
        data_mgr: StudioDataManager,
        optimizer: BundleAdjustmentOptimizer,
        manifest_repo: ManifestRepository,
        map_path: str,
        manifest_path: str,
        marker_size_mm: float = 50.0,
        on_status_change: Optional[Callable[[str], None]] = None
    ):
        self.data_mgr = data_mgr
        self.optimizer = optimizer
        self.manifest_repo = manifest_repo
        self.map_path = map_path
        self.manifest_path = manifest_path
        self.marker_size_mm = marker_size_mm
        self.on_status_change = on_status_change

        # 运行状态与指标
        self.is_ba_running: bool = False
        self.ba_thread: Optional[threading.Thread] = None
        self.ba_result_queue: Optional[Tuple[bool, str]] = None
        self.ba_progress: float = 0.0
        self.ba_stage_text: str = ""
        self.ba_sub_progress: float = 0.0
        self.ba_sub_text: str = ""

    def _notify(self, msg: str):
        if self.on_status_change is not None:
            try:
                self.on_status_change(msg)
            except Exception:
                pass

    def start(self) -> bool:
        """
        启动后台线程执行两阶段全局 BA 平差优化
        :return: True 如果成功启动，False 如果已有任务在运行
        """
        if self.is_ba_running:
            self._notify("BA 全局平差优化已在运行中，请稍候...")
            return False

        self.is_ba_running = True
        self.ba_progress = 0.05
        self.ba_sub_progress = 0.0
        self.ba_stage_text = "正在启动两阶段全局 BA 平差优化计算..."
        self.ba_sub_text = "初始化优化工作空间..."
        self._notify("正在启动两阶段全局 BA 平差优化计算...")
        print("\n[*] [STUDIO] 正在启动异步 BA 全局平差优化计算...")

        def _worker():
            try:
                self.ba_progress = 0.10
                self.ba_sub_progress = 0.30
                self.ba_stage_text = "阶段 1/4: 准备观测清单与共视拓扑分析..."
                self.ba_sub_text = "校验数据清单并分析共视连通性图..."
                self.data_mgr._save_manifest()
                time.sleep(0.05)

                self.ba_progress = 0.25
                self.ba_sub_progress = 0.60
                self.ba_stage_text = "阶段 2/4: 过滤已剔除样本并构建全局初值..."
                self.ba_sub_text = "构建超定 PnP 初始机位与标靶三维姿态..."
                frame_detections, valid_frame_names, _ = self.manifest_repo.load_manifest(self.manifest_path)
                if len(frame_detections) < 2:
                    self.ba_result_queue = (False, "有效图像帧不足 2 帧，无法执行 BA 平差")
                    return

                self.ba_progress = 0.40
                self.ba_sub_progress = 0.0
                self.ba_stage_text = "阶段 3/4: 两阶段 Cauchy 稳健核平差全局收敛求解..."
                self.ba_sub_text = "准备启动非线性平差求解器..."

                def on_ba_callback(info: Dict[str, Any]):
                    stg = info.get("stage", 1)
                    stg_name = info.get("stage_name", "")
                    cur_it = info.get("iter", 0)
                    max_it = info.get("max_iter", 1)
                    cur_rmse = info.get("rmse", 0.0)
                    sub_pct = max(0.0, min(1.0, info.get("sub_progress", 0.0)))

                    # 全局大进度条在阶段 3/4 期间平滑推进：阶段一占 40%~62%，阶段二占 62%~84%
                    if stg == 1:
                        self.ba_progress = 0.40 + 0.22 * sub_pct
                    else:
                        self.ba_progress = 0.62 + 0.22 * sub_pct

                    self.ba_sub_progress = sub_pct
                    self.ba_sub_text = f"[{stg_name}] 轮次 #{cur_it}/{max_it} | 实时 RMSE: {cur_rmse:.3f} px"

                opt_res = self.optimizer.optimize(frame_detections, valid_frame_names, callback=on_ba_callback)

                self.ba_progress = 0.88
                self.ba_sub_progress = 0.50
                self.ba_stage_text = "阶段 4/4: 重映射空间立体地图与外参反算..."
                self.ba_sub_text = "对齐世界原点并写入立体几何地图缓存..."
                if opt_res and "final_tag_poses_aligned" in opt_res:
                    tags_dict = {}
                    for tid, T in opt_res["final_tag_poses_aligned"].items():
                        tags_dict[tid] = {
                            "transform_matrix": T.tolist(),
                            "position_mm": T[:3, 3].tolist()
                        }
                    new_map = {
                        "marker_size_mm": self.marker_size_mm,
                        "tags": tags_dict,
                        "rmse_px": opt_res.get("final_rmse", 0.0)
                    }
                    ManifestRepository.save_map(new_map, self.map_path)
                    self.data_mgr.tags_map_data = new_map
                    if self.data_mgr.engine:
                        self.data_mgr.engine.tags_map = new_map

                    self.ba_progress = 1.0
                    self.ba_sub_progress = 1.0
                    self.ba_stage_text = "全局平差完成！正在同步就地热更新..."
                    self.ba_sub_text = f"最终全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px (已就地生效)"
                    msg = f"BA 优化成功！新全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px (地图已就地热重载)"
                    self.ba_result_queue = (True, msg)
                else:
                    self.ba_result_queue = (False, "BA 优化未能收敛，请检查有效观测标靶数")
            except Exception as e:
                self.ba_result_queue = (False, f"BA 平差优化异常: {e}")

        self.ba_thread = threading.Thread(target=_worker, daemon=True)
        self.ba_thread.start()
        return True

    def poll_result(self) -> Optional[Tuple[bool, str]]:
        """
        检查异步 BA 任务是否完成
        :return: (is_success, message) 或 None
        """
        if self.ba_result_queue is not None:
            result = self.ba_result_queue
            self.ba_result_queue = None
            self.is_ba_running = False
            return result
        return None
