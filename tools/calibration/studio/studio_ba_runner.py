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
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.calibration.ba_optimizer import BundleAdjustmentOptimizer
from src.calibration.manifest_repository import ManifestRepository
from tools.calibration.studio.studio_state import StudioDataManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


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

        # 智能剪枝平差专属状态
        self.is_auto_pruning: bool = False
        self.prune_round: int = 0
        self.max_prune_rounds: int = 10
        self.min_improvement_px: float = 0.01
        self.should_stop_pruning: bool = False
        self.prune_history: List[Dict[str, Any]] = []
        self.prune_settlement_data: Optional[Dict[str, Any]] = None
        self.current_pruning_target: str = ""

        # 世界系对齐锚定配置 (从 config.yaml 动态加载，杜绝幽灵 Tag 1)
        self.origin_tag_id: int = 0
        self.x_align_tag_id: int = 28
        self.world_anchor: Optional[Dict[str, Any]] = None
        self._load_alignment_config()

    def _load_alignment_config(self):
        try:
            import yaml
            cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    c = yaml.safe_load(f) or {}
                calib = c.get("calibration", {})
                self.origin_tag_id = int(calib.get("origin_tag_id", 0))
                self.x_align_tag_id = int(calib.get("x_axis_tag_id", 28))
                # FR-9.6 世界系绝对锚定 (Tag0/Tag1 已知机械臂坐标, 配置缺失时退化为相对对齐)
                wa = calib.get("world_anchor")
                if isinstance(wa, dict) and wa.get("origin_xyz_mm") and wa.get("align_xyz_mm"):
                    self.world_anchor = {
                        "origin_tag_id": int(wa.get("origin_tag_id", 0)),
                        "origin_xyz_mm": [float(v) for v in wa["origin_xyz_mm"]],
                        "align_tag_id": int(wa.get("align_tag_id", 1)),
                        "align_xyz_mm": [float(v) for v in wa["align_xyz_mm"]]
                    }
        except Exception as e:
            print(f"[WARN] [STUDIO] 读取对齐标靶配置异常，采用默认值 (0, 28): {e}")

    def _notify(self, msg: str):
        if self.on_status_change is not None:
            try:
                self.on_status_change(msg)
            except Exception:
                pass

    def _execute_ba_solve(self, callback: Optional[Any] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """执行单次核心两阶段 BA 平差求解计算并保存完整地图 Schema"""
        frame_detections, valid_frame_names, _ = self.manifest_repo.load_manifest(self.manifest_path)
        if len(frame_detections) < 2:
            return False, None, "有效图像帧不足 2 帧，无法执行 BA 平差"

        opt_res = self.optimizer.optimize(
            frame_detections=frame_detections,
            active_frame_names=valid_frame_names,
            origin_tag_id=self.origin_tag_id,
            x_align_tag_id=self.x_align_tag_id,
            world_anchor=self.world_anchor,
            callback=callback
        )
        if opt_res and "tags" in opt_res:
            # 完整继承优化器产出的全量标准 schema (保留 is_dynamic_yaw, is_origin, rpy_deg 等)
            raw_tags = opt_res.get("tags", {})
            tags_dict = {}
            for tid, t_info in raw_tags.items():
                tags_dict[int(tid)] = {
                    "transform_matrix": t_info.get("transform_matrix"),
                    "position_mm": t_info.get("position_mm"),
                    "rpy_deg": t_info.get("rpy_deg", [0.0, 0.0, 0.0]),
                    "is_origin": bool(t_info.get("is_origin", False)),
                    "is_dynamic_yaw": bool(t_info.get("is_dynamic_yaw", False))
                }
            new_map = {
                "origin_tag_id": opt_res.get("origin_tag_id", self.origin_tag_id),
                "x_axis_align_tag_id": opt_res.get("x_axis_align_tag_id", self.x_align_tag_id),
                "marker_size_mm": opt_res.get("marker_size_mm", self.marker_size_mm),
                "rmse_px": opt_res.get("final_rmse", 0.0),
                "rmse_reprojection_px": opt_res.get("rmse_reprojection_px", opt_res.get("final_rmse", 0.0)),
                "calibrated_images_count": opt_res.get("calibrated_images_count", len(valid_frame_names)),
                "tags": tags_dict
            }
            if opt_res.get("world_anchor"):
                new_map["world_anchor"] = opt_res["world_anchor"]
            ManifestRepository.save_map(new_map, self.map_path)
            self.data_mgr.tags_map_data = new_map
            if self.data_mgr.engine:
                self.data_mgr.engine.tags_map = new_map
            # 同步 BA 反算的真实边长到引擎模型, 保证理论/实测棱柱比例与空间偏差解算一致
            if new_map.get("marker_size_mm"):
                self.data_mgr.set_marker_size_mm(new_map["marker_size_mm"])
            self.data_mgr.refresh_all_frame_metrics()
            return True, opt_res, f"平差收敛成功，全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px"
        return False, None, "BA 优化未能收敛，请检查有效观测标靶数"

    def start(self) -> bool:
        """
        启动后台线程执行两阶段全局 BA 平差优化
        :return: True 如果成功启动，False 如果已有任务在运行
        """
        if self.is_ba_running:
            self._notify("平差优化已在运行中，请稍候...")
            return False

        self.is_ba_running = True
        self.is_auto_pruning = False
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

                def on_ba_callback(info: Dict[str, Any]):
                    stg = info.get("stage", 1)
                    stg_name = info.get("stage_name", "")
                    cur_it = info.get("iter", 0)
                    max_it = info.get("max_iter", 1)
                    cur_rmse = info.get("rmse", 0.0)
                    sub_pct = max(0.0, min(1.0, info.get("sub_progress", 0.0)))

                    if stg == 1:
                        self.ba_progress = 0.40 + 0.22 * sub_pct
                    else:
                        self.ba_progress = 0.62 + 0.22 * sub_pct

                    self.ba_sub_progress = sub_pct
                    self.ba_sub_text = f"[{stg_name}] 轮次 #{cur_it}/{max_it} | 实时 RMSE: {cur_rmse:.3f} px"

                self.ba_progress = 0.40
                self.ba_stage_text = "阶段 3/4: 两阶段 Cauchy 稳健核平差全局收敛求解..."
                succ, opt_res, msg = self._execute_ba_solve(callback=on_ba_callback)

                if succ:
                    self.ba_progress = 1.0
                    self.ba_sub_progress = 1.0
                    self.ba_stage_text = "全局平差完成！正在同步就地热更新..."
                    self.ba_sub_text = f"最终全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px (已就地生效)"
                    self.ba_result_queue = (True, f"BA 优化成功！新全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px")
                else:
                    self.ba_result_queue = (False, msg)
            except Exception as e:
                self.ba_result_queue = (False, f"BA 平差优化异常: {e}")

        self.ba_thread = threading.Thread(target=_worker, daemon=True)
        self.ba_thread.start()
        return True

    def start_auto_prune(self, max_rounds: int = 10, min_improvement_px: float = 0.01) -> bool:
        """
        启动后台线程执行基于边际收益收敛与共视拓扑守门的全自动迭代剪枝平差
        """
        if self.is_ba_running:
            self._notify("平差优化计算中，请稍候...")
            return False

        # 启动前自动制作状态快照
        self.data_mgr.create_manifest_snapshot()
        self.data_mgr.sort_mode = "err_desc"

        self.is_ba_running = True
        self.is_auto_pruning = True
        self.should_stop_pruning = False
        self.prune_round = 0
        self.max_prune_rounds = max_rounds
        self.min_improvement_px = min_improvement_px
        self.prune_history = []
        self.prune_settlement_data = None
        self.current_pruning_target = ""
        self.initial_rmse = self.data_mgr.global_rmse

        self.ba_progress = 0.05
        self.ba_sub_progress = 0.0
        self.ba_stage_text = "正在启动工序 5-Auto: 迭代残差剪枝平差..."
        self.ba_sub_text = "创建状态快照并准备首轮基准平差..."
        self._notify("智能剪枝平差启动: 已制作状态快照")
        print("\n[*] [STUDIO AUTO-PRUNE] 启动自动迭代残差剪枝平差...")

        def _auto_prune_worker():
            try:
                # 1. 确保有初始有效基准 RMSE
                init_rmse = self.data_mgr.global_rmse
                init_mm = self.data_mgr.global_median_mm
                if init_rmse <= 0.0:
                    self.ba_stage_text = "初始化: 运行基准平差获取初始误差网格..."
                    succ, _, msg = self._execute_ba_solve()
                    if not succ:
                        self.ba_result_queue = (False, f"初始基准平差失败: {msg}")
                        self.is_auto_pruning = False
                        return
                    init_rmse = self.data_mgr.global_rmse
                    init_mm = self.data_mgr.global_median_mm

                self.initial_rmse = init_rmse
                prev_rmse = init_rmse
                stop_reason = "达到最大预设轮数"

                # 初始化逐帧多轮残差收敛矩阵: 首列为 R0(基准)
                self.data_mgr.convergence_headers = ["R0"]
                self.data_mgr.frame_convergence_matrix = {}
                for p in self.data_mgr.image_files:
                    bn = os.path.basename(p)
                    meta = self.data_mgr.frame_metrics_cache.get(bn, {})
                    err = meta.get("mean_err", 0.0)
                    is_excl = meta.get("is_excluded", False)
                    val = round(float(err), 2) if (not is_excl and err is not None) else None
                    self.data_mgr.frame_convergence_matrix[bn] = [val]

                # 2. 迭代剪枝主循环
                for r in range(1, self.max_prune_rounds + 1):
                    if self.should_stop_pruning:
                        stop_reason = "用户手动急停"
                        break

                    self.prune_round = r
                    self.ba_progress = min(0.95, 0.10 + 0.85 * (r / self.max_prune_rounds))
                    self.ba_stage_text = f"智能剪枝平差 (第 {r}/{self.max_prune_rounds} 轮): 寻找最大离差安全样本..."

                    # 寻找本轮最大离差的 1~2 个安全标靶
                    prunable = self.data_mgr.find_worst_prunable_observations(top_k=2)
                    if not prunable:
                        stop_reason = "拓扑安全守门触发: 已无安全可剔除标靶"
                        print(f"[*] [AUTO-PRUNE] 轮次 #{r}: 已无安全可剔除项，安全停机。")
                        break

                    # 格式化剔除描述
                    p_desc = ", ".join([f"{bname} 的 #{tid} ({err:.2f}px)" for bname, tid, err, _ in prunable])
                    self.current_pruning_target = p_desc
                    self.ba_stage_text = f"智能剪枝平差 (第 {r}/{self.max_prune_rounds} 轮): 正在重平差求解全场景优化..."
                    self.ba_sub_text = f"本轮淘汰: {p_desc} -> 全局 BA 优化求解中..."
                    print(f"\n[======== [AUTO-PRUNE] 剪枝平差 第 {r}/{self.max_prune_rounds} 轮 ========]", flush=True)
                    print(f"[*] 拟剔除坏样本: {p_desc}", flush=True)
                    print(f"[*] 剔除前全局基准 RMSE: {prev_rmse:.3f} px (中位数重投影误差: {self.data_mgr.global_median_mm:.3f} mm)", flush=True)

                    # 轮次前制作即时快照 (用于防反弹单调保护)
                    round_snapshot = self.data_mgr.create_manifest_snapshot()

                    # 执行剔除
                    self.data_mgr.prune_observations(prunable)

                    # 重新运行平差求解
                    print(f"[*] 正在重平差求解全场景优化...", flush=True)
                    succ, _, msg = self._execute_ba_solve()
                    self.current_pruning_target = ""
                    if not succ:
                        stop_reason = f"平差求解发散: {msg} (已自动恢复本轮前状态)"
                        print(f"[!] [AUTO-PRUNE] 求解发散: {msg}，自动回滚本轮剔除并停机。", flush=True)
                        self.data_mgr.restore_manifest_snapshot(round_snapshot)
                        break

                    new_rmse = self.data_mgr.global_rmse
                    new_mm = self.data_mgr.global_median_mm
                    delta_rmse = prev_rmse - new_rmse

                    # 防反弹单调刚性保护: 若剔除导致误差反弹上升 (delta_rmse < 0)，说明该标靶是关键拓扑支撑点，自动回滚并锁定最优停机
                    if delta_rmse < -0.01:
                        stop_reason = f"触发刚性拓扑保护: 拟淘汰标靶属于关键支撑骨架，剔除后残差反弹 (+{abs(delta_rmse):.2f}px)，已自动回滚并锁定最优收敛状态"
                        print(f"\n[!] [AUTO-PRUNE] 警告: 本轮剔除导致平差残差反弹 (从 {prev_rmse:.3f} px 恶化至 {new_rmse:.3f} px)!", flush=True)
                        print(f"[*] 正在自动回滚撤销本轮剔除，并精准恢复至最优地图状态...", flush=True)
                        self.data_mgr.restore_manifest_snapshot(round_snapshot)
                        print(f"[✓] 已安全恢复至最优 RMSE: {prev_rmse:.3f} px，自动触发最优收敛停机！\n", flush=True)
                        break

                    # 本轮求解成功且有效，横向自动增加一列记录各图像在求解后的最新残差
                    self.data_mgr.convergence_headers.append(f"R{r}")
                    for p in self.data_mgr.image_files:
                        bn = os.path.basename(p)
                        meta = self.data_mgr.frame_metrics_cache.get(bn, {})
                        err = meta.get("mean_err", 0.0)
                        is_excl = meta.get("is_excluded", False)
                        val = round(float(err), 2) if (not is_excl and err is not None) else None
                        if bn not in self.data_mgr.frame_convergence_matrix:
                            self.data_mgr.frame_convergence_matrix[bn] = []
                        self.data_mgr.frame_convergence_matrix[bn].append(val)

                    self.prune_history.append({
                        "round": r,
                        "pruned": prunable,
                        "rmse_before": prev_rmse,
                        "rmse_after": new_rmse,
                        "delta_rmse": delta_rmse,
                        "median_mm": new_mm
                    })

                    print(f"[✓] 轮次 #{r} 完成: RMSE 从 {prev_rmse:.3f} px -> {new_rmse:.3f} px (改善幅度: {delta_rmse:+.3f} px)", flush=True)

                    # 核心终止判定: 边际收益见顶
                    if delta_rmse < self.min_improvement_px:
                        stop_reason = f"边际收益见顶 (本轮改善 {delta_rmse:.4f} px < 门限 {self.min_improvement_px:.3f} px)"
                        print(f"[*] [AUTO-PRUNE] 改善幅度低于边际门限，已达最优收敛状态，自动停机！", flush=True)
                        break

                    prev_rmse = new_rmse

                # 3. 生成结算对比卡片数据包
                import copy
                final_rmse = self.data_mgr.global_rmse
                final_mm = self.data_mgr.global_median_mm
                total_pruned = sum(len(item["pruned"]) for item in self.prune_history)

                self.prune_settlement_data = {
                    "initial_rmse": init_rmse,
                    "initial_mm": init_mm,
                    "final_rmse": final_rmse,
                    "final_mm": final_mm,
                    "rounds_executed": len(self.prune_history),
                    "total_pruned_count": total_pruned,
                    "stop_reason": stop_reason,
                    "history": self.prune_history,
                    "convergence_headers": list(self.data_mgr.convergence_headers),
                    "frame_convergence_matrix": copy.deepcopy(self.data_mgr.frame_convergence_matrix)
                }

                self.ba_progress = 1.0
                self.ba_sub_progress = 1.0
                self.ba_stage_text = "智能剪枝平差完成！请在结算卡片中确认采纳或撤销"
                self.ba_sub_text = f"优化成效: RMSE {init_rmse:.2f}px -> {final_rmse:.2f}px (剔除 {total_pruned} 个外点)"
                self.ba_result_queue = (True, f"智能剪枝平差结束: {stop_reason}，累计剔除 {total_pruned} 个外点")
                print("\n==================================================", flush=True)
                print(f"[★] [AUTO-PRUNE] 智能剪枝平差全流程完成！", flush=True)
                print(f"    - 执行轮次: {len(self.prune_history)} 轮", flush=True)
                print(f"    - 累计剔除外点: {total_pruned} 个", flush=True)
                print(f"    - 全局 RMSE: {init_rmse:.3f} px  ==>  {final_rmse:.3f} px", flush=True)
                print(f"    - 停止原因: {stop_reason}", flush=True)
                print("==================================================\n", flush=True)
            except Exception as e:
                import traceback
                print(f"\n[!] [AUTO-PRUNE ERROR] 智能剪枝平差发生未捕获异常: {e}", flush=True)
                traceback.print_exc()
                self.ba_result_queue = (False, f"智能剪枝平差异常: {e}")
            finally:
                self.is_auto_pruning = False

        self.ba_thread = threading.Thread(target=_auto_prune_worker, daemon=True)
        self.ba_thread.start()
        return True

    def request_stop_pruning(self):
        """请求中途安全急停智能剪枝平差"""
        if self.is_auto_pruning:
            self.should_stop_pruning = True
            self._notify("已发送急停请求，将在当前平差轮次完成后安全停止...")

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
