#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Studio 异步工作流 Mixin (studio_workflows.py)
=============================================
从 app.py 拆分出的长耗时工作流调度职责模块，由 TagOfflineStudio 以 Mixin 方式继承：
  - 全量超精重提取的异步调度与结果轮询 (start_async_super_extract_all / poll_super_extract_result)
  - 智能剪枝平差的启动 / 采纳 / 撤销 (start_auto_prune_ba / accept_prune_results / undo_prune_results)
  - 异步全局 BA 平差启动 (start_async_bundle_adjustment)
  - 生产地图发布与全景质检报告导出 (publish_to_production / export_verification_report)
无 __init__、无新增实例属性，全部通过宿主 self 与主控制器协作。
"""

import os
import time
import threading
from typing import Optional, Tuple

from src.calibration.manifest_repository import ManifestRepository
from tools.studio.studio_app_meta import PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger(__name__)


class StudioWorkflowMixin:

    """异步工作流调度职责 (无状态，依赖宿主 self 属性)"""

    def start_async_super_extract_all(self) -> bool:
        """启动后台异步线程执行全量采图工序 3 工业级超精重提取并从头重建"""
        if self.is_extracting_all:
            self.set_toast("全量超精提取已在后台运行中，请稍候...")
            return False
        if self.is_ba_running:
            self.set_toast("全局平差计算中，请待平差完成后再执行提取")
            return False
        if not self.image_files:
            self.set_toast("未扫描到采图文件，无法执行超精重提取")
            return False

        self.is_extracting_all = True
        self.extract_progress = 0.01
        self.extract_stage_text = f"正在启动全局全量超精提取 (共 {len(self.image_files)} 帧)..."
        self.set_toast(self.extract_stage_text)

        def _worker():
            try:
                def on_progress(cur, total, bname, count):
                    self.extract_progress = cur / max(1, total)
                    self.extract_stage_text = f"全量超精提取 ({cur}/{total}): {bname} (检出 {count} 个标靶)"

                total_frames, total_tags = self.data_mgr.super_extract_all_frames(progress_callback=on_progress)
                self.extract_progress = 1.0
                msg = f"全局超精提取完成！处理 {total_frames} 帧，累计提取 {total_tags} 个高精标靶"
                self.extract_result_queue = (True, msg)
            except Exception as e:
                self.extract_result_queue = (False, f"全量超精重提取失败: {e}")

        self.extract_thread = threading.Thread(target=_worker, daemon=True)
        self.extract_thread.start()
        return True

    def poll_super_extract_result(self) -> Optional[Tuple[bool, str]]:
        """检查异步全量超精提取任务是否完成"""
        if self.extract_result_queue is not None:
            res = self.extract_result_queue
            self.extract_result_queue = None
            self.is_extracting_all = False
            return res
        return None

    def start_auto_prune_ba(self) -> bool:
        """启动全自动基于边际收益与共视拓扑守门的残差剪枝平差"""
        if self.is_ba_running or self.is_extracting_all:
            self.set_toast("后台任务正在计算中，请稍候...")
            return False
        # 联动质检视角: 自动将左侧图像序列切换为【残差降序 (最差优先 ↓)】并展开多轮残差矩阵视图
        self.sort_mode = "err_desc"
        self.matrix_view_mode = True
        self.left_bar_w = self.dynamic_left_bar_w
        res = self.ba_runner.start_auto_prune(max_rounds=10, min_improvement_px=0.01)
        if res:
            # 自动将主视口聚焦至残差最大、最亟待排查的首张图像
            f_indices = self._get_filtered_indices()
            if f_indices:
                self.current_img_idx = f_indices[0]
        return res

    def accept_prune_results(self):
        """采纳智能剪枝平差结果并清空结算单"""
        self.ba_runner.prune_settlement_data = None
        self.set_toast("已采纳智能剪枝平差结果！可按 [M] 保存为最新地图")
        log.info("[*] [STUDIO] 操作员确认采纳智能剪枝平差结果。")

    def undo_prune_results(self):
        """一键无损撤销智能剪枝，回滚至快照状态"""
        succ = self.data_mgr.restore_manifest_snapshot()
        self.ba_runner.prune_settlement_data = None
        if succ:
            self.set_toast("已撤销智能剪枝！观测清单与地图已完全恢复至剪枝前状态")
            log.warning("[*] [STUDIO] 操作员已撤销智能剪枝，状态已无损回滚。")
        else:
            self.set_toast("未找到有效快照，撤销未执行")

    def start_async_bundle_adjustment(self):
        """启动后台线程执行两阶段全局 BA 平差优化，前台持续平滑响应"""
        return self.ba_runner.start()

    def publish_to_production(self):
        """将当前工作站优化好的地图一键发布至全局生产环境 (config/tags_map.yaml)"""
        if not self.tags_map_data:
            self.set_toast("当前尚无有效地图，请先按 [B] 进行 BA 平差！")
            return

        ManifestRepository.save_map(self.tags_map_data, self.map_path)
        target_sc = self.current_scene
        if self.scene_mgr and target_sc:
            ok, msg = self.scene_mgr.publish_to_production(target_sc.workspace_id)
            if ok:
                self.set_toast(f"★ 成功将【{target_sc.name}】发布为生产全局地图！")
            else:
                self.set_toast(f"发布失败: {msg}")
        else:
            self.set_toast("未连接工位管理器，已保存至本工位地图")

    def export_verification_report(self):
        """导出 Markdown 全景精度质检单"""
        if self.current_scene:
            report_dir = self.current_scene.calib_reports_dir
        else:
            report_dir = os.path.join(PROJECT_ROOT, "data", "tag_calibration_verification")
        os.makedirs(report_dir, exist_ok=True)
        ts = int(time.time())
        report_path = os.path.join(report_dir, f"studio_qa_report_{ts}.md")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# AprilTag 离线标定与建图全景质检单 (Offline Studio)\n\n")
                f.write(f"- **质检时间**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
                f.write(f"- **总采图集**: `{len(self.image_files)} 帧`\n")
                f.write(f"- **全景 RMSE**: `{self.global_rmse:.3f} px`\n")
                f.write(f"- **已知标靶数**: `{len(self.tags_map_data.get('tags', {}))} 个`\n")
                f.write(f"- **空间地图**: `{self.map_path}`\n\n")
                f.write(f"## 图像帧逐项质检明细\n\n")
                f.write(f"| 图像帧 | 观测标靶数 | 平均残差 | 最大残差 | 状态 |\n")
                f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")

                for p in self.image_files:
                    bname = os.path.basename(p)
                    meta = self.frame_metrics_cache.get(bname, {})
                    status_str = "❌ 已剔除" if meta.get("is_excluded", False) else "✅ 参与解算"
                    f.write(f"| `{bname}` | {meta.get('tag_count', 0)} | {meta.get('mean_err', 0.0):.2f} px | {meta.get('max_err', 0.0):.2f} px | {status_str} |\n")

                # 2. 智能剪枝平差逐帧多轮残差收敛矩阵 (若存在多轮历史)
                headers = getattr(self.data_mgr, "convergence_headers", [])
                matrix = getattr(self.data_mgr, "frame_convergence_matrix", {})
                if headers and matrix and len(headers) >= 1:
                    f.write(f"\n## 2. 智能剪枝平差逐帧多轮残差收敛矩阵 (Per-Frame Convergence Matrix)\n\n")
                    f.write(f"> 记录各图像帧在每一轮平差求解后的残差演进变化情况：\n\n")
                    header_cols = ["图像帧", "标靶数"] + headers + ["累计降幅"]
                    f.write("| " + " | ".join(header_cols) + " |\n")
                    f.write("| " + " | ".join([":---"] + [":---:"] * (len(header_cols) - 1)) + " |\n")

                    for p in self.image_files:
                        bname = os.path.basename(p)
                        meta = self.frame_metrics_cache.get(bname, {})
                        tag_cnt = meta.get("tag_count", 0)
                        row_vals = matrix.get(bname, [])
                        r_strs = []
                        for val in row_vals:
                            r_strs.append(f"{val:.2f} px" if val is not None else "--")
                        while len(r_strs) < len(headers):
                            r_strs.append("--")

                        first_val = row_vals[0] if (row_vals and row_vals[0] is not None) else None
                        last_val = None
                        for v in reversed(row_vals):
                            if v is not None:
                                last_val = v
                                break
                        if first_val is not None and last_val is not None and first_val > 0.001:
                            drop_px = first_val - last_val
                            drop_pct = (drop_px / first_val) * 100.0
                            drop_str = f"↓{drop_pct:.1f}% ({drop_px:+.2f}px)"
                        else:
                            drop_str = "--"

                        f.write(f"| `{bname}` | {tag_cnt} | " + " | ".join(r_strs) + f" | {drop_str} |\n")

            self.set_toast("全景质检报告已成功导出至 data/tag_calibration_verification/！")
            log.info(f"[OK] 质检报告导出成功: {report_path}")
        except Exception as e:
            self.set_toast(f"导出质检报告失败: {e}")
            log.warning(f"导出质检报告异常: {e}")
