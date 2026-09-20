"""
技术路线 B1：基于垂直扫描与正负梯度极性配对的感知流水线 (Polarity Scanline Pipeline)
===================================================================================
核心原理:
  1. 传送带核心 ROI 裁切 + 保边双边滤波与微小开运算去噪
  2. Sobel-Y 梯度极性分解与方向导向滤波：解耦上边缘 (+Gy) 与下边缘 (-Gy)
  3. 垂直列扫描极性配对 (Polarity Scanline Matching)：在单列内按物理直径配对上下沿
  4. 横向中轴点阵聚类与鲁棒直线拟合：得到单根连续主轴线与物理直径
  5. 纯 2D 叠压与上下层拓扑剥层 (Occlusion Peeling)：锁定无遮挡最顶层物料
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.occlusion_peeler import CandidateSpine, OcclusionPeeler
from src.vision.pipelines.registry import PipelineRegistry


@PipelineRegistry.register("polarity_scanline", "路线 B1: 极性扫描法 (Polarity Scanline)")
class PolarityScanlinePipeline(BaseAsparagusPipeline):
    """路线 B1：基于垂直扫描线与正负梯度极性配对的纯 2D 感知流水线"""

    name = "路线 B1: 极性扫描法 (Polarity Scanline)"
    description = "垂直列扫描 + 梯度极性正负跃变配对 (+Gy/-Gy) + 拓扑剥层顶层仲裁"

    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        super().__init__(fx, fy, cx, cy)
        self.min_length_mm = 60.0
        self.max_length_mm = 600.0
        self.min_diam_mm = 6.0
        self.max_diam_mm = 45.0
        self.peeler = OcclusionPeeler(t_junction_radius=18.0)

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep("stage1_prep", "1.预处理", "ROI 区域裁切、双边滤波与开运算去噪"),
            PipelineStep("stage2_polarity", "梯度极性", "Sobel-Y 正负极性边缘分解 (+Gy 上沿 / -Gy 下沿)"),
            PipelineStep("stage3_scanline", "极性配对", "垂直列扫描与双侧极性配对提取中心脊点"),
            PipelineStep("stage4_spines", "2.主干拟合", "横向点阵聚类与主轴鲁棒直线拟合"),
            PipelineStep("stage5_top_poses", "3.顶层位姿", "纯 2D 叠压拓扑剥层与 Top 3 顶层抓取位姿输出")
        ]

    def run(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        plane_coeff: Optional[np.ndarray] = None,
        frame_transform: Optional[np.ndarray] = None,
        frame_calib_source: str = "uncalibrated",
        nominal_z_mm: float = 640.0
    ) -> PipelineResult:
        t0 = time.perf_counter()
        h, w = color_bgr.shape[:2]
        step_images: Dict[str, Optional[np.ndarray]] = {}

        # ---------------- 步骤 1: 预处理 (去杂散点与平滑) ----------------
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        # 双边滤波保护物料边界同时消除传送带杂散反光
        filtered = cv2.bilateralFilter(roi_bgr, d=5, sigmaColor=35, sigmaSpace=35)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        # 形态学小开运算消除细微高亮反光杂点
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gray_clean = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k_open)

        vis_1 = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        vis_1[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.cvtColor(gray_clean, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_1, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_1, (12, 12), (560, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_1, (12, 12), (560, 48), (40, 230, 240), 2)
        put_text(vis_1, "STAGE 1: PREPROCESSING (ROI Bilateral & Morph Clean)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage1_prep"] = vis_1

        # ---------------- 步骤 2: 垂直梯度极性分解与方向导向滤波 ----------------
        grad_y = cv2.Sobel(gray_clean, cv2.CV_32F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(gray_clean, cv2.CV_32F, 1, 0, ksize=3)

        # 芦笋上边缘：由暗(传送带)到亮(芦笋)，Gy 显著为正
        # 芦笋下边缘：由亮(芦笋)到暗(传送带)，Gy 显著为负
        # 方向约束：排除垂直端面截断与竖向干扰 (|Gy| 显著大于 |Gx| * 0.6)
        top_edges = (grad_y > 28.0) & (np.abs(grad_y) > np.abs(grad_x) * 0.6)
        bottom_edges = (grad_y < -28.0) & (np.abs(grad_y) > np.abs(grad_x) * 0.6)

        # 连通域小面积/短碎屑过滤
        top_clean = self._filter_small_segments(top_edges.astype(np.uint8) * 255, min_len=25)
        bottom_clean = self._filter_small_segments(bottom_edges.astype(np.uint8) * 255, min_len=25)

        vis_2 = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        patch_2 = vis_2[roi_y1:roi_y2, roi_x1:roi_x2]
        patch_2[top_clean > 0] = (0, 165, 255)       # 橙色：上边缘 (+Gy)
        patch_2[bottom_clean > 0] = (255, 220, 40)   # 青蓝：下边缘 (-Gy)
        vis_2[roi_y1:roi_y2, roi_x1:roi_x2] = patch_2
        cv2.rectangle(vis_2, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_2, (12, 12), (620, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (620, 48), (0, 255, 255), 2)
        put_text(vis_2, "STAGE 2: GRADIENT POLARITY (Orange: +Gy Top | Blue: -Gy Bottom)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        step_images["stage2_polarity"] = vis_2

        # ---------------- 步骤 3: 垂直列扫描与极性配对 ----------------
        scale_2d = nominal_z_mm / self.fx
        min_diam_px = self.min_diam_mm / scale_2d
        max_diam_px = self.max_diam_mm / scale_2d

        roi_h, roi_w = gray_clean.shape
        scan_step_x = 6
        paired_ridge_points: List[Tuple[float, float, float]] = []  # (x, y, diam_px)

        vis_3 = (color_bgr.astype(np.float32) * 0.40).astype(np.uint8)

        for sx in range(10, roi_w - 10, scan_step_x):
            # 获取该列的所有上边缘与下边缘 Y 坐标
            col_top_y = np.where(top_clean[:, sx] > 0)[0]
            col_bot_y = np.where(bottom_clean[:, sx] > 0)[0]

            if len(col_top_y) == 0 or len(col_bot_y) == 0:
                continue

            # 聚合同一边缘的连续连通块
            clustered_tops = self._cluster_1d_coords(col_top_y)
            clustered_bots = self._cluster_1d_coords(col_bot_y)

            # 按物理直径配对
            used_bots = set()
            for ty in clustered_tops:
                best_by = None
                best_diff = 9999.0
                for bi, by in enumerate(clustered_bots):
                    if bi in used_bots:
                        continue
                    span_y = by - ty
                    if min_diam_px <= span_y <= max_diam_px:
                        # 物料中心区域亮度检查 (芦笋体应比传送带亮)
                        mid_y = int((ty + by) * 0.5)
                        if gray_clean[mid_y, sx] > 40:
                            diff = abs(span_y - (min_diam_px + max_diam_px) * 0.5)
                            if diff < best_diff:
                                best_diff = diff
                                best_by = (bi, by)

                if best_by is not None:
                    b_idx, chosen_by = best_by
                    used_bots.add(b_idx)
                    mid_y = float((ty + chosen_by) * 0.5)
                    diam_val = float(chosen_by - ty)
                    gx = float(sx + roi_x1)
                    gy = float(mid_y + roi_y1)
                    paired_ridge_points.append((gx, gy, diam_val))

                    # 可视化配对线段
                    cv2.line(vis_3, (int(gx), int(ty + roi_y1)), (int(gx), int(chosen_by + roi_y1)), (80, 240, 120), 1)
                    cv2.circle(vis_3, (int(gx), int(gy)), 2, (0, 255, 255), -1)

        cv2.rectangle(vis_3, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_3, (12, 12), (600, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_3, (12, 12), (600, 48), (80, 240, 120), 2)
        put_text(vis_3, f"STAGE 3: POLARITY SCANLINE PAIRS | Ridge Points: {len(paired_ridge_points)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 240, 120), 2)
        step_images["stage3_scanline"] = vis_3

        # ---------------- 步骤 4: 横向主轴聚类与鲁棒拟合 ----------------
        clusters = self._cluster_ridge_points(paired_ridge_points, max_dx=22.0, max_dy=10.0)
        candidates: List[CandidateSpine] = []
        vis_4 = color_bgr.copy()
        cand_id = 1
        palette = [(255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120), (255, 210, 40)]

        for cluster_pts in clusters:
            if len(cluster_pts) < 10:
                continue

            pts_arr = np.array([[p[0], p[1]] for p in cluster_pts], dtype=np.float32)
            diam_vals = [p[2] for p in cluster_pts]

            # 直线拟合
            [vx_v, vy_v, x0_v, y0_v] = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx_v[0]), float(vy_v[0])
            if abs(vx) < 0.45:
                continue  # 排除非主要输送方向
            if vx < 0:
                vx, vy = -vx, -vy

            # 沿轴向投影计算真实长度
            mean_pt = np.mean(pts_arr, axis=0)
            proj = np.dot(pts_arr - mean_pt, np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            len_px = float(max_p - min_p)
            len_mm = len_px * scale_2d
            diam_px = float(np.median(diam_vals))
            diam_mm = diam_px * scale_2d

            if not (self.min_length_mm <= len_mm <= self.max_length_mm and
                    self.min_diam_mm <= diam_mm <= self.max_diam_mm):
                continue

            cx_val = float(mean_pt[0] + (min_p + max_p) * 0.5 * vx)
            cy_val = float(mean_pt[1] + (min_p + max_p) * 0.5 * vy)
            yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
            if yaw_deg > 90.0: yaw_deg -= 180.0
            elif yaw_deg < -90.0: yaw_deg += 180.0

            # 构造定向外框角点
            u_vec = np.array([vx, vy])
            v_vec = np.array([-vy, vx])
            half_l = len_px * 0.5
            half_w = diam_px * 0.5
            c_pt = np.array([cx_val, cy_val])
            p1 = c_pt - half_l * u_vec - half_w * v_vec
            p2 = c_pt + half_l * u_vec - half_w * v_vec
            p3 = c_pt + half_l * u_vec + half_w * v_vec
            p4 = c_pt - half_l * u_vec + half_w * v_vec
            box_corners = np.array([p1, p2, p3, p4], dtype=np.int32)

            cand = CandidateSpine(
                id=cand_id,
                center_px=(cx_val, cy_val),
                length_px=len_px,
                diam_px=diam_px,
                yaw_deg=yaw_deg,
                axis_vector=(vx, vy),
                box_corners=box_corners,
                extra_data={'l_mm': len_mm, 'd_mm': diam_mm, 'scale': scale_2d}
            )
            candidates.append(cand)

            col = palette[(cand_id - 1) % len(palette)]
            cv2.polylines(vis_4, [box_corners], True, col, 2)
            cv2.line(vis_4,
                     (int(cx_val - half_l * vx), int(cy_val - half_l * vy)),
                     (int(cx_val + half_l * vx), int(cy_val + half_l * vy)),
                     (255, 255, 255), 2)
            put_text(vis_4, f"#{cand_id} D:{diam_mm:.1f} L:{len_mm:.0f}",
                     (int(cx_val - 35), int(cy_val - half_w - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cand_id += 1

        cv2.rectangle(vis_4, (12, 12), (600, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_4, (12, 12), (600, 48), (40, 230, 240), 2)
        put_text(vis_4, f"STAGE 4: SPINES FITTED | Candidates: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage4_spines"] = vis_4

        # ---------------- 步骤 5: 纯 2D 叠压拓扑剥层与 Top 3 输出 ----------------
        combined_edges = cv2.bitwise_or(top_clean, bottom_clean)
        full_edge_map = np.zeros((h, w), dtype=np.uint8)
        full_edge_map[roi_y1:roi_y2, roi_x1:roi_x2] = combined_edges

        # 执行循环剥层算法
        peeled_layers = self.peeler.peel_layers(candidates, edge_image=full_edge_map)

        # 展平分层序列：最顶层 (Layer 0) 拥有最高抓取优先级
        ordered_candidates: List[Tuple[CandidateSpine, int]] = []
        for layer_idx, layer_cands in enumerate(peeled_layers):
            for c_item in layer_cands:
                ordered_candidates.append((c_item, layer_idx))

        from src.vision.asparagus_analyzer import AsparagusTarget, AsparagusAnalyzer
        targets: List[AsparagusTarget] = []

        for rank_idx, (cand, layer_level) in enumerate(ordered_candidates):
            cx_val, cy_val = cand.center_px
            vx, vy = cand.axis_vector
            len_mm = cand.extra_data['l_mm']
            diam_mm = cand.extra_data['d_mm']

            # 纯 2D 标称物理高度映射：顶层赋予安全抓取高度 (+35mm)，次层 (+20mm)
            rel_h = max(10.0, 35.0 - layer_level * 15.0)
            z_nominal = nominal_z_mm - rel_h

            grip_x = float((cx_val - self.cx) * z_nominal / self.fx)
            grip_y = float((cy_val - self.cy) * z_nominal / self.fy)
            grip_z = float(z_nominal)

            if frame_transform is not None:
                p_cam_h = np.array([grip_x, grip_y, grip_z, 1.0])
                p_robot_h = frame_transform @ p_cam_h
                robot_x = float(p_robot_h[0])
                robot_y = float(p_robot_h[1])
                robot_z = float(p_robot_h[2])
                r_mat = frame_transform[:3, :3]
                v_robot = r_mat @ np.array([vx, vy, 0.0])
                robot_r = float(np.degrees(np.arctan2(v_robot[1], v_robot[0])))
                if robot_r > 90.0: robot_r -= 180.0
                elif robot_r < -90.0: robot_r += 180.0
            else:
                robot_x, robot_y, robot_z = grip_x, grip_y, rel_h
                robot_r = cand.yaw_deg

            targets.append(AsparagusTarget(
                id=rank_idx + 1,
                center_px=(cx_val, cy_val),
                length_px=cand.length_px,
                diam_px=cand.diam_px,
                yaw_deg=round(cand.yaw_deg, 1),
                axis_vector=(vx, vy),
                box_corners=cand.box_corners,
                contour=cand.box_corners,
                length_mm=round(len_mm, 1),
                diam_mm=round(diam_mm, 1),
                grip_x=round(grip_x, 1),
                grip_y=round(grip_y, 1),
                grip_z=round(grip_z, 1),
                z_top=round(z_nominal, 1),
                rel_height_mm=round(rel_h, 1),
                robot_x=round(robot_x, 1),
                robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1),
                robot_r=round(robot_r, 1),
                is_topmost=(rank_idx == 0),
                calibration_source=frame_calib_source
            ))

        # 严格保留排名前三位 (Top 3)
        top_targets = targets[:3]
        dummy_analyzer = AsparagusAnalyzer(self.fx, self.fy, self.cx, self.cy)
        vis_5 = dummy_analyzer.draw_detections(color_bgr, top_targets, sel_target_idx=0)
        step_images["stage5_top_poses"] = vis_5

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=top_targets,
            elapsed_ms=round(elapsed_ms, 1),
            step_snapshots=step_images,
            extra_metrics={
                "ridge_points_paired": len(paired_ridge_points),
                "candidates_found": len(candidates),
                "peeled_layers": len(peeled_layers)
            }
        )

    def _filter_small_segments(self, binary_mask: np.ndarray, min_len: int = 25) -> np.ndarray:
        """过滤长度过短的散斑碎片"""
        cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_mask = np.zeros_like(binary_mask)
        for c in cnts:
            if cv2.arcLength(c, False) >= min_len or cv2.contourArea(c) >= 20:
                cv2.drawContours(out_mask, [c], -1, 255, -1)
        return out_mask

    def _cluster_1d_coords(self, coords: np.ndarray, max_gap: int = 3) -> List[float]:
        """对一维坐标进行紧邻聚合并取均值"""
        if len(coords) == 0:
            return []
        clusters = []
        curr = [coords[0]]
        for val in coords[1:]:
            if val - curr[-1] <= max_gap:
                curr.append(val)
            else:
                clusters.append(float(np.mean(curr)))
                curr = [val]
        clusters.append(float(np.mean(curr)))
        return clusters

    def _cluster_ridge_points(
        self,
        pts: List[Tuple[float, float, float]],
        max_dx: float = 22.0,
        max_dy: float = 10.0
    ) -> List[List[Tuple[float, float, float]]]:
        """沿横向对配对脊点聚类成线段簇"""
        if not pts:
            return []
        # 按 X 坐标排序
        pts_sorted = sorted(pts, key=lambda p: p[0])
        clusters: List[List[Tuple[float, float, float]]] = []

        for p in pts_sorted:
            px, py, pd = p
            assigned = False
            best_c = None
            min_dist = 9999.0
            for c in clusters:
                last_x, last_y, _ = c[-1]
                dx = px - last_x
                dy = abs(py - last_y)
                if 0 <= dx <= max_dx and dy <= max_dy:
                    dist = dx + dy
                    if dist < min_dist:
                        min_dist = dist
                        best_c = c

            if best_c is not None:
                best_c.append(p)
                assigned = True
            if not assigned:
                clusters.append([p])

        return clusters
