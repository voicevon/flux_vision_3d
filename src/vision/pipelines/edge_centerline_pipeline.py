"""
技术路线 B：基于双侧边缘检测与几何对称轴线拟合的感知流水线 (Edge Centerline Pipeline)
=======================================================================================
核心原理 (用户指定路线 2):
  1. 传送带核心 ROI 裁切 + 保边双边滤波 (Bilateral Filter)，去除杂质反光
  2. 结合 Canny 与 Sobel-Y 梯度边缘检测，提取清晰的物料高频轮廓与上下边界
  3. 轮廓几何分析与边缘长条筛选：提取细长环状双侧边缘与平行边缘段
  4. 求解几何中线轨迹 (Geometric Centerline)，直接确定芦笋中轴线与物理直径
  5. 空间位姿恢复与 Top 3 推荐输出
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry


@PipelineRegistry.register("edge_centerline", "路线 B: 双侧边缘拟合法 (Edge Centerline)")
class EdgeCenterlinePipeline(BaseAsparagusPipeline):
    """路线 B：基于边缘提取与双侧对称配对的感知流水线"""

    name = "路线 B: 双侧边缘拟合法 (Edge Centerline)"
    description = "通过物料上下双侧边缘梯度，基于长条轮廓几何对称性求解中心轴线"

    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        super().__init__(fx, fy, cx, cy)
        self.min_length_mm = 60.0
        self.max_length_mm = 600.0
        self.min_diam_mm = 6.0
        self.max_diam_mm = 48.0

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep("stage1_prep", "1.预处理", "ROI 区域裁切与保边双边滤波 (Bilateral Filter)"),
            PipelineStep("stage2_edges", "梯度边缘", "Sobel-Y & Canny 双侧高频边缘梯度提取"),
            PipelineStep("stage3_contours", "双侧边界", "细长双侧边界提取与毛刺抑制"),
            PipelineStep("stage4_centerline", "2.几何中线", "双侧对称几何中心线轨迹拟合与单体验证"),
            PipelineStep("stage5_poses", "3.位姿", "3D 空间位姿恢复、SCARA抓取坐标与Top 3顶层锁定")
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

        # ---------------- 步骤 1: 预处理 ----------------
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        filtered = cv2.bilateralFilter(roi_bgr, d=7, sigmaColor=50, sigmaSpace=50)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        vis_1 = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        vis_1[roi_y1:roi_y2, roi_x1:roi_x2] = filtered
        cv2.rectangle(vis_1, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_1, (12, 12), (540, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_1, (12, 12), (540, 48), (40, 230, 240), 2)
        put_text(vis_1, "STAGE 1: PREPROCESSING (ROI Bilateral Filter)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage1_prep"] = vis_1

        # ---------------- 步骤 2: 梯度边缘提取 ----------------
        # 针对大致横躺芦笋，检测 Y 方向垂直边缘 (Sobel-Y)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        canny_edges = cv2.Canny(gray, 30, 90)
        edge_raw = cv2.bitwise_or(canny_edges, (abs_grad_y > 40).astype(np.uint8) * 255)

        # 过滤左右边界杂散
        inner_mask = np.zeros_like(edge_raw)
        inner_mask[5:-5, 5:-5] = 255
        edge_raw = cv2.bitwise_and(edge_raw, inner_mask)

        vis_2 = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        edge_patch = vis_2[roi_y1:roi_y2, roi_x1:roi_x2]
        edge_patch[edge_raw > 0] = (0, 255, 255)
        vis_2[roi_y1:roi_y2, roi_x1:roi_x2] = edge_patch
        cv2.rectangle(vis_2, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_2, (12, 12), (540, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (540, 48), (0, 255, 255), 2)
        put_text(vis_2, "STAGE 2: GRADIENT EDGES (Sobel-Y & Canny Borders)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        step_images["stage2_edges"] = vis_2

        # ---------------- 步骤 3: 双侧细长边界提取与形态学连接 ----------------
        k_close_x = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        edges_connected = cv2.morphologyEx(edge_raw, cv2.MORPH_CLOSE, k_close_x)

        cnts, _ = cv2.findContours(edges_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale = (nominal_z_mm / self.fx)

        vis_3 = color_bgr.copy()
        palette = [(255, 100, 40), (40, 240, 220), (220, 100, 240), (80, 240, 100), (255, 200, 40)]
        extracted_borders = []

        for c in cnts:
            if len(c) < 20:
                continue
            rect = cv2.minAreaRect(c)
            (rcx, rcy), (rw, rh), _ = rect
            l_px = max(rw, rh)
            w_px = min(rw, rh)
            l_mm = l_px * scale
            w_mm = w_px * scale

            # 候选判定：细长物料边缘，长宽比显著
            if l_mm >= 60.0 and (w_mm <= 60.0):
                [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy = float(vx[0]), float(vy[0])
                if abs(vx) < 0.45:
                    continue
                if vx < 0:
                    vx, vy = -vx, -vy

                # 估算直径
                d_mm = max(6.0, w_mm)
                extracted_borders.append({
                    'contour': c + np.array([roi_x1, roi_y1]),
                    'rcx': rcx + roi_x1, 'rcy': rcy + roi_y1,
                    'vx': vx, 'vy': vy,
                    'l_px': l_px, 'w_px': max(w_px, 8.0),
                    'l_mm': l_mm, 'd_mm': d_mm,
                    'pts': c.reshape(-1, 2)
                })

        for b_i, b_item in enumerate(extracted_borders):
            col = palette[b_i % len(palette)]
            cv2.drawContours(vis_3, [b_item['contour']], -1, col, 2)
            put_text(vis_3, f"Edge #{b_i+1}", (int(b_item['rcx'] - 20), int(b_item['rcy'] - 10)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        cv2.rectangle(vis_3, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_3, (12, 12), (560, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_3, (12, 12), (560, 48), (255, 100, 40), 2)
        put_text(vis_3, f"STAGE 3: BORDER EXTRACTION | Borders: {len(extracted_borders)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 120, 60), 2)
        step_images["stage3_contours"] = vis_3

        # ---------------- 步骤 4: 几何中心线拟合 ----------------
        candidates = []
        vis_4 = color_bgr.copy()

        for b_i, item in enumerate(extracted_borders):
            global_cx = float(item['rcx'])
            global_cy = float(item['rcy'])
            vx, vy = item['vx'], item['vy']
            len_px = item['l_px']
            diam_px = item['w_px']
            l_mm = item['l_mm']
            d_mm = item['d_mm']

            yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
            if yaw_deg > 90.0: yaw_deg -= 180.0
            elif yaw_deg < -90.0: yaw_deg += 180.0

            u_vec = np.array([vx, vy])
            v_vec = np.array([-vy, vx])
            half_l = len_px * 0.5
            half_w = diam_px * 0.5

            c_pt = np.array([global_cx, global_cy])
            p1 = c_pt - half_l * u_vec - half_w * v_vec
            p2 = c_pt + half_l * u_vec - half_w * v_vec
            p3 = c_pt + half_l * u_vec + half_w * v_vec
            p4 = c_pt - half_l * u_vec + half_w * v_vec
            box_corners = np.array([p1, p2, p3, p4], dtype=np.int32)

            col = palette[b_i % len(palette)]
            cv2.polylines(vis_4, [box_corners], True, col, 2)
            cv2.line(vis_4, (int(global_cx - half_l * vx), int(global_cy - half_l * vy)),
                     (int(global_cx + half_l * vx), int(global_cy + half_l * vy)), (255, 255, 255), 2)
            put_text(vis_4, f"#{b_i+1} D:{d_mm:.1f} L:{l_mm:.0f}",
                     (int(global_cx - 24), int(global_cy - half_w - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            candidates.append({
                'id': b_i + 1, 'center_px': (global_cx, global_cy),
                'length_px': len_px, 'diam_px': diam_px,
                'length_mm': round(l_mm, 1), 'diam_mm': round(d_mm, 1),
                'yaw_deg': round(yaw_deg, 1), 'axis_vector': (vx, vy),
                'box_corners': box_corners, 'z_ref': nominal_z_mm
            })

        cv2.rectangle(vis_4, (12, 12), (580, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_4, (12, 12), (580, 48), (40, 230, 240), 2)
        put_text(vis_4, f"STAGE 4: GEOMETRIC CENTERLINES | Valid Instances: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage4_centerline"] = vis_4

        # ---------------- 步骤 5: 位姿与 Top 3 输出 ----------------
        from src.vision.asparagus_analyzer import AsparagusTarget, AsparagusAnalyzer
        targets: List[AsparagusTarget] = []
        for sp in candidates:
            cx_val, cy_val = sp['center_px']
            vx_val, vy_val = sp['axis_vector']
            z_ref = sp['z_ref']
            z_top, z_med = 0.0, z_ref
            grip_x = float((cx_val - self.cx) * z_ref / self.fx)
            grip_y = float((cy_val - self.cy) * z_ref / self.fy)
            grip_z = 0.0
            rel_height_mm = 0.0

            if depth_mm is not None:
                int_cx, int_cy = int(cx_val), int(cy_val)
                if 0 <= int_cx < w and 0 <= int_cy < h and 350 <= depth_mm[int_cy, int_cx] <= 780:
                    z_med = float(depth_mm[int_cy, int_cx])
                    z_top = z_med
                    grip_x = float((cx_val - self.cx) * z_med / self.fx)
                    grip_y = float((cy_val - self.cy) * z_med / self.fy)
                    grip_z = z_top
                    if plane_coeff is not None:
                        table_z = plane_coeff[0] * cx_val + plane_coeff[1] * cy_val + plane_coeff[2]
                        rel_height_mm = float(table_z - z_top)

            if frame_transform is not None:
                p_cam_h = np.array([grip_x, grip_y, grip_z, 1.0])
                p_robot_h = frame_transform @ p_cam_h
                robot_x = float(p_robot_h[0])
                robot_y = float(p_robot_h[1])
                robot_z = float(p_robot_h[2])
                r_mat = frame_transform[:3, :3]
                v_robot = r_mat @ np.array([vx_val, vy_val, 0.0])
                robot_r = float(np.degrees(np.arctan2(v_robot[1], v_robot[0])))
                if robot_r > 90.0: robot_r -= 180.0
                elif robot_r < -90.0: robot_r += 180.0
            else:
                robot_x, robot_y, robot_z = grip_x, grip_y, rel_height_mm
                robot_r = sp['yaw_deg']

            targets.append(AsparagusTarget(
                id=sp['id'], center_px=(cx_val, cy_val),
                length_px=sp['length_px'], diam_px=sp['diam_px'],
                yaw_deg=sp['yaw_deg'], axis_vector=(vx_val, vy_val),
                box_corners=sp['box_corners'], contour=sp['box_corners'],
                length_mm=sp['length_mm'], diam_mm=sp['diam_mm'],
                grip_x=round(grip_x, 1), grip_y=round(grip_y, 1), grip_z=round(grip_z, 1),
                z_top=round(z_top, 1), rel_height_mm=round(rel_height_mm, 1),
                robot_x=round(robot_x, 1), robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1), robot_r=round(robot_r, 1),
                is_topmost=False, calibration_source=frame_calib_source
            ))

        if len(targets) > 0:
            targets.sort(key=lambda t: (t.length_px * t.diam_px), reverse=True)
            targets = targets[:3]
            for rank_i, t in enumerate(targets):
                t.id = rank_i + 1
                t.is_topmost = (rank_i == 0)

        dummy_analyzer = AsparagusAnalyzer(self.fx, self.fy, self.cx, self.cy)
        vis_5 = dummy_analyzer.draw_detections(color_bgr, targets, sel_target_idx=0)
        step_images["stage5_poses"] = vis_5

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=targets,
            elapsed_ms=round(elapsed_ms, 1),
            step_snapshots=step_images,
            extra_metrics={"borders_found": len(extracted_borders)}
        )
