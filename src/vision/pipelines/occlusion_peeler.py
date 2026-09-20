"""
纯 2D 芦笋叠压与上下层遮挡拓扑剥层器 (Occlusion Graph Peeler)
============================================================
核心职责：
  1. 在纯 2D 彩色/灰度图像约束下，计算多根芦笋候选之间的交叉干涉与空间几何重叠
  2. 基于边缘连续性与 T 型截断特征 (T-Junction & Continuity)，判定上层 (Upper) 与下层 (Lower)
  3. 构建遮挡有向无环图 (Occlusion DAG)，计算每个物料的被压入度 (In-Degree)
  4. 执行循环拓扑剥层 (Topological Peeling Loop)，由顶向底分层排序，锁定最顶层可抓取目标
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


@dataclass
class CandidateSpine:
    """待判定的单根芦笋几何候选数据结构"""
    id: int
    center_px: Tuple[float, float]
    length_px: float
    diam_px: float
    yaw_deg: float
    axis_vector: Tuple[float, float]
    box_corners: np.ndarray          # 4x2 坐标数组
    edge_pts: Optional[np.ndarray] = None  # 属于该候选的边缘像素集合 (Nx2)
    extra_data: Dict[str, Any] = field(default_factory=dict)


def line_segment_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    """计算二维空间中两条线段 [p1, p2] 与 [p3, p4] 的交点，若无交点返回 None"""
    d = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if abs(d) < 1e-6:
        return None
    t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / d
    u = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])) / d
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = p1[0] + t * (p2[0] - p1[0])
        iy = p1[1] + t * (p2[1] - p1[1])
        return np.array([ix, iy], dtype=np.float32)
    return None


class OcclusionPeeler:
    """纯 2D 芦笋遮挡关系裁决与循环拓扑剥层器"""

    def __init__(self, t_junction_radius: float = 18.0):
        """
        :param t_junction_radius: 交叉点周围分析边缘 T 型截断的局部邻域半径 (像素)
        """
        self.t_junction_radius = t_junction_radius

    def detect_overlap_and_crossing(
        self,
        c1: CandidateSpine,
        c2: CandidateSpine
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        检测两个芦笋候选之间是否存在交叉重叠
        :return: (is_overlapping, cross_point_xy, overlap_area)
        """
        # 1. 轴线相交测试
        c1_pt = np.array(c1.center_px, dtype=np.float32)
        c2_pt = np.array(c2.center_px, dtype=np.float32)
        v1 = np.array(c1.axis_vector, dtype=np.float32)
        v2 = np.array(c2.axis_vector, dtype=np.float32)

        p1_a = c1_pt - v1 * (c1.length_px * 0.5)
        p1_b = c1_pt + v1 * (c1.length_px * 0.5)
        p2_a = c2_pt - v2 * (c2.length_px * 0.5)
        p2_b = c2_pt + v2 * (c2.length_px * 0.5)

        cross_pt = line_segment_intersection(p1_a, p1_b, p2_a, p2_b)

        # 2. 定向外包围盒凸多边形相交面积
        box1 = np.ascontiguousarray(c1.box_corners, dtype=np.float32)
        box2 = np.ascontiguousarray(c2.box_corners, dtype=np.float32)
        has_intersect, inter_poly = cv2.intersectConvexConvex(box1, box2)

        area = 0.0
        if has_intersect and inter_poly is not None and len(inter_poly) >= 3:
            area = float(cv2.contourArea(inter_poly))

        is_overlapping = (cross_pt is not None) or (area > 50.0)
        return is_overlapping, cross_pt, area

    def arbitrate_stacking_order(
        self,
        c1: CandidateSpine,
        c2: CandidateSpine,
        cross_pt: Optional[np.ndarray],
        edge_image: Optional[np.ndarray] = None
    ) -> int:
        """
        仲裁两根重叠芦笋谁在上层、谁在下层
        :return:
          +1: c1 压在 c2 上面 (c1 是上层)
          -1: c2 压在 c1 上面 (c2 是上层)
           0: 无法确信 (平级相撞或平行贴合)
        """
        # 若提供了 2D 边缘二值图，利用交叉处“T型截断断点”与“连续贯穿度”判决
        if edge_image is not None and cross_pt is not None:
            cx, cy = int(round(cross_pt[0])), int(round(cross_pt[1]))
            h, w = edge_image.shape[:2]
            rad = int(max(self.t_junction_radius, max(c1.diam_px, c2.diam_px) + 8.0))
            y1, y2 = max(0, cy - rad), min(h, cy + rad)
            x1, x2 = max(0, cx - rad), min(w, cx + rad)

            if y2 > y1 and x2 > x1:
                patch = edge_image[y1:y2, x1:x2]
                v1 = np.array(c1.axis_vector)
                v2 = np.array(c2.axis_vector)

                # 传入各自的半宽 diam_px * 0.5，在真实边界附近采样
                c1_cont_score = self._measure_axis_continuity(patch, (cx - x1, cy - y1), v1, c1.diam_px * 0.5)
                c2_cont_score = self._measure_axis_continuity(patch, (cx - x1, cy - y1), v2, c2.diam_px * 0.5)

                if c1_cont_score > c2_cont_score + 0.15:
                    return 1
                elif c2_cont_score > c1_cont_score + 0.15:
                    return -1

        # 兜底纯几何启发式仲裁：
        # 上层芦笋通常长宽比更标准、几何完整度更高（未被遮挡截断导致尺寸虚缩）
        aspect1 = c1.length_px / max(c1.diam_px, 1.0)
        aspect2 = c2.length_px / max(c2.diam_px, 1.0)
        if aspect1 > aspect2 * 1.25:
            return 1
        elif aspect2 > aspect1 * 1.25:
            return -1

        return 0

    def _measure_axis_continuity(
        self,
        patch: np.ndarray,
        center: Tuple[int, int],
        axis_v: np.ndarray,
        half_w: float
    ) -> float:
        """评估某轴线在交叉局部 Patch 内的双侧边缘连续度 (0.0 ~ 1.0)"""
        cx, cy = center
        h, w = patch.shape[:2]
        vx, vy = axis_v
        nx, ny = -vy, vx
        hit_count = 0
        test_samples = 0
        # 沿长轴方向采样几个点，在两侧边界附近采样
        steps = [-8, -4, 0, 4, 8]
        offsets = [-half_w, half_w]
        for step in steps:
            for offset_side in offsets:
                # 容差 +/- 2 像素
                hit_this = False
                for delta in [-2, 0, 2]:
                    px = int(round(cx + step * vx + (offset_side + delta) * nx))
                    py = int(round(cy + step * vy + (offset_side + delta) * ny))
                    if 0 <= px < w and 0 <= py < h:
                        if patch[py, px] > 0:
                            hit_this = True
                            break
                test_samples += 1
                if hit_this:
                    hit_count += 1
        return (hit_count / max(test_samples, 1))

    def peel_layers(
        self,
        candidates: List[CandidateSpine],
        edge_image: Optional[np.ndarray] = None
    ) -> List[List[CandidateSpine]]:
        """
        循环迭代拓扑剥层主流程 (Topological Peeling Loop)
        :param candidates: 所有识别出的芦笋候选
        :param edge_image: 2D 边缘二值图 (可选，用于高精度 T-Junction 断点检测)
        :return: 分层芦笋列表 [Layer_0(最顶层), Layer_1(次顶层), ...]
        """
        if not candidates:
            return []

        n = len(candidates)
        # 1. 构建有向遮挡图 (DAG)
        # graph[u] 存储被 u 压在下面的物料集合 (u -> v 表示 u 压 v)
        graph: Dict[int, Set[int]] = {c.id: set() for c in candidates}
        # in_degree 记录物料被多少个上层物料压住
        in_degree: Dict[int, int] = {c.id: 0 for c in candidates}
        cand_map: Dict[int, CandidateSpine] = {c.id: c for c in candidates}

        # 2. 两两配对仲裁
        for i in range(n):
            for j in range(i + 1, n):
                c_i, c_j = candidates[i], candidates[j]
                is_overlap, cross_pt, area = self.detect_overlap_and_crossing(c_i, c_j)
                if not is_overlap:
                    continue

                order = self.arbitrate_stacking_order(c_i, c_j, cross_pt, edge_image)
                if order > 0:
                    # c_i 压在 c_j 上面: c_i -> c_j
                    if c_j.id not in graph[c_i.id]:
                        graph[c_i.id].add(c_j.id)
                        in_degree[c_j.id] += 1
                elif order < 0:
                    # c_j 压在 c_i 上面: c_j -> c_i
                    if c_i.id not in graph[c_j.id]:
                        graph[c_j.id].add(c_i.id)
                        in_degree[c_i.id] += 1

        # 3. 循环剥层 (Peeling Loop)
        layers: List[List[CandidateSpine]] = []
        remaining_ids = set(cand_map.keys())

        while remaining_ids:
            # 查找当前所有未被任何物料压住的物料 (in_degree == 0)
            current_top = [cand_map[cid] for cid in remaining_ids if in_degree[cid] == 0]

            if not current_top:
                # 环状依赖破环防御 (极端罕见交叉循环)，选择被压次数最少的物料破环
                min_in = min(in_degree[cid] for cid in remaining_ids)
                current_top = [cand_map[cid] for cid in remaining_ids if in_degree[cid] == min_in][:1]

            # 记录当前顶层
            layers.append(current_top)

            # 从待处理集合中剥离当前顶层，并释放其覆盖的下层节点
            for top_c in current_top:
                remaining_ids.remove(top_c.id)
                for child_id in graph[top_c.id]:
                    if child_id in in_degree:
                        in_degree[child_id] = max(0, in_degree[child_id] - 1)

        return layers
