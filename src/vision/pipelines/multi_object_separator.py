"""
多根芦笋分离工具集 (Multi-Object Separator Utilities)
====================================================
为 H1-H6 算法流水线提供通用的多目标分离功能：
  1. Watershed 预分割：将粘连二值掩膜按距离变换极大值分割为独立区域
  2. 骨架分叉点切割：检测 degree>=3 节点并切断骨架为独立路径
  3. DBSCAN 聚类：替代贪心聚类，自动发现任意数量的簇
"""

import numpy as np
import cv2
from typing import List, Tuple


def watershed_presegment(binary_mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    """使用 Watershed 分水岭预分割粘连前景掩膜。

    Parameters
    ----------
    binary_mask : np.ndarray
        二值前景掩膜 (uint8, 255=前景)。
    min_area : int
        最小区域面积（像素），小于此值的区域将被丢弃。

    Returns
    -------
    labels : np.ndarray
        分割标签图 (int32)，背景=0，各区域=1,2,3...
    """
    # 距离变换
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # 局部极大值作为种子 (使用膨胀检测)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    local_max = cv2.dilate(dist, kernel)
    seeds = ((dist == local_max) & (dist > 3.0)).astype(np.uint8)

    # 标记种子连通分量
    num_seeds, seed_labels = cv2.connectedComponents(seeds)

    # Watershed 需要 markers: 背景=1, 未知=0, 前景种子=2,3,4...
    markers = np.zeros_like(binary_mask, dtype=np.int32)
    markers[binary_mask == 0] = 1  # 确定背景
    for i in range(1, num_seeds):
        markers[seed_labels == i] = i + 1  # 各种子标记

    # 构造 3 通道图像供 Watershed
    img_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_3ch, markers)

    # 后处理：边界线与背景置 0，前景标签归 1, 2, ...
    fg_markers = np.where(markers > 1, markers - 1, 0)

    # 过滤微小噪点并重排为连续标签
    unique, counts = np.unique(fg_markers, return_counts=True)
    valid_labels = [u for u, c in zip(unique, counts) if u > 0 and c >= min_area]
    relabeled = np.zeros_like(markers, dtype=np.int32)
    for new_id, old_id in enumerate(valid_labels, start=1):
        relabeled[fg_markers == old_id] = new_id

    return relabeled


def cut_skeleton_junctions(skeleton: np.ndarray, radius: int = 2) -> np.ndarray:
    """检测骨架分叉点 (degree >= 3) 并在分叉点周围切断骨架。

    Parameters
    ----------
    skeleton : np.ndarray
        二值骨架掩膜 (uint8, 255=骨架)。
    radius : int
        在分叉点周围擦除的半径（像素）。

    Returns
    -------
    cut_skeleton : np.ndarray
        切断分叉后的骨架 (uint8, 255=骨架)。
    """
    skel_f32 = (skeleton > 0).astype(np.float32)

    # 3×3 邻域求和检测连通度
    kernel = np.ones((3, 3), dtype=np.float32)
    neighbor_count = cv2.filter2D(skel_f32, -1, kernel) - skel_f32

    # 分叉点：骨架像素且邻居数 >= 3
    junctions = (skel_f32 > 0) & (neighbor_count >= 3.0)

    # 在分叉点周围擦除
    result = skeleton.copy()
    if radius > 0:
        junction_mask = junctions.astype(np.uint8) * 255
        dilated = cv2.dilate(junction_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
        result[dilated > 0] = 0
    else:
        result[junctions] = 0

    return result


def dbscan_cluster_ridge_points(
    points: List[Tuple[float, float, float]],
    eps: float = 0.0,
    min_samples: int = 8
) -> List[List[Tuple[float, float, float]]]:
    """使用 DBSCAN 聚类离散脊点，自动发现任意数量的芦笋簇。

    Parameters
    ----------
    points : list of (gx, gy, diam_est)
        脊线离散点列表。
    eps : float
        DBSCAN 邻域半径。若为 0 则自动计算。
    min_samples : int
        DBSCAN 最小簇样本数。

    Returns
    -------
    clusters : list of list of (gx, gy, diam_est)
        各簇的点列表。
    """
    if len(points) < min_samples:
        return [points] if points else []

    coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)

    # 自动估算 eps：基于 Y 方向中位直径的 60%
    if eps <= 0.0:
        diams = [p[2] for p in points]
        median_diam = float(np.median(diams)) if diams else 20.0
        eps = max(12.0, median_diam * 0.6)

    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_
    except ImportError:
        # 降级为简单的 Y 方向分箱聚类
        return _fallback_y_cluster(points, eps)

    clusters: List[List[Tuple[float, float, float]]] = []
    unique_labels = set(labels)
    for label in sorted(unique_labels):
        if label == -1:
            continue  # 噪声点
        cluster = [points[i] for i in range(len(points)) if labels[i] == label]
        if len(cluster) >= min_samples:
            clusters.append(cluster)

    return clusters


def _fallback_y_cluster(
    points: List[Tuple[float, float, float]],
    max_dy: float = 15.0
) -> List[List[Tuple[float, float, float]]]:
    """当 sklearn 不可用时的降级 Y 方向分箱聚类。"""
    if not points:
        return []

    sorted_pts = sorted(points, key=lambda p: p[1])  # 按 Y 排序
    clusters: List[List[Tuple[float, float, float]]] = [[sorted_pts[0]]]

    for pt in sorted_pts[1:]:
        merged = False
        for cluster in clusters:
            cluster_mean_y = np.mean([p[1] for p in cluster])
            if abs(pt[1] - cluster_mean_y) <= max_dy:
                cluster.append(pt)
                merged = True
                break
        if not merged:
            clusters.append([pt])

    return [c for c in clusters if len(c) >= 5]


def apply_per_label_skeleton(
    labels: np.ndarray,
    binary_mask: np.ndarray
) -> np.ndarray:
    """对 Watershed 分割的掩膜执行高效骨架化，并物理切断分水岭交界线。

    性能优化：
      单次全局 skeletonize + Sobel 标签交界切断，
      替代老版本逐标签全图循环细化（耗时由 28s 骤降至 0.04s）。

    Parameters
    ----------
    labels : np.ndarray
        Watershed 分割标签图 (int32)。
    binary_mask : np.ndarray
        原始二值前景掩膜 (uint8)。

    Returns
    -------
    skeleton : np.ndarray
        合并切断后的骨架 (uint8, 255=骨架)。
    """
    from skimage.morphology import skeletonize

    # 单次全局骨架化
    skel = skeletonize(binary_mask > 0).astype(np.uint8) * 255

    # 仅当存在多个分水岭区域时，擦除标签之间的分界线
    if labels.max() > 1:
        gx = cv2.Sobel(labels.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(labels.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        boundaries = ((np.abs(gx) + np.abs(gy)) > 0.1).astype(np.uint8)
        k_b = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundaries = cv2.dilate(boundaries, k_b)
        skel[boundaries > 0] = 0

    return skel

