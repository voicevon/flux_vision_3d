import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def compute_sato_vesselness(volume: np.ndarray, scales: list = [1.5, 2.5, 3.5]) -> np.ndarray:
    """Compute Sato vesselness for a 2D grayscale image using vectorized eigenvalue analysis.

    Parameters
    ----------
    volume : np.ndarray
        Input gray image (H, W) in uint8 or float format.
    scales : list
        List of sigma values for Gaussian smoothing.

    Returns
    -------
    vesselness : np.ndarray
        Vesselness response image normalized to [0, 255] (uint8).
    """
    img = volume.astype(np.float32)
    best = np.zeros_like(img, dtype=np.float32)

    for sigma in scales:
        # 二阶高斯导数
        Ixx = gaussian_filter(img, sigma, order=(2, 0))
        Iyy = gaussian_filter(img, sigma, order=(0, 2))
        Ixy = gaussian_filter(img, sigma, order=(1, 1))

        # 2x2 实对称矩阵 [[Ixx, Ixy], [Ixy, Iyy]] 的特征值闭式解析解
        trace = Ixx + Iyy
        diff = Ixx - Iyy
        disc = np.sqrt(np.maximum(0.0, diff * diff + 4.0 * (Ixy * Ixy)))

        ev1 = 0.5 * (trace + disc)
        ev2 = 0.5 * (trace - disc)

        # Sato 原则：管状明亮结构的二阶导向内凹，两个特征值均小于 0
        cond = (ev1 < 0.0) & (ev2 < 0.0)
        vessel = np.zeros_like(img)
        vessel[cond] = np.abs(ev1[cond] * ev2[cond])

        best = np.maximum(best, vessel)

    b_max = float(best.max())
    b_min = float(best.min())
    if b_max > b_min:
        best_u8 = ((best - b_min) / (b_max - b_min) * 255.0).astype(np.uint8)
    else:
        best_u8 = np.zeros_like(img, dtype=np.uint8)

    return best_u8
