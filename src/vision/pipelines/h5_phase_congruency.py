import numpy as np
import cv2
from skimage import filters

def phase_congruency(image: np.ndarray, scales: list = [1.5, 3.0], orientations: int = 4) -> np.ndarray:
    """Fast approximate phase‑congruency edge detection with multiscale pyramid.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8 or float).
    scales : list
        List of sigma values for multi‑scale analysis.
    orientations : int
        Number of orientations to compute.

    Returns
    -------
    pc_map : np.ndarray
        Edge strength map normalized to 0‑255 (uint8).
    """
    orig_h, orig_w = image.shape[:2]

    # 若尺寸较大，降采样到 1/2 尺寸加速 4 倍，抗噪性更优
    downscale = (orig_h > 400 or orig_w > 400)
    if downscale:
        img_small = cv2.resize(image, (orig_w // 2, orig_h // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        img_small = image.astype(np.float32)

    accum = np.zeros_like(img_small, dtype=np.float32)
    for sigma in scales:
        s = sigma * 0.7 if downscale else sigma
        ksize = int(6 * s + 1)
        if ksize % 2 == 0:
            ksize += 1
        for theta in np.linspace(0, np.pi, orientations, endpoint=False):
            kernel = cv2.getGaborKernel((ksize, ksize), s, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img_small, cv2.CV_32F, kernel)
            accum = np.maximum(accum, np.abs(filtered))

    pc_small = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if downscale:
        pc = cv2.resize(pc_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        pc = pc_small

    return pc
