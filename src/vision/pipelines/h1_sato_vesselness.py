import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def compute_sato_vesselness(volume: np.ndarray, scales: list = [1, 2, 3]) -> np.ndarray:
    """Compute Sato vesselness for a 2D grayscale image.

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
        # second‑order derivatives
        Ixx = gaussian_filter(img, sigma, order=(2, 0))
        Iyy = gaussian_filter(img, sigma, order=(0, 2))
        Ixy = gaussian_filter(img, sigma, order=(1, 1))
        # Hessian matrix per pixel
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                H = np.array([[Ixx[y, x], Ixy[y, x]], [Ixy[y, x], Iyy[y, x]]])
                ev = np.linalg.eigvalsh(H)
                # Sato: both eigenvalues negative for bright tubular structures
                if ev[0] < 0 and ev[1] < 0:
                    vessel = np.abs(ev[0] * ev[1])
                    if vessel > best[y, x]:
                        best[y, x] = vessel
    # normalize to 0-255 uint8
    if best.max() > best.min():
        best = ((best - best.min()) / (best.max() - best.min()) * 255).astype(np.uint8)
    else:
        best = best.astype(np.uint8)
    return best
