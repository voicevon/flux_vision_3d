import numpy as np
import cv2
from skimage import filters

def phase_congruency(image: np.ndarray, scales: list = [1, 2, 4], orientations: int = 4) -> np.ndarray:
    """Approximate phase‑congruency edge detection.

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
    img = image.astype(np.float32)
    rows, cols = img.shape
    accum = np.zeros_like(img, dtype=np.float32)
    for sigma in scales:
        for theta in np.linspace(0, np.pi, orientations, endpoint=False):
            # Gabor kernel approximates a band‑pass filter for a given orientation
            ksize = int(6 * sigma + 1)
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            accum = np.maximum(accum, np.abs(filtered))
    pc = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return pc
