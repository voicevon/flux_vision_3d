import numpy as np
import cv2
try:
    import skfmm  # scikit-fmm for fast marching
except ImportError:
    skfmm = None

def fast_marching_centerline(cost_image: np.ndarray, seeds: list) -> np.ndarray:
    """Compute centreline using Fast‑Marching from given seed points.

    Parameters
    ----------
    cost_image : np.ndarray
        2‑D cost/speed image where lower values indicate easier traversal.
    seeds : list of (int, int)
        Seed points (y, x) to start the propagation.

    Returns
    -------
    centreline_mask : np.ndarray
        Binary mask (uint8) of the extracted centreline.
    """
    if skfmm is None:
        raise ImportError("scikit‑fmm is required for fast marching. Please install via 'pip install scikit-fmm'.")
    # Initialise phi: negative inside seeds, positive elsewhere
    phi = np.full(cost_image.shape, np.inf, dtype=np.float64)
    for y, x in seeds:
        phi[y, x] = -1.0
    # Speed function: inverse of cost (avoid division by zero)
    speed = np.where(cost_image > 0, 1.0 / cost_image.astype(np.float64), 0.0)
    # Compute travel time
    travel_time = skfmm.travel_time(phi, speed, dx=1.0)
    # Threshold to obtain centreline (e.g., gradient of travel time)
    grad_y, grad_x = np.gradient(travel_time)
    magnitude = np.hypot(grad_y, grad_x)
    centreline = (magnitude < np.percentile(magnitude, 5)).astype(np.uint8) * 255
    return centreline
