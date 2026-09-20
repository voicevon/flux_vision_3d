import numpy as np
import cv2
from skimage.morphology import skeletonize

def distance_transform_skeleton(mask: np.ndarray, min_length: int = 30) -> np.ndarray:
    """Compute skeleton of a binary mask using distance transform.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8) where foreground = 255.
    min_length : int, optional
        Minimum length of skeleton branches to keep (in pixels).

    Returns
    -------
    skeleton : np.ndarray
        Binary skeleton mask (uint8) with the same shape as input.
    """
    # Ensure binary
    binary = (mask > 0).astype(np.uint8)
    # Compute distance transform (for possible further use)
    _ = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Skeletonize (2‑D only)
    skel = skeletonize(binary).astype(np.uint8) * 255
    # Remove short branches using simple length filter (connected components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    output = np.zeros_like(skel, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_length:
            output[labels == i] = 255
    return output
