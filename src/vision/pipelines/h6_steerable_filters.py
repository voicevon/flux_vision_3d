import numpy as np
import cv2

def apply_steerable_filters(image: np.ndarray, directions: int = 8, sigma: float = 1.0) -> np.ndarray:
    """Apply a bank of oriented Gaussian derivative filters (steerable) to enhance linear structures.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8 or float).
    directions : int
        Number of orientations to sample (e.g., 8 means every 22.5°).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    enhanced : np.ndarray
        Image with the maximum response over all orientations, normalized to 0‑255 uint8.
    """
    img = image.astype(np.float32)
    # Generate derivative of Gaussian kernels for each orientation
    ksize = int(6 * sigma + 1)
    responses = []
    for i in range(directions):
        theta = i * np.pi / directions
        # Rotate gradient kernels
        gx = cv2.getDerivKernels(1, 0, ksize)[0].astype(np.float32)
        gy = cv2.getDerivKernels(0, 1, ksize)[0].astype(np.float32)
        # Rotate kernels
        rot_mat = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), np.degrees(theta), 1.0)
        gx_rot = cv2.warpAffine(gx, rot_mat, (ksize, ksize))
        gy_rot = cv2.warpAffine(gy, rot_mat, (ksize, ksize))
        # Combine to oriented derivative
        g = np.cos(theta) * gx_rot + np.sin(theta) * gy_rot
        resp = cv2.filter2D(img, cv2.CV_32F, g)
        responses.append(np.abs(resp))
    # Max over orientations
    max_resp = np.max(np.stack(responses, axis=0), axis=0)
    # Normalize to 0‑255
    enhanced = cv2.normalize(max_resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced
