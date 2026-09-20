import cv2
import numpy as np

img = cv2.imread('data/workspaces/20260916_145246_Home_real/production/raw_images/view_0006.png')
h, w = img.shape[:2]
roi_x1, roi_x2 = int(w * 0.36), int(w * 0.79)
roi = img[:, roi_x1:roi_x2].copy()

b, g, r = cv2.split(roi.astype(float))
exg = 2.0 * g - r - b
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
fg = ((exg > 10) | ((g > b * 1.05) & (gray > 45))).astype(np.uint8) * 255
fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

kernel_v = np.ones((7, 1), np.uint8)
dist_dil_v = cv2.dilate(dist, kernel_v)
peaks = (dist == dist_dil_v) & (dist >= 5.0)

# 横向桥接
k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
peaks_connected = cv2.morphologyEx(peaks.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_h)

cnts, _ = cv2.findContours(peaks_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Candidate contours:', len(cnts))

scale = 640.0 / 909.12
targets = []
for c in cnts:
    pts = c.reshape(-1, 2)
    if len(pts) < 10:
        continue
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect
    l_px = max(rw, rh)
    if l_px < 150:
        continue
    [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx[0]), float(vy[0])
    if abs(vx) < 0.5:
        continue
    if vx < 0:
        vx, vy = -vx, -vy
    
    proj = np.dot(pts - np.array([cx, cy]), np.array([vx, vy]))
    min_p, max_p = np.min(proj), np.max(proj)
    len_px = max_p - min_p
    
    sampled_radii = []
    for s in np.linspace(min_p * 0.2, max_p * 0.8, 15):
        sx = int(round(cx + s * vx))
        sy = int(round(cy + s * vy))
        if 0 <= sx < roi.shape[1] and 0 <= sy < roi.shape[0]:
            sampled_radii.append(dist[sy, sx])
    
    avg_rad = np.median(sampled_radii) if sampled_radii else 10.0
    diam_px = float(avg_rad * 2.0)
    
    l_mm = len_px * scale
    d_mm = diam_px * scale
    if 6.0 <= d_mm <= 45.0 and l_mm >= 120.0:
        targets.append({
            'cx': cx + roi_x1, 'cy': cy,
            'vx': vx, 'vy': vy,
            'len_px': len_px, 'diam_px': diam_px,
            'l_mm': round(l_mm, 1), 'd_mm': round(d_mm, 1),
            'yaw': round(float(np.degrees(np.arctan2(vy, vx))), 1)
        })

print('Valid single asparagus targets:', len(targets))
for i, t in enumerate(targets):
    print(f"Target #{i+1}: D={t['d_mm']}mm, L={t['l_mm']}mm, Yaw={t['yaw']}deg, center=({t['cx']:.1f}, {t['cy']:.1f})")
