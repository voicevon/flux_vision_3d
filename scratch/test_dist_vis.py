import cv2
import numpy as np

img = cv2.imread('data/workspaces/20260916_145246_Home_real/production/raw_images/view_0006.png')
h, w = img.shape[:2]
roi_x1, roi_x2 = int(w * 0.35), int(w * 0.81)
roi_y1, roi_y2 = int(h * 0.02), int(h * 0.98)
roi_bgr = img[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
b, g, r = roi_bgr[:, :, 0], roi_bgr[:, :, 1], roi_bgr[:, :, 2]
exg = 2.0 * g - r - b
gray = cv2.cvtColor(img[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
fg_roi = ((exg > 10.0) | ((g > b * 1.05) & (gray > 42))).astype(np.uint8) * 255
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_OPEN, k_open)

# 1. 距离变换
dist = cv2.distanceTransform(fg_roi, cv2.DIST_L2, 5)

# 生成 vis_dist: 全黑底图或微弱暗色原图
vis_dist = (img.astype(np.float32) * 0.25).astype(np.uint8)
# 归一化距离场 (0 ~ 28px 映射到 0 ~ 255)
dist_norm = np.clip(dist / 28.0 * 255.0, 0, 255).astype(np.uint8)
dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
# 仅在前景区域混合距离场颜色
dist_roi_canvas = vis_dist[roi_y1:roi_y2, roi_x1:roi_x2]
fg_bool = fg_roi > 0
dist_roi_canvas[fg_bool] = dist_color[fg_bool]
vis_dist[roi_y1:roi_y2, roi_x1:roi_x2] = dist_roi_canvas
cv2.rectangle(vis_dist, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
cv2.putText(vis_dist, "EUCLIDEAN DISTANCE FIELD (Radius Energy)", (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 230, 240), 2)

# 2. 垂向极大值提取
kernel_v = np.ones((7, 1), np.uint8)
dist_dil_v = cv2.dilate(dist, kernel_v)
peaks = (dist == dist_dil_v) & (dist >= 4.0)

# 生成 vis_peaks: 在距离场或原图上点亮高亮极大值峰点
vis_peaks = (img.astype(np.float32) * 0.35).astype(np.uint8)
# 绘制微弱物料轮廓
cnts_fg, _ = cv2.findContours(fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cf in cnts_fg:
    cf_shift = cf + np.array([roi_x1, roi_y1])
    cv2.drawContours(vis_peaks, [cf_shift], -1, (60, 100, 70), 1)

# 点亮极大值点
py, px = np.where(peaks)
for y_pt, x_pt in zip(py, px):
    gx, gy = x_pt + roi_x1, y_pt + roi_y1
    cv2.drawMarker(vis_peaks, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 4, 1)

cv2.rectangle(vis_peaks, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
cv2.putText(vis_peaks, "TRANSVERSE NMS RIDGE PEAKS (Peak Points)", (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv2.imwrite("scratch/demo_vis_dist.png", vis_dist)
cv2.imwrite("scratch/demo_vis_peaks.png", vis_peaks)
print("Demo distance & peak images created successfully!")
