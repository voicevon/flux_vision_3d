import os

file_path = r"d:\Software\antigravity\flux_vision_3d\src\vision\asparagus_analyzer.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 在 __init__ 增加 vis_dist 和 vis_peaks
old_init = """        # 三阶段透明流水线调试图与中间缓存
        self.vis_stage1: Optional[np.ndarray] = None
        self.vis_stage2: Optional[np.ndarray] = None
        self.vis_stage3: Optional[np.ndarray] = None
        self.last_pipeline_targets: List[AsparagusTarget] = []"""

new_init = """        # CV 视觉识别流水线调试图与中间过程缓存
        self.vis_stage1: Optional[np.ndarray] = None
        self.vis_dist: Optional[np.ndarray] = None       # 欧氏距离变换场 (Radius Energy Map)
        self.vis_peaks: Optional[np.ndarray] = None      # 垂向极大值峰脊线 (Transverse Ridge Peaks)
        self.vis_stage2: Optional[np.ndarray] = None
        self.vis_stage3: Optional[np.ndarray] = None
        self.last_pipeline_targets: List[AsparagusTarget] = []"""

assert old_init in code, "old_init not found"
code = code.replace(old_init, new_init)

# 2. 在 extract_stage2_spines 中渲染 vis_dist 和 vis_peaks
old_dist_block = """        # 1. 距离变换
        dist = cv2.distanceTransform(fg_roi, cv2.DIST_L2, 5)

        # 2. 垂向局部极大值提取 (切分并排挨着的芦笋)
        kernel_v = np.ones((7, 1), np.uint8)
        dist_dil_v = cv2.dilate(dist, kernel_v)
        peaks = (dist == dist_dil_v) & (dist >= 4.0)

        # 3. 沿芦笋主轴方向横向桥接脊线
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        peaks_connected = cv2.morphologyEx(peaks.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_h)"""

new_dist_block = """        # 1. 距离变换 (计算前景像素到背景边界的最短欧氏距离，值即代表截面半径)
        dist = cv2.distanceTransform(fg_roi, cv2.DIST_L2, 5)

        # 生成【距离变换场】可视化图 (COLORMAP_TURBO 热力图，直观展现物料半径能量分布与贴合鞍部)
        vis_d = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        dist_norm = np.clip(dist / 28.0 * 255.0, 0, 255).astype(np.uint8)
        dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
        roi_patch = vis_d[roi_y1:roi_y2, roi_x1:roi_x2]
        fg_bool = fg_roi > 0
        roi_patch[fg_bool] = dist_color[fg_bool]
        vis_d[roi_y1:roi_y2, roi_x1:roi_x2] = roi_patch
        cv2.rectangle(vis_d, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_d, (12, 12), (560, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_d, (12, 12), (560, 48), (40, 230, 240), 2)
        put_text(vis_d, "CV: DISTANCE TRANSFORM (Radius Field & Seam Valleys)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        self.vis_dist = vis_d

        # 2. 垂向局部极大值提取 (沿垂直芦笋长轴跨度方向做非极大值抑制 NMS，在贴合处天然出现凹陷谷底，只在物料中轴取峰值)
        kernel_v = np.ones((7, 1), np.uint8)
        dist_dil_v = cv2.dilate(dist, kernel_v)
        peaks = (dist == dist_dil_v) & (dist >= 4.0)

        # 生成【极大值峰脊线】可视化图 (点亮非极大值抑制后的中轴峰值点阵，证明并排缝隙处的数学解耦)
        vis_p = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        cnts_fg, _ = cv2.findContours(fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cf in cnts_fg:
            cv2.drawContours(vis_p, [cf + np.array([roi_x1, roi_y1])], -1, (60, 110, 75), 1)
        py, px = np.where(peaks)
        for y_pt, x_pt in zip(py, px):
            gx, gy = x_pt + roi_x1, y_pt + roi_y1
            cv2.drawMarker(vis_p, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 4, 1)
        cv2.rectangle(vis_p, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_p, (12, 12), (580, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_p, (12, 12), (580, 48), (0, 255, 255), 2)
        put_text(vis_p, "CV: TRANSVERSE NMS RIDGE PEAKS (De-coupling Seams)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        self.vis_peaks = vis_p

        # 3. 沿芦笋主轴方向横向形态学闭运算桥接 (形成连续中轴骨架)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        peaks_connected = cv2.morphologyEx(peaks.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_h)"""

assert old_dist_block in code, "old_dist_block not found"
code = code.replace(old_dist_block, new_dist_block)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated asparagus_analyzer.py with intermediate CV visualizations!")
