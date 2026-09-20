import os
import re

file_path = r"d:\Software\antigravity\flux_vision_3d\src\vision\asparagus_analyzer.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 确保 __init__ 中有 vis_stage1, vis_stage2, vis_stage3 等属性
init_old = """        # AprilTag 在线相机定位器 (优先级最高的标定来源)
        self.tag_localizer = None  # type: Optional["TagLocalizer"]
        self.last_valid_tag_transform: Optional[np.ndarray] = None  # 历史锁定外参缓存
        self.last_tag_info: dict = {}  # 上一帧 AprilTag 定位的诊断信息"""

init_new = """        # AprilTag 在线相机定位器 (优先级最高的标定来源)
        self.tag_localizer = None  # type: Optional["TagLocalizer"]
        self.last_valid_tag_transform: Optional[np.ndarray] = None  # 历史锁定外参缓存
        self.last_tag_info: dict = {}  # 上一帧 AprilTag 定位的诊断信息

        # 三阶段透明流水线调试图与中间缓存
        self.vis_stage1: Optional[np.ndarray] = None
        self.vis_stage2: Optional[np.ndarray] = None
        self.vis_stage3: Optional[np.ndarray] = None
        self.last_pipeline_targets: List[AsparagusTarget] = []"""

assert init_old in code, "init_old not found!"
code = code.replace(init_old, init_new)

# 2. 替换 analyze, _analyze_2d 及相关逻辑，实现三阶段透明流水线
# 寻找从 def segment_and_separate 到 def draw_detections 之前的内容
target_start_marker = "    def segment_and_separate("
target_end_marker = "    def draw_detections("

idx_start = code.find(target_start_marker)
idx_end = code.find(target_end_marker)
assert idx_start != -1 and idx_end != -1, "markers not found!"

new_pipeline_code = '''    def extract_stage1_foreground(self, color_bgr: np.ndarray, depth_mm: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        【阶段 1】前景物料提取 (ExG 超绿 + 传送带 ROI 约束)
        :return: (fg_mask_full, (roi_x1, roi_x2, roi_y1, roi_y2))
        """
        h, w = color_bgr.shape[:2]
        # 传送带核心作业区域 ROI (自动剔除左右两侧支架和反光区域)
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
        b, g, r = roi_bgr[:, :, 0], roi_bgr[:, :, 1], roi_bgr[:, :, 2]

        # 超绿特征 ExG 与亮度/色差联合约束
        exg = 2.0 * g - r - b
        gray = cv2.cvtColor(color_bgr[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
        
        # 纯植物绿/嫩黄绿提取：ExG > 10 或 G 显著大于 B 且具有基本亮度
        fg_roi = ((exg > 10.0) | ((g > b * 1.05) & (gray > 42))).astype(np.uint8) * 255

        # 微弱开运算消除极微小反光毛刺 (严禁大闭运算，坚决保护并排芦笋之间的缝隙！)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_OPEN, k_open)

        # 若有深度，排除非工作台深度区域 (350mm ~ 780mm)
        if depth_mm is not None:
            depth_roi = depth_mm[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depth = (depth_roi >= 350) & (depth_roi <= 780)
            # 仅在有深度的区域进行深度约束
            fg_roi = np.where((depth_roi > 0) & (~valid_depth), 0, fg_roi)

        fg_full = np.zeros((h, w), dtype=np.uint8)
        fg_full[roi_y1:roi_y2, roi_x1:roi_x2] = fg_roi

        # 阶段 1 可视化渲染：原图压暗 + 绿色荧光高亮物料 + ROI 引导框
        vis = (color_bgr.astype(np.float32) * 0.45).astype(np.uint8)
        green_layer = vis.copy()
        green_layer[fg_full > 0] = (40, 235, 90)
        vis = cv2.addWeighted(green_layer, 0.65, vis, 0.35, 0)
        
        # 绘制传送带作业 ROI
        cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.putText(vis, "CONVEYOR WORKSPACE ROI", (roi_x1 + 8, roi_y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        # 统计前景像素面积
        fg_pixels = int(np.count_nonzero(fg_full))
        badge = f"STAGE 1: FOREGROUND (ExG+ROI) | Pixels: {fg_pixels}"
        cv2.rectangle(vis, (12, 12), (520, 48), (20, 20, 20), -1)
        cv2.rectangle(vis, (12, 12), (520, 48), (0, 235, 90), 2)
        put_text(vis, badge, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

        self.vis_stage1 = vis
        return fg_full, (roi_x1, roi_x2, roi_y1, roi_y2)

    def extract_stage2_spines(
        self,
        color_bgr: np.ndarray,
        fg_mask: np.ndarray,
        roi_box: Tuple[int, int, int, int],
        depth_mm: Optional[np.ndarray] = None,
        nominal_z_mm: float = 640.0
    ) -> List[dict]:
        """
        【阶段 2】独立脊线骨架与单体验证 (Spine & Ridge Tracing)
        原理：
          在欧氏距离变换场 (Distance Transform) 中，不论多根芦笋如何并排挨着，
          每一根芦笋的中轴线上都是截面半径的局部极大值峰（Ridge）。
          沿垂向 (Y 轴) 提取局部极大值峰线，横向 (X 轴) 桥接，即可彻底切分并排粘连物料！
        :return: 候选芦笋脊线字典列表
        """
        h, w = color_bgr.shape[:2]
        roi_x1, roi_x2, roi_y1, roi_y2 = roi_box
        fg_roi = fg_mask[roi_y1:roi_y2, roi_x1:roi_x2]

        if np.count_nonzero(fg_roi) < 200:
            self.vis_stage2 = color_bgr.copy()
            return []

        # 1. 距离变换
        dist = cv2.distanceTransform(fg_roi, cv2.DIST_L2, 5)

        # 2. 垂向局部极大值提取 (切分并排挨着的芦笋)
        kernel_v = np.ones((7, 1), np.uint8)
        dist_dil_v = cv2.dilate(dist, kernel_v)
        peaks = (dist == dist_dil_v) & (dist >= 4.0)

        # 3. 沿芦笋主轴方向横向桥接脊线
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        peaks_connected = cv2.morphologyEx(peaks.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_h)

        cnts, _ = cv2.findContours(peaks_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale = (nominal_z_mm / self.fx) if depth_mm is None else None

        candidates = []
        # 可视化图底图
        vis = color_bgr.copy()
        overlay = vis.copy()

        # 调色盘区分不同芦笋实例
        palette = [
            (255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120),
            (255, 210, 40), (120, 160, 255), (200, 255, 80), (255, 80, 140)
        ]

        cand_id = 1
        for c in cnts:
            pts = c.reshape(-1, 2)
            if len(pts) < 8:
                continue

            rect = cv2.minAreaRect(c)
            (rcx, rcy), (rw, rh), _ = rect
            l_approx = max(rw, rh)
            if l_approx < 80:
                continue

            # 主轴拟合
            [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx[0]), float(vy[0])
            if abs(vx) < 0.45:
                continue  # 芦笋应大致平行传送带输送方向
            if vx < 0:
                vx, vy = -vx, -vy

            # 投影计算脊线长度
            proj = np.dot(pts - np.array([rcx, rcy]), np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            len_px = float(max_p - min_p)
            if len_px < 100:
                continue

            # 在脊线上多点采样距离场半径
            sampled_radii = []
            spine_pts_img = []
            for s in np.linspace(min_p * 0.15, max_p * 0.85, 16):
                sx = int(round(rcx + s * vx))
                sy = int(round(rcy + s * vy))
                if 0 <= sx < fg_roi.shape[1] and 0 <= sy < fg_roi.shape[0]:
                    sampled_radii.append(float(dist[sy, sx]))
                    spine_pts_img.append((sx + roi_x1, sy + roi_y1))

            if not sampled_radii:
                continue

            avg_rad = float(np.median(sampled_radii))
            diam_px = float(avg_rad * 2.0)

            # 图像绝对中心
            global_cx = float(rcx + roi_x1)
            global_cy = float(rcy + roi_y1)

            # 物理尺度换算与先验尺寸过滤
            if depth_mm is not None:
                # 采样脊线上的真实深度
                sample_depths = []
                for (px_x, px_y) in spine_pts_img:
                    if 0 <= px_x < w and 0 <= px_y < h:
                        d_val = depth_mm[px_y, px_x]
                        if 350 <= d_val <= 780:
                            sample_depths.append(d_val)
                z_ref = float(np.median(sample_depths)) if len(sample_depths) >= 4 else nominal_z_mm
                local_scale = z_ref / self.fx
            else:
                z_ref = nominal_z_mm
                local_scale = scale

            l_mm = float(len_px * local_scale)
            d_mm = float(diam_px * local_scale)

            # 物理尺寸合规性校验：充分支持 6mm ~ 48mm 粗细 (涵盖特级粗笋 35mm)，长度 > 100mm
            if not (6.0 <= d_mm <= 48.0 and 100.0 <= l_mm <= 600.0):
                continue

            yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
            if yaw_deg > 90.0: yaw_deg -= 180.0
            elif yaw_deg < -90.0: yaw_deg += 180.0

            # 构造紧凑外接定向矩形角点
            u_vec = np.array([vx, vy])
            v_vec = np.array([-vy, vx])
            half_l = len_px * 0.5
            half_w = max(diam_px * 0.5, 4.0)

            c_pt = np.array([global_cx, global_cy])
            p1 = c_pt - half_l * u_vec - half_w * v_vec
            p2 = c_pt + half_l * u_vec - half_w * v_vec
            p3 = c_pt + half_l * u_vec + half_w * v_vec
            p4 = c_pt - half_l * u_vec + half_w * v_vec
            box_corners = np.array([p1, p2, p3, p4], dtype=np.int32)

            color_theme = palette[(cand_id - 1) % len(palette)]

            # 阶段 2 可视化绘制：半透明定向外框 + 脊线中轴 + 采样半径圈
            cv2.fillPoly(overlay, [box_corners], color_theme)
            cv2.polylines(vis, [box_corners], True, color_theme, 2)
            
            # 白色高亮中心脊线
            sp_p1 = (int(global_cx - half_l * vx), int(global_cy - half_l * vy))
            sp_p2 = (int(global_cx + half_l * vx), int(global_cy + half_l * vy))
            cv2.line(vis, sp_p1, sp_p2, (255, 255, 255), 2)

            # 采样圆指示
            for (px_x, px_y) in spine_pts_img[::4]:
                cv2.circle(vis, (px_x, px_y), max(2, int(diam_px * 0.5)), (255, 255, 200), 1)

            # 实例标签
            tag_str = f"#{cand_id} D:{d_mm:.1f} L:{l_mm:.0f}"
            put_text(vis, tag_str, (int(global_cx - 30), int(global_cy - half_w - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            candidates.append({
                'id': cand_id,
                'center_px': (global_cx, global_cy),
                'length_px': len_px,
                'diam_px': diam_px,
                'length_mm': round(l_mm, 1),
                'diam_mm': round(d_mm, 1),
                'yaw_deg': round(yaw_deg, 1),
                'axis_vector': (vx, vy),
                'box_corners': box_corners,
                'z_ref': z_ref,
                'spine_pts': spine_pts_img
            })
            cand_id += 1

        # 混合半透明图层
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        badge2 = f"STAGE 2: SPINES & RIDGES | Identified Instances: {len(candidates)}"
        cv2.rectangle(vis, (12, 12), (540, 48), (20, 20, 20), -1)
        cv2.rectangle(vis, (12, 12), (540, 48), (40, 230, 240), 2)
        put_text(vis, badge2, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 230, 240), 2)

        self.vis_stage2 = vis
        return candidates

    def estimate_stage3_poses(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        spines: List[dict],
        plane_coeff: Optional[np.ndarray] = None,
        frame_transform: Optional[np.ndarray] = None,
        frame_calib_source: str = "uncalibrated"
    ) -> List[AsparagusTarget]:
        """
        【阶段 3】位姿解算与排名前三位 (Top 3) 可抓取物料输出
        :return: 严格按优先级排序的排名前三位 AsparagusTarget 列表
        """
        targets: List[AsparagusTarget] = []

        for sp in spines:
            cx_val, cy_val = sp['center_px']
            vx_val, vy_val = sp['axis_vector']
            length_px = sp['length_px']
            diam_px = sp['diam_px']
            length_mm = sp['length_mm']
            diam_mm = sp['diam_mm']
            yaw_deg = sp['yaw_deg']
            box_corners = sp['box_corners']
            z_ref = sp['z_ref']

            if depth_mm is not None:
                # 采样沿中轴脊线的顶层深度
                spine_pts = sp.get('spine_pts', [])
                valid_ds = []
                for (px_x, px_y) in spine_pts:
                    if 0 <= px_x < color_bgr.shape[1] and 0 <= px_y < color_bgr.shape[0]:
                        dv = depth_mm[px_y, px_x]
                        if 350 <= dv <= 780:
                            valid_ds.append(dv)
                
                if len(valid_ds) >= 3:
                    z_top = float(np.percentile(valid_ds, 15))
                    z_med = float(np.median(valid_ds))
                else:
                    z_top = z_ref
                    z_med = z_ref

                # 计算相机坐标系下的 3D 抓取中心点 (X, Y, Z)
                grip_x = float((cx_val - self.cx) * z_med / self.fx)
                grip_y = float((cy_val - self.cy) * z_med / self.fy)
                grip_z = float(z_top)

                # 计算相对工作台的凸起净高度 (mm)
                if plane_coeff is not None:
                    table_z_local = plane_coeff[0] * cx_val + plane_coeff[1] * cy_val + plane_coeff[2]
                    rel_height_mm = float(table_z_local - z_top)
                else:
                    rel_height_mm = float(640.0 - z_top)
            else:
                z_top = 0.0
                z_med = z_ref
                grip_x = float((cx_val - self.cx) * z_ref / self.fx)
                grip_y = float((cy_val - self.cy) * z_ref / self.fy)
                grip_z = 0.0
                rel_height_mm = 0.0

            # SCARA 抓取坐标系转换
            if frame_transform is not None:
                p_cam_h = np.array([grip_x, grip_y, grip_z, 1.0])
                p_robot_h = frame_transform @ p_cam_h
                robot_x = float(p_robot_h[0])
                robot_y = float(p_robot_h[1])
                robot_z = float(p_robot_h[2])

                r_mat = frame_transform[:3, :3]
                v_cam = np.array([vx_val, vy_val, 0.0])
                v_robot = r_mat @ v_cam
                r_rad = np.arctan2(v_robot[1], v_robot[0])
                robot_r = float(np.degrees(r_rad))
                if robot_r > 90.0: robot_r -= 180.0
                elif robot_r < -90.0: robot_r += 180.0
            else:
                robot_x = float(grip_x)
                robot_y = float(grip_y)
                robot_z = float(rel_height_mm)
                robot_r = float(yaw_deg)

            target = AsparagusTarget(
                id=sp['id'],
                center_px=(cx_val, cy_val),
                length_px=length_px,
                diam_px=diam_px,
                yaw_deg=round(yaw_deg, 1),
                axis_vector=(vx_val, vy_val),
                box_corners=box_corners,
                contour=box_corners,
                length_mm=round(length_mm, 1),
                diam_mm=round(diam_mm, 1),
                grip_x=round(grip_x, 1),
                grip_y=round(grip_y, 1),
                grip_z=round(grip_z, 1),
                z_top=round(z_top, 1),
                rel_height_mm=round(rel_height_mm, 1),
                robot_x=round(robot_x, 1),
                robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1),
                robot_r=round(robot_r, 1),
                is_topmost=False,
                calibration_source=frame_calib_source
            )
            targets.append(target)

        # 排序：优先按相对台面凸起高度降序；纯 2D 时按面积和居中度排序
        if len(targets) > 0:
            if depth_mm is not None:
                targets.sort(key=lambda t: t.rel_height_mm, reverse=True)
            else:
                targets.sort(key=lambda t: (t.length_px * t.diam_px), reverse=True)

            # 严格保留排名前三位 (Top 3)
            targets = targets[:3]
            for rank_i, t in enumerate(targets):
                t.id = rank_i + 1
                t.is_topmost = (rank_i == 0)

        self.last_pipeline_targets = targets
        self.vis_stage3 = self.draw_detections(color_bgr, targets, sel_target_idx=0)
        return targets

    def analyze(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        stages: Tuple[bool, bool, bool] = (True, True, True)
    ) -> List[AsparagusTarget]:
        """
        三阶段透明流水线端到端解算入口
        :param stages: (run_stage1, run_stage2, run_stage3) 是否执行各阶段
        :return: 最终排名前三位的识别目标 (若未执行阶段 3 则返回空列表)
        """
        run_s1, run_s2, run_s3 = stages

        # 标定与平面拟合准备
        frame_transform, frame_calib_source = self._resolve_calibration(color_bgr)
        plane_coeff = self.fit_table_plane(depth_mm) if depth_mm is not None else None

        # 阶段 1：前景物料提取
        if not run_s1:
            self.vis_stage1 = None
            self.vis_stage2 = None
            self.vis_stage3 = None
            return []

        fg_mask, roi_box = self.extract_stage1_foreground(color_bgr, depth_mm)

        # 阶段 2：独立脊线骨架与单体验证
        if not run_s2:
            self.vis_stage2 = None
            self.vis_stage3 = None
            return []

        nominal_z = 640.0
        if plane_coeff is not None and abs(plane_coeff[2]) > 300:
            nominal_z = float(plane_coeff[2])

        spines = self.extract_stage2_spines(
            color_bgr, fg_mask, roi_box, depth_mm=depth_mm, nominal_z_mm=nominal_z
        )

        # 阶段 3：位姿解算与 Top 3 输出
        if not run_s3:
            self.vis_stage3 = None
            return []

        targets = self.estimate_stage3_poses(
            color_bgr, depth_mm, spines,
            plane_coeff=plane_coeff,
            frame_transform=frame_transform,
            frame_calib_source=frame_calib_source
        )

        return targets

'''

new_full_code = code[:idx_start] + new_pipeline_code + code[idx_end:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_full_code)

print("Updated asparagus_analyzer.py successfully!")
