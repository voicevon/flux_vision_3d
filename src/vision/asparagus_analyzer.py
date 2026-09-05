"""
芦笋 3D 视觉特征分析与最顶层目标解算模块
=====================================================
核心职责：
  1. 2D 目标轮廓分割与细长体形态学滤波
  2. 工作台面最小二乘平面拟合，解算物料相对工作台的凸起净高度 (Relative Height)
  3. 基于 fitLine 主成分直线拟合，高精度解算芦笋中轴倾角 (Yaw deg) 与沿轴中心线
  4. 芦笋物理尺寸解算：物理长度 (Length mm)、截面直径 (Diameter mm)
  5. 掩膜形态学腐蚀与中轴脊线抗噪深度采样 (Z_top, Z_center)
  6. 叠压拓扑分析，锁定最上层可抓取目标 (Topmost Pickable Target)
  7. 解算并输出 SCARA 机械臂夹爪目标位姿 (X, Y, Z, R)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class AsparagusTarget:
    """单根芦笋感知与几何特征数据"""
    id: int
    center_px: Tuple[float, float]       # 图像像素中心 (u, v)
    length_px: float                     # 像素长度
    diam_px: float                       # 像素直径
    yaw_deg: float                       # 轴线在水平面的倾角 [-90°, 90°]，即夹爪目标旋转角 R
    axis_vector: Tuple[float, float]     # 沿长轴的归一化方向向量 (vx, vy)
    box_corners: np.ndarray              # 最小外接矩形 4 个角点像素坐标
    contour: np.ndarray                  # 轮廓多边形
    
    # 3D 物理尺寸 (单位: mm)
    length_mm: float                     # 物理实际长度
    diam_mm: float                       # 物理平均直径
    
    # 相机坐标系下的物理抓取位姿 (单位: mm, deg)
    grip_x: float                        # 夹取点 X
    grip_y: float                        # 夹取点 Y
    grip_z: float                        # 夹取点 Z (下探目标高度)
    z_top: float                         # 芦笋顶面绝对深度 (越小越靠近相机)
    rel_height_mm: float                 # 相对工作台面的凸起净高度 (越大越在上层)
    
    is_topmost: bool = False             # 是否被判定为最顶层目标


class AsparagusAnalyzer:
    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        """
        初始化分析器
        :param fx, fy, cx, cy: 彩色相机内参矩阵参数
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # 尺寸与滤波超参数 (匹配真实芦笋形态)
        self.min_area = 5000             # 最小像素面积
        self.max_area = 25000            # 最大像素面积 (过滤大块背景边框)
        self.min_length_mm = 180.0       # 最小物理长度 (mm)
        self.max_length_mm = 320.0       # 最大物理长度 (mm)
        self.min_diam_mm = 10.0          # 最小物理直径 (mm)
        self.max_diam_mm = 35.0          # 最大物理直径 (mm)
        self.min_aspect_ratio = 4.0      # 最小长宽比 (细长物料)
        self.erosion_kernel = np.ones((5, 5), np.uint8)

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """动态更新内参"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def fit_table_plane(self, depth_mm: np.ndarray) -> Optional[np.ndarray]:
        """
        使用最小二乘法拟合工作台背景平面方程: Z_table(x, y) = a*x + b*y + d
        消除相机俯视时轻微俯仰/横滚角度引起的整幅图像渐变倾斜
        """
        h, w = depth_mm.shape
        valid = (depth_mm > 350) & (depth_mm < 650)
        if np.count_nonzero(valid) < 5000:
            return None
        
        # 降采样快速拟合
        step = 8
        y_grid, x_grid = np.indices((h, w))
        x_sub = x_grid[::step, ::step][valid[::step, ::step]]
        y_sub = y_grid[::step, ::step][valid[::step, ::step]]
        z_sub = depth_mm[::step, ::step][valid[::step, ::step]]
        
        if len(z_sub) < 500:
            return None
            
        A = np.column_stack([x_sub, y_sub, np.ones_like(x_sub)])
        plane_coeff, _, _, _ = np.linalg.lstsq(A, z_sub, rcond=None)
        return plane_coeff

    def segment_foreground(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        """
        前背景多特征自适应分割：融合深度距离门限与颜色反差
        兼容新鲜绿色芦笋以及蓝绿色仿真教学试棒，彻底隔离黄色工作台/木板
        """
        h, w = color_bgr.shape[:2]
        
        # 1. 深度有效区间门限 (350mm ~ 620mm)
        depth_valid = (depth_mm >= 350) & (depth_mm <= 620)
        
        # 2. 颜色特征提取：
        # 特征 A: 真实嫩绿芦笋 HSV 空间 (Hue 36~85 为纯绿，严格避开 20~35 的黄色木纹)
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, np.array([36, 40, 40]), np.array([85, 255, 255]))
        
        # 特征 B: 冷色差分 (针对浅黄木纹/牛皮纸/桌面背景，芦笋呈高 G/B、低 R 的鲜明冷色)
        b, g, r = cv2.split(color_bgr)
        cold_diff = (b.astype(float) + g.astype(float)) / 2.0 - r.astype(float)
        mask_cold = (cold_diff > 24).astype(np.uint8) * 255
        
        # 综合颜色掩膜
        color_mask = cv2.bitwise_or(mask_green, mask_cold)
        
        # 3. 限制在中心作业有效 ROI (过滤四周机箱线缆与桌面边缘)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h * 0.08):int(h * 0.85), int(w * 0.12):int(w * 0.88)] = 255
        
        combined = cv2.bitwise_and(color_mask, roi_mask)
        combined = cv2.bitwise_and(combined, combined, mask=depth_valid.astype(np.uint8))
        
        # 仅使用 3x3 开运算滤除噪点，保留芦笋紧致轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return combined

    def analyze(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> List[AsparagusTarget]:
        """
        端到端全流程分析：
          1. 分割芦笋轮廓
          2. 基于 fitLine 解算轴线角度与长径尺寸
          3. 脊线采样与工作台倾斜补偿
          4. 叠压拓扑分析并锁定最顶层目标
        """
        plane_coeff = self.fit_table_plane(depth_mm)
        fg_mask = self.segment_foreground(color_bgr, depth_mm)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        targets: List[AsparagusTarget] = []
        target_idx = 1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            
            # 使用 fitLine 进行鲁棒主轴拟合，消除 minAreaRect 的 90 度跳变歧义
            [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            vx_val, vy_val = float(vx[0]), float(vy[0])
            cx_val, cy_val = float(x0[0]), float(y0[0])
            
            # 计算轴线朝向角 (范围 [-90°, 90°])
            angle_rad = np.arctan2(vy_val, vx_val)
            yaw_deg = float(np.degrees(angle_rad))
            if yaw_deg > 90.0: yaw_deg -= 180.0
            elif yaw_deg < -90.0: yaw_deg += 180.0
            
            # 沿轴线与垂轴投影计算长径
            pts = cnt.reshape(-1, 2).astype(float)
            diff = pts - np.array([cx_val, cy_val])
            proj_len = np.dot(diff, np.array([vx_val, vy_val]))
            proj_wid = np.dot(diff, np.array([-vy_val, vx_val]))
            
            length_px = float(np.max(proj_len) - np.min(proj_len))
            diam_px = float(np.max(proj_wid) - np.min(proj_wid))
            
            aspect_ratio = length_px / max(1.0, diam_px)
            if aspect_ratio < self.min_aspect_ratio:
                continue
            
            # 提取中心轴脊线
            c_mask = np.zeros_like(fg_mask)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            spine_mask = cv2.erode(c_mask, self.erosion_kernel, iterations=2)
            
            # 脊线深度统计
            spine_depths = depth_mm[(spine_mask > 0) & (depth_mm > 200) & (depth_mm < 700)]
            if len(spine_depths) < 15:
                spine_depths = depth_mm[(c_mask > 0) & (depth_mm > 200) & (depth_mm < 700)]
            
            if len(spine_depths) == 0:
                continue
            
            # 取前 15% 分位数作为最顶面高度 Z_top，中位数作为中心轴高度
            z_top = float(np.percentile(spine_depths, 15))
            z_med = float(np.median(spine_depths))
            
            # 计算物理尺寸 (mm)
            scale = z_med / self.fx
            length_mm = float(length_px * scale)
            diam_mm = float(diam_px * scale)
            
            # 过滤不符合芦笋物理尺寸的杂物
            if not (self.min_length_mm <= length_mm <= self.max_length_mm):
                continue
            if not (self.min_diam_mm <= diam_mm <= self.max_diam_mm):
                continue
            
            # 计算相机坐标系下的 3D 抓取中心点 (X, Y, Z)
            grip_x = float((cx_val - self.cx) * z_med / self.fx)
            grip_y = float((cy_val - self.cy) * z_med / self.fy)
            grip_z = float(z_top)
            
            # 计算相对工作台的凸起净高度 (mm)
            if plane_coeff is not None:
                table_z_local = plane_coeff[0] * cx_val + plane_coeff[1] * cy_val + plane_coeff[2]
                rel_height_mm = float(table_z_local - z_top)
            else:
                rel_height_mm = float(530.0 - z_top)
            
            rect = cv2.minAreaRect(cnt)
            box_corners = cv2.boxPoints(rect).astype(np.int32)
            
            target = AsparagusTarget(
                id=target_idx,
                center_px=(cx_val, cy_val),
                length_px=length_px,
                diam_px=diam_px,
                yaw_deg=round(yaw_deg, 1),
                axis_vector=(vx_val, vy_val),
                box_corners=box_corners,
                contour=cnt,
                length_mm=round(length_mm, 1),
                diam_mm=round(diam_mm, 1),
                grip_x=round(grip_x, 1),
                grip_y=round(grip_y, 1),
                grip_z=round(grip_z, 1),
                z_top=round(z_top, 1),
                rel_height_mm=round(rel_height_mm, 1),
                is_topmost=False
            )
            targets.append(target)
            target_idx += 1
            
        # 叠压拓扑分析与最顶层判决：
        # 判据：相对工作台面凸起高度最高（rel_height 最大）者为最顶层
        if len(targets) > 0:
            targets.sort(key=lambda t: t.rel_height_mm, reverse=True)
            targets[0].is_topmost = True
            
        return targets

    def draw_detections(self, image: np.ndarray, targets: List[AsparagusTarget]) -> np.ndarray:
        """
        在图像上绘制芦笋轮廓、中轴线、长径尺寸、抓取夹爪十字与最顶层卡片
        """
        annotated = image.copy()
        
        for t in targets:
            is_top = t.is_topmost
            color = (0, 255, 120) if is_top else (220, 180, 50)
            thickness = 3 if is_top else 1
            
            # 1. 绘制最小外接矩形框
            cv2.polylines(annotated, [t.box_corners], True, color, thickness)
            
            # 2. 绘制沿芦笋中心轴线的指引线 (与芦笋主干严格平行)
            cx_int, cy_int = int(t.center_px[0]), int(t.center_px[1])
            vx, vy = t.axis_vector
            half_len = int(t.length_px * 0.45)
            p1 = (int(cx_int - half_len * vx), int(cy_int - half_len * vy))
            p2 = (int(cx_int + half_len * vx), int(cy_int + half_len * vy))
            cv2.line(annotated, p1, p2, (0, 255, 255) if is_top else (180, 180, 180), 2)
            
            # 3. 绘制垂直于轴线的夹爪示意短线 (展示机械臂开合面)
            half_diam = int(max(15, t.diam_px * 0.75))
            perp_p1 = (int(cx_int - half_diam * (-vy)), int(cy_int - half_diam * vx))
            perp_p2 = (int(cx_int + half_diam * (-vy)), int(cy_int + half_diam * vx))
            cv2.line(annotated, perp_p1, perp_p2, (0, 0, 255) if is_top else (200, 200, 0), 2)
            
            # 4. 绘制抓取瞄准十字
            marker_color = (0, 0, 255) if is_top else (200, 200, 0)
            cv2.circle(annotated, (cx_int, cy_int), 6, marker_color, -1)
            cv2.drawMarker(annotated, (cx_int, cy_int), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
            
            # 5. 标注尺寸与位姿标签
            if is_top:
                label_header = f"[TOPMOST] L:{t.length_mm}mm D:{t.diam_mm}mm (+{t.rel_height_mm}mm)"
                label_pose = f"Grip (X:{t.grip_x}, Y:{t.grip_y}, Z:{t.grip_z})mm | R:{t.yaw_deg}deg"
                
                # 顶部高亮大文本卡片
                cv2.rectangle(annotated, (cx_int - 165, cy_int - 56), (cx_int + 225, cy_int - 6), (20, 20, 20), -1)
                cv2.rectangle(annotated, (cx_int - 165, cy_int - 56), (cx_int + 225, cy_int - 6), (0, 255, 120), 2)
                cv2.putText(annotated, label_header, (cx_int - 156, cy_int - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2)
                cv2.putText(annotated, label_pose, (cx_int - 156, cy_int - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)
            else:
                label_simple = f"#{t.id} L:{t.length_mm} D:{t.diam_mm} R:{t.yaw_deg}"
                cv2.putText(annotated, label_simple, (cx_int - 45, cy_int - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        return annotated
