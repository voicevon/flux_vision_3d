"""
芦笋 3D 视觉特征分析与最顶层目标解算模块 (工业级黑帽暗缝实例切分与顶层解算)
=======================================================================
核心职责：
  1. 3D 深度浮凸主导 + 植物色自适应宽容度前景提取 (彻底免疫黑色/深色工作台)
  2. 工作台面最小二乘平面拟合，解算物料相对工作台的真实凸起净高度 (Relative Height)
  3. 基于水平黑帽暗缝 (Black-Hat Seam Detection) 与形态学切分的多根贴合芦笋实例分离 (Instance Separation)
  4. 基于 fitLine 主成分直线拟合，高精度解算芦笋中轴倾角 (Yaw deg) 与沿轴中心线
  5. 芦笋物理尺寸解算：物理长度 (Length mm)、截面直径 (Diameter mm)
  6. 掩膜中轴脊线抗噪深度采样 (Z_top, Z_center)
  7. 锁定最上层可抓取目标 (Topmost Pickable Target) 并输出 SCARA 夹爪位姿 (X, Y, Z, R)
  8. 集成 AprilTag 多标靶在线相机外参定位 (TagLocalizer)，三级标定降级链
"""

from dataclasses import dataclass, field
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
    length_mm: float                     # 物理实际长度 (经 3D 欧氏反投影恢复)
    diam_mm: float                       # 物理平均直径
    
    # 相机坐标系下的物理空间点 (单位: mm, 原点为镜头光心)
    grip_x: float                        # 相机系 X
    grip_y: float                        # 相机系 Y
    grip_z: float                        # 相机系 Z (沿光轴的镜头绝对深度，如 530mm)
    z_top: float                         # 芦笋顶面绝对深度 (越小越靠近相机)
    rel_height_mm: float                 # 相对工作台面的凸起净高度 (越大越在上层, 如 +45mm)
    
    # 机械臂 SCARA 世界抓取坐标系 (基座法兰原点, 运动面平行于传送带)
    robot_x: float = 0.0                 # SCARA 抓取 X (mm)
    robot_y: float = 0.0                 # SCARA 抓取 Y (mm)
    robot_z: float = 0.0                 # SCARA 垂直下探高度 (mm, 相对传送带台面)
    robot_r: float = 0.0                 # SCARA 夹爪末端偏航角 (deg)
    
    is_topmost: bool = False             # 是否被判定为最顶层目标
    calibration_source: str = "uncalibrated"  # 标定来源: "tag_online" | "tag_cached" | "hand_eye" | "uncalibrated"

    def generate_gcode(self, safe_z: float = 80.0, drop_x: float = 220.0, drop_y: float = 0.0) -> str:
        """
        生成规范、具备防撞防护的 SCARA 抓取 G-code 指令
        杜绝直接输出相机 500+mm 镜头深度导致撞机的严重隐患！
        """
        lines = []
        lines.append(f"; ==============================================================================")
        lines.append(f"; SCARA 抓取指令 (物料 #{self.id} | 长:{self.length_mm}mm 直径:{self.diam_mm}mm 凸起:+{self.rel_height_mm}mm)")
        if self.calibration_source == "uncalibrated":
            lines.append(f"; [安全警告] 当前手眼矩阵尚未标定 (UNCALIBRATED)！")
            lines.append(f"; 坐标模式: 传送带物理基准系 (Z 轴采用凸起高度 {self.robot_z:.1f}mm，已拦截相机 500+mm 深度)")
            lines.append(f"; 实机运行前请完成 AprilTag 建图 (tools/tag_map_builder.py) 或手工标定 (tools/hand_eye_calibration.py)！")
        elif self.calibration_source == "tag_online":
            lines.append(f"; [状态] AprilTag 在线标靶定位 (实时 PnP 外参) 转换至机械臂基座坐标系")
        elif self.calibration_source == "tag_cached":
            lines.append(f"; [状态] AprilTag 历史缓存外参 (标靶暂不可见，沿用上帧锁定值)")
        elif self.calibration_source == "hand_eye":
            lines.append(f"; [状态] 手工 SVD 点触标定矩阵 (T_cam_to_scara) 转换至机械臂基座坐标系")
        else:
            lines.append(f"; [状态] 标定来源: {self.calibration_source}")
        lines.append(f"; ==============================================================================")
        lines.append(f"G90                     ; 绝对坐标模式")
        lines.append(f"G0 Z{safe_z:.1f} F4000          ; 提升至安全过渡高度 (避免平移推撞物料)")
        lines.append(f"G0 X{self.robot_x:.2f} Y{self.robot_y:.2f} R{self.robot_r:.2f} F4000 ; 快速平移旋转对准中轴")
        lines.append(f"M3                      ; 预先开启夹爪 (气动/伺服张开就位)")
        lines.append(f"G1 Z{self.robot_z:.2f} F1500          ; 垂直平稳下探至抓持夹取高度")
        lines.append(f"M4                      ; 闭合夹爪 (牢固夹持物料)")
        lines.append(f"G4 P200                 ; 保压延时 200ms 确保稳固抓持")
        lines.append(f"G0 Z{safe_z:.1f} F4000          ; 提起物料脱离堆叠区")
        lines.append(f"G0 X{drop_x:.2f} Y{drop_y:.2f} R0.00 F4000 ; 移动至分级分料口上方")
        lines.append(f"M3                      ; 释放物料落料")
        return "\n".join(lines)


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
        
        # 尺寸与滤波超参数 (匹配真实多根堆叠细长/特长与粗干芦笋)
        self.min_area = 600              # 最小像素面积
        self.min_length_mm = 60.0        # 最小物理长度 (mm, 涵盖短切与端头)
        self.max_length_mm = 550.0       # 最大物理长度 (mm, 充分容纳 400~500mm 特长芦笋)
        self.min_diam_mm = 5.0           # 最小物理直径 (mm, 涵盖笋尖)
        self.max_diam_mm = 65.0          # 最大物理直径 (mm, 涵盖大直径特级笋及并排局部接触)
        self.min_aspect_ratio = 1.8      # 最小长宽比 (兼顾倾斜投影与局部遮挡)
        self.table_margin_mm = 8.0       # 相对工作台面的凸起门限 (mm, 排除底板杂散反光，放行扁平下压笋)

        # 手眼标定矩阵 (Eye-to-Hand: 相机坐标系 -> SCARA 基座坐标系, 4x4 齐次矩阵)
        # 作为 AprilTag 在线定位不可用时的回退方案
        self.t_cam_to_scara: Optional[np.ndarray] = None
        self.is_hand_eye_calibrated: bool = False

        # AprilTag 在线相机定位器 (优先级最高的标定来源)
        self.tag_localizer = None  # type: Optional["TagLocalizer"]
        self.last_valid_tag_transform: Optional[np.ndarray] = None  # 历史锁定外参缓存
        self.last_tag_info: dict = {}  # 上一帧 AprilTag 定位的诊断信息

    def set_hand_eye_matrix(self, t_matrix: Optional[np.ndarray]):
        """
        设置或更新相机到 SCARA 机械臂基座的手眼标定矩阵 (4x4)
        自动校验矩阵有效性；若仍为单位阵或非合法矩阵，则标记为未标定安全模式
        """
        if t_matrix is not None:
            mat = np.array(t_matrix, dtype=float)
            if mat.shape == (4, 4):
                self.t_cam_to_scara = mat
                is_identity = np.allclose(mat, np.eye(4), atol=1e-3)
                self.is_hand_eye_calibrated = not is_identity
                return
        self.t_cam_to_scara = None
        self.is_hand_eye_calibrated = False

    def set_tag_localizer(self, localizer):
        """
        设置 AprilTag 在线相机定位器实例
        :param localizer: TagLocalizer 实例 (已加载 tags_map.yaml)
        """
        self.tag_localizer = localizer

    def _resolve_calibration(self, color_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        三级标定降级链：解算当前帧的最优坐标变换矩阵与标定来源
        优先级：AprilTag 在线定位 > AprilTag 历史缓存 > 手工 SVD 标定 > 未标定防撞
        :return: (transform_matrix_4x4 或 None, calibration_source 字符串)
        """
        # 第一级：AprilTag 在线定位 (每帧实时 PnP 解算)
        if self.tag_localizer is not None:
            try:
                success, t_cam_to_world, info = self.tag_localizer.localize_camera(color_bgr)
                self.last_tag_info = info
                if success and t_cam_to_world is not None:
                    self.last_valid_tag_transform = t_cam_to_world.copy()
                    return t_cam_to_world, "tag_online"
            except Exception:
                pass  # 定位器异常不应中断主流程

            # 第二级：AprilTag 历史缓存外参 (标靶暂时不可见时沿用上帧)
            if self.last_valid_tag_transform is not None:
                return self.last_valid_tag_transform, "tag_cached"

        # 第三级：手工 SVD 点触标定矩阵 (config.yaml 中的 t_cam_to_scara)
        if self.is_hand_eye_calibrated and self.t_cam_to_scara is not None:
            return self.t_cam_to_scara, "hand_eye"

        # 第四级：完全未标定 — 启用防撞保护模式
        return None, "uncalibrated"

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """动态更新内参"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def fit_table_plane(self, depth_mm: np.ndarray) -> Optional[np.ndarray]:
        """
        使用最小二乘法鲁棒拟合工作台背景平面方程: Z_table(x, y) = a*x + b*y + d
        消除相机俯视时轻微俯仰/横滚角度引起的整幅图像渐变倾斜
        """
        h, w = depth_mm.shape
        # 实测工作台底板距离主要分布在 610mm ~ 670mm 之间
        valid = (depth_mm >= 610) & (depth_mm <= 670)
        if np.count_nonzero(valid) < 1500:
            # 备用方案：取深度较大（后 20% 分位）的有效点作为底板候选
            non_zero = depth_mm[depth_mm > 400]
            if len(non_zero) < 1000:
                return None
            p80 = np.percentile(non_zero, 80)
            valid = (depth_mm >= p80 - 20) & (depth_mm <= p80 + 30)
        
        # 降采样快速拟合
        step = 8
        y_grid, x_grid = np.indices((h, w))
        x_sub = x_grid[::step, ::step][valid[::step, ::step]]
        y_sub = y_grid[::step, ::step][valid[::step, ::step]]
        z_sub = depth_mm[::step, ::step][valid[::step, ::step]]
        
        if len(z_sub) < 300:
            return None
            
        A = np.column_stack([x_sub, y_sub, np.ones_like(x_sub)])
        plane_coeff, _, _, _ = np.linalg.lstsq(A, z_sub, rcond=None)
        return plane_coeff

    def get_table_tilt_angles(self, plane_coeff: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """
        根据传送带拟合平面反算当前相机的物理安装倾角
        返回: (pitch_deg 俯仰角, roll_deg 横滚角, total_tilt_deg 综合空间倾角)
        """
        if plane_coeff is None:
            return (0.0, 0.0, 0.0)
        a, b, d = float(plane_coeff[0]), float(plane_coeff[1]), float(plane_coeff[2])
        if abs(d) < 1e-6:
            return (0.0, 0.0, 0.0)
        roll_deg = float(np.degrees(np.arctan2(a * self.fx, d)))
        pitch_deg = float(np.degrees(np.arctan2(b * self.fy, d)))
        norm_val = np.sqrt(1.0 + (a * self.fx / d)**2 + (b * self.fy / d)**2)
        total_tilt = float(np.degrees(np.arccos(1.0 / norm_val)))
        return (round(pitch_deg, 1), round(roll_deg, 1), round(total_tilt, 1))

    def compute_relative_height(self, depth_mm: np.ndarray, plane_coeff: Optional[np.ndarray]) -> np.ndarray:
        """
        计算每个像素相对工作台面的凸起净高度 (mm)
        Height_rel(x, y) = Z_table(x, y) - Z_actual(x, y)
        """
        h, w = depth_mm.shape
        y_grid, x_grid = np.indices((h, w))
        
        if plane_coeff is not None:
            table_z = plane_coeff[0] * x_grid + plane_coeff[1] * y_grid + plane_coeff[2]
        else:
            valid_depths = depth_mm[depth_mm > 400]
            table_z = np.median(valid_depths) if len(valid_depths) > 0 else 640.0
            
        rel_h = np.where(depth_mm > 0, table_z - depth_mm, 0.0)
        return np.maximum(0.0, rel_h)

    def render_height_map(self, depth_mm: np.ndarray, plane_coeff: Optional[np.ndarray], max_h_mm: float = 80.0) -> np.ndarray:
        """
        渲染传送带校准相对高度热力图 (Table-Relative Height / Elevation Map)
        原理：
          以拟合出的传送带黑色平面作为 Z=0 基准参考系，彻底消除相机大倾角俯视引起的底板倾斜渐变！
          - 台面表面 (rel_h <= 2.5mm)：统一置为平坦无畸变的深冷色基准底板；
          - 凸起物料 (rel_h > 2.5mm)：按相对台面垂直净高度 [0 ~ max_h_mm] 映射为渐变热力色彩：
            * 0~15mm: 青绿 (低层物料/细支)
            * 15~35mm: 翠绿/草绿 (底层芦笋)
            * 35~55mm: 明黄/橙色 (中层芦笋)
            * 55~80mm+: 鲜艳暖红/亮红 (最顶层优先抓取芦笋)
          并在画面右侧绘制精美的高度色标尺 (Height Color Bar)，标注毫米物理刻度。
        """
        h, w = depth_mm.shape
        rel_h = self.compute_relative_height(depth_mm, plane_coeff)
        
        # 归一化映射 (0 ~ max_h_mm) -> (0 ~ 255)
        clamped_h = np.clip(rel_h - 2.5, 0.0, max_h_mm)
        norm_h = (clamped_h / max_h_mm * 255.0).astype(np.uint8)
        
        # 使用 COLORMAP_TURBO 进行高动态对比色彩映射
        h_color = cv2.applyColorMap(norm_h, cv2.COLORMAP_TURBO)
        
        # 无效深度点 (0mm 盲区) 设为深灰色
        h_color[depth_mm == 0] = (20, 20, 20)
        
        # 相对台面 <= 2.5mm 的台面背景区域，赋予均匀的暗深蓝色底色 (消除相机倾角渐变)
        table_bg_mask = (depth_mm > 0) & (rel_h <= 2.5)
        h_color[table_bg_mask] = (35, 20, 10)
        
        # 在画面右上角/右侧绘制高度色阶标尺 (Height Color Bar)
        bar_w = 18
        bar_h = min(220, int(h * 0.45))
        bar_x = w - 85
        bar_y = 50
        
        # 生成垂直颜色阶梯 (从上到下: max_h_mm -> 0mm)
        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
        gradient_img = np.repeat(gradient, bar_w, axis=1)
        bar_bgr = cv2.applyColorMap(gradient_img, cv2.COLORMAP_TURBO)
        
        # 叠加标尺半透明背景框
        cv2.rectangle(h_color, (bar_x - 10, bar_y - 28), (w - 10, bar_y + bar_h + 20), (15, 15, 15), -1)
        cv2.rectangle(h_color, (bar_x - 10, bar_y - 28), (w - 10, bar_y + bar_h + 20), (70, 70, 70), 1)
        
        # 标题
        cv2.putText(h_color, "Height(mm)", (bar_x - 6, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        
        # 标尺贴图与边框
        h_color[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = bar_bgr
        cv2.rectangle(h_color, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        
        # 标尺刻度与文本
        steps = [
            (0.0, f"+{int(max_h_mm)}mm"),
            (0.25, f"+{int(max_h_mm * 0.75)}mm"),
            (0.5, f"+{int(max_h_mm * 0.5)}mm"),
            (0.75, f"+{int(max_h_mm * 0.25)}mm"),
            (1.0, "0(台面)")
        ]
        for ratio, text in steps:
            curr_y = bar_y + int(ratio * bar_h)
            cv2.line(h_color, (bar_x + bar_w, curr_y), (bar_x + bar_w + 4, curr_y), (255, 255, 255), 1)
            cv2.putText(h_color, text, (bar_x + bar_w + 7, curr_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
            
        return h_color

    def segment_and_separate(self, color_bgr: np.ndarray, depth_mm: np.ndarray, rel_h: np.ndarray) -> List[np.ndarray]:
        """
        3D 深度浮凸主导 + 黑帽横向暗缝切分实例分离算法：
        1. 提取高于台面 (rel_h > 8mm) 且具备植物色特征的前景物料区
        2. 基于 Black-Hat 细长水平卷积核提取芦笋与芦笋接触面的纵向阴影缝隙
        3. 用暗缝掩膜对并排粘连的连通块进行断开切分，输出单根芦笋实例轮廓列表
        """
        h, w = color_bgr.shape[:2]
        b, g, r = cv2.split(color_bgr)
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. 前景掩膜：深度高于台面且属于植物色域 (兼顾嫩黄绿、深绿与白绿笋体)
        color_valid = (g.astype(float) >= b.astype(float) * 0.90) | ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 100))
        fg_mask = (color_valid & (rel_h >= self.table_margin_mm) & (hsv[:, :, 2] > 25) & (depth_mm >= 350) & (depth_mm <= 780)).astype(np.uint8) * 255
        
        # 限制在中心作业有效 ROI
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h * 0.04):int(h * 0.96), int(w * 0.04):int(w * 0.96)] = 255
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
        
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close)
        
        # 2. 黑帽变换提取芦笋之间的水平暗缝
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        k_seam = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 5))
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_seam)
        _, seams = cv2.threshold(black_hat, 8, 255, cv2.THRESH_BINARY)
        
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        seams_dil = cv2.dilate(seams, k_dil, iterations=1)
        
        # 3. 切分粘连并做形态学去噪
        cut_mask = cv2.bitwise_and(fg_clean, cv2.bitwise_not(seams_dil))
        k_op = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        cut_clean = cv2.morphologyEx(cut_mask, cv2.MORPH_OPEN, k_op)
        
        cnts, _ = cv2.findContours(cut_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) >= self.min_area]

    def segment_foreground(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        """
        兼容外部接口调用的单张二值掩膜生成
        """
        plane_coeff = self.fit_table_plane(depth_mm)
        rel_h = self.compute_relative_height(depth_mm, plane_coeff)
        b, g, r = cv2.split(color_bgr)
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        color_valid = (g.astype(float) >= b.astype(float) * 0.90) | ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 100))
        fg = (color_valid & (rel_h >= self.table_margin_mm) & (hsv[:, :, 2] > 25) & (depth_mm >= 400) & (depth_mm <= 670)).astype(np.uint8) * 255
        return fg

    def analyze(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> List[AsparagusTarget]:
        """
        端到端全流程分析：
          0. AprilTag 三级标定降级链解算当前帧坐标变换
          1. 拟合工作台平面并计算逐像素相对高度
          2. 黑帽暗缝检测切开并排粘连，分离出独立单根芦笋轮廓
          3. 基于 fitLine 解算各根芦笋轴线角度与长径尺寸
          4. 脊线深度采样与工作台倾斜补偿
          5. 叠压拓扑分析，锁定最顶层可抓取目标 (Topmost Pickable Target)
        """
        # 步骤 0：三级标定降级链 — 解算当前帧最优坐标变换
        frame_transform, frame_calib_source = self._resolve_calibration(color_bgr)

        plane_coeff = self.fit_table_plane(depth_mm)
        rel_h = self.compute_relative_height(depth_mm, plane_coeff)
        contours = self.segment_and_separate(color_bgr, depth_mm, rel_h)
        
        targets: List[AsparagusTarget] = []
        target_idx = 1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
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
            
            # 提取中轴脊线采样掩膜
            c_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            spine_mask = cv2.erode(c_mask, k_erode, iterations=1)
            
            # 深度统计采样
            valid_depth_mask = (spine_mask > 0) & (depth_mm > 350) & (depth_mm < 700)
            spine_depths = depth_mm[valid_depth_mask]
            if len(spine_depths) < 10:
                spine_depths = depth_mm[(c_mask > 0) & (depth_mm > 350) & (depth_mm < 700)]
            
            if len(spine_depths) == 0:
                continue
            
            # 取前 15% 分位数作为最顶面高度 Z_top (抗反光噪点)
            z_top = float(np.percentile(spine_depths, 15))
            z_med = float(np.median(spine_depths))
            
            # --- 3D 空间真实欧氏测距 (彻底消除 ±30° 大倾角透视短缩误差) ---
            # 芦笋两端点在图像上的精确亚像素坐标
            p_min, p_max = float(np.min(proj_len)), float(np.max(proj_len))
            u1, v1 = cx_val + p_min * vx_val, cy_val + p_min * vy_val
            u2, v2 = cx_val + p_max * vx_val, cy_val + p_max * vy_val
            
            # 结合传送带平面方程估计两端点的真实深度 Z1, Z2
            if plane_coeff is not None:
                # 局部台面高度减去物料凸起
                z1 = float(plane_coeff[0] * u1 + plane_coeff[1] * v1 + plane_coeff[2] - (z_med / self.fx * 2.0))
                z2 = float(plane_coeff[0] * u2 + plane_coeff[1] * v2 + plane_coeff[2] - (z_med / self.fx * 2.0))
            else:
                z1 = z_med
                z2 = z_med
            
            # 反投影至 3D 空间计算无损真实欧氏长度
            x1_3d = (u1 - self.cx) * z1 / self.fx
            y1_3d = (v1 - self.cy) * z1 / self.fy
            x2_3d = (u2 - self.cx) * z2 / self.fx
            y2_3d = (v2 - self.cy) * z2 / self.fy
            
            length_mm = float(np.sqrt((x1_3d - x2_3d)**2 + (y1_3d - y2_3d)**2 + (z1 - z2)**2))
            
            # 直径按中心深度尺度解算
            scale_center = z_med / self.fx
            diam_mm = float(diam_px * scale_center)
            
            # 过滤不符合芦笋物理尺寸的杂散区域
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
                rel_height_mm = float(640.0 - z_top)
            
            rect = cv2.minAreaRect(cnt)
            box_corners = cv2.boxPoints(rect).astype(np.int32)

            # 计算机械臂 SCARA 抓取坐标系参数 (三级标定降级链坐标变换)
            if frame_transform is not None:
                p_cam_h = np.array([grip_x, grip_y, grip_z, 1.0])
                p_robot_h = frame_transform @ p_cam_h
                robot_x = float(p_robot_h[0])
                robot_y = float(p_robot_h[1])
                robot_z = float(p_robot_h[2])

                # 经过标定旋转矩阵变换芦笋主轴方向，解算夹爪在 SCARA 水平面的目标旋转角
                r_mat = frame_transform[:3, :3]
                v_cam = np.array([vx_val, vy_val, 0.0])
                v_robot = r_mat @ v_cam
                r_rad = np.arctan2(v_robot[1], v_robot[0])
                robot_r = float(np.degrees(r_rad))
                if robot_r > 90.0: robot_r -= 180.0
                elif robot_r < -90.0: robot_r += 180.0
            else:
                # [核心安全防撞机制] 未标定安全模式：
                # 机械臂 Z 轴绝对禁止直接使用相机镜头深度 (grip_z ~530mm)，否则必撞机毁机！
                # 强制采用相对传送带凸起净高度 (rel_height_mm, 通常 15~40mm) 作为安全下探参考
                robot_x = float(grip_x)
                robot_y = float(grip_y)
                robot_z = float(rel_height_mm)
                robot_r = float(yaw_deg)

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
                robot_x=round(robot_x, 1),
                robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1),
                robot_r=round(robot_r, 1),
                is_topmost=False,
                calibration_source=frame_calib_source
            )
            targets.append(target)
            target_idx += 1
            
        # 叠压拓扑分析与最顶层判决：
        # 判据：相对工作台面凸起高度最高（rel_height_mm 最大）者为最顶层优先抓取目标
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
                cv2.rectangle(annotated, (cx_int - 170, cy_int - 56), (cx_int + 230, cy_int - 6), (20, 20, 20), -1)
                cv2.rectangle(annotated, (cx_int - 170, cy_int - 56), (cx_int + 230, cy_int - 6), (0, 255, 120), 2)
                cv2.putText(annotated, label_header, (cx_int - 160, cy_int - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2)
                cv2.putText(annotated, label_pose, (cx_int - 160, cy_int - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)
            else:
                label_simple = f"#{t.id} L:{t.length_mm} D:{t.diam_mm} R:{t.yaw_deg}"
                cv2.putText(annotated, label_simple, (cx_int - 45, cy_int - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        return annotated

    def diagnose(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> str:
        """
        现场感知即时诊断工具：
        当未检测到有效芦笋时，全流程逐步检查并输出详细根因分析与调整建议
        """
        lines = []
        lines.append("【芦笋 3D 感知系统即时诊断报告】")
        h, w = depth_mm.shape
        
        # 1. 检查深度数据流健康度
        valid_depth = depth_mm[depth_mm > 0]
        valid_ratio = len(valid_depth) / (h * w) * 100
        lines.append(f"1. 深度点云健康度: 有效深度像素占比 {valid_ratio:.1f}% ({len(valid_depth)}/{h*w})")
        if len(valid_depth) < 5000:
            lines.append("   [!] 严重警告: 深度点云极为稀疏 (<5000点)！可能原因: 黑色传送带反光散射/激光散斑被遮挡/距离超出相机量程。")
            lines.append("   [建议] 检查相机镜头前有无异物，按 [L] 确认激光器已开启，或微调激光功率。")
            return "\n".join(lines)
            
        d_min = np.min(valid_depth)
        d_max = np.max(valid_depth)
        d_med = np.median(valid_depth)
        lines.append(f"   - 深度分布: 最近 {d_min}mm, 中位数 {d_med:.0f}mm, 最远 {d_max}mm")
        
        # 2. 传送带底板拟合诊断
        plane_coeff = self.fit_table_plane(depth_mm)
        if plane_coeff is None:
            lines.append("2. 传送带底板平面拟合: 【失败】")
            lines.append(f"   [!] 未能在 610~670mm 区间或高位深层找到足够密集的基准点。当前视野中位深度为 {d_med:.0f}mm。")
            lines.append("   [建议] 当前相机安装高度可能偏离 640mm 标准高度，建议检查机台物理安装或微调拟合区间。")
        else:
            pitch, roll, tilt = self.get_table_tilt_angles(plane_coeff)
            lines.append(f"2. 传送带底板平面拟合: 【成功】 (Pitch={pitch:+.1f}°, Roll={roll:+.1f}°, 综合倾角={tilt:.1f}°)")
            
        # 3. 前景物料提取与相对凸起高度
        rel_h = self.compute_relative_height(depth_mm, plane_coeff)
        h_max = float(np.max(rel_h))
        h_p98 = float(np.percentile(rel_h[rel_h > 0], 98)) if np.count_nonzero(rel_h > 0) > 100 else 0.0
        lines.append(f"3. 相对底板净凸起高度: 最高点={h_max:.1f}mm, 98分位凸起={h_p98:.1f}mm (门限: >={self.table_margin_mm}mm)")
        if h_p98 < self.table_margin_mm:
            lines.append(f"   [!] 警告: 视野内几乎无高于台面 {self.table_margin_mm}mm 的凸起物体！传送带上可能未放置物料，或物料过于贴平。")
            return "\n".join(lines)
            
        # 4. 植物色域与连通域切分
        contours = self.segment_and_separate(color_bgr, depth_mm, rel_h)
        lines.append(f"4. 实例分割与黑帽暗缝切分: 提取出 {len(contours)} 个候选轮廓 (最小面积门限: {self.min_area}px)")
        if len(contours) == 0:
            lines.append("   [!] 警告: 前景掩膜提取为空！可能原因: 物料颜色不满足嫩绿/黄绿植物色域，或光照严重过曝/欠曝。")
            return "\n".join(lines)
            
        # 5. 逐个候选轮廓的几何过滤诊断
        lines.append("5. 候选轮廓几何规格过滤详情:")
        for idx, cnt in enumerate(contours[:6]):  # 最多打印前 6 个
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                lines.append(f"   - 轮廓 #{idx+1}: 面积 {area:.0f}px < {self.min_area}px [过滤: 碎片杂质]")
                continue
            [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            pts = cnt.reshape(-1, 2).astype(float)
            diff = pts - np.array([float(x0[0]), float(y0[0])])
            proj_len = np.dot(diff, np.array([float(vx[0]), float(vy[0])]))
            proj_wid = np.dot(diff, np.array([-float(vy[0]), float(vx[0])]))
            l_px = float(np.max(proj_len) - np.min(proj_len))
            d_px = float(np.max(proj_wid) - np.min(proj_wid))
            asp = l_px / max(1.0, d_px)
            if asp < self.min_aspect_ratio:
                lines.append(f"   - 轮廓 #{idx+1}: 长宽比 {asp:.2f} < {self.min_aspect_ratio} [过滤: 非细长棒体/块状杂物]")
                continue
            # 深度采样
            c_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            valid_d = depth_mm[(c_mask > 0) & (depth_mm > 350) & (depth_mm < 700)]
            if len(valid_d) == 0:
                lines.append(f"   - 轮廓 #{idx+1}: 区域内无有效深度数据 [过滤: 测距盲区]")
                continue
            z_med = float(np.median(valid_d))
            l_mm = float(l_px * z_med / self.fx)
            d_mm = float(d_px * z_med / self.fx)
            if not (self.min_length_mm <= l_mm <= self.max_length_mm):
                lines.append(f"   - 轮廓 #{idx+1}: 估算长度 {l_mm:.1f}mm 超出范围 [{self.min_length_mm}, {self.max_length_mm}]mm [过滤]")
                continue
            if not (self.min_diam_mm <= d_mm <= self.max_diam_mm):
                lines.append(f"   - 轮廓 #{idx+1}: 估算直径 {d_mm:.1f}mm 超出范围 [{self.min_diam_mm}, {self.max_diam_mm}]mm [过滤]")
                continue
            lines.append(f"   - 轮廓 #{idx+1}: 【合规目标】 L={l_mm:.1f}mm, D={d_mm:.1f}mm")
            
        lines.append("------------------------------------------------------------")
        lines.append("【调试建议】")
        lines.append("  1. 若芦笋偏长或偏粗，系统已将长径放宽至 550mm / 65mm。")
        lines.append("  2. 如自动识别暂未锁定，您可直接用【鼠标左键点击】画面上任意芦笋，")
        lines.append("     准心将锁定该点，按 [G] 键将直接输出该示教点的防撞抓取 G-code！")
        return "\n".join(lines)
