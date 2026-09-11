#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标靶 (AprilTag 16h5) 空间地图建图与全局平差求解工具 (Tag Map Builder)
- 基于多视角重叠图像构建标靶共视连通图 (Co-visibility Graph)
- 基于 Bundle Adjustment (BA) 联合优化静止标靶 3D 空间位姿与相机位姿
- 支持 Tag 0 (SCARA J1 旋转中心原点) 与 Tag 1 (世界 +X 轴基准) 的刚体对齐闭环
- 导出 config/tags_map.yaml 供运行时毫秒级在线相机定位
"""

import os
import sys
import glob
import math
import yaml
import argparse
from datetime import datetime
import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional, Any, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None


class CovisibilityGraphError(Exception):
    """标靶共视连通图拓扑异常（如出现孤立子图、断网或约束退化）"""
    pass


class TagMapBuilder:
    def __init__(self, 
                 marker_size_mm: float = 50.0,
                 tag_family: int = cv2.aruco.DICT_APRILTAG_16h5,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        初始化建图求解器
        :param marker_size_mm: 标靶黑白边框物理边长 (毫米)
        :param tag_family: OpenCV ArUco 字典枚举 (默认 AprilTag 16h5)
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数
        """
        self.marker_size_mm = float(marker_size_mm)
        self.dictionary = cv2.aruco.getPredefinedDictionary(tag_family)
        
        # 现代 OpenCV 4.7+ ArucoDetector 接口与工业级高灵敏度超参数
        # 尝试读取 config.yaml 中的反差门限配置
        contrast_boost = 1.6
        clahe_clip = 4.0
        min_otsu = 1.5
        min_perim = 0.006
        thresh_const = 3.0
        valid_tag_ids = []
        try:
            cfg_file = "config.yaml"
            if os.path.exists(cfg_file):
                with open(cfg_file, "r", encoding="utf-8") as f:
                    c = yaml.safe_load(f) or {}
                valid_tag_ids = [int(x) for x in c.get("calibration", {}).get("valid_tag_ids", [])]
                d_cfg = c.get("calibration", {}).get("tag_detection", {})
                contrast_boost = float(d_cfg.get("contrast_boost", contrast_boost))
                clahe_clip = float(d_cfg.get("clahe_clip_limit", clahe_clip))
                min_otsu = float(d_cfg.get("min_otsu_std_dev", min_otsu))
                min_perim = float(d_cfg.get("min_perimeter_rate", min_perim))
                thresh_const = float(d_cfg.get("adaptive_thresh_constant", thresh_const))
        except Exception:
            pass

        self.valid_tag_ids = valid_tag_ids
        self.contrast_boost = contrast_boost
        self.min_otsu_std_dev = min_otsu

        # 基础参数模版 (兼顾高精度亚像素角点与大透视/低反差/发灰墨色)
        def make_params(thresh_c, min_otsu):
            p = cv2.aruco.DetectorParameters()
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 43
            p.adaptiveThreshWinSizeStep = 8
            p.adaptiveThreshConstant = thresh_c
            p.minOtsuStdDev = min_otsu
            p.minMarkerPerimeterRate = max(0.008, min_perim)
            p.maxMarkerPerimeterRate = 4.0
            p.polygonalApproxAccuracyRate = 0.09
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            p.perspectiveRemovePixelPerCell = 10
            p.errorCorrectionRate = 0.50
            p.perspectiveRemoveIgnoredMarginPerCell = 0.15
            p.maxErroneousBitsInBorderRate = 0.30
            return p

        # 构建两路互补检测器：
        # 路 A: 抗反光/高亮清晰路 (C=5.5, 适合高光、灯光直射视角)
        self.params_bright = make_params(thresh_c=5.5, min_otsu=0.55)
        self.detector_bright = cv2.aruco.ArucoDetector(self.dictionary, self.params_bright)
        
        # 路 B: 低反差/黑度不纯路 (C=2.5, min_otsu=0.45, 适合背光、发灰视角)
        self.params_dark = make_params(thresh_c=2.5, min_otsu=0.45)
        self.detector_dark = cv2.aruco.ArucoDetector(self.dictionary, self.params_dark)
        self.detector = self.detector_bright

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

        # 相机内参与畸变 (若未指定，优先从 config_guard 加载并自适应)
        if camera_matrix is None:
            if resolve_camera_intrinsics is not None:
                K, dist, _ = resolve_camera_intrinsics("config.yaml")
                self.camera_matrix = K
                self.dist_coeffs = dist
            else:
                self.camera_matrix = np.array([
                    [1363.68, 0.0, 971.19],
                    [0.0, 1361.19, 566.26],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float64)

        # 标靶局部坐标系下的 4 个角点物理坐标 (逆时针, Z=0)
        s = self.marker_size_mm / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

    def detect_tags(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """
        双路互补融合全景检测（彻底解决反光、黑度不纯、倾斜与远景漏检）：
        路 1 (抗反光/高光路): 原图灰度 + 较严二值门限 C=5.5，精准捕获高光、灯光直射区域标靶
        路 2 (低反差/暗部路): 动态直方图拉伸 + 宽松二值门限 C=2.5 + minOtsu=0.45，攻克发灰、低反差、暗部标靶
        白名单机制硬锁保底，两路检测结果快速取并集，杜绝误检与漏检。
        :return: {tag_id: corners_4x2}
        """
        if len(image.shape) == 3:
            gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_raw = image

        results = {}

        # 路 1: 针对清晰/高光/反光区域
        c1, ids1, _ = self.detector_bright.detectMarkers(gray_raw)
        if ids1 is not None and len(ids1) > 0:
            for idx, tag_id in enumerate(ids1.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                results[tid] = c1[idx].reshape((4, 2))

        # 路 2: 针对暗部/低反差/打印黑度不够纯区域 (动态拉伸 + 宽松门限)
        p_low, p_high = np.percentile(gray_raw[::4, ::4], (2, 98))
        if p_high > p_low + 10:
            gray_stretch = np.clip((gray_raw.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            gray_stretch = gray_raw

        c2, ids2, _ = self.detector_dark.detectMarkers(gray_stretch)
        if ids2 is not None and len(ids2) > 0:
            for idx, tag_id in enumerate(ids2.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                if tid not in results:
                    results[tid] = c2[idx].reshape((4, 2))

        return results

    def solve_single_tag_pnp(self, corners: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        对单标靶执行 PnP 获得其在相机系下的位姿 (rvec, tvec)
        """
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return success, rvec, tvec

    def render_tag_3d_axes(self, img: np.ndarray, corners: np.ndarray, tag_id: int):
        """
        在图像上绘制标靶 3D 空间坐标系与实心正四棱柱：
        - X 轴 (红色, 25mm), Y 轴 (绿色, 25mm)
        - Z 轴指示: 边长 20.0mm x 20.0mm (原始标靶 1/2)、长 120.0mm 的实心正四棱柱 (半透明实心柱体 + 12条高亮棱线 + 顶盖透视截面)
        """
        try:
            corners_2d = corners.reshape((4, 2)).astype(np.float64)
            ok, rvec, tvec = self.solve_single_tag_pnp(corners_2d)
            if not ok:
                return

            hw = 10.0   # 截面半宽 10mm，整体截面边长为 20.0mm x 20.0mm
            L = 120.0   # 柱体长度 120mm

            # 8 个 3D 角点: 底面 4 点 (Z=0), 顶面 4 点 (Z=L)
            pts_3d = np.array([
                # 底面 4 点
                [-hw, -hw, 0.0],
                [ hw, -hw, 0.0],
                [ hw,  hw, 0.0],
                [-hw,  hw, 0.0],
                # 顶面 4 点
                [-hw, -hw, L],
                [ hw, -hw, L],
                [ hw,  hw, L],
                [-hw,  hw, L],
                # 顶面中心
                [0.0, 0.0, L],
                # X 轴与 Y 轴参考端点 (从中心伸出 25mm，突出棱柱外侧)
                [25.0, 0.0, 0.0],
                [0.0, 25.0, 0.0],
                [0.0, 0.0, 0.0]
            ], dtype=np.float64)

            proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            proj = proj.reshape((-1, 2)).astype(int)

            b_pts = proj[0:4] # 底面 4 点
            t_pts = proj[4:8] # 顶面 4 点
            top_center = tuple(proj[8])
            p_x = tuple(proj[9])
            p_y = tuple(proj[10])
            p_orig = tuple(proj[11])

            # 1. 绘制 X 轴 (红色) 和 Y 轴 (绿色)
            cv2.line(img, p_orig, p_x, (0, 0, 240), 2, cv2.LINE_AA)
            cv2.putText(img, 'X', p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(img, p_orig, p_y, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(img, 'Y', p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 2. 半透明填充 4 个侧面与顶面 (呈现实心方柱体立体质感)
            overlay = img.copy()
            for i in range(4):
                next_i = (i + 1) % 4
                side_poly = np.array([b_pts[i], b_pts[next_i], t_pts[next_i], t_pts[i]], dtype=np.int32)
                cv2.fillPoly(overlay, [side_poly], (240, 160, 30)) # BGR: 浅蓝/青
            cv2.fillPoly(overlay, [t_pts], (255, 220, 90))         # 顶面高光
            cv2.addWeighted(overlay, 0.42, img, 0.58, 0, img)

            # 3. 绘制 12 条棱线 (高清晰边框)
            cv2.polylines(img, [b_pts], isClosed=True, color=(180, 80, 0), thickness=2, lineType=cv2.LINE_AA)
            for i in range(4):
                cv2.line(img, tuple(b_pts[i]), tuple(t_pts[i]), (255, 130, 0), 2, cv2.LINE_AA)
            cv2.polylines(img, [t_pts], isClosed=True, color=(255, 240, 120), thickness=2, lineType=cv2.LINE_AA)

            # 4. 顶面中心标注点与文字 (简洁工业标定, 仅保留 Z 轴标识)
            cv2.circle(img, top_center, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, 'Z', (top_center[0] + 5, top_center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 80), 2, cv2.LINE_AA)
        except Exception:
            pass

    @staticmethod
    def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """旋转向量与平移向量转 4x4 齐次矩阵"""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    @staticmethod
    def matrix_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """4x4 齐次矩阵转旋转向量与平移向量"""
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3].reshape((3, 1))
        return rvec, tvec

    def compute_tag_metrics(self, corners: np.ndarray) -> Dict[str, Any]:
        """
        根据标靶 4 个角点物理投影计算空间像元分辨率及几何指标
        """
        pts = corners.reshape((4, 2)).astype(np.float64)
        l01 = float(np.linalg.norm(pts[1] - pts[0]))
        l12 = float(np.linalg.norm(pts[2] - pts[1]))
        l23 = float(np.linalg.norm(pts[3] - pts[2]))
        l30 = float(np.linalg.norm(pts[0] - pts[3]))
        cell_w = int(round((l01 + l23) / 12.0))
        cell_h = int(round((l30 + l12) / 12.0))
        center_x = float(np.mean(pts[:, 0]))
        center_y = float(np.mean(pts[:, 1]))
        area = float(cv2.contourArea(pts.astype(np.float32)))
        return {
            "cell_size_px": [cell_w, cell_h],
            "center_px": [round(center_x, 1), round(center_y, 1)],
            "area_px": round(area, 1),
            "edge_lengths_px": [round(l01, 1), round(l12, 1), round(l23, 1), round(l30, 1)]
        }

    def render_annotated_frame(self, image: np.ndarray, detected_tags: Dict[int, np.ndarray]) -> np.ndarray:
        """
        渲染完整图示化分析图像 (带绿色轮廓线、彩色角点、像元分辨率标牌与 3D 正四棱柱)
        """
        disp = image.copy()
        dot_colors = [
            (0, 0, 255),    # 角点 0: 红色
            (0, 255, 0),    # 角点 1: 绿色
            (255, 0, 0),    # 角点 2: 蓝色
            (0, 255, 255)   # 角点 3: 黄色
        ]
        for tag_id, corners in detected_tags.items():
            pts = corners.reshape((4, 2)).astype(np.int32)
            # 绘制绿色轮廓线
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            # 绘制 4 个角点彩色圆点
            for pt_idx, pt in enumerate(pts):
                cv2.circle(disp, tuple(pt), 5, dot_colors[pt_idx], -1)

            # 计算机械标靶单元方格像素尺寸 (AprilTag 16h5 为 6x6 网格)
            metrics = self.compute_tag_metrics(corners)
            cell_w, cell_h = metrics["cell_size_px"]

            # 绘制 ID 与最小单元方格像素分辨率标牌
            cx = int(metrics["center_px"][0])
            min_y = int(np.min(pts[:, 1]))
            tag_text = f"Tag {tag_id}" + (" [ORIGIN]" if tag_id == 0 else "")
            cell_text = f"Cell: {cell_w}x{cell_h}px"
            
            badge_x = cx - 50
            badge_y = max(42, min_y - 12)
            cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 106, badge_y + 8), (20, 20, 20), -1)
            cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 106, badge_y + 8), (0, 255, 255), 1)
            cv2.putText(disp, tag_text, (badge_x, badge_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, cell_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

            # 绘制 3D 空间坐标系 (实心正四棱柱)
            self.render_tag_3d_axes(disp, corners, tag_id)

        return disp

    def export_observations_manifest(self, 
                                     image_paths: List[str], 
                                     manifest_path: str = "data/tag_calibration_images/tag_observations.yaml",
                                     generate_visualized: bool = True) -> str:
        """
        两阶段建图流水线 - 阶段一：
        扫描多视角图像，提取标靶观测数据并导出为结构化审核清单 (YAML)
        - 自动保留用户此前修改过的 keep: false 剔除状态与人工备注 (note)
        - 自动输出单元方格分辨率 (Cell: WxH px)、中心坐标与面积
        - 支持同步生成高清图示化标注图片至 visualized/ 目录
        """
        # 读取已有清单以保留用户此前的手工标记
        existing_prefs = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old_manifest = yaml.safe_load(f) or {}
                for img_key, img_info in old_manifest.get("images", {}).items():
                    for obs in img_info.get("observations", []):
                        tid = obs.get("tag_id")
                        keep = obs.get("keep", True)
                        note = obs.get("note", "")
                        existing_prefs[(img_key, tid)] = (keep, note)
                print(f"[+] 检测到已有审核清单，已成功加载 {len(existing_prefs)} 条历史人工保留/剔除标记")
            except Exception as e:
                print(f"[WARN] 读取已有清单配置失败: {e}，将生成全新清单。")

        manifest_data = {
            "summary": {},
            "images": {}
        }
        total_obs = 0
        total_kept = 0
        total_excluded = 0

        vis_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "visualized")
        if generate_visualized:
            os.makedirs(vis_dir, exist_ok=True)

        for path in sorted(image_paths):
            base_name = os.path.basename(path)
            img = cv2.imread(path)
            if img is None:
                continue

            detected = self.detect_tags(img)
            obs_list = []

            for tid in sorted(detected.keys()):
                corners = detected[tid]
                metrics = self.compute_tag_metrics(corners)
                
                # 保留历史人工决策
                keep_val, note_val = existing_prefs.get((base_name, tid), (True, ""))
                total_obs += 1
                if keep_val:
                    total_kept += 1
                else:
                    total_excluded += 1

                obs_entry = {
                    "tag_id": int(tid),
                    "keep": bool(keep_val),
                    "cell_size_px": metrics["cell_size_px"],
                    "center_px": metrics["center_px"],
                    "area_px": metrics["area_px"],
                    "note": str(note_val),
                    "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                }
                obs_list.append(obs_entry)

            annotated_name = os.path.splitext(base_name)[0] + "_annotated.png"
            annotated_path = os.path.join(vis_dir, annotated_name)
            
            if generate_visualized:
                ann_img = self.render_annotated_frame(img, detected)
                cv2.imwrite(annotated_path, ann_img)

            manifest_data["images"][base_name] = {
                "file_name": base_name,
                "image_path": path.replace("\\", "/"),
                "annotated_path": annotated_path.replace("\\", "/"),
                "detected_count": len(obs_list),
                "observations": obs_list
            }

        manifest_data["summary"] = {
            "total_images": len(manifest_data["images"]),
            "total_observations": total_obs,
            "total_kept": total_kept,
            "total_excluded": total_excluded,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 写入带有清晰中文使用指南的 YAML
        header_comments = (
            "# ==============================================================================\n"
            "# AprilTag 离线建图与 BA 平差观测数据清单 (Observations Manifest)\n"
            "# \n"
            "# 【人工审核与坏样本剔除指南】：\n"
            "# 1. 请在下方列表中查找需要剔除的劣质标靶观测（如透视大倾角、模糊、边缘拉丝畸变）；\n"
            "# 2. 将对应的 `keep: true` 修改为 `keep: false` 即可，求解器将自动忽略该项；\n"
            "# 3. 可在 `note: \"\"` 中记录剔除原因（如 \"大倾角发灰\"、\"遮挡严重\" 等）；\n"
            "# 4. 修改保存后，在控制台选择继续执行平差即可；\n"
            "# 5. 【拓扑连通性安全守门员】：系统会在平差前进行连通图校验，防止误删导致断网崩溃！\n"
            "# ==============================================================================\n\n"
        )
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(header_comments)
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"[OK] 观测清单已成功导出至: {manifest_path}")
        print(f"     统计: 共 {len(manifest_data['images'])} 张图像，{total_obs} 次标靶观测 (保留: {total_kept}, 排除: {total_excluded})")
        return manifest_path

    def load_observations_manifest(self, manifest_path: str = "data/tag_calibration_images/tag_observations.yaml") -> Tuple[List[Dict[int, np.ndarray]], List[str], Dict[str, Any]]:
        """
        两阶段建图流水线 - 阶段二：
        从审核清单中加载已审核的标靶观测数据，并过滤掉 keep: false 的坏样本。
        :return: (frame_detections, valid_frame_names, stats)
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"未找到观测清单文件: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        frame_detections = []
        valid_frame_names = []
        stats = {
            "total_images": len(data.get("images", {})),
            "total_observations": 0,
            "total_kept": 0,
            "total_excluded": 0,
            "dropped_single_tag_frames": [],
            "excluded_items": []
        }

        for img_name, img_info in data.get("images", {}).items():
            tags_in_frame = {}
            for obs in img_info.get("observations", []):
                stats["total_observations"] += 1
                tid = int(obs["tag_id"])
                keep = obs.get("keep", True)
                if not keep:
                    stats["total_excluded"] += 1
                    stats["excluded_items"].append({
                        "image": img_name,
                        "tag_id": tid,
                        "note": obs.get("note", "")
                    })
                    continue

                stats["total_kept"] += 1
                corners = np.array(obs["corners"], dtype=np.float64)
                tags_in_frame[tid] = corners

            # 约束检查：一张图必须至少观测到 2 个 Tag 才能形成相对刚体约束
            if len(tags_in_frame) >= 2:
                frame_detections.append(tags_in_frame)
                valid_frame_names.append(img_name)
            else:
                stats["dropped_single_tag_frames"].append((img_name, len(tags_in_frame)))

        return frame_detections, valid_frame_names, stats

    def validate_covisibility(self, 
                              frame_detections: List[Dict[int, np.ndarray]], 
                              frame_names: Optional[List[str]] = None,
                              origin_tag_id: int = 0,
                              x_align_tag_id: int = 1) -> Dict[str, Any]:
        """
        共视连通性安全守门员 (Co-visibility Graph Connectivity Guard)
        在 BA 平差前全面审查标靶共视图拓扑健康度：
        1. 检查各 Tag 是否在一个连通分量内，防止误删导致孤岛断网引发奇异矩阵；
        2. 识别单视角关键桥梁 (Critical Bridges)，当桥梁仅由 1 张图片支撑时警示用户；
        3. 给出详细结构化拓扑报告。
        """
        all_tags = set()
        for tags in frame_detections:
            all_tags.update(tags.keys())

        if len(all_tags) == 0:
            return {
                "is_valid": False,
                "all_tags": [],
                "connected_tags": [],
                "unconnected_tags": [],
                "adj_list": {},
                "edge_counts": {},
                "critical_bridges": [],
                "valid_frames_count": 0,
                "components": [],
                "message": "有效标靶集合为空，无法建图！"
            }

        # 构建无向图与共视重叠帧数统计
        adj_list = {t: set() for t in all_tags}
        edge_counts = {}

        for tags in frame_detections:
            t_ids = list(tags.keys())
            for i in range(len(t_ids)):
                for j in range(i + 1, len(t_ids)):
                    u, v = min(t_ids[i], t_ids[j]), max(t_ids[i], t_ids[j])
                    adj_list[u].add(v)
                    adj_list[v].add(u)
                    edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

        # 识别关键单一支撑桥梁 (只有 1 帧同时观测到该标靶对)
        critical_bridges = [(u, v) for (u, v), count in edge_counts.items() if count == 1]

        # 寻找连通分量 (BFS)
        remaining = set(all_tags)
        components = []
        while remaining:
            root = next(iter(remaining))
            comp = set()
            q = [root]
            while q:
                curr = q.pop(0)
                if curr not in comp:
                    comp.add(curr)
                    q.extend(adj_list[curr] - comp)
            components.append(sorted(list(comp)))
            remaining -= comp

        # 确定主基准分量 (优先包含 x_align_tag_id 或 origin_tag_id)
        preferred_base = x_align_tag_id if x_align_tag_id in all_tags else (origin_tag_id if origin_tag_id in all_tags else min(all_tags))
        main_component = []
        for comp in components:
            if preferred_base in comp:
                main_component = comp
                break
        if not main_component and components:
            main_component = max(components, key=len)

        unconnected = sorted(list(all_tags - set(main_component)))
        is_valid = (len(components) == 1) and (len(all_tags) >= 2) and (len(frame_detections) >= 2)

        if not is_valid:
            if len(all_tags) < 2:
                msg = f"有效标靶总数不足 2 个 (仅检出 {list(all_tags)})，无法构建相对空间地图！"
            elif len(frame_detections) < 2:
                msg = f"有效多视角图像不足 2 张 (当前仅 {len(frame_detections)} 张)，无法构建空间刚体约束！"
            else:
                msg = (f"【严重风险】标靶共视图发生断裂！共发现 {len(components)} 个孤立分量。\n"
                       f"主连通网络: {main_component}\n"
                       f"失联断网标靶: {unconnected}\n"
                       f"提示：请检查 tag_observations.yaml，恢复连接上述失联标靶的关键视角观测 (设为 keep: true)。")
        else:
            msg = f"共视连通性检查完全通过！全部 {len(all_tags)} 个标靶处于统一连通拓扑网中。"

        return {
            "is_valid": is_valid,
            "all_tags": sorted(list(all_tags)),
            "connected_tags": main_component,
            "unconnected_tags": unconnected,
            "components": components,
            "adj_list": {k: sorted(list(v)) for k, v in adj_list.items()},
            "edge_counts": {f"{u}-{v}": c for (u, v), c in edge_counts.items()},
            "critical_bridges": critical_bridges,
            "valid_frames_count": len(frame_detections),
            "message": msg
        }

    def optimize_bundle_adjustment(self, 
                                   frame_detections: List[Dict[int, np.ndarray]], 
                                   active_frame_names: Optional[List[str]] = None,
                                   origin_tag_id: int = 0,
                                   x_align_tag_id: int = 1,
                                   baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict[str, Any]:
        """
        基于非线性最小二乘 (Bundle Adjustment) 联合优化所有标靶位姿与相机位姿
        平差前严格调用共视连通性安全守门员校验，杜绝奇异矩阵。
        """
        if active_frame_names is None:
            active_frame_names = [f"frame_{i:04d}" for i in range(len(frame_detections))]

        # 1. 守门员审查
        report = self.validate_covisibility(frame_detections, active_frame_names, origin_tag_id, x_align_tag_id)
        if not report["is_valid"]:
            raise CovisibilityGraphError(report["message"])

        all_detected_tags = set(report["all_tags"])
        if report["critical_bridges"]:
            print(f"[NOTE] 提示：发现 {len(report['critical_bridges'])} 对标靶仅由单张图共视支撑 (关键桥梁): {report['critical_bridges']}")

        # 2. 生成初值 (以 base_static_id 为参考基准)
        base_static_id = x_align_tag_id if x_align_tag_id in all_detected_tags else min(all_detected_tags)
        tag_poses_init = {base_static_id: np.eye(4, dtype=np.float64)}
        camera_poses_init = {} # {frame_idx: T_w_cam}

        # 迭代传递位姿初值
        changed = True
        while changed:
            changed = False
            for f_idx, tags in enumerate(frame_detections):
                if f_idx not in camera_poses_init:
                    for t_id, corners in tags.items():
                        if t_id in tag_poses_init:
                            succ, rvec, tvec = self.solve_single_tag_pnp(corners)
                            if succ:
                                T_c_t = self.rvec_tvec_to_matrix(rvec, tvec)
                                T_w_c = tag_poses_init[t_id] @ np.linalg.inv(T_c_t)
                                camera_poses_init[f_idx] = T_w_c
                                changed = True
                                break

                if f_idx in camera_poses_init:
                    T_w_c = camera_poses_init[f_idx]
                    for t_id, corners in tags.items():
                        if t_id not in tag_poses_init:
                            succ, rvec, tvec = self.solve_single_tag_pnp(corners)
                            if succ:
                                T_c_t = self.rvec_tvec_to_matrix(rvec, tvec)
                                T_w_t = T_w_c @ T_c_t
                                tag_poses_init[t_id] = T_w_t
                                changed = True

        print(f"[+] 初值生成完毕：成功初始化 {len(tag_poses_init)} 个 Tag，{len(camera_poses_init)} 个相机机位")

        # 3. 构建 BA 优化变量
        static_tags_to_opt = [t for t in tag_poses_init.keys() if t != base_static_id]
        active_frames = [f for f in camera_poses_init.keys()]

        x0 = []
        for t in static_tags_to_opt:
            rv, tv = self.matrix_to_rvec_tvec(tag_poses_init[t])
            x0.extend(rv.flatten().tolist())
            x0.extend(tv.flatten().tolist())

        for f in active_frames:
            rv, tv = self.matrix_to_rvec_tvec(camera_poses_init[f])
            x0.extend(rv.flatten().tolist())
            x0.extend(tv.flatten().tolist())

        x0 = np.array(x0, dtype=np.float64)

        def unpack_params(x):
            tags_pose = {base_static_id: np.eye(4, dtype=np.float64)}
            offset = 0
            for t in static_tags_to_opt:
                rv = x[offset:offset + 3]
                tv = x[offset + 3:offset + 6]
                tags_pose[t] = self.rvec_tvec_to_matrix(rv, tv)
                offset += 6

            cams_pose = {}
            for f in active_frames:
                rv = x[offset:offset + 3]
                tv = x[offset + 3:offset + 6]
                cams_pose[f] = self.rvec_tvec_to_matrix(rv, tv)
                offset += 6
            return tags_pose, cams_pose

        def residuals_func(x):
            tags_pose, cams_pose = unpack_params(x)
            residuals = []
            for f in active_frames:
                T_w_c = cams_pose[f]
                T_c_w = np.linalg.inv(T_w_c)
                tags = frame_detections[f]
                for t_id, corners_img in tags.items():
                    if t_id in tags_pose:
                        T_w_t = tags_pose[t_id]
                        T_c_t = T_c_w @ T_w_t
                        rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                        proj_pts, _ = cv2.projectPoints(
                            self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs
                        )
                        proj_pts = proj_pts.reshape((4, 2))
                        diff = (proj_pts - corners_img).flatten()
                        residuals.extend(diff)
            return np.array(residuals, dtype=np.float64)

        print("[*] 开始进行非线性最小二乘 (Bundle Adjustment) 全局平差优化...")
        res = least_squares(
            residuals_func, x0,
            method='trf',
            loss='huber',
            f_scale=1.0,
            ftol=1e-3,
            xtol=1e-3,
            max_nfev=120,
            verbose=1
        )

        optimized_tags_pose, optimized_cams_pose = unpack_params(res.x)
        final_residuals = residuals_func(res.x)
        rmse_px = float(np.sqrt(np.mean(final_residuals ** 2)))
        print(f"[OK] BA 优化完成！重投影均方根误差 RMSE: {rmse_px:.3f} 像素")

        # 4. 双标靶中心基线测距尺度修正 (Metric Baseline Gauge)
        scale_factor = 1.0
        real_marker_size = self.marker_size_mm
        baseline_info = None

        if baseline_pair is not None:
            id_a, id_b, real_dist_mm = baseline_pair
            optimized_tags_pose, scale_factor, real_marker_size = self.apply_baseline_scale(
                optimized_tags_pose, id_a, id_b, real_dist_mm
            )
            baseline_info = {
                "tag_a": int(id_a),
                "tag_b": int(id_b),
                "measured_dist_mm": float(real_dist_mm),
                "scale_factor": round(float(scale_factor), 6)
            }
            self.marker_size_mm = real_marker_size

        # 5. 坐标系对齐闭环：将世界坐标原点绑定到 Tag 0 中心，X 轴对齐到 Tag 1
        final_tags_map = self._align_to_scara_world(
            optimized_tags_pose, 
            origin_tag_id=origin_tag_id, 
            x_align_tag_id=x_align_tag_id
        )

        final_tags_map["rmse_reprojection_px"] = rmse_px
        final_tags_map["marker_size_mm"] = round(float(self.marker_size_mm), 3)
        final_tags_map["tag_family"] = "DICT_APRILTAG_16h5"
        final_tags_map["calibrated_images_count"] = len(active_frames)
        if baseline_info:
            final_tags_map["baseline_gauge"] = baseline_info
        return final_tags_map

    def build_map_from_manifest(self, 
                                manifest_path: str = "data/tag_calibration_images/tag_observations.yaml",
                                origin_tag_id: int = 0,
                                x_align_tag_id: int = 1,
                                baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict[str, Any]:
        """
        从审核清单直接加载过滤后的观测数据并执行 BA 优化建图
        """
        frame_detections, valid_frames, stats = self.load_observations_manifest(manifest_path)
        print(f"[+] 从审核清单成功载入: {len(valid_frames)} 张有效图像，保留观测 {stats['total_kept']} 次，排除观测 {stats['total_excluded']} 次")
        if stats["total_excluded"] > 0:
            print(f"    【已人工剔除的坏样本】:")
            for exc in stats["excluded_items"]:
                note_str = f" (备注: {exc['note']})" if exc['note'] else ""
                print(f"      - [{exc['image']}] Tag #{exc['tag_id']}{note_str}")
        if stats["dropped_single_tag_frames"]:
            for d_name, d_cnt in stats["dropped_single_tag_frames"]:
                print(f"[WARN] 图像 {d_name} 有效标靶少于 2 个 (实际={d_cnt})，已自动不参与相对刚体平差约束")

        return self.optimize_bundle_adjustment(
            frame_detections=frame_detections,
            active_frame_names=valid_frames,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )

    def build_map_from_images(self, 
                              image_paths: List[str],
                              origin_tag_id: int = 0,
                              x_align_tag_id: int = 1,
                              baseline_pair: Optional[Tuple[int, int, float]] = None,
                              manifest_path: str = "data/tag_calibration_images/tag_observations.yaml",
                              use_manifest: bool = True) -> Dict[str, Any]:
        """
        从一组多视角图像构建标靶全局地图并进行 BA 全局平差优化
        自动联动审核清单流水线 (若清单不存在则先导出，若存在则使用清单的过滤规则)
        """
        print(f"[*] 开始处理 {len(image_paths)} 张多视角标定图像...")
        
        # 探测输入图像真实分辨率，执行内参动态自适应与防呆校验
        for path in image_paths:
            probe_img = cv2.imread(path)
            if probe_img is not None:
                if resolve_camera_intrinsics is not None:
                    self.camera_matrix, self.dist_coeffs, _ = resolve_camera_intrinsics(
                        "config.yaml", actual_image_shape=probe_img.shape[:2]
                    )
                break

        if use_manifest:
            if not os.path.exists(manifest_path):
                print(f"[*] 未检测到观测清单，正在生成初始观测数据清单: {manifest_path} ...")
                self.export_observations_manifest(image_paths, manifest_path=manifest_path)
            return self.build_map_from_manifest(
                manifest_path=manifest_path,
                origin_tag_id=origin_tag_id,
                x_align_tag_id=x_align_tag_id,
                baseline_pair=baseline_pair
            )

        # 直接全量检测（备用快速通道）
        frame_detections = []
        valid_frames = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            tags = self.detect_tags(img)
            if len(tags) >= 2:
                frame_detections.append(tags)
                valid_frames.append(os.path.basename(path))

        return self.optimize_bundle_adjustment(
            frame_detections=frame_detections,
            active_frame_names=valid_frames,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )

    def apply_baseline_scale(self, 
                             tag_poses: Dict[int, np.ndarray],
                             tag_id_a: int, 
                             tag_id_b: int, 
                             real_distance_mm: float) -> Tuple[Dict[int, np.ndarray], float, float]:
        """
        利用两个标靶中心物理测量距离锁定绝对尺度 (Metric Baseline Gauge)
        :param tag_poses: 各标靶 4x4 位姿矩阵字典
        :param tag_id_a: 标靶 A 的 ID
        :param tag_id_b: 标靶 B 的 ID
        :param real_distance_mm: 现场实际测量的中心物理直线距离 (mm)
        :return: (scaled_tag_poses, scale_factor, real_marker_size_mm)
        """
        if tag_id_a not in tag_poses or tag_id_b not in tag_poses:
            print(f"[WARN] 尺度标定失败：标靶 {tag_id_a} 或 {tag_id_b} 未在重构地图中！保持名义尺度。")
            return tag_poses, 1.0, self.marker_size_mm

        p_a = tag_poses[tag_id_a][:3, 3]
        p_b = tag_poses[tag_id_b][:3, 3]
        nominal_dist = float(np.linalg.norm(p_a - p_b))

        if nominal_dist < 1e-4:
            print(f"[WARN] 标靶 {tag_id_a} 与 {tag_id_b} 距离过近，无法用作尺度基线！")
            return tag_poses, 1.0, self.marker_size_mm

        scale_factor = float(real_distance_mm) / nominal_dist
        real_marker_size = self.marker_size_mm * scale_factor

        print(f"\n[+] ====== 双标靶中心基线绝对尺度校准 (Metric Baseline Gauge) ======")
        print(f"  -> 基准标靶对: Tag #{tag_id_a} <---> Tag #{tag_id_b}")
        print(f"  -> 当前名义欧氏距离: {nominal_dist:.2f} mm")
        print(f"  -> 现场测量实际距离: {real_distance_mm:.2f} mm")
        print(f"  -> 尺度修正系数 (Scale): {scale_factor:.6f}")
        print(f"  -> 反算单个 Tag 真实物理边长: {real_marker_size:.2f} mm (名义初值: {self.marker_size_mm:.2f} mm)")
        print(f"===================================================================\n")

        # 缩放所有 Tag 的平移位置
        scaled_poses = {}
        for t_id, T in tag_poses.items():
            T_scaled = T.copy()
            T_scaled[:3, 3] = T[:3, 3] * scale_factor
            scaled_poses[t_id] = T_scaled

        return scaled_poses, scale_factor, real_marker_size

    def _align_to_scara_world(self, 
                              tag_poses: Dict[int, np.ndarray], 
                              origin_tag_id: int, 
                              x_align_tag_id: int) -> Dict:
        """
        通过刚体变换将地图整体平移旋转，使得：
        1. Tag 0 的中心处于 (0.0, 0.0)
        2. Tag 0 -> Tag 1 的水平向量严格处于 +X 轴 (Y=0, X>0)
        """
        aligned_map = {
            "origin_tag_id": origin_tag_id,
            "x_axis_align_tag_id": x_align_tag_id,
            "tags": {}
        }

        # 如果没有检测到 Tag 0，则默认以 base 标靶对齐
        if origin_tag_id in tag_poses:
            p_origin = tag_poses[origin_tag_id][:3, 3].copy()
        else:
            p_origin = np.zeros(3)
            print(f"[WARN] 未在有效图像中检出 Tag {origin_tag_id}，将以参考标靶相对对齐！")

        # 计算对齐旋转角 (绕 Z 轴旋转使得 Tag 1 的 Y 坐标归零)
        yaw_rad = 0.0
        if origin_tag_id in tag_poses and x_align_tag_id in tag_poses:
            vec_x = tag_poses[x_align_tag_id][:3, 3] - p_origin
            # 水平面夹角
            yaw_rad = math.atan2(vec_x[1], vec_x[0])
            print(f"[+] 坐标系 X 轴对齐旋转角: {-math.degrees(yaw_rad):.2f}°")

        # 构建整体刚体变换矩阵 T_world_aligned
        cos_y = math.cos(-yaw_rad)
        sin_y = math.sin(-yaw_rad)
        R_align = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        for t_id, T_w_t in tag_poses.items():
            # 1. 平移至原点
            pos_rel = T_w_t[:3, 3] - p_origin
            # 2. 旋转对齐 X 轴
            pos_aligned = R_align @ pos_rel
            R_aligned = R_align @ T_w_t[:3, :3]
            
            # 提取欧拉角 (RPY deg)
            sy = math.sqrt(R_aligned[0, 0] * R_aligned[0, 0] + R_aligned[1, 0] * R_aligned[1, 0])
            singular = sy < 1e-6
            if not singular:
                roll = math.atan2(R_aligned[2, 1], R_aligned[2, 2])
                pitch = math.atan2(-R_aligned[2, 0], sy)
                yaw = math.atan2(R_aligned[1, 0], R_aligned[0, 0])
            else:
                roll = math.atan2(-R_aligned[1, 2], R_aligned[1, 1])
                pitch = math.atan2(-R_aligned[2, 0], sy)
                yaw = 0.0

            aligned_map["tags"][t_id] = {
                "position_mm": [round(float(v), 2) for v in pos_aligned],
                "rpy_deg": [round(float(math.degrees(v)), 2) for v in [roll, pitch, yaw]],
                "transform_matrix": [[round(float(val), 5) for val in row] for row in np.vstack([np.hstack([R_aligned, pos_aligned.reshape(3, 1)]), [0, 0, 0, 1]])],
                "is_origin": bool(t_id == origin_tag_id),
                "is_dynamic_yaw": bool(t_id == origin_tag_id)
            }

        return aligned_map

    def save_map(self, map_data: Dict, output_path: str = "config/tags_map.yaml"):
        """保存标靶地图至 YAML 文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(map_data, f, allow_unicode=True, sort_keys=False)
        print(f"[OK] 标靶空间立体地图已成功保存至: {output_path}")


def print_topology_report(covis_report: Dict[str, Any], stats: Dict[str, Any], valid_frames: List[str]):
    """打印详细共视拓扑分析诊断报告"""
    print("\n" + "=" * 70)
    print("           标靶共视连通图拓扑结构深度诊断报告")
    print("=" * 70)
    print(f" 状态评估: {'[ 连通健康 (PASS) ]' if covis_report['is_valid'] else '[ 断网告警 (FAIL) ]'}")
    print(f" 诊断说明: {covis_report['message']}")
    print(f" 参与帧数: {len(valid_frames)} 张图像满足 >= 2 标靶相对刚体约束")
    print("-" * 70)
    print(" [标靶节点与连通度 (Degree)]")
    for tid in covis_report.get("all_tags", []):
        neighbors = covis_report.get("adj_list", {}).get(tid, [])
        is_conn = tid in covis_report.get("connected_tags", [])
        status_str = "正常连通" if is_conn else "【断网孤立】"
        print(f"   - Tag #{tid:02d}: 相邻标靶 {neighbors} (度数: {len(neighbors)}) -> {status_str}")

    print("\n [共视桥梁与重叠帧数 (Edge Co-visibility Count)]")
    for edge_str, count in covis_report.get("edge_counts", {}).items():
        bridge_warn = " \033[93m[单图支撑关键桥梁 - 切勿剔除]\033[0m" if count == 1 else ""
        print(f"   - 标靶对 ({edge_str}): 在 {count} 帧图像中同时出现{bridge_warn}")

    if stats.get("excluded_items"):
        print("\n [当前已人工剔除的观测清单]")
        for exc in stats["excluded_items"]:
            note = f" (备注: {exc['note']})" if exc.get("note") else ""
            print(f"   - [{exc['image']}] Tag #{exc['tag_id']}{note}")
    print("=" * 70)


def interactive_workflow(args, builder: TagMapBuilder, image_paths: List[str], baseline_pair: Optional[Tuple[int, int, float]]):
    """交互式两阶段建图与人工质量审核工作流"""
    manifest_path = args.manifest
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "visualized")

    # 1. 若清单不存在，先自动导出
    if not os.path.exists(manifest_path):
        print(f"[*] 首次运行，正在自动提取并生成观测数据清单: {manifest_path} ...")
        builder.export_observations_manifest(image_paths, manifest_path=manifest_path)

    while True:
        # 加载清单与状态
        try:
            frame_detections, valid_frames, stats = builder.load_observations_manifest(manifest_path)
            covis_report = builder.validate_covisibility(frame_detections, valid_frames, args.origin_id, args.x_axis_id)
        except Exception as e:
            print(f"[ERROR] 读取或解析观测清单异常: {e}")
            covis_report = {"is_valid": False, "message": str(e), "all_tags": [], "critical_bridges": []}
            stats = {"total_images": 0, "total_observations": 0, "total_kept": 0, "total_excluded": 0, "excluded_items": []}

        # 终端横幅
        print("\n" + "=" * 76)
        print("     【AprilTag 两阶段人工审核与 BA 空间平差控制台】(Tag Map Builder)")
        print("=" * 76)
        print(f" 采图目录: {args.image_dir} ({len(image_paths)} 帧)")
        print(f" 审核清单: {manifest_path}")
        print(f" 观测统计: 共 {stats['total_observations']} 次标靶检出 | 保留: {stats['total_kept']} | 人工剔除: {stats['total_excluded']}")
        
        if covis_report["is_valid"]:
            status_tag = f"\033[92m[ 连通网健康 (PASS) ]\033[0m"
            print(f" 拓扑状态: {status_tag} 全部 {len(covis_report['all_tags'])} 个标靶完全连通 (有效参与: {len(valid_frames)} 帧)")
        else:
            status_tag = f"\033[91m[ 拓扑断网告警 (FAIL) ]\033[0m"
            print(f" 拓扑状态: {status_tag} {covis_report['message']}")

        if covis_report.get("critical_bridges"):
            print(f" 关键桥梁: 发现 {len(covis_report['critical_bridges'])} 对标靶仅由单图连接: {covis_report['critical_bridges']} (注意不要剔除桥梁帧)")

        print("-" * 76)
        print(" 【第一阶段：数据审核与拓扑评估】")
        print("   [1] 启动交互审核画板 (鼠标点击标靶直接剔除/恢复，实时拓扑安全红绿灯)")
        print("   [2] 查看共视拓扑结构深度诊断报告 (节点度数、共视重叠帧数、关键桥梁)")
        print("")
        print(" 【辅助维护工具集 (无固定顺序，按需调用)】")
        print("   [3] 刷新/重新扫描图像并更新清单 (自动保留已有 keep: false 与备注)")
        print("   [4] 在系统资源管理器中打开可视化标注图目录 (visualized/) 查看像元标牌")
        print("   [5] (或 [E]) 在系统文本编辑器中直接编辑清单 (tag_observations.yaml)")
        print("")
        print(" 【最终收官：终审求解与成果导出】")
        print("   [6] (或 [B]) 立即执行 BA 全局平差优化并导出地图 (config/tags_map.yaml)")
        print("----------------------------------------------------------------------------")
        print("   [Q] 退出")
        print("=" * 76)

        try:
            choice = input("请输入操作编号 [1-6, E, B, Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出] 操作已终止。")
            break

        if choice == '1':
            try:
                from tools.tag_manifest_reviewer import TagManifestReviewer
                print("\n[启动] 正在启动 AprilTag 观测样本交互审核画板...")
                reviewer = TagManifestReviewer(manifest_path=manifest_path, builder=builder)
                reviewer.run()
            except Exception as e:
                print(f"\n[WARN] 启动交互画板失败或当前环境无显示服务: {e}")
                print("已自动回退到调用系统文本编辑器打开 YAML 清单。")
                abs_manifest = os.path.abspath(manifest_path)
                try:
                    if sys.platform == "win32":
                        os.startfile(abs_manifest)
                    elif sys.platform == "darwin":
                        import subprocess
                        subprocess.run(["open", abs_manifest])
                    else:
                        import subprocess
                        subprocess.run(["xdg-open", abs_manifest])
                except Exception:
                    pass
                try:
                    input("\n修改保存完毕后，请按回车键刷新...")
                except (EOFError, KeyboardInterrupt):
                    pass

        elif choice == '2':
            print_topology_report(covis_report, stats, valid_frames)
            try:
                input("\n按回车键返回菜单...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == '3':
            print(f"\n[*] 正在重新扫描所有标定图像并刷新清单 (历史人工标记将安全保留)...")
            builder.export_observations_manifest(image_paths, manifest_path=manifest_path)
            try:
                input("\n清单刷新完成，按回车键继续...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == '4':
            os.makedirs(vis_dir, exist_ok=True)
            print(f"\n[浏览] 正在打开可视化标注目录: {vis_dir} ...")
            try:
                if sys.platform == "win32":
                    os.startfile(vis_dir)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", vis_dir])
                else:
                    import subprocess
                    subprocess.run(["xdg-open", vis_dir])
            except Exception as e:
                print(f"[WARN] 打开目录失败: {e}")
            try:
                input("\n按回车键返回菜单...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice in ('5', 'E'):
            abs_manifest = os.path.abspath(manifest_path)
            print(f"\n[打开] 正在尝试调用系统默认编辑器打开: {abs_manifest} ...")
            try:
                if sys.platform == "win32":
                    os.startfile(abs_manifest)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", abs_manifest])
                else:
                    import subprocess
                    subprocess.run(["xdg-open", abs_manifest])
            except Exception as e:
                print(f"[WARN] 无法自动拉起编辑器: {e}，请手动打开该文件。")
            try:
                input("\n修改保存完毕后，请按回车键重新读取并刷新拓扑连通状态...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice in ('6', 'B'):
            if not covis_report["is_valid"]:
                print(f"\n\033[91m[拦截] 共视拓扑检查未通过，禁止执行 BA 优化！\033[0m")
                print(f"原因: {covis_report['message']}")
                print("请先选择选项 [1] 打开交互画板，恢复失联标靶的关键视角后再继续。")
                try:
                    input("\n按回车键返回菜单...")
                except (EOFError, KeyboardInterrupt):
                    pass
                continue

            try:
                tags_map = builder.optimize_bundle_adjustment(
                    frame_detections=frame_detections,
                    active_frame_names=valid_frames,
                    origin_tag_id=args.origin_id,
                    x_align_tag_id=args.x_axis_id,
                    baseline_pair=baseline_pair
                )
                builder.save_map(tags_map, args.output)
                print(f"\n\033[92m[SUCCESS] 标靶空间地图优化求解大功告成！已成功输出至: {args.output}\033[0m")
                break
            except Exception as e:
                print(f"\n\033[91m[ERROR] BA 平差求解失败: {e}\033[0m")
                try:
                    input("\n按回车键返回菜单...")
                except (EOFError, KeyboardInterrupt):
                    pass

        elif choice in ('Q', '0'):
            print("\n[退出] 已退出建图工具。")
            break


def main():
    parser = argparse.ArgumentParser(description="AprilTag 16h5 多标靶离线两阶段建图与 BA 平差工具")
    parser.add_argument("--image_dir", type=str, default="data/tag_calibration_images", help="多视角标定图片目录")
    parser.add_argument("--manifest", type=str, default="data/tag_calibration_images/tag_observations.yaml", help="观测数据审核清单路径")
    parser.add_argument("--marker_size", type=float, default=50.0, help="标靶黑白边框名义边长 (mm)")
    parser.add_argument("--origin_id", type=int, default=0, help="SCARA 原点锚定标靶 ID")
    parser.add_argument("--x_axis_id", type=int, default=1, help="世界 X 轴对齐基准标靶 ID")
    parser.add_argument("--baseline_pair", nargs=3, type=float, metavar=('TAG_A', 'TAG_B', 'DIST_MM'),
                        help="双标靶基线尺度校准参数: TAG_A TAG_B 真实距离(mm), 例如: --baseline_pair 1 5 620.5")
    parser.add_argument("--output", type=str, default="config/tags_map.yaml", help="导出的图谱文件路径")
    parser.add_argument("--export-manifest", action="store_true", help="仅扫描图像导出观测数据清单并退出")
    parser.add_argument("--solve-manifest", action="store_true", help="直接读取清单执行 BA 平差 (非交互批处理)")
    parser.add_argument("--inspect", action="store_true", help="仅执行连通性深度诊断并输出报告后退出")
    parser.add_argument("--no-interactive", action="store_true", help="静默非交互模式运行")
    args = parser.parse_args()

    # 寻找图像文件
    patterns = [os.path.join(args.image_dir, ext) for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat))

    if not image_paths and not os.path.exists(args.manifest):
        print(f"[!] 目录 '{args.image_dir}' 下未找到任何标定图像，且未找到已有清单 '{args.manifest}'！")
        print(f"[*] 提示：请使用相机采集覆盖多标靶的图像放入该目录后重试。")
        sys.exit(1)

    builder = TagMapBuilder(marker_size_mm=args.marker_size)

    baseline_pair = None
    if args.baseline_pair is not None:
        baseline_pair = (int(args.baseline_pair[0]), int(args.baseline_pair[1]), float(args.baseline_pair[2]))

    # 单独模式 1: 仅导出清单
    if args.export_manifest:
        builder.export_observations_manifest(image_paths, manifest_path=args.manifest)
        print("[OK] 清单导出完毕，已退出。")
        return

    # 单独模式 2: 仅深度诊断
    if args.inspect:
        if not os.path.exists(args.manifest):
            builder.export_observations_manifest(image_paths, manifest_path=args.manifest)
        f_det, v_frames, stats = builder.load_observations_manifest(args.manifest)
        report = builder.validate_covisibility(f_det, v_frames, args.origin_id, args.x_axis_id)
        print_topology_report(report, stats, v_frames)
        return

    # 单独模式 3: 直接静默求解
    if args.solve_manifest or args.no_interactive or (not sys.stdin.isatty()):
        if not os.path.exists(args.manifest):
            builder.export_observations_manifest(image_paths, manifest_path=args.manifest)
        tags_map = builder.build_map_from_manifest(
            manifest_path=args.manifest,
            origin_tag_id=args.origin_id,
            x_align_tag_id=args.x_axis_id,
            baseline_pair=baseline_pair
        )
        builder.save_map(tags_map, args.output)
        return

    # 默认模式: 启动交互式审核与建图控制台
    interactive_workflow(args, builder, image_paths, baseline_pair)


if __name__ == "__main__":
    main()
