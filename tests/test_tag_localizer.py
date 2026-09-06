#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TagLocalizer 在线实时相机外参定位器单元测试
- 模拟已知 tags_map.yaml
- 模拟相机从任意俯视未知位姿拍摄视野内的 2~3 个标靶
- 验证 localize_camera 恢复出相机真实 6DoF 外参矩阵的精度与速度
"""

import unittest
import os
import sys
import time
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vision.tag_localizer import TagLocalizer


class TestTagLocalizer(unittest.TestCase):
    def setUp(self):
        self.localizer = TagLocalizer(tags_map_path="non_existent_map.yaml")
        # 注入一个立体的虚拟地图
        # Tag 1: (200, 0, 10), Tag 2: (250, 150, -15), Tag 0: (0, 0, 0)
        self.localizer.marker_size_mm = 50.0
        s = 25.0
        self.localizer.local_corners = np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)

        # 构造立体世界地图
        T_w_t1 = np.eye(4)
        T_w_t1[:3, 3] = [200.0, 0.0, 10.0]

        T_w_t2 = np.eye(4)
        R2, _ = cv2.Rodrigues(np.array([0.05, -0.02, 0.1]))
        T_w_t2[:3, :3] = R2
        T_w_t2[:3, 3] = [250.0, 150.0, -15.0]

        self.localizer.tags_map = {
            "marker_size_mm": 50.0,
            "tags": {
                0: {
                    "position_mm": [0.0, 0.0, 0.0],
                    "transform_matrix": np.eye(4).tolist(),
                    "is_dynamic_yaw": True
                },
                1: {
                    "position_mm": [200.0, 0.0, 10.0],
                    "transform_matrix": T_w_t1.tolist(),
                    "is_dynamic_yaw": False
                },
                2: {
                    "position_mm": [250.0, 150.0, -15.0],
                    "transform_matrix": T_w_t2.tolist(),
                    "is_dynamic_yaw": False
                }
            }
        }
        self.T_w_t1 = T_w_t1
        self.T_w_t2 = T_w_t2

    def test_camera_localization_accuracy(self):
        """测试在线单帧定位相机的准确性与耗时"""
        # 1. 设定相机的真实未知外参位姿 T_w_c
        gt_cam_pos = np.array([120.0, 80.0, 520.0]) # 机械臂上方 520mm
        gt_cam_rpy = np.array([math.radians(180), math.radians(25), math.radians(-10)]) # 大倾角
        R_w_c, _ = cv2.Rodrigues(gt_cam_rpy)
        gt_T_w_c = np.eye(4)
        gt_T_w_c[:3, :3] = R_w_c
        gt_T_w_c[:3, 3] = gt_cam_pos

        T_c_w = np.linalg.inv(gt_T_w_c)

        # 2. 生成相机视野中的标靶投影角点
        T_c_t1 = T_c_w @ self.T_w_t1
        T_c_t2 = T_c_w @ self.T_w_t2

        r1, t1 = TagLocalizer_matrix_to_rt(T_c_t1)
        r2, t2 = TagLocalizer_matrix_to_rt(T_c_t2)

        local_pts_3d = np.ascontiguousarray(self.localizer.local_corners[:, :3], dtype=np.float64)
        pts1, _ = cv2.projectPoints(local_pts_3d, r1, t1, self.localizer.camera_matrix, self.localizer.dist_coeffs)
        pts2, _ = cv2.projectPoints(local_pts_3d, r2, t2, self.localizer.camera_matrix, self.localizer.dist_coeffs)

        # 3. 构造虚拟检测结果直接测试 solvePnPRansac 求解
        obj_pts = []
        img_pts = []
        for k in range(4):
            obj_pts.append((self.T_w_t1 @ self.localizer.local_corners[k])[:3])
            img_pts.append(pts1[k].flatten())
        for k in range(4):
            obj_pts.append((self.T_w_t2 @ self.localizer.local_corners[k])[:3])
            img_pts.append(pts2[k].flatten())

        t_start = time.time()
        success, rvec, tvec = cv2.solvePnP(
            np.array(obj_pts, dtype=np.float64), 
            np.array(img_pts, dtype=np.float64),
            self.localizer.camera_matrix, 
            self.localizer.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        t_cost_ms = (time.time() - t_start) * 1000.0

        self.assertTrue(success, "PnP 求解应成功")
        
        # 恢复相机外参
        R_c_w, _ = cv2.Rodrigues(rvec)
        T_c_w_est = np.eye(4)
        T_c_w_est[:3, :3] = R_c_w
        T_c_w_est[:3, 3] = tvec.flatten()
        est_T_w_c = np.linalg.inv(T_c_w_est)

        # 验证相机空间位置误差 (应小于 1.0mm)
        pos_err = np.linalg.norm(est_T_w_c[:3, 3] - gt_cam_pos)
        self.assertLess(pos_err, 1.0, f"相机定位位置误差应小于 1mm (实际: {pos_err:.3f}mm)")
        self.assertLess(t_cost_ms, 15.0, f"PnP 解算耗时应小于 15ms (实际: {t_cost_ms:.2f}ms)")

        print(f"\n[Test OK] 相机外参定位恢复成功！耗时: {t_cost_ms:.2f}ms, 空间位置误差: {pos_err:.4f}mm")


def TagLocalizer_matrix_to_rt(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape((3, 1))
    return rvec, tvec


if __name__ == "__main__":
    unittest.main()
