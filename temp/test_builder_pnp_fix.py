#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归测试: TagMapBuilder.solve_single_tag_pnp 平面二义性翻转纠偏
路径 A: 无显式先验 -> 内置"朝天"先验 (R[1,2]<0, 相机正立俯视工作台)
路径 B: 显式 expected_z_cam -> 法向同半球过滤 (与引擎版同接口)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from tools.calibration.tag_map_builder import TagMapBuilder

rng = np.random.default_rng(7)
b = TagMapBuilder.__new__(TagMapBuilder)
b.camera_matrix = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1.0]])
b.dist_coeffs = np.zeros((5, 1))
s = 25.0
b.obj_points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

SIZE_ERR = 1.02
s_t = s * SIZE_ERR
obj_true = np.array([[-s_t, s_t, 0], [s_t, s_t, 0], [s_t, -s_t, 0], [-s_t, -s_t, 0]], dtype=np.float64)

def make_obs(n, dist=800.0, noise_px=0.6, yaw_deg=0.0):
    """n: 真值法向 (相机系, 单位向量); 靶心在相机正前方 dist 处"""
    axis = np.cross([0.0, 0.0, 1.0], n)
    sn = float(np.linalg.norm(axis))
    ang = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    R_align = np.eye(3) if sn < 1e-9 else cv2.Rodrigues(axis / sn * ang)[0]
    R_t_c = R_align @ cv2.Rodrigues(np.array([0.0, 0.0, np.radians(yaw_deg)]))[0]
    t_t_c = np.array([8.0, -5.0, dist])
    proj, _ = cv2.projectPoints(obj_true, cv2.Rodrigues(R_t_c)[0], t_t_c,
                                b.camera_matrix, b.dist_coeffs)
    return proj.reshape(4, 2) + rng.normal(0, noise_px, (4, 2)), R_t_c

n_cases, fail_a, fail_b = 600, 0, 0
for i in range(n_cases):
    th = np.radians(rng.uniform(30, 70))          # 有效视角 30-70°
    yaw = rng.uniform(0, 360)
    # 路径 A 场景: 相机正立俯视, 标靶法向朝上 -> 相机系 Y 分量必为负 (R[1,2]<0)
    ph = np.radians(rng.uniform(180, 360))        # 方位限制在上半空间 (n_y<0)
    n_a = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    c, R_true = make_obs(n_a, yaw_deg=yaw)
    ok, r, _ = b.solve_single_tag_pnp(c)          # 内置朝天先验
    if ok and float(cv2.Rodrigues(r)[0][1, 2]) >= 0:
        fail_a += 1

    # 路径 B 场景: 任意方位法向, 显式传入真值法向
    ph2 = np.radians(rng.uniform(0, 360))
    n_b = np.array([np.sin(th) * np.cos(ph2), np.sin(th) * np.sin(ph2), np.cos(th)])
    c2, R_true2 = make_obs(n_b, yaw_deg=yaw)
    ok2, r2, _ = b.solve_single_tag_pnp(c2, expected_z_cam=R_true2[:, 2])
    if ok2 and float(cv2.Rodrigues(r2)[0][:, 2] @ n_b) < 0:
        fail_b += 1

print(f"样本 {n_cases} (视角 30-70°, 噪声 0.6px, 边长偏差 2%)")
print(f"路径 A 内置朝天先验翻转: {fail_a}")
print(f"路径 B 显式先验翻转:    {fail_b}")
assert fail_a == 0 and fail_b == 0, "建图版先验纠偏失效!"
print("[PASS] TagMapBuilder.solve_single_tag_pnp 两条先验路径均无 180° 翻转")
