#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归测试: 单靶 PnP 平面二义性 180° 翻转的先验纠偏
合成一个 45° 斜视标靶, 加入像素噪声, 验证:
1. 无先验时可能选中翻转解 (法向反 180°)
2. 传入 expected_z_cam 后必然选中与先验同向的解
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2
from calibration.offline_engine import OfflineVerificationEngine

rng = np.random.default_rng(42)
engine = OfflineVerificationEngine.__new__(OfflineVerificationEngine)
engine.camera_matrix = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1.0]])
engine.dist_coeffs = np.zeros((5, 1))
s = 25.0
engine.obj_points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

def make_obs(theta_eff_deg, az_deg, dist=420.0, noise_px=0.5, yaw_deg=0.0, obj=None):
    """构造斜视观测: theta_eff = 标靶法向与相机光轴的夹角 (有效视角, 保证非掠射)"""
    th, ph, yaw = np.radians([theta_eff_deg, az_deg, yaw_deg])
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    axis = np.cross([0.0, 0.0, 1.0], n)
    s = float(np.linalg.norm(axis))
    ang = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    R_align = np.eye(3) if s < 1e-9 else cv2.Rodrigues(axis / s * ang)[0]
    R_t_c = R_align @ cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]  # 含靶面内自转
    t_t_c = np.array([8.0, -5.0, dist])   # 标靶中心在相机系
    pts = engine.obj_points if obj is None else obj
    proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(R_t_c)[0], t_t_c,
                                engine.camera_matrix, engine.dist_coeffs)
    c = proj.reshape(4, 2) + rng.normal(0, noise_px, (4, 2))
    return c, R_t_c, t_t_c

n = 300
print(f"{'距离':>6} {'噪声':>5} {'边长偏差':>7} | 无先验翻转 / 带先验翻转")
for dist in (500.0, 800.0, 1200.0):
    for noise in (0.3, 0.6, 1.0):
        for size_err in (1.0, 1.02):
            fails_no_prior, fails_prior = 0, 0
            s_true = 25.0 * size_err
            obj_true = np.array([[-s_true, s_true, 0], [s_true, s_true, 0],
                                 [s_true, -s_true, 0], [-s_true, -s_true, 0]], dtype=np.float64)
            for i in range(n):
                tilt = rng.uniform(30, 70)    # 有效视角 30-70°, 45° 附近为二义性高发区
                az = rng.uniform(0, 360)
                yaw = rng.uniform(0, 360)
                c, R_true, t_true = make_obs(tilt, az, dist=dist, noise_px=noise,
                                             yaw_deg=yaw, obj=obj_true)
                z_true_cam = R_true[:, 2]     # 真值法向 (相机系)

                ok0, r0, _ = engine.solve_single_tag_pnp(c)                    # 无先验
                ok1, r1, _ = engine.solve_single_tag_pnp(c, expected_z_cam=z_true_cam)  # 带先验

                if ok0 and float(cv2.Rodrigues(r0)[0][:, 2] @ z_true_cam) < 0:
                    fails_no_prior += 1
                if ok1 and float(cv2.Rodrigues(r1)[0][:, 2] @ z_true_cam) < 0:
                    fails_prior += 1
                    print(f"\n[先验失效] dist={dist} noise={noise} size_err={size_err} "
                          f"tilt={tilt:.1f} az={az:.1f} yaw={yaw:.1f}")
                    succ, rvs, tvs, _ = cv2.solvePnPGeneric(
                        engine.obj_points, c, engine.camera_matrix, engine.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    for k, (rk, tk) in enumerate(zip(rvs, tvs)):
                        Rk = cv2.Rodrigues(rk)[0]
                        proj, _ = cv2.projectPoints(engine.obj_points, rk, tk,
                                                    engine.camera_matrix, engine.dist_coeffs)
                        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - c, axis=1)))
                        print(f"  解{k}: dot(法向,先验)={float(Rk[:, 2] @ z_true_cam):+.3f} "
                              f"深度={tk[2, 0]:.1f}mm 误差={err:.3f}px")
            print(f"{dist:6.0f} {noise:5.1f} {size_err:7.2f} | "
                  f"{fails_no_prior:4d} ({fails_no_prior/n*100:4.1f}%) / "
                  f"{fails_prior:4d} ({fails_prior/n*100:4.1f}%)")

assert fails_prior == 0, "先验纠偏失效!"
print(f"\n[PASS] expected_z_cam 先验在全部 18 组参数 ({18*n} 样本) 下完全消除 180° 翻转")
