# -*- coding: utf-8 -*-
"""验证 _tag_local_frame 法向方向: 世界角点重构坐标系应与单靶 PnP 位姿同向 (修复绿/蓝棱柱反向)"""
import sys
import numpy as np
import cv2

sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")
from tools.tracker.common import _tag_local_frame, _PRISM_PTS
from src.calibration.offline_engine import OfflineVerificationEngine

# 任意外参: 相机看着一块斜置标靶
K = np.array([[1363.68, 0, 971.19], [0, 1361.19, 566.26], [0, 0, 1.0]])
dist = np.zeros((5, 1))
r_true, t_true = np.array([[0.3], [0.0], [0.0]]), np.array([[10.0], [-20.0], [600.0]])

engine = OfflineVerificationEngine(
    tags_map={}, camera_matrix=K, dist_coeffs=dist, marker_size_mm=40.0)
proj, _ = cv2.projectPoints(engine.obj_points, r_true, t_true, K, dist)
corners_img = proj.reshape(4, 2)

# 1) 蓝色路径: 单靶 PnP 位姿
ok, r_pnp, t_pnp = engine.solve_single_tag_pnp(corners_img)
assert ok, "PnP 失败"

# 2) 绿色路径: 构造世界角点 (真值位姿) -> _tag_local_frame -> 合成相机系位姿
T_w_t = np.eye(4)
R_t, _ = cv2.Rodrigues(r_true)
T_w_t[:3, :3], T_w_t[:3, 3] = R_t.reshape(3, 3), t_true.reshape(3)
s = engine.marker_size_mm / 2.0
local = np.array([[-s, s, 0, 1], [s, s, 0, 1], [s, -s, 0, 1], [-s, -s, 0, 1]], dtype=np.float64)
wc = (T_w_t @ local.T).T[:, :3]  # 世界角点 (4,3), 顺序同 get_tag_world_corners

R_frame, c_w = _tag_local_frame(wc)
# 法向一致性: 重构 Z 轴 与 PnP 旋转第三列 点积应为 +1 (同向)
R_pnp_m, _ = cv2.Rodrigues(r_pnp)
dot = float(R_frame[:, 2] @ R_pnp_m[:, 2])
print(f"重构法向 · PnP法向 = {dot:+.4f}")
assert dot > 0.99, "法向仍反向!"

# 合成相机系位姿 vs 直接 PnP 位姿: 平移应一致 (mm 级)
R_wc = np.eye(3)  # 世界系 = 标靶真值所在系 (无额外变换)
t_ct = R_wc @ c_w
print(f"绿色合成平移: {t_ct.round(2)} | 蓝色 PnP 平移: {np.asarray(t_pnp).reshape(3).round(2)}")
assert np.allclose(t_ct, np.asarray(t_pnp).reshape(3), atol=0.5), "平移不一致"

# 棱柱顶面中心应沿 +Z 朝外 (远离安装面): 顶面中心世界坐标 = c + 75*法向
top_center_w = c_w + R_frame[:, 2] * 75.0
assert np.allclose(top_center_w - c_w, R_t.reshape(3, 3) @ np.array([0, 0, 75.0]), atol=1e-6)
print("[PASS] 绿/蓝棱柱法向同向, 顶面中心沿面外 +75mm")
