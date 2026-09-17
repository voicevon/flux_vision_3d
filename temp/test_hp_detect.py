# -*- coding: utf-8 -*-
"""验证 Tag 2 高精度二次识别 (ROI 放大重检): 合成图对比粗检 vs 高精度角点误差
引擎使用 tracker 真实引擎 OfflineVerificationEngine"""
import sys
import numpy as np
import cv2

sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")
from src.calibration.offline_engine import OfflineVerificationEngine
from tools.tracker.app import RobotOnlineTracker

# 合成 1280x720 图, Tag 2 (16h5) 以带透视的四边形贴入
K = np.array([[1363.68, 0, 971.19], [0, 1361.19, 566.26], [0, 0, 1.0]])
engine = OfflineVerificationEngine(
    tags_map={}, camera_matrix=K, dist_coeffs=np.zeros((5, 1)), marker_size_mm=40.0)

marker = cv2.aruco.generateImageMarker(engine.dictionary, 2, 200)
marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

img = np.full((720, 1280, 3), 210, dtype=np.uint8)
src = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
gt = np.array([[420, 300], [580, 282], [596, 442], [412, 462]], dtype=np.float32)  # GT 四边形
M = cv2.getPerspectiveTransform(src, gt)
warp = cv2.warpPerspective(marker, M, (1280, 720), borderValue=(210, 210, 210))
mask = np.full((720, 1280), 255, np.uint8)
cv2.fillPoly(mask, [gt.astype(np.int32)], 0)
img[mask == 0] = warp[mask == 0]

# 模拟真实成像: 轻微模糊 + 噪声
img = cv2.GaussianBlur(img, (3, 3), 0.8)
noise = np.random.default_rng(7).normal(0, 5, img.shape).astype(np.float32)
img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# 1) 全景粗检 (真实引擎, 无白名单过滤)
det = engine.detect_tags(img)
assert 2 in det, f"全景未检出 Tag 2, 实际: {list(det.keys())}"
coarse = det[2].reshape(4, 2)
err_coarse = float(np.mean(np.linalg.norm(coarse - gt, axis=1)))

# 2) 高精度二次识别 (走 tracker 方法)
app = RobotOnlineTracker.__new__(RobotOnlineTracker)
app.target_tag_id = 2
app.engine = engine
det_hp = app._detect_high_precision(img, dict(det))
assert 2 in det_hp, "高精度流程丢失 Tag 2"
hp = det_hp[2].reshape(4, 2)
err_hp = float(np.mean(np.linalg.norm(hp - gt, axis=1)))

print(f"粗检   平均角点误差: {err_coarse:.3f} px")
print(f"高精度 平均角点误差: {err_hp:.3f} px")
print(f"提升: {err_coarse / max(err_hp, 1e-9):.2f}x")
assert err_hp <= err_coarse * 1.05, "高精度结果未达粗检水平"
print("[PASS] 高精度二次识别有效")
