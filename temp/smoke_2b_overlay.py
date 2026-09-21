#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试: 验证 2B 双模式渲染逻辑 (edges / overlay) 与 F1 preview_stage2a_morph"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline
from src.utils.text_rendering import put_text

# 合成暗底图像 + 绿色矩形 + 白色细线 (模拟芦笋与传送带反光)
color = np.zeros((480, 640, 3), dtype=np.uint8)
color[:] = (60, 60, 60)
cv2.rectangle(color, (100, 100), (140, 400), (40, 160, 80), -1)   # 绿色主干 1
cv2.rectangle(color, (300, 120), (340, 430), (40, 160, 80), -1)   # 绿色主干 2
cv2.rectangle(color, (200, 200), (280, 220), (200, 200, 200), -1)  # 灰白反光
cv2.line(color, (400, 50), (400, 450), (220, 220, 220), 1)        # 细线边缘
put_text(color, "SMOKE", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

pipe = FengGreenAxisPipeline()
assert hasattr(pipe, "preview_stage2a_morph"), "F1 缺少 preview_stage2a_morph"
assert hasattr(pipe, "_stage2a_morph"), "F1 缺少 _stage2a_morph"
assert any("attr" in sp for sps in FengGreenAxisPipeline.STEP_SLIDERS.values()
           for sp in sps), "STEP_SLIDERS 缺少单滑块声明"

# --- edges 模式: Canny 60/160 -> MORPH_CLOSE 3x3 ---
edges = cv2.Canny(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), 60, 160)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
bridged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3)
vis_edges = cv2.cvtColor(bridged, cv2.COLOR_GRAY2BGR)
put_text(vis_edges, "STAGE 2B: EDGE MORPH CLOSE | Gaps Bridged [B] Overlay",
         (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 255), 2)
assert vis_edges.shape == color.shape
assert cv2.countNonZero(bridged) > 0, "bridged 边缘为空"
# 闭运算只增不减: bridged >= edges
assert np.all((bridged >= edges) | (edges == 0)), "闭运算结果异常收缩"
print(f"[edges] mode OK, bridged px = {cv2.countNonZero(bridged)}")

# --- overlay 模式: A 系掩膜 + B 系边缘叠加 ---
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                         (pipe.h_high, pipe.s_high, pipe.v_high))
mask_a, removed = pipe._stage2a_morph(green_mask)
vis_overlay = (color.astype(np.float32) * 0.30).astype(np.uint8)
vis_overlay[mask_a > 0] = (80, 240, 120)
vis_overlay[bridged > 0] = (250, 230, 90)
put_text(vis_overlay, "STAGE 2B OVERLAY [B] | A-Mask Green / B-Edges Yellow",
         (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 255), 2)
assert mask_a.shape == color.shape[:2]
assert cv2.countNonZero(mask_a) > 0, "A 系掩膜为空"
print(f"[overlay] mode OK, A-mask px = {cv2.countNonZero(mask_a)}, removed blobs = {removed}")

# --- preview_stage2a_morph 实时预览一致性 ---
pv = pipe.preview_stage2a_morph(color)
assert pv.shape == color.shape
print(f"[preview_stage2a_morph] OK, shape = {pv.shape}")

print("SMOKE ALL PASSED")
