#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试: 验证 F1 八步流水线 (含 3.融合) 与 Studio 双分支视图逻辑"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline
from src.utils.text_rendering import put_text

# 合成图: 暗底 + 绿色主干 x2 + 灰白反光 + 细线
color = np.zeros((480, 640, 3), dtype=np.uint8)
color[:] = (60, 60, 60)
cv2.rectangle(color, (100, 100), (140, 400), (40, 160, 80), -1)
cv2.rectangle(color, (300, 120), (340, 430), (40, 160, 80), -1)
cv2.rectangle(color, (200, 200), (280, 220), (200, 200, 200), -1)
cv2.line(color, (400, 50), (400, 450), (220, 220, 220), 1)
put_text(color, "SMOKE", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

pipe = FengGreenAxisPipeline()

# --- get_steps 十步声明 (含 2S/3A) ---
steps = pipe.get_steps()
keys = [s.key for s in steps]
names = [s.name for s in steps]
expect_keys = ["stage1_hsv_mask", "stage2a_morph", "stage2s_split", "stage3a_axis", "stage3_fuse",
               "stage4_region", "stage5_boxes", "stage6_axis", "stage7_stem", "stage8_midpoint"]
assert keys == expect_keys, f"步骤键不符: {keys}"
print(f"[get_steps] 10 步 OK: {names}")

# --- STEP_SLIDERS 含 stage3_fuse Canny 滑条 ---
assert "stage3_fuse" in pipe.STEP_SLIDERS
attrs = {sp["attr"] for sp in pipe.STEP_SLIDERS["stage3_fuse"]}
assert attrs == {"canny_lo", "canny_hi"}, attrs
assert hasattr(pipe, "canny_lo") and hasattr(pipe, "canny_hi")
print(f"[STEP_SLIDERS] stage3_fuse OK, canny {pipe.canny_lo}/{pipe.canny_hi}")

# --- run 全流程: 8 快照齐全 ---
result = pipe.run(color, None, nominal_z_mm=640.0)
snap_keys = set(result.step_snapshots.keys())
assert snap_keys == set(expect_keys), f"快照键不符: {snap_keys}"
print(f"[run] 8 快照齐全, targets={len(result.targets)}, "
      f"fused_px={result.extra_metrics['fused_pixels']}, "
      f"boxes={result.extra_metrics['boxes_found']}")

# --- 融合并集正确性: mask_fused == mask_open | edges_bridged ---
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                         (pipe.h_high, pipe.s_high, pipe.v_high))
mask_a, _ = pipe._stage2a_morph(green_mask)
bridged = pipe._stage2b_edge_morph(pipe._stage1b_edges(color))
fused_ref = cv2.bitwise_or(mask_a, bridged)
assert result.extra_metrics["fused_pixels"] == int(cv2.countNonZero(fused_ref)), "融合像素数不一致"
# A 子集关系: mask_a 全部包含于 fused
assert cv2.countNonZero(cv2.bitwise_and(mask_a, fused_ref)) == cv2.countNonZero(mask_a)
print(f"[fuse] 并集 OK: A={cv2.countNonZero(mask_a)}px, B-bridged={cv2.countNonZero(bridged)}px, "
      f"Fused={cv2.countNonZero(fused_ref)}px")

# --- canny 滑条联动: 阈值升高后边缘像素单调不增 ---
pipe.canny_lo, pipe.canny_hi = 100, 260
result2 = pipe.run(color, None, nominal_z_mm=640.0)
assert result2.extra_metrics["fused_pixels"] <= result.extra_metrics["fused_pixels"]
print(f"[canny tuning] 100/260 fused_px={result2.extra_metrics['fused_pixels']} (<= 60/160) OK")
pipe.canny_lo, pipe.canny_hi = 60, 160

# --- 实时预览器齐全 ---
for m in ("preview_stage1_hsv_mask", "preview_stage2a_morph", "preview_stage3_fuse",
          "preview_stage3a_axis"):
    pv = getattr(pipe, m)(color)
    assert pv.shape == color.shape, m
print("[preview] 1A / 2A / 3 融合 / 3A 实时预览 OK")

# --- 持久化采集模拟: canny 参数会被 STEP_SLIDERS 机制捕获 ---
attrs_saved = set()
for specs in FengGreenAxisPipeline.STEP_SLIDERS.values():
    for sp in specs:
        for key in ("attr", "attr_low", "attr_high"):
            if key in sp:
                attrs_saved.add(sp[key])
assert {"canny_lo", "canny_hi", "morph_ksize", "min_blob_area",
        "h_low", "h_high", "s_min", "s_high", "v_min", "v_high"} <= attrs_saved
print(f"[persist] 可持久化参数: {sorted(attrs_saved)}")

print("SMOKE ALL PASSED")
