#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试: 验证 F1 九步流水线 (含 3A.轴线提取)"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline
from src.utils.text_rendering import put_text

# 合成图: 暗底 + 绿色主干 x2 (斜置一根) + 灰白反光 + 细线
color = np.zeros((480, 640, 3), dtype=np.uint8)
color[:] = (60, 60, 60)
cv2.rectangle(color, (100, 100), (140, 400), (40, 160, 80), -1)
cv2.line(color, (280, 430), (360, 120), (40, 160, 80), 40)   # 斜置主干
cv2.rectangle(color, (200, 200), (280, 220), (200, 200, 200), -1)
cv2.line(color, (400, 50), (400, 450), (220, 220, 220), 1)
put_text(color, "SMOKE", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

pipe = FengGreenAxisPipeline()

# --- get_steps 十步声明: 3A 位于 2S 与 3 融合之间 ---
steps = pipe.get_steps()
keys = [s.key for s in steps]
names = [s.name for s in steps]
expect_keys = ["stage1_hsv_mask", "stage2a_morph", "stage2s_split", "stage3a_axis", "stage3_fuse",
               "stage4_region", "stage5_boxes", "stage6_axis", "stage7_stem", "stage8_midpoint"]
assert keys == expect_keys, f"步骤键不符: {keys}"
assert names[3] == "3A.轴线提取", names
print(f"[get_steps] 10 步 OK: {names}")

# --- run 全流程: 10 快照齐全 ---
result = pipe.run(color, None, nominal_z_mm=640.0)
snap_keys = set(result.step_snapshots.keys())
assert snap_keys == set(expect_keys), f"快照键不符: {snap_keys}"
print(f"[run] 10 快照齐全, axes_early={result.extra_metrics['axes_early']}, "
      f"boxes={result.extra_metrics['boxes_found']}, targets={len(result.targets)}")

# --- 3A 轴线提取正确性: 手工复算对比 (新接口: instances 标签图) ---
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                         (pipe.h_high, pipe.s_high, pipe.v_high))
mask_open, _ = pipe._stage2a_morph(green_mask)
instances, n_ind = pipe._stage2s_split(mask_open)
axes = pipe._stage3a_axis(instances)
assert len(axes) >= 2, f"斜置+竖直主干应至少提取 2 条轴线, 实际 {len(axes)}"
a1 = axes[0]      # 最长轴
vx, vy = a1["axis"]
ang = abs(np.degrees(np.arctan2(vy, vx)))
# 斜置主干 (近垂直, atan2 vy/vx → 接近 ±90°)
assert 60 <= ang <= 120, f"最长轴应为斜置主干 (近垂直), 实际角度 {ang:.1f}°"
assert a1["len_px"] > 200, f"主干轴长应 > 200px, 实际 {a1['len_px']:.0f}px"
# 轴长降序排列
lens = [a["len_px"] for a in axes]
assert lens == sorted(lens, reverse=True), "轴线未按长度降序"
print(f"[3A axis] {len(axes)} 条轴线, 最长轴 L={a1['len_px']:.0f}px ang={ang:.1f}° OK")

# --- 步骤 3A 快照可视化非空且带白线 ---
snap = result.step_snapshots["stage3a_axis"]
white = np.all(snap > 240, axis=2).sum()
assert white > 50, "3A 快照缺少白色轴线像素"
print(f"[3A vis] 白色轴线像素 {white} OK")

print("SMOKE ALL PASSED")
