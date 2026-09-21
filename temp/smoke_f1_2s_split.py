#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试: 验证 F1 十步流水线 (含 2S.个体分离) —— 并排贴合双条被正确分离"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline
from src.utils.text_rendering import put_text

# 合成图: 两根圆头竖条(胶囊形)黏连 (模拟 0011 芦笋圆头贴合) + 一根独立单条
color = np.zeros((480, 640, 3), dtype=np.uint8)
color[:] = (60, 60, 60)
green = (40, 160, 80)
# 贴合对: 左中心 x=170, 右中心 x=224, 中段 13px 缝 (CLOSE 7 不焊死), 由一块黏连桥连接
for cx in (170, 224):
    cv2.rectangle(color, (cx - 20, 100), (cx + 20, 400), green, -1)
cv2.rectangle(color, (193, 245), (201, 255), green, -1)          # 黏连桥 (9px 宽, 开运算存活)
# 独立单根
cv2.rectangle(color, (410, 100), (450, 400), green, -1)
cv2.circle(color, (430, 100), 21, green, -1)
cv2.circle(color, (430, 400), 21, green, -1)
put_text(color, "SMOKE", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

pipe = FengGreenAxisPipeline()

# --- 2A 后确实黏连为一个连通域 ---
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                         (pipe.h_high, pipe.s_high, pipe.v_high))
mask_open, _ = pipe._stage2a_morph(green_mask)
n_cc = cv2.connectedComponents(mask_open, connectivity=8)[0] - 1
print(f"[2A] 连通域数 = {n_cc} (黏连证据: 贴合对合并)")
assert n_cc == 2, f"期望黏连后 2 个连通域(贴合对+单根), 实际 {n_cc}"

# --- get_steps 十步声明 ---
steps = pipe.get_steps()
keys = [s.key for s in steps]
expect = ["stage1_hsv_mask", "stage2a_morph", "stage2s_split", "stage3a_axis", "stage3_fuse",
          "stage4_region", "stage5_boxes", "stage6_axis", "stage7_stem", "stage8_midpoint"]
assert keys == expect, keys
print(f"[get_steps] 10 步 OK: {[s.name for s in steps]}")

# --- run 全流程: 10 快照 + 2S 分出 3 个体 ---
result = pipe.run(color, None, nominal_z_mm=640.0)
assert set(result.step_snapshots.keys()) == set(expect), result.step_snapshots.keys()
n_ind = result.extra_metrics["individuals"]
n_axes = result.extra_metrics["axes_early"]
print(f"[run] 10 快照齐全, individuals={n_ind}, axes_early={n_axes}")
assert n_ind == 3, f"期望分离出 3 个个体, 实际 {n_ind}"
assert n_axes == 3, f"期望 3 条轴线, 实际 {n_axes}"

# --- 2S 分离正确性: 贴合对拆成 2 个体, 且 3 条轴都近垂直 ---
instances, _ = pipe._stage2s_split(mask_open)
lbl_pairs = []
for lbl in range(1, n_ind + 1):
    ys, xs = np.nonzero(instances == lbl)
    cx = xs.mean()
    if 140 <= cx <= 250:
        lbl_pairs.append(lbl)
assert len(lbl_pairs) == 2, f"贴合对应拆成 2 个个体, 实际 {len(lbl_pairs)}"
for ax in pipe._stage3a_axis(instances):
    ang = abs(np.degrees(np.arctan2(ax["axis"][1], ax["axis"][0])))
    assert 60 <= ang <= 120, f"轴线应近垂直, 实际 {ang:.1f}°"
print(f"[2S split] 贴合对拆分为 {len(lbl_pairs)} 个体, 轴线全部近垂直 OK")

# --- 2S 退化: 单连通域图不触发 watershed ---
single = np.zeros((100, 100), np.uint8)
cv2.rectangle(single, (20, 10), (30, 90), 255, -1)
inst1, n1 = pipe._stage2s_split(single)
assert n1 == 1 and set(np.unique(inst1)) == {0, 1}
print("[2S degrade] 单根退化路径 OK")

print("SMOKE ALL PASSED")
