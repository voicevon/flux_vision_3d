#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""插桩复现 _stage2s_split: 打印种子簇与分水岭分界位置"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2
from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline

pipe = FengGreenAxisPipeline()
color = np.zeros((480, 640, 3), dtype=np.uint8)
color[:] = (60, 60, 60)
green = (40, 160, 80)
for cx in (170, 224):
    cv2.rectangle(color, (cx - 20, 100), (cx + 20, 400), green, -1)
cv2.rectangle(color, (193, 245), (201, 255), green, -1)
cv2.rectangle(color, (410, 100), (450, 400), green, -1)
cv2.circle(color, (430, 100), 21, green, -1)
cv2.circle(color, (430, 400), 21, green, -1)
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
gm = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                 (pipe.h_high, pipe.s_high, pipe.v_high))
mo, _ = pipe._stage2a_morph(gm)

# ---- 只看贴合对区域 x 140..260 ----
sub = mo[90:410, 140:260].copy()
fg = sub > 0
dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
dist = cv2.GaussianBlur(dist, (15, 15), 0)
k = max(3, int(pipe.split_peak_k)) | 1
peak = (dist >= cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) - 0.75) \
    & (dist >= float(pipe.split_min_ridge))
n_raw, raw_lbl = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
print("peak clusters (sub):", n_raw - 1)
for lbl in range(1, n_raw):
    m = raw_lbl == lbl
    a = int(m.sum())
    if a >= 20:
        ys, xs = np.nonzero(m)
        print(f"  keep c{lbl}: area={a} cx={xs.mean()+140:.0f} cy={ys.mean()+90:.0f} maxd={dist[m].max():.1f}")
    else:
        print(f"  filt c{lbl}: area={a}")

peak_keep = np.zeros(peak.shape, np.uint8)
for lbl in range(1, n_raw):
    if int((raw_lbl == lbl).sum()) >= 20:
        peak_keep[raw_lbl == lbl] = 255
seeds = cv2.dilate(peak_keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
n_seeds, seed_lbl = cv2.connectedComponents(seeds, connectivity=8)
print("seeds:", n_seeds - 1)
for lbl in range(1, n_seeds):
    ys, xs = np.nonzero(seed_lbl == lbl)
    print(f"  seed{lbl}: area={int(len(xs))} cx={xs.mean()+140:.0f} cy={ys.mean()+90:.0f}")
