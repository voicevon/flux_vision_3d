#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 2S 合成例: 贴合对(13px缝+9px桥) 为何分出过多个体"""
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
inst, n = pipe._stage2s_split(mo)
print("individuals:", n)
for lbl in range(1, n + 1):
    ys, xs = np.nonzero(inst == lbl)
    print(f"  inst{lbl}: area={len(xs)}, cx={xs.mean():.0f}, cy={ys.mean():.0f}, "
          f"x[{xs.min()}..{xs.max()}] y[{ys.min()}..{ys.max()}]")

# 手动复现 2S 内部: 看峰簇
fg = mo > 0
dist = cv2.distanceTransform(mo, cv2.DIST_L2, 5)
dist = cv2.GaussianBlur(dist, (15, 15), 0)
k = max(3, int(pipe.split_peak_k)) | 1
peak = (cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) == dist) \
    & (dist >= float(pipe.split_min_ridge))
n_raw, raw_lbl = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
print("raw peak clusters:", n_raw - 1)
for lbl in range(1, n_raw):
    m = raw_lbl == lbl
    if int(m.sum()) >= 20:
        ys, xs = np.nonzero(m)
        print(f"  cluster{lbl}: area={int(m.sum())} cx={xs.mean():.0f} cy={ys.mean():.0f} "
              f"maxdist={dist[m].max():.1f}")
    else:
        print(f"  cluster{lbl}: area={int(m.sum())} (filtered)")
