#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全图插桩 _stage2s_split 分水岭: 找出右胶囊泛洪失败的原因"""
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

fg = mo > 0
dist = cv2.distanceTransform(mo, cv2.DIST_L2, 5)
dist = cv2.GaussianBlur(dist, (15, 15), 0)
k = max(3, int(pipe.split_peak_k)) | 1
peak = (dist >= cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) - 0.75) \
    & (dist >= float(pipe.split_min_ridge))
n_raw, raw_lbl = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
peak_keep = np.zeros(peak.shape, np.uint8)
for lbl in range(1, n_raw):
    if int((raw_lbl == lbl).sum()) >= 20:
        peak_keep[raw_lbl == lbl] = 255
seeds = cv2.dilate(peak_keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
n_seeds, seed_lbl = cv2.connectedComponents(seeds, connectivity=8)
print("full-image seeds:", n_seeds - 1)
for lbl in range(1, n_seeds):
    ys, xs = np.nonzero(seed_lbl == lbl)
    print(f"  seed{lbl}: cx={xs.mean():.0f} cy={ys.mean():.0f} area={len(xs)}")

ws = np.zeros(mo.shape, np.int32)
ws[~fg] = 1
for lbl in range(1, n_seeds):
    ws[(seed_lbl == lbl) & fg] = lbl + 1
dmax = float(dist.max())
feed = np.zeros(dist.shape, np.uint8)
feed[fg] = ((1.0 - dist[fg] / dmax) * 255).astype(np.uint8)
cv2.watershed(cv2.cvtColor(feed, cv2.COLOR_GRAY2BGR), ws)
inst = np.where(fg, ws - 1, 0)
print("ws labels in pair region row y=250, x=185..245:", ws[250, 185:246])
print("feed row y=250, x=185..245:", feed[250, 185:246])
print("inst row y=250, x=150..250:", inst[250, 150:251])
# 桥行的 ws/feed
print("ws row y=250 x=186..210:", ws[250, 186:211])
print("feed row y=250 x=186..210:", feed[250, 186:210])
