#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""feed 变体对照: 找出让分水岭在桥鞍部正确切开的喂图方案"""
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
print("seeds:", n_seeds - 1)
dmax = float(dist.max())


def run_ws(name, feed):
    ws = np.zeros(mo.shape, np.int32)
    ws[~fg] = 1
    for lbl in range(1, n_seeds):
        ws[(seed_lbl == lbl) & fg] = lbl + 1
    cv2.watershed(cv2.cvtColor(feed.astype(np.uint8), cv2.COLOR_GRAY2BGR), ws)
    inst = np.where(fg, ws - 1, 0)
    row = inst[250, 150:250]
    # 统计 y=250 行的段
    segs = []
    cur, start = None, 0
    for i, v in enumerate(list(row) + [999]):
        if v != cur:
            if cur is not None:
                segs.append((cur, start + 150, i - 1 + 150))
            cur, start = v, i
    print(f"[{name}] y=250 row segments: {segs}")


feed_a = np.zeros(dist.shape, np.uint8)
feed_a[fg] = ((1.0 - dist[fg] / dmax) * 255).astype(np.uint8)
run_ws("A bg=0 (current)", feed_a)

feed_b = np.full(dist.shape, 255, np.uint8)
feed_b[fg] = ((1.0 - dist[fg] / dmax) * 255).astype(np.uint8)
run_ws("B bg=255 wall", feed_b)

# C: feed 用原始 dist (不模糊) 归一化
dist_raw = cv2.distanceTransform(mo, cv2.DIST_L2, 5)
feed_c = np.full(dist.shape, 255, np.uint8)
feed_c[fg] = ((1.0 - dist_raw[fg] / float(dist_raw.max())) * 255).astype(np.uint8)
run_ws("C raw dist, bg=255", feed_c)
