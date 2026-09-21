#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: 合成冒烟场景 2S 内部插桩 —— 种子簇分布 / 泛洪区域 / 鞍深度量"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline

# 与 smoke_f1_2s_split.py 完全一致的场景
color = np.zeros((480, 640, 3), dtype=np.uint8)
color[:] = (60, 60, 60)
green = (40, 160, 80)
for cx in (170, 224):
    cv2.rectangle(color, (cx - 20, 100), (cx + 20, 400), green, -1)
cv2.rectangle(color, (193, 245), (201, 255), green, -1)
cv2.rectangle(color, (410, 100), (450, 400), green, -1)
cv2.circle(color, (430, 100), 21, green, -1)
cv2.circle(color, (430, 400), 21, green, -1)

pipe = FengGreenAxisPipeline()
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
gm = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                 (pipe.h_high, pipe.s_high, pipe.v_high))
mask_open, _ = pipe._stage2a_morph(gm)
print("2A cc:", cv2.connectedComponents(mask_open, connectivity=8)[0] - 1)

fg = mask_open > 0
dist = cv2.distanceTransform(mask_open, cv2.DIST_L2, 5)
dist = cv2.GaussianBlur(dist, (15, 15), 0)
k = max(3, int(pipe.split_peak_k)) | 1
peak = (cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) == dist) \
    & (dist >= float(pipe.split_min_ridge))
n_raw, raw_lbl = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
print(f"peak raw clusters: {n_raw - 1}")
peak_keep = np.zeros(peak.shape, np.uint8)
for lbl in range(1, n_raw):
    area = int((raw_lbl == lbl).sum())
    if area >= 20:
        peak_keep[raw_lbl == lbl] = 255
    else:
        ys, xs = np.nonzero(raw_lbl == lbl)
        print(f"  dropped cluster: area={area} bbox=x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
n_seeds, seed_lbl = cv2.connectedComponents(peak_keep, cv2.CV_32S) if False else cv2.connectedComponents(peak_keep, connectivity=8)
print(f"kept seed clusters: {n_seeds - 1}")
for lbl in range(1, n_seeds):
    ys, xs = np.nonzero(seed_lbl == lbl)
    print(f"  seed {lbl}: area={len(ys)} bbox=x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] "
          f"maxdist={dist[seed_lbl == lbl].max():.2f}")

dmax = float(dist.max())
feed = np.full(dist.shape, 255.0, np.float32)
feed[fg] = (1.0 - dist[fg] / dmax) * 255.0
zones = pipe._flood_minimax(feed, seed_lbl, n_seeds - 1, fg)
print(f"zones after flood: {int(zones.max())}")
for lbl in range(1, int(zones.max()) + 1):
    ys, xs = np.nonzero(zones == lbl)
    print(f"  zone {lbl}: area={len(ys)} bbox=x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] "
          f"maxdist={dist[zones == lbl].max():.2f}")

# 鞍部度量插值: 每对相邻区域的接触边 min feed 与两侧区域脊顶
saddle_feed = {}
for a, b, f in ((zones[:-1, :], zones[1:, :], feed[:-1, :]),
                (zones[:, :-1], zones[:, 1:], feed[:, :-1])):
    m = (a > 0) & (b > 0) & (a != b)
    if not m.any():
        continue
    lo = np.minimum(a, b)[m].astype(np.int64)
    hi = np.maximum(a, b)[m].astype(np.int64)
    fv = f[m]
    order = np.lexsort((fv, hi, lo))
    lo_s, hi_s, fv_s = lo[order], hi[order], fv[order]
    first = np.ones(len(lo_s), bool)
    first[1:] = (lo_s[1:] != lo_s[:-1]) | (hi_s[1:] != hi_s[:-1])
    for L, H, F in zip(lo_s[first].tolist(), hi_s[first].tolist(), fv_s[first].tolist()):
        saddle_feed[(L, H)] = F
zone_peak = {lbl: float(dist[zones == lbl].max()) for lbl in range(1, int(zones.max()) + 1)}
print(f"dmax={dmax:.2f}  contact pairs:")
for (za, zb), fmin in saddle_feed.items():
    saddle_dist = (1.0 - fmin / 255.0) * dmax
    depth = min(zone_peak[za], zone_peak[zb]) - saddle_dist
    print(f"  {za}<->{zb}: saddle_dist={saddle_dist:.2f} peaks=({zone_peak[za]:.2f},{zone_peak[zb]:.2f}) "
          f"depth={depth:.2f} -> {'MERGE' if depth < pipe.split_merge_saddle else 'keep'}")

zones_m = pipe._merge_shallow_saddles(zones.copy(), feed, dmax, float(pipe.split_merge_saddle))
print(f"zones after merge: {int(zones_m.max())}")

# 探针: 黏连桥行附近的模糊 dist 剖面 (行 y=235..265, 列 x=165..200)
print("\nblur-dist profile (rows 235..265 step5, cols 165..200 step 3):")
for y in range(235, 266, 5):
    row = " ".join(f"{dist[y, x]:5.2f}" for x in range(165, 201, 3))
    print(f"  y={y}: {row}")
print("\nraw-dist profile (before blur) same grid:")
dist_raw = cv2.distanceTransform(mask_open, cv2.DIST_L2, 5)
for y in range(235, 266, 5):
    row = " ".join(f"{dist_raw[y, x]:5.2f}" for x in range(165, 201, 3))
    print(f"  y={y}: {row}")
print("\nmask rows 240..260, cols 186..206 (1=fg):")
for y in range(240, 261, 2):
    row = "".join("#" if mask_open[y, x] else "." for x in range(186, 207))
    print(f"  y={y}: {row}")
