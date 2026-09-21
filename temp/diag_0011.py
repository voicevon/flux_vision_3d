#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐层诊断 0011: 1A 像素 -> 2A 连通域 -> 2S 种子簇 -> instances"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline

color = cv2.imread(r"d:\Software\antigravity\flux_vision_3d\data\workspaces\20260916_145246_Home_real\production\raw_images\view_0011.png")
pipe = FengGreenAxisPipeline()

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
gm = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                 (pipe.h_high, pipe.s_high, pipe.v_high))
print(f"1A green px = {cv2.countNonZero(gm)}")

mo, removed = pipe._stage2a_morph(gm)
n2a, lab2a, st2a, _ = cv2.connectedComponentsWithStats(mo, connectivity=8)
areas = sorted((int(st2a[i, cv2.CC_STAT_AREA]) for i in range(1, n2a)), reverse=True)
print(f"2A domains = {n2a - 1} (removed {removed}), areas top20 = {areas[:20]}")
big = [i for i in range(1, n2a) if st2a[i, cv2.CC_STAT_AREA] >= 500]
print(f"domains >= 500px: {len(big)}")
for i in big[:10]:
    x, y, w2, h2, a = st2a[i]
    print(f"  dom{i}: rect=({x},{y},{w2}x{h2}) area={a} aspect={max(w2,h2)/max(1,min(w2,h2)):.1f}")

# 2S 内部: 平滑 dist 后峰簇数
dist = cv2.distanceTransform(mo, cv2.DIST_L2, 5)
dist = cv2.GaussianBlur(dist, (15, 15), 0)
k = 7
peak = (cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) == dist) & (dist >= 3.0)
n_raw, raw_lbl = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
sizes = [(int((raw_lbl == i).sum()), i) for i in range(1, n_raw)]
sizes.sort(reverse=True)
print(f"2S raw peak clusters = {n_raw - 1}, sizes top15 = {s for s, _ in sizes[:15]}" if False else
      f"2S raw peak clusters = {n_raw - 1}, sizes top15 = {[s for s, _ in sizes[:15]]}")
kept = [s for s, _ in sizes if s >= 20]
print(f"clusters kept (>=20px): {len(kept)}")
