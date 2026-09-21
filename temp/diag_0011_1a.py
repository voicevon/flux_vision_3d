#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 1A 层: 开运算前的掩膜连通域结构 + 不同开运算核的对比"""
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

n1a, lab1a, st1a, _ = cv2.connectedComponentsWithStats(gm, connectivity=8)
areas = sorted((int(st1a[i, cv2.CC_STAT_AREA]) for i in range(1, n1a)), reverse=True)
print(f"1A raw domains = {n1a - 1}, total px = {cv2.countNonZero(gm)}")
print(f"1A areas top15 = {areas[:15]}")
big = [(int(st1a[i, cv2.CC_STAT_AREA]), int(st1a[i, cv2.CC_STAT_WIDTH]), int(st1a[i, cv2.CC_STAT_HEIGHT]))
       for i in range(1, n1a) if st1a[i, cv2.CC_STAT_AREA] >= 800]
big.sort(reverse=True)
print(f"1A domains >= 800px: {len(big)} -> (area, w, h): {big[:10]}")

# 不同开运算核下的大域数量对比
for ks in (3, 5, 7):
    k = ks | 1
    mo = cv2.morphologyEx(gm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    nk, labk, stk, _ = cv2.connectedComponentsWithStats(mo, connectivity=8)
    bigk = sum(1 for i in range(1, nk) if stk[i, cv2.CC_STAT_AREA] >= 800)
    top = sorted((int(stk[i, cv2.CC_STAT_AREA]) for i in range(1, nk)), reverse=True)[:6]
    print(f"OPEN k={k}: domains={nk - 1}, >=800px: {bigk}, top areas={top}")

# 闭运算优先 (先弥合断裂再去散点): CLOSE 后再 OPEN
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
co = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, k3)
mo2 = cv2.morphologyEx(co, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
nc, _, stc, _ = cv2.connectedComponentsWithStats(mo2, connectivity=8)
bigc = sum(1 for i in range(1, nc) if stc[i, cv2.CC_STAT_AREA] >= 800)
topc = sorted((int(stc[i, cv2.CC_STAT_AREA]) for i in range(1, nc)), reverse=True)[:6]
print(f"CLOSE7->OPEN5: domains={nc - 1}, >=800px: {bigc}, top areas={topc}")
