#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: (1) 合成弯曲条骨架是否单轴 (2) 0011 最大实例骨架轴数可视化 (3) 全图实例轴数分布"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline

pipe = FengGreenAxisPipeline()

# ---------- (1) 合成弯曲条: 弧长 ~300px, 宽 14px, 轻微弯曲 ----------
img = np.zeros((200, 400), np.uint8)
ts = np.linspace(0, np.pi, 300)
cx = 60 + 120 * ts / np.pi
cy = 100 - 55 * np.sin(ts)
for x, y in zip(cx, cy):
    cv2.circle(img, (int(x), int(y)), 7, 255, -1)
ys, xs = np.nonzero(img)
crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
segs = pipe._instance_axes(crop, int(xs.min()), int(ys.min()), int(img.sum()))
print(f"[synthetic curved bar] axes: {len(segs)}, lens: {[round(s['len_px']) for s in segs]}")

# ---------- (2) 0011 最大实例骨架轴可视化 ----------
color = cv2.imread(r"d:\Software\antigravity\flux_vision_3d\data\workspaces\20260916_145246_Home_real\production\raw_images\view_0011.png")
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
gm = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                 (pipe.h_high, pipe.s_high, pipe.v_high))
mo, _ = pipe._stage2a_morph(gm)
inst, n = pipe._stage2s_split(mo)
big = max(range(1, n + 1), key=lambda L: int((inst == L).sum()))
m = (inst == big).astype(np.uint8)
ys, xs = np.nonzero(m)
y0b, x0b = int(ys.min()), int(xs.min())
crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
segs = pipe._instance_axes(crop, x0b, y0b, int(m.sum()))

pad = 20
crop_vis = color[max(0, y0b - pad):ys.max() + 1 + pad, max(0, x0b - pad):xs.max() + 1 + pad].copy()
sub = crop_vis[max(0, y0b - pad) - max(0, y0b - pad):, :]
for i, s in enumerate(segs):
    p1 = (int(s["p1"][0]) - max(0, x0b - pad), int(s["p1"][1]) - max(0, y0b - pad))
    p2 = (int(s["p2"][0]) - max(0, x0b - pad), int(s["p2"][1]) - max(0, y0b - pad))
    cv2.line(crop_vis, p1, p2, (255, 255, 255), 2)
    print(f"  axis{i + 1}: len={s['len_px']:.0f} area={s['area_px']}")
cv2.imwrite(r"d:\Software\antigravity\flux_vision_3d\temp\out_diag_big.png", crop_vis)
print("saved temp/out_diag_big.png")

# ---------- (3) 全图各实例轴数分布 ----------
multi = []
for lbl in range(1, n + 1):
    mm = (inst == lbl).astype(np.uint8)
    if int(mm.sum()) < 30:
        continue
    ys2, xs2 = np.nonzero(mm)
    cr = mm[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
    ss = pipe._instance_axes(cr, int(xs2.min()), int(ys2.min()), int(mm.sum()))
    if len(ss) >= 2:
        multi.append((lbl, len(ss), int(mm.sum())))
multi.sort(key=lambda t: -t[2])
print(f"instances with >=2 axes: {len(multi)}")
for lbl, ns, area in multi[:12]:
    print(f"  inst{lbl}: {ns} axes, {area}px")
