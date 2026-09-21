#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实样本 0011 验证: 2S 平滑过滤 + 3A 逐根递归拟合"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

from src.vision.pipelines.f1_feng_green_axis_pipeline import FengGreenAxisPipeline

color = cv2.imread(r"d:\Software\antigravity\flux_vision_3d\data\workspaces\20260916_145246_Home_real\production\raw_images\view_0011.png")
print("image:", color.shape)

pipe = FengGreenAxisPipeline()
result = pipe.run(color, None, nominal_z_mm=640.0)
em = result.extra_metrics
print(f"individuals(2S)={em['individuals']}  axes(3A)={em['axes_early']}  "
      f"boxes(5)={em['boxes_found']}  stems(7)={em['stems_confirmed']}  targets={len(result.targets)}")

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
gm = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                 (pipe.h_high, pipe.s_high, pipe.v_high))
mo, _ = pipe._stage2a_morph(gm)
inst, n = pipe._stage2s_split(mo)
axes = pipe._stage3a_axis(inst)
lens = [round(a["len_px"]) for a in axes]
print("axis lengths desc:", lens[:25], "..." if len(lens) > 25 else "")
print(f"axes >= 100px: {sum(1 for L in lens if L >= 100)} / total {len(lens)}")

# 逐实例骨架验证: 任取最大实例, 轴数应 <= 3
big = max(range(1, n + 1), key=lambda L: int((inst == L).sum()))
m = (inst == big).astype(np.uint8)
ys, xs = np.nonzero(m)
y0, x0 = int(ys.min()), int(xs.min())
crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
segs = pipe._instance_axes(crop, x0, y0, int(m.sum()))
print(f"largest instance ({int(m.sum())}px) -> {len(segs)} axis segment(s)")
