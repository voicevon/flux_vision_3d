#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桥附着区 dist/peak 数值检查"""
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

dist = cv2.distanceTransform(mo, cv2.DIST_L2, 5)
distb = cv2.GaussianBlur(dist, (15, 15), 0)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
peak = (cv2.dilate(distb, k) == distb) & (distb >= 3)

np.set_printoptions(precision=1, suppress=True, linewidth=260)
print("dist blurred (x=166..176, y=238..262):")
print(distb[238:263, 166:177])
print("peak:")
print(peak[238:263, 166:177].astype(int))
# 桥区中心行的 dist 原始值
print("raw dist row y=250, x=185..205:", np.round(dist[250, 185:206], 1))
print("blurred    row y=250, x=185..205:", np.round(distb[250, 185:206], 1))
