#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试: 验证 F2 冯氏二代轮廓中心线三步流水线 (1.HSV绿分割 -> 2.绘制 -> 3.中心线提取)"""
import sys
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")

import numpy as np
import cv2

import src.vision.pipelines  # 触发全部注册
from src.vision.pipelines.registry import PipelineRegistry
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline
from src.vision.pipelines.f2_feng_green_axis_v2_pipeline import FengGreenAxisV2Pipeline

# 合成图 (640x480): 直条 + 弯曲半圆弧 + 贴合对黏连桥 (H 型)
color = np.zeros((480, 640, 3), dtype=np.uint8)
color[:] = (60, 60, 60)
green = (40, 160, 80)
cv2.rectangle(color, (60, 100), (100, 400), green, -1)            # 场景1: 直条 (40x300)
cv2.ellipse(color, (320, 250), (90, 90), 0, 180, 360, green, 24)  # 场景2: 弯曲半圆弧 (R90 厚24)
cv2.rectangle(color, (460, 100), (482, 400), green, -1)           # 场景3: 贴合对左条 (22x300)
cv2.rectangle(color, (512, 100), (534, 400), green, -1)           # 场景3: 贴合对右条 (22x300)
cv2.rectangle(color, (482, 246), (512, 254), green, -1)           # 场景3: 黏连桥 (H 横档 30x8)

# --- 注册与下拉顺序: F1 第 1 位, F2 第 2 位 ---
opts = PipelineRegistry.list_options()
keys = [k for k, _ in opts]
assert keys[0] == "feng_green_axis" and keys[1] == "feng_green_axis_v2", keys
print(f"[registry] 下拉顺序 OK: {[n for _, n in opts[:2]]}")

# --- 独立类: F2 直接继承 BaseAsparagusPipeline, 不再继承 F1 ---
pipe = FengGreenAxisV2Pipeline()
assert FengGreenAxisV2Pipeline.__bases__ == (BaseAsparagusPipeline,), FengGreenAxisV2Pipeline.__bases__
for m in ("_stage2a_morph", "_skeleton_groups", "_stage2_draw",
          "preview_stage1_hsv_mask", "preview_stage2_draw", "preview_stage3_centerline"):
    assert callable(getattr(pipe, m)), m
print("[inherit] F2 独立类 (直接继承 Base), 处理核齐备 OK")

# --- get_steps: 三步声明 (1.HSV绿分割 / 2.绘制 / 3.中心线提取) ---
steps = pipe.get_steps()
keys3 = [s.key for s in steps]
assert keys3 == ["stage1_hsv_mask", "stage2_draw", "stage3_centerline"], keys3
names3 = [s.name for s in steps]
assert names3 == ["1.HSV绿分割", "2.绘制", "3.中心线提取"], names3
print(f"[get_steps] 三步编号 OK: {names3}")

# --- STEP_SLIDERS: 步骤2 = 2A 形态学三滑条, 步骤3 = 最小线长 ---
assert set(pipe.STEP_SLIDERS.keys()) == {"stage1_hsv_mask", "stage2_draw", "stage3_centerline"}
attrs_2 = [sp["attr"] for sp in pipe.STEP_SLIDERS["stage2_draw"]]
assert attrs_2 == ["morph_close_k", "morph_ksize", "min_blob_area"], attrs_2
attrs_3 = [sp["attr"] for sp in pipe.STEP_SLIDERS["stage3_centerline"]]
assert attrs_3 == ["min_axis_len"], attrs_3
print(f"[STEP_SLIDERS] 步骤2={attrs_2} 步骤3={attrs_3} OK")

# --- run 全流程: 3 快照, targets 为空, 3 轮廓 4 中心线 (横档被 80px 门槛过滤) ---
result = pipe.run(color, None, nominal_z_mm=640.0)
assert set(result.step_snapshots.keys()) == set(keys3), result.step_snapshots.keys()
assert result.targets == []
em = result.extra_metrics
assert em["green_pixels"] > 0
assert em["contours"] == 3, em
assert em["centerlines"] == 4, em
print(f"[run] 3 快照齐全, targets={len(result.targets)}, green_px={em['green_pixels']}, "
      f"contours={em['contours']}, centerlines={em['centerlines']}")

# --- 中心线几何性质: 3 条近垂直 (直条 + H 两竖) + 1 条弯曲弧, 弧长降序 ---
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (pipe.h_low, pipe.s_min, pipe.v_min),
                         (pipe.h_high, pipe.s_high, pipe.v_high))
mask_clean, contours, _ = pipe._stage2_draw(green_mask)
lines = pipe._stage_centerlines(mask_clean)
arcs = [ln["len_px"] for ln in lines]
assert arcs == sorted(arcs, reverse=True), arcs
n_vert = sum(1 for ln in lines
             if abs(ln["p2"][0] - ln["p1"][0]) <= 25 and abs(ln["p2"][1] - ln["p1"][1]) >= 200)
assert n_vert == 3, [(ln["p1"], ln["p2"], ln["len_px"]) for ln in lines]
# 弯曲弧: 弧长 > 弦长 + 40 且最大偏弦距 > 60 (沿弯曲中心行走而非直线)
for ln in lines:
    p1, p2 = np.array(ln["p1"]), np.array(ln["p2"])
    chord = float(np.linalg.norm(p2 - p1))
    pts = np.array(ln["path"])
    dev = np.abs((p2[0] - p1[0]) * (pts[:, 1] - p1[1])
                 - (p2[1] - p1[1]) * (pts[:, 0] - p1[0])) / max(chord, 1e-6)
    if ln["len_px"] > chord + 40.0:
        assert dev.max() > 60, (ln["len_px"], chord, dev.max())
        print(f"[bend] 弯曲弧 OK: arc={ln['len_px']:.0f} chord={chord:.0f} dev={dev.max():.0f}")
        break
else:
    raise AssertionError("未找到弯曲弧中心线")
print(f"[geometry] 近垂直中心线 {n_vert}/4, 弧长降序 {['%.0f' % a for a in arcs]} OK")

# --- 调参预览方法不崩溃 ---
p2v = pipe.preview_stage2_draw(color)
p3v = pipe.preview_stage3_centerline(color)
assert p2v.shape == color.shape and p3v.shape == color.shape
print("[preview] 2 绘制 / 3 中心线 实时预览 OK")

# --- 弧长门槛过滤: min_axis_len=0 时 H 横档中心线出现 (5 条), =80 过滤回 4 条 ---
pipe.min_axis_len = 0.0
lines_all = pipe._stage_centerlines(mask_clean)
assert len(lines_all) == 5, len(lines_all)
pipe.min_axis_len = 80.0
print(f"[tuning] min_axis_len=0 -> {len(lines_all)} 条 (含 H 横档), =80 过滤为 4 条 OK")

print("SMOKE ALL PASSED")
