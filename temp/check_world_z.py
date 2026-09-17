# -*- coding: utf-8 -*-
"""诊断: 世界坐标系 Z 轴真实朝向 (用地图标靶局部轴反推)"""
import yaml
import numpy as np

d = yaml.safe_load(open(
    r'd:\Software\antigravity\flux_vision_3d\data\calibration_scenes\20260916_145246_Home_real\tags_map.yaml',
    encoding='utf-8'))
tags = d['tags']
for tid in (0, 1, 18, 23):
    t = tags[tid]
    T = np.array(t['transform_matrix'], dtype=float)
    R = T[:3, :3]
    print(f"Tag {tid}: pos={t['position_mm']} rpy={t['rpy_deg']}")
    print(f"   localX(右)={R[:, 0].round(3)}  localY(上)={R[:, 1].round(3)}  localZ(法向)={R[:, 2].round(3)}")

# 结论判定: 标靶贴墙直立时 localY 应接近世界 +Z (指向天)
t0 = np.array(tags[0]['transform_matrix'], dtype=float)[:3, :3]
up_component = float(t0[2, 1])  # localY 的世界 Z 分量
print(f"\nTag0 localY 的世界Z分量 = {up_component:+.3f}  "
      f"({'世界+Z≈天(上)' if up_component > 0.5 else '世界+Z≈地(下) ← 用户感觉反了' if up_component < -0.5 else '斜的'})")
