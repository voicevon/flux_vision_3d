# -*- coding: utf-8 -*-
"""冒烟测试: robot_online_tracker 单帧识别叠加渲染 (不实际开相机)"""
import sys
import numpy as np
import cv2

sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d")
sys.path.insert(0, r"d:\Software\antigravity\flux_vision_3d\tools\calibration")

from robot_online_tracker import RobotOnlineTracker  # noqa: E402

app = RobotOnlineTracker()
print(f"[OK] 实例化成功 | 地图锚点: {sorted(app.anchor_positions)} | 理论Tag2: {app.theoretical is not None}")

# 模拟单帧识别结果: 伪造 Tag 0/1 检测角点 + 一个合理的相机位姿 (平移沿Z)
app.static_frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
det = {
    0: np.array([[300, 200], [380, 200], [380, 280], [300, 280]], dtype=np.float32),
    1: np.array([[700, 260], [780, 260], [780, 340], [700, 340]], dtype=np.float32),
}
app.static_det = det
app.world_locked = True
app.locked_rvec = np.array([0.6, 0.0, 0.0], dtype=np.float64)  # 前倾看地面 (旋转向量)
app.locked_tvec = np.array([[0.0], [200.0], [900.0]])  # 相机在世界系前方 900mm
app.lock_info = "单帧识别 | RMSE 0.42px | 支撑 [0, 1]"

canvas = app.static_frame.copy()
app._handle_action("DD_PLANE_7", 150)  # 模拟下拉框选择 Z=150 (索引7: 不绘制,405,196,350,300,250,200,150)
print(f"[OK] 下拉选择后 plane_z = {app.plane_z} | show_xy_plane_on = {app.show_xy_plane_on}")
app._draw_xy_plane_overlay(canvas, None)
app._handle_action("DD_PLANE_2", 196)  # Tag 1 高度平面 (触发等高红色 X 轴辅助线)
app._draw_xy_plane_overlay(canvas, None)
print(f"[OK] Z=196 平面渲染完成 (Tag 1 辅助红线) | plane_z = {app.plane_z}")
app._handle_action("DD_PLANE_0", None)  # 选择"不绘制 XY 平面"
print(f"[OK] 选择'不绘制'后 show_xy_plane_on = {app.show_xy_plane_on}")
app._handle_action("DD_PLANE_7", 150)  # 恢复 Z=150 用于最终渲染
app._draw_recognition_overlay(canvas)
full = app._compose_canvas(canvas)
app._draw_info_panel(full, y_off=app.TOOLBAR_H if hasattr(app, "TOOLBAR_H") else 44)
app.active_dropdown = "PLANE_DROPDOWN"  # 验证下拉浮层渲染
app._draw_toolbar(full)

out = r"d:\Software\antigravity\flux_vision_3d\temp\recog_overlay_smoke.png"
cv2.imwrite(out, full)
n_green = sum(1 for t in app.anchor_positions if app.engine.get_tag_world_corners(t) is not None)
print(f"[OK] 渲染成功 -> {out} | 绿理论Tag候选 {n_green} 枚 | 蓝实测 {len(det)} 枚")
