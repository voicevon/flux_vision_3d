import cv2
from tools.asparagus_offline import AsparagusOfflineApp

app = AsparagusOfflineApp()
# 执行识别解算 (一次性全阶段解算)
app.run_analyze()

# 验证当前右侧面板结果已就绪
assert len(app.targets) > 0, "解算结果为空!"
print(f"Top 3 目标数量: {len(app.targets)}")
for t in app.targets:
    print(f"  #{t.id}: D={t.diam_mm}mm, L={t.length_mm}mm, R={t.yaw_deg}deg, H={t.rel_height_mm}mm")

# 1. 切换至 1.前景
app._select_view_mode(0)
canvas_1 = app.render()
cv2.imwrite("scratch/gui_view1_foreground.png", canvas_1)
assert app.vis_img is app.stage_vis[0]

# 2. 切换至 2.骨架
app._select_view_mode(1)
canvas_2 = app.render()
cv2.imwrite("scratch/gui_view2_spines.png", canvas_2)
assert app.vis_img is app.stage_vis[1]

# 3. 切换至 3.位姿
app._select_view_mode(2)
canvas_3 = app.render()
cv2.imwrite("scratch/gui_view3_poses.png", canvas_3)
assert app.vis_img is app.stage_vis[2]

# 验证在各个视图下右侧 targets 数据均保持不变且有效
assert len(app.targets) > 0

print("所有视图切换与独立显示测试全部通过！三张界面图已保存至 scratch/")
