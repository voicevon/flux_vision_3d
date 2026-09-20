import os
import cv2
from tools.asparagus_offline import AsparagusOfflineApp

app = AsparagusOfflineApp()
app.run_analyze()

# 保存三阶段调试图
cv2.imwrite("scratch/stage1_foreground.png", app.stage_vis[0])
cv2.imwrite("scratch/stage2_spines.png", app.stage_vis[1])
cv2.imwrite("scratch/stage3_poses.png", app.stage_vis[2])

print("保存三阶段效果图成功:")
print("1. scratch/stage1_foreground.png")
print("2. scratch/stage2_spines.png")
print("3. scratch/stage3_poses.png")
