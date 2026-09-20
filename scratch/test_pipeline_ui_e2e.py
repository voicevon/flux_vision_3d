import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from tools.asparagus_offline import AsparagusOfflineApp
from src.vision.pipelines import PipelineRegistry

print("=== 开始多技术路线与自适应 GUI 端到端验证 ===")
app = AsparagusOfflineApp()
print(f"当前工位: {app.current_workspace_name}")
print(f"当前激活算法: {app.pipeline_key}")
assert app.pipeline_key == "ridge_tracing"

# 1. 验证路线 A 的动态步骤
steps_a = app.pipeline.get_steps()
step_names_a = [s.name for s in steps_a]
print("路线 A 动态步骤:", step_names_a)
assert step_names_a == ["1.前景", "距离场", "峰脊线", "2.骨架", "3.位姿"]

# 运行路线 A
app.run_analyze()
assert app.pipeline_result is not None
print(f"路线 A 解算耗时: {app.pipeline_result.elapsed_ms}ms, 检出目标: {len(app.targets)}")
assert len(app.targets) > 0

# 保存路线 A 的位姿与距离场截图
app._select_step("stage3_poses")
canvas_a3 = app.render()
cv2.imwrite("scratch/route_a_poses.png", canvas_a3)

app._select_step("stage1_dist")
canvas_a2 = app.render()
cv2.imwrite("scratch/route_a_dist.png", canvas_a2)

# 2. 切换至路线 B (双侧边缘拟合法)
print("--- 切换至路线 B: edge_centerline ---")
app.switch_pipeline("edge_centerline")
assert app.pipeline_key == "edge_centerline"

steps_b = app.pipeline.get_steps()
step_names_b = [s.name for s in steps_b]
print("路线 B 动态步骤:", step_names_b)
assert step_names_b == ["1.预处理", "梯度边缘", "双侧边界", "2.几何中线", "3.位姿"]

print(f"路线 B 解算耗时: {app.pipeline_result.elapsed_ms}ms, 检出目标: {len(app.targets)}")
assert len(app.targets) > 0

# 保存路线 B 的位姿与梯度边缘截图
app._select_step("stage5_poses")
canvas_b5 = app.render()
cv2.imwrite("scratch/route_b_poses.png", canvas_b5)

app._select_step("stage2_edges")
canvas_b2 = app.render()
cv2.imwrite("scratch/route_b_edges.png", canvas_b2)

print("=== 多技术路线动态自适应 GUI 验证全部通过 (ALL PASSED) ===")
