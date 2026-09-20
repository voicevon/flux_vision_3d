import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from tools.asparagus_offline import AsparagusOfflineApp
from src.vision.pipelines import PipelineRegistry

print("=== 开始三技术路线 (A, B1, B2) 与自适应 GUI 端到端验证 ===")
app = AsparagusOfflineApp()
print(f"当前工位: {app.current_workspace_name}")
print(f"所有已注册路线: {PipelineRegistry.list_options()}")

# 1. 验证路线 A: 距离场脊线法
print("\n--- 1. 测试路线 A: ridge_tracing ---")
assert app.pipeline_key == "ridge_tracing"
steps_a = [s.name for s in app.pipeline.get_steps()]
print("路线 A 动态步骤:", steps_a)
assert steps_a == ["1.前景", "距离场", "峰脊线", "2.骨架", "3.位姿"]

app.run_analyze()
assert app.pipeline_result is not None
print(f"路线 A 解算耗时: {app.pipeline_result.elapsed_ms}ms, 检出目标: {len(app.targets)}")
assert len(app.targets) > 0

app._select_step("stage3_poses")
canvas_a = app.render()
cv2.imwrite("scratch/route_a_poses.png", canvas_a)

# 2. 切换至路线 B1: 极性扫描法 (Polarity Scanline)
print("\n--- 2. 测试路线 B1: polarity_scanline ---")
app.switch_pipeline("polarity_scanline")
assert app.pipeline_key == "polarity_scanline"
steps_b1 = [s.name for s in app.pipeline.get_steps()]
print("路线 B1 动态步骤:", steps_b1)
assert steps_b1 == ["1.预处理", "梯度极性", "极性配对", "2.主干拟合", "3.顶层位姿"]

print(f"路线 B1 解算耗时: {app.pipeline_result.elapsed_ms}ms, 检出目标: {len(app.targets)}")
assert len(app.targets) > 0

# 保存路线 B1 极性配对与顶层位姿快照
app._select_step("stage3_scanline")
canvas_b1_pairs = app.render()
cv2.imwrite("scratch/route_b1_scanline.png", canvas_b1_pairs)

app._select_step("stage5_top_poses")
canvas_b1_poses = app.render()
cv2.imwrite("scratch/route_b1_poses.png", canvas_b1_poses)

# 3. 切换至路线 B2: 分线段提取法 (Segment Topology)
print("\n--- 3. 测试路线 B2: segment_topology ---")
app.switch_pipeline("segment_topology")
assert app.pipeline_key == "segment_topology"
steps_b2 = [s.name for s in app.pipeline.get_steps()]
print("路线 B2 动态步骤:", steps_b2)
assert steps_b2 == ["1.预处理", "线段提取", "双轨配对", "2.中线拓扑", "3.顶层位姿"]

print(f"路线 B2 解算耗时: {app.pipeline_result.elapsed_ms}ms, 检出目标: {len(app.targets)}")
assert len(app.targets) > 0

app._select_step("stage3_pairing")
canvas_b2_pairs = app.render()
cv2.imwrite("scratch/route_b2_pairing.png", canvas_b2_pairs)

app._select_step("stage5_poses")
canvas_b2_poses = app.render()
cv2.imwrite("scratch/route_b2_poses.png", canvas_b2_poses)

print("\n=== 三技术路线 (A, B1, B2) 端到端 GUI 切换与解算全部通过 (ALL PASSED) ===")
