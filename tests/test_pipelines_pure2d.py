"""
纯 2D 感知流水线与叠压拓扑剥层自动化测试
=============================================
验证内容：
  1. PipelineRegistry 成功注册并枚举三条路线 (A, B1, B2)
  2. OcclusionPeeler 纯 2D 叠压交叉检测与循环剥层逻辑
  3. 各技术路线在 depth_mm=None 纯 2D 输入下的端到端执行与步骤快照生成
"""

import cv2
import numpy as np
from src.vision.pipelines import PipelineRegistry
from src.vision.pipelines.occlusion_peeler import CandidateSpine, OcclusionPeeler


def test_pipeline_registry_contains_three_routes():
    """验证注册中心包含 A, B1, B2 三条算法路线"""
    options = dict(PipelineRegistry.list_options())
    assert "ridge_tracing" in options, "缺少路线 A"
    assert "polarity_scanline" in options, "缺少路线 B1"
    assert "segment_topology" in options, "缺少路线 B2"
    assert len(options) >= 3


def test_occlusion_peeler_layer_separation():
    """验证 OcclusionPeeler 在两根交叉芦笋时的遮挡检测与分层剥离"""
    peeler = OcclusionPeeler(t_junction_radius=15.0)

    # 构造芦笋 A (横卧在上方, 中心 (500, 200), 长 200, 宽 20, 水平)
    box_a = np.array([
        [400, 190], [600, 190], [600, 210], [400, 210]
    ], dtype=np.int32)
    cand_a = CandidateSpine(
        id=1,
        center_px=(500.0, 200.0),
        length_px=200.0,
        diam_px=20.0,
        yaw_deg=0.0,
        axis_vector=(1.0, 0.0),
        box_corners=box_a
    )

    # 构造芦笋 B (斜穿被 A 压在下方, 中心 (500, 200), 长 180, 宽 20, 倾斜 30 度)
    rad = np.radians(30.0)
    vx, vy = float(np.cos(rad)), float(np.sin(rad))
    c_pt = np.array([500.0, 200.0])
    u_v = np.array([vx, vy])
    v_v = np.array([-vy, vx])
    p1 = c_pt - 90.0 * u_v - 10.0 * v_v
    p2 = c_pt + 90.0 * u_v - 10.0 * v_v
    p3 = c_pt + 90.0 * u_v + 10.0 * v_v
    p4 = c_pt - 90.0 * u_v + 10.0 * v_v
    box_b = np.array([p1, p2, p3, p4], dtype=np.int32)

    cand_b = CandidateSpine(
        id=2,
        center_px=(500.0, 200.0),
        length_px=180.0,
        diam_px=20.0,
        yaw_deg=30.0,
        axis_vector=(vx, vy),
        box_corners=box_b
    )

    # 1. 交叉检测
    is_overlap, cross_pt, area = peeler.detect_overlap_and_crossing(cand_a, cand_b)
    assert is_overlap is True, "未检测出两根交叉芦笋的重叠"
    assert cross_pt is not None, "未计算出轴线交叉点"

    # 2. 模拟边缘图：cand_a 的水平边缘连续穿过，cand_b 被切断
    edge_map = np.zeros((400, 800), dtype=np.uint8)
    # A 的连续双侧边界
    cv2.line(edge_map, (400, 190), (600, 190), 255, 2)
    cv2.line(edge_map, (400, 210), (600, 210), 255, 2)

    # 3. 剥层测试
    layers = peeler.peel_layers([cand_a, cand_b], edge_image=edge_map)
    assert len(layers) == 2, f"剥层数应为 2 层，实际: {len(layers)}"
    assert layers[0][0].id == 1, "最顶层 (Layer 0) 应当是连续穿过的芦笋 A"
    assert layers[1][0].id == 2, "次层 (Layer 1) 应当是被压住的芦笋 B"


def _create_synthetic_asparagus_image() -> np.ndarray:
    """生成带有黑色传送带和两根绿色横卧芦笋的合成图像"""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # 传送带区域背景稍亮暗灰色
    img[:, int(1280 * 0.35):int(1280 * 0.81)] = (25, 25, 25)

    # 绘制第一根芦笋 (绿色横条)
    cv2.rectangle(img, (520, 260), (920, 285), (35, 155, 55), -1)
    # 绘制第二根并排芦笋 (微有间隙)
    cv2.rectangle(img, (540, 305), (900, 328), (40, 160, 60), -1)

    return img


def test_polarity_scanline_pure2d_execution():
    """验证路线 B1 在纯 2D 图像输入下成功运行且不依赖深度"""
    pipeline = PipelineRegistry.create("polarity_scanline")
    assert pipeline is not None

    img = _create_synthetic_asparagus_image()
    res = pipeline.run(img, depth_mm=None)

    assert res is not None
    assert res.elapsed_ms > 0
    # 检查步骤快照
    steps = pipeline.get_steps()
    for s in steps:
        assert s.key in res.step_snapshots, f"缺少步骤快照: {s.key}"
        assert res.step_snapshots[s.key] is not None


def test_segment_topology_pure2d_execution():
    """验证路线 B2 在纯 2D 图像输入下成功运行且不依赖深度"""
    pipeline = PipelineRegistry.create("segment_topology")
    assert pipeline is not None

    img = _create_synthetic_asparagus_image()
    res = pipeline.run(img, depth_mm=None)

    assert res is not None
    assert res.elapsed_ms > 0
    steps = pipeline.get_steps()
    for s in steps:
        assert s.key in res.step_snapshots, f"缺少步骤快照: {s.key}"
        assert res.step_snapshots[s.key] is not None


if __name__ == "__main__":
    print("Running test_pipeline_registry_contains_three_routes...")
    test_pipeline_registry_contains_three_routes()
    print("Passed!")
    print("Running test_occlusion_peeler_layer_separation...")
    test_occlusion_peeler_layer_separation()
    print("Passed!")
    print("Running test_polarity_scanline_pure2d_execution...")
    test_polarity_scanline_pure2d_execution()
    print("Passed!")
    print("Running test_segment_topology_pure2d_execution...")
    test_segment_topology_pure2d_execution()
    print("Passed!")
    print("ALL PURE 2D PIPELINE TESTS PASSED!")
