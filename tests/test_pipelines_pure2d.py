"""
纯 2D 感知流水线与叠压拓扑剥层自动化测试
=============================================
验证内容：
  1. PipelineRegistry 成功注册并枚举三条路线 (A, B1, B2)
  2. OcclusionPeeler 纯 2D 叠压交叉检测与循环剥层逻辑
  3. 各技术路线在 depth_mm=None 纯 2D 输入下的端到端执行与步骤快照生成
"""

import os
import sys
import unittest
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.pipelines import PipelineRegistry
from src.vision.pipelines.occlusion_peeler import CandidateSpine, OcclusionPeeler


class TestPipelinesPure2D(unittest.TestCase):

    def test_pipeline_registry_contains_all_algorithms(self):
        """验证注册中心包含 A, B1, B2 以及规划中的 C1, C2, Y1 算法路线"""
        options = dict(PipelineRegistry.list_options())
        self.assertIn("ridge_tracing", options, "缺少算法 A")
        self.assertIn("polarity_scanline", options, "缺少算法 B1")
        self.assertIn("segment_topology", options, "缺少算法 B2")
        self.assertIn("skeleton_thinning", options, "缺少算法 C1")
        self.assertIn("frangi_vesselness", options, "缺少算法 C2")
        self.assertIn("yolo_deeplearning", options, "缺少算法 Y1")
        self.assertGreaterEqual(len(options), 6)


    def test_occlusion_peeler_layer_separation(self):
        """验证 OcclusionPeeler 在两根交叉芦笋时的遮挡检测与分层剥离"""
        peeler = OcclusionPeeler(t_junction_radius=15.0)

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
        self.assertTrue(is_overlap)
        self.assertIsNotNone(cross_pt)

        # 2. 模拟边缘图：cand_a 的水平边缘连续穿过，cand_b 被切断
        edge_map = np.zeros((400, 800), dtype=np.uint8)
        cv2.line(edge_map, (400, 190), (600, 190), 255, 2)
        cv2.line(edge_map, (400, 210), (600, 210), 255, 2)

        # 3. 剥层测试
        layers = peeler.peel_layers([cand_a, cand_b], edge_image=edge_map)
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0][0].id, 1)
        self.assertEqual(layers[1][0].id, 2)

    def test_polarity_scanline_pure2d_execution(self):
        """验证算法 B1 在纯 2D 图像输入下成功运行且所有细分步骤完整生成快照"""
        pipeline = PipelineRegistry.create("polarity_scanline")
        self.assertIsNotNone(pipeline)

        img = _create_synthetic_asparagus_image()
        res = pipeline.run(img, depth_mm=None)

        self.assertIsNotNone(res)
        self.assertGreater(res.elapsed_ms, 0)
        # 检查 8 个细化步骤快照全部存在
        steps = pipeline.get_steps()
        self.assertEqual(len(steps), 8)
        for s in steps:
            self.assertIn(s.key, res.step_snapshots, f"缺少步骤快照: {s.key}")
            self.assertIsNotNone(res.step_snapshots[s.key])

    def test_segment_topology_pure2d_execution(self):
        """验证算法 B2 在纯 2D 图像输入下成功运行且不依赖深度"""
        pipeline = PipelineRegistry.create("segment_topology")
        self.assertIsNotNone(pipeline)

        img = _create_synthetic_asparagus_image()
        res = pipeline.run(img, depth_mm=None)

        self.assertIsNotNone(res)
        self.assertGreater(res.elapsed_ms, 0)
        steps = pipeline.get_steps()
        for s in steps:
            self.assertIn(s.key, res.step_snapshots, f"缺少步骤快照: {s.key}")
            self.assertIsNotNone(res.step_snapshots[s.key])

    def test_skeleton_thinning_pure2d_execution(self):
        """验证算法 C1 (形态学骨架细化法) 在纯 2D 图像输入下成功运行且各步骤完整输出快照"""
        pipeline = PipelineRegistry.create("skeleton_thinning")
        self.assertIsNotNone(pipeline)

        img = _create_synthetic_asparagus_image()
        res = pipeline.run(img, depth_mm=None)

        self.assertIsNotNone(res)
        self.assertGreater(res.elapsed_ms, 0)
        steps = pipeline.get_steps()
        self.assertEqual(len(steps), 6)
        for s in steps:
            self.assertIn(s.key, res.step_snapshots, f"缺少步骤快照: {s.key}")
            self.assertIsNotNone(res.step_snapshots[s.key])

    def test_frangi_vesselness_pure2d_execution(self):
        """验证算法 C2 (Frangi管状滤波法) 在纯 2D 图像输入下成功运行且各步骤完整输出快照"""
        pipeline = PipelineRegistry.create("frangi_vesselness")
        self.assertIsNotNone(pipeline)

        img = _create_synthetic_asparagus_image()
        res = pipeline.run(img, depth_mm=None)

        self.assertIsNotNone(res)
        self.assertGreater(res.elapsed_ms, 0)
        steps = pipeline.get_steps()
        self.assertEqual(len(steps), 6)
        for s in steps:
            self.assertIn(s.key, res.step_snapshots, f"缺少步骤快照: {s.key}")
            self.assertIsNotNone(res.step_snapshots[s.key])

    def test_yolo_stub_execution(self):
        """验证规划中的算法 Y1 能安全实例化并输出规范规划视图"""
        img = _create_synthetic_asparagus_image()
        pipe = PipelineRegistry.create("yolo_deeplearning")
        self.assertIsNotNone(pipe, "创建 yolo_deeplearning 失败")
        res = pipe.run(img, depth_mm=None)
        self.assertIsNotNone(res)
        self.assertIn("stage_overview", res.step_snapshots)
        self.assertIsNotNone(res.step_snapshots["stage_overview"])


def _create_synthetic_asparagus_image() -> np.ndarray:
    """生成带有黑色传送带和两根绿色横卧芦笋的合成图像"""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, int(1280 * 0.35):int(1280 * 0.81)] = (25, 25, 25)
    cv2.rectangle(img, (520, 260), (920, 285), (35, 155, 55), -1)
    cv2.rectangle(img, (540, 305), (900, 328), (40, 160, 60), -1)
    return img


if __name__ == "__main__":
    unittest.main()

