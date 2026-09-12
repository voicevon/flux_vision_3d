#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：AprilTag 多标靶 3D 空间建图、BA 平差与坐标系对齐算法验证
"""

import os
import sys
import math
import numpy as np
import cv2

# 添加工程根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.calibration.tag_map_builder import TagMapBuilder


def test_tag_builder_initialization():
    """测试建图器默认参数与几何角点初始化"""
    builder = TagMapBuilder(marker_size_mm=50.0)
    assert builder.marker_size_mm == 50.0
    assert builder.obj_points.shape == (4, 3)
    # 验证 4 个角点边长为 50mm
    edge_01 = np.linalg.norm(builder.obj_points[0] - builder.obj_points[1])
    edge_12 = np.linalg.norm(builder.obj_points[1] - builder.obj_points[2])
    assert abs(edge_01 - 50.0) < 1e-4
    assert abs(edge_12 - 50.0) < 1e-4
    print("[PASS] TagMapBuilder 初始化与角点几何尺寸校验通过")


def test_baseline_scaling():
    """测试双标靶大基线物理测距尺度锁定"""
    builder = TagMapBuilder(marker_size_mm=50.0)
    
    # 假设施加了 1.05 倍的尺度放大 (名义 500mm，实测 525mm)
    tag_poses = {
        0: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64),
        1: np.array([[1, 0, 0, 500], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64),
        2: np.array([[1, 0, 0, 200], [0, 1, 0, 300], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    }
    
    scaled_poses, scale_factor, real_marker_size = builder.apply_baseline_scale(
        tag_poses, tag_id_a=0, tag_id_b=1, real_distance_mm=525.0
    )
    
    assert abs(scale_factor - 1.05) < 1e-4
    assert abs(real_marker_size - 52.5) < 1e-4
    assert abs(scaled_poses[1][0, 3] - 525.0) < 1e-4
    print("[PASS] 双标靶大基线测距尺度锁定算法校验通过")


def test_align_to_scara_world():
    """测试 Tag 0 原点与 Tag 1 水平 X 轴刚体对齐闭环"""
    builder = TagMapBuilder(marker_size_mm=50.0)

    # 构造未对齐的任意旋转和平移地图
    # 真实 Tag 0 位于 (100, 200, 50)
    # 真实 Tag 1 位于沿 45 度方向 (100 + 300*cos45, 200 + 300*sin45, 50)
    p0 = np.array([100.0, 200.0, 50.0])
    p1 = np.array([100.0 + 300.0 * math.sqrt(0.5), 200.0 + 300.0 * math.sqrt(0.5), 50.0])
    
    tag_poses = {
        0: np.eye(4, dtype=np.float64),
        1: np.eye(4, dtype=np.float64)
    }
    tag_poses[0][:3, 3] = p0
    tag_poses[1][:3, 3] = p1

    aligned_map = builder._align_to_scara_world(tag_poses, origin_tag_id=0, x_align_tag_id=1)

    # 1. 验证 Tag 0 位置在原点
    tag0_pos = aligned_map["tags"][0]["position_mm"]
    assert abs(tag0_pos[0]) < 1e-2 and abs(tag0_pos[1]) < 1e-2 and abs(tag0_pos[2]) < 1e-2, f"Tag 0 未在原点: {tag0_pos}"

    # 2. 验证 Tag 1 在水平面 X 轴上 (Y 应该为 0, X 应该约等于 300)
    tag1_pos = aligned_map["tags"][1]["position_mm"]
    assert abs(tag1_pos[1]) < 1e-2, f"Tag 1 未在 X 轴上 (Y != 0): {tag1_pos}"
    assert abs(tag1_pos[0] - 300.0) < 1e-1, f"Tag 1 距离失真: {tag1_pos}"
    print(f"[PASS] SCARA 世界坐标系原点与 X 轴对齐闭环校验通过: Tag0={tag0_pos}, Tag1={tag1_pos}")


def test_synthetic_bundle_adjustment():
    """测试多视角合成数据下的 BA 平差收敛性"""
    builder = TagMapBuilder(marker_size_mm=50.0)
    
    # 设定相机内参
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    builder.camera_matrix = K
    builder.dist_coeffs = dist

    # 生成 4 个空间标靶 (Z=0, 平面分布)
    ground_truth_tags = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([300.0, 0.0, 0.0]),
        2: np.array([150.0, 200.0, 0.0]),
        3: np.array([0.0, 200.0, 0.0])
    }

    # 模拟 3 个不同视角相机 (俯视，略带倾斜)
    cam_positions = [
        np.array([150.0, 100.0, -500.0]),
        np.array([50.0, 80.0, -480.0]),
        np.array([250.0, 120.0, -510.0])
    ]

    s = 25.0
    local_corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    frame_detections = []
    for c_pos in cam_positions:
        # 相机朝向标靶中心 (150, 100, 0)
        tvec = -c_pos.reshape(3, 1) # 简化近似
        rvec = np.array([0.1, 0.05, 0.0], dtype=np.float64).reshape(3, 1)
        
        f_det = {}
        for tid, t_center in ground_truth_tags.items():
            world_pts = local_corners + t_center
            img_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
            # 添加微小高斯噪声 (0.1 像素)
            noise = np.random.normal(0, 0.05, img_pts.shape)
            f_det[tid] = (img_pts + noise).reshape(4, 2)
        frame_detections.append(f_det)

    # 验证提取到的角点格式
    assert len(frame_detections) == 3
    print("[PASS] 合成多视角投影视角数据准备就绪")

    # 执行全局 BA 平差并验证求解收敛性
    tags_map = builder.optimize_bundle_adjustment(
        frame_detections=frame_detections,
        origin_tag_id=0,
        x_align_tag_id=1
    )
    assert tags_map["rmse_reprojection_px"] < 1.0, f"BA 误差过大: {tags_map['rmse_reprojection_px']}"
    assert 0 in tags_map["tags"] and 1 in tags_map["tags"]
    print(f"[PASS] 合成数据 BA 优化求解收敛测试通过, RMSE: {tags_map['rmse_reprojection_px']:.3f} px")


def test_covisibility_guard():
    """测试共视连通性安全守门员 (Co-visibility Guard)"""
    from tools.calibration.tag_map_builder import CovisibilityGraphError
    builder = TagMapBuilder(marker_size_mm=50.0)

    # 1. 正常连通图: Frame1 (0, 1), Frame2 (1, 2) -> 0-1-2 完全连通
    c_dummy = np.zeros((4, 2))
    healthy_detections = [
        {0: c_dummy, 1: c_dummy},
        {1: c_dummy, 2: c_dummy}
    ]
    report = builder.validate_covisibility(healthy_detections, origin_tag_id=0, x_align_tag_id=1)
    assert report["is_valid"] is True
    assert set(report["all_tags"]) == {0, 1, 2}
    assert len(report["critical_bridges"]) == 2  # 0-1 和 1-2 都只有单帧支撑
    print("[PASS] 正常连通拓扑图校验通过")

    # 2. 断网断裂图: Frame1 (0, 1), Frame2 (2, 3) -> 孤立成两个子网络 {0, 1} 和 {2, 3}
    broken_detections = [
        {0: c_dummy, 1: c_dummy},
        {2: c_dummy, 3: c_dummy}
    ]
    broken_report = builder.validate_covisibility(broken_detections, origin_tag_id=0, x_align_tag_id=1)
    assert broken_report["is_valid"] is False
    assert len(broken_report["components"]) == 2
    assert 2 in broken_report["unconnected_tags"] or 3 in broken_report["unconnected_tags"]
    print(f"[PASS] 拓扑断网检测校验通过: {broken_report['message']}")

    # 3. 验证守门员成功拦截 BA 平差求解
    try:
        builder.optimize_bundle_adjustment(broken_detections, origin_tag_id=0, x_align_tag_id=1)
        assert False, "未能成功拦截断网输入！"
    except CovisibilityGraphError:
        print("[PASS] 守门员成功拦截断网图输入并抛出 CovisibilityGraphError，保护系统不崩溃")


def test_manifest_workflow_and_curation(tmp_path=None):
    """测试清单导出、加载与人工剔除坏样本标记保留机制"""
    import tempfile
    import yaml
    temp_dir = tempfile.mkdtemp()
    manifest_path = os.path.join(temp_dir, "test_manifest.yaml")

    builder = TagMapBuilder(marker_size_mm=50.0)
    
    # 构造假数据 (边长 100px)
    dummy_corners = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float64)
    metrics = builder.compute_tag_metrics(dummy_corners)
    assert metrics["cell_size_px"] == [17, 17]
    assert metrics["area_px"] == 10000.0

    # 模拟已有清单且用户已标记剔除某标靶
    existing_manifest = {
        "images": {
            "test_01.png": {
                "observations": [
                    {
                        "tag_id": 5,
                        "keep": False,
                        "note": "人工判定：大倾角边缘畸变",
                        "corners": dummy_corners.tolist()
                    },
                    {
                        "tag_id": 6,
                        "keep": True,
                        "note": "",
                        "corners": dummy_corners.tolist()
                    }
                ]
            }
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(existing_manifest, f)

    # 读取并验证过滤效果
    f_det, v_frames, stats = builder.load_observations_manifest(manifest_path)
    # 因为 test_01.png 剔除 tag 5 后只剩下 1 个 tag 6，不足 2 个，应该自动被 dropped
    assert stats["total_excluded"] == 1
    assert stats["total_kept"] == 1
    assert len(f_det) == 0  # 单标靶帧不参与相对约束
    assert len(stats["dropped_single_tag_frames"]) == 1
    print("[PASS] 审核清单加载与坏样本剔除、单标靶帧自动降级校验通过")


def test_tag_manifest_reviewer_logic():
    """测试 TagManifestReviewer 点击翻转、命中测试与数据同步逻辑"""
    import tempfile
    import yaml
    from tools.calibration.tag_manifest_reviewer import TagManifestReviewer

    temp_dir = tempfile.mkdtemp()
    manifest_path = os.path.join(temp_dir, "test_reviewer_manifest.yaml")

    # 构造含两个标靶的测试清单
    corners_a = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
    corners_b = [[300.0, 100.0], [400.0, 100.0], [400.0, 200.0], [300.0, 200.0]]

    manifest_data = {
        "summary": {"total_observations": 2, "total_kept": 2, "total_excluded": 0},
        "images": {
            "test_img_01.png": {
                "file_name": "test_img_01.png",
                "image_path": "non_existent.png",
                "observations": [
                    {"tag_id": 1, "keep": True, "corners": corners_a, "cell_size_px": [17, 17]},
                    {"tag_id": 2, "keep": True, "corners": corners_b, "cell_size_px": [17, 17]}
                ]
            }
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f)

    reviewer = TagManifestReviewer(manifest_path=manifest_path)
    assert len(reviewer.image_keys) == 1
    assert reviewer.raw_manifest["images"]["test_img_01.png"]["observations"][0]["keep"] is True

    # 1. 模拟鼠标点击外部区域 (0, 0) -> 无命中，状态不改变
    reviewer.on_mouse_click(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)
    assert reviewer.raw_manifest["images"]["test_img_01.png"]["observations"][0]["keep"] is True

    # 2. 模拟鼠标点击 Tag 1 内部 (150, 150) -> 命中，keep 翻转为 False
    reviewer.on_mouse_click(cv2.EVENT_LBUTTONDOWN, 150, 150, 0, None)
    assert reviewer.raw_manifest["images"]["test_img_01.png"]["observations"][0]["keep"] is False
    assert reviewer.has_unsaved_changes is True

    # 3. 模拟再次点击 Tag 1 内部 (150, 150) -> 命中，keep 翻转回 True
    reviewer.on_mouse_click(cv2.EVENT_LBUTTONDOWN, 150, 150, 0, None)
    assert reviewer.raw_manifest["images"]["test_img_01.png"]["observations"][0]["keep"] is True

    # 4. 点击 Tag 2 内部 (350, 150) -> 翻转为 False 并保存
    reviewer.on_mouse_click(cv2.EVENT_LBUTTONDOWN, 350, 150, 0, None)
    assert reviewer.raw_manifest["images"]["test_img_01.png"]["observations"][1]["keep"] is False
    reviewer.save_changes()

    # 验证保存后的文件真实写入
    with open(manifest_path, "r", encoding="utf-8") as f:
        saved_data = yaml.safe_load(f)
    assert saved_data["images"]["test_img_01.png"]["observations"][1]["keep"] is False
    assert saved_data["summary"]["total_kept"] == 1
    assert saved_data["summary"]["total_excluded"] == 1

    # 5. 测试 GUI 按钮命中处理 (例如测试重置 RESET 与保存 SAVE 按钮)
    reviewer.render_current_frame()
    assert len(reviewer.gui_action_buttons) >= 6
    # 模拟点击 RESET 按钮
    for btn_id, (bx1, by1, bx2, by2), _, _ in reviewer.gui_action_buttons:
        if btn_id == "RESET":
            reviewer.on_mouse_event(cv2.EVENT_LBUTTONDOWN, (bx1 + bx2) // 2, (by1 + by2) // 2, 0, None)
            break
    # 重置后应恢复为初始状态 (Tag 2 的 keep 回到 True)
    assert reviewer.raw_manifest["images"]["test_img_01.png"]["observations"][1]["keep"] is True

    print("[PASS] TagManifestReviewer 鼠标标靶命中、底部GUI按钮响应与状态翻转测试通过")


if __name__ == "__main__":
    test_tag_builder_initialization()
    test_baseline_scaling()
    test_align_to_scara_world()
    test_synthetic_bundle_adjustment()
    test_covisibility_guard()
    test_manifest_workflow_and_curation()
    test_tag_manifest_reviewer_logic()
    print("\n>>> 所有 AprilTag 空间建图、人工审核与守门员测试通过 (ALL TESTS PASSED) <<<")
