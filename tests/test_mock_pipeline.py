"""
单元测试：验证模拟数据生成与对齐帧存储
"""
import os
import sys
import numpy as np

# 添加工程根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.d435_viewer import D435Viewer

def test_mock_generation_and_snapshot():
    viewer = D435Viewer(mock_mode=True)
    viewer.start()
    
    color, depth = viewer.generate_mock_frame()
    assert color.shape == (720, 1280, 3), f"Color shape mismatch: {color.shape}"
    assert depth.shape == (720, 1280), f"Depth shape mismatch: {depth.shape}"
    assert depth.dtype == np.uint16, f"Depth dtype mismatch: {depth.dtype}"
    
    # 验证三层芦笋高度关系 (工作台 640mm, 顶层应当 <= 530mm)
    min_z = np.min(depth)
    max_z = np.max(depth)
    print(f"[TEST] 深度范围: {min_z}mm ~ {max_z}mm")
    assert min_z <= 530, f"最顶层芦笋高度异常: {min_z}"
    assert max_z >= 630, f"工作台底面高度异常: {max_z}"

    # 验证视觉分析器对模拟帧的解析能力
    targets = viewer.analyzer.analyze(color, depth)
    print(f"[TEST] 模拟帧检出芦笋物料: {len(targets)} 根")
    assert len(targets) > 0, "模拟多层芦笋未能被检出"
    top = next((t for t in targets if t.is_topmost), None)
    assert top is not None, "未锁定最顶层芦笋"
    print(f"[TEST] 最顶层芦笋: L={top.length_mm}mm, D={top.diam_mm}mm, 凸起=+{top.rel_height_mm}mm")
    
    print("[TEST] 模拟帧生成与管线解析校验通过！")

if __name__ == "__main__":
    test_mock_generation_and_snapshot()
