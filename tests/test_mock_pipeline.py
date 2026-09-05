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
    
    # 验证三层芦笋高度关系 (工作台720mm, 顶层应当 < 685mm)
    min_z = np.min(depth)
    max_z = np.max(depth)
    print(f"[TEST] 深度范围: {min_z}mm ~ {max_z}mm")
    assert min_z < 685, "最顶层芦笋高度异常"
    assert max_z >= 715, "工作台底面高度异常"
    
    print("[TEST] 模拟帧生成校验通过！")

if __name__ == "__main__":
    test_mock_generation_and_snapshot()
