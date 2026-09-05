"""
使用用户真实捕获的数据帧测试芦笋识别、尺寸测量与最顶层抓取点解算
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.asparagus_analyzer import AsparagusAnalyzer

def test_on_real_snapshot():
    color_path = "data/snapshots/color_20260905_224312.png"
    depth_path = "data/snapshots/depth_raw_20260905_224312.npy"
    
    if not os.path.exists(color_path) or not os.path.exists(depth_path):
        print("[SKIP] 未找到真实快照文件，跳过实测。")
        return
        
    color = cv2.imread(color_path)
    depth = np.load(depth_path)
    
    analyzer = AsparagusAnalyzer(fx=909.12, fy=907.46, cx=647.46, cy=377.51)
    targets = analyzer.analyze(color, depth)
    
    print(f"\n[RESULT] 检测到 {len(targets)} 个目标:")
    for t in targets:
        top_str = "[TOPMOST]" if t.is_topmost else "         "
        print(f"{top_str} ID#{t.id}: L={t.length_mm:.1f}mm, D={t.diam_mm:.1f}mm, Yaw={t.yaw_deg:.1f}deg, Grip=(X:{t.grip_x:.1f}, Y:{t.grip_y:.1f}, Z:{t.grip_z:.1f})mm")
        
    assert len(targets) == 3, f"预期检测到 3 根芦笋，实际检测到 {len(targets)} 根"
    
    # 渲染带检测结果的画面并保存
    annotated = analyzer.draw_detections(color, targets)
    out_path = "data/snapshots/analyzed_result_224312.png"
    cv2.imwrite(out_path, annotated)
    print(f"[SAVE] 渲染结果已保存至: {out_path}\n")

if __name__ == "__main__":
    test_on_real_snapshot()
