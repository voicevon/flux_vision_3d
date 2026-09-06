#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 16h5 标靶高清生成与排版工具
- 生成 ID 0 ~ 19 的高清标靶图片 (PNG)
- 标明标靶 ID 和几何中心十字线 (特别用于 Tag 0 对齐 SCARA 旋转中心)
- 生成适合 A4 打印的排版拼图
"""

import os
import cv2
import numpy as np


def generate_tags(output_dir: str = "data/apriltags_16h5",
                  tag_count: int = 20,
                  tag_pixel_size: int = 800,
                  border_bits: int = 2):
    """
    生成高清 AprilTag 16h5 标靶
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取 OpenCV 内置的 AprilTag 16h5 字典
    # 注意: cv2.aruco.DICT_APRILTAG_16h5
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    
    generated_files = []
    print(f"[*] 开始生成 AprilTag 16h5 标靶 (ID 0 ~ {tag_count - 1})...")

    # 准备拼图画布 (4 行 5 列)
    cols = 5
    rows = (tag_count + cols - 1) // cols
    card_w = 400
    card_h = 460
    grid_img = np.ones((rows * card_h, cols * card_w, 3), dtype=np.uint8) * 255

    for tag_id in range(tag_count):
        # 1. 生成纯标靶黑白图
        marker_img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_pixel_size, borderBits=border_bits)
        
        # 2. 转换为带边距和文字标注的卡片
        margin = int(tag_pixel_size * 0.15)
        total_w = tag_pixel_size + 2 * margin
        total_h = tag_pixel_size + 2 * margin + 80
        card = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
        
        # 贴入标靶
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        card[margin:margin + tag_pixel_size, margin:margin + tag_pixel_size] = marker_bgr
        
        # 画出边界细外框 (便于人工沿框裁切，测量物理边长)
        cv2.rectangle(card, (margin, margin), 
                      (margin + tag_pixel_size, margin + tag_pixel_size), 
                      (180, 180, 180), 2)
        
        # 如果是 Tag 0，特别在外围画出精密对齐参考刻度标记 (SCARA 旋转原点)
        if tag_id == 0:
            center_x = margin + tag_pixel_size // 2
            center_y = margin + tag_pixel_size // 2
            # 在白边外侧标注中心辅助线
            cv2.line(card, (center_x, 10), (center_x, margin - 10), (0, 0, 255), 2)
            cv2.line(card, (center_x, margin + tag_pixel_size + 10), 
                     (center_x, margin + tag_pixel_size + margin - 10), (0, 0, 255), 2)
            cv2.line(card, (10, center_y), (margin - 10, center_y), (0, 0, 255), 2)
            cv2.line(card, (margin + tag_pixel_size + 10, center_y), 
                     (total_w - 10, center_y), (0, 0, 255), 2)
            label = f"Tag #0 [SCARA Origin Anchor] (16h5)"
        elif tag_id == 1:
            label = f"Tag #1 [World +X Axis Ref] (16h5)"
        else:
            label = f"Tag #{tag_id} (16h5)"
            
        # 写入底部文字
        cv2.putText(card, label, (margin, margin + tag_pixel_size + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        # 保存单独高分辨率图片
        out_path = os.path.join(output_dir, f"tag16h5_id_{tag_id:02d}.png")
        cv2.imwrite(out_path, card)
        generated_files.append(out_path)
        
        # 缩放到排版小格
        r = tag_id // cols
        c = tag_id % cols
        small_card = cv2.resize(card, (card_w, card_h), interpolation=cv2.INTER_AREA)
        grid_img[r * card_h:(r + 1) * card_h, c * card_w:(c + 1) * card_w] = small_card

    # 保存总览排版图
    grid_path = os.path.join(output_dir, "apriltags_16h5_all_grid.png")
    cv2.imwrite(grid_path, grid_img)
    print(f"[OK] 成功生成 {tag_count} 个独立高清标靶文件于: {output_dir}")
    print(f"[OK] 成功生成总览排版大图: {grid_path}")
    return grid_path


if __name__ == "__main__":
    generate_tags()
