#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 16h5 标靶高清生成与排版工具
- 生成 ID 0 ~ 19 的高清标靶图片 (PNG)
- 标明标靶 ID 和几何中心十字线 (特别用于 Tag 0 对齐 SCARA 旋转中心)
- 生成适合 A4 打印的排版拼图
"""

import os
import sys
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.text_rendering import measure_text, put_text
from src.utils.logger import get_logger

log = get_logger(__name__)


def generate_tags(output_dir: str = "data/apriltags_16h5",
                  tag_count: int = 30,
                  tag_pixel_size: int = 800,
                  border_bits: int = 2):
    """
    生成高清 AprilTag 16h5 标靶 (PNG 格式) 并排版为高精度严格 1:1 的 A4 PDF 文件。
    
    参数:
        output_dir: 输出目录
        tag_count: 标靶总数 (默认 30，对应 ID 00 ~ 29)
        tag_pixel_size: 单个标靶高分辨率位图的像素尺寸 (默认 800x800)
        border_bits: AprilTag 黑色边框宽度 (默认 2)
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_marker_dir = os.path.join(output_dir, "_raw_markers")
    os.makedirs(raw_marker_dir, exist_ok=True)
    
    # 获取 OpenCV 内置的 AprilTag 16h5 字典
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    
    generated_png_files = []
    raw_marker_files = []
    log.info(f"[*] 开始生成 AprilTag 16h5 标靶 (ID 00 ~ {tag_count - 1:02d})...")

    # 准备 5 列 x 6 行 PNG 总览网格图 (方便快速屏幕查看)
    cols = 5
    rows = (tag_count + cols - 1) // cols
    card_w = 400
    card_h = 460
    grid_img = np.ones((rows * card_h, cols * card_w, 3), dtype=np.uint8) * 255

    for tag_id in range(tag_count):
        # 1. 生成纯标靶正方形黑白图像 (严格 1:1)
        marker_img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_pixel_size, borderBits=border_bits)
        raw_marker_path = os.path.join(raw_marker_dir, f"raw_marker_{tag_id:02d}.png")
        cv2.imwrite(raw_marker_path, marker_img)
        raw_marker_files.append(raw_marker_path)
        
        # 2. 转换为带边距和文字标注的独立 PNG 卡片
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
        
        # Tag 0 特别标注 SCARA 旋转中心与对齐刻度
        if tag_id == 0:
            center_x = margin + tag_pixel_size // 2
            center_y = margin + tag_pixel_size // 2
            cv2.line(card, (center_x, 10), (center_x, margin - 10), (0, 0, 255), 2)
            cv2.line(card, (center_x, margin + tag_pixel_size + 10), 
                     (center_x, margin + tag_pixel_size + margin - 10), (0, 0, 255), 2)
            cv2.line(card, (10, center_y), (margin - 10, center_y), (0, 0, 255), 2)
            cv2.line(card, (margin + tag_pixel_size + 10, center_y), 
                     (total_w - 10, center_y), (0, 0, 255), 2)
            label = f"Tag #00 [SCARA Origin Anchor] (16h5)"
        elif tag_id == 1:
            label = f"Tag #01 [World +X Axis Ref] (16h5)"
        else:
            label = f"Tag #{tag_id:02d} (16h5)"
            
        put_text(card, label, (margin, margin + tag_pixel_size + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        
        out_png_path = os.path.join(output_dir, f"tag16h5_id_{tag_id:02d}.png")
        cv2.imwrite(out_png_path, card)
        generated_png_files.append(out_png_path)
        
        # 填充到 PNG 总览图
        r = tag_id // cols
        c = tag_id % cols
        small_card = cv2.resize(card, (card_w, card_h), interpolation=cv2.INTER_AREA)
        grid_img[r * card_h:(r + 1) * card_h, c * card_w:(c + 1) * card_w] = small_card

    # 保存 PNG 总览大图
    grid_png_path = os.path.join(output_dir, "apriltags_16h5_all_grid.png")
    cv2.imwrite(grid_png_path, grid_img)
    log.info(f"[OK] 成功生成 {tag_count} 个独立高清标靶 PNG 文件: tag16h5_id_00.png ~ tag16h5_id_{tag_count - 1:02d}.png")
    log.info(f"[OK] 成功生成总览排版 PNG: {grid_png_path}")

    # =========================================================================
    # 生成高精度、严格 1:1 比例的 A4 打印 PDF 文件 (30 个标靶排于同一页)
    # =========================================================================
    pdf_path = os.path.join(output_dir, "apriltags_16h5_grid_a4.pdf")
    build_a4_pdf(pdf_path, raw_marker_files, tag_count=tag_count)
    log.info(f"[OK] 成功生成严格 1:1 A4 排版 PDF 文件: {pdf_path}")

    # 清理临时 raw_markers
    for f in raw_marker_files:
        try:
            os.remove(f)
        except OSError:
            pass  # 清理容错：临时文件删除失败不影响成品交付
    try:
        os.rmdir(raw_marker_dir)
    except OSError:
        pass  # 清理容错：临时目录删除失败不影响成品交付

    return pdf_path


def build_a4_pdf(pdf_path: str, raw_marker_files: list, tag_count: int = 30):
    """
    使用 ReportLab 构建高精度、绝对严格 1:1 长宽比的两页 A4 PDF 标靶纸。
    - 每页排版: 3 列 x 5 行 = 15 个标靶 (共 2 页，满载 30 个)
    - 标靶物理尺寸放大至: 40.0 mm x 40.0 mm (面积为之前 28mm 的 2 倍以上，极大提升视觉识别距离与精度)
    - 每页顶部均配备 100.0 mm 物理打印校验尺，供游标卡尺测量验证 100% 打印无失真。
    """
    page_w, page_h = A4  # 210mm x 297mm
    c = canvas.Canvas(pdf_path, pagesize=A4)
    
    tags_per_page = 15
    cols = 3
    rows = 5
    num_pages = (tag_count + tags_per_page - 1) // tags_per_page  # 2 页
    
    # 网格与标靶物理尺寸参数
    tag_size_mm = 40.0
    tag_draw_size = tag_size_mm * mm
    
    margin_x = 12.0 * mm
    grid_w = page_w - 2 * margin_x         # 210 - 24 = 186.0 mm
    col_w = grid_w / cols                  # 186 / 3 = 62.0 mm
    
    top_grid_y = page_h - 26.0 * mm        # 顶部留 26mm 放置标题、说明与 100mm 校验尺
    bottom_grid_y = 8.0 * mm               # 底部留 8mm 边距
    grid_h = top_grid_y - bottom_grid_y    # 263.0 mm
    row_h = grid_h / rows                  # 263 / 5 = 52.6 mm

    for page_idx in range(num_pages):
        start_id = page_idx * tags_per_page
        end_id = min(start_id + tags_per_page, tag_count)
        
        # ---------------- 1. 顶部标题、页码与说明 ----------------
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(page_w / 2.0, page_h - 11.0 * mm, 
                            f"AprilTag 16h5 Calibration Sheet (ID {start_id:02d} - {end_id - 1:02d}) [Page {page_idx + 1}/{num_pages}]")
        
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#333333"))
        c.drawCentredString(page_w / 2.0, page_h - 15.0 * mm, 
                            "Print Setting: Set scale to '100% / Actual Size' (Do NOT 'Fit to Paper'). Aspect ratio: Strictly 1:1.")

        # ---------------- 2. 100.0 mm 打印比例校验标尺 ----------------
        scale_len_mm = 100.0
        scale_x1 = (210.0 * mm - scale_len_mm * mm) / 2.0
        scale_x2 = scale_x1 + scale_len_mm * mm
        scale_y = page_h - 20.5 * mm
        
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.8)
        # 主水平线与端点垂直卡线
        c.line(scale_x1, scale_y, scale_x2, scale_y)
        c.line(scale_x1, scale_y - 2.5 * mm, scale_x1, scale_y + 2.5 * mm)
        c.line(scale_x2, scale_y - 2.5 * mm, scale_x2, scale_y + 2.5 * mm)
        # 中间 50mm 刻度线
        c.line(scale_x1 + 50.0 * mm, scale_y - 1.5 * mm, scale_x1 + 50.0 * mm, scale_y + 1.5 * mm)
        
        # 10mm 细刻度
        c.setLineWidth(0.4)
        for tick_idx in range(1, 10):
            if tick_idx == 5:
                continue
            tx = scale_x1 + tick_idx * 10.0 * mm
            c.line(tx, scale_y - 1.0 * mm, tx, scale_y + 1.0 * mm)

        c.setFont("Helvetica-Bold", 6.5)
        c.setFillColor(colors.black)
        c.drawCentredString(page_w / 2.0, scale_y + 1.2 * mm, "|<--- 100.0 mm Scale Verification Bar (Measure with Caliper) --->|")

        # ---------------- 3. 本页 15 个标靶网格排版 (3 列 x 5 行) ----------------
        for idx in range(start_id, end_id):
            local_idx = idx - start_id
            r = local_idx // cols
            col_idx = local_idx % cols
            
            cell_x = margin_x + col_idx * col_w
            # ReportLab 原点在左下角，第 0 行在最上方
            cell_y = top_grid_y - (r + 1) * row_h
            
            # 绘制单元格轻微裁切辅助虚线
            c.setStrokeColor(colors.HexColor("#D0D0D0"))
            c.setLineWidth(0.3)
            c.setDash(2, 2)
            c.rect(cell_x, cell_y, col_w, row_h, stroke=1, fill=0)
            c.setDash()  # 恢复实线
            
            # 标靶居中水平放置，下方预留 9.5mm 给文字
            tag_x = cell_x + (col_w - tag_draw_size) / 2.0
            tag_y = cell_y + 9.5 * mm
            
            # 绘制严格 1:1 正方形 AprilTag 图片
            marker_file = raw_marker_files[idx]
            c.drawImage(marker_file, tag_x, tag_y, 
                        width=tag_draw_size, height=tag_draw_size, 
                        preserveAspectRatio=True)
            
            # 标靶外框细线 (0.4pt 浅灰，方便卡尺实测外沿)
            c.setStrokeColor(colors.HexColor("#999999"))
            c.setLineWidth(0.4)
            c.rect(tag_x, tag_y, tag_draw_size, tag_draw_size, stroke=1, fill=0)
            
            # 特别对 Tag 0 绘制几何中心十字刻度辅助红线 (用于机械臂原点对准)
            if idx == 0:
                cx = tag_x + tag_draw_size / 2.0
                cy = tag_y + tag_draw_size / 2.0
                c.setStrokeColor(colors.red)
                c.setLineWidth(0.7)
                c.line(cx, tag_y + tag_draw_size, cx, tag_y + tag_draw_size + 2.5 * mm)
                c.line(cx, tag_y, cx, tag_y - 2.5 * mm)
                c.line(tag_x - 2.5 * mm, cy, tag_x, cy)
                c.line(tag_x + tag_draw_size, cy, tag_x + tag_draw_size + 2.5 * mm, cy)
            
            # 底部文字信息
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 8.0)
            if idx == 0:
                label_title = "Tag #00 [SCARA Origin]"
            elif idx == 1:
                label_title = "Tag #01 [Ref +X Axis]"
            else:
                label_title = f"Tag #{idx:02d}"
            c.drawCentredString(cell_x + col_w / 2.0, cell_y + 5.5 * mm, label_title)
            
            c.setFont("Helvetica", 6.0)
            c.setFillColor(colors.HexColor("#555555"))
            c.drawCentredString(cell_x + col_w / 2.0, cell_y + 2.4 * mm, f"16h5 | Size: {tag_size_mm:.1f} x {tag_size_mm:.1f} mm")

        c.showPage()
        
    c.save()


if __name__ == "__main__":
    generate_tags()
