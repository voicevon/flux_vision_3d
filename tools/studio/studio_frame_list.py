"""
AprilTag 离线标定工作站 - 左栏帧列表渲染 Mixin (StudioFrameListMixin)
================================================================================
承载 StudioUIRenderer 的左栏绘制分区：
1. render_left_frame_list: 高信息密度垂直紧凑帧列表 与 逐帧多轮残差演进矩阵宽表
仅包含纯绘制方法, 不持有任何状态; 通过 self 依赖宿主 StudioUIRenderer 的其他方法,
由 MRO 解析跨分区调用。
"""

import os
from typing import Any
import cv2
import numpy as np

from src.utils.text_rendering import measure_text, put_text
from src.utils.viewport_manager import draw_styled_button
from src.utils.gui_components import draw_dropdown_button
from tools.studio.studio_ui_common import FILTER_MODE_OPTIONS, SORT_MODE_OPTIONS


class StudioFrameListMixin:
    """左栏帧列表渲染 Mixin (由宿主类 StudioUIRenderer 组合)"""

    def render_left_frame_list(self, studio: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """左栏：高信息密度垂直紧凑帧列表 或 逐帧多轮残差演进矩阵宽表大视图"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x + w, y), (x + w, y + h), (50, 54, 66), 1)

        is_matrix = getattr(studio, "matrix_view_mode", False)
        headers = getattr(studio.data_mgr, "convergence_headers", [])
        matrix = getattr(studio.data_mgr, "frame_convergence_matrix", {})

        header_h = 36
        dd_y1 = y + 6
        dd_y2 = y + header_h

        # 1. 顶部操作栏自适应排版：切换按钮 + 筛选下拉框 + 排序下拉框
        btn_w = 98 if is_matrix else 86
        btn_x1 = x + w - btn_w - 8
        btn_x2 = x + w - 8

        avail_w = max(120, btn_x1 - (x + 8) - 8)
        dd1_w = avail_w // 2
        dd2_w = avail_w - dd1_w - 6

        dd1_x1 = x + 8
        dd1_x2 = dd1_x1 + dd1_w
        dd2_x1 = dd1_x2 + 6
        dd2_x2 = dd2_x1 + dd2_w

        # 筛选范围下拉框
        cur_filter_label = dict(FILTER_MODE_OPTIONS).get(studio.filter_mode, "全部帧")
        is_f_open = (studio.active_dropdown == "FILTER_DROPDOWN")
        draw_dropdown_button(canvas, (dd1_x1, dd_y1, dd1_x2, dd_y2), cur_filter_label,
                             is_open=is_f_open, mouse_pos=studio.mouse_pos)
        studio.dropdown_boxes["FILTER_DROPDOWN"] = {
            "rect": (dd1_x1, dd_y1, dd1_x2, dd_y2),
            "options": FILTER_MODE_OPTIONS,
            "active_key": studio.filter_mode
        }
        studio.gui_buttons.append(("TOGGLE_FILTER_DROPDOWN", (dd1_x1, dd_y1, dd1_x2, dd_y2), "FILTER_DROPDOWN"))

        # 排序方式下拉框
        cur_sort_label = dict(SORT_MODE_OPTIONS).get(studio.sort_mode, "文件名升序")
        short_sort = cur_sort_label.split(" ")[0] if "(" in cur_sort_label else cur_sort_label
        is_s_open = (studio.active_dropdown == "SORT_DROPDOWN")
        draw_dropdown_button(canvas, (dd2_x1, dd_y1, dd2_x2, dd_y2), short_sort,
                             is_open=is_s_open, mouse_pos=studio.mouse_pos)
        studio.dropdown_boxes["SORT_DROPDOWN"] = {
            "rect": (dd2_x1, dd_y1, dd2_x2, dd_y2),
            "options": SORT_MODE_OPTIONS,
            "active_key": studio.sort_mode
        }
        studio.gui_buttons.append(("TOGGLE_SORT_DROPDOWN", (dd2_x1, dd_y1, dd2_x2, dd_y2), "SORT_DROPDOWN"))

        # 矩阵模式切换按钮 (一键展开/收起 10 轮残差对比大表)
        btn_txt = "⊟ 紧凑 (X)" if is_matrix else "⊞ 矩阵 (X)"
        btn_type = "primary" if is_matrix else "secondary"
        draw_styled_button(canvas, (btn_x1, dd_y1, btn_x2, dd_y2), btn_txt,
                           mouse_pos=studio.mouse_pos, btn_type=btn_type)
        studio.gui_buttons.append(("TOGGLE_MATRIX_VIEW", (btn_x1, dd_y1, btn_x2, dd_y2), "TOGGLE_MATRIX_VIEW"))

        filtered_indices = studio._get_filtered_indices()
        if not filtered_indices:
            put_text(canvas, "当前筛选条件下无图像", (x + 60, y + header_h + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1, cv2.LINE_AA)
            return

        # 2. 列表内容区布局与渲染
        if is_matrix:
            # ==================== 【模式 A: 逐帧多轮残差演进矩阵宽表】 ====================
            th_h = 24
            th_y1 = y + header_h + 6
            th_y2 = th_y1 + th_h
            list_y = th_y2 + 4
            item_h = 30
            visible_count = (h - (list_y - y) - 10) // item_h

            studio.scroll_offset = max(0, min(studio.scroll_offset, len(filtered_indices) - visible_count))

            # 绘制矩阵表头背景
            cv2.rectangle(canvas, (x + 6, th_y1), (x + w - 6, th_y2), (32, 36, 46), -1)
            cv2.rectangle(canvas, (x + 6, th_y1), (x + w - 6, th_y2), (52, 58, 72), 1)

            c_name_w = 110
            c_tag_w = 34
            c_round_w = 54
            c_drop_w = 68

            # 表头固定列
            put_text(canvas, "图像帧", (x + 14, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)
            put_text(canvas, "Tag", (x + 14 + c_name_w, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)

            cur_col_x = x + 14 + c_name_w + c_tag_w
            # 动态平差轮次表头列 (R0, R1, R2, ..., R10)
            active_headers = headers if headers else ["基准(R0)"]
            for col_idx, h_name in enumerate(active_headers):
                is_latest_col = (col_idx == len(active_headers) - 1)
                th_color = (0, 240, 255) if is_latest_col else (180, 185, 195)
                # 最新一轮列高亮底框
                if is_latest_col and len(active_headers) > 1:
                    cv2.rectangle(canvas, (cur_col_x - 2, th_y1 + 2), (cur_col_x + c_round_w - 4, th_y2 - 2), (25, 60, 75), -1)
                put_text(canvas, h_name, (cur_col_x + 6, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, th_color, 1 if not is_latest_col else 2, cv2.LINE_AA)
                cur_col_x += c_round_w

            # 累计改善降幅列
            put_text(canvas, "累计降幅", (cur_col_x + 4, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 120), 1, cv2.LINE_AA)

            # 绘制逐帧矩阵数据行
            for row_idx in range(visible_count):
                list_idx = studio.scroll_offset + row_idx
                if list_idx >= len(filtered_indices):
                    break

                orig_img_idx = filtered_indices[list_idx]
                p = studio.image_files[orig_img_idx]
                bname = os.path.basename(p)
                meta = studio.frame_metrics_cache.get(bname, {})

                iy1 = list_y + row_idx * item_h
                iy2 = iy1 + item_h - 2
                is_selected = (orig_img_idx == studio.current_img_idx)
                is_hover = (x + 6 <= studio.mouse_pos[0] <= x + w - 6 and iy1 <= studio.mouse_pos[1] <= iy2)

                if is_selected:
                    row_bg = (48, 42, 28)
                    border_c = (0, 220, 255)
                elif is_hover:
                    row_bg = (35, 38, 48)
                    border_c = (55, 60, 75)
                else:
                    row_bg = (24, 26, 33) if row_idx % 2 == 0 else (20, 22, 28)
                    border_c = (36, 38, 48)

                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), row_bg, -1)
                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), border_c, 1)

                btn_id = f"SELECT_FRAME_{orig_img_idx}"
                studio.gui_buttons.append((btn_id, (x + 6, iy1, x + w - 6, iy2), orig_img_idx))

                # 状态小圆点
                dot_y = iy1 + item_h // 2
                is_excl = meta.get("is_excluded", False)
                if is_excl:
                    dot_c = (0, 0, 240)
                elif meta.get("mean_err", 0.0) > 0.5:
                    dot_c = (0, 180, 255)
                else:
                    dot_c = (0, 230, 80)
                cv2.circle(canvas, (x + 14, dot_y), 3, dot_c, -1)

                # 文件名 (截取前10个字符)
                name_stem = bname.replace(".png", "").replace(".jpg", "")
                short_name = name_stem if len(name_stem) <= 10 else name_stem[:9] + "…"
                name_col = (255, 255, 255) if is_selected else (200, 205, 215)
                put_text(canvas, short_name, (x + 22, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, name_col, 1, cv2.LINE_AA)

                # Tag 数量
                tag_cnt = meta.get("tag_count", 0)
                put_text(canvas, f"{tag_cnt}T", (x + 14 + c_name_w, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 150, 165), 1, cv2.LINE_AA)

                # 各轮次残差单元格值 (R0, R1, ..., R10)
                row_vals = matrix.get(bname, [])
                r_col_x = x + 14 + c_name_w + c_tag_w

                for c_idx in range(len(active_headers)):
                    is_latest_c = (c_idx == len(active_headers) - 1)
                    val = row_vals[c_idx] if c_idx < len(row_vals) else None

                    # 最新一轮单元格微高亮底框
                    if is_latest_c and len(active_headers) > 1 and not is_selected:
                        cv2.rectangle(canvas, (r_col_x - 2, iy1 + 2), (r_col_x + c_round_w - 6, iy2 - 2), (20, 42, 54), -1)

                    if val is None or is_excl:
                        v_txt = "EXCL" if is_excl else "--"
                        v_col = (90, 95, 110) if not is_excl else (0, 0, 200)
                        put_text(canvas, v_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, v_col, 1, cv2.LINE_AA)
                    else:
                        v_txt = f"{val:.1f}" if val >= 100.0 else f"{val:.2f}"
                        if val > 1.0:
                            v_col = (0, 180, 255)
                        elif val > 0.5:
                            v_col = (0, 220, 255)
                        else:
                            v_col = (0, 240, 100)
                        put_text(canvas, v_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, v_col, 1, cv2.LINE_AA)
                    r_col_x += c_round_w

                # 累计降幅百分比
                first_v = row_vals[0] if (row_vals and row_vals[0] is not None) else None
                last_v = None
                for rv in reversed(row_vals):
                    if rv is not None:
                        last_v = rv
                        break

                if first_v is not None and last_v is not None and first_v > 0.001:
                    drop_val = first_v - last_v
                    drop_pct = (drop_val / first_v) * 100.0
                    if drop_pct >= 5.0:
                        pct_txt = f"↓{drop_pct:.0f}%" if drop_pct >= 10.0 else f"↓{drop_pct:.1f}%"
                        pct_col = (0, 240, 255)
                    elif drop_pct <= -5.0:
                        pct_txt = f"↑{abs(drop_pct):.0f}%"
                        pct_col = (0, 100, 255)
                    else:
                        pct_txt = "0%"
                        pct_col = (150, 160, 175)
                else:
                    pct_txt = "--"
                    pct_col = (110, 115, 125)

                put_text(canvas, pct_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pct_col, 1, cv2.LINE_AA)

        else:
            # ==================== 【模式 B: 高信息密度垂直紧凑帧列表】 ====================
            list_y = y + header_h + 8
            item_h = 36
            visible_count = (h - header_h - 16) // item_h

            studio.scroll_offset = max(0, min(studio.scroll_offset, len(filtered_indices) - visible_count))

            for row_idx in range(visible_count):
                list_idx = studio.scroll_offset + row_idx
                if list_idx >= len(filtered_indices):
                    break

                orig_img_idx = filtered_indices[list_idx]
                p = studio.image_files[orig_img_idx]
                bname = os.path.basename(p)
                meta = studio.frame_metrics_cache.get(bname, {})

                iy1 = list_y + row_idx * item_h
                iy2 = iy1 + item_h - 2
                is_selected = (orig_img_idx == studio.current_img_idx)
                is_hover = (x + 4 <= studio.mouse_pos[0] <= x + w - 4 and iy1 <= studio.mouse_pos[1] <= iy2)

                if is_selected:
                    row_bg = (48, 42, 28)
                    border_c = (0, 220, 255)
                elif is_hover:
                    row_bg = (35, 38, 48)
                    border_c = (55, 60, 75)
                else:
                    row_bg = (25, 27, 34)
                    border_c = (38, 40, 50)

                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), row_bg, -1)
                cv2.rectangle(canvas, (x + 6, iy1), (x + w - 6, iy2), border_c, 1)

                btn_id = f"SELECT_FRAME_{orig_img_idx}"
                studio.gui_buttons.append((btn_id, (x + 6, iy1, x + w - 6, iy2), orig_img_idx))

                # 状态小圆点
                dot_y = iy1 + item_h // 2
                is_excl = meta.get("is_excluded", False)
                if is_excl:
                    dot_c = (0, 0, 240)
                elif meta.get("mean_err", 0.0) > 0.5:
                    dot_c = (0, 180, 255)
                else:
                    dot_c = (0, 230, 80)
                cv2.circle(canvas, (x + 18, dot_y), 4, dot_c, -1)

                # 文件名
                txt_col = (255, 255, 255) if is_selected else (200, 200, 200)
                short_name = bname if len(bname) <= 15 else bname[:12] + "..."
                put_text(canvas, short_name, (x + 28, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

                # Tag 计数
                tag_cnt = meta.get("tag_count", 0)
                t_str = f"{tag_cnt}T"
                put_text(canvas, t_str, (x + 168, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

                # 平均残差数值与降幅
                if is_excl:
                    put_text(canvas, "EXCL", (x + 218, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 240), 1, cv2.LINE_AA)
                else:
                    err_val = meta.get('mean_err', 0.0)
                    err_str = f"{err_val:.2f}px"
                    err_col = (0, 200, 255) if err_val > 0.5 else (0, 240, 100)
                    put_text(canvas, err_str, (x + 218, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, err_col, 1, cv2.LINE_AA)
