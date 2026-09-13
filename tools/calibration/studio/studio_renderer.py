"""
AprilTag 离线标定工作站 - UI 界面渲染器 (StudioUIRenderer)
================================================================================
负责工作站现代深色全景三栏界面的几何排版与所有视觉元素绘制：
1. 顶栏 (Top Navigation Bar: 状态指标与全局 RMSE)
2. 底栏 (Bottom Control Toolbar: 现代圆角微质感快捷动作按钮)
3. 左栏 (高信息密度垂直紧凑帧列表 + 双下拉菜单过滤与排序)
4. 中栏视口 (自适应平移缩放、ROI 裁剪、3D 双棱柱与残差矢量投影)
5. 右栏 (180px 瘦身属性面板、逐 Tag 剔除打叉、单帧病因切片诊断)
6. 浮层 (科技感居中异步 BA 双轨进度卡片、Toast 提示、置顶下拉菜单)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.viewport_manager import draw_styled_button


# 下拉菜单选项定义
VIEW_MODE_OPTIONS = [
    ("3d", "3D 双四棱柱对比"),
    ("2d", "2D 识别框与残差矢量"),
    ("mix", "混合透视模式")
]

FILTER_MODE_OPTIONS = [
    ("all", "全部帧"),
    ("warning", "高残差 (>0.5px)"),
    ("excluded", "已剔除帧")
]

SORT_MODE_OPTIONS = [
    ("name_asc", "文件名升序"),
    ("err_desc", "残差降序 (最差优先 ↓)"),
    ("err_asc", "残差升序 (最优优先 ↑)"),
    ("tags_desc", "标靶数量降序")
]

BA_VIEW_OPTIONS = [
    ("3d", "3D 翡翠绿棱柱"),
    ("2d", "2D 理论投影框"),
    ("off", "隐藏 (关闭显示)")
]

OBS_VIEW_OPTIONS = [
    ("3d", "3D 科技天蓝棱柱"),
    ("2d", "2D 实测识别框"),
    ("off", "隐藏 (关闭显示)")
]


def draw_dropdown_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    is_open: bool,
    mouse_pos: Tuple[int, int],
    prefix: str = ""
):
    """绘制现代扁平化微质感下拉菜单头部按钮"""
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

    if is_open:
        bg_col = (48, 56, 72)
        border_col = (0, 220, 255)
        text_col = (255, 255, 255)
        arrow = "▲"
    elif is_hover:
        bg_col = (36, 40, 52)
        border_col = (0, 180, 220)
        text_col = (240, 240, 240)
        arrow = "▼"
    else:
        bg_col = (28, 30, 38)
        border_col = (55, 60, 75)
        text_col = (200, 200, 200)
        arrow = "▼"

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

    display_txt = f"{prefix}{label} {arrow}" if prefix else f"{label} {arrow}"
    (tw, th), _ = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    tx = x1 + max(6, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(canvas, display_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)


class StudioUIRenderer:
    """Offline Studio UI 渲染器"""

    def __init__(self):
        pass

    def render(self, studio: Any, canvas: np.ndarray):
        """完整渲染 Offline Studio 顶栏、左栏、中栏、右栏与底栏"""
        w, h = studio.win_w, studio.win_h
        top_h = studio.viewport.top_bar_h
        bot_h = studio.viewport.bottom_bar_h
        studio.gui_buttons.clear()

        # 异步 BA 结果轮询
        ba_res = studio.ba_runner.poll_result()
        if ba_res is not None:
            succ, msg = ba_res
            studio.set_toast(msg)
            if succ:
                studio.refresh_all_frame_metrics()

        # 1. 顶栏
        self.render_top_bar(studio, canvas, w, top_h)

        # 2. 底栏
        self.render_bottom_toolbar(studio, canvas, w, h, bot_h)

        # 3. 工作区尺寸
        content_y1 = top_h
        content_y2 = h - bot_h
        content_h = content_y2 - content_y1

        # 左栏：紧凑帧序列列表
        self.render_left_frame_list(studio, canvas, 0, content_y1, studio.left_bar_w, content_h)

        # 右栏：180px 瘦身属性与单帧诊断面板
        self.render_right_inspector(studio, canvas, w - studio.right_bar_w, content_y1, studio.right_bar_w, content_h)

        # 中栏：视口
        mid_x1 = studio.left_bar_w
        mid_w = w - studio.left_bar_w - studio.right_bar_w
        self.render_center_viewport(studio, canvas, mid_x1, content_y1, mid_w, content_h)

        # 4. 居中展示异步 BA 运行中进度卡片
        if studio.is_ba_running:
            self.render_ba_loading_card(studio, canvas, w, h)

        # 5. Toast 浮层
        if time.time() - studio.status_toast_time < 3.0 and studio.status_toast:
            self.render_toast(studio, canvas, w, h, bot_h)

        # 6. 置顶悬浮下拉列表
        if studio.active_dropdown and studio.active_dropdown in studio.dropdown_boxes:
            dd_info = studio.dropdown_boxes[studio.active_dropdown]
            self.render_dropdown_popup(studio, canvas, studio.active_dropdown,
                                      dd_info["rect"], dd_info["options"], dd_info["active_key"])

    def render_top_bar(self, studio: Any, canvas: np.ndarray, w: int, top_h: int):
        cv2.rectangle(canvas, (0, 0), (w, top_h), (24, 26, 32), -1)
        cv2.line(canvas, (0, top_h), (w, top_h), (55, 60, 72), 1)

        cv2.putText(canvas, "OFFLINE STUDIO", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "| AprilTag 离线标定与空间建图综合工作站", (185, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        tag_num = len(studio.tags_map_data.get("tags", {}))
        stat_txt = f"采图集: {len(studio.image_files)} 帧  |  已知标靶: {tag_num} 个  |  全局 RMSE: {studio.global_rmse:.2f} px"
        (tw, _), _ = cv2.getTextSize(stat_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(canvas, stat_txt, (w - tw - 20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1, cv2.LINE_AA)

    def render_bottom_toolbar(self, studio: Any, canvas: np.ndarray, w: int, h: int, bot_h: int):
        y1 = h - bot_h
        cv2.rectangle(canvas, (0, y1), (w, h), (20, 22, 28), -1)
        cv2.line(canvas, (0, y1), (w, y1), (60, 65, 78), 1)

        btn_y_top = y1 + 8
        btn_y_bot = h - 8
        bx = 16
        mx, my = studio.mouse_pos

        # [B] 全局平差
        ba_w = 135
        ba_type = "warning" if studio.is_ba_running else "primary"
        ba_txt = "正在平差..." if studio.is_ba_running else "全局平差 (B)"
        draw_styled_button(canvas, (bx, btn_y_top, bx + ba_w, btn_y_bot), ba_txt,
                           mouse_pos=(mx, my), btn_type=ba_type)
        studio.gui_buttons.append(("RUN_BA", (bx, btn_y_top, bx + ba_w, btn_y_bot), "RUN_BA"))
        bx += ba_w + 10

        # [P] 全量精度体检重算
        p_w = 125
        draw_styled_button(canvas, (bx, btn_y_top, bx + p_w, btn_y_bot), "全量体检 (P)",
                           mouse_pos=(mx, my), btn_type="normal")
        studio.gui_buttons.append(("RECOMPUTE_METRICS", (bx, btn_y_top, bx + p_w, btn_y_bot), "RECOMPUTE_METRICS"))
        bx += p_w + 10

        # [R] 导出质检报告
        r_w = 115
        draw_styled_button(canvas, (bx, btn_y_top, bx + r_w, btn_y_bot), "导出报告 (R)",
                           mouse_pos=(mx, my), btn_type="normal")
        studio.gui_buttons.append(("EXPORT_REPORT", (bx, btn_y_top, bx + r_w, btn_y_bot), "EXPORT_REPORT"))
        bx += r_w + 10

        # [S] 保存/发布地图
        s_w = 115
        draw_styled_button(canvas, (bx, btn_y_top, bx + s_w, btn_y_bot), "保存地图 (S)",
                           mouse_pos=(mx, my), btn_type="success")
        studio.gui_buttons.append(("SAVE_MAP", (bx, btn_y_top, bx + s_w, btn_y_bot), "SAVE_MAP"))
        bx += s_w + 10

        # 右侧 [Q] 退出工作台
        exit_w = 90
        exit_x1 = w - exit_w - 16
        draw_styled_button(canvas, (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "退出 (Q)",
                           mouse_pos=(mx, my), btn_type="danger")
        studio.gui_buttons.append(("EXIT", (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "EXIT"))

    def render_left_frame_list(self, studio: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """左栏：高信息密度垂直紧凑帧列表"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x + w, y), (x + w, y + h), (50, 54, 66), 1)

        header_h = 36
        dd_y1 = y + 6
        dd_y2 = y + header_h
        dd1_w = (w - 24) // 2
        dd1_x1 = x + 8
        dd1_x2 = dd1_x1 + dd1_w
        dd2_x1 = dd1_x2 + 8
        dd2_x2 = x + w - 8

        # 1. 筛选范围下拉框
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

        # 2. 排序方式下拉框
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

        list_y = y + header_h + 8
        item_h = 36
        visible_count = (h - header_h - 16) // item_h

        filtered_indices = studio._get_filtered_indices()
        if not filtered_indices:
            cv2.putText(canvas, "当前筛选条件下无图像", (x + 60, list_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1, cv2.LINE_AA)
            return

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

            # 行底色
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
            short_name = bname if len(bname) <= 16 else bname[:13] + "..."
            cv2.putText(canvas, short_name, (x + 30, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_col, 1, cv2.LINE_AA)

            # Tag 计数
            tag_cnt = meta.get("tag_count", 0)
            t_str = f"{tag_cnt}T"
            cv2.putText(canvas, t_str, (x + 175, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

            # 平均残差数值
            if is_excl:
                cv2.putText(canvas, "EXCL", (x + 225, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 240), 1, cv2.LINE_AA)
            else:
                err_str = f"{meta.get('mean_err', 0.0):.2f}px"
                err_col = (0, 200, 255) if meta.get('mean_err', 0.0) > 0.5 else (0, 240, 100)
                cv2.putText(canvas, err_str, (x + 225, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, err_col, 1, cv2.LINE_AA)

    def render_center_viewport(self, studio: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """中栏：高清工作视口，等比居中自适应渲染"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (14, 15, 18), -1)

        if not studio.image_files:
            cv2.putText(canvas, "未扫描到采图图像 (data/tag_calibration_images/ 为空)", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
            return

        cur_file = studio.image_files[studio.current_img_idx]
        bgr = cv2.imread(cur_file)
        if bgr is None:
            cv2.putText(canvas, f"读取图像文件失败: {cur_file}", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            return

        disp_frame = bgr.copy()
        base_name = os.path.basename(cur_file)
        meta = studio.frame_metrics_cache.get(base_name, {})
        obs_list = meta.get("observations", [])

        # 叠加标靶与 3D 双棱柱
        self.overlay_visual_elements(studio, disp_frame, obs_list, meta.get("is_excluded", False), meta=meta)

        # 视口等比与平移缩放渲染 (委托给 viewport 控制器)
        frame_h, frame_w = disp_frame.shape[:2]
        rois = studio.viewport.compute_viewport_render_rois((x, y, w, h), frame_w, frame_h)
        if rois is not None:
            (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = rois
            src_roi = disp_frame[src_y1:src_y2, src_x1:src_x2]
            dst_w = dst_x2 - dst_x1
            dst_h = dst_y2 - dst_y1
            if dst_w > 0 and dst_h > 0 and src_roi.size > 0:
                interp = cv2.INTER_LINEAR if studio.zoom_level > 1.2 else cv2.INTER_AREA
                resized_roi = cv2.resize(src_roi, (dst_w, dst_h), interpolation=interp)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized_roi

        # 视口外边框
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 60, 70), 1)

        # 视口左上角：双独立下拉菜单 (BA 理论值控制 + 单帧实测值控制)
        ba_x1 = x + 12
        ba_y1 = y + 10
        ba_x2 = ba_x1 + 148
        ba_y2 = ba_y1 + 28
        cur_ba_label = dict(BA_VIEW_OPTIONS).get(studio.ba_view_mode, "3D 翡翠绿棱柱")
        is_ba_open = (studio.active_dropdown == "BA_VIEW_DROPDOWN")
        draw_dropdown_button(canvas, (ba_x1, ba_y1, ba_x2, ba_y2), cur_ba_label,
                             is_open=is_ba_open, mouse_pos=studio.mouse_pos, prefix="BA理论: ")
        studio.dropdown_boxes["BA_VIEW_DROPDOWN"] = {
            "rect": (ba_x1, ba_y1, ba_x2, ba_y2),
            "options": BA_VIEW_OPTIONS,
            "active_key": studio.ba_view_mode
        }
        studio.gui_buttons.append(("TOGGLE_BA_VIEW_DROPDOWN", (ba_x1, ba_y1, ba_x2, ba_y2), "BA_VIEW_DROPDOWN"))

        obs_x1 = ba_x2 + 8
        obs_y1 = y + 10
        obs_x2 = obs_x1 + 148
        obs_y2 = obs_y1 + 28
        cur_obs_label = dict(OBS_VIEW_OPTIONS).get(studio.obs_view_mode, "3D 科技天蓝棱柱")
        is_obs_open = (studio.active_dropdown == "OBS_VIEW_DROPDOWN")
        draw_dropdown_button(canvas, (obs_x1, obs_y1, obs_x2, obs_y2), cur_obs_label,
                             is_open=is_obs_open, mouse_pos=studio.mouse_pos, prefix="实测识别: ")
        studio.dropdown_boxes["OBS_VIEW_DROPDOWN"] = {
            "rect": (obs_x1, obs_y1, obs_x2, obs_y2),
            "options": OBS_VIEW_OPTIONS,
            "active_key": studio.obs_view_mode
        }
        studio.gui_buttons.append(("TOGGLE_OBS_VIEW_DROPDOWN", (obs_x1, obs_y1, obs_x2, obs_y2), "OBS_VIEW_DROPDOWN"))

        # 视口右上角悬浮提示胶囊
        zoom_badge = f"缩放: {studio.zoom_level:.1f}x | 点击Tag: 剔除/恢复(打叉) | 切换模式: V | 拖拽: 右键/中键 | 双击/Z: 重置"
        (zw, zh), _ = cv2.getTextSize(zoom_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        bx1 = x + w - zw - 24
        by1 = y + 10
        bx2 = bx1 + zw + 14
        by2 = by1 + zh + 10
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 24, 32), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (70, 75, 88), 1)
        cv2.putText(canvas, zoom_badge, (bx1 + 7, by1 + zh + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 210, 230), 1, cv2.LINE_AA)

    def overlay_visual_elements(
        self,
        studio: Any,
        disp_frame: np.ndarray,
        observations: List[Dict[str, Any]],
        is_frame_excluded: bool,
        meta: Optional[Dict[str, Any]] = None
    ):
        """依据 ba_view_mode 与 obs_view_mode 双独立维度解耦渲染，剔除标靶显著打红叉"""
        ba_mode = studio.ba_view_mode
        obs_mode = studio.obs_view_mode

        obj_pts = []
        img_pts = []
        valid_obs = []

        for obs in observations:
            tid = obs["tag_id"]
            pts = np.array(obs["corners"], dtype=np.int32).reshape((-1, 2))
            keep = obs.get("keep", True) and not is_frame_excluded

            # 1. 剔除状态下在标靶上绘制鲜红显著的大叉号 (打叉审核模式)
            if not keep:
                cv2.line(disp_frame, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0, 0, 235), 3, cv2.LINE_AA)
                cv2.line(disp_frame, (pts[1][0], pts[1][1]), (pts[3][0], pts[3][1]), (0, 0, 235), 3, cv2.LINE_AA)
                cv2.polylines(disp_frame, [pts], isClosed=True, color=(40, 40, 180), thickness=2, lineType=cv2.LINE_AA)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.putText(disp_frame, f"Tag #{tid} [EXCL]", (cx - 42, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 240), 2, cv2.LINE_AA)
                continue

            # 2. 正常保留状态：根据 obs_view_mode 绘制 2D 实测角点多边形与标牌
            if obs_mode == "2d":
                cv2.polylines(disp_frame, [pts], isClosed=True, color=(0, 230, 80), thickness=2, lineType=cv2.LINE_AA)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.putText(disp_frame, f"Tag #{tid}", (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 80), 2, cv2.LINE_AA)

            # 收集参与三维解算的已知标靶
            w_c = studio.get_tag_world_corners(tid)
            if w_c is not None:
                obj_pts.append(w_c)
                img_pts.append(np.array(obs["corners"], dtype=np.float64))
                valid_obs.append(obs)

        # 3. 3D 棱柱与残差矢量投影
        if len(obj_pts) >= 1:
            obj_flat = np.concatenate(obj_pts, axis=0)
            img_flat = np.concatenate(img_pts, axis=0)
            rvec, tvec, success = studio.engine.solve_pnp(obj_flat, img_flat)
            if success:
                need_3d = (ba_mode == "3d" or obs_mode == "3d")
                if need_3d:
                    for obs in valid_obs:
                        tid = obs["tag_id"]
                        T_w_t = studio.get_tag_transform(tid)

                        # 计算 BA 理论世界位姿 (翡翠绿)
                        r_tag, t_tag = None, None
                        if ba_mode == "3d" and T_w_t is not None:
                            R_c_w, _ = cv2.Rodrigues(rvec)
                            T_c_w = np.eye(4, dtype=np.float64)
                            T_c_w[:3, :3] = R_c_w
                            T_c_w[:3, 3] = tvec.flatten()
                            T_c_t = T_c_w @ T_w_t
                            r_tag, _ = cv2.Rodrigues(T_c_t[:3, :3])
                            t_tag = T_c_t[:3, 3].reshape((3, 1))

                        # 计算单标靶本地实测位姿 (用于科技天蓝 OBS 棱柱)
                        obs_r, obs_t = None, None
                        c_arr = np.array(obs["corners"], dtype=np.float64).reshape((4, 2))
                        succ_single = False
                        if obs_mode == "3d":
                            succ_single, obs_r, obs_t = studio.engine.solve_single_tag_pnp(c_arr)

                        err_mm = 0.0
                        if t_tag is not None and succ_single and obs_t is not None:
                            err_mm = float(np.linalg.norm(t_tag - obs_t))
                        err_px = (meta or {}).get("tag_errors", {}).get(tid, 0.2)

                        studio.visualizer.render_tag_dual_prisms(
                            img=disp_frame,
                            ba_rvec=r_tag if ba_mode == "3d" else None,
                            ba_tvec=t_tag if ba_mode == "3d" else None,
                            obs_rvec=obs_r if (obs_mode == "3d" and succ_single) else None,
                            obs_tvec=obs_t if (obs_mode == "3d" and succ_single) else None,
                            tag_id=tid,
                            err_px=err_px,
                            err_mm=err_mm,
                            observed_corners=c_arr
                        )

                # 4. 2D 理论重投影框与残差矢量
                proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, studio.engine.camera_matrix, studio.engine.dist_coeffs)
                proj_flat = proj_pts.reshape((-1, 2))

                if ba_mode == "2d":
                    for i in range(len(valid_obs)):
                        p4 = proj_flat[i * 4:(i + 1) * 4].astype(np.int32)
                        cv2.polylines(disp_frame, [p4], isClosed=True, color=(0, 210, 255), thickness=1, lineType=cv2.LINE_AA)

                if (ba_mode != "off" and obs_mode != "off") and (ba_mode == "2d" or obs_mode == "2d"):
                    if hasattr(studio.visualizer, "draw_reprojection_vectors"):
                        studio.visualizer.draw_reprojection_vectors(disp_frame, img_flat, proj_flat, scale_factor=40.0)

    def render_right_inspector(self, studio: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """右栏：精简瘦身属性与标靶残差清单面板 (瘦身宽度: 180px)"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x, y), (x, y + h), (50, 54, 66), 1)

        if not studio.image_files:
            return

        cur_file = studio.image_files[studio.current_img_idx]
        bname = os.path.basename(cur_file)
        meta = studio.frame_metrics_cache.get(bname, {})

        # 1. 顶部当前帧摘要卡片
        cv2.putText(canvas, bname, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        # 状态切换按钮 (保留 / 剔除)
        is_excl = meta.get("is_excluded", False)
        b_type = "danger" if is_excl else "success"
        b_label = "恢复此帧 (T)" if is_excl else "剔除此帧 (T)"
        draw_styled_button(canvas, (x + 8, y + 30, x + w - 8, y + 56), b_label,
                           mouse_pos=studio.mouse_pos, btn_type=b_type)
        studio.gui_buttons.append(("TOGGLE_FRAME_STATUS", (x + 8, y + 30, x + w - 8, y + 56), bname))

        # 2. 标靶细目清单 (精简高信息密度表格)
        list_y = y + 66
        cv2.line(canvas, (x + 8, list_y), (x + w - 8, list_y), (45, 48, 58), 1)
        obs_list = meta.get("observations", [])
        tag_errors = meta.get("tag_errors", {})
        cv2.putText(canvas, f"标靶与残差 ({len(obs_list)})", (x + 8, list_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 185, 195), 1, cv2.LINE_AA)

        row_y = list_y + 24
        row_h = 24
        for obs in obs_list:
            tid = obs["tag_id"]
            keep = obs.get("keep", True)
            err_val = tag_errors.get(tid, 0.0)

            rx1, ry1, rx2, ry2 = x + 6, row_y, x + w - 6, row_y + row_h - 2
            is_hover = (rx1 <= studio.mouse_pos[0] <= rx2 and ry1 <= studio.mouse_pos[1] <= ry2)
            bg_col = (34, 38, 48) if is_hover else (26, 28, 36)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), bg_col, -1)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (48, 52, 64), 1)
            studio.gui_buttons.append((f"TOGGLE_TAG_{tid}", (rx1, ry1, rx2, ry2), tid))

            dot_c = (0, 220, 80) if keep else (0, 0, 220)
            cv2.circle(canvas, (rx1 + 10, ry1 + 11), 3, dot_c, -1)

            t_col = (230, 230, 230) if keep else (120, 120, 120)
            cv2.putText(canvas, f"#{tid}", (rx1 + 18, ry1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_col, 1, cv2.LINE_AA)

            err_str = f"{err_val:.2f}px" if keep else "EXCL"
            err_c = (120, 120, 120) if not keep else ((0, 200, 255) if err_val > 0.5 else (0, 230, 80))
            (ew, _), _ = cv2.getTextSize(err_str, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            cv2.putText(canvas, err_str, (rx2 - ew - 6, ry1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, err_c, 1, cv2.LINE_AA)

            row_y += row_h
            if row_y > y + h - 80:
                break

        # 3. 底部紧凑快捷动作
        diag_y = y + h - 68
        cv2.line(canvas, (x + 8, diag_y), (x + w - 8, diag_y), (45, 48, 58), 1)

        draw_styled_button(canvas, (x + 8, diag_y + 8, x + w - 8, diag_y + 32), "超精提取 (E)",
                           mouse_pos=studio.mouse_pos, btn_type="primary")
        studio.gui_buttons.append(("SUPER_EXTRACT_FRAME", (x + 8, diag_y + 8, x + w - 8, diag_y + 32), bname))

        draw_styled_button(canvas, (x + 8, diag_y + 36, x + w - 8, diag_y + 60), "病因诊断 (D)",
                           mouse_pos=studio.mouse_pos, btn_type="warning")
        studio.gui_buttons.append(("DIAGNOSE_FRAME", (x + 8, diag_y + 36, x + w - 8, diag_y + 60), bname))

    def render_ba_loading_card(self, studio: Any, canvas: np.ndarray, w: int, h: int):
        """居中展示异步 BA 全局平差双轨进度卡片 (大阶段主进度条 + 求解器子进度条与实时收敛指标)"""
        card_w, card_h = 640, 162
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (0, 220, 255), 2)

        # 1. 主阶段总流程进度条
        pct = max(0.0, min(1.0, studio.ba_progress))
        pct_int = int(round(pct * 100))

        cv2.putText(canvas, "[BA] 全局平差整体收敛进度", (cx1 + 22, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA)

        bar_x1 = cx1 + 22
        bar_y1 = cy1 + 34
        bar_x2 = cx1 + card_w - 22
        bar_y2 = bar_y1 + 12
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (55, 62, 78), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 210, 255), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (180, 245, 255), 1)

        stage_txt = studio.ba_stage_text or "准备进入优化平差管线..."
        cv2.putText(canvas, stage_txt, (cx1 + 22, cy1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

        cv2.line(canvas, (cx1 + 22, cy1 + 74), (cx1 + card_w - 22, cy1 + 74), (45, 48, 60), 1)

        # 2. 求解器内部子阶段与迭代进度条
        sub_pct = max(0.0, min(1.0, studio.ba_sub_progress))
        sub_pct_int = int(round(sub_pct * 100))

        cv2.putText(canvas, "优化器实时迭代收敛监控 (Sub-Iteration)", (cx1 + 22, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 205, 215), 1, cv2.LINE_AA)
        sub_pct_str = f"{sub_pct_int}%"
        (spw, _), _ = cv2.getTextSize(sub_pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        cv2.putText(canvas, sub_pct_str, (cx1 + card_w - 22 - spw, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 180, 255), 2, cv2.LINE_AA)

        sbar_y1 = cy1 + 102
        sbar_y2 = sbar_y1 + 10
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (25, 28, 36), -1)
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (48, 54, 68), 1)

        sub_fill_w = int(bar_w * sub_pct)
        if sub_fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y2), (0, 160, 255), -1)
            cv2.line(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y1), (120, 220, 255), 1)

        sub_txt = studio.ba_sub_text or "等待当前阶段迭代步进推进..."
        cv2.putText(canvas, sub_txt, (cx1 + 22, cy1 + 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

    def render_toast(self, studio: Any, canvas: np.ndarray, w: int, h: int, bot_h: int):
        (tw, _), _ = cv2.getTextSize(studio.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        tx1 = (w - tw) // 2 - 16
        ty1 = h - bot_h - 46
        tx2 = tx1 + tw + 32
        ty2 = ty1 + 32

        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (120, 30, 100), -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (220, 60, 180), 1)
        cv2.putText(canvas, studio.status_toast, (tx1 + 16, ty1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    def render_dropdown_popup(
        self,
        studio: Any,
        canvas: np.ndarray,
        pop_name: str,
        rect: Tuple[int, int, int, int],
        options: List[Tuple[str, str]],
        active_key: str
    ):
        """置顶悬浮下拉列表浮层"""
        rx1, ry1, rx2, ry2 = rect
        item_h = 32
        pop_w = max(rx2 - rx1, 190)
        pop_h = len(options) * item_h + 6
        pop_x1 = rx1
        pop_y1 = ry2 + 2
        pop_x2 = pop_x1 + pop_w
        pop_y2 = pop_y1 + pop_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), (0, 200, 255), 1)

        mx, my = studio.mouse_pos
        for idx, (opt_key, opt_label) in enumerate(options):
            iy1 = pop_y1 + 3 + idx * item_h
            iy2 = iy1 + item_h - 1
            is_active = (opt_key == active_key)
            is_hover = (pop_x1 <= mx <= pop_x2 and iy1 <= my <= iy2)

            if is_active:
                row_bg = (52, 45, 20)
                txt_col = (0, 230, 255)
            elif is_hover:
                row_bg = (36, 42, 56)
                txt_col = (255, 255, 255)
            else:
                row_bg = (22, 25, 32)
                txt_col = (190, 190, 190)

            cv2.rectangle(canvas, (pop_x1 + 3, iy1), (pop_x2 - 3, iy2), row_bg, -1)
            prefix = "✔ " if is_active else "  "
            (tw, th), _ = cv2.getTextSize(prefix + opt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(canvas, prefix + opt_label, (pop_x1 + 8, iy1 + (item_h + th) // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

            btn_id = f"DD_SELECT_{studio.active_dropdown}_{opt_key}"
            studio.gui_buttons.append((btn_id, (pop_x1, iy1, pop_x2, iy2), (studio.active_dropdown, opt_key)))
