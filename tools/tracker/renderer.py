#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot 在线跟踪 - 渲染器:
工具栏 / 下拉浮层 / Studio 同款棱柱叠加 / XY 平面网格 / 信息面板 / Toast。
只读主控制器 RobotOnlineTracker 的状态并绘制, 不修改业务状态;
按钮命中表 (buttons) 每帧由 draw_toolbar 重建, 供主控制器鼠标分发使用。
"""

import os
import time

import cv2
import numpy as np

from tools.tracker.common import (
    COLOR_ACCENT, COLOR_BG, COLOR_BORDER, COLOR_BORDER_SEL, COLOR_CARD_BG,
    COLOR_CARD_SEL, COLOR_TEXT_SUB, COL_BLUE, COL_CYAN, COL_GRAY, COL_GREEN,
    COL_PANEL_BG, COL_PANEL_EDGE, COL_RED, COL_WHITE, COL_YELLOW, TOOLBAR_H,
    _PRISM_PTS, _tag_local_frame, draw_text, fmt_point)


class TrackerRenderer:
    """Robot 在线跟踪渲染器: 输入主控制器状态, 输出画布与按钮命中表"""

    def __init__(self, tracker):
        self.tr = tracker               # 主控制器状态引用 (只读)
        self.buttons = []               # [(btn_id, (x1,y1,x2,y2), payload), ...] 每帧重建
        self.mouse_pos = (-1, -1)
        self._camera_type_rect = None
        self._resolution_rect = None
        self._plane_z_rect = None

    # ------------------------------ 鼠标辅助 ------------------------------
    def on_mouse_move(self, x, y):
        self.mouse_pos = (x, y)

    def hit_test(self, x, y):
        """命中检测: 返回 (btn_id, payload) 或 None"""
        for btn_id, rect, payload in self.buttons:
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return btn_id, payload
        return None

    # ------------------------------ 工具栏 ------------------------------
    def _draw_dropdown_button(self, canvas, rect, label, is_open):
        """扁平化下拉按钮 (借鉴 d435_viewer / studio_renderer)"""
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)
        if is_open:
            bg_col, border_col, text_col = (48, 56, 72), COLOR_BORDER_SEL, COL_WHITE
            arrow = "▲"
        elif is_hover:
            bg_col, border_col, text_col = (36, 40, 52), (0, 180, 220), COL_WHITE
            arrow = "▼"
        else:
            bg_col, border_col, text_col = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB
            arrow = "▼"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)
        draw_text(canvas, f"{label} {arrow}", (x1 + 8, y1 + (y2 - y1) // 2 - 8), 14, text_col)

    def _render_dropdown_popup(self, canvas, rect, options, active_key, btn_prefix):
        """置顶悬浮下拉列表浮层 (借鉴 d435_viewer)"""
        rx1, ry1, rx2, ry2 = rect
        item_h = 30
        pop_w = max(rx2 - rx1, 210)
        pop_x1, pop_y1 = rx1, ry2 + 2
        pop_x2, pop_y2 = pop_x1 + pop_w, pop_y1 + len(options) * item_h + 6
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (30, 34, 42), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), COLOR_BORDER_SEL, 1)

        for i, (key, label) in enumerate(options):
            iy1 = pop_y1 + 3 + i * item_h
            iy2 = iy1 + item_h
            is_active = (key == active_key)
            if is_active:
                cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), COLOR_CARD_SEL, -1)
            draw_text(canvas, label, (pop_x1 + 10, iy1 + (item_h - 16) // 2 - 2), 15,
                      COLOR_ACCENT if is_active else COL_WHITE, bold=is_active)
            self.buttons.append((f"{btn_prefix}{i}", (pop_x1 + 2, iy1, pop_x2 - 2, iy2), key))

    def draw_toolbar(self, canvas):
        """顶部工具栏: 相机类型 ▼ | 分辨率 ▼ | 开启/关闭 | 识别 | 确定世界坐标系 | XY平面 ▼ | 显示已知Tag | 识别 Tag 2 | 连接机械臂 | 跟踪 | 退出 X"""
        tr = self.tr
        tw = canvas.shape[1]
        self.buttons = []
        cv2.rectangle(canvas, (0, 0), (tw, TOOLBAR_H), COLOR_BG, -1)
        cv2.line(canvas, (0, TOOLBAR_H - 1), (tw, TOOLBAR_H - 1), COLOR_BORDER, 1)

        y1, btn_h = 6, TOOLBAR_H - 12
        y2 = y1 + btn_h
        gap = 6

        # 1. 相机类型下拉 (最左)
        cam_x1, cam_x2 = 8, 8 + 150
        cam_label = dict(tr.camera.camera_options).get(tr.camera.camera_type, tr.camera.camera_type)
        self._draw_dropdown_button(canvas, (cam_x1, y1, cam_x2, y2), cam_label,
                                   is_open=(tr.active_dropdown == "CAMERA_TYPE_DROPDOWN"))
        self.buttons.append(("TOGGLE_CAM_DD", (cam_x1, y1, cam_x2, y2), "CAMERA_TYPE_DROPDOWN"))
        self._camera_type_rect = (cam_x1, y1, cam_x2, y2)

        # 2. 分辨率下拉
        res_x1 = cam_x2 + gap
        res_x2 = res_x1 + 110
        self._draw_dropdown_button(canvas, (res_x1, y1, res_x2, y2), tr.camera.resolution,
                                   is_open=(tr.active_dropdown == "RES_DROPDOWN"))
        self.buttons.append(("TOGGLE_RES_DD", (res_x1, y1, res_x2, y2), "RES_DROPDOWN"))
        self._resolution_rect = (res_x1, y1, res_x2, y2)

        # 3. 开启/关闭乒乓按钮
        sw_x1 = res_x2 + gap
        sw_x2 = sw_x1 + 70
        if tr.camera.pipeline_running:
            sw_bg, sw_border, sw_txt, sw_label = (55, 45, 30), (255, 160, 40), (255, 200, 80), "关闭"
        else:
            sw_bg, sw_border, sw_txt, sw_label = (30, 50, 40), (0, 200, 120), (80, 230, 160), "开启"
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_bg, -1)
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_border, 1)
        draw_text(canvas, sw_label, (sw_x1 + 22, y1 + (btn_h - 16) // 2 - 1), 15, sw_txt, True)
        self.buttons.append(("TOGGLE_CAMERA", (sw_x1, y1, sw_x2, y2), None))

        # 4. "识别" 一键单帧闭环按钮: 开相机→拍一张→关相机→识别Tag→定世界系→蓝/绿棱柱
        rc_x1 = sw_x2 + gap
        rc_x2 = rc_x1 + 76
        if tr.recognizing:
            rc_label = tr.recog_stage or "识别中..."
            rc_bg, rc_border, rc_txt = COLOR_CARD_BG, (255, 160, 40), (255, 200, 80)
        elif tr.static_frame is not None:
            rc_label = "识别"
            rc_bg, rc_border, rc_txt = COLOR_CARD_SEL, COLOR_BORDER_SEL, COLOR_ACCENT
        else:
            rc_label = "识别"
            rc_bg, rc_border, rc_txt = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB
        cv2.rectangle(canvas, (rc_x1, y1), (rc_x2, y2), rc_bg, -1)
        cv2.rectangle(canvas, (rc_x1, y1), (rc_x2, y2), rc_border, 1)
        draw_text(canvas, rc_label,
                  (rc_x1 + (8 if tr.recognizing else 20), y1 + (btn_h - 16) // 2 - 1),
                  14, rc_txt, True)
        self.buttons.append(("TRIGGER_RECOG", (rc_x1, y1, rc_x2, y2), None))

        # 5. "确定世界坐标系" 一键流程按钮 (FR-12.5)
        lk_x1 = rc_x2 + gap
        lk_x2 = lk_x1 + 130
        if tr.sampling:
            lk_label, lk_border, lk_txt = (tr.sample_stage or "采样中..."), (255, 160, 40), (255, 200, 80)
        elif tr.world_locked:
            lk_label, lk_border, lk_txt = "解除锁定", COLOR_BORDER_SEL, COLOR_ACCENT
        else:
            lk_label, lk_border, lk_txt = "确定世界坐标系", COLOR_BORDER, COLOR_TEXT_SUB
        cv2.rectangle(canvas, (lk_x1, y1), (lk_x2, y2), COLOR_CARD_SEL if tr.world_locked else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (lk_x1, y1), (lk_x2, y2), lk_border, 1)
        draw_text(canvas, lk_label, (lk_x1 + 8, y1 + (btn_h - 16) // 2 - 1), 14, lk_txt, True)
        self.buttons.append(("TOGGLE_LOCK", (lk_x1, y1, lk_x2, y2), None))

        # 6. "XY平面" 下拉框: 合并原[绘制XY平面]开关与[Z高度]下拉, 最高项"不绘制", 其余为绘制高度
        pl_x1 = lk_x2 + gap
        pl_x2 = pl_x1 + 110
        pl_label = f"Z={tr.plane_z}" if tr.show_xy_plane_on else "XY平面"
        self._draw_dropdown_button(canvas, (pl_x1, y1, pl_x2, y2), pl_label,
                                   is_open=(tr.active_dropdown == "PLANE_DROPDOWN"))
        self.buttons.append(("TOGGLE_PLANE_DD", (pl_x1, y1, pl_x2, y2), "PLANE_DROPDOWN"))
        self._plane_z_rect = (pl_x1, y1, pl_x2, y2)

        # 7. "显示已知 Tag" 乒乓开关 (绿色=BA理论位置 / 蓝色=当帧实测位置)
        an_x1 = pl_x2 + gap
        an_x2 = an_x1 + 120
        cv2.rectangle(canvas, (an_x1, y1), (an_x2, y2),
                      (30, 50, 40) if tr.show_anchors_on else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (an_x1, y1), (an_x2, y2),
                      (0, 200, 120) if tr.show_anchors_on else COLOR_BORDER, 1)
        draw_text(canvas, "显示已知Tag", (an_x1 + 12, y1 + (btn_h - 16) // 2 - 1), 14,
                  (80, 230, 160) if tr.show_anchors_on else COLOR_TEXT_SUB, tr.show_anchors_on)
        self.buttons.append(("TOGGLE_ANCHORS", (an_x1, y1, an_x2, y2), None))

        # 8. "识别 Tag 2" 乒乓开关 (默认关, FR-12.6)
        rg_x1 = an_x2 + gap
        rg_x2 = rg_x1 + 110
        cv2.rectangle(canvas, (rg_x1, y1), (rg_x2, y2),
                      (30, 50, 40) if tr.recog_tag2_on else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (rg_x1, y1), (rg_x2, y2),
                      (0, 200, 120) if tr.recog_tag2_on else COLOR_BORDER, 1)
        draw_text(canvas, "识别 Tag 2", (rg_x1 + 14, y1 + (btn_h - 16) // 2 - 1), 14,
                  (80, 230, 160) if tr.recog_tag2_on else COLOR_TEXT_SUB, tr.recog_tag2_on)
        self.buttons.append(("TOGGLE_RECOG", (rg_x1, y1, rg_x2, y2), None))

        # 9. 机械臂按钮
        rb_x1 = rg_x2 + gap
        rb_x2 = rb_x1 + 110
        connected = tr.robot.is_connected
        rb_bg = COLOR_CARD_SEL if connected else COLOR_CARD_BG
        rb_border = COLOR_BORDER_SEL if connected else COLOR_BORDER
        rb_txt = COLOR_ACCENT if connected else COLOR_TEXT_SUB
        cv2.rectangle(canvas, (rb_x1, y1), (rb_x2, y2), rb_bg, -1)
        cv2.rectangle(canvas, (rb_x1, y1), (rb_x2, y2), rb_border, 1)
        draw_text(canvas, "断开机械臂" if connected else "连接机械臂",
                  (rb_x1 + 8, y1 + (btn_h - 16) // 2 - 1), 14, rb_txt, connected)
        self.buttons.append(("TOGGLE_ROBOT", (rb_x1, y1, rb_x2, y2), None))

        # 10. 跟踪按钮
        tk_x1 = rb_x2 + gap
        tk_x2 = tk_x1 + 90
        busy = tr.tracking
        cv2.rectangle(canvas, (tk_x1, y1), (tk_x2, y2), COLOR_CARD_SEL if busy else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (tk_x1, y1), (tk_x2, y2), (255, 160, 40) if busy else COLOR_BORDER, 1)
        draw_text(canvas, "跟踪中..." if busy else f"跟踪 Tag {tr.target_tag_id}",
                  (tk_x1 + 8, y1 + (btn_h - 16) // 2 - 1), 14,
                  (255, 200, 80) if busy else COL_GREEN, True)
        self.buttons.append(("TRIGGER_TRACK", (tk_x1, y1, tk_x2, y2), None))

        # 11. 退出按钮 (最右)
        exit_x1, exit_x2 = tw - 90, tw - 8
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2), (60, 60, 80), 1)
        draw_text(canvas, "退出 X", (exit_x1 + 14, y1 + (btn_h - 16) // 2 - 1), 14,
                  (190, 190, 200), True)
        self.buttons.append(("QUIT", (exit_x1, y1, exit_x2, y2), None))

        # 展开的下拉浮层
        if tr.active_dropdown == "CAMERA_TYPE_DROPDOWN" and self._camera_type_rect:
            self._render_dropdown_popup(canvas, self._camera_type_rect,
                                        tr.camera.camera_options, tr.camera.camera_type, "DD_CAM_")
        elif tr.active_dropdown == "RES_DROPDOWN" and self._resolution_rect:
            self._render_dropdown_popup(canvas, self._resolution_rect,
                                        tr.camera.resolution_options, tr.camera.resolution, "DD_RES_")
        elif tr.active_dropdown == "PLANE_DROPDOWN" and self._plane_z_rect:
            self._render_dropdown_popup(canvas, self._plane_z_rect,
                                        tr.plane_options,
                                        tr.plane_z if tr.show_xy_plane_on else None, "DD_PLANE_")

    # ------------------------------ 棱柱与叠加层 ------------------------------
    def _draw_studio_prism(self, canvas, rvec, tvec, is_theory, is_target=False):
        """Offline Studio 同款四棱柱 (VerificationVisualizer 参数一致):
        截面 30x30mm (半宽15) x 生长高度 75mm, 半透明填充 + 棱线描边 + 顶面中心点;
        is_theory=True 翡翠绿(BA理论) / False 科技天蓝(实测);
        目标 Tag 额外绘制底面中心点与中心生长轴 (至固定高顶面中心)。
        """
        K, dist = self.tr.engine.camera_matrix, self.tr.engine.dist_coeffs
        proj = cv2.projectPoints(_PRISM_PTS, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(int)
        b, t, tc = proj[0:4], proj[4:8], tuple(proj[8])
        if is_theory:
            side_c, cap_c, edge_c, top_c, dot_c = \
                (0, 185, 60), (50, 240, 100), (0, 255, 100), (120, 255, 160), (0, 255, 120)
        else:
            side_c, cap_c, edge_c, top_c, dot_c = \
                (235, 125, 20), (255, 175, 50), (255, 195, 70), (255, 255, 255), (255, 200, 60)
        overlay = canvas.copy()
        for i in range(4):
            j = (i + 1) % 4
            cv2.fillPoly(overlay, [np.array([b[i], b[j], t[j], t[i]], dtype=np.int32)], side_c)
        cv2.fillPoly(overlay, [t], cap_c)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.polylines(canvas, [b], True, edge_c, 2, cv2.LINE_AA)
        cv2.polylines(canvas, [t], True, top_c, 2, cv2.LINE_AA)
        for i in range(4):
            cv2.line(canvas, tuple(b[i]), tuple(t[i]), edge_c, 2, cv2.LINE_AA)
        cv2.circle(canvas, tc, 4, dot_c, -1, cv2.LINE_AA)      # 生长固定高后的顶面中心点
        if is_target:                                          # 目标 Tag: 底面中心点 + 中心生长轴
            pb = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec, K, dist)[0]
            pb = tuple(pb.reshape(2).astype(int))
            cv2.circle(canvas, pb, 4, dot_c, -1, cv2.LINE_AA)
            cv2.line(canvas, pb, tc, dot_c, 1, cv2.LINE_AA)

    def draw_overlay(self, canvas, det):
        """叠加层: 目标 Tag 绿色高亮框 + Studio 同款蓝色实测棱柱 (含底面/顶面中心点)"""
        tr = self.tr
        corners = det.get(tr.target_tag_id)
        if corners is None:
            return
        pts = corners.reshape((-1, 2)).astype(np.int32)
        cv2.polylines(canvas, [pts], True, COL_GREEN, 3, cv2.LINE_AA)
        if tr.recog_tag2_on or tr.show_anchors_on:
            # 目标 Tag 法向先验: 世界系已锁定时用"朝向天空"先验消除 IPPE 二义性 180° 翻转
            z_exp = None
            if tr.world_locked and tr.locked_rvec is not None:
                R_lock, _ = cv2.Rodrigues(tr.locked_rvec)
                z_exp = R_lock @ np.array([0.0, 0.0, 1.0])
            ok_b, rvec_b, tvec_b = tr.engine.solve_single_tag_pnp(corners, expected_z_cam=z_exp)
            if ok_b:
                self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False,
                                        is_target=True)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.drawMarker(canvas, (cx, cy), COL_GREEN, cv2.MARKER_CROSS, 18, 2)
        draw_text(canvas, f"Tag {tr.target_tag_id}", (cx + 12, cy - 24), 17, COL_GREEN, True)

    def draw_anchor_overlay(self, canvas, det):
        """已知标靶棱柱叠加 (与 Offline Studio 同款棱柱参数, 数据源=相机实时帧):
        绿色棱柱=BA 理论位姿棱柱; 蓝色棱柱=当帧实测单靶 PnP 位姿棱柱。
        位姿: 锁定后用锁定位姿, 未锁定用当帧锚定 PnP。
        """
        tr = self.tr
        if not tr.show_anchors_on:
            return
        # 位姿来源 (世界->相机): 优先锁定位姿, 未锁定用当帧锚定 PnP
        rvec, tvec = tr.get_world_pose(det)
        R, _ = cv2.Rodrigues(rvec) if rvec is not None else (None, None)
        t_flat = np.asarray(tvec).reshape(3) if tvec is not None else None

        def _world_tag_pose(wc):
            """世界角点 -> 标靶相机系位姿 (rvec, tvec)"""
            R_t, c_w = _tag_local_frame(wc)
            return cv2.Rodrigues(R @ R_t)[0], R @ c_w + t_flat

        for tid in tr.anchor_positions:
            wc = tr.engine.get_tag_world_corners(tid)
            if wc is None or R is None:
                continue
            rvec_t, tvec_t = _world_tag_pose(wc)
            if tvec_t[2] <= 1e-6:
                continue
            # 绿色理论棱柱 (Offline Studio 同款: 30x30 截面 x 75mm 生长高)
            self._draw_studio_prism(canvas, rvec_t, tvec_t, True)
            if tid in det:
                # 蓝色实测棱柱 (单靶 PnP 位姿); 传入地图理论法向, 消除 IPPE 平面二义性 180° 翻转
                R_exp, _ = cv2.Rodrigues(rvec_t)
                ok_b, rvec_b, tvec_b = tr.engine.solve_single_tag_pnp(
                    det[tid], expected_z_cam=R_exp[:, 2])
                if ok_b:
                    self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False)
                    c = det[tid].reshape(4, 2).mean(axis=0).astype(int)
                    draw_text(canvas, str(tid), (int(c[0]) + 8, int(c[1]) - 22), 14, COL_BLUE, True)
            else:
                # 未入镜的理论 Tag: 在标靶中心标注编号
                pc = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec_t, tvec_t,
                                       tr.engine.camera_matrix, tr.engine.dist_coeffs)[0]
                pc = pc.reshape(2).astype(int)
                draw_text(canvas, str(tid), (int(pc[0]) + 8, int(pc[1]) - 22), 14, COL_GREEN, True)

    def draw_recognition_overlay(self, canvas):
        """单帧识别结果叠加 (静态照片, 与 Offline Studio 同款棱柱参数):
        绿色棱柱 = 地图白名单全部 Tag 的 BA 理论位姿棱柱 (需已确定世界坐标系);
        蓝色棱柱 = 当帧识别 Tag 的单靶 PnP 位姿棱柱; 目标 Tag 附带底面/顶面中心点。
        """
        tr = self.tr
        R = t_flat = None
        if tr.world_locked and tr.locked_rvec is not None:
            R, _ = cv2.Rodrigues(tr.locked_rvec)
            t_flat = np.asarray(tr.locked_tvec, dtype=np.float64).reshape(3)
        K = tr.engine.camera_matrix

        def _world_tag_pose(wc):
            """世界角点 -> 标靶相机系位姿 (rvec, tvec)"""
            R_t, c_w = _tag_local_frame(wc)
            return cv2.Rodrigues(R @ R_t)[0], R @ c_w + t_flat

        # 1. 绿色理论棱柱: 地图白名单全部 Tag (含目标 Tag, 若在地图中)
        map_ids = sorted(set(tr.anchor_positions) |
                         ({tr.target_tag_id} if tr.theoretical is not None else set()))
        for tid in map_ids:
            wc = tr.engine.get_tag_world_corners(tid)
            if wc is None or R is None:
                continue
            rvec_t, tvec_t = _world_tag_pose(wc)
            if tvec_t[2] <= 1e-6:
                continue
            self._draw_studio_prism(canvas, rvec_t, tvec_t, True,
                                    is_target=(tid == tr.target_tag_id))
            if tid not in (tr.static_det or {}):
                # 未入镜的理论 Tag: 在标靶中心标注编号
                pc = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec_t, tvec_t, K,
                                       tr.engine.dist_coeffs)[0].reshape(2).astype(int)
                draw_text(canvas, str(tid), (int(pc[0]) + 8, int(pc[1]) - 22), 14, COL_GREEN, True)

        # 2. 蓝色实测棱柱: 当帧识别到的全部 Tag (单靶 PnP)
        # 法向先验: 地图内 Tag 用理论位姿法向, 地图外 Tag (如动态目标) 退用"朝向天空"先验
        sky_z = R @ np.array([0.0, 0.0, 1.0]) if R is not None else None
        for tid, corners in (tr.static_det or {}).items():
            z_exp = None
            if R is not None:
                wc = tr.engine.get_tag_world_corners(tid)
                if wc is not None:
                    R_t, _ = _tag_local_frame(wc)
                    z_exp = R @ R_t[:, 2]
                else:
                    z_exp = sky_z
            ok_b, rvec_b, tvec_b = tr.engine.solve_single_tag_pnp(corners, expected_z_cam=z_exp)
            if ok_b:
                self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False,
                                        is_target=(tid == tr.target_tag_id))
                c = corners.reshape(4, 2).mean(axis=0).astype(int)
                draw_text(canvas, str(tid), (int(c[0]) + 8, int(c[1]) - 22), 14, COL_BLUE, True)
            else:
                pts = corners.reshape((-1, 2)).astype(np.int32)
                cv2.polylines(canvas, [pts], True, COL_BLUE, 2, cv2.LINE_AA)

    def draw_xy_plane_overlay(self, canvas, det):
        """世界 XY 平面透视网格叠加 (高度由下拉框选择, 默认 Z=0 地面):
        两组互相垂直的平行线网格 + 三轴加粗高亮 (X红/Y绿/Z蓝) + 原点标记, 直观透视世界系。
        位姿: 优先锁定位姿, 未锁定用当帧锚定 PnP。
        """
        tr = self.tr
        if not tr.show_xy_plane_on:
            return
        rvec, tvec = tr.get_world_pose(det)
        if rvec is None:
            return
        R, _ = cv2.Rodrigues(rvec)
        t_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
        K = tr.engine.camera_matrix
        h_c, w_c = canvas.shape[:2]
        ext, step, z0 = tr.PLANE_EXTENT_MM, tr.PLANE_STEP_MM, float(tr.plane_z)

        def _project(p_w):
            p_cam = R @ np.asarray(p_w, dtype=np.float64).reshape(3) + t_flat
            if p_cam[2] <= 1e-6:
                return None
            uv = K @ p_cam
            u, v = int(uv[0] / uv[2]), int(uv[1] / uv[2])
            return (u, v) if (0 <= u < w_c and 0 <= v < h_c) else None

        def _seg(p0, p1, color, thick):
            """长线段沿线采样投影连线 (自动处理出画与近裁剪)"""
            prev = None
            for k in range(25):
                s = k / 24.0
                p = (p0[0] + (p1[0] - p0[0]) * s,
                     p0[1] + (p1[1] - p0[1]) * s,
                     p0[2] + (p1[2] - p0[2]) * s)
                uv = _project(p)
                if uv is not None and prev is not None:
                    cv2.line(canvas, prev, uv, color, thick, cv2.LINE_AA)
                prev = uv

        # 平行线网格 (绘制高度 z0): 平行于 X 轴 (Y=-ext..ext) 与平行于 Y 轴 (X=-ext..ext) 两组
        for i in range(-ext, ext + 1, step):
            _seg((-ext, i, z0), (ext, i, z0), COL_GRAY, 1)
            _seg((i, -ext, z0), (i, ext, z0), COL_GRAY, 1)
        # 坐标轴加粗高亮: X 红 / Y 绿 (随平面高度) / Z 蓝 (0→600mm), 明显可见
        _seg((-ext, 0, z0), (ext, 0, z0), (60, 60, 245), 3)
        _seg((0, -ext, z0), (0, ext, z0), COL_GREEN, 3)
        _seg((0, 0, 0), (0, 0, tr.PLANE_Z_MM), COL_BLUE, 4)
        # Tag 等高辅助红线: 平面高度与某锚定标靶中心 Z 重合且该靶不在原点时,
        # 平移一条红色 X 轴穿过该标靶 (如 Z=196 平面过 Tag 1); Tag 0 在原点, 主 X 轴已穿过
        for tid, c_t in tr.anchor_positions.items():
            if abs(float(c_t[2]) - z0) < 2.0 and (abs(float(c_t[0])) > 1.0 or abs(float(c_t[1])) > 1.0):
                _seg((c_t[0] - ext, c_t[1], z0), (c_t[0] + ext, c_t[1], z0), (60, 60, 245), 2)
                uv = _project((c_t[0] + ext, c_t[1], z0))
                if uv is not None:
                    draw_text(canvas, f"X (Tag {tid})", (uv[0] + 6, uv[1] - 8), 13, (60, 60, 245), True)
        # Z 轴高度刻度 (每 100mm) + 顶端箭头: 俯视相机下高度轴指向镜头呈放射状,
        # 刻度数值让"向上生长"方向一目了然, 消除透视歧义
        for hz in range(100, tr.PLANE_Z_MM, 100):
            tp = _project((0, 0, hz))
            if tp is not None:
                cv2.line(canvas, (tp[0] - 5, tp[1]), (tp[0] + 5, tp[1]), COL_BLUE, 2, cv2.LINE_AA)
                draw_text(canvas, str(hz), (tp[0] + 8, tp[1] - 6), 12, COL_BLUE, True)
        p_top = _project((0, 0, tr.PLANE_Z_MM))
        p_base = _project((0, 0, 0))
        if p_top is not None and p_base is not None:
            d = np.array(p_top, dtype=np.float64) - np.array(p_base, dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n > 24:
                d /= n
                perp = np.array([-d[1], d[0]])
                tip = np.array(p_top, dtype=np.float64)
                wing = 14.0 * d
                arrow = np.array([tip, tip - wing + 6.0 * perp, tip - wing - 6.0 * perp],
                                 dtype=np.int32)
                cv2.fillPoly(canvas, [arrow], COL_BLUE)
        for label, p, col in (("X", (ext + 70, 0, z0), (60, 60, 245)),
                              ("Y", (0, ext + 70, z0), COL_GREEN),
                              ("Z", (0, 0, tr.PLANE_Z_MM + 70), COL_BLUE),
                              ("0", (0, 0, 0), COL_WHITE)):
            uv = _project(p)
            if uv is not None:
                draw_text(canvas, label, (uv[0] + 6, uv[1] - 8), 15, col, True)

    # ------------------------------ 信息面板与 Toast ------------------------------
    def draw_info_panel(self, canvas, y_off=0):
        """左上信息面板 (实测 / 理论 / 偏差 / 世界系状态 / 机械臂状态)"""
        tr = self.tr
        x1, y1 = 14, y_off + 14
        x2, y2 = 478, y_off + 296
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_PANEL_EDGE, 1)

        y = y1 + 26
        draw_text(canvas, f"Robot 在线跟踪 | 目标 Tag {tr.target_tag_id}",
                  (x1 + 14, y), 19, COL_WHITE, True)
        y += 28
        draw_text(canvas, f"世界系地图: {os.path.basename(tr.map_path)}",
                  (x1 + 14, y), 15, COL_GRAY)
        y += 24
        if tr.support_ids and tr.rmse is not None:
            draw_text(canvas, f"支撑: {len(tr.support_ids)} 靶 {tr.support_ids}"
                              f" | RMSE {tr.rmse:.2f}px", (x1 + 14, y), 15, COL_GRAY)
        else:
            draw_text(canvas, "支撑: 无已知标靶入镜, 世界位姿失效",
                      (x1 + 14, y), 15, COL_RED)
        y += 26
        draw_text(canvas, f"实测 {fmt_point(tr.measured)}",
                  (x1 + 14, y), 17, COL_GREEN if tr.measured is not None else COL_GRAY, True)
        y += 26
        theo_txt = f"理论 {fmt_point(tr.theoretical)}" if tr.theoretical is not None \
            else "理论      (地图中无该 Tag)"
        draw_text(canvas, theo_txt, (x1 + 14, y), 17, COL_YELLOW)
        y += 26
        if tr.measured is not None and tr.theoretical is not None:
            dev = tr.measured - tr.theoretical
            draw_text(canvas, f"偏差 {fmt_point(dev, signed=True)}",
                      (x1 + 14, y), 17, COL_CYAN)
        else:
            draw_text(canvas, "偏差          --", (x1 + 14, y), 17, COL_GRAY)
        y += 26
        if tr.sampling:
            draw_text(canvas, f"世界坐标系: {tr.sample_stage}", (x1 + 14, y), 16, COL_YELLOW, True)
        elif tr.world_locked:
            draw_text(canvas, f"世界坐标系: 已锁定 ({tr.lock_info})", (x1 + 14, y), 16, COLOR_ACCENT)
        else:
            draw_text(canvas, "世界坐标系: 未确定 [确定世界坐标系]", (x1 + 14, y), 16, COL_GRAY)
        y += 26
        if tr.robot.is_connected:
            pos_txt = f"末端 {fmt_point(tr.robot_pos)}" if tr.robot_pos is not None \
                else "末端 --"
            draw_text(canvas, f"机械臂: {tr.robot.port} | {pos_txt}",
                      (x1 + 14, y), 15, COL_GREEN)
        else:
            draw_text(canvas, "机械臂: 未连接 [C] 连接", (x1 + 14, y), 15, COL_GRAY)

    def draw_toast(self, canvas):
        """右下角浮动通知"""
        tr = self.tr
        h, w = canvas.shape[:2]
        if time.time() - tr.toast_time < 6.0:
            col = COL_RED if tr.toast_err else COL_WHITE
            tw = int(len(tr.toast) * 16 * 1.05) + 16
            tx = max(w - tw - 20, 20)
            ty = h - 40
            overlay = canvas.copy()
            cv2.rectangle(overlay, (tx - 10, ty - 8), (w - 10, ty + 24), COL_PANEL_BG, -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
            draw_text(canvas, tr.toast, (tx, ty), 16, col)

    def compose_canvas(self, frame):
        """视频帧 + 顶部工具栏拼合 (工具栏独立于画面, 不遮挡视频内容)"""
        fh, fw = frame.shape[:2]
        tool_area = np.full((TOOLBAR_H, fw, 3), COLOR_BG, dtype=np.uint8)
        full = np.vstack((tool_area, frame))
        return full
