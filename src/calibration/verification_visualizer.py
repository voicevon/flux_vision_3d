# -*- coding: utf-8 -*-
"""
标定精度验证 3D/2D 视觉渲染器 (VerificationVisualizer)
===================================================
单一职责设计：
  1. 3D 双四棱柱虚实位姿对比立体渲染 (BA 理论真值 vs 实测抓取)；
  2. 2D 角点残差矢量放大图绘制与误差标牌合成；
  3. 纯图像处理与 OpenCV 画布合成，绝不参与 PnP 解算逻辑，亦不管理窗口生命周期。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class VerificationVisualizer:
    """标定验证视觉呈现与双模态渲染器"""

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        初始化渲染器
        :param camera_matrix: 3x3 相机内参
        :param dist_coeffs: 畸变系数
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)

    def render_tag_dual_prisms(
        self,
        img: np.ndarray,
        ba_rvec: Optional[np.ndarray],
        ba_tvec: Optional[np.ndarray],
        obs_rvec: Optional[np.ndarray],
        obs_tvec: Optional[np.ndarray],
        tag_id: int,
        err_px: float,
        err_mm: float,
        observed_corners: Optional[np.ndarray] = None
    ):
        """
        绘制全局 BA 平差理论位姿 (绿色) 与单帧实测抓取位姿 (金色/橙红) 的 3D 双四棱柱空间对比
        """
        try:
            hw = 15.0   # 截面半宽 15mm，整体截面 30.0mm x 30.0mm
            L = 75.0    # 柱体高度 75mm (与实际尺寸协调)

            pts_3d = np.array([
                # 底面 4 点 (Z=0)
                [-hw, -hw, 0.0],
                [ hw, -hw, 0.0],
                [ hw,  hw, 0.0],
                [-hw,  hw, 0.0],
                # 顶面 4 点 (Z=L)
                [-hw, -hw, L],
                [ hw, -hw, L],
                [ hw,  hw, L],
                [-hw,  hw, L],
                # 顶面中心
                [0.0, 0.0, L]
            ], dtype=np.float64)

            # 1. 投影 BA 理论棱柱
            proj_ba = None
            if ba_rvec is not None and ba_tvec is not None:
                p, _ = cv2.projectPoints(pts_3d, ba_rvec, ba_tvec, self.camera_matrix, self.dist_coeffs)
                proj_ba = p.reshape((-1, 2)).astype(int)

            # 2. 投影实测观测棱柱
            proj_obs = None
            if obs_rvec is not None and obs_tvec is not None:
                p, _ = cv2.projectPoints(pts_3d, obs_rvec, obs_tvec, self.camera_matrix, self.dist_coeffs)
                proj_obs = p.reshape((-1, 2)).astype(int)

            is_good = (err_px <= 1.5 and err_mm <= 1.5)
            is_moderate = (err_px <= 3.0 and err_mm <= 2.5)

            overlay = img.copy()

            # A. 良好达标样本 (BA 与实测高度吻合，渲染纯正翠绿立体方柱)
            if is_good:
                pts = proj_ba if proj_ba is not None else proj_obs
                if pts is not None:
                    b_pts = pts[0:4]
                    t_pts = pts[4:8]
                    top_c = tuple(pts[8])

                    # 翠绿色半透明实心柱体
                    side_color = (0, 190, 50)
                    cap_color = (80, 255, 120)
                    for i in range(4):
                        next_i = (i + 1) % 4
                        side_poly = np.array([b_pts[i], b_pts[next_i], t_pts[next_i], t_pts[i]], dtype=np.int32)
                        cv2.fillPoly(overlay, [side_poly], side_color)
                    cv2.fillPoly(overlay, [t_pts], cap_color)
                    cv2.addWeighted(overlay, 0.42, img, 0.58, 0, img)

                    # 纯净亮白/亮绿棱线描边
                    edge_c = (0, 245, 100)
                    cv2.polylines(img, [b_pts], True, edge_c, 2, cv2.LINE_AA)
                    cv2.polylines(img, [t_pts], True, (255, 255, 255), 2, cv2.LINE_AA)
                    for i in range(4):
                        cv2.line(img, tuple(b_pts[i]), tuple(t_pts[i]), edge_c, 2, cv2.LINE_AA)

                    # 顶盖中心与标识
                    cv2.circle(img, top_c, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(img, "BA", (top_c[0] + 5, top_c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

                    # 底面实测角点连线与四色圆点
                    if observed_corners is not None:
                        c_int = observed_corners.reshape((4, 2)).astype(np.int32)
                        cv2.polylines(img, [c_int], True, (0, 255, 100), 2, cv2.LINE_AA)
                        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                        for pt_i, pt in enumerate(c_int):
                            cv2.circle(img, tuple(pt), 4, dot_colors[pt_i], -1)

                    # 悬浮高对比度稳态绿色标牌
                    min_x = min(np.min(b_pts[:, 0]), np.min(t_pts[:, 0]))
                    min_y = min(np.min(b_pts[:, 1]), np.min(t_pts[:, 1]))
                    bx = max(10, int(min_x - 10))
                    by = max(40, int(min_y - 14))

                    label = f"Tag#{tag_id} [PASS] {err_mm:.2f}mm ({err_px:.2f}px)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), (10, 42, 16), -1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), (0, 240, 90), 1)
                    cv2.putText(img, label, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # B. 偏差/需回审样本 (绿色 BA 棱柱 vs 金黄/橙红实测棱柱，形成空间错位拉线)
            else:
                if proj_ba is not None:
                    ba_b = proj_ba[0:4]
                    ba_t = proj_ba[4:8]
                    ba_c = tuple(proj_ba[8])

                    side_c = (0, 180, 80)
                    for i in range(4):
                        next_i = (i + 1) % 4
                        side_poly = np.array([ba_b[i], ba_b[next_i], ba_t[next_i], ba_t[i]], dtype=np.int32)
                        cv2.fillPoly(overlay, [side_poly], side_c)
                    cv2.fillPoly(overlay, [ba_t], (50, 250, 140))
                    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

                    cv2.polylines(img, [ba_b], True, (0, 220, 80), 2, cv2.LINE_AA)
                    cv2.polylines(img, [ba_t], True, (120, 255, 160), 2, cv2.LINE_AA)
                    for i in range(4):
                        cv2.line(img, tuple(ba_b[i]), tuple(ba_t[i]), (0, 220, 80), 2, cv2.LINE_AA)
                    cv2.circle(img, ba_c, 4, (0, 255, 100), -1, cv2.LINE_AA)
                    cv2.putText(img, "BA", (ba_c[0] + 6, ba_c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)

                if proj_obs is not None:
                    obs_b = proj_obs[0:4]
                    obs_t = proj_obs[4:8]
                    obs_c = tuple(proj_obs[8])

                    warn_c = (0, 80, 240) if not is_moderate else (0, 160, 255)
                    overlay2 = img.copy()
                    for i in range(4):
                        next_i = (i + 1) % 4
                        side_poly = np.array([obs_b[i], obs_b[next_i], obs_t[next_i], obs_t[i]], dtype=np.int32)
                        cv2.fillPoly(overlay2, [side_poly], warn_c)
                    cv2.fillPoly(overlay2, [obs_t], (0, 210, 255))
                    cv2.addWeighted(overlay2, 0.30, img, 0.70, 0, img)

                    cv2.polylines(img, [obs_b], True, warn_c, 2, cv2.LINE_AA)
                    cv2.polylines(img, [obs_t], True, (255, 255, 255), 2, cv2.LINE_AA)
                    for i in range(4):
                        cv2.line(img, tuple(obs_b[i]), tuple(obs_t[i]), warn_c, 2, cv2.LINE_AA)
                    cv2.circle(img, obs_c, 4, (0, 200, 255), -1, cv2.LINE_AA)
                    cv2.putText(img, "OBS", (obs_c[0] + 6, obs_c[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)

                    if proj_ba is not None:
                        cv2.line(img, obs_c, ba_c, (0, 50, 255), 3, cv2.LINE_AA)
                        cv2.circle(img, obs_c, 5, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(img, ba_c, 5, (0, 255, 0), -1, cv2.LINE_AA)

                anchor_pts = proj_ba if proj_ba is not None else proj_obs
                if anchor_pts is not None:
                    min_x = np.min(anchor_pts[:, 0])
                    min_y = np.min(anchor_pts[:, 1])
                    bx = max(10, int(min_x - 10))
                    by = max(40, int(min_y - 14))
                    tag_status = "[WARN]" if is_moderate else "[FAIL-回审]"
                    border_c = (0, 160, 255) if is_moderate else (0, 0, 255)
                    bg_c = (15, 30, 60) if is_moderate else (15, 15, 65)
                    label = f"Tag#{tag_id} {tag_status} {err_mm:.2f}mm ({err_px:.2f}px)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), bg_c, -1)
                    cv2.rectangle(img, (bx - 6, by - th - 6), (bx + tw + 8, by + 4), border_c, 2)
                    cv2.putText(img, label, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    def render_verification_frame(
        self,
        img_path: str,
        frame_results: List[Dict],
        view_mode_3d: bool = True,
        manifest_data: Optional[Dict] = None,
        solve_single_tag_pnp_func = None,
        out_path: Optional[str] = None
    ) -> np.ndarray:
        """
        渲染单帧 LOO 盲测可视化图 (支持 3D 双棱柱空间对比视图 与 2D 角点残差矢量视图)
        """
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
        disp = img.copy()
        h, w = disp.shape[:2]

        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

        # 若当前帧 frame_results 为空（如孤立单靶帧），依然尽可能从清单提取有效样本渲染实测 3D 棱柱
        if not frame_results and manifest_data and solve_single_tag_pnp_func:
            fname = os.path.basename(img_path)
            img_entry = manifest_data.get("images", {}).get(fname, {})
            for obs in img_entry.get("observations", []):
                if obs.get("keep", True):
                    tid = int(obs.get("tag_id", -1))
                    c = np.array(obs.get("corners", []), dtype=np.float64)
                    if len(c) == 4:
                        ok_pnp, r_obs, t_obs = solve_single_tag_pnp_func(c)
                        if ok_pnp:
                            self.render_tag_dual_prisms(
                                disp, ba_rvec=None, ba_tvec=None,
                                obs_rvec=r_obs, obs_tvec=t_obs,
                                tag_id=tid, err_px=0.0, err_mm=0.0,
                                observed_corners=c
                            )

        if view_mode_3d:
            # 模式 A: 3D 双四棱柱虚实位姿对比
            for r in frame_results:
                self.render_tag_dual_prisms(
                    disp, ba_rvec=r.get("ba_rvec"), ba_tvec=r.get("ba_tvec"),
                    obs_rvec=r.get("obs_rvec"), obs_tvec=r.get("obs_tvec"),
                    tag_id=r["blind_tag_id"], err_px=r["err_px"], err_mm=r["err_mm"],
                    observed_corners=r.get("obs_corners")
                )

            mode_badge = "[ 视图: 3D双四棱柱空间对比模式 (按 T 键切换 2D 残差矢量) ]"
            (mw, mh), _ = cv2.getTextSize(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (18, 22, 30), -1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (0, 220, 180), 1)
            cv2.putText(disp, mode_badge, (22, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 200), 1, cv2.LINE_AA)

        else:
            # 模式 B: 2D 角点残差矢量模式
            for r in frame_results:
                obs = r["obs_corners"].astype(np.int32)
                proj = r["proj_corners"].astype(np.int32)
                tid = r["blind_tag_id"]
                err_px = r["err_px"]
                err_mm = r["err_mm"]

                is_good = (err_px <= 1.5)
                is_moderate = (err_px <= 3.0)

                obs_line_color = (0, 230, 80) if is_good else ((180, 180, 180) if is_moderate else (120, 120, 120))
                cv2.polylines(disp, [obs], True, obs_line_color, 2 if is_good else 1, cv2.LINE_AA)

                proj_color = (0, 255, 100) if is_good else ((0, 180, 255) if is_moderate else (0, 0, 255))
                cv2.polylines(disp, [proj], True, proj_color, 2, cv2.LINE_AA)

                scale = 15.0
                for i in range(4):
                    p_obs = r["obs_corners"][i]
                    p_proj = r["proj_corners"][i]
                    dx = (p_proj[0] - p_obs[0]) * scale
                    dy = (p_proj[1] - p_obs[1]) * scale
                    pt_s = (int(p_obs[0]), int(p_obs[1]))
                    pt_e = (int(p_obs[0] + dx), int(p_obs[1] + dy))
                    cv2.circle(disp, pt_s, 4, dot_colors[i], -1)
                    cv2.arrowedLine(disp, pt_s, pt_e, proj_color, 2, tipLength=0.25)

                cx = int(np.mean(obs[:, 0]))
                min_y = int(np.min(obs[:, 1]))
                badge_y = max(75, min_y - 14)
                badge_x = max(10, cx - 85)

                status_text = "[PASS]" if is_good else ("[WARN]" if is_moderate else "[FAIL]")
                label = f"Tag#{tid} {status_text} {err_px:.2f}px / {err_mm:.2f}mm"
                badge_bg = (12, 42, 16) if is_good else ((15, 35, 75) if is_moderate else (15, 15, 75))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 18), (badge_x + tw + 8, badge_y + 4), badge_bg, -1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 18), (badge_x + tw + 8, badge_y + 4), proj_color, 1 if is_good else 2)
                cv2.putText(disp, label, (badge_x, badge_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            mode_badge = "[ 视图: 2D角点残差矢量模式 (按 T 键切换 3D 双棱柱) ]"
            (mw, mh), _ = cv2.getTextSize(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (18, 22, 30), -1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (0, 200, 255), 1)
            cv2.putText(disp, mode_badge, (22, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        if out_path is not None:
            cv2.imwrite(out_path, disp)

        return disp
