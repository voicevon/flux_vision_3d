"""
Scene Hub 视觉渲染引擎 (HubRenderer)
===================================
提供 1280x720 高清深色科技控制台双缓冲 Canvas 渲染
"""

import os
import time
import cv2
import numpy as np

from tools.calibration.scene_hub.hub_state import HubState
from tools.calibration.scene_hub.hub_capture_stream import HubCaptureStream


class HubRenderer:
    """Scene Hub 统一界面渲染器"""

    def __init__(self):
        self.canvas_w = 1280
        self.canvas_h = 720
        self.capture_stream_renderer = HubCaptureStream()

        # 调色板定义 (深色科技风)
        self.COLOR_BG = (18, 20, 24)           # 全局底色
        self.COLOR_PANEL = (24, 28, 36)        # 侧边栏/卡片底色
        self.COLOR_CARD_ACTIVE = (38, 48, 64)  # 选中卡片底色
        self.COLOR_BORDER = (45, 52, 68)       # 普通线框
        self.COLOR_ACTIVE_BORDER = (0, 240, 120)  # 选中项荧光绿
        self.COLOR_CYAN = (230, 200, 0)        # 科技青 (BGR: 0, 200, 230)
        self.COLOR_GOLD = (50, 190, 255)       # 金黄色 (BGR)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_GRAY = (140, 145, 155)
        self.COLOR_DARK_GRAY = (70, 75, 85)

    def render(self, state: HubState) -> np.ndarray:
        """主绘制入口，返回 1280x720 BGR 图像"""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        # 1. 顶部状态栏
        self._render_header(canvas, state)

        # 2. 左侧场景列表栏 (宽度 380)
        self._render_left_scene_list(canvas, state)

        # 3. 垂直分割线
        cv2.line(canvas, (380, 52), (380, 670), self.COLOR_BORDER, 1)

        # 4. 右侧主舞台
        if state.mode == HubState.MODE_CAPTURE:
            self._render_capture_viewport(canvas, state)
        else:
            self._render_inspector_stage(canvas, state)

        # 5. 底部状态与快捷键导航栏
        self._render_footer(canvas, state)

        return canvas

    def _render_header(self, canvas: np.ndarray, state: HubState):
        """渲染顶部标题栏 (0~52px)"""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 52), (14, 16, 20), -1)
        cv2.line(canvas, (0, 52), (self.canvas_w, 52), self.COLOR_BORDER, 1)

        # 系统标题与图标
        cv2.circle(canvas, (24, 26), 7, (0, 255, 180), -1)
        cv2.putText(canvas, "flux_vision_3d", (38, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 2, cv2.LINE_AA)
        cv2.putText(canvas, "| 标定采样场景综合管理驾驶舱 (Scene Hub) v1.0", (168, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 活动场景微章
        act_id = state.active_scene_id or "未设定"
        act_text = f"全局生产活动场景: {act_id}"
        cv2.putText(canvas, act_text, (self.canvas_w - 480, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 120), 1, cv2.LINE_AA)

        # 相机硬件指示
        cam_status = "MOCK" if state.camera_streamer.is_mock else "ONLINE"
        cam_col = (0, 180, 255) if state.camera_streamer.is_mock else (0, 240, 100)
        cv2.circle(canvas, (self.canvas_w - 60, 26), 5, cam_col, -1)
        cv2.putText(canvas, f"CAM {cam_status}", (self.canvas_w - 48, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GRAY, 1, cv2.LINE_AA)

    def _render_left_scene_list(self, canvas: np.ndarray, state: HubState):
        """渲染左侧场景卡片流 (x: 0~380, y: 52~670)"""
        cv2.rectangle(canvas, (0, 52), (380, 670), self.COLOR_PANEL, -1)

        # 栏目标题
        cv2.putText(canvas, f"工况与采样场景列表 ({len(state.scenes)})", (16, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 2, cv2.LINE_AA)

        if not state.scenes:
            cv2.putText(canvas, "暂无场景，按 [N] 键新建新工况场景", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_GRAY, 1, cv2.LINE_AA)
            return

        card_h = 76
        start_y = 96
        max_cards = 7
        
        # 简单分页滚动
        scroll_start = max(0, state.selected_scene_idx - max_cards + 1)
        visible_scenes = state.scenes[scroll_start: scroll_start + max_cards]

        for i, sc in enumerate(visible_scenes):
            real_idx = scroll_start + i
            is_selected = (real_idx == state.selected_scene_idx)
            is_active = (sc.scene_id == state.active_scene_id)

            cy = start_y + i * (card_h + 8)
            card_col = self.COLOR_CARD_ACTIVE if is_selected else (28, 32, 42)
            border_col = self.COLOR_ACTIVE_BORDER if is_selected else self.COLOR_BORDER

            # 绘制卡片底板与边框
            cv2.rectangle(canvas, (12, cy), (368, cy + card_h), card_col, -1)
            cv2.rectangle(canvas, (12, cy), (368, cy + card_h), border_col, 2 if is_selected else 1)

            # 左侧高亮指示条
            if is_selected:
                cv2.rectangle(canvas, (12, cy), (16, cy + card_h), (0, 255, 160), -1)

            # 序号与名称
            prefix = f"{real_idx + 1:02d}."
            name_text = f"{prefix} {sc.scene_id}"
            title_col = (0, 255, 200) if is_selected else self.COLOR_WHITE
            cv2.putText(canvas, name_text[:28], (24, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, title_col, 2 if is_selected else 1, cv2.LINE_AA)

            # 状态徽章条
            img_badge = f"{sc.image_count} 帧"
            ba_badge = f"RMSE: {sc.global_rmse_px:.2f}px" if sc.ba_solved else "未平差"
            ba_col = (0, 220, 100) if sc.ba_solved else self.COLOR_GRAY
            cv2.putText(canvas, img_badge, (24, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"| {ba_badge}", (88, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ba_col, 1, cv2.LINE_AA)

            # 发布标志 / 当前活动标志
            if is_active:
                cv2.putText(canvas, "[当前活动]", (270, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
            if sc.is_published:
                cv2.putText(canvas, "★生产", (308, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GOLD, 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "草稿", (314, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_DARK_GRAY, 1, cv2.LINE_AA)

    def _render_inspector_stage(self, canvas: np.ndarray, state: HubState):
        """渲染右侧画廊与体检看板 (x: 380~1280, y: 52~670)"""
        sc = state.get_selected_scene()
        if not sc:
            cv2.putText(canvas, "未选中有效场景", (680, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_GRAY, 2, cv2.LINE_AA)
            return

        # ---- 区域 1: 顶部横向照片滚动带 (y: 60 ~ 180) ----
        self._render_thumbnail_strip(canvas, state, sc)

        # ---- 区域 2: 下半部 (单帧大图预览 + 数据看板) (y: 190 ~ 660) ----
        # 左侧：单帧高清大图预览 (x: 395 ~ 835)
        self._render_single_photo_preview(canvas, state, sc)

        # 右侧：场景综合体检仪表盘 (x: 850 ~ 1265)
        self._render_scene_metrics_panel(canvas, state, sc)

    def _render_thumbnail_strip(self, canvas: np.ndarray, state: HubState, sc):
        """绘制顶部照片缩略图带"""
        cv2.putText(canvas, f"场景采样相册 (共 {len(state.current_images)} 帧图片)", (398, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 2, cv2.LINE_AA)

        if not state.current_images:
            # 暂无图片提示
            box_x, box_y, box_w, box_h = 398, 92, 860, 84
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (26, 30, 40), -1)
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_BORDER, 1)
            cv2.putText(canvas, "当前场景尚未采集照片！", (box_x + 20, box_y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, "请直接按下键盘 [C] 键，原地进入相机实时连拍向导拍摄照片。", (box_x + 20, box_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_GRAY, 1, cv2.LINE_AA)
            return

        # 展示 6 张缩略图
        tw, th = 118, 72
        pad = 12
        visible_count = 6
        offset = state.image_strip_offset
        visible_imgs = state.current_images[offset: offset + visible_count]

        for idx, img_path in enumerate(visible_imgs):
            real_idx = offset + idx
            is_cur = (real_idx == state.selected_image_idx)
            x = 398 + idx * (tw + pad)
            y = 96

            thumb = state.get_thumbnail(img_path, tw, th)
            if thumb is not None:
                canvas[y:y + th, x:x + tw] = thumb

            border_col = (0, 240, 200) if is_cur else self.COLOR_BORDER
            cv2.rectangle(canvas, (x, y), (x + tw, y + th), border_col, 2 if is_cur else 1)

            # 图片标签
            base_name = os.path.basename(img_path)
            cv2.rectangle(canvas, (x, y + th - 16), (x + tw, y + th), (10, 10, 14), -1)
            cv2.putText(canvas, base_name, (x + 4, y + th - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 180) if is_cur else self.COLOR_GRAY, 1, cv2.LINE_AA)

            if is_cur:
                # 选中小箭头
                cv2.line(canvas, (x + tw // 2 - 4, y + th + 4), (x + tw // 2 + 4, y + th + 4), (0, 255, 200), 2)

    def _render_single_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """渲染选中的单帧图片大视口"""
        box_x, box_y, box_w, box_h = 398, 194, 435, 460
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (22, 25, 34), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_BORDER, 1)

        if not state.current_images:
            cv2.putText(canvas, "无图片预览", (box_x + 150, box_y + 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_DARK_GRAY, 1, cv2.LINE_AA)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=415, max_h=370)

        # 头部标题
        img_title = f"预览: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        cv2.putText(canvas, img_title, (box_x + 14, box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_CYAN, 1, cv2.LINE_AA)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 42 + (375 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (50, 56, 72), 1)

        # 底部翻页提示
        cv2.putText(canvas, "[← / →] 左右键切换查看相册大图", (box_x + 80, box_y + box_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GRAY, 1, cv2.LINE_AA)

    def _render_scene_metrics_panel(self, canvas: np.ndarray, state: HubState, sc):
        """绘制当前场景体检指标与快捷动作卡片"""
        box_x, box_y, box_w, box_h = 848, 194, 412, 460
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (22, 25, 34), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_BORDER, 1)

        # 标题
        cv2.putText(canvas, "场景综合体检与几何健康报告", (box_x + 16, box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.line(canvas, (box_x + 16, box_y + 38), (box_x + box_w - 16, box_y + 38), self.COLOR_BORDER, 1)

        # 4 个指标小卡片
        cards = [
            ("采样数据集", f"总采集 {sc.image_count} 帧 (有效参与 {sc.active_image_count} 帧)", (0, 200, 255)),
            ("空间标靶覆盖", f"原点: Tag #{sc.origin_tag_id} | X轴: Tag #{sc.x_axis_tag_id}", (0, 240, 100)),
            ("两阶段 BA 平差", f"RMSE: {sc.global_rmse_px:.3f} px (优秀)" if sc.ba_solved else "尚未执行离线平差",
             (0, 255, 180) if sc.ba_solved else self.COLOR_GRAY),
            ("全局生产部署", "★ 已发布至生产环境 (运行中)" if sc.is_published else "草稿状态 (尚未发布)",
             self.COLOR_GOLD if sc.is_published else self.COLOR_DARK_GRAY),
        ]

        cy = box_y + 54
        for title, val, col in cards:
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 52), (28, 32, 44), -1)
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 52), (40, 46, 60), 1)
            cv2.putText(canvas, title, (box_x + 28, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_GRAY, 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (box_x + 28, cy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
            cy += 62

        # 快捷动作入口卡片
        cv2.rectangle(canvas, (box_x + 16, cy + 6), (box_x + box_w - 16, box_y + box_h - 16), (18, 22, 30), -1)
        cv2.rectangle(canvas, (box_x + 16, cy + 6), (box_x + box_w - 16, box_y + box_h - 16), (0, 180, 200), 1)
        cv2.putText(canvas, "快捷工作流通道 (Workflows):", (box_x + 26, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_CYAN, 2, cv2.LINE_AA)
        cv2.putText(canvas, "* 按 [C] : 原地切入相机，开启空格连拍向导", (box_x + 26, cy + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "* 按 [S] : 启动 AprilTag 离线 Studio 深度平差", (box_x + 26, cy + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, "* 按 [P] : 一键将该场景地图发布为生产运行地图", (box_x + 26, cy + 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_GOLD, 1, cv2.LINE_AA)

    def _render_capture_viewport(self, canvas: np.ndarray, state: HubState):
        """渲染原地相机取流视口 (x: 380~1280, y: 52~670)"""
        vw, vh = 900, 618
        stream_view = self.capture_stream_renderer.render_stream_viewport(state, vw, vh)
        canvas[52:52 + vh, 380:380 + vw] = stream_view

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部状态与快捷键导航栏 (670~720px)"""
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)

        # 状态 Toast
        now = time.time()
        if state.toast_time > now:
            cv2.putText(canvas, f"[INFO] {state.toast_msg}", (20, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1, cv2.LINE_AA)
        else:
            # 默认操作说明
            if state.mode == HubState.MODE_CAPTURE:
                cv2.putText(canvas, "[Space] 抓拍并存入当前场景   |   [ESC / C] 退出采图返回看板   |   [S] 立即平差",
                            (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_WHITE, 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "[↑/↓] 选场景  [Enter] 设活动  [C] 采图  [S] 平差  [P] 发布生产  [N] 新建  [K] 克隆  [V] 打开目录  [ESC] 退出",
                            (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_GRAY, 1, cv2.LINE_AA)
