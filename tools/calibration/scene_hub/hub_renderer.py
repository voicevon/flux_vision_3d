"""
Scene Hub 视觉渲染引擎 (HubRenderer)
===================================
提供 1280x720 高清深色科技控制台双缓冲 Canvas 渲染
支持：
- 中文字体高质量渲染 (PIL + msyh/simhei 自动矢量抗锯齿)
- 场景名称/友好别名高亮显示与物理 ID 区分
- 显眼的 [+] 新建工况场景 (按 [N] 键) 交互卡片
- 单帧大图全宽占满自适应预览 (按 [F] 键切换) 与并排看板双模态
"""

import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.calibration.scene_hub.hub_state import HubState
from tools.calibration.scene_hub.hub_capture_stream import HubCaptureStream


# 字体内存缓存
_FONT_CACHE = {}


def draw_text(img: np.ndarray, text: str, pos: tuple[int, int], font_size: int = 16,
              color: tuple[int, int, int] = (240, 240, 240), bold: bool = False):
    """在 OpenCV BGR 图像上绘制高质量中文或西文字符"""
    if not text:
        return

    # 判断是否包含非 ASCII 字符 (如中文)
    if any(ord(c) > 127 for c in text):
        key = (font_size, bold)
        if key not in _FONT_CACHE:
            try:
                font_path = "C:/Windows/Fonts/msyh.ttc"
                if not os.path.exists(font_path):
                    font_path = "C:/Windows/Fonts/simhei.ttf"
                _FONT_CACHE[key] = ImageFont.truetype(font_path, font_size)
            except Exception:
                _FONT_CACHE[key] = ImageFont.load_default()
        font = _FONT_CACHE[key]

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        # BGR -> RGB
        rgb_col = (int(color[2]), int(color[1]), int(color[0]))
        draw.text(pos, text, font=font, fill=rgb_col)
        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img[:] = res[:]
    else:
        scale = font_size / 28.0
        cv2.putText(img, text, (pos[0], pos[1] + int(font_size * 0.85)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)


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
        draw_text(canvas, "| 标定采样场景综合管理驾驶舱 (Scene Hub) v1.1", (168, 16), font_size=16, color=self.COLOR_WHITE)

        # 活动场景微章
        act_sc = state.scene_mgr.get_active_scene()
        act_name = act_sc.name if act_sc else "未设定"
        act_id = state.active_scene_id or "无"
        draw_text(canvas, f"生产运行场景: 【{act_name}】({act_id})", (self.canvas_w - 530, 17),
                  font_size=15, color=(0, 240, 140))

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
        draw_text(canvas, f"工况场景批次列表 ({len(state.scenes)})", (16, 68), font_size=17, color=self.COLOR_WHITE, bold=True)

        if not state.scenes:
            draw_text(canvas, "暂无场景，按 [N] 键新建新工况场景", (20, 140), font_size=15, color=self.COLOR_GRAY)
            return

        card_h = 76
        start_y = 96
        max_cards = 6

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

            # 优先突出显示友好别名 (支持中文)，第二行用浅灰色显示物理 ID
            prefix = f"{real_idx + 1:02d}."
            display_title = f"{prefix} {sc.name}"
            title_col = (0, 255, 200) if is_selected else self.COLOR_WHITE
            draw_text(canvas, display_title, (24, cy + 8), font_size=17, color=title_col, bold=is_selected)

            # 第二行：物理唯一 ID 与张数
            id_subtitle = f"ID: {sc.scene_id} | {sc.image_count}帧"
            cv2.putText(canvas, id_subtitle[:34], (24, cy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_GRAY, 1, cv2.LINE_AA)

            # 第三行：平差状态徽章
            ba_badge = f"RMSE: {sc.global_rmse_px:.2f}px" if sc.ba_solved else "未平差"
            ba_col = (0, 220, 100) if sc.ba_solved else self.COLOR_DARK_GRAY
            cv2.putText(canvas, ba_badge, (24, cy + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ba_col, 1, cv2.LINE_AA)

            # 发布标志 / 当前活动标志
            if is_active:
                draw_text(canvas, "[当前活动]", (270, cy + 8), font_size=13, color=(0, 255, 120))
            if sc.is_published:
                draw_text(canvas, "★生产", (308, cy + 48), font_size=14, color=self.COLOR_GOLD, bold=True)
            else:
                draw_text(canvas, "草稿", (314, cy + 48), font_size=13, color=self.COLOR_DARK_GRAY)

        # 醒目的 [+] 新建工况场景交互卡片
        btn_y = start_y + len(visible_scenes) * (card_h + 8)
        if btn_y + 44 < 665:
            cv2.rectangle(canvas, (12, btn_y), (368, btn_y + 42), (20, 28, 38), -1)
            cv2.rectangle(canvas, (12, btn_y), (368, btn_y + 42), (0, 220, 240), 1)
            draw_text(canvas, "[+] 新建工况场景 (按 [N] 键)", (75, btn_y + 11), font_size=16, color=(0, 240, 220), bold=True)

    def _render_inspector_stage(self, canvas: np.ndarray, state: HubState):
        """渲染右侧画廊与体检看板 (x: 380~1280, y: 52~670)"""
        sc = state.get_selected_scene()
        if not sc:
            draw_text(canvas, "未选中有效场景，按 [N] 键新建", (680, 360), font_size=20, color=self.COLOR_GRAY)
            return

        # ---- 区域 1: 顶部横向照片滚动带 (y: 60 ~ 180) ----
        self._render_thumbnail_strip(canvas, state, sc)

        # ---- 区域 2: 下半部 (大图预览与看板) ----
        if state.expanded_preview_mode:
            # 全宽沉浸式大图占满模式
            self._render_expanded_photo_preview(canvas, state, sc)
        else:
            # 标准双分栏模式 (左大图预览 + 右综合体检仪表盘)
            self._render_single_photo_preview(canvas, state, sc)
            self._render_scene_metrics_panel(canvas, state, sc)

    def _render_thumbnail_strip(self, canvas: np.ndarray, state: HubState, sc):
        """绘制顶部照片缩略图带"""
        draw_text(canvas, f"采样相册 (已采 {len(state.current_images)} 帧照片)", (398, 68), font_size=17, color=self.COLOR_WHITE, bold=True)

        if not state.current_images:
            box_x, box_y, box_w, box_h = 398, 92, 860, 84
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (26, 30, 40), -1)
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_BORDER, 1)
            draw_text(canvas, "当前场景尚未采集照片！", (box_x + 20, box_y + 16), font_size=17, color=(0, 200, 240))
            draw_text(canvas, "请直接按下键盘 [C] 键，原地进入相机实时连拍向导拍摄照片。", (box_x + 20, box_y + 46), font_size=15, color=self.COLOR_GRAY)
            return

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

            base_name = os.path.basename(img_path)
            cv2.rectangle(canvas, (x, y + th - 16), (x + tw, y + th), (10, 10, 14), -1)
            cv2.putText(canvas, base_name, (x + 4, y + th - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 180) if is_cur else self.COLOR_GRAY, 1, cv2.LINE_AA)

            if is_cur:
                cv2.line(canvas, (x + tw // 2 - 4, y + th + 4), (x + tw // 2 + 4, y + th + 4), (0, 255, 200), 2)

    def _render_single_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """标准分栏模式下的单帧大图视口"""
        box_x, box_y, box_w, box_h = 398, 194, 435, 460
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (22, 25, 34), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_BORDER, 1)

        if not state.current_images:
            draw_text(canvas, "无图片预览", (box_x + 150, box_y + 200), font_size=18, color=self.COLOR_DARK_GRAY)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=415, max_h=370)

        img_title = f"单帧: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 14, box_y + 10), font_size=15, color=self.COLOR_CYAN)

        # 放大提示
        draw_text(canvas, "[F 键全宽放大]", (box_x + box_w - 120, box_y + 12), font_size=13, color=(0, 255, 180))

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 42 + (375 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (50, 56, 72), 1)

        draw_text(canvas, "[← / →] 切换选片  |  [F] 放大预览占满空间", (box_x + 36, box_y + box_h - 22), font_size=13, color=self.COLOR_GRAY)

    def _render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """全宽自适应大图视口 (按 F 键展开，占满右侧全部空间)"""
        box_x, box_y, box_w, box_h = 398, 194, 862, 460
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 30), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)

        if not state.current_images:
            draw_text(canvas, "无图片预览", (box_x + 360, box_y + 200), font_size=20, color=self.COLOR_DARK_GRAY)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=840, max_h=390)

        img_title = f"全宽沉浸式预览: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 18, box_y + 12), font_size=16, color=(0, 255, 200), bold=True)
        draw_text(canvas, "[按 F 键退出全宽，恢复指标看板]", (box_x + box_w - 260, box_y + 12), font_size=14, color=self.COLOR_GOLD)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 44 + (390 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 70, 90), 1)

        draw_text(canvas, "[← / →] 左右键切换查看相册大图   |   [Space] 空间抓拍   |   [F] 退出全宽预览",
                  (box_x + 180, box_y + box_h - 22), font_size=14, color=self.COLOR_WHITE)

    def _render_scene_metrics_panel(self, canvas: np.ndarray, state: HubState, sc):
        """绘制当前场景体检指标与快捷动作卡片"""
        box_x, box_y, box_w, box_h = 848, 194, 412, 460
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (22, 25, 34), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_BORDER, 1)

        draw_text(canvas, "场景综合体检与几何健康报告", (box_x + 16, box_y + 12), font_size=16, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (box_x + 16, box_y + 38), (box_x + box_w - 16, box_y + 38), self.COLOR_BORDER, 1)

        cards = [
            ("采样数据集", f"总采集 {sc.image_count} 帧 (有效参与 {sc.active_image_count} 帧)", (0, 200, 255)),
            ("空间标靶覆盖", f"原点: Tag #{sc.origin_tag_id} | X轴: Tag #{sc.x_axis_tag_id}", (0, 240, 100)),
            ("两阶段 BA 平差", f"RMSE: {sc.global_rmse_px:.3f} px (优秀)" if sc.ba_solved else "尚未执行离线平差",
             (0, 255, 180) if sc.ba_solved else self.COLOR_GRAY),
            ("全局生产部署", "★ 已发布至生产环境 (运行中)" if sc.is_published else "草稿状态 (尚未发布)",
             self.COLOR_GOLD if sc.is_published else self.COLOR_DARK_GRAY),
        ]

        cy = box_y + 48
        for title, val, col in cards:
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 46), (28, 32, 44), -1)
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 46), (40, 46, 60), 1)
            draw_text(canvas, title, (box_x + 28, cy + 4), font_size=13, color=self.COLOR_GRAY)
            draw_text(canvas, val, (box_x + 28, cy + 22), font_size=14, color=col, bold=True)
            cy += 54

        # 快捷动作入口卡片
        cv2.rectangle(canvas, (box_x + 16, cy + 6), (box_x + box_w - 16, box_y + box_h - 16), (18, 22, 30), -1)
        cv2.rectangle(canvas, (box_x + 16, cy + 6), (box_x + box_w - 16, box_y + box_h - 16), (0, 180, 200), 1)
        draw_text(canvas, "快捷工作流通道 (Workflows):", (box_x + 24, cy + 12), font_size=15, color=self.COLOR_CYAN, bold=True)
        draw_text(canvas, "* 按 [N] : 新建采样场景与工况批次 (支持中文)", (box_x + 24, cy + 34), font_size=13, color=self.COLOR_WHITE)
        draw_text(canvas, "* 按 [R] : 修改当前场景显示名称 (支持中文)", (box_x + 24, cy + 54), font_size=13, color=(0, 255, 200))
        draw_text(canvas, "* 按 [F] : 放大/占满预览大图 (全宽自适应)", (box_x + 24, cy + 74), font_size=13, color=(0, 220, 255))
        draw_text(canvas, "* 按 [C] : 原地切入相机，开启空格连拍向导", (box_x + 24, cy + 94), font_size=13, color=(0, 255, 160))
        draw_text(canvas, "* 按 [S] : 启动 AprilTag 离线 Studio 深度平差", (box_x + 24, cy + 114), font_size=13, color=(0, 255, 160))
        draw_text(canvas, "* 按 [P] : 一键将该场景地图发布为生产运行地图", (box_x + 24, cy + 134), font_size=13, color=self.COLOR_GOLD)

    def _render_capture_viewport(self, canvas: np.ndarray, state: HubState):
        """渲染原地相机取流视口 (x: 380~1280, y: 52~670)"""
        vw, vh = 900, 618
        stream_view = self.capture_stream_renderer.render_stream_viewport(state, vw, vh)
        canvas[52:52 + vh, 380:380 + vw] = stream_view

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部状态与快捷键导航栏 (670~720px)"""
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)

        now = time.time()
        if state.toast_time > now:
            draw_text(canvas, f"[提示] {state.toast_msg}", (20, 684), font_size=16, color=(0, 255, 200), bold=True)
        else:
            if state.mode == HubState.MODE_CAPTURE:
                draw_text(canvas, "[Space] 抓拍并存入当前场景   |   [ESC / C] 退出采图返回看板   |   [S] 立即平差",
                          (20, 686), font_size=15, color=self.COLOR_WHITE)
            else:
                draw_text(canvas, "[N] 新建场景  [R] 改名(支持中文)  [F] 放大预览  [↑/↓] 选场景  [Enter] 设活动  [C] 连拍  [S] 平差  [P] 发布  [ESC] 退出",
                          (20, 686), font_size=14, color=self.COLOR_WHITE)
