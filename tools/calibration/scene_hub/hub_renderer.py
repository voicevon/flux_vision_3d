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
        """主绘制入口，返回 1280x720 BGR 图像 (左-中-右三栏布局)"""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        # 1. 顶部状态栏 (y: 0~50)
        self._render_header(canvas, state)

        # 2. 如果处于相机连拍向导模式，则使用全屏采图视口
        if state.mode == HubState.MODE_CAPTURE:
            self._render_capture_viewport(canvas, state)
            self._render_footer(canvas, state)
            return canvas

        # 3. 左侧综合导航栏 (x: 0~340, y: 50~670) - 场景列表 + 常用操作 + 核心工作流
        self._render_left_panel(canvas, state)
        cv2.line(canvas, (340, 50), (340, 670), self.COLOR_BORDER, 1)

        # 4. 中间栏与右侧栏 (按 F 键可全宽扩展预览，否则为标准三栏布局)
        sc = state.get_selected_scene()
        if state.expanded_preview_mode:
            # 全宽沉浸式大图视口 (跨越中间和右侧, x: 340~1280)
            self._render_expanded_photo_preview(canvas, state, sc)
        else:
            # 标准三栏结构:
            # 中间栏: 场景体检报告与几何健康看板 (x: 340~800)
            self._render_center_report_panel(canvas, state, sc)
            cv2.line(canvas, (800, 50), (800, 670), self.COLOR_BORDER, 1)

            # 最右侧栏: 采样相册缩略图与大图预览 (x: 800~1280)
            self._render_right_album_panel(canvas, state, sc)

        # 5. 底部状态与快捷键导航栏 (y: 670~720)
        self._render_footer(canvas, state)

        # 6. 如果打开了标定工具箱总菜单，则渲染置顶半透明浮层
        if state.is_toolbox_open:
            self._render_toolbox_modal(canvas, state)

        # 7. 如果打开了生产系统生效机制说明弹窗，则渲染置顶半透明浮层
        if state.is_help_modal_open:
            self._render_help_modal(canvas, state)

        return canvas

    def _render_header(self, canvas: np.ndarray, state: HubState):
        """渲染顶部标题栏 (0~50px)"""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 50), (14, 16, 20), -1)
        cv2.line(canvas, (0, 50), (self.canvas_w, 50), self.COLOR_BORDER, 1)

        # 系统标题与状态点
        cv2.circle(canvas, (22, 25), 6, (0, 255, 180), -1)
        cv2.putText(canvas, "flux_vision_3d", (36, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 2, cv2.LINE_AA)
        draw_text(canvas, "| 标定场景管理与深度平差驾驶舱 (Scene Hub)", (166, 16), font_size=16, color=self.COLOR_WHITE)

        # 全局生产运行场景信息 (向左挪以腾出工具箱按钮空间)
        act_sc = state.scene_mgr.get_active_scene()
        act_name = act_sc.name if act_sc else "未设定"
        act_id = state.active_scene_id or "无"
        draw_text(canvas, f"生产运行场景: 【{act_name}】({act_id})", (self.canvas_w - 680, 16),
                  font_size=14, color=(0, 240, 140))

        # [M] 标定工具箱总菜单按钮 (x: 1040~1180, y: 8~42)
        btn_bg = (32, 48, 64) if state.is_toolbox_open else (22, 28, 38)
        border_col = (0, 240, 220) if state.is_toolbox_open else (0, 180, 200)
        cv2.rectangle(canvas, (1040, 8), (1180, 42), btn_bg, -1)
        cv2.rectangle(canvas, (1040, 8), (1180, 42), border_col, 2 if state.is_toolbox_open else 1)
        draw_text(canvas, "[M] 综合工具箱", (1050, 16), font_size=14, color=(0, 240, 220), bold=True)

        # 相机硬件状态指示
        cam_status = "MOCK" if state.camera_streamer.is_mock else "ONLINE"
        cam_col = (0, 180, 255) if state.camera_streamer.is_mock else (0, 240, 100)
        cv2.circle(canvas, (self.canvas_w - 65, 25), 5, cam_col, -1)
        cv2.putText(canvas, f"CAM {cam_status}", (self.canvas_w - 55, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GRAY, 1, cv2.LINE_AA)

    def _render_left_panel(self, canvas: np.ndarray, state: HubState):
        """渲染左侧综合导航栏 (x: 0~340, y: 50~670)
        包含三层结构:
        1. 场景批次列表 (可滚动卡片流)
        2. 场景常用操作按钮 (新建 [N] / 改名 [R] / 克隆 [K])
        3. 核心工作流通道栏 (相机采图 [C] / 离线Studio [S] / 生效到生产 [P])
        """
        cv2.rectangle(canvas, (0, 50), (340, 670), self.COLOR_PANEL, -1)

        # ==== 1. 场景批次列表 ====
        draw_text(canvas, f"工况场景批次 ({len(state.scenes)})", (16, 62), font_size=16, color=self.COLOR_WHITE, bold=True)

        card_h = 66
        start_y = 88
        max_cards = 4  # 限制展示4个以留出充足空间放置操作按钮

        scroll_start = max(0, state.selected_scene_idx - max_cards + 1)
        visible_scenes = state.scenes[scroll_start: scroll_start + max_cards]

        for i, sc in enumerate(visible_scenes):
            real_idx = scroll_start + i
            is_selected = (real_idx == state.selected_scene_idx)
            is_active = (sc.scene_id == state.active_scene_id)
            cy = start_y + i * (card_h + 6)

            card_col = self.COLOR_CARD_ACTIVE if is_selected else (28, 32, 42)
            border_col = self.COLOR_ACTIVE_BORDER if is_selected else self.COLOR_BORDER

            cv2.rectangle(canvas, (10, cy), (330, cy + card_h), card_col, -1)
            cv2.rectangle(canvas, (10, cy), (330, cy + card_h), border_col, 2 if is_selected else 1)

            if is_selected:
                cv2.rectangle(canvas, (10, cy), (14, cy + card_h), (0, 255, 160), -1)

            # 主标题突出显示友好中文名称
            prefix = f"{real_idx + 1:02d}."
            display_title = f"{prefix} {sc.name}"
            title_col = (0, 255, 200) if is_selected else self.COLOR_WHITE
            draw_text(canvas, display_title, (20, cy + 6), font_size=15, color=title_col, bold=is_selected)

            # 第二行：物理唯一 ID 与张数
            id_subtitle = f"ID: {sc.scene_id[:18]} | {sc.image_count}帧"
            cv2.putText(canvas, id_subtitle, (20, cy + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GRAY, 1, cv2.LINE_AA)

            # 第三行：平差精度指标徽章
            ba_badge = f"RMSE: {sc.global_rmse_px:.2f}px" if sc.ba_solved else "未平差"
            ba_col = (0, 220, 100) if sc.ba_solved else self.COLOR_DARK_GRAY
            cv2.putText(canvas, ba_badge, (20, cy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ba_col, 1, cv2.LINE_AA)

            # 当前活动标记 / 生产发布徽章
            if is_active:
                draw_text(canvas, "[活动]", (250, cy + 6), font_size=13, color=(0, 255, 120))
            if sc.is_published:
                draw_text(canvas, "★生产", (280, cy + 42), font_size=13, color=self.COLOR_GOLD, bold=True)
            else:
                draw_text(canvas, "草稿", (285, cy + 42), font_size=12, color=self.COLOR_DARK_GRAY)

    def _draw_button(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str,
                     mouse_pos: tuple[int, int], is_active: bool = False,
                     theme_color: tuple[int, int, int] = (0, 200, 140)) -> bool:
        """统一绘制现代科技风交互按钮，带精准 Hover 高亮检测，返回是否处于 hover 状态"""
        bx, by, bw, bh = rect
        mx, my = mouse_pos
        is_hover = (bx <= mx <= bx + bw and by <= my <= by + bh)

        if is_hover:
            bg_col = (34, 46, 56)
            border_col = (0, 255, 180)  # 荧光亮绿高亮
            text_col = (0, 255, 200)
            thickness = 2
        elif is_active:
            bg_col = (28, 40, 48)
            border_col = theme_color
            text_col = self.COLOR_WHITE
            thickness = 2
        else:
            bg_col = (22, 28, 36)
            border_col = (45, 68, 62)   # 统一沉稳科技暗绿框
            text_col = (205, 225, 220)
            thickness = 1

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, thickness)

        # 文字垂直居中
        draw_text(canvas, text, (bx + 14, by + (bh - 18) // 2), font_size=14,
                  color=text_col, bold=is_hover)
        return is_hover

    def _render_left_panel(self, canvas: np.ndarray, state: HubState):
        """渲染左侧综合导航栏 (x: 0~340, y: 50~670)
        包含三层结构:
        1. 场景批次列表 (可滚动卡片流)
        2. 场景常用操作按钮 (统一科技绿风格 + Hover高亮)
        3. 核心工作流通道栏 (带 [? Help] 专属说明按钮)
        """
        cv2.rectangle(canvas, (0, 50), (340, 670), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # ==== 1. 场景批次列表 ====
        draw_text(canvas, f"工况场景批次 ({len(state.scenes)})", (16, 62), font_size=16, color=self.COLOR_WHITE, bold=True)

        card_h = 66
        start_y = 88
        max_cards = 4  # 限制展示4个以留出充足空间放置操作按钮

        scroll_start = max(0, state.selected_scene_idx - max_cards + 1)
        visible_scenes = state.scenes[scroll_start: scroll_start + max_cards]

        for i, sc in enumerate(visible_scenes):
            real_idx = scroll_start + i
            is_selected = (real_idx == state.selected_scene_idx)
            is_active = (sc.scene_id == state.active_scene_id)
            cy = start_y + i * (card_h + 6)

            card_col = self.COLOR_CARD_ACTIVE if is_selected else (28, 32, 42)
            border_col = self.COLOR_ACTIVE_BORDER if is_selected else self.COLOR_BORDER

            cv2.rectangle(canvas, (10, cy), (330, cy + card_h), card_col, -1)
            cv2.rectangle(canvas, (10, cy), (330, cy + card_h), border_col, 2 if is_selected else 1)

            if is_selected:
                cv2.rectangle(canvas, (10, cy), (14, cy + card_h), (0, 255, 160), -1)

            # 主标题突出显示友好中文名称
            prefix = f"{real_idx + 1:02d}."
            display_title = f"{prefix} {sc.name}"
            title_col = (0, 255, 200) if is_selected else self.COLOR_WHITE
            draw_text(canvas, display_title, (20, cy + 6), font_size=15, color=title_col, bold=is_selected)

            # 第二行：物理唯一 ID 与张数
            id_subtitle = f"ID: {sc.scene_id[:18]} | {sc.image_count}帧"
            cv2.putText(canvas, id_subtitle, (20, cy + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GRAY, 1, cv2.LINE_AA)

            # 第三行：平差精度指标徽章
            ba_badge = f"RMSE: {sc.global_rmse_px:.2f}px" if sc.ba_solved else "未平差"
            ba_col = (0, 220, 100) if sc.ba_solved else self.COLOR_DARK_GRAY
            cv2.putText(canvas, ba_badge, (20, cy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ba_col, 1, cv2.LINE_AA)

            # 当前活动标记 / 生产发布徽章 (交互式胶囊按钮，支持 hover 高亮与点击弹窗说明)
            badge_ax, badge_ay, badge_aw, badge_ah = 252, cy + 6, 68, 22
            a_hover = (badge_ax <= mpos[0] <= badge_ax + badge_aw and badge_ay <= mpos[1] <= badge_ay + badge_ah)
            if is_active:
                cv2.rectangle(canvas, (badge_ax, badge_ay), (badge_ax + badge_aw, badge_ay + badge_ah),
                              (24, 42, 34) if a_hover else (18, 30, 24), -1)
                cv2.rectangle(canvas, (badge_ax, badge_ay), (badge_ax + badge_aw, badge_ay + badge_ah),
                              (0, 255, 180) if a_hover else (0, 200, 120), 2 if a_hover else 1)
                draw_text(canvas, "[活动] ?", (badge_ax + 8, badge_ay + 3), font_size=12,
                          color=(0, 255, 200) if a_hover else (0, 240, 140), bold=True)
            else:
                if a_hover:
                    cv2.rectangle(canvas, (badge_ax, badge_ay), (badge_ax + badge_aw, badge_ay + badge_ah), (26, 32, 42), -1)
                    cv2.rectangle(canvas, (badge_ax, badge_ay), (badge_ax + badge_aw, badge_ay + badge_ah), (0, 200, 160), 1)
                    draw_text(canvas, "设活动 ?", (badge_ax + 8, badge_ay + 3), font_size=12, color=(0, 240, 180))

            badge_px, badge_py, badge_pw, badge_ph = 252, cy + 36, 68, 22
            p_hover = (badge_px <= mpos[0] <= badge_px + badge_pw and badge_py <= mpos[1] <= badge_py + badge_ph)
            if sc.is_published:
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (36, 40, 26) if p_hover else (26, 28, 18), -1)
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (0, 255, 255) if p_hover else self.COLOR_GOLD, 2 if p_hover else 1)
                draw_text(canvas, "★生产 ?", (badge_px + 7, badge_py + 3), font_size=12,
                          color=(100, 255, 255) if p_hover else self.COLOR_GOLD, bold=True)
            else:
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (32, 36, 44) if p_hover else (22, 25, 32), -1)
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (0, 220, 200) if p_hover else (50, 58, 72), 1)
                draw_text(canvas, "草稿 ?", (badge_px + 14, badge_py + 3), font_size=12,
                          color=(0, 220, 200) if p_hover else self.COLOR_DARK_GRAY)

        # 分割线
        div_y1 = 380
        cv2.line(canvas, (10, div_y1), (330, div_y1), self.COLOR_BORDER, 1)

        # ==== 2. 场景管理常用操作按钮 (风格统一为科技绿框，带精准 Hover 高亮) ====
        draw_text(canvas, "场景管理快捷操作", (16, div_y1 + 8), font_size=15, color=self.COLOR_CYAN, bold=True)

        btn1_y = div_y1 + 32
        self._draw_button(canvas, (10, btn1_y, 155, 36), "[+] 新建 [N]", mpos)
        self._draw_button(canvas, (175, btn1_y, 155, 36), "[R] 改名", mpos)

        btn2_y = btn1_y + 40
        self._draw_button(canvas, (10, btn2_y, 155, 36), "[K] 克隆", mpos)
        self._draw_button(canvas, (175, btn2_y, 155, 36), "[V] 目录", mpos)

        # 分割线
        div_y2 = btn2_y + 42
        cv2.line(canvas, (10, div_y2), (330, div_y2), self.COLOR_BORDER, 1)

        # ==== 3. 核心快捷工作流通道 ====
        draw_text(canvas, "核心快捷工作流通道", (16, div_y2 + 8), font_size=15, color=self.COLOR_WHITE, bold=True)

        wf_items = [
            ("[C] 原地切入相机连拍向导", "空格抓拍，直接采集到当前场景", (0, 255, 180)),
            ("[S] 启动离线 Studio 深度平差", "智能剪枝/重投影质检/立体几何", (0, 220, 255)),
            ("[P] 生效到生产系统 (发布运行地图)", "覆盖全局 config/tags_map.yaml", self.COLOR_GOLD),
        ]

        wfy = div_y2 + 32
        for idx, (title, desc, col) in enumerate(wf_items):
            is_hover = (10 <= mpos[0] <= 330 and wfy <= mpos[1] <= wfy + 44)
            card_border = (0, 255, 180) if is_hover else (38, 46, 62)
            cv2.rectangle(canvas, (10, wfy), (330, wfy + 44), (26, 32, 42) if is_hover else (22, 26, 36), -1)
            cv2.rectangle(canvas, (10, wfy), (330, wfy + 44), card_border, 2 if is_hover else 1)
            draw_text(canvas, title, (18, wfy + 4), font_size=13, color=col, bold=True)
            draw_text(canvas, desc, (18, wfy + 24), font_size=12, color=self.COLOR_GRAY)

            # 在第 3 项 [P] 生效到生产系统 右侧增加专属 [? Help] 按钮 (点击弹出说明弹窗)
            if idx == 2:
                hx, hy, hw, hh = 262, wfy + 6, 60, 32
                h_hover = (hx <= mpos[0] <= hx + hw and hy <= mpos[1] <= hy + hh)
                cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), (38, 48, 62) if h_hover else (26, 32, 42), -1)
                cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), (0, 255, 200) if h_hover else self.COLOR_GOLD, 2 if h_hover else 1)
                draw_text(canvas, "? Help", (hx + 6, hy + 7), font_size=13,
                          color=(0, 255, 200) if h_hover else self.COLOR_GOLD, bold=True)

            wfy += 48

    def _render_center_report_panel(self, canvas: np.ndarray, state: HubState, sc):
        """渲染中间栏：场景综合体检报告与几何健康看板 (x: 340~800, y: 50~670)
        移除底部冗长常驻文字，指标卡片舒展呈现，增加质检放行仪表盘！
        """
        box_x, box_y, box_w, box_h = 340, 50, 460, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 23, 30), -1)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建场景", (box_x + 120, box_y + 260), font_size=18, color=self.COLOR_GRAY)
            return

        # 栏目标题与当前场景标识
        draw_text(canvas, "场景综合体检与几何健康报告", (box_x + 16, box_y + 14), font_size=17, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (box_x + 16, box_y + 44), (box_x + box_w - 16, box_y + 44), self.COLOR_BORDER, 1)

        # 场景核心元数据卡片
        meta_y = box_y + 54
        cv2.rectangle(canvas, (box_x + 16, meta_y), (box_x + box_w - 16, meta_y + 58), (26, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 16, meta_y), (box_x + box_w - 16, meta_y + 58), (0, 180, 220), 1)
        draw_text(canvas, f"当前场景: 【{sc.name}】", (box_x + 26, meta_y + 8), font_size=16, color=(0, 240, 220), bold=True)
        draw_text(canvas, f"物理唯一ID: {sc.scene_id}", (box_x + 26, meta_y + 34), font_size=13, color=self.COLOR_GRAY)

        # 4 大体检与健康指标卡片 (舒展间距)
        cards = [
            ("1. 采样数据集规模与有效性",
             f"总采集: {sc.image_count} 帧  |  有效参与: {sc.active_image_count} 帧",
             "状态评级: 样本充足 (≥10帧达标)" if sc.image_count >= 10 else "状态评级: 样本偏少 (建议按 [C] 继续采图)",
             (0, 240, 100) if sc.image_count >= 10 else (0, 180, 255)),

            ("2. 空间标靶拓扑与参考基准",
             f"基准原点: Tag #{sc.origin_tag_id}  |  X轴对准: Tag #{sc.x_axis_tag_id}",
             "已知标靶拓扑已锚定，空间立体几何约束锁定",
             (0, 220, 255)),

            ("3. 两阶段 BA 平差与重投影精度",
             f"全局 RMSE 误差: {sc.global_rmse_px:.3f} px" if sc.ba_solved else "尚未执行离线平差 (暂无精度数据)",
             "精度评级: 极优 (误差 < 0.20px)" if (sc.ba_solved and sc.global_rmse_px < 0.2) else
             ("精度评级: 良好" if sc.ba_solved else "待平差: 请按 [S] 启动 Studio 计算"),
             (0, 255, 160) if sc.ba_solved else self.COLOR_GRAY),

            ("4. 生产系统生效与运行状态",
             "★ 已生效为全局生产运行地图 (生效中)" if sc.is_published else "草稿沙盒状态 (尚未生效至生产配置)",
             "全局生产路径: config/tags_map.yaml  (可点击左侧 [? Help] 查看详情)",
             self.COLOR_GOLD if sc.is_published else (140, 150, 165))
        ]

        cy = meta_y + 70
        for title, val_line, sub_line, col in cards:
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 62), (26, 30, 40), -1)
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 62), (38, 44, 58), 1)
            draw_text(canvas, title, (box_x + 28, cy + 6), font_size=14, color=self.COLOR_WHITE, bold=True)
            draw_text(canvas, val_line, (box_x + 28, cy + 26), font_size=13, color=col, bold=True)
            draw_text(canvas, sub_line, (box_x + 28, cy + 44), font_size=12, color=self.COLOR_GRAY)
            cy += 74

        # 底部升级为清爽的【工程质量放行评定面板】
        eval_y = cy + 6
        cv2.rectangle(canvas, (box_x + 16, eval_y), (box_x + box_w - 16, box_y + box_h - 16), (24, 28, 38), -1)
        cv2.rectangle(canvas, (box_x + 16, eval_y), (box_x + box_w - 16, box_y + box_h - 16), (0, 180, 200), 1)

        draw_text(canvas, "★ 场景质量综合评定与放行指引", (box_x + 26, eval_y + 12), font_size=15, color=(0, 240, 220), bold=True)

        if sc.ba_solved and sc.global_rmse_px < 0.2:
            verdict_text = "🟢 [极优放行] 该场景平差精度达标 (RMSE < 0.20px)，几何精度稳定！"
            action_advice = "建议操作: 按 [P] 键一键生效到生产系统，或进入在线 AR [A] 实景验收。"
            v_col = (0, 255, 160)
        elif sc.ba_solved:
            verdict_text = "🟡 [常规放行] 该场景平差已收敛，可直接投入常规抓取定位。"
            action_advice = "建议操作: 可按 [S] 启动 Studio 执行智能残差剪枝以进一步压低误差。"
            v_col = self.COLOR_GOLD
        else:
            verdict_text = "⚪ [未求解] 当前场景尚未执行离线两阶段 BA 空间建图与平差。"
            action_advice = "建议操作: 确保采图 ≥10 帧后，按下 [S] 键启动离线平差工作站。"
            v_col = self.COLOR_GRAY

        draw_text(canvas, verdict_text, (box_x + 26, eval_y + 40), font_size=13, color=v_col, bold=True)
        draw_text(canvas, action_advice, (box_x + 26, eval_y + 64), font_size=12, color=self.COLOR_WHITE)
        draw_text(canvas, "提示: 点击左侧 [? Help] 或按 [H] 键可随时了解生产生效机制。", (box_x + 26, eval_y + 88), font_size=12, color=self.COLOR_GRAY)

    def _render_right_album_panel(self, canvas: np.ndarray, state: HubState, sc):
        """渲染最右侧栏：采样相册画廊与大图预览视口 (x: 800~1280, y: 50~670)"""
        box_x, box_y, box_w, box_h = 800, 50, 480, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题
        draw_text(canvas, f"采样相册 ({len(state.current_images)} 帧)", (box_x + 16, box_y + 14), font_size=17, color=self.COLOR_WHITE, bold=True)

        # 顶部实体按钮组: [◀] [▶] [⛶ 全宽放大 [F]] (统一科技绿框 + Hover 高亮)
        self._draw_button(canvas, (box_x + box_w - 224, box_y + 8, 40, 32), "[<]", mpos)
        self._draw_button(canvas, (box_x + box_w - 178, box_y + 8, 40, 32), "[>]", mpos)
        self._draw_button(canvas, (box_x + box_w - 132, box_y + 8, 118, 32), "[F] 全宽放大", mpos)

        if not state.current_images:
            empty_box_y = box_y + 50
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 100), (22, 26, 36), -1)
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 100), self.COLOR_BORDER, 1)
            draw_text(canvas, "当前场景尚未采集任何照片！", (box_x + 30, empty_box_y + 24), font_size=16, color=(0, 200, 240), bold=True)
            draw_text(canvas, "请直接按键盘 [C] 键，原地进入相机连拍向导抓拍照片。", (box_x + 30, empty_box_y + 56), font_size=13, color=self.COLOR_GRAY)
            return

        # 1. 顶部缩略图水平滚动带 (y: 84~168)
        tw, th = 98, 62
        pad = 8
        visible_count = 4
        offset = state.image_strip_offset
        visible_imgs = state.current_images[offset: offset + visible_count]

        for idx, img_path in enumerate(visible_imgs):
            real_idx = offset + idx
            is_cur = (real_idx == state.selected_image_idx)
            x = box_x + 16 + idx * (tw + pad)
            y = box_y + 46

            thumb = state.get_thumbnail(img_path, tw, th)
            if thumb is not None:
                canvas[y:y + th, x:x + tw] = thumb

            border_col = (0, 255, 180) if is_cur else self.COLOR_BORDER
            cv2.rectangle(canvas, (x, y), (x + tw, y + th), border_col, 2 if is_cur else 1)

            base_name = os.path.basename(img_path)
            cv2.rectangle(canvas, (x, y + th - 15), (x + tw, y + th), (10, 10, 14), -1)
            cv2.putText(canvas, base_name[:12], (x + 3, y + th - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 180) if is_cur else self.COLOR_GRAY, 1, cv2.LINE_AA)

            if is_cur:
                cv2.line(canvas, (x + tw // 2 - 4, y + th + 3), (x + tw // 2 + 4, y + th + 3), (0, 255, 180), 2)

        # 2. 单帧照片高画质大图视口 (y: 172~656)
        prev_box_y = box_y + 118
        prev_box_h = box_h - 130
        cv2.rectangle(canvas, (box_x + 16, prev_box_y), (box_x + box_w - 16, prev_box_y + prev_box_h), (18, 21, 28), -1)
        cv2.rectangle(canvas, (box_x + 16, prev_box_y), (box_x + box_w - 16, prev_box_y + prev_box_h), self.COLOR_BORDER, 1)

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=430, max_h=390)

        img_title = f"单帧预览: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 28, prev_box_y + 10), font_size=15, color=self.COLOR_CYAN, bold=True)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + 16 + (box_w - 32 - pw) // 2
            py = prev_box_y + 38 + (400 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (50, 56, 72), 1)

        draw_text(canvas, "[< / >] 左右键选片  |  滚轮快速切片  |  [F] 放大预览",
                  (box_x + 60, prev_box_y + prev_box_h - 22), font_size=13, color=self.COLOR_GRAY)

    def _render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """全宽自适应大图视口 (按 F 键展开，横跨中间和右侧，x: 340~1280)"""
        box_x, box_y, box_w, box_h = 340, 50, 940, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 20, 26), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)
        mpos = (state.mouse_x, state.mouse_y)

        # 标题与右上角实体按钮
        if not state.current_images:
            draw_text(canvas, "当前场景无图片", (box_x + 400, box_y + 280), font_size=20, color=self.COLOR_DARK_GRAY)
            self._draw_button(canvas, (box_x + box_w - 180, box_y + 10, 160, 32), "[F] 退出全宽放大", mpos)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=910, max_h=520)

        img_title = f"全宽自适应大图预览: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 20, box_y + 14), font_size=17, color=(0, 255, 200), bold=True)

        # 右上角实体按钮组: [◀ 上一张] [下一张 ▶] [F 退出放大]
        self._draw_button(canvas, (box_x + box_w - 364, box_y + 10, 84, 32), "[<] 上张", mpos)
        self._draw_button(canvas, (box_x + box_w - 274, box_y + 10, 84, 32), "[>] 下张", mpos)
        self._draw_button(canvas, (box_x + box_w - 184, box_y + 10, 164, 32), "[F] 退出全宽放大", mpos)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 48 + (520 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 70, 90), 1)

        draw_text(canvas, "[< / >] 切换相册大图   |   滚轮顺畅切换   |   [Space] 空间抓拍   |   [F] 恢复标准看板",
                  (box_x + 210, box_y + box_h - 24), font_size=14, color=self.COLOR_WHITE)

    def _render_capture_viewport(self, canvas: np.ndarray, state: HubState):
        """渲染原地相机取流视口 (x: 0~1280, y: 50~670)"""
        vw, vh = 1280, 620
        stream_view = self.capture_stream_renderer.render_stream_viewport(state, vw, vh)
        canvas[50:50 + vh, 0:vw] = stream_view

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部状态与快捷键导航栏 (670~720px)"""
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)

        now = time.time()
        if state.toast_time > now:
            draw_text(canvas, f"[系统反馈] {state.toast_msg}", (20, 684), font_size=16, color=(0, 255, 200), bold=True)
        else:
            if state.is_help_modal_open:
                draw_text(canvas, "【生产机制解析窗已激活】[ESC / H] 关闭说明窗口  |  按 [P] 可直接一键生效到生产系统",
                          (20, 686), font_size=14, color=self.COLOR_GOLD, bold=True)
            elif state.is_toolbox_open:
                draw_text(canvas, "【工具箱已激活】[S] Studio平差  [A] 在线AR验证  [L] 盲测体检  [D] 漏检切片  [T] 标靶图纸  [W] 白名单  [ESC / M] 关闭",
                          (20, 686), font_size=14, color=(0, 255, 200), bold=True)
            elif state.mode == HubState.MODE_CAPTURE:
                draw_text(canvas, "[Space] 抓拍存入当前场景   |   [ESC / C] 退出采图返回三栏看板   |   [S] 立即平差",
                          (20, 686), font_size=15, color=self.COLOR_WHITE)
            else:
                draw_text(canvas, "[M] 工具箱  [N] 新建  [R] 改名  [K] 克隆  [↑/↓] 选场景  [C] 连拍  [S] 平差  [P] 生效生产  [H] 生产说明  [F] 放大",
                          (20, 686), font_size=14, color=self.COLOR_WHITE)

    def _render_toolbox_modal(self, canvas: np.ndarray, state: HubState):
        """渲染中央悬浮置顶的【标定工具箱综合菜单】(880x520)"""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_w, self.canvas_h), (8, 10, 14), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        modal_w, modal_h = 880, 520
        mx = (self.canvas_w - modal_w) // 2
        my = (self.canvas_h - modal_h) // 2
        mpos = (state.mouse_x, state.mouse_y)

        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (20, 24, 32), -1)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (0, 200, 240), 2)
        cv2.rectangle(canvas, (mx + 4, my + 4), (mx + modal_w - 4, my + modal_h - 4), (40, 50, 66), 1)

        # 标题栏
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + 54), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 54), (mx + modal_w, my + 54), self.COLOR_BORDER, 1)

        cv2.circle(canvas, (mx + 22, my + 27), 6, (0, 240, 220), -1)
        draw_text(canvas, "★ AprilTag 视觉标定综合工具箱总菜单 (Toolbox Hub)", (mx + 36, my + 15),
                  font_size=18, color=self.COLOR_WHITE, bold=True)

        # 右上角 [X] 关闭按钮
        self._draw_button(canvas, (mx + modal_w - 120, my + 11, 104, 32), "[X] 关闭 [ESC]", mpos)

        draw_text(canvas, "一站式极速唤起系统内所有独立 OpenCV 交互式图形化诊断、平差、图纸生成与在线验证工具：",
                  (mx + 28, my + 66), font_size=13, color=(0, 200, 240))

        tools = [
            ("[S] 离线 Studio 深度平差工作站",
             "整合多视角样本网格、两阶段 BA 平差求解、智能残差剪枝与全量质检体检报告",
             (0, 240, 160), "tag_offline_studio.py"),

            ("[A] 在线 AR 精度体检与 3D 虚实融合",
             "相机实时高帧率取流、3D坐标轴/立体棱柱空间叠加、多帧平滑毫米级位姿锁定",
             (0, 220, 255), "tag_calibration_verifier.py"),

            ("[L] 离线留一交叉验证盲测工作台",
             "对当前场景全量执行留一盲测 (LOO)，绘制残差矢量分布并评估相机外参鲁棒性",
             self.COLOR_GOLD, "tag_offline_verifier.py"),

            ("[D] 单帧漏检病因深度切片与梯度诊断",
             "CLAHE 双尺度增强、16级网格自适应阈值，深度切片排查候选四边形淘汰病因",
             (200, 140, 255), "tag_image_diagnostics.py"),

            ("[T] 标靶图纸生成与 1:1 A4 打印排版",
             "自动生成 0~29 号高精 AprilTag 矢量图纸及工业 1:1 A4 标定板排版 PDF 文件",
             (120, 240, 100), "generate_tags_and_docs.py"),

            ("[W] 标靶 ID 白名单管理与探索放行",
             "查看或指定有效标靶 ID 集合，屏蔽车间杂乱反光外点干扰，锚定空间参考系",
             (80, 160, 255), "whitelist_manager"),
        ]

        cw, ch = 398, 86
        col_xs = [mx + 28, mx + 454]
        row_ys = [my + 96, my + 196, my + 296]

        for idx, (title, desc, col, script) in enumerate(tools):
            col_i = idx % 2
            row_i = idx // 2
            x = col_xs[col_i]
            y = row_ys[row_i]

            is_hover = (x <= mpos[0] <= x + cw and y <= mpos[1] <= y + ch)
            cv2.rectangle(canvas, (x, y), (x + cw, y + ch), (32, 40, 52) if is_hover else (26, 31, 42), -1)
            cv2.rectangle(canvas, (x, y), (x + cw, y + ch), (0, 255, 180) if is_hover else (44, 52, 70), 2 if is_hover else 1)
            cv2.rectangle(canvas, (x, y), (x + 4, y + ch), col, -1)

            draw_text(canvas, title, (x + 14, y + 10), font_size=15, color=col, bold=True)
            desc_line1 = desc[:28]
            desc_line2 = desc[28:56]
            draw_text(canvas, desc_line1, (x + 14, y + 36), font_size=12, color=self.COLOR_WHITE)
            if desc_line2:
                draw_text(canvas, desc_line2, (x + 14, y + 56), font_size=12, color=self.COLOR_GRAY)

            cv2.rectangle(canvas, (x + cw - 72, y + ch - 22), (x + cw - 8, y + ch - 6), (16, 22, 30), -1)
            cv2.putText(canvas, "CLICK", (x + cw - 62, y + ch - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

        footer_y = my + modal_h - 40
        cv2.line(canvas, (mx + 20, footer_y), (mx + modal_w - 20, footer_y), (36, 44, 58), 1)
        draw_text(canvas, "★ 操作提示: 鼠标直接点击对应工具卡片，或直接按下键盘快捷键 [S / A / L / D / T / W] 即可秒级拉起！",
                  (mx + 30, footer_y + 12), font_size=13, color=self.COLOR_GOLD)

    def _render_help_modal(self, canvas: np.ndarray, state: HubState):
        """渲染置顶居中的【场景状态机制解析：活动 (Active) vs 生产 (Production)】深度说明看板 (940x530)"""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_w, self.canvas_h), (8, 10, 14), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        modal_w, modal_h = 940, 530
        mx = (self.canvas_w - modal_w) // 2
        my = (self.canvas_h - modal_h) // 2
        mpos = (state.mouse_x, state.mouse_y)

        # 底板与多层线框
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (20, 24, 32), -1)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (0, 220, 160), 2)
        cv2.rectangle(canvas, (mx + 4, my + 4), (mx + modal_w - 4, my + modal_h - 4), (40, 50, 66), 1)

        # 标题栏
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + 54), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 54), (mx + modal_w, my + 54), self.COLOR_BORDER, 1)

        cv2.circle(canvas, (mx + 24, my + 27), 6, self.COLOR_GOLD, -1)
        draw_text(canvas, "★ 工业级场景状态机制解析:【活动 (Active)】与【生产 (Production)】的区别", (mx + 38, my + 15),
                  font_size=17, color=self.COLOR_WHITE, bold=True)

        # 右上角 [X] 关闭按钮
        self._draw_button(canvas, (mx + modal_w - 116, my + 11, 100, 32), "[X] 关闭 [H]", mpos)

        # 1. 顶部核心理念
        intro_text = "核心架构：严格实行【研发实验沙盒】与【车间流水线作业】的物理安全隔离与闭环发布！"
        draw_text(canvas, intro_text, (mx + 26, my + 66), font_size=14, color=(0, 240, 220), bold=True)

        # 2. 左右两大核心对比卡片 (高度 236px)
        card_y = my + 94
        card_w = (modal_w - 68) // 2  # 436

        # 2.1 左卡片：【活动】场景 (Active Workspace)
        cx1 = mx + 26
        cv2.rectangle(canvas, (cx1, card_y), (cx1 + card_w, card_y + 236), (22, 32, 36), -1)
        cv2.rectangle(canvas, (cx1, card_y), (cx1 + card_w, card_y + 236), (0, 220, 140), 2)
        cv2.rectangle(canvas, (cx1, card_y), (cx1 + card_w, card_y + 36), (18, 26, 30), -1)
        draw_text(canvas, "🟢 【活动】场景 (Active Workspace)", (cx1 + 14, card_y + 8), font_size=15, color=(0, 255, 160), bold=True)

        active_points = [
            ("概念定义", "当前研发与标定聚焦的操作台沙盒 (类似 Git 本地分支)"),
            ("连拍归档", "按 [C] 进入相机连拍抓拍的照片，自动保存于此场景"),
            ("离线平差", "启动 Studio 平差、留一盲测、诊断切片默认载入此数据"),
            ("如何切换", "在场景列表中按 [Enter] 回车键或点击徽章即可随时切换"),
            ("安全边界", "完全沙盒隔离！无论如何采图平差，流水线机械臂零影响"),
        ]
        py = card_y + 44
        for label, desc in active_points:
            draw_text(canvas, f"• {label}:", (cx1 + 14, py), font_size=12, color=(0, 220, 180), bold=True)
            d1 = desc[:28]
            d2 = desc[28:]
            draw_text(canvas, d1, (cx1 + 84, py), font_size=12, color=self.COLOR_WHITE)
            if d2:
                py += 18
                draw_text(canvas, d2, (cx1 + 84, py), font_size=11, color=self.COLOR_GRAY)
            py += 24

        # 2.2 右卡片：【★生产】地图 (Production Release)
        cx2 = cx1 + card_w + 16
        cv2.rectangle(canvas, (cx2, card_y), (cx2 + card_w, card_y + 236), (32, 28, 20), -1)
        cv2.rectangle(canvas, (cx2, card_y), (cx2 + card_w, card_y + 236), self.COLOR_GOLD, 2)
        cv2.rectangle(canvas, (cx2, card_y), (cx2 + card_w, card_y + 36), (24, 20, 14), -1)
        draw_text(canvas, "★ 【生产】地图 (Production Release)", (cx2 + 14, card_y + 8), font_size=15, color=self.COLOR_GOLD, bold=True)

        prod_points = [
            ("概念定义", "车间现场机械臂定位唯一信赖的真实世界几何基准"),
            ("物理路径", "对应项目根目录下的全局唯一文件 config/tags_map.yaml"),
            ("生效机制", "选中满意场景后，按 [P] 键一键安全原子覆盖发布"),
            ("历史保护", "发布瞬间自动创建时间戳 .bak 备份文件，确保可追溯"),
            ("放行标准", "必须经多视角采图平差、RMSE 达标后方可发布 (极优放行)"),
        ]
        py = card_y + 44
        for label, desc in prod_points:
            draw_text(canvas, f"• {label}:", (cx2 + 14, py), font_size=12, color=self.COLOR_GOLD, bold=True)
            d1 = desc[:28]
            d2 = desc[28:]
            draw_text(canvas, d1, (cx2 + 84, py), font_size=12, color=self.COLOR_WHITE)
            if d2:
                py += 18
                draw_text(canvas, d2, (cx2 + 84, py), font_size=11, color=self.COLOR_GRAY)
            py += 24

        # 3. 底部完整闭环工作流导引 (y: card_y + 246)
        flow_y = card_y + 246
        cv2.rectangle(canvas, (mx + 26, flow_y), (mx + modal_w - 26, flow_y + 92), (20, 25, 34), -1)
        cv2.rectangle(canvas, (mx + 26, flow_y), (mx + modal_w - 26, flow_y + 92), (40, 55, 75), 1)
        cv2.rectangle(canvas, (mx + 26, flow_y), (mx + 30, flow_y + 92), (0, 200, 240), -1)

        draw_text(canvas, "💡 工业工程标准作业流 (SOP 黄金闭环):", (mx + 42, flow_y + 8), font_size=14, color=(0, 220, 255), bold=True)
        draw_text(canvas, "步骤 1: 新建/克隆场景 -> 按 [Enter] 设为【活动】场景 -> 按 [C] 原地抓拍多视角照片 (≥10帧)", (mx + 42, flow_y + 32), font_size=12, color=self.COLOR_WHITE)
        draw_text(canvas, "步骤 2: 按 [S] 启动 Studio 离线平差工作站 -> 智能残差剪枝 -> 质检评定 RMSE < 0.20px 极优放行", (mx + 42, flow_y + 52), font_size=12, color=(0, 240, 180))
        draw_text(canvas, "步骤 3: 达到精度指标后，按 [P] 键一键发布为【★生产】地图，现场机械臂秒级热更新！", (mx + 42, flow_y + 72), font_size=12, color=self.COLOR_GOLD, bold=True)

        # 4. 底部关闭操作指引
        footer_y = my + modal_h - 36
        draw_text(canvas, "★ 提示: 点击左侧卡片上的 [活动] 或 ★生产 徽章、点击 [? Help] 或直接按键盘 [ESC / H] 即可秒级开关！",
                  (mx + 32, footer_y), font_size=13, color=self.COLOR_GRAY)



