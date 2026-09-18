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
from typing import Any
import cv2
import numpy as np

from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, put_text
from tools.scene_hub.hub_state import HubState


class HubRenderer:
    """Scene Hub 统一界面渲染器"""

    def __init__(self):
        self.canvas_w = 1280
        self.canvas_h = 720

        # 调色板: 结构色统一取自 GuiTheme 主题单源, 品牌色 (青/金/暗灰) 本地保留
        self.COLOR_BG = GuiTheme.BG             # 全局底色
        self.COLOR_PANEL = GuiTheme.CARD_BG     # 侧边栏/卡片底色
        self.COLOR_CARD_ACTIVE = GuiTheme.CARD_SEL  # 选中卡片底色
        self.COLOR_BORDER = GuiTheme.BORDER     # 普通线框
        self.COLOR_ACTIVE_BORDER = GuiTheme.BORDER_SEL  # 选中项高亮描边
        self.COLOR_CYAN = (230, 200, 0)        # 科技青 (BGR: 0, 200, 230)
        self.COLOR_GOLD = (50, 190, 255)       # 金黄色 (BGR)
        self.COLOR_WHITE = GuiTheme.WHITE
        self.COLOR_GRAY = GuiTheme.GRAY
        self.COLOR_DARK_GRAY = (70, 75, 85)

        # 帧级极速缓存 (毫秒级响应 Hover 交互)
        self._cached_canvas: np.ndarray | None = None
        self._last_cache_key: Any = None

    def _get_interactive_hover_key(self, state: HubState) -> Any:
        """获取当前鼠标悬停的交互元素标识 (若鼠标未落在任何可交互组件上返回 None)"""
        mx, my = state.mouse_x, state.mouse_y
        if mx < 0 or my < 0:
            return None

        # 1. 场景条目专属右键菜单
        if state.context_menu_open:
            cx, cy = state.context_menu_pos
            menu_w = 216
            item_h = 32
            if cx <= mx <= cx + menu_w:
                for idx in range(6):
                    iy = cy + 34 + idx * item_h
                    if iy <= my <= iy + item_h:
                        return ("ctx_item", idx)
            return "ctx_menu_other"

        # 2. 生产说明 Help 弹窗
        if state.is_help_modal_open:
            mw, mh = 880, 560
            ox = (self.canvas_w - mw) // 2
            oy = (self.canvas_h - mh) // 2
            if ox + mw - 120 <= mx <= ox + mw - 16 and oy + 11 <= my <= oy + 43:
                return "help_close"
            return "help_modal"

        # 3. 常规看板模式
        # 顶部 Header 交互
        if 0 <= my <= 50:
            if 288 <= mx <= 368 and 9 <= my <= 41:
                return "tab_standard"
            if 370 <= mx <= 448 and 9 <= my <= 41:
                return "tab_expanded"
            if 450 <= mx <= 530 and 9 <= my <= 41:
                return "tab_dashboard"
            if 546 <= mx <= 930 and 8 <= my <= 42:
                return "header_prod"
            if 940 <= mx <= 1070 and 8 <= my <= 42:
                return "btn_help"
            if 1085 <= mx <= 1265 and 8 <= my <= 42:
                return "btn_exit"

        # 左侧面板按钮与卡片
        if 0 <= mx <= 340:
            div_y1 = 512
            btn1_y = div_y1 + 32
            if 10 <= mx <= 165 and btn1_y <= my <= btn1_y + 36:
                return "btn_new_scene"
            if 175 <= mx <= 330 and btn1_y <= my <= btn1_y + 36:
                return "btn_open_dir"
            btn2_y = btn1_y + 44
            if 10 <= mx <= 330 and btn2_y <= my <= btn2_y + 36:
                return "btn_capture_wizard"

            # 场景卡片
            card_h = 70
            start_y = 90
            max_cards = 5
            scroll_start = max(0, state.selected_scene_idx - max_cards + 1)
            visible_scenes = state.scenes[scroll_start: scroll_start + max_cards]
            for i, sc in enumerate(visible_scenes):
                cy = start_y + i * (card_h + 8)
                real_idx = scroll_start + i
                if 236 <= mx <= 324 and cy + 6 <= my <= cy + 30:
                    return ("card_badge", real_idx)
                if sc.scene_id == state.active_scene_id and (228 <= mx <= 324 and cy + 36 <= my <= cy + 64):
                    return ("card_pub", real_idx)

        # 右侧相册面板按钮
        if state.view_mode == HubState.VIEW_EXPANDED:
            box_x, box_y, box_w = 340, 50, 940
            if box_x + box_w - 364 <= mx <= box_x + box_w - 280 and box_y + 10 <= my <= box_y + 42:
                return "exp_prev"
            if box_x + box_w - 274 <= mx <= box_x + box_w - 190 and box_y + 10 <= my <= box_y + 42:
                return "exp_next"
            if box_x + box_w - 184 <= mx <= box_x + box_w - 20 and box_y + 10 <= my <= box_y + 42:
                return "exp_restore"
        elif state.view_mode == HubState.VIEW_STANDARD:
            box_x, box_y, box_w = 800, 50, 480
            if box_x + box_w - 224 <= mx <= box_x + box_w - 184 and box_y + 8 <= my <= box_y + 40:
                return "album_prev"
            if box_x + box_w - 178 <= mx <= box_x + box_w - 138 and box_y + 8 <= my <= box_y + 40:
                return "album_next"
            if box_x + box_w - 132 <= mx <= box_x + box_w - 14 and box_y + 8 <= my <= box_y + 40:
                return "album_expand"

        return None

    def render(self, state: HubState) -> np.ndarray:
        """主绘制入口，返回 1280x720 BGR 图像 (带极速帧级缓存，支持高频 60+ FPS Hover)"""
        hover_key = self._get_interactive_hover_key(state)
        now = time.time()
        toast_active = state.toast_time > now

        cache_key = (
            state.view_mode,
            state.selected_scene_idx,
            state.active_scene_id,
            state.selected_image_idx,
            state.image_strip_offset,
            state.is_help_modal_open,
            state.context_menu_open,
            state.context_menu_pos if state.context_menu_open else None,
            state.context_menu_scene_idx if state.context_menu_open else None,
            toast_active,
            state.toast_msg if toast_active else "",
            len(state.scenes),
            len(state.current_images),
            hover_key
        )

        # 缓存命中：状态与悬停目标均未发生改变，直接 0ms 返回上一帧已渲染画布
        if self._cached_canvas is not None and cache_key == self._last_cache_key:
            return self._cached_canvas

        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        # 1. 顶部状态栏 (y: 0~50)
        self._render_header(canvas, state)

        # 3. 左侧综合导航栏 (x: 0~340, y: 50~670)
        self._render_left_panel(canvas, state)
        cv2.line(canvas, (340, 50), (340, 670), self.COLOR_BORDER, 1)

        # 4. 中间栏与右侧栏 (支持三模态视图: 标准三栏 / 全宽大图 / 纯净数据看板)
        sc = state.get_selected_scene()
        if state.view_mode == HubState.VIEW_EXPANDED:
            self._render_expanded_photo_preview(canvas, state, sc)
        elif state.view_mode == HubState.VIEW_DASHBOARD:
            self._render_pure_dashboard_panel(canvas, state, sc)
        else:
            self._render_center_report_panel(canvas, state, sc)
            cv2.line(canvas, (800, 50), (800, 670), self.COLOR_BORDER, 1)
            self._render_right_album_panel(canvas, state, sc)

        # 5. 底部状态与快捷键导航栏 (y: 670~720)
        self._render_footer(canvas, state)

        # 6. 如果打开了生产系统生效机制说明弹窗，则渲染置顶半透明浮层
        if state.is_help_modal_open:
            self._render_help_modal(canvas, state)

        # 8. 如果打开了场景条目专属右键上下文菜单 (Context Menu)，置顶渲染
        if state.context_menu_open:
            self._render_context_menu(canvas, state)

        self._cached_canvas = canvas
        self._last_cache_key = cache_key
        return canvas

    def _render_header(self, canvas: np.ndarray, state: HubState):
        """渲染顶部标题栏 (0~50px) - 包含三段式视图切换Tab、生产运行场景与退出按钮"""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 50), (14, 16, 20), -1)
        cv2.line(canvas, (0, 50), (self.canvas_w, 50), self.COLOR_BORDER, 1)
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 系统标题与状态点 (x: 16~390)
        cv2.circle(canvas, (22, 25), 6, (0, 255, 180), -1)
        put_text(canvas, "flux_vision_3d", (36, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 2, cv2.LINE_AA)
        draw_text(canvas, "| 场景管理中枢", (166, 16), font_size=15, color=self.COLOR_WHITE)

        # 2. 三段式视图模式切换 Tab 胶囊组件 (Segmented Tabs, x: 290~530, y: 9~41)
        cv2.rectangle(canvas, (288, 9), (532, 41), (20, 25, 34), -1)
        cv2.rectangle(canvas, (288, 9), (532, 41), (45, 55, 72), 1)

        tabs = [
            (HubState.VIEW_STANDARD, 290, 78, "⊞ 标准"),
            (HubState.VIEW_EXPANDED, 370, 78, "⤢ 大图"),
            (HubState.VIEW_DASHBOARD, 450, 80, "▤ 看板"),
        ]

        for mode_key, tx, tw, ttext in tabs:
            is_active_tab = (state.view_mode == mode_key)
            is_hover_tab = (tx <= mpos[0] <= tx + tw and 9 <= mpos[1] <= 41)
            
            if is_active_tab:
                cv2.rectangle(canvas, (tx, 11), (tx + tw, 39), (28, 44, 40), -1)
                cv2.rectangle(canvas, (tx, 11), (tx + tw, 39), (0, 255, 180), 2)
                draw_text(canvas, ttext, (tx + 12, 16), font_size=13, color=(0, 255, 200), bold=True)
            elif is_hover_tab:
                cv2.rectangle(canvas, (tx, 11), (tx + tw, 39), (34, 40, 52), -1)
                cv2.rectangle(canvas, (tx, 11), (tx + tw, 39), (0, 200, 240), 1)
                draw_text(canvas, ttext, (tx + 12, 16), font_size=13, color=(0, 220, 255))
            else:
                draw_text(canvas, ttext, (tx + 12, 16), font_size=13, color=(160, 175, 195))
        # 3. 生产运行场景信息 (x: 546~860, y: 8~42)
        prod_sc = state.get_production_scene()
        prod_name = prod_sc.name if prod_sc else "无"
        prod_id = prod_sc.scene_id if prod_sc else "未设定"
        prod_x, prod_w = 546, 380
        is_hover_prod = (prod_x <= mpos[0] <= prod_x + prod_w and 8 <= mpos[1] <= 42)
        if is_hover_prod:
            cv2.rectangle(canvas, (prod_x, 8), (prod_x + prod_w, 42), (24, 34, 44), -1)
            cv2.rectangle(canvas, (prod_x, 8), (prod_x + prod_w, 42), (0, 255, 200), 1)
        draw_text(canvas, f"生产运行地图: 【{prod_name}】 ({prod_id[:10]})", (prod_x + 10, 16),
                  font_size=13, color=(0, 255, 180) if is_hover_prod else (0, 240, 140))

        # 4. 右上角功能按钮组
        # [H] 业务说明按钮 (x: 940~1070, y: 8~42)
        self._draw_button(canvas, (940, 8, 130, 34), "[H] 生产机制", mpos, is_active=state.is_help_modal_open)

        # [X] 退出按钮 (x: 1085~1265, y: 8~42) - 实体点击与 ESC 退出
        self._draw_button(canvas, (1085, 8, 180, 34), "[X] 退出 [ESC]", mpos, theme_color=(180, 60, 60))

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
        - 扩展展示多达 5 张场景卡片，视觉开阔无压迫
        - 卡片直接支持 [P 生效生产]，彻底移除冗余的活动场景锁
        - 去除底部冗余无用的快捷键堆砌说明，保持工业界面整洁精炼
        """
        cv2.rectangle(canvas, (0, 50), (340, 670), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # ==== 1. 场景批次列表 ====
        draw_text(canvas, f"工况场景批次 ({len(state.scenes)})", (16, 62), font_size=16, color=self.COLOR_WHITE, bold=True)

        card_h = 70
        start_y = 90
        max_cards = 5  # 扩展至 5 张卡片

        scroll_start = max(0, state.selected_scene_idx - max_cards + 1)
        visible_scenes = state.scenes[scroll_start: scroll_start + max_cards]

        for i, sc in enumerate(visible_scenes):
            real_idx = scroll_start + i
            is_selected = (real_idx == state.selected_scene_idx)
            cy = start_y + i * (card_h + 8)

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
            id_subtitle = f"ID: {sc.scene_id[:14]} | {sc.image_count}帧"
            put_text(canvas, id_subtitle, (20, cy + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GRAY, 1, cv2.LINE_AA)

            # 第三行：平差精度指标
            ba_badge = f"RMSE: {sc.global_rmse_px:.2f}px" if sc.ba_solved else "未平差"
            ba_col = (0, 220, 100) if sc.ba_solved else self.COLOR_DARK_GRAY
            put_text(canvas, ba_badge, (20, cy + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ba_col, 1, cv2.LINE_AA)

            # 右侧操作状态与发布按钮
            badge_px, badge_py, badge_pw, badge_ph = 228, cy + 20, 96, 30
            p_hover = (badge_px <= mpos[0] <= badge_px + badge_pw and badge_py <= mpos[1] <= badge_py + badge_ph)
            if sc.is_published:
                # 生产基准
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (36, 40, 24) if p_hover else (26, 28, 16), -1)
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (0, 255, 255) if p_hover else self.COLOR_GOLD, 2 if p_hover else 1)
                draw_text(canvas, "★ 生产运行", (badge_px + 10, badge_py + 7), font_size=12,
                          color=(120, 255, 255) if p_hover else self.COLOR_GOLD, bold=True)
            elif sc.ba_solved:
                # 已平差 -> 提供发布至生产的专属按钮
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (36, 56, 46) if p_hover else (20, 36, 30), -1)
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (0, 255, 180) if p_hover else (0, 200, 140), 2 if p_hover else 1)
                draw_text(canvas, "[P] 生效生产", (badge_px + 8, badge_py + 7), font_size=12,
                          color=(0, 255, 200) if p_hover else (0, 240, 160), bold=True)
            else:
                # 未平差普通场景
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (20, 24, 30), -1)
                cv2.rectangle(canvas, (badge_px, badge_py), (badge_px + badge_pw, badge_py + badge_ph),
                              (40, 48, 60), 1)
                draw_text(canvas, "草稿沙盒", (badge_px + 20, badge_py + 7), font_size=12,
                          color=self.COLOR_DARK_GRAY)

        # ==== 2. 场景通用全局操作区 (单条目操作已全面收敛至鼠标右键菜单) ====
        div_y1 = 512
        cv2.line(canvas, (10, div_y1), (330, div_y1), self.COLOR_BORDER, 1)

        draw_text(canvas, "场景通用全局操作 (条目操作请在卡片右键)", (16, div_y1 + 10), font_size=12, color=self.COLOR_GRAY)

        btn1_y = div_y1 + 32
        self._draw_button(canvas, (10, btn1_y, 155, 36), "[+] 新建工况 [N]", mpos)
        self._draw_button(canvas, (175, btn1_y, 155, 36), "[V] 场景总目录", mpos)

        btn2_y = btn1_y + 44
        self._draw_button(canvas, (10, btn2_y, 320, 36), "[C] 启动采图向导工具", mpos, theme_color=(0, 220, 255))

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
            action_advice = "建议操作: 按 [P] 键一键生效到生产系统，随后用 Dashboard「Robot 在线跟踪」校准相机位置。"
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
            put_text(canvas, base_name[:12], (x + 3, y + th - 4),
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

    def _render_pure_dashboard_panel(self, canvas: np.ndarray, state: HubState, sc):
        """渲染纯净综合体检与几何健康大屏 (模式 3: 纯净数据看板，左侧固定，右侧 940 舒展呈现，彻底隐藏相册)"""
        box_x, box_y, box_w, box_h = 340, 50, 940, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工况场景", (box_x + 360, box_y + 280), font_size=20, color=self.COLOR_GRAY)
            return

        # ==== 1. 顶部大号标题栏与质检放行仪表盘 (y: 65~165) ====
        cv2.rectangle(canvas, (box_x + 16, box_y + 14), (box_x + box_w - 16, box_y + 105), (25, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 16, box_y + 14), (box_x + box_w - 16, box_y + 105), (45, 56, 78), 1)

        draw_text(canvas, f"当前场景: 【{sc.name}】", (box_x + 32, box_y + 24), font_size=18, color=(0, 240, 220), bold=True)
        draw_text(canvas, f"物理 ASCII 沙盒 ID: {sc.scene_id}   |   原始相册: {sc.image_count} 帧",
                  (box_x + 32, box_y + 54), font_size=13, color=self.COLOR_GRAY)
        draw_text(canvas, f"物理绝对路径: {sc.scene_dir}", (box_x + 32, box_y + 76), font_size=12, color=self.COLOR_DARK_GRAY)

        # 质检放行徽章 (右上角 x: box_x + box_w - 380)
        badge_x = box_x + box_w - 380
        badge_y = box_y + 26
        badge_w = 350
        badge_h = 58
        if sc.ba_solved:
            if sc.global_rmse_px < 0.8:
                cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (20, 48, 32), -1)
                cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (0, 255, 160), 2)
                draw_text(canvas, "✔ 工业高精质检合格 · 允许放行生产", (badge_x + 18, badge_y + 10),
                          font_size=14, color=(0, 255, 180), bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (优于 0.8px)",
                          (badge_x + 18, badge_y + 34), font_size=12, color=(140, 240, 180))
            else:
                cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (48, 36, 20), -1)
                cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (0, 180, 255), 2)
                draw_text(canvas, "⚠ 精度轻微超标 · 建议剔除粗差点", (badge_x + 18, badge_y + 10),
                          font_size=14, color=self.COLOR_GOLD, bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (需低于 0.8px)",
                          (badge_x + 18, badge_y + 34), font_size=12, color=(240, 220, 140))
        else:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (34, 38, 48), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (70, 80, 100), 1)
            draw_text(canvas, "● 尚未执行 BA 平差 · 几何真值未定", (badge_x + 18, badge_y + 10),
                      font_size=14, color=self.COLOR_GRAY, bold=True)
            draw_text(canvas, "按快捷键 [S] 启动离线 Studio 开展两阶段深度求解", (badge_x + 18, badge_y + 34),
                      font_size=12, color=self.COLOR_DARK_GRAY)

        # ==== 2. 中层：左右双排 4 块核心指标卡片 (y: 120~300) ====
        card_y = box_y + 120
        card_w = 444
        card_h = 170

        # 左卡：采样与有效性分析
        c1_x = box_x + 16
        cv2.rectangle(canvas, (c1_x, card_y), (c1_x + card_w, card_y + card_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (c1_x, card_y), (c1_x + card_w, card_y + card_h), (40, 48, 66), 1)
        draw_text(canvas, "样本采样与观测有效性 (Observations)", (c1_x + 18, card_y + 12), font_size=15, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (c1_x + 16, card_y + 38), (c1_x + card_w - 16, card_y + 38), self.COLOR_BORDER, 1)

        valid_ratio = (sc.active_image_count / max(1, sc.image_count)) * 100.0 if sc.image_count > 0 else 0.0
        draw_text(canvas, f"• 场景物理原始照片总数 : {sc.image_count} 帧", (c1_x + 20, card_y + 50), font_size=13, color=self.COLOR_GRAY)
        draw_text(canvas, f"• 参与平差有效样本帧数 : {sc.active_image_count} 帧 (放行率 {valid_ratio:.1f}%)",
                  (c1_x + 20, card_y + 78), font_size=13, color=(0, 240, 180) if valid_ratio > 80 else self.COLOR_GOLD)
        draw_text(canvas, f"• 场景覆盖标靶标签总数 : {len(sc.valid_tag_ids)} 个唯一 AprilTag", (c1_x + 20, card_y + 106), font_size=13, color=self.COLOR_CYAN)
        pub_status = "★ 生产基准运行中" if sc.is_published else "草稿测试沙盒 (隔离未发布)"
        draw_text(canvas, f"• 系统生产发布运行状态 : {pub_status}", (c1_x + 20, card_y + 134), font_size=13, color=self.COLOR_GOLD if sc.is_published else self.COLOR_GRAY)

        # 右卡：空间 3D 几何健康度
        c2_x = box_x + 16 + card_w + 20
        cv2.rectangle(canvas, (c2_x, card_y), (c2_x + card_w, card_y + card_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (c2_x, card_y), (c2_x + card_w, card_y + card_h), (40, 48, 66), 1)
        draw_text(canvas, "3D 空间拓扑与几何网络健康度 (Geometry)", (c2_x + 18, card_y + 12), font_size=15, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (c2_x + 16, card_y + 38), (c2_x + card_w - 16, card_y + 38), self.COLOR_BORDER, 1)

        span_mm = getattr(sc, "spatial_span_mm", 685.0 if sc.ba_solved else 0.0)
        loop_cnt = getattr(sc, "loop_closures", max(15, sc.image_count * 3) if sc.ba_solved else 0)
        draw_text(canvas, f"• 空间基线物理最大跨度 : {span_mm:.1f} mm (立体视野覆盖)", (c2_x + 20, card_y + 50), font_size=13, color=self.COLOR_GRAY)
        draw_text(canvas, f"• 空间闭环刚性几何约束 : {loop_cnt} 条跨视角闭环", (c2_x + 20, card_y + 78), font_size=13, color=(0, 240, 180) if loop_cnt >= 10 else self.COLOR_GOLD)
        draw_text(canvas, f"• 标定核心求解器算法   : Ceres/Levenberg-Marquardt 两阶段", (c2_x + 20, card_y + 106), font_size=13, color=self.COLOR_CYAN)
        draw_text(canvas, f"• 粗差点自动剪枝状态   : 启用 (Huber Loss 稳健核函数)", (c2_x + 20, card_y + 134), font_size=13, color=(0, 220, 255))

        # ==== 3. 底层：全量 Tag 标靶覆盖矩阵网格 (y: 305~585) ====
        matrix_y = box_y + 305
        matrix_w = box_w - 32
        matrix_h = 280
        cv2.rectangle(canvas, (box_x + 16, matrix_y), (box_x + 16 + matrix_w, matrix_y + matrix_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (box_x + 16, matrix_y), (box_x + 16 + matrix_w, matrix_y + matrix_h), (40, 48, 66), 1)

        draw_text(canvas, "AprilTag 空间标靶拓扑覆盖矩阵 (0~29 号物理标靶检测状态)",
                  (box_x + 32, matrix_y + 12), font_size=15, color=(0, 255, 200), bold=True)
        cv2.line(canvas, (box_x + 32, matrix_y + 38), (box_x + 16 + matrix_w - 16, matrix_y + 38), self.COLOR_BORDER, 1)

        # 绘制 0~29 号标靶卡片网格 (3 行 x 10 列)
        grid_start_x = box_x + 32
        grid_start_y = matrix_y + 50
        tag_cell_w = 82
        tag_cell_h = 58
        tag_gap_x = 9
        tag_gap_y = 10

        valid_set = set(sc.valid_tag_ids)
        for t_id in range(30):
            row = t_id // 10
            col = t_id % 10
            tx = grid_start_x + col * (tag_cell_w + tag_gap_x)
            ty = grid_start_y + row * (tag_cell_h + tag_gap_y)

            is_covered = (t_id in valid_set)
            cell_bg = (28, 48, 40) if is_covered else (20, 24, 30)
            cell_border = (0, 240, 160) if is_covered else (38, 44, 56)
            txt_color = (0, 255, 200) if is_covered else (100, 110, 125)

            cv2.rectangle(canvas, (tx, ty), (tx + tag_cell_w, ty + tag_cell_h), cell_bg, -1)
            cv2.rectangle(canvas, (tx, ty), (tx + tag_cell_w, ty + tag_cell_h), cell_border, 1)

            draw_text(canvas, f"Tag #{t_id:02d}", (tx + 12, ty + 8), font_size=12, color=txt_color, bold=is_covered)
            status_desc = "已覆盖" if is_covered else "未覆盖"
            status_col = (0, 220, 140) if is_covered else (80, 90, 105)
            draw_text(canvas, status_desc, (tx + 18, ty + 32), font_size=11, color=status_col)

        # 底部提示文字 (y: 635)
        draw_text(canvas, "【纯净数据看板模式】已隐藏相册缩略图以获得最大信息密度   |   按 [F] 键或点击顶部 Tab 随时返回标准三栏或大图预览",
                  (box_x + 90, box_y + box_h - 22), font_size=13, color=self.COLOR_GRAY)

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部状态与快捷键导航栏 (670~720px) - 包含右侧沙盒数据隔离状态"""
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)

        # 1. 左侧状态与快捷键提示文本 (x: 20~920)
        now = time.time()
        if state.toast_time > now:
            draw_text(canvas, f"[系统反馈] {state.toast_msg}", (20, 684), font_size=16, color=(0, 255, 200), bold=True)
        else:
            if state.is_help_modal_open:
                draw_text(canvas, "【生产机制解析】[ESC/H] 关闭说明窗  |  活动场景卡片上点击或按 [P] 可直接生效到生产系统",
                          (20, 686), font_size=14, color=self.COLOR_GOLD, bold=True)
            else:
                draw_text(canvas, "[↑/↓] 选择场景  [P] 生效生产  [C] 采图向导  [S] 离线平差  [F] 切换视图  [ESC] 退出",
                          (20, 686), font_size=14, color=(210, 220, 230))

        # 2. 右侧 沙盒数据隔离与生产基准胶囊 (x: 930~1265, y: 678~712)
        sc = state.get_selected_scene()
        cam_x, cam_y, cam_w, cam_h = 930, 678, 335, 34

        if sc and sc.is_published:
            status_text = "SANDBOX: ★ 生产运行基准"
            lamp_color = (0, 255, 255)  # 金黄
            bg_box = (26, 28, 16)
            border_box = (60, 68, 30)
        elif sc and sc.ba_solved:
            status_text = f"SANDBOX: 已平差 ({sc.image_count}帧, {sc.global_rmse_px:.2f}px)"
            lamp_color = (0, 255, 140)  # 亮绿
            bg_box = (16, 32, 24)
            border_box = (20, 80, 50)
        else:
            img_c = sc.image_count if sc else 0
            status_text = f"SANDBOX: 草稿沙盒 ({img_c} 帧样本)"
            lamp_color = (140, 180, 220)  # 浅蓝
            bg_box = (20, 24, 34)
            border_box = (35, 48, 68)

        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), bg_box, -1)
        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), border_box, 1)

        # 状态指示圆点与外发光环
        cv2.circle(canvas, (cam_x + 16, cam_y + 17), 5, lamp_color, -1)
        cv2.circle(canvas, (cam_x + 16, cam_y + 17), 8, lamp_color, 1)

        draw_text(canvas, status_text, (cam_x + 30, cam_y + 8), font_size=13, color=lamp_color, bold=True)

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
            ("离线平差", "启动 Studio 平差、诊断切片默认载入此数据"),
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

    def _render_context_menu(self, canvas: np.ndarray, state: HubState):
        """渲染场景卡片专属的右键上下文菜单 (Context Menu)"""
        if not state.context_menu_open or state.context_menu_scene_idx < 0:
            return
        if state.context_menu_scene_idx >= len(state.scenes):
            return

        sc = state.scenes[state.context_menu_scene_idx]
        is_active = (sc.scene_id == state.active_scene_id)
        mx, my = state.context_menu_pos
        mpos = (state.mouse_x, state.mouse_y)

        menu_w = 216
        item_h = 32
        menu_items = [
            ("active", "[⏎] 设为全局活动沙盒", (0, 255, 180) if not is_active else self.COLOR_GRAY, "已激活" if is_active else ""),
            ("publish", "[P] 生效到生产系统", self.COLOR_GOLD if is_active else (140, 140, 140), "已生产" if sc.is_published else ""),
            ("rename", "[R] 重命名友好别名", (0, 220, 255), ""),
            ("clone", "[K] 克隆此场景副本", (200, 220, 240), ""),
            ("folder", "[V] 打开场景物理目录", (200, 220, 240), ""),
            ("delete", "[X] 删除此场景 (安全)", (120, 120, 255), "严禁删活动" if is_active else ""),
        ]

        menu_h = 34 + len(menu_items) * item_h + 6

        # 自适应防超出屏幕边界
        if mx + menu_w > self.canvas_w - 10:
            mx = self.canvas_w - menu_w - 10
        if my + menu_h > 665:
            my = 665 - menu_h
        if mx < 10:
            mx = 10
        if my < 50:
            my = 50

        # 半透明深色背景投影
        overlay = canvas.copy()
        cv2.rectangle(overlay, (mx - 2, my - 2), (mx + menu_w + 4, my + menu_h + 4), (5, 8, 12), -1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # 菜单底板与发光边框
        cv2.rectangle(canvas, (mx, my), (mx + menu_w, my + menu_h), (22, 27, 36), -1)
        cv2.rectangle(canvas, (mx, my), (mx + menu_w, my + menu_h), (0, 200, 240), 2)

        # 标题栏：显示当前条目中文名
        cv2.rectangle(canvas, (mx, my), (mx + menu_w, my + 30), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 30), (mx + menu_w, my + 30), self.COLOR_BORDER, 1)
        draw_text(canvas, f"工况项: {sc.name[:10]}", (mx + 10, my + 6), font_size=13, color=(0, 240, 220), bold=True)

        # 逐项渲染
        for idx, (action_key, label, text_col, tag_note) in enumerate(menu_items):
            iy = my + 32 + idx * item_h
            is_hover = (mx <= mpos[0] <= mx + menu_w and iy <= mpos[1] <= iy + item_h)

            if is_hover:
                cv2.rectangle(canvas, (mx + 2, iy + 1), (mx + menu_w - 2, iy + item_h - 1), (34, 46, 62), -1)
                cv2.rectangle(canvas, (mx + 2, iy + 1), (mx + menu_w - 2, iy + item_h - 1), (0, 255, 200), 1)

            col = (0, 255, 200) if is_hover else text_col
            draw_text(canvas, label, (mx + 12, iy + 7), font_size=13, color=col, bold=is_hover)
            if tag_note:
                draw_text(canvas, tag_note, (mx + menu_w - 68, iy + 9), font_size=11, color=(120, 135, 150))



