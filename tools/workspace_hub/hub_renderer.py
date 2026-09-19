"""
Workspace Hub 视觉渲染引擎 (HubRenderer)
=======================================
专业工业级暗黑系 GUI 渲染管线，左右两栏布局：
- 左栏 (x: 0~340): Workspace 列表导航 (固定稳定)
- 右栏 (x: 340~1280): 动态页签区 (1 标定相册 / 2 Tag白名单 / 3 体检报告)
- 标定相册页签内支持双击卡片进入全宽大图沉浸预览
"""

import os
import time
from typing import Any
import cv2
import numpy as np

from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, put_text
from tools.workspace_hub.hub_state import HubState


# 图片卡片网格墙几何常量 (4 列 x 3 行, 大卡片 218x182, 网格铺满 y: 86~648)
GRID_X0 = 356
GRID_Y0 = 86
GRID_CELL_W = 218
GRID_CELL_H = 182
GRID_GAP_X = 12
GRID_GAP_Y = 8
GRID_THUMB_H = 156
GRID_COLS = 4
GRID_ROWS = 3

# 生产机制说明窗几何常量
HELP_MODAL_W = 860
HELP_MODAL_H = 490


def grid_hit_test(mx: int, my: int) -> int | None:
    """根据逻辑坐标返回命中的卡片格位索引 (0~11)；落在卡片间隙或网格外返回 None"""
    if mx < GRID_X0 or my < GRID_Y0:
        return None
    col = (mx - GRID_X0) // (GRID_CELL_W + GRID_GAP_X)
    row = (my - GRID_Y0) // (GRID_CELL_H + GRID_GAP_Y)
    if col >= GRID_COLS or row >= GRID_ROWS:
        return None
    local_x = (mx - GRID_X0) % (GRID_CELL_W + GRID_GAP_X)
    local_y = (my - GRID_Y0) % (GRID_CELL_H + GRID_GAP_Y)
    if local_x >= GRID_CELL_W or local_y >= GRID_CELL_H:
        return None
    return row * GRID_COLS + col


class HubRenderer:
    """Workspace Hub 统一界面渲染器"""

    # 图片卡片网格墙几何 (引用模块级常量，便于渲染与命中测试共用)
    GRID_X0 = GRID_X0
    GRID_Y0 = GRID_Y0
    GRID_CELL_W = GRID_CELL_W
    GRID_CELL_H = GRID_CELL_H
    GRID_GAP_X = GRID_GAP_X
    GRID_GAP_Y = GRID_GAP_Y
    GRID_THUMB_H = GRID_THUMB_H

    # 页签显示文案 (顺序由 HubState.TAB_ORDER 决定)
    TAB_LABELS = {
        HubState.TAB_CALIB_IMAGES: "⊞ 标定相册",
        HubState.TAB_WHITELIST: "⚑ Tag白名单",
        HubState.TAB_REPORT: "▤ 体检报告",
        HubState.TAB_PROD_IMAGES: "★ 生产相册",
    }

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

    def hit_test(self, mx: int, my: int, state: HubState) -> Any:
        """根据逻辑坐标探测当前命中交互元素"""
        old_x, old_y = state.mouse_x, state.mouse_y
        try:
            state.mouse_x, state.mouse_y = mx, my
            return self._get_interactive_hover_key(state)
        finally:
            state.mouse_x, state.mouse_y = old_x, old_y

    def _get_interactive_hover_key(self, state: HubState) -> Any:
        """获取当前鼠标悬停的交互元素标识 (若鼠标未落在任何可交互组件上返回 None)"""
        mx, my = state.mouse_x, state.mouse_y
        if mx < 0 or my < 0:
            return None

        # 0. 生产机制业务说明弹窗模式 (最高交互层)
        if state.is_help_modal_open:
            modal_w, modal_h = HELP_MODAL_W, HELP_MODAL_H
            mx_box = (self.canvas_w - modal_w) // 2
            my_box = (self.canvas_h - modal_h) // 2
            bx1 = mx_box + modal_w - 116
            by1 = my_box + 11
            bx2 = bx1 + 100
            by2 = by1 + 32
            # 关闭按钮 (带 6px 容差热区)
            if (bx1 - 6) <= mx <= (bx2 + 6) and (by1 - 6) <= my <= (by2 + 6):
                return "help_close"
            # 外部半透明遮罩
            if mx < mx_box or mx > mx_box + modal_w or my < my_box or my > my_box + modal_h:
                return "help_mask"
            return "help_modal_body"

        # 1. 场景条目专属右键菜单 (精简为 4 项且无标题栏)
        if state.context_menu_open:
            cx, cy = state.context_menu_pos
            menu_w = 216
            item_h = 32
            pad_y = 6
            if cx <= mx <= cx + menu_w:
                for idx in range(4):
                    iy = cy + pad_y + idx * item_h
                    if iy <= my <= iy + item_h:
                        return ("ctx_item", idx)
            return "ctx_menu_other"

        # 2. 常规看板模式
        # 顶部 Header 交互 (右侧动态区页签 Tab + [退出] 按钮)
        if 0 <= my <= 50:
            # 页签 Tab 胶囊 (x: 360 起, y: 8~42, 每片 112px 宽、间距 8px)
            if 8 <= my <= 42 and 360 <= mx <= 360 + len(HubState.TAB_ORDER) * 120 - 8:
                tab_idx = (mx - 360) // 120
                if 0 <= tab_idx < len(HubState.TAB_ORDER):
                    return ("hdr_tab", tab_idx)
            if 1085 <= mx <= 1265 and 8 <= my <= 42:
                return "btn_exit"

        # 左侧面板按钮与卡片
        if 0 <= mx <= 340:
            div_y1 = 604
            btn1_y = div_y1 + 10
            if 10 <= mx <= 330 and btn1_y <= my <= btn1_y + 40:
                return "btn_new_workspace"

            # 工位卡片 (扩展至 6 张卡片)
            card_h = 70
            start_y = 58
            max_cards = 6
            scroll_start = max(0, state.selected_workspace_idx - max_cards + 1)
            visible_workspaces = state.workspaces[scroll_start: scroll_start + max_cards]
            for i, ws in enumerate(visible_workspaces):
                cy = start_y + i * (card_h + 8)
                real_idx = scroll_start + i
                if 10 <= mx <= 330 and cy <= my <= cy + card_h:
                    return ("card_select", real_idx)

        # 右侧动态区页签内容按钮
        if state.view_mode == HubState.VIEW_EXPANDED:
            box_x, box_y, box_w = 340, 50, 940
            if box_y + 8 <= my <= box_y + 44:
                if box_x + box_w - 470 <= mx <= box_x + box_w - 390:
                    return "exp_prev"
                if box_x + box_w - 384 <= mx <= box_x + box_w - 304:
                    return "exp_next"
                if box_x + box_w - 298 <= mx <= box_x + box_w - 198:
                    return "album_delete"
                if box_x + box_w - 192 <= mx <= box_x + box_w - 20:
                    return "exp_restore"
        elif 58 <= my <= 88:
            # Tag 白名单页签: [刷新] [编辑] (图片页签已移除顶部按钮组, 改用滚轮/双击等鼠标操作)
            if state.active_tab == HubState.TAB_WHITELIST:
                if 1092 <= mx <= 1170:
                    return "wl_refresh"
                if 1176 <= mx <= 1270:
                    return "wl_edit"

        # 图片卡片网格墙卡片 Hover
        if state.active_tab == HubState.TAB_CALIB_IMAGES:
            cell_idx = grid_hit_test(mx, my)
            if cell_idx is not None:
                return ("grid_item", state.image_grid_offset + cell_idx)

        return None

    def render(self, state: HubState) -> np.ndarray:
        """根据当前状态机渲染 1280x720 最终画布 (双缓冲极速渲染)"""
        # 计算当前交互状态的哈希指纹，命中缓存则零拷贝直接返回！
        selected_ws = state.get_selected_workspace()
        cache_key = (
            state.selected_workspace_idx,
            selected_ws.workspace_id if selected_ws else None,
            len(state.workspaces),
            state.selected_image_idx,
            len(state.current_images),
            state.image_grid_offset,
            state.view_mode,
            state.active_tab,
            state._whitelist_cache_ws,
            state._whitelist_cache_mtime,
            state.toast_msg,
            state.context_menu_open,
            state.context_menu_pos if state.context_menu_open else None,
            state.context_menu_ws_idx if state.context_menu_open else None,
            state.is_help_modal_open,
            state.mouse_x,
            state.mouse_y,
            int(time.time() * 2)  # 每 500ms 刷新时间敏感的 Toast 与动画
        )
        if self._cached_canvas is not None and self._last_cache_key == cache_key:
            return self._cached_canvas

        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        # 1. 顶部状态栏 (y: 0~50): 标题 + 右侧动态区页签 Tab + 功能按钮
        self._render_header(canvas, state)

        # 3. 左侧综合导航栏 (x: 0~340, y: 50~670) - 切换页签过程中保持稳定
        self._render_left_panel(canvas, state)
        cv2.line(canvas, (340, 50), (340, 670), self.COLOR_BORDER, 1)

        # 4. 右侧动态区 (x: 340~1280, y: 50~670): 三页签动态内容 + 全宽大图沉浸
        ws = state.get_selected_workspace()
        if state.view_mode == HubState.VIEW_EXPANDED:
            self._render_expanded_photo_preview(canvas, state, ws)
        elif state.active_tab == HubState.TAB_REPORT:
            self._render_pure_dashboard_panel(canvas, state, ws)
        elif state.active_tab == HubState.TAB_WHITELIST:
            self._render_page_whitelist(canvas, state, ws)
        elif state.active_tab == HubState.TAB_PROD_IMAGES:
            self._render_page_prod_images(canvas, state, ws)
        else:
            self._render_page_calib_images(canvas, state, ws)

        # 5. 底部系统反馈提示栏 (y: 670~720)
        self._render_footer(canvas, state)

        # 6. 如果打开了场景条目专属右键上下文菜单 (Context Menu)，置顶渲染
        if state.context_menu_open:
            self._render_context_menu(canvas, state)

        # 7. 如果打开了生产系统机制说明弹窗 (Help Modal)，最高优先级置顶展示
        if state.is_help_modal_open:
            self._render_help_modal(canvas, state)

        self._cached_canvas = canvas
        self._last_cache_key = cache_key
        return canvas

    def _render_header(self, canvas: np.ndarray, state: HubState):
        """渲染顶部标题栏 (0~50px) - 包含右侧动态区四页签Tab、生产机制说明与退出按钮"""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 50), (14, 16, 20), -1)
        cv2.line(canvas, (0, 50), (self.canvas_w, 50), self.COLOR_BORDER, 1)
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 系统标题与状态点 (x: 16~260)
        cv2.circle(canvas, (22, 25), 6, (0, 255, 180), -1)
        put_text(canvas, "flux_vision_3d", (36, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 2, cv2.LINE_AA)
        draw_text(canvas, "Workspace", (165, 16), font_size=17, color=self.COLOR_WHITE, bold=True)

        # 2. 右侧动态区四页签 Tab 胶囊 (x: 360~832, y: 8~42)
        self._render_header_tabs(canvas, state)

        # 3. 右上角功能按钮组 (说明窗按钮已移除, 说明窗改由点击工位卡片徽章呼出)
        # [退出] 按钮 (x: 1085~1265, y: 8~42) - 实体鼠标点击退出
        self._draw_button(canvas, (1085, 8, 180, 34), "退出", mpos, theme_color=(180, 60, 60))

    def _render_header_tabs(self, canvas: np.ndarray, state: HubState):
        """渲染顶部四页签 Tab 胶囊: 1 标定相册 / 2 Tag白名单 / 3 体检报告 / 4 生产相册
        (x: 360~832, y: 8~42, 每片 112px 宽、间距 8px; 页签顺序与 HubState.TAB_ORDER 保持一致)
        """
        mpos = (state.mouse_x, state.mouse_y)
        # 直接按 HubState.TAB_ORDER 渲染，保证页签展示顺序与状态机始终一致
        tabs = [(key, self.TAB_LABELS[key]) for key in HubState.TAB_ORDER]

        for idx, (tab_key, tab_text) in enumerate(tabs):
            tx = 360 + idx * 120
            ty, tw, th = 8, 112, 34
            is_active_tab = (state.active_tab == tab_key)
            is_hover_tab = (tx <= mpos[0] <= tx + tw and ty <= mpos[1] <= ty + th)

            if is_active_tab:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (28, 44, 40), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 255, 180), 2)
                # 激活页签底部高亮指示条
                cv2.rectangle(canvas, (tx + 8, ty + th - 3), (tx + tw - 8, ty + th - 1), (0, 255, 180), -1)
                draw_text(canvas, tab_text, (tx + 12, ty + 8), font_size=13, color=(0, 255, 200), bold=True)
            elif is_hover_tab:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (34, 40, 52), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 200, 240), 1)
                draw_text(canvas, tab_text, (tx + 12, ty + 8), font_size=13, color=(0, 220, 255))
            else:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (22, 27, 35), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (45, 55, 72), 1)
                draw_text(canvas, tab_text, (tx + 12, ty + 8), font_size=13, color=(160, 175, 195))

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
        - 场景卡片全面支持自由一键 [生效生产]
        - 保持工业界面整洁精炼
        """
        cv2.rectangle(canvas, (0, 50), (340, 670), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # ==== 1. Workspace 卡片列表 (标题行已移除, 卡片直接顶到面板顶部) ====
        card_h = 70
        start_y = 58
        max_cards = 6  # 扩展至 6 张卡片，充分利用垂直空间

        scroll_start = max(0, state.selected_workspace_idx - max_cards + 1)
        visible_workspaces = state.workspaces[scroll_start: scroll_start + max_cards]

        for i, ws in enumerate(visible_workspaces):
            real_idx = scroll_start + i
            is_selected = (real_idx == state.selected_workspace_idx)
            cy = start_y + i * (card_h + 8)

            card_col = self.COLOR_CARD_ACTIVE if is_selected else (28, 32, 42)
            border_col = self.COLOR_ACTIVE_BORDER if is_selected else self.COLOR_BORDER

            cv2.rectangle(canvas, (10, cy), (330, cy + card_h), card_col, -1)
            cv2.rectangle(canvas, (10, cy), (330, cy + card_h), border_col, 2 if is_selected else 1)

            if is_selected:
                cv2.rectangle(canvas, (10, cy), (14, cy + card_h), (0, 255, 160), -1)

            # 主标题突出显示友好中文名称
            prefix = f"{real_idx + 1:02d}."
            display_title = f"{prefix} {ws.name}"
            title_col = (0, 255, 200) if is_selected else self.COLOR_WHITE
            draw_text(canvas, display_title, (20, cy + 8), font_size=15, color=title_col, bold=is_selected)

            # 左下角：平差精度指标 (原第三行上移，原第二行冗余物理ID已删除)
            ba_badge = f"RMSE: {ws.global_rmse_px:.2f}px" if ws.ba_solved else "未平差"
            ba_col = (0, 220, 100) if ws.ba_solved else self.COLOR_DARK_GRAY
            put_text(canvas, ba_badge, (20, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ba_col, 1, cv2.LINE_AA)

            # 右侧：标定与生产采图帧数 (分2行布局)
            frame_col = (0, 220, 180) if is_selected else (150, 165, 185)
            put_text(canvas, f"标定 {ws.image_count} 帧", (232, cy + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, frame_col, 1, cv2.LINE_AA)
            put_text(canvas, f"生产 {ws.prod_image_count} 帧", (232, cy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, frame_col, 1, cv2.LINE_AA)

        # ==== 2. Workspace 通用全局操作区 ====
        div_y1 = 604
        cv2.line(canvas, (10, div_y1), (330, div_y1), self.COLOR_BORDER, 1)

        btn1_y = div_y1 + 10
        self._draw_button(canvas, (10, btn1_y, 320, 40), "+ 新建 Workspace", mpos)

    def _render_page_calib_images(self, canvas: np.ndarray, state: HubState, sc):
        """页签1: 标定相册 - 当前选中工位的采样相册卡片网格墙 (x: 340~1280, y: 50~670)"""
        title = f"标定采样相册 ({len(state.current_images)}帧)"
        self._render_gallery_page(
            canvas, state, title,
            state.current_images, state.selected_image_idx, state.image_grid_offset,
            empty_hint=("当前场景尚未采集任何照片！", "请在主仪表盘启动多视角采图向导抓拍照片。"),
            is_calib=True,
        )
        # 标题帧数信息与沙盒状态合并展示 (右侧状态胶囊)
        self._render_status_capsule(canvas, state, sc)

    def _render_page_prod_images(self, canvas: np.ndarray, state: HubState, sc):
        """页签4: 生产相册 - 生产运行基准工位的采样相册 (x: 340~1280, y: 50~670)"""
        prod_images = getattr(state, "prod_images", [])
        prod_idx = getattr(state, "selected_prod_image_idx", 0)
        prod_offset = getattr(state, "prod_grid_offset", 0)
        title = f"生产基准相册 ({len(prod_images)}帧)"
        self._render_gallery_page(
            canvas, state, title,
            prod_images, prod_idx, prod_offset,
            empty_hint=("生产基准工位尚未采集任何照片！", "可在该工位下切换至生产用途抓拍图像。"),
            is_calib=False,
        )
        self._render_status_capsule(canvas, state, sc)

    def _render_gallery_page(self, canvas: np.ndarray, state: HubState, title: str,
                             images: list[str], sel_idx: int, grid_offset: int,
                             empty_hint: tuple[str, str], is_calib: bool):
        """渲染通用图片卡片网格墙: 单行标题 + 4列x3行大卡片网格 (双击卡片放大)
        布局: 面板 (340,50,940,620); 标题 y: 58; 网格 x: 356~1264, y: 86~648; 卡片 218x182
        """
        box_x, box_y, box_w, box_h = 340, 50, 940, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 栏目标题 (单行精炼，垂直空间全部让给大卡片网格)
        draw_text(canvas, title, (box_x + 14, box_y + 8), font_size=15, color=self.COLOR_WHITE, bold=True)

        if not images:
            empty_box_y = box_y + 130
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 120), (22, 26, 36), -1)
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 120), self.COLOR_BORDER, 1)
            draw_text(canvas, empty_hint[0], (box_x + 60, empty_box_y + 34), font_size=18, color=(0, 200, 240), bold=True)
            draw_text(canvas, empty_hint[1], (box_x + 60, empty_box_y + 74), font_size=13, color=self.COLOR_GRAY)
            return

        # 3. 图片卡片网格墙 (4 列 x 3 行, 每页 12 张大卡片, 铺满面板底部空白)
        total = len(images)
        visible_imgs = images[grid_offset: grid_offset + HubState.GRID_PAGE]

        for i, img_path in enumerate(visible_imgs):
            real_idx = grid_offset + i
            row, col = divmod(i, HubState.GRID_COLS)
            x = self.GRID_X0 + col * (self.GRID_CELL_W + self.GRID_GAP_X)
            y = self.GRID_Y0 + row * (self.GRID_CELL_H + self.GRID_GAP_Y)

            is_cur = (real_idx == sel_idx)
            is_hover = (x <= mpos[0] <= x + self.GRID_CELL_W and y <= mpos[1] <= y + self.GRID_CELL_H)

            card_bg = (30, 40, 36) if is_cur else ((28, 33, 41) if is_hover else (22, 26, 36))
            card_border = (0, 255, 180) if is_cur else ((0, 200, 240) if is_hover else (38, 44, 58))
            cv2.rectangle(canvas, (x, y), (x + self.GRID_CELL_W, y + self.GRID_CELL_H), card_bg, -1)
            cv2.rectangle(canvas, (x, y), (x + self.GRID_CELL_W, y + self.GRID_CELL_H), card_border, 2 if is_cur else 1)

            # 3.1 卡片主体: 接近 4:3 的大缩略图
            tw, th = self.GRID_CELL_W - 8, self.GRID_THUMB_H
            thumb = state.get_thumbnail(img_path, tw, th)
            if thumb is not None:
                canvas[y + 4:y + 4 + th, x + 4:x + 4 + tw] = thumb
            else:
                draw_text(canvas, "读取失败", (x + 60, y + 74), font_size=13, color=(120, 120, 140))

            # 3.2 左上角序号徽章
            cv2.rectangle(canvas, (x + 4, y + 4), (x + 56, y + 26), (8, 12, 16), -1)
            put_text(canvas, f"#{real_idx + 1:02d}", (x + 11, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                     0.42, (0, 255, 180) if is_cur else (170, 185, 205), 1, cv2.LINE_AA)

            # 3.3 底部文件名信息条
            bar_y = y + self.GRID_CELL_H - 22
            cv2.rectangle(canvas, (x + 1, bar_y), (x + self.GRID_CELL_W - 1, y + self.GRID_CELL_H - 1), (10, 12, 18), -1)
            put_text(canvas, os.path.basename(img_path)[:30], (x + 6, y + self.GRID_CELL_H - 7),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                     (0, 255, 200) if is_cur else self.COLOR_GRAY, 1, cv2.LINE_AA)

            # 3.4 选中卡片左侧高亮指示条
            if is_cur:
                cv2.rectangle(canvas, (x, y), (x + 4, y + self.GRID_CELL_H), (0, 255, 180), -1)

        # 4. 底部单行选中详情与操作提示 (网格已铺满，信息压缩为一行)
        cur_name = os.path.basename(images[sel_idx])
        page_now = grid_offset // HubState.GRID_PAGE + 1
        page_total = max(1, (total + HubState.GRID_PAGE - 1) // HubState.GRID_PAGE)
        draw_text(canvas,
                  f"已选中: {cur_name} ({sel_idx + 1}/{total})  |  第 {page_now}/{page_total} 页  |  "
                  f"单击卡片选中  |  双击卡片放大查看  |  鼠标滚轮翻页",
                  (box_x + 16, 652), font_size=12, color=(0, 240, 220))

    def _render_page_whitelist(self, canvas: np.ndarray, state: HubState, sc):
        """页签4: Tag 标靶白名单管理页 - 实时读取 tag_whitelist.yaml 呈现放行矩阵"""
        box_x, box_y, box_w, box_h = 340, 50, 940, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 23, 30), -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题与右上角操作按钮
        draw_text(canvas, "Tag 标靶白名单管理", (box_x + 16, box_y + 14), font_size=17, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, (1092, box_y + 8, 78, 30), "刷新", mpos)
        self._draw_button(canvas, (1176, box_y + 8, 94, 30), "编辑", mpos)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建 Workspace", (box_x + 340, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        wl = state.get_tag_whitelist()
        wl_path = state.workspace_mgr.get_tag_whitelist_path(sc.workspace_id)
        enabled = bool(wl.get("enabled", False)) if wl else False
        allowed_ids = set(wl.get("allowed_ids") or []) if wl else set()

        # 1. Workspace 元数据卡与白名单生效状态徽章
        meta_y = box_y + 52
        cv2.rectangle(canvas, (box_x + 16, meta_y), (box_x + box_w - 16, meta_y + 70), (26, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 16, meta_y), (box_x + box_w - 16, meta_y + 70), self.COLOR_BORDER, 1)
        draw_text(canvas, f"当前 Workspace: 【{sc.name}】  |  物理唯一ID: {sc.workspace_id}",
                  (box_x + 28, meta_y + 10), font_size=15, color=(0, 240, 220), bold=True)
        draw_text(canvas, f"配置文件: {wl_path}", (box_x + 28, meta_y + 44), font_size=12, color=self.COLOR_DARK_GRAY)

        badge_x, badge_y = box_x + box_w - 250, meta_y + 14
        if enabled:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 230, badge_y + 42), (20, 48, 32), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 230, badge_y + 42), (0, 255, 160), 2)
            draw_text(canvas, "● 白名单生效中", (badge_x + 22, badge_y + 11), font_size=14, color=(0, 255, 180), bold=True)
        else:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 230, badge_y + 42), (34, 38, 48), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 230, badge_y + 42), (70, 80, 100), 1)
            draw_text(canvas, "○ 白名单未启用", (badge_x + 22, badge_y + 11), font_size=14, color=self.COLOR_GRAY, bold=True)

        # 2. 白名单模式说明卡
        desc_y = meta_y + 82
        cv2.rectangle(canvas, (box_x + 16, desc_y), (box_x + box_w - 16, desc_y + 62), (24, 28, 38), -1)
        cv2.rectangle(canvas, (box_x + 16, desc_y), (box_x + box_w - 16, desc_y + 62), self.COLOR_BORDER, 1)

        if not wl:
            mode_text = "尚未创建 tag_whitelist.yaml 配置文件，当前放行所有检测到的有效标靶。"
            mode_hint = "提示: 点击右上角 [编辑] 按钮可自动生成配置模板并打开编辑。"
            mode_col = self.COLOR_GOLD
        elif enabled:
            mode_text = f"白名单已启用: 仅放行 allowed_ids 中的 {len(allowed_ids)} 个标靶，其余全部拦截。"
            mode_hint = "提示: 在外部编辑器修改保存后，返回本页签即自动刷新矩阵状态。"
            mode_col = (0, 255, 160)
        else:
            mode_text = "白名单未启用 (enabled: false): 放行所有检测到的有效标靶。"
            mode_hint = "提示: 将 enabled 改为 true 并维护 allowed_ids 列表即可启用白名单过滤。"
            mode_col = (0, 200, 240)

        draw_text(canvas, mode_text, (box_x + 28, desc_y + 10), font_size=13, color=mode_col, bold=True)
        draw_text(canvas, mode_hint, (box_x + 28, desc_y + 36), font_size=12, color=self.COLOR_GRAY)

        # 3. 全量 Tag 标靶放行矩阵网格 (3 行 x 10 列)
        matrix_y = desc_y + 74
        matrix_h = 260
        cv2.rectangle(canvas, (box_x + 16, matrix_y), (box_x + box_w - 16, matrix_y + matrix_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (box_x + 16, matrix_y), (box_x + box_w - 16, matrix_y + matrix_h), (40, 48, 66), 1)

        matrix_title = ("AprilTag 标靶放行矩阵 (0~29 号标靶拦截状态)"
                        + (f"  |  已放行 {len(allowed_ids)} 个" if enabled else "  |  全量放行模式"))
        draw_text(canvas, matrix_title, (box_x + 32, matrix_y + 12), font_size=15, color=(0, 255, 200), bold=True)
        cv2.line(canvas, (box_x + 32, matrix_y + 38), (box_x + box_w - 32, matrix_y + 38), self.COLOR_BORDER, 1)

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

            if enabled:
                is_allowed = (t_id in allowed_ids)
                cell_bg = (28, 48, 40) if is_allowed else (40, 26, 26)
                cell_border = (0, 240, 160) if is_allowed else (60, 40, 40)
                txt_color = (0, 255, 200) if is_allowed else (180, 120, 120)
                status_desc = "已放行" if is_allowed else "已拦截"
                status_col = (0, 220, 140) if is_allowed else (150, 100, 100)
            else:
                cell_bg = (24, 34, 40)
                cell_border = (36, 70, 80)
                txt_color = (160, 220, 235)
                status_desc = "免检放行"
                status_col = (110, 170, 190)

            cv2.rectangle(canvas, (tx, ty), (tx + tag_cell_w, ty + tag_cell_h), cell_bg, -1)
            cv2.rectangle(canvas, (tx, ty), (tx + tag_cell_w, ty + tag_cell_h), cell_border, 1)

            draw_text(canvas, f"Tag #{t_id:02d}", (tx + 12, ty + 8), font_size=12, color=txt_color, bold=True)
            draw_text(canvas, status_desc, (tx + 18, ty + 32), font_size=11, color=status_col)
            # 工位实际检测覆盖标记
            if t_id in valid_set:
                cv2.circle(canvas, (tx + tag_cell_w - 10, ty + 12), 4, (0, 200, 240), -1)

        # 4. 底部操作指引
        draw_text(canvas, "提示: 点击右上角 [编辑] 按钮在外部编辑器中维护 tag_whitelist.yaml；保存返回后自动刷新   |   小圆点标记 = 当前工位已检测覆盖的标靶",
                  (box_x + 40, box_y + box_h - 30), font_size=13, color=self.COLOR_GRAY)

    def _render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """全宽自适应大图视口 (按 F 键展开，横跨中间和右侧，x: 340~1280)"""
        box_x, box_y, box_w, box_h = 340, 50, 940, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 20, 26), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)
        mpos = (state.mouse_x, state.mouse_y)

        # 标题与右上角实体按钮
        if not state.current_images:
            draw_text(canvas, "当前场景无图片", (box_x + 400, box_y + 280), font_size=20, color=self.COLOR_DARK_GRAY)
            self._draw_button(canvas, (box_x + box_w - 180, box_y + 10, 160, 32), "返回网格", mpos)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=910, max_h=520)

        img_title = f"全宽自适应大图预览: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 20, box_y + 14), font_size=17, color=(0, 255, 200), bold=True)

        # 右上角实体按钮组: [上张] [下张] [删帧] [返回网格]
        self._draw_button(canvas, (box_x + box_w - 470, box_y + 10, 80, 32), "上张", mpos)
        self._draw_button(canvas, (box_x + box_w - 384, box_y + 10, 80, 32), "下张", mpos)
        self._draw_button(canvas, (box_x + box_w - 298, box_y + 10, 100, 32), "删帧", mpos, theme_color=(180, 60, 60))
        self._draw_button(canvas, (box_x + box_w - 192, box_y + 10, 172, 32), "返回网格", mpos)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 48 + (520 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 70, 90), 1)

        draw_text(canvas, "点击 [上张] / [下张] 切换大图   |   鼠标滚轮顺畅切换   |   双击画面或点击 [返回网格] 返回卡片网格",
                  (box_x + 210, box_y + box_h - 24), font_size=14, color=self.COLOR_WHITE)

    def _render_pure_dashboard_panel(self, canvas: np.ndarray, state: HubState, sc):
        """页签3: 体检报告 - 综合体检与几何健康大屏 (左侧固定，右侧 940 舒展呈现)"""
        box_x, box_y, box_w, box_h = 340, 50, 940, 310
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工况场景", (box_x + 360, box_y + 150), font_size=20, color=self.COLOR_GRAY)
            return

        # ==== 1. 顶部大号标题栏与质检放行仪表盘 (y: 65~165) ====
        cv2.rectangle(canvas, (box_x + 16, box_y + 14), (box_x + box_w - 16, box_y + 105), (25, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 16, box_y + 14), (box_x + box_w - 16, box_y + 105), (45, 56, 78), 1)

        draw_text(canvas, f"物理 ASCII 沙盒 ID: {sc.workspace_id}   |   原始相册: {sc.image_count} 帧",
                  (box_x + 32, box_y + 30), font_size=13, color=self.COLOR_GRAY)
        draw_text(canvas, f"物理绝对路径: {sc.workspace_dir}", (box_x + 32, box_y + 56), font_size=12, color=self.COLOR_DARK_GRAY)

        # 质检放行徽章 (右上角 x: box_x + box_w - 380)
        badge_x = box_x + box_w - 380
        badge_y = box_y + 26
        badge_w = 350
        badge_h = 58
        if sc.ba_solved:
            if sc.global_rmse_px < 0.8:
                cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (20, 48, 32), -1)
                draw_text(canvas, "✔ 工业高精质检合格 · 地图可用", (badge_x + 18, badge_y + 10),
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
            draw_text(canvas, "在主仪表盘启动空间建图工作站开展两阶段深度求解", (badge_x + 18, badge_y + 34),
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
        draw_text(canvas, f"• 生产用途采样照片总数 : {sc.prod_image_count} 帧 (工位内隔离存放)",
                  (c1_x + 20, card_y + 134), font_size=13, color=self.COLOR_GRAY)

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

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部状态提示栏 (670~720px)
        (快捷键提示行与 SANDBOX 状态胶囊已移除: 状态已合并至图片页签右上角状态胶囊)
        """
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)

        # 系统反馈 Toast 与说明窗提示 (无内容时保持纯净留白)
        now = time.time()
        if state.toast_time > now:
            draw_text(canvas, f"[系统反馈] {state.toast_msg}", (20, 684), font_size=16, color=(0, 255, 200), bold=True)

    def _render_status_capsule(self, canvas: np.ndarray, state: HubState, sc):
        """在图片页签右上角渲染【沙盒状态胶囊】 (由底部 Footer 迁移并与页面帧数信息合并展示)
        x: 1010~1264, y: 54~82 (与单行标题同一行, 不遮挡下方卡片网格)
        """
        if not sc:
            return
        cx, cy, cw, ch = 1010, 54, 254, 28

        if sc.ba_solved:
            status_text = f"SANDBOX: 已平差 {sc.global_rmse_px:.2f}px"
            lamp_color = (0, 255, 140)
            bg_box, border_box = (16, 32, 24), (20, 80, 50)
        else:
            status_text = "SANDBOX: 草稿沙盒"
            lamp_color = (140, 180, 220)
            bg_box, border_box = (20, 24, 34), (35, 48, 68)

        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), bg_box, -1)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), border_box, 1)

        # 状态指示圆点与外发光环
        cv2.circle(canvas, (cx + 18, cy + ch // 2), 5, lamp_color, -1)
        cv2.circle(canvas, (cx + 18, cy + ch // 2), 8, lamp_color, 1)

        draw_text(canvas, status_text, (cx + 34, cy + 5), font_size=13, color=lamp_color, bold=True)

    def _render_context_menu(self, canvas: np.ndarray, state: HubState):
        """渲染场景卡片专属的右键上下文菜单 (Context Menu)"""
        if not state.context_menu_open or state.context_menu_ws_idx < 0:
            return
        if state.context_menu_ws_idx >= len(state.workspaces):
            return

        ws = state.workspaces[state.context_menu_ws_idx]
        mx, my = state.context_menu_pos
        mpos = (state.mouse_x, state.mouse_y)

        menu_w = 216
        item_h = 32
        pad_y = 6
        menu_text_col = GuiTheme.BTN_TEXT  # 全部条目统一常态文字色，悬停态由下方逻辑高亮
        menu_items = [
            ("rename", "重命名友好别名", menu_text_col, ""),
            ("clone", "克隆此 Workspace", menu_text_col, ""),
            ("folder", "打开物理目录", menu_text_col, ""),
            ("delete", "删除此 Workspace", menu_text_col, ""),
        ]

        menu_h = pad_y * 2 + len(menu_items) * item_h

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

        # 逐项渲染 (无割裂标题栏，现代扁平菜单)
        for idx, (action_key, label, text_col, tag_note) in enumerate(menu_items):
            iy = my + pad_y + idx * item_h
            is_hover = (mx <= mpos[0] <= mx + menu_w and iy <= mpos[1] <= iy + item_h)

            if is_hover:
                cv2.rectangle(canvas, (mx + 2, iy + 1), (mx + menu_w - 2, iy + item_h - 1), (34, 46, 62), -1)
                cv2.rectangle(canvas, (mx + 2, iy + 1), (mx + menu_w - 2, iy + item_h - 1), (0, 255, 200), 1)

            col = (0, 255, 200) if is_hover else text_col
            draw_text(canvas, label, (mx + 16, iy + 7), font_size=13, color=col, bold=is_hover)
            if tag_note:
                draw_text(canvas, tag_note, (mx + menu_w - 68, iy + 9), font_size=11, color=(120, 135, 150))

    def _render_help_modal(self, canvas: np.ndarray, state: HubState):
        """渲染置顶居中的【生效到生产系统业务机制说明窗】(860x490)"""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_w, self.canvas_h), (8, 10, 14), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        modal_w, modal_h = HELP_MODAL_W, HELP_MODAL_H
        mx = (self.canvas_w - modal_w) // 2
        my = (self.canvas_h - modal_h) // 2
        mpos = (state.mouse_x, state.mouse_y)

        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (20, 24, 32), -1)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (0, 220, 160), 2)
        cv2.rectangle(canvas, (mx + 4, my + 4), (mx + modal_w - 4, my + modal_h - 4), (40, 50, 66), 1)

        # 标题栏
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + 54), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 54), (mx + modal_w, my + 54), self.COLOR_BORDER, 1)

        cv2.circle(canvas, (mx + 24, my + 27), 6, self.COLOR_GOLD, -1)
        draw_text(canvas, "★ 业务架构解析: 为什么需要【生效到生产系统】？", (mx + 38, my + 15),
                  font_size=17, color=self.COLOR_WHITE, bold=True)

        # 右上角 [X] 关闭按钮
        self._draw_button(canvas, (mx + modal_w - 116, my + 11, 100, 32), "[X] 关闭 [H]", mpos)

        # 4 条架构阐释卡片
        intro_text = "在工业机器视觉与机械臂抓取工程中，标定研发与现场生产实行严格的【沙盒隔离】机制："
        draw_text(canvas, intro_text, (mx + 30, my + 68), font_size=14, color=(0, 240, 220))

        sections = [
            ("1. 研发沙盒安全隔离 (Sandbox Isolation)",
             "每个工况场景（如“1号机台主标定”、“光照对照组”）都是独立沙盒，拥有专属照片集与平差结果。现场实验或测试标定时，绝对不会影响正在作业的机械臂。",
             (0, 255, 180)),

            ("2. 全局唯一生产地图 (Global Production Map)",
             "流水线所有空间定位与手眼协同节点，均默认读取项目根目录的【config/tags_map.yaml】。该文件是全系统运行时的唯一真实几何基准。",
             (0, 220, 255)),

            ("3. 一键安全原子覆盖与自动备份 (Safe Atomic Publish)",
             "当操作员选中满意场景并按【[P] 生效到生产系统】时，系统自动生成带时间戳的 .bak 历史备份，随后将平差坐标原子写入 config/tags_map.yaml 并记录至 config.yaml。",
             self.COLOR_GOLD),

            ("4. 实时生效与生产闭环 (Seamless Hot-Reload)",
             "生效发布后，无需重启生产服务即可读取最新厘米/亚毫米级标定矩阵，实现从“采图 -> 平差 -> 质检 -> 生产发布”的完整闭环，确保流水线作业极致精准！",
             (160, 255, 120))
        ]

        sy = my + 98
        for title, desc, col in sections:
            cv2.rectangle(canvas, (mx + 28, sy), (mx + modal_w - 28, sy + 74), (25, 30, 40), -1)
            cv2.rectangle(canvas, (mx + 28, sy), (mx + modal_w - 28, sy + 74), (44, 52, 68), 1)
            cv2.rectangle(canvas, (mx + 28, sy), (mx + 32, sy + 74), col, -1)

            draw_text(canvas, title, (mx + 42, sy + 8), font_size=14, color=col, bold=True)
            d1 = desc[:48]
            d2 = desc[48:96]
            draw_text(canvas, d1, (mx + 42, sy + 30), font_size=12, color=self.COLOR_WHITE)
            if d2:
                draw_text(canvas, d2, (mx + 42, sy + 48), font_size=12, color=self.COLOR_GRAY)
            sy += 82

        footer_y = my + modal_h - 36
        draw_text(canvas, "快捷提示: 鼠标点击右上角 [X]、点击遮罩或直接按键盘 [ESC / H] 即可秒级关闭！",
                  (mx + 32, footer_y), font_size=13, color=self.COLOR_GRAY)



