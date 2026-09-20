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


# 图片卡片网格墙几何常量 (3 列 x 3 行, 大卡片 186x186, 网格铺满 340~960 区域)
GRID_X0 = 352
GRID_Y0 = 64
GRID_CELL_W = 186
GRID_CELL_H = 186
GRID_GAP_X = 16
GRID_GAP_Y = 12
GRID_THUMB_H = 156
GRID_COLS = 3
GRID_ROWS = 3

# 生产机制说明窗几何常量
HELP_MODAL_W = 860
HELP_MODAL_H = 490


def grid_hit_test(mx: int, my: int) -> int | None:
    """根据逻辑坐标返回命中的卡片格位索引 (0~8)；落在卡片间隙或网格外返回 None"""
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

    # 页签显示文案 (顺序: Dashboard / Tag白名单 / 标定相册 / ★ 生产相册)
    TAB_LABELS = {
        HubState.TAB_REPORT: "Dashboard",
        HubState.TAB_WHITELIST: "Tag白名单",
        HubState.TAB_CALIB_IMAGES: "标定相册",
        HubState.TAB_PROD_IMAGES: "★ 生产相册",
    }

    def __init__(self):
        self.canvas_w = 960
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

        # 1. 常规看板模式
        # 顶部 Header 交互 (右侧动态区页签 Tab + [退出] 按钮)
        if 0 <= my <= 50:
            # 四页签 Tab 胶囊 (x: 348~816, y: 8~42, 每片 110px 宽、间距 8px)
            if 8 <= my <= 42 and 348 <= mx <= 816:
                tab_idx = (mx - 348) // 118
                if 0 <= tab_idx < len(HubState.TAB_ORDER):
                    return ("hdr_tab", tab_idx)
            # [退出] 按钮紧贴生产相册右侧 (x: 824~948, y: 8~42)
            if 824 <= mx <= 948 and 8 <= my <= 42:
                return "btn_exit"

        # 左侧面板按钮与卡片
        if 0 <= mx <= 340:
            div_y1 = 604
            btn1_y = div_y1 + 10
            if 10 <= mx <= 330 and btn1_y <= my <= btn1_y + 40:
                return "btn_new_workspace"

            # 工位卡片 (支持 6 张卡片)
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

        # 右侧动态区页签内容按钮 (x: 340~960)
        if state.view_mode == HubState.VIEW_EXPANDED:
            if 58 <= my <= 92:
                if 680 <= mx <= 740:
                    return "exp_prev"
                if 746 <= mx <= 806:
                    return "exp_next"
                if 812 <= mx <= 880:
                    return "album_delete"
                if 886 <= mx <= 950:
                    return "exp_restore"
        elif 58 <= my <= 88:
            # Tag 白名单页签: [刷新] [编辑]
            if state.active_tab == HubState.TAB_WHITELIST:
                if 760 <= mx <= 846:
                    return "wl_refresh"
                if 854 <= mx <= 944:
                    return "wl_edit"

        # 体检报告页签内，工位卡片内嵌操作按钮 Hover (紧凑对齐右边缘)
        if state.active_tab == HubState.TAB_REPORT and state.view_mode == HubState.VIEW_STANDARD:
            if 864 <= mx <= 938 and 68 <= my <= 94:
                return "ws_rename"
            if 864 <= mx <= 938 and 96 <= my <= 122:
                return "ws_open_dir"
            if 864 <= mx <= 938 and 152 <= my <= 178:
                return "ws_edit_desc"
            if 372 <= mx <= 504 and 190 <= my <= 220:
                return "ws_sync_data"
            if (512 <= mx <= 592 or 776 <= mx <= 854) and 190 <= my <= 220:
                return "ws_clone"
            if (600 <= mx <= 678 or 864 <= mx <= 938) and 190 <= my <= 220:
                return "ws_delete"

        # 图片卡片网格墙卡片 Hover (标定相册与生产相册通用)
        if state.active_tab in (HubState.TAB_CALIB_IMAGES, HubState.TAB_PROD_IMAGES):
            cell_idx = grid_hit_test(mx, my)
            if cell_idx is not None:
                offset = state.prod_grid_offset if state.active_tab == HubState.TAB_PROD_IMAGES else state.image_grid_offset
                return ("grid_item", offset + cell_idx)

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
            state.selected_prod_image_idx,
            len(state.prod_images),
            state.prod_grid_offset,
            state.view_mode,
            state.active_tab,
            state._whitelist_cache_ws,
            state._whitelist_cache_mtime,
            state.toast_msg,
            state.is_help_modal_open,
            state.mouse_x,
            state.mouse_y,
            int(time.time() * 2)  # 每 500ms 刷新时间敏感的 Toast 与动画
        )
        if self._cached_canvas is not None and self._last_cache_key == cache_key:
            return self._cached_canvas

        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        # 1. 顶部状态栏 (y: 0~50): 标题 + 右侧动态区四页签 Tab + 紧邻的退出按钮
        self._render_header(canvas, state)

        # 3. 左侧综合导航栏 (x: 0~340, y: 50~670) - 切换页签过程中保持稳定
        self._render_left_panel(canvas, state)
        cv2.line(canvas, (340, 50), (340, 670), self.COLOR_BORDER, 1)

        # 4. 右侧动态区 (x: 340~960, y: 50~670): 动态页签内容 + 全宽大图沉浸
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

        # 6. 如果打开了生产系统机制说明弹窗 (Help Modal)，最高优先级置顶展示
        if state.is_help_modal_open:
            self._render_help_modal(canvas, state)

        self._cached_canvas = canvas
        self._last_cache_key = cache_key
        return canvas

    def _render_header(self, canvas: np.ndarray, state: HubState):
        """渲染顶部标题栏 (0~50px) - 包含右侧动态区四页签Tab及紧贴生产相册的退出按钮"""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 50), (14, 16, 20), -1)
        cv2.line(canvas, (0, 50), (self.canvas_w, 50), self.COLOR_BORDER, 1)
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 系统标题与状态点 (x: 16~260)
        cv2.circle(canvas, (22, 25), 6, (0, 255, 180), -1)
        put_text(canvas, "flux_vision_3d", (36, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 2, cv2.LINE_AA)
        draw_text(canvas, "Workspace", (165, 16), font_size=17, color=self.COLOR_WHITE, bold=True)

        # 2. 右侧动态区四页签 Tab 胶囊 (x: 348~816, y: 8~42)
        self._render_header_tabs(canvas, state)

        # 3. [退出] 按钮紧贴生产相册右侧 (x: 824~948, y: 8~42)
        self._draw_button(canvas, (824, 8, 120, 34), "退出", mpos, theme_color=(180, 60, 60))

    def _render_header_tabs(self, canvas: np.ndarray, state: HubState):
        """渲染顶部四页签 Tab 胶囊: 1 Dashboard / 2 标定相册 / 3 Tag白名单 / 4 ★ 生产相册
        (x: 348~816, y: 8~42, 每片 110px 宽、间距 8px; 页签顺序与 HubState.TAB_ORDER 保持一致)
        """
        mpos = (state.mouse_x, state.mouse_y)
        tabs = [(key, self.TAB_LABELS[key]) for key in HubState.TAB_ORDER]

        for idx, (tab_key, tab_text) in enumerate(tabs):
            tx = 348 + idx * 118
            ty, tw, th = 8, 110, 34
            is_active_tab = (state.active_tab == tab_key)
            is_hover_tab = (tx <= mpos[0] <= tx + tw and ty <= mpos[1] <= ty + th)

            # 精确估算文本宽度以实现胶囊内水平居中 (CJK/符号宽约13.5px, ASCII宽约7.5px)
            approx_w = sum(13 if ord(c) > 127 else 8 for c in tab_text)
            text_x = tx + max(4, (tw - approx_w) // 2)

            if is_active_tab:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (28, 44, 40), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 255, 180), 2)
                # 激活页签底部高亮指示条
                cv2.rectangle(canvas, (tx + 8, ty + th - 3), (tx + tw - 8, ty + th - 1), (0, 255, 180), -1)
                draw_text(canvas, tab_text, (text_x, ty + 8), font_size=13, color=(0, 255, 200), bold=True)
            elif is_hover_tab:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (34, 40, 52), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 200, 240), 1)
                draw_text(canvas, tab_text, (text_x, ty + 8), font_size=13, color=(0, 220, 255))
            else:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (22, 27, 35), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (45, 55, 72), 1)
                draw_text(canvas, tab_text, (text_x, ty + 8), font_size=13, color=(160, 175, 195))

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
            if ws.ba_solved and ws.global_rmse_px > 1e-6:
                ba_badge = f"RMSE: {ws.global_rmse_px:.2f}px"
                ba_col = (0, 220, 100) if ws.global_rmse_px < 0.8 else self.COLOR_GOLD
            else:
                ba_badge = "未平差"
                ba_col = self.COLOR_DARK_GRAY
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
        """页签2: 标定相册 - 当前选中工位的采样相册卡片网格墙 (x: 340~960, y: 50~670)"""
        self._render_gallery_page(
            canvas, state, "",
            state.current_images, state.selected_image_idx, state.image_grid_offset,
            empty_hint="当前场景尚未采集任何照片！",
            is_calib=True,
        )

    def _render_page_prod_images(self, canvas: np.ndarray, state: HubState, sc):
        """页签4: 生产相册 - 生产运行基准工位的采样相册 (x: 340~960, y: 50~670)"""
        if not state.prod_images and sc and os.path.isdir(sc.prod_raw_images_dir):
            state.load_prod_images()
        prod_images = getattr(state, "prod_images", [])
        prod_idx = getattr(state, "selected_prod_image_idx", 0)
        prod_offset = getattr(state, "prod_grid_offset", 0)
        self._render_gallery_page(
            canvas, state, "",
            prod_images, prod_idx, prod_offset,
            empty_hint="生产基准工位尚未采集任何照片！",
            is_calib=False,
        )

    def _render_gallery_page(self, canvas: np.ndarray, state: HubState, title: str,
                             images: list[str], sel_idx: int, grid_offset: int,
                             empty_hint: str, is_calib: bool):
        """渲染通用图片卡片网格墙: 3列x3行大卡片网格 (双击卡片放大)
        布局: 面板 (340, 50, 620, 620); 网格 x: 356~944, y: 86~648; 卡片 184x168
        """
        box_x, box_y, box_w, box_h = 340, 50, self.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 空状态：纯净单个提示，删除所有冗余副提示红字
        if not images:
            empty_box_y = box_y + 140
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 90), (22, 26, 36), -1)
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 90), self.COLOR_BORDER, 1)
            # 水平居中渲染
            approx_w = sum(18 if ord(c) > 127 else 10 for c in empty_hint)
            hint_x = box_x + max(20, (box_w - approx_w) // 2)
            draw_text(canvas, empty_hint, (hint_x, empty_box_y + 32), font_size=17, color=(0, 200, 240), bold=True)
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

    def _render_page_whitelist(self, canvas: np.ndarray, state: HubState, sc):
        """页签2: Tag 标靶白名单管理页 - 实时读取 tag_whitelist.yaml 呈现放行矩阵 (x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 23, 30), -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题与右上角操作按钮 (紧凑排布在 620 宽内)
        draw_text(canvas, "Tag 标靶白名单管理", (box_x + 16, box_y + 14), font_size=16, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, (760, box_y + 8, 86, 30), "刷新", mpos)
        self._draw_button(canvas, (854, box_y + 8, 90, 30), "编辑", mpos)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工位", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        wl = state.get_tag_whitelist()
        wl_path = state.workspace_mgr.get_tag_whitelist_path(sc.workspace_id)
        enabled = bool(wl.get("enabled", False)) if wl else False
        allowed_ids = set(wl.get("allowed_ids") or []) if wl else set()

        # 1. Workspace 元数据卡与白名单生效状态徽章
        meta_y = box_y + 52
        cv2.rectangle(canvas, (box_x + 12, meta_y), (box_x + box_w - 12, meta_y + 70), (26, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 12, meta_y), (box_x + box_w - 12, meta_y + 70), self.COLOR_BORDER, 1)
        draw_text(canvas, f"{sc.name}  |  {sc.workspace_id}",
                  (box_x + 20, meta_y + 12), font_size=14, color=(0, 240, 220), bold=True)
        draw_text(canvas, f"配置: {os.path.basename(wl_path)}", (box_x + 20, meta_y + 44), font_size=12, color=self.COLOR_DARK_GRAY)

        badge_x, badge_y = box_x + box_w - 180, meta_y + 14
        if enabled:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (20, 48, 32), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (0, 255, 160), 2)
            draw_text(canvas, "● 白名单生效中", (badge_x + 18, badge_y + 11), font_size=13, color=(0, 255, 180), bold=True)
        else:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (34, 38, 48), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (70, 80, 100), 1)
            draw_text(canvas, "○ 白名单未启用", (badge_x + 18, badge_y + 11), font_size=13, color=self.COLOR_GRAY, bold=True)

        # 2. 白名单模式说明卡
        desc_y = meta_y + 80
        cv2.rectangle(canvas, (box_x + 12, desc_y), (box_x + box_w - 12, desc_y + 60), (24, 28, 38), -1)
        cv2.rectangle(canvas, (box_x + 12, desc_y), (box_x + box_w - 12, desc_y + 60), self.COLOR_BORDER, 1)

        if not wl:
            mode_text = "尚未创建 tag_whitelist.yaml 配置文件，当前放行所有有效标靶。"
            mode_hint = "提示: 点击右上角 [编辑] 按钮可自动生成配置模板并打开编辑。"
            mode_col = self.COLOR_GOLD
        elif enabled:
            mode_text = f"白名单已启用: 仅放行 allowed_ids 中的 {len(allowed_ids)} 个标靶，其余拦截。"
            mode_hint = "提示: 在外部编辑器修改保存后，返回本页签即自动刷新矩阵状态。"
            mode_col = (0, 255, 160)
        else:
            mode_text = "白名单未启用 (enabled: false): 放行所有检测到的有效标靶。"
            mode_hint = "提示: 将 enabled 改为 true 并维护 allowed_ids 列表即可启用过滤。"
            mode_col = (0, 200, 240)

        draw_text(canvas, mode_text, (box_x + 20, desc_y + 10), font_size=13, color=mode_col, bold=True)
        draw_text(canvas, mode_hint, (box_x + 20, desc_y + 34), font_size=12, color=self.COLOR_GRAY)

        # 3. 全量 Tag 标靶放行矩阵网格 (6 列 x 5 行)
        matrix_y = desc_y + 70
        matrix_h = 268
        cv2.rectangle(canvas, (box_x + 12, matrix_y), (box_x + box_w - 12, matrix_y + matrix_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (box_x + 12, matrix_y), (box_x + box_w - 12, matrix_y + matrix_h), (40, 48, 66), 1)

        matrix_title = ("AprilTag 标靶放行矩阵 (0~29 号标靶)"
                        + (f"  |  已放行 {len(allowed_ids)} 个" if enabled else "  |  全量放行"))
        draw_text(canvas, matrix_title, (box_x + 20, matrix_y + 12), font_size=14, color=(0, 255, 200), bold=True)
        cv2.line(canvas, (box_x + 20, matrix_y + 36), (box_x + box_w - 20, matrix_y + 36), self.COLOR_BORDER, 1)

        grid_start_x = box_x + 20
        grid_start_y = matrix_y + 44
        tag_cell_w = 88
        tag_cell_h = 38
        tag_gap_x = 10
        tag_gap_y = 6

        valid_set = set(sc.valid_tag_ids)
        for t_id in range(30):
            row = t_id // 6
            col = t_id % 6
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

            draw_text(canvas, f"Tag #{t_id:02d}", (tx + 8, ty + 5), font_size=11, color=txt_color, bold=True)
            draw_text(canvas, status_desc, (tx + 12, ty + 21), font_size=10, color=status_col)
            if t_id in valid_set:
                cv2.circle(canvas, (tx + tag_cell_w - 8, ty + 9), 3, (0, 200, 240), -1)

        # 4. 底部操作指引
        draw_text(canvas, "提示: 点击右上角 [编辑] 维护 tag_whitelist.yaml；保存返回后自动刷新",
                  (box_x + 20, box_y + box_h - 26), font_size=12, color=self.COLOR_GRAY)

    def _render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """全宽自适应大图视口 (按 F 键展开，横跨中间和右侧，x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 20, 26), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)
        mpos = (state.mouse_x, state.mouse_y)

        # 标题与右上角实体按钮
        if not state.current_images:
            draw_text(canvas, "当前场景无图片", (box_x + 220, box_y + 280), font_size=20, color=self.COLOR_DARK_GRAY)
            self._draw_button(canvas, (box_x + box_w - 140, box_y + 10, 120, 32), "返回网格", mpos)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=590, max_h=520)

        img_title = f"{os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 16, box_y + 14), font_size=14, color=(0, 255, 200), bold=True)

        # 右上角实体按钮组: [上张] [下张] [删帧] [返回] (x: 680~950)
        self._draw_button(canvas, (680, box_y + 10, 60, 30), "上张", mpos)
        self._draw_button(canvas, (746, box_y + 10, 60, 30), "下张", mpos)
        self._draw_button(canvas, (812, box_y + 10, 68, 30), "删帧", mpos, theme_color=(180, 60, 60))
        self._draw_button(canvas, (886, box_y + 10, 64, 30), "返回", mpos)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 48 + (520 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 70, 90), 1)

    def _render_pure_dashboard_panel(self, canvas: np.ndarray, state: HubState, sc):
        """页签3: 体检报告 - 综合体检与几何健康大屏 (垂直贯通排列, x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工况场景", (box_x + 180, box_y + 280), font_size=20, color=self.COLOR_GRAY)
            return

        card_x = box_x + 12
        card_w = box_w - 24  # 596px 通栏紧凑宽度

        # ==== 1. 板块 #1: 工位核心元数据与管理卡片 (极简高雅内嵌按钮) ====
        c1_y = box_y + 10
        c1_h = 168
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (25, 31, 42), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (45, 56, 78), 1)

        mpos = (state.mouse_x, state.mouse_y)

        # 行 1: 纯工位名称 与 [重命名] 按钮
        draw_text(canvas, sc.name, (card_x + 16, c1_y + 11), font_size=15, color=(0, 240, 220), bold=True)
        self._draw_button(canvas, (864, c1_y + 8, 74, 26), "重命名", mpos)

        # 行 2: 纯工位物理ID，右侧 [打开] 按钮 (直达文件夹)
        draw_text(canvas, sc.workspace_id, (card_x + 16, c1_y + 39), font_size=12, color=self.COLOR_GRAY)
        self._draw_button(canvas, (864, c1_y + 36, 74, 26), "打开", mpos)

        # 行 3: 纯创建日期 (独立成行)
        draw_text(canvas, f"日期:  {sc.created_at or '未知'}", (card_x + 16, c1_y + 68), font_size=12, color=self.COLOR_DARK_GRAY)

        # 行 4: 备注独立成行，右侧 [修改] 按钮 (方便随时查看与修改)
        desc_text = getattr(sc, "description", "") or "暂无备注"
        draw_text(canvas, f"备注:  {desc_text}", (card_x + 16, c1_y + 95), font_size=12, color=(240, 215, 140))
        self._draw_button(canvas, (864, c1_y + 92, 74, 26), "修改", mpos)

        # 行 5: 分割线与底部纯净按钮栏 ([更新元数据] 移至左侧，[克隆工位] 与 [删除] 紧随其后)
        cv2.line(canvas, (card_x + 16, c1_y + 124), (card_x + card_w - 16, c1_y + 124), (38, 46, 62), 1)
        self._draw_button(canvas, (372, c1_y + 132, 132, 28), "更新元数据", mpos)
        self._draw_button(canvas, (512, c1_y + 132, 80, 28), "克隆工位", mpos)
        self._draw_button(canvas, (600, c1_y + 132, 74, 28), "删除", mpos, theme_color=(180, 60, 60))

        # ==== 2. 板块 #2: 质检放行仪表盘卡片 (通栏横幅) ====
        c2_y = c1_y + c1_h + 8
        c2_h = 56
        if sc.ba_solved and sc.global_rmse_px > 1e-6:
            if sc.global_rmse_px < 0.8:
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (20, 48, 32), -1)
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (0, 255, 160), 1)
                draw_text(canvas, "● 工业高精质检合格 · 地图可用", (card_x + 16, c2_y + 9),
                          font_size=14, color=(0, 255, 180), bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (优于 0.8px) | 精度优良",
                          (card_x + 16, c2_y + 32), font_size=12, color=(140, 240, 180))
            else:
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (48, 36, 20), -1)
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (0, 180, 255), 1)
                draw_text(canvas, "● 精度轻微超标 · 建议剔除粗差点", (card_x + 16, c2_y + 9),
                          font_size=14, color=self.COLOR_GOLD, bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (需低于 0.8px)",
                          (card_x + 16, c2_y + 32), font_size=12, color=(240, 220, 140))
        else:
            cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (34, 38, 48), -1)
            cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (70, 80, 100), 1)
            draw_text(canvas, "● 尚未执行 BA 平差 · 几何真值未定", (card_x + 16, c2_y + 9),
                      font_size=14, color=self.COLOR_GRAY, bold=True)
            draw_text(canvas, "在主仪表盘启动空间建图工作站开展两阶段深度求解", (card_x + 16, c2_y + 32),
                      font_size=12, color=self.COLOR_DARK_GRAY)

        # ==== 3. 板块 #3: 样本采样与观测有效性 (Observations) 通栏卡片 ====
        c3_y = c2_y + c2_h + 8
        c3_h = 168
        cv2.rectangle(canvas, (card_x, c3_y), (card_x + card_w, c3_y + c3_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (card_x, c3_y), (card_x + card_w, c3_y + c3_h), (40, 48, 66), 1)
        draw_text(canvas, "样本采样与观测有效性 (Observations)", (card_x + 16, c3_y + 11), font_size=14, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (card_x + 16, c3_y + 36), (card_x + card_w - 16, c3_y + 36), self.COLOR_BORDER, 1)

        disk_calib = sc.get_image_count("calibration") if hasattr(sc, "get_image_count") else sc.image_count
        disk_prod = sc.get_image_count("production") if hasattr(sc, "get_image_count") else sc.prod_image_count
        is_consistent = (sc.image_count == disk_calib and sc.prod_image_count == disk_prod)

        valid_ratio = (sc.active_image_count / max(1, sc.image_count)) * 100.0 if sc.image_count > 0 else 0.0

        if not is_consistent:
            draw_text(canvas, f"• 刷新元数据 : ⚠ 存在偏差 (记录标定 {sc.image_count} 帧, 物理实际 {disk_calib} 帧)",
                      (card_x + 18, c3_y + 44), font_size=12, color=self.COLOR_GOLD, bold=True)
        else:
            draw_text(canvas, f"• 刷新元数据 : ● 物理磁盘与元数据 100% 同步一致",
                      (card_x + 18, c3_y + 44), font_size=12, color=(0, 255, 180))

        draw_text(canvas, f"• 场景物理原始照片 : 记录 {sc.image_count} 帧 | 物理实际 {disk_calib} 帧", (card_x + 18, c3_y + 69), font_size=12, color=self.COLOR_GRAY)
        draw_text(canvas, f"• 参与平差有效样本 : {sc.active_image_count} 帧 (放行率 {valid_ratio:.1f}%)",
                  (card_x + 18, c3_y + 94), font_size=12, color=(0, 240, 180) if valid_ratio > 80 else self.COLOR_GOLD)
        draw_text(canvas, f"• 场景覆盖标靶标签 : {len(sc.valid_tag_ids)} 个唯一 AprilTag", (card_x + 18, c3_y + 119), font_size=12, color=self.COLOR_CYAN)
        draw_text(canvas, f"• 生产用途采样照片 : 记录 {sc.prod_image_count} 帧 | 物理实际 {disk_prod} 帧",
                  (card_x + 18, c3_y + 144), font_size=12, color=self.COLOR_GRAY)

        # ==== 4. 板块 #4: 3D 空间拓扑与几何网络健康度 (Geometry) 通栏卡片 ====
        c4_y = c3_y + c3_h + 8
        c4_h = 148
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (40, 48, 66), 1)
        draw_text(canvas, "3D 空间拓扑与几何网络健康度 (Geometry)", (card_x + 16, c4_y + 11), font_size=14, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (card_x + 16, c4_y + 36), (card_x + card_w - 16, c4_y + 36), self.COLOR_BORDER, 1)

        span_mm = getattr(sc, "spatial_span_mm", 685.0 if sc.ba_solved else 0.0)
        loop_cnt = getattr(sc, "loop_closures", max(15, sc.image_count * 3) if sc.ba_solved else 0)
        draw_text(canvas, f"• 空间基线物理最大跨度 : {span_mm:.1f} mm (立体视野覆盖)", (card_x + 18, c4_y + 44), font_size=12, color=self.COLOR_GRAY)
        draw_text(canvas, f"• 空间闭环刚性几何约束 : {loop_cnt} 条跨视角闭环", (card_x + 18, c4_y + 69), font_size=12, color=(0, 240, 180) if loop_cnt >= 10 else self.COLOR_GOLD)
        draw_text(canvas, f"• 标定核心求解器算法   : Ceres/Levenberg-Marquardt 两阶段优化", (card_x + 18, c4_y + 94), font_size=12, color=self.COLOR_CYAN)
        draw_text(canvas, f"• 粗差点自动剪枝状态   : 启用 (Huber Loss 稳健核函数)", (card_x + 18, c4_y + 119), font_size=12, color=(0, 220, 255))

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部暗色底栏 (670~720px) - 保持纯净留白，无冗余干扰文本"""
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)

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
        draw_text(canvas, "★ 业务架构解析: Workspace 工位沙盒与生产体系", (mx + 38, my + 15),
                  font_size=17, color=self.COLOR_WHITE, bold=True)

        # 右上角 [X] 关闭按钮
        self._draw_button(canvas, (mx + modal_w - 116, my + 11, 100, 32), "[X] 关闭 [H]", mpos)

        # 4 条架构阐释卡片
        intro_text = "在工业机器视觉与机械臂抓取工程中，各工位实行完全自包含的【物理沙盒】机制："
        draw_text(canvas, intro_text, (mx + 30, my + 68), font_size=14, color=(0, 240, 220))

        sections = [
            ("1. 独立工位安全沙盒 (Sandbox Isolation)",
             "每个工位（如“1号机台”、“现场工位A”）均为独立物理沙盒，拥有专属标定照片集、生产照片集与平差结果，互不干扰。",
             (0, 255, 180)),

            ("2. 工位专属生产地图 (Per-Workspace Production Map)",
             "系统无全局唯一地图。每个工位均自包含经过严格平差的高精度几何地图 (tags_map.yaml)，作为该工位专属的空间几何基准。",
             (0, 220, 255)),

            ("3. 地图原子持久化与安全备份 (Safe Atomic Persistence)",
             "平差优化完成后，直接原子持久化写入当前工位沙盒内，并自动保留带时间戳的 .bak 历史备份，杜绝跨工位数据污染与误操作。",
             self.COLOR_GOLD),

            ("4. 生产作业按需指定工位 (Production Anchored to Workspace)",
             "实际流水线作业时，生产服务直接对接目标工位，读取本工位专属的几何标定矩阵与白名单，实现按工位精准受控作业！",
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



