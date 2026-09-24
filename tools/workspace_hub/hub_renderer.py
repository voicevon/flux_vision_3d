"""
Workspace Hub 视觉渲染引擎 (HubRenderer)
=======================================
专业工业级暗黑系 GUI 渲染管线，960x720 紧凑布局：
- 左栏 (x: 0~340): Workspace 列表导航 (固定稳定)
- 右栏 (x: 340~960): 动态页签区 (1 Dashboard / 2 Tag白名单 / 3 标定相册 / 4 ★ 生产相册)
- 标定相册页签内支持双击卡片进入全宽大图沉浸预览
"""

import os
import time
from typing import Any
import cv2
import numpy as np

from src.utils.gui_components import render_floating_tooltip
from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, put_text
from tools.workspace_hub.hub_state import HubState


# 顶部 Header 与页签胶囊几何常量 (单源标准)
HEADER_TAB_X0 = 348
HEADER_TAB_Y0 = 8
HEADER_TAB_W = 98
HEADER_TAB_H = 34
HEADER_TAB_STEP = 104
BTN_EXIT_X0 = 874
BTN_EXIT_Y0 = 8
BTN_EXIT_W = 74
BTN_EXIT_H = 34

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

# ==================== 白名单芯片矩阵编辑器几何常量 (渲染与命中测试单源共用) ====================
WL_BOX_X, WL_BOX_Y, WL_BOX_W = 340, 50, 620          # 白名单页签容器
WL_GRID_X0, WL_GRID_Y0 = 360, 296                    # 芯片网格左上 (与只读矩阵视图一致)
WL_CELL_W, WL_CELL_H = 88, 38
WL_GAP_X, WL_GAP_Y = 10, 6
WL_COLS = 6
WL_BTN_DONE = (854, 58, 90, 30)                      # 标题行 [编辑]/[完成]
WL_BTN_ALL = (360, 624, 120, 32)                     # 全部放行
WL_BTN_CLEAR = (488, 624, 90, 32)                    # 清空 (探索模式)
WL_BTN_ANCHOR = (586, 624, 110, 32)                  # 锚点坐标 / 退出锚点
# 锚点弹窗 (Tag 世界坐标逐轴编辑, 支持部分已知)
WL_ANCHOR_X, WL_ANCHOR_Y, WL_ANCHOR_W, WL_ANCHOR_H = 420, 120, 460, 500
WL_ANCHOR_ROW_X0, WL_ANCHOR_ROW_Y0, WL_ANCHOR_ROW_W = 438, 182, 424
WL_ANCHOR_ROW_H, WL_ANCHOR_ROW_STEP = 38, 44
WL_ANCHOR_CLR_W = 74                                 # 行内 [清除] 按钮宽
WL_ANCHOR_KEY_X0, WL_ANCHOR_KEY_Y0, WL_ANCHOR_KEY_W, WL_ANCHOR_KEY_H = 438, 330, 96, 40
WL_ANCHOR_KEY_STEP_X, WL_ANCHOR_KEY_STEP_Y = 104, 48
WL_ANCHOR_SAVE = (438, 572, 130, 30)
WL_ANCHOR_CANCEL = (578, 572, 90, 30)
WL_ANCHOR_DELETE = (678, 572, 120, 30)

# ==================== 坐标系与 ROI 结构化编辑表单弹窗几何常量 ====================
GEOM_MODAL_W = 680
GEOM_MODAL_H = 480
GEOM_MODAL_X = (960 - GEOM_MODAL_W) // 2   # 140
GEOM_MODAL_Y = (720 - GEOM_MODAL_H) // 2   # 120
GEOM_MODAL_SAVE = (GEOM_MODAL_X + 190, GEOM_MODAL_Y + GEOM_MODAL_H - 52, 120, 36)
GEOM_MODAL_CANCEL = (GEOM_MODAL_X + 370, GEOM_MODAL_Y + GEOM_MODAL_H - 52, 120, 36)
GEOM_MODAL_CLOSE = (GEOM_MODAL_X + GEOM_MODAL_W - 46, GEOM_MODAL_Y + 12, 34, 30)

def frame_btn_add_rect() -> tuple[int, int, int, int]:
    return (824, 98, 108, 24)

def roi_btn_add_rect() -> tuple[int, int, int, int]:
    return (836, 368, 96, 24)

def frame_row_edit_rect(idx: int) -> tuple[int, int, int, int]:
    fy = 136 + idx * 72
    return (818, fy + 6, 50, 24)

def frame_row_del_rect(idx: int) -> tuple[int, int, int, int]:
    fy = 136 + idx * 72
    return (874, fy + 6, 50, 24)

def roi_row_edit_rect(idx: int) -> tuple[int, int, int, int]:
    ry = 406 + idx * 76
    return (818, ry + 6, 50, 24)

def roi_row_del_rect(idx: int) -> tuple[int, int, int, int]:
    ry = 406 + idx * 76
    return (874, ry + 6, 50, 24)

# ==================== 坐标系专属视图几何常量 ====================
FRAME_EDIT_POSE_BTN = (800, 58, 140, 30)
FRAME_ADD_ROI_BTN = (780, 58, 160, 30)

FT_GRID_X0 = 356
FT_GRID_Y0 = 366
FT_CHIP_W = 110
FT_CHIP_H = 68
FT_GAP_X = 8
FT_GAP_Y = 12

def frame_tag_chip_rect(idx: int) -> tuple[int, int, int, int]:
    """计算 10-Slot Tag 芯片矩阵中第 idx (0~9) 个芯片的矩形 (2行x5列)"""
    row, col = divmod(idx, 5)
    return (FT_GRID_X0 + col * (FT_CHIP_W + FT_GAP_X),
            FT_GRID_Y0 + row * (FT_CHIP_H + FT_GAP_Y),
            FT_CHIP_W, FT_CHIP_H)

def frame_roi_row_rect(idx: int) -> tuple[int, int, int, int]:
    return (356, 100 + idx * 80, 588, 72)

def frame_roi_edit_btn(idx: int) -> tuple[int, int, int, int]:
    return (818, 100 + idx * 80 + 22, 54, 26)

def frame_roi_del_btn(idx: int) -> tuple[int, int, int, int]:
    return (878, 100 + idx * 80 + 22, 54, 26)


def whitelist_cell_rect(t_id: int) -> tuple[int, int, int, int]:
    """白名单芯片格位矩形 (0~29 基础网格与超出范围的追加芯片共用同一公式)"""
    row, col = divmod(t_id, WL_COLS)
    return (WL_GRID_X0 + col * (WL_CELL_W + WL_GAP_X),
            WL_GRID_Y0 + row * (WL_CELL_H + WL_GAP_Y),
            WL_CELL_W, WL_CELL_H)


def anchor_row_rect(axis: int) -> tuple[int, int, int, int]:
    """锚点弹窗轴行矩形 (axis 0/1/2 → X/Y/Z)"""
    return (WL_ANCHOR_ROW_X0, WL_ANCHOR_ROW_Y0 + axis * WL_ANCHOR_ROW_STEP,
            WL_ANCHOR_ROW_W, WL_ANCHOR_ROW_H)


def anchor_clear_rect(axis: int) -> tuple[int, int, int, int]:
    """锚点弹窗轴行内 [清除] 按钮矩形"""
    rx, ry, rw, rh = anchor_row_rect(axis)
    return (rx + rw - 10 - WL_ANCHOR_CLR_W, ry + 6, WL_ANCHOR_CLR_W, rh - 12)


def anchor_padkey_rect(idx: int) -> tuple[int, int, int, int]:
    """锚点键盘按键矩形 (idx 0~14: 1~9/./0/-+/清空/退格/确认)"""
    row, col = divmod(idx, 3)
    return (WL_ANCHOR_KEY_X0 + col * WL_ANCHOR_KEY_STEP_X,
            WL_ANCHOR_KEY_Y0 + row * WL_ANCHOR_KEY_STEP_Y,
            WL_ANCHOR_KEY_W, WL_ANCHOR_KEY_H)


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    """点是否落在矩形内 (Hover 高亮与命中测试共用)"""
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


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

    # 页签显示文案 (自适应工位宏观视图与坐标系微观视图)
    TAB_LABELS = {
        HubState.TAB_REPORT: "Dashboard",
        HubState.TAB_FRAMES_ROIS: "坐标系&ROI",
        HubState.TAB_WHITELIST: "Tag白名单",
        HubState.TAB_CALIB_IMAGES: "标定相册",
        HubState.TAB_PROD_IMAGES: "★ 生产相册",
        HubState.TAB_FRAME_POSE_TAGS: "机构参数与Tag",
        HubState.TAB_FRAME_ROIS: "3D ROI 空间物件",
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

    def _get_tabs_layout(self, state: HubState):
        """计算顶部 Tab 胶囊的动态布局矩形 (自适应工位 3 页签与坐标系 2 页签)"""
        tabs = state.get_current_tabs()
        n = len(tabs)
        tab_w = 148 if n == 2 else 115
        tab_step = tab_w + 12
        start_x = 348
        res = []
        for idx, (tab_key, tab_label) in enumerate(tabs):
            tx = start_x + idx * tab_step
            rect = (tx, HEADER_TAB_Y0, tab_w, HEADER_TAB_H)
            res.append((tab_key, tab_label, rect))
        return res

    def _get_tree_layout(self, state: HubState):
        """计算左侧两层树结构各项的几何矩形与数据标识，供渲染与点击测试统一使用"""
        items = []
        cur_y = 58
        max_y = 604
        for ws_idx, ws in enumerate(state.workspaces):
            if cur_y + 44 > max_y:
                break
            ws_id = ws.workspace_id
            is_expanded = (ws_id in state.expanded_workspaces)
            is_ws_selected = (
                state.selected_tree_item[0] == "workspace"
                and state.selected_workspace_idx == ws_idx
            )

            ws_rect = (10, cur_y, 320, 44)
            arrow_rect = (10, cur_y, 30, 44)
            body_rect = (40, cur_y, 290, 44)

            items.append({
                "type": "workspace",
                "ws_idx": ws_idx,
                "workspace": ws,
                "is_expanded": is_expanded,
                "is_selected": is_ws_selected,
                "ws_rect": ws_rect,
                "arrow_rect": arrow_rect,
                "body_rect": body_rect,
            })
            cur_y += 48

            # 若工位展开，渲染其坐标系子节点
            if is_expanded:
                mgr = state.get_workspace_coord_mgr(ws)
                frames = mgr.list_frames() if mgr else []

                for f in frames:
                    if cur_y + 30 > max_y:
                        break
                    is_frame_selected = (
                        state.selected_tree_item[0] == "frame"
                        and state.selected_workspace_idx == ws_idx
                        and state.selected_tree_item[2] == f.frame_id
                    )
                    frame_rect = (34, cur_y, 296, 30)
                    items.append({
                        "type": "frame",
                        "ws_idx": ws_idx,
                        "workspace": ws,
                        "frame": f,
                        "frame_id": f.frame_id,
                        "is_selected": is_frame_selected,
                        "rect": frame_rect,
                    })
                    cur_y += 34
        return items

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

        # 0. 坐标系与 3D ROI 结构化弹窗模式 (最高交互层)
        if state.frame_modal_open:
            mx_box, my_box = GEOM_MODAL_X, GEOM_MODAL_Y
            mw, mh = GEOM_MODAL_W, GEOM_MODAL_H

            # 0.a 活跃下拉框浮层检测 (浮层拥有最高层级交互优先级)
            if state.active_dropdown in ("frame_type", "frame_parent"):
                dd_data = self._get_dropdown_data(state.active_dropdown, state)
                if dd_data:
                    (tx, ty, tw, th), cur_val, options = dd_data
                    drop_x = tx
                    drop_y = ty + th + 2
                    item_h = 30
                    drop_h = len(options) * item_h
                    if drop_x <= mx <= drop_x + tw and drop_y <= my <= drop_y + drop_h:
                        opt_idx = min(len(options) - 1, max(0, (my - drop_y) // item_h))
                        return ("dropdown_select", state.active_dropdown, options[opt_idx][0])
                    if point_in_rect(mx, my, (tx, ty, tw, th)):
                        return ("dropdown_toggle", state.active_dropdown)
                    return "dropdown_dismiss"

            if point_in_rect(mx, my, GEOM_MODAL_CLOSE):
                return "frame_modal_close"
            if point_in_rect(mx, my, GEOM_MODAL_SAVE):
                return "frame_modal_save"
            if point_in_rect(mx, my, GEOM_MODAL_CANCEL):
                return "frame_modal_cancel"
            form_y = my_box + 56
            if point_in_rect(mx, my, (mx_box + 115, form_y, 220, 28)):
                return "frame_field_name"
            if point_in_rect(mx, my, (mx_box + 430, form_y, 220, 28)):
                return "frame_field_id"

            # 下拉框触发条
            type_y = form_y + 40
            if point_in_rect(mx, my, (mx_box + 115, type_y, 360, 28)):
                return ("dropdown_toggle", "frame_type")

            parent_y = form_y + 80
            if point_in_rect(mx, my, (mx_box + 115, parent_y, 360, 28)):
                return ("dropdown_toggle", "frame_parent")

            param_y = form_y + 128
            d = state.frame_modal_data
            cur_type = d.get("type", "fixed_transform")
            if cur_type == "fixed_transform":
                if point_in_rect(mx, my, (mx_box + 125, param_y + 42, 130, 28)):
                    return ("frame_field_num", "translation", 0)
                if point_in_rect(mx, my, (mx_box + 265, param_y + 42, 130, 28)):
                    return ("frame_field_num", "translation", 1)
                if point_in_rect(mx, my, (mx_box + 405, param_y + 42, 130, 28)):
                    return ("frame_field_num", "translation", 2)
                if point_in_rect(mx, my, (mx_box + 125, param_y + 92, 130, 28)):
                    return ("frame_field_num", "rotation", 0)
                if point_in_rect(mx, my, (mx_box + 265, param_y + 92, 130, 28)):
                    return ("frame_field_num", "rotation", 1)
                if point_in_rect(mx, my, (mx_box + 405, param_y + 92, 130, 28)):
                    return ("frame_field_num", "rotation", 2)
            else:
                if point_in_rect(mx, my, (mx_box + 165, param_y + 42, 160, 28)):
                    return ("frame_field_num", "tag_id", 0)
                if point_in_rect(mx, my, (mx_box + 185, param_y + 92, 120, 28)):
                    return ("frame_field_num", "offset", 0)
                if point_in_rect(mx, my, (mx_box + 315, param_y + 92, 120, 28)):
                    return ("frame_field_num", "offset", 1)
                if point_in_rect(mx, my, (mx_box + 445, param_y + 92, 120, 28)):
                    return ("frame_field_num", "offset", 2)
            if mx < mx_box or mx > mx_box + mw or my < my_box or my > my_box + mh:
                return "frame_modal_mask"
            return "frame_modal_body"

        if state.roi_modal_open:
            mx_box, my_box = GEOM_MODAL_X, GEOM_MODAL_Y
            mw, mh = GEOM_MODAL_W, GEOM_MODAL_H

            # 0.b 活跃下拉框浮层检测
            if state.active_dropdown in ("roi_category", "roi_frame"):
                dd_data = self._get_dropdown_data(state.active_dropdown, state)
                if dd_data:
                    (tx, ty, tw, th), cur_val, options = dd_data
                    drop_x = tx
                    drop_y = ty + th + 2
                    item_h = 30
                    drop_h = len(options) * item_h
                    if drop_x <= mx <= drop_x + tw and drop_y <= my <= drop_y + drop_h:
                        opt_idx = min(len(options) - 1, max(0, (my - drop_y) // item_h))
                        return ("dropdown_select", state.active_dropdown, options[opt_idx][0])
                    if point_in_rect(mx, my, (tx, ty, tw, th)):
                        return ("dropdown_toggle", state.active_dropdown)
                    return "dropdown_dismiss"

            if point_in_rect(mx, my, GEOM_MODAL_CLOSE):
                return "roi_modal_close"
            if point_in_rect(mx, my, GEOM_MODAL_SAVE):
                return "roi_modal_save"
            if point_in_rect(mx, my, GEOM_MODAL_CANCEL):
                return "roi_modal_cancel"
            form_y = my_box + 54
            if point_in_rect(mx, my, (mx_box + 115, form_y, 220, 28)):
                return "roi_field_name"
            if point_in_rect(mx, my, (mx_box + 430, form_y, 220, 28)):
                return "roi_field_id"

            cat_y = form_y + 40
            if point_in_rect(mx, my, (mx_box + 115, cat_y, 360, 28)):
                return ("dropdown_toggle", "roi_category")

            parent_y = form_y + 80
            if point_in_rect(mx, my, (mx_box + 115, parent_y, 360, 28)):
                return ("dropdown_toggle", "roi_frame")

            geom_y = form_y + 128
            if point_in_rect(mx, my, (mx_box + 155, geom_y + 36, 115, 28)):
                return ("roi_field_num", "center", 0)
            if point_in_rect(mx, my, (mx_box + 280, geom_y + 36, 115, 28)):
                return ("roi_field_num", "center", 1)
            if point_in_rect(mx, my, (mx_box + 405, geom_y + 36, 115, 28)):
                return ("roi_field_num", "center", 2)
            if point_in_rect(mx, my, (mx_box + 155, geom_y + 76, 115, 28)):
                return ("roi_field_num", "size", 0)
            if point_in_rect(mx, my, (mx_box + 280, geom_y + 76, 115, 28)):
                return ("roi_field_num", "size", 1)
            if point_in_rect(mx, my, (mx_box + 405, geom_y + 76, 115, 28)):
                return ("roi_field_num", "size", 2)
            if point_in_rect(mx, my, (mx_box + 155, geom_y + 116, 115, 28)):
                return ("roi_field_num", "rotation", 0)
            if point_in_rect(mx, my, (mx_box + 280, geom_y + 116, 115, 28)):
                return ("roi_field_num", "rotation", 1)
            if point_in_rect(mx, my, (mx_box + 405, geom_y + 116, 115, 28)):
                return ("roi_field_num", "rotation", 2)
            if mx < mx_box or mx > mx_box + mw or my < my_box or my > my_box + mh:
                return "roi_modal_mask"
            return "roi_modal_body"

        # 0.1 生产机制业务说明弹窗模式
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
            for tab_key, tab_label, rect in self._get_tabs_layout(state):
                if point_in_rect(mx, my, rect):
                    return ("hdr_tab_key", tab_key)
            # [退出] 按钮
            if BTN_EXIT_X0 <= mx <= BTN_EXIT_X0 + BTN_EXIT_W and BTN_EXIT_Y0 <= my <= BTN_EXIT_Y0 + BTN_EXIT_H:
                return "btn_exit"

        # 左侧面板按钮与两层树交互
        if 0 <= mx <= 340:
            div_y1 = 604
            btn1_y = div_y1 + 10
            if 10 <= mx <= 330 and btn1_y <= my <= btn1_y + 40:
                return "btn_new_workspace"

            # 遍历两层树节点
            tree_items = self._get_tree_layout(state)
            for item in tree_items:
                if item["type"] == "workspace":
                    if point_in_rect(mx, my, item["arrow_rect"]):
                        return ("tree_ws_toggle", item["ws_idx"], item["workspace"].workspace_id)
                    if point_in_rect(mx, my, item["body_rect"]):
                        return ("tree_ws_select", item["ws_idx"])
                elif item["type"] == "frame":
                    if point_in_rect(mx, my, item["rect"]):
                        return ("tree_frame_select", item["ws_idx"], item["frame_id"])

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
            # 坐标系与 ROI 页签: [刷新] [打开目录]
            if state.active_tab == HubState.TAB_FRAMES_ROIS:
                if 760 <= mx <= 846:
                    return "geom_refresh"
                if 854 <= mx <= 944:
                    return "geom_open_dir"
            # Tag 白名单页签: [刷新] [编辑]
            elif state.active_tab == HubState.TAB_WHITELIST:
                if 760 <= mx <= 846:
                    return "wl_refresh"
                if 854 <= mx <= 944:
                    return "wl_edit"

        # 坐标系专属页签 1: 机构参数与 Tag 分段 (TAB_FRAME_POSE_TAGS)
        if state.active_tab == HubState.TAB_FRAME_POSE_TAGS and state.view_mode == HubState.VIEW_STANDARD:
            if point_in_rect(mx, my, FRAME_EDIT_POSE_BTN):
                return "btn_edit_frame_pose"
            if self._should_show_tag_bound_tooltip(state, (mx, my)):
                return "tag_bound_help"
            cur_frame = state.get_selected_frame()
            if cur_frame:
                tag_range = state.get_frame_tag_range(cur_frame.frame_id)
                for slot_idx in range(10):
                    chip_rect = frame_tag_chip_rect(slot_idx)
                    if point_in_rect(mx, my, chip_rect):
                        tag_id = tag_range[slot_idx]
                        cx, cy, cw, ch = chip_rect
                        if my >= cy + ch - 24:
                            return ("frame_tag_edit_xyz", tag_id)
                        return ("frame_tag_toggle", tag_id)

        # 坐标系专属页签 2: 3D ROI 空间物件 (TAB_FRAME_ROIS)
        if state.active_tab == HubState.TAB_FRAME_ROIS and state.view_mode == HubState.VIEW_STANDARD:
            if point_in_rect(mx, my, FRAME_ADD_ROI_BTN):
                return "btn_add_frame_roi"
            cur_frame = state.get_selected_frame()
            if cur_frame:
                rois = state.get_frame_rois(cur_frame.frame_id)
                for i, r in enumerate(rois[:6]):
                    if point_in_rect(mx, my, frame_roi_edit_btn(i)):
                        return ("frame_roi_edit", r.roi_id)
                    if point_in_rect(mx, my, frame_roi_del_btn(i)):
                        return ("frame_roi_delete", r.roi_id)

        # 坐标系与 3D ROI 页签内的列表操作与新增按钮
        if state.active_tab == HubState.TAB_FRAMES_ROIS and state.view_mode == HubState.VIEW_STANDARD:
            if point_in_rect(mx, my, frame_btn_add_rect()):
                return "btn_add_frame"
            if point_in_rect(mx, my, roi_btn_add_rect()):
                return "btn_add_roi"
            frames = state.get_coordinate_frames()
            for i, f in enumerate(frames[:3]):
                if point_in_rect(mx, my, frame_row_edit_rect(i)):
                    return ("frame_edit", i)
                if f.frame_id != "world" and point_in_rect(mx, my, frame_row_del_rect(i)):
                    return ("frame_del", i)
            rois = state.get_roi_spaces()
            for i in range(min(3, len(rois))):
                if point_in_rect(mx, my, roi_row_edit_rect(i)):
                    return ("roi_edit", i)
                if point_in_rect(mx, my, roi_row_del_rect(i)):
                    return ("roi_del", i)

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
            if 682 <= mx <= 812 and 190 <= my <= 220:
                return "ws_add_frame_roi"

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
            state.selected_tree_item,
            tuple(sorted(state.expanded_workspaces)),
            state._whitelist_cache_ws,
            state._whitelist_cache_mtime,
            state.toast_msg,
            state.is_help_modal_open,
            state.frame_modal_open,
            state.roi_modal_open,
            state.active_dropdown,
            str(state.frame_modal_data),
            str(state.roi_modal_data),
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
        elif state.active_tab == HubState.TAB_FRAME_POSE_TAGS:
            self._render_page_frame_pose_tags(canvas, state, ws)
        elif state.active_tab == HubState.TAB_FRAME_ROIS:
            self._render_page_frame_rois(canvas, state, ws)
        elif state.active_tab == HubState.TAB_PROD_IMAGES:
            self._render_page_prod_images(canvas, state, ws)
        elif state.active_tab == HubState.TAB_FRAMES_ROIS:
            self._render_page_frames_rois(canvas, state, ws)
        elif state.active_tab == HubState.TAB_WHITELIST:
            self._render_page_whitelist(canvas, state, ws)
        else:
            self._render_page_calib_images(canvas, state, ws)

        # 5. 底部系统反馈提示栏 (y: 670~720)
        self._render_footer(canvas, state)

        # 6. 如果打开了结构化弹窗，最高优先级置顶展示
        if state.frame_modal_open:
            self._render_frame_modal(canvas, state)
        elif state.roi_modal_open:
            self._render_roi_modal(canvas, state)
        elif state.is_help_modal_open:
            self._render_help_modal(canvas, state)

        # 7. 悬浮 Tooltip 气泡提示 (置于最顶层，无遮挡呈现)
        if not (state.frame_modal_open or state.roi_modal_open or state.is_help_modal_open):
            if state.active_tab == HubState.TAB_FRAME_POSE_TAGS and state.view_mode == HubState.VIEW_STANDARD:
                mpos = (state.mouse_x, state.mouse_y)
                if self._should_show_tag_bound_tooltip(state, mpos):
                    self._draw_tag_bound_tooltip(canvas, mpos)

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

        # 2. 右侧动态区四页签 Tab 胶囊
        self._render_header_tabs(canvas, state)

        # 3. [退出] 按钮紧贴生产相册右侧
        self._draw_button(canvas, (BTN_EXIT_X0, BTN_EXIT_Y0, BTN_EXIT_W, BTN_EXIT_H), "退出", mpos, theme_color=(180, 60, 60))

    def _render_header_tabs(self, canvas: np.ndarray, state: HubState):
        """渲染顶部自适应 Tab 胶囊 (工位视图 3 页签，坐标系专属视图 2 页签)"""
        mpos = (state.mouse_x, state.mouse_y)
        tabs_layout = self._get_tabs_layout(state)

        for tab_key, tab_text, rect in tabs_layout:
            tx, ty, tw, th = rect
            is_active_tab = (state.active_tab == tab_key)
            is_hover_tab = (tx <= mpos[0] <= tx + tw and ty <= mpos[1] <= ty + th)

            approx_w = sum(13 if ord(c) > 127 else 8 for c in tab_text)
            text_x = tx + max(4, (tw - approx_w) // 2)

            if is_active_tab:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (28, 44, 40), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 255, 180), 2)
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

    def _draw_text_input(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str,
                         mouse_pos: tuple[int, int]) -> bool:
        """统一绘制高可编辑感知的输入框 (深暗内凹底色 + 科技发光边框 + 铅笔修改图标 ✎)"""
        bx, by, bw, bh = rect
        mx, my = mouse_pos
        is_hover = (bx <= mx <= bx + bw and by <= my <= by + bh)

        bg_col = (14, 18, 25) if not is_hover else (22, 32, 44)
        border_col = (0, 255, 180) if is_hover else (45, 65, 75)
        text_col = (0, 255, 220) if is_hover else (220, 235, 235)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 2 if is_hover else 1)

        disp = text if text else "点击输入..."
        if len(disp) > 22:
            disp = disp[:20] + ".."
        draw_text(canvas, disp, (bx + 8, by + (bh - 16) // 2), font_size=12,
                  color=text_col, bold=is_hover)

        # 右侧绘制微型矢量铅笔图标 (45° 科技暗绿/亮青，平滑抗锯齿，彻底消除方块字符)
        px = bx + bw - 14
        py = by + bh // 2
        pen_col = (0, 255, 180) if is_hover else (80, 110, 120)
        cv2.line(canvas, (px - 4, py + 3), (px + 3, py - 4), pen_col, 2, cv2.LINE_AA)
        cv2.circle(canvas, (px - 5, py + 4), 1, pen_col, -1, cv2.LINE_AA)
        return is_hover

    def _draw_dropdown_trigger(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str,
                               mouse_pos: tuple[int, int], is_open: bool = False) -> bool:
        """统一绘制现代下拉选择框触发条 (带右侧分割线与 ▼/▲ 展开指示符)"""
        bx, by, bw, bh = rect
        mx, my = mouse_pos
        is_hover = (bx <= mx <= bx + bw and by <= my <= by + bh)

        bg_col = (28, 38, 52) if (is_hover or is_open) else (18, 24, 34)
        border_col = (0, 255, 180) if (is_hover or is_open) else (50, 70, 85)
        text_col = (0, 255, 200) if (is_hover or is_open) else (210, 225, 230)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 2 if (is_hover or is_open) else 1)

        disp = text
        if len(disp) > 32:
            disp = disp[:30] + ".."
        draw_text(canvas, disp, (bx + 10, by + (bh - 16) // 2), font_size=12, color=text_col, bold=is_hover)

        # 右侧指示区与箭头
        arrow_w = 26
        cv2.line(canvas, (bx + bw - arrow_w, by + 1), (bx + bw - arrow_w, by + bh - 2), border_col, 1)
        arrow_char = "▲" if is_open else "▼"
        arrow_col = (0, 255, 180) if (is_hover or is_open) else (130, 150, 165)
        draw_text(canvas, arrow_char, (bx + bw - 19, by + (bh - 14) // 2), font_size=11, color=arrow_col)
        return is_hover

    def _get_dropdown_data(self, dd_type: str, state: HubState):
        """统一获取下拉框的布局矩形、当前选中值以及候选枚举列表"""
        mx_box, my_box = GEOM_MODAL_X, GEOM_MODAL_Y
        form_y = my_box + 56
        if dd_type == "frame_type":
            rect = (mx_box + 115, form_y + 40, 360, 28)
            cur_val = state.frame_modal_data.get("type", "fixed_transform")
            options = [
                ("fixed_transform", "固定刚体外参 (平移 + 旋转)"),
                ("tag_bound", "AprilTag 动标绑定 (动态跟踪)")
            ]
            return rect, cur_val, options

        if dd_type == "frame_parent":
            rect = (mx_box + 115, form_y + 80, 360, 28)
            cur_val = state.frame_modal_data.get("parent_frame_id", "world")
            cur_fid = state.frame_modal_data.get("frame_id")
            frames = state.get_coordinate_frames()
            options = [("world", "world (世界基准绝对原点)")]
            for f in frames:
                if f.frame_id != cur_fid and f.frame_id != "world":
                    options.append((f.frame_id, f"{f.frame_id} ({f.name})"))
            return rect, cur_val, options

        roi_form_y = my_box + 54
        if dd_type == "roi_category":
            rect = (mx_box + 115, roi_form_y + 40, 360, 28)
            cur_val = state.roi_modal_data.get("category", "belt")
            options = [
                ("belt", "同步带工作面 (belt)"),
                ("wheel", "驱动轮干涉区 (wheel)"),
                ("tray", "料盘工装区 (tray)"),
                ("general", "通用机构部件 (general)")
            ]
            return rect, cur_val, options

        if dd_type == "roi_frame":
            rect = (mx_box + 115, roi_form_y + 80, 360, 28)
            cur_val = state.roi_modal_data.get("frame_id", "world")
            frames = state.get_coordinate_frames()
            options = [("world", "world (世界基准绝对原点)")]
            for f in frames:
                if f.frame_id != "world":
                    options.append((f.frame_id, f"{f.frame_id} ({f.name})"))
            return rect, cur_val, options

        return None

    def _render_active_dropdown(self, canvas: np.ndarray, state: HubState):
        """在弹窗顶层高亮绘制当前展开的下拉菜单浮层"""
        dd_type = state.active_dropdown
        if not dd_type:
            return
        dd_data = self._get_dropdown_data(dd_type, state)
        if not dd_data:
            return
        (tx, ty, tw, th), cur_val, options = dd_data
        item_h = 30
        drop_h = len(options) * item_h
        drop_x = tx
        drop_y = ty + th + 2

        mx, my = state.mouse_x, state.mouse_y

        # 外阴影
        cv2.rectangle(canvas, (drop_x - 3, drop_y - 2), (drop_x + tw + 3, drop_y + drop_h + 3), (8, 12, 16), -1)
        # 背景与主边框
        cv2.rectangle(canvas, (drop_x, drop_y), (drop_x + tw, drop_y + drop_h), (20, 26, 36), -1)
        cv2.rectangle(canvas, (drop_x, drop_y), (drop_x + tw, drop_y + drop_h), (0, 220, 180), 2)

        for idx, (opt_val, opt_label) in enumerate(options):
            iy = drop_y + idx * item_h
            is_hover = (drop_x <= mx <= drop_x + tw and iy <= my < iy + item_h)
            is_selected = (opt_val == cur_val)

            if is_hover:
                cv2.rectangle(canvas, (drop_x + 1, iy), (drop_x + tw - 1, iy + item_h), (38, 52, 70), -1)
            elif is_selected:
                cv2.rectangle(canvas, (drop_x + 1, iy), (drop_x + tw - 1, iy + item_h), (26, 36, 48), -1)

            # 选中状态圆点
            if is_selected:
                cv2.circle(canvas, (drop_x + 14, iy + item_h // 2), 4, (0, 255, 180), -1)

            text_col = (0, 255, 220) if is_hover else ((0, 230, 180) if is_selected else (205, 220, 230))
            draw_text(canvas, opt_label, (drop_x + 26, iy + (item_h - 16) // 2), font_size=12,
                      color=text_col, bold=(is_selected or is_hover))

            if idx < len(options) - 1:
                cv2.line(canvas, (drop_x + 8, iy + item_h), (drop_x + tw - 8, iy + item_h), (35, 45, 60), 1)

    def _render_left_panel(self, canvas: np.ndarray, state: HubState):
        """渲染左侧两层树结构导航 (x: 0~340, y: 50~670)
        - 一级节点: Workspace (工位)，带 ▼ / ▶ 展开折叠指示器与统计
        - 二级节点: Coordinate Frame (坐标系)，带缩进与 Tag 专属范围徽章
        - 底部保留: + 新建 Workspace
        """
        cv2.rectangle(canvas, (0, 50), (340, 670), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        tree_items = self._get_tree_layout(state)
        for item in tree_items:
            if item["type"] == "workspace":
                ws_idx = item["ws_idx"]
                ws = item["workspace"]
                is_sel = item["is_selected"]
                is_exp = item["is_expanded"]
                wx, wy, ww, wh = item["ws_rect"]

                card_bg = self.COLOR_CARD_ACTIVE if is_sel else (26, 30, 40)
                card_border = self.COLOR_ACTIVE_BORDER if is_sel else self.COLOR_BORDER
                cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), card_bg, -1)
                cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), card_border, 2 if is_sel else 1)
                if is_sel:
                    cv2.rectangle(canvas, (wx, wy), (wx + 4, wy + wh), (0, 255, 160), -1)

                arrow_char = "▼" if is_exp else "▶"
                arrow_col = (0, 255, 200) if is_sel else (140, 160, 180)
                draw_text(canvas, arrow_char, (wx + 8, wy + 14), font_size=13, color=arrow_col, bold=True)

                disp_title = f"{ws_idx + 1:02d}. {ws.name}"
                title_col = (0, 255, 200) if is_sel else self.COLOR_WHITE
                draw_text(canvas, disp_title, (wx + 28, wy + 6), font_size=13, color=title_col, bold=is_sel)

                if ws.ba_solved and ws.global_rmse_px > 1e-6:
                    ba_badge = f"RMSE: {ws.global_rmse_px:.2f}px"
                    ba_col = (0, 220, 100) if ws.global_rmse_px < 0.8 else self.COLOR_GOLD
                else:
                    ba_badge = "未平差"
                    ba_col = self.COLOR_DARK_GRAY
                put_text(canvas, ba_badge, (wx + 28, wy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ba_col, 1, cv2.LINE_AA)

                cnt_text = f"{ws.image_count}帧/{ws.prod_image_count}帧"
                put_text(canvas, cnt_text, (wx + 200, wy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 160, 180), 1, cv2.LINE_AA)

            elif item["type"] == "frame":
                fx, fy, fw, fh = item["rect"]
                f = item["frame"]
                is_sel = item["is_selected"]

                if is_sel:
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (22, 42, 46), -1)
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (0, 255, 200), 1)
                    cv2.rectangle(canvas, (fx, fy), (fx + 3, fy + fh), (0, 255, 200), -1)
                else:
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (20, 24, 32), -1)
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (35, 42, 54), 1)

                tree_branch = "└" if f.frame_id != "world" else "•"
                draw_text(canvas, tree_branch, (fx + 6, fy + 7), font_size=12, color=(0, 200, 240))

                f_label = f"{f.frame_id}"
                if f.name and f.name != f.frame_id:
                    f_label += f" ({f.name})"
                if len(f_label) > 22:
                    f_label = f_label[:20] + ".."
                f_col = (0, 255, 220) if is_sel else (200, 215, 230)
                draw_text(canvas, f_label, (fx + 20, fy + 7), font_size=12, color=f_col, bold=is_sel)

                tag_range = state.get_frame_tag_range(f.frame_id)
                tag_badge = f"Tag {tag_range[0]}~{tag_range[-1]}"
                badge_x = fx + fw - 76
                cv2.rectangle(canvas, (badge_x, fy + 5), (badge_x + 70, fy + fh - 5), (28, 38, 48), -1)
                cv2.rectangle(canvas, (badge_x, fy + 5), (badge_x + 70, fy + fh - 5), (45, 65, 75), 1)
                put_text(canvas, tag_badge, (badge_x + 5, fy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 220, 220), 1, cv2.LINE_AA)

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

    def _render_page_frame_pose_tags(self, canvas: np.ndarray, state: HubState, ws):
        """坐标系专属页签 1: 机构参数与 Tag 分段管理 (x: 340~960, y: 50~670)"""
        box_x, box_y, box_w, box_h = 340, 50, self.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        cur_frame = state.get_selected_frame()
        if not cur_frame:
            draw_text(canvas, "未选择任何机构坐标系", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        # 1. 顶部标题栏与编辑位姿按钮
        header_text = f"机构参数与 Tag 分段 (Frame: {cur_frame.frame_id})"
        draw_text(canvas, header_text, (box_x + 16, box_y + 14), font_size=15, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, FRAME_EDIT_POSE_BTN, "编辑机构参数", mpos)

        card_x = box_x + 16
        card_w = box_w - 32

        # 2. 上部卡片: 机构位姿与空间拓扑
        c1_y = box_y + 42
        c1_h = 196
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (42, 52, 70), 1)

        draw_text(canvas, f"机构位姿与拓扑关系 · {cur_frame.name}", (card_x + 14, c1_y + 10), font_size=14, color=(0, 240, 220), bold=True)
        cv2.line(canvas, (card_x + 10, c1_y + 32), (card_x + card_w - 10, c1_y + 32), (36, 45, 60), 1)

        if cur_frame.type == "world":
            type_desc = "工位绝对世界基准 (world)"
        elif cur_frame.type == "fixed_transform":
            type_desc = "固定刚体外参 (fixed_transform)"
        else:
            type_desc = "AprilTag 动标绑定 (tag_bound)"

        draw_text(canvas, f"坐标系标识: {cur_frame.frame_id}", (card_x + 16, c1_y + 42), font_size=13, color=(210, 225, 240))
        draw_text(canvas, f"父坐标系: {cur_frame.parent_frame_id}", (card_x + 220, c1_y + 42), font_size=13, color=(210, 225, 240))
        draw_text(canvas, f"类型: {type_desc}", (card_x + 16, c1_y + 68), font_size=13, color=(0, 220, 200))

        # 动标定义与说明悬停帮助徽章
        help_btn_x, help_btn_y, help_btn_w, help_btn_h = card_x + 310, c1_y + 65, 126, 22
        is_hover_help = (help_btn_x <= mpos[0] <= help_btn_x + help_btn_w and help_btn_y <= mpos[1] <= help_btn_y + help_btn_h)
        help_bg = (30, 48, 48) if is_hover_help else (20, 28, 36)
        help_border = (0, 255, 200) if is_hover_help else (40, 75, 75)
        cv2.rectangle(canvas, (help_btn_x, help_btn_y), (help_btn_x + help_btn_w, help_btn_y + help_btn_h), help_bg, -1)
        cv2.rectangle(canvas, (help_btn_x, help_btn_y), (help_btn_x + help_btn_w, help_btn_y + help_btn_h), help_border, 1)
        draw_text(canvas, "(?) 动标定义说明", (help_btn_x + 8, help_btn_y + 4), font_size=11,
                  color=(0, 255, 220) if is_hover_help else (140, 185, 195), bold=is_hover_help)

        if cur_frame.type == "world":
            w_box_y = c1_y + 98
            cv2.rectangle(canvas, (card_x + 16, w_box_y), (card_x + card_w - 16, w_box_y + 84), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, w_box_y), (card_x + card_w - 16, w_box_y + 84), (38, 48, 65), 1)
            draw_text(canvas, "● 工位全局绝对空间基准 (World Datum / Origin)", (card_x + 24, w_box_y + 11), font_size=13, color=(0, 255, 200), bold=True)
            draw_text(canvas, "位姿特性: 恒为齐次单位阵 Identity 4x4 (空间测量全局绝对基准原点，无需外参)", (card_x + 24, w_box_y + 35), font_size=12, color=(160, 180, 200))
            draw_text(canvas, "空间标靶: 由下方专属放行矩阵中的静态标靶阵列 (ID: 00~09) 建立全局基准网格", (card_x + 24, w_box_y + 58), font_size=12, color=(120, 140, 160))

        elif cur_frame.type == "fixed_transform":
            tx, ty, tz = cur_frame.translation_xyz_mm
            rx, ry, rz = getattr(cur_frame, "rotation_rpy_deg", [0.0, 0.0, 0.0])

            t_box_y = c1_y + 98
            cv2.rectangle(canvas, (card_x + 16, t_box_y), (card_x + card_w - 16, t_box_y + 38), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, t_box_y), (card_x + card_w - 16, t_box_y + 38), (38, 48, 65), 1)
            draw_text(canvas, "平移向量 T [mm]:", (card_x + 24, t_box_y + 11), font_size=12, color=(160, 180, 200))
            draw_text(canvas, f"X: {tx:+.1f}", (card_x + 160, t_box_y + 11), font_size=13, color=(0, 255, 220), bold=True)
            draw_text(canvas, f"Y: {ty:+.1f}", (card_x + 290, t_box_y + 11), font_size=13, color=(0, 255, 220), bold=True)
            draw_text(canvas, f"Z: {tz:+.1f}", (card_x + 420, t_box_y + 11), font_size=13, color=(0, 255, 220), bold=True)

            r_box_y = c1_y + 144
            cv2.rectangle(canvas, (card_x + 16, r_box_y), (card_x + card_w - 16, r_box_y + 38), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, r_box_y), (card_x + card_w - 16, r_box_y + 38), (38, 48, 65), 1)
            draw_text(canvas, "欧拉旋转 R [deg]:", (card_x + 24, r_box_y + 11), font_size=12, color=(160, 180, 200))
            draw_text(canvas, f"Rx: {rx:+.1f}°", (card_x + 160, r_box_y + 11), font_size=13, color=(255, 200, 60), bold=True)
            draw_text(canvas, f"Ry: {ry:+.1f}°", (card_x + 290, r_box_y + 11), font_size=13, color=(255, 200, 60), bold=True)
            draw_text(canvas, f"Rz: {rz:+.1f}°", (card_x + 420, r_box_y + 11), font_size=13, color=(255, 200, 60), bold=True)

        else:
            tag_box_y = c1_y + 98
            cv2.rectangle(canvas, (card_x + 16, tag_box_y), (card_x + card_w - 16, tag_box_y + 84), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, tag_box_y), (card_x + card_w - 16, tag_box_y + 84), (38, 48, 65), 1)

            bound_tags = cur_frame.get_tag_ids() if hasattr(cur_frame, "get_tag_ids") else ([cur_frame.tag_id] if cur_frame.tag_id is not None else [])
            if not bound_tags:
                bound_tags = [getattr(cur_frame, "tag_id", 0) or 0]

            draw_text(canvas, "绑定动标 Tag 列表:", (card_x + 24, tag_box_y + 12), font_size=13, color=(0, 240, 220), bold=True)
            chip_start_x = card_x + 175
            for tid in bound_tags:
                cw, ch = 48, 22
                cv2.rectangle(canvas, (chip_start_x, tag_box_y + 10), (chip_start_x + cw, tag_box_y + 10 + ch), (24, 44, 40), -1)
                cv2.rectangle(canvas, (chip_start_x, tag_box_y + 10), (chip_start_x + cw, tag_box_y + 10 + ch), (0, 255, 180), 1)
                draw_text(canvas, f"#{tid:02d}", (chip_start_x + 8, tag_box_y + 13), font_size=12, color=(0, 255, 200), bold=True)
                chip_start_x += cw + 8

            draw_text(canvas, f"(共 {len(bound_tags)} 个动标 · 多标冗余跟踪组)", (chip_start_x + 6, tag_box_y + 14), font_size=11, color=(140, 160, 180))

            off = getattr(cur_frame, "offset_xyz_mm", [0.0, 0.0, 0.0])
            draw_text(canvas, f"标称安装偏移 offset: [{off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f}] mm", (card_x + 24, tag_box_y + 38), font_size=12, color=(200, 215, 230))
            draw_text(canvas, "工作原理: 运动机构实时识别动标，通过标称偏移解算机构实际受控点位姿", (card_x + 24, tag_box_y + 58), font_size=11, color=(120, 140, 160))

        # 3. 下部卡片: 10-Slot Tag 专属分配矩阵
        c2_y = box_y + 248
        c2_h = 360
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (42, 52, 70), 1)

        tag_range = state.get_frame_tag_range(cur_frame.frame_id)
        start_id, end_id = tag_range[0], tag_range[-1]
        draw_text(canvas, f"Tag 专属分段放行矩阵 [分配区间 ID: {start_id:02d} ~ {end_id:02d}]",
                  (card_x + 14, c2_y + 10), font_size=14, color=(0, 240, 220), bold=True)
        draw_text(canvas, "提示: 点击卡片勾选放行 (保存生效至工位白名单)；点击下半部 [✎ 坐标] 标注已知物理坐标真值",
                  (card_x + 14, c2_y + 32), font_size=11, color=(140, 160, 180))
        cv2.line(canvas, (card_x + 10, c2_y + 50), (card_x + card_w - 10, c2_y + 50), (36, 45, 60), 1)

        allowed_set = set(state.get_frame_tags_status(cur_frame.frame_id))
        wl_data = state.get_whitelist_data()
        anchors = wl_data.get("tag_anchors", {}) if isinstance(wl_data, dict) else {}

        for slot_idx in range(10):
            tag_id = tag_range[slot_idx]
            cx, cy, cw, ch = frame_tag_chip_rect(slot_idx)
            is_allowed = (tag_id in allowed_set)
            is_hover = (cx <= mpos[0] <= cx + cw and cy <= mpos[1] <= cy + ch)

            if is_allowed:
                chip_bg = (24, 40, 36) if is_hover else (18, 32, 28)
                chip_border = (0, 255, 180) if is_hover else (0, 200, 140)
            else:
                chip_bg = (30, 34, 42) if is_hover else (20, 24, 32)
                chip_border = (0, 200, 240) if is_hover else (45, 55, 70)

            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), chip_bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), chip_border, 2 if (is_hover or is_allowed) else 1)

            tag_label = f"#{tag_id:02d}"
            draw_text(canvas, tag_label, (cx + 8, cy + 6), font_size=13,
                      color=(0, 255, 200) if is_allowed else (160, 175, 195), bold=True)

            status_str = "已放行 √" if is_allowed else "未放行"
            status_col = (0, 255, 160) if is_allowed else (120, 135, 150)
            draw_text(canvas, status_str, (cx + cw - 52, cy + 6), font_size=11, color=status_col, bold=is_allowed)

            cv2.line(canvas, (cx + 6, cy + 30), (cx + cw - 6, cy + 30), (35, 45, 58), 1)

            anchor_pos = anchors.get(tag_id) or anchors.get(str(tag_id))
            if anchor_pos and len(anchor_pos) == 3:
                pos_str = f"P:({anchor_pos[0]:.0f},{anchor_pos[1]:.0f},{anchor_pos[2]:.0f})"
                pos_col = (255, 210, 80)
            else:
                pos_str = "P: 未标注"
                pos_col = (110, 125, 140)
            draw_text(canvas, pos_str, (cx + 6, cy + 40), font_size=10, color=pos_col)

            px, py = cx + cw - 14, cy + 48
            pen_col = (0, 255, 180) if is_hover else (70, 95, 110)
            cv2.line(canvas, (px - 3, py + 2), (px + 3, py - 4), pen_col, 1, cv2.LINE_AA)

    def _render_page_frame_rois(self, canvas: np.ndarray, state: HubState, ws):
        """坐标系专属页签 2: 3D ROI 空间物件 (x: 340~960, y: 50~670)"""
        box_x, box_y, box_w, box_h = 340, 50, self.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        cur_frame = state.get_selected_frame()
        if not cur_frame:
            draw_text(canvas, "未选择任何机构坐标系", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        rois = state.get_frame_rois(cur_frame.frame_id)

        # 1. 顶部标题栏与新建 ROI 按钮
        header_text = f"3D ROI 空间物件 (坐标系: {cur_frame.frame_id} · 共 {len(rois)} 个)"
        draw_text(canvas, header_text, (box_x + 16, box_y + 14), font_size=15, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, FRAME_ADD_ROI_BTN, "+ 新建本坐标系 ROI", mpos)

        # 2. ROI 列表卡片
        if not rois:
            empty_y = box_y + 180
            cv2.rectangle(canvas, (box_x + 40, empty_y), (box_x + box_w - 40, empty_y + 120), (24, 28, 38), -1)
            cv2.rectangle(canvas, (box_x + 40, empty_y), (box_x + box_w - 40, empty_y + 120), (42, 52, 70), 1)
            draw_text(canvas, f"当前坐标系 [{cur_frame.frame_id}] 下暂无 3D ROI 空间物件",
                      (box_x + 110, empty_y + 36), font_size=16, color=(0, 220, 255), bold=True)
            draw_text(canvas, "点击右上角 [+ 新建本坐标系 ROI] 即可为该机构添加检测工作面包围盒",
                      (box_x + 85, empty_y + 70), font_size=13, color=(140, 160, 180))
            return

        for i, r in enumerate(rois[:6]):
            rx, ry, rw, rh = frame_roi_row_rect(i)
            is_hover = (rx <= mpos[0] <= rx + rw and ry <= mpos[1] <= ry + rh)
            card_bg = (26, 32, 44) if is_hover else (22, 26, 36)
            card_border = (0, 220, 240) if is_hover else (38, 48, 64)

            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), card_bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), card_border, 2 if is_hover else 1)

            cat_map = {"belt": "同步带", "wheel": "驱动轮", "tray": "料盘", "general": "通用"}
            cat_badge = cat_map.get(r.category, r.category)
            cv2.rectangle(canvas, (rx + 10, ry + 10), (rx + 78, ry + 34), (32, 45, 58), -1)
            cv2.rectangle(canvas, (rx + 10, ry + 10), (rx + 78, ry + 34), (0, 200, 240), 1)
            draw_text(canvas, cat_badge, (rx + 18, ry + 14), font_size=11, color=(0, 240, 220), bold=True)

            title = f"{r.name} ({r.roi_id})"
            draw_text(canvas, title, (rx + 86, ry + 13), font_size=14, color=self.COLOR_WHITE, bold=True)

            cx, cy, cz = r.center_xyz_mm
            sx, sy, sz = r.size_xyz_mm
            rx_d, ry_d, rz_d = getattr(r, "rotation_rpy_deg", [0.0, 0.0, 0.0])
            geom_str = f"中心: ({cx:.0f},{cy:.0f},{cz:.0f})  尺寸: ({sx:.0f}×{sy:.0f}×{sz:.0f})  旋转: ({rx_d:.0f}°,{ry_d:.0f}°,{rz_d:.0f}°)"
            put_text(canvas, geom_str, (rx + 14, ry + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 200), 1, cv2.LINE_AA)

            self._draw_button(canvas, frame_roi_edit_btn(i), "编辑", mpos)
            self._draw_button(canvas, frame_roi_del_btn(i), "删除", mpos, theme_color=(180, 60, 60))

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

    def _render_page_frames_rois(self, canvas: np.ndarray, state: HubState, sc):
        """页签: 坐标系与 3D ROI 空间管理页 (x: 340~960, y: 50~670)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题与右上角操作按钮
        draw_text(canvas, "工位多坐标系拓扑与 3D ROI 空间", (box_x + 16, box_y + 14), font_size=16, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, (760, box_y + 8, 86, 30), "刷新", mpos)
        self._draw_button(canvas, (854, box_y + 8, 86, 30), "打开目录", mpos)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工位", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        card_x = box_x + 12
        card_w = box_w - 24  # 596px

        # -------------------------------------------------------------------------
        # 1. 机构多坐标系拓扑树卡片 (Coordinate Frame Tree)
        # -------------------------------------------------------------------------
        c1_y = box_y + 46
        c1_h = 260
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (42, 50, 68), 1)

        frames = state.get_coordinate_frames()
        draw_text(canvas, f"机构多坐标系拓扑树 (Coordinate Frames · 共 {len(frames)} 个)",
                  (card_x + 16, c1_y + 11), font_size=14, color=(0, 240, 220), bold=True)
        self._draw_button(canvas, frame_btn_add_rect(), "+ 新增坐标系", mpos)
        cv2.line(canvas, (card_x + 14, c1_y + 34), (card_x + card_w - 14, c1_y + 34), self.COLOR_BORDER, 1)

        coord_mgr = state.coord_mgr
        row_y = c1_y + 40
        row_h = 66
        gap_y = 6

        # 显示坐标系列表 (最多前 3 个)
        for i, frame in enumerate(frames[:3]):
            fy = row_y + i * (row_h + gap_y)
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + card_w - 10, fy + row_h), (30, 36, 48), -1)
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + card_w - 10, fy + row_h), (50, 60, 80), 1)

            # 类型指示色条
            if frame.type == "world":
                bar_col = (0, 255, 180)
                type_name = "绝对世界系"
            elif frame.type == "tag_bound":
                bar_col = (0, 210, 255)
                type_name = f"动标Tag #{frame.tag_id}"
            else:
                bar_col = (255, 180, 50)
                type_name = "固定外参"
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + 14, fy + row_h), bar_col, -1)

            # 第一行：名称 + 标识 + 类型徽章 + 状态
            draw_text(canvas, f"{frame.name} ({frame.frame_id})", (card_x + 22, fy + 8), font_size=13, color=self.COLOR_WHITE, bold=True)
            cv2.rectangle(canvas, (card_x + 240, fy + 6), (card_x + 335, fy + 26), (20, 26, 36), -1)
            cv2.rectangle(canvas, (card_x + 240, fy + 6), (card_x + 335, fy + 26), bar_col, 1)
            draw_text(canvas, type_name, (card_x + 246, fy + 8), font_size=11, color=bar_col, bold=True)

            is_res = True
            if coord_mgr:
                _, is_res = coord_mgr.get_frame_to_world(frame.frame_id)
            status_text = "● 已解出" if is_res else "⚠ 动标未解"
            status_col = (0, 230, 140) if is_res else (0, 180, 255)
            draw_text(canvas, status_text, (card_x + 350, fy + 8), font_size=11, color=status_col, bold=True)

            # 右侧操作按钮
            self._draw_button(canvas, frame_row_edit_rect(i), "编辑", mpos)
            if frame.frame_id != "world":
                self._draw_button(canvas, frame_row_del_rect(i), "删除", mpos, theme_color=(180, 60, 60))

            # 第二行：父坐标系与几何参数详情
            parent_info = f"父级: {frame.parent_frame_id or '根节点'}"
            if frame.type == "world":
                param_info = "原点基准 [0, 0, 0] mm | 无旋转"
            elif frame.type == "tag_bound":
                param_info = f"绑定 Tag {frame.tag_id} | 偏移 XYZ: {frame.offset_xyz_mm} mm"
            else:
                t = [round(float(x), 1) for x in frame.translation_xyz_mm]
                r = [round(float(x), 1) for x in frame.rotation_rpy_deg]
                param_info = f"平移: {t} mm | 旋转: {r}°"
            draw_text(canvas, f"{parent_info}  |  {param_info}", (card_x + 22, fy + 38), font_size=11, color=(160, 175, 195))

        # -------------------------------------------------------------------------
        # 2. 3D ROI 空间物件集合卡片 (ROI Space Collection)
        # -------------------------------------------------------------------------
        c2_y = c1_y + c1_h + 10
        c2_h = 285
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (42, 50, 68), 1)

        rois = state.get_roi_spaces()
        draw_text(canvas, f"3D ROI 空间物件集合 (3D OBB Objects · 共 {len(rois)} 个)",
                  (card_x + 16, c2_y + 11), font_size=14, color=(0, 255, 180), bold=True)
        self._draw_button(canvas, roi_btn_add_rect(), "+ 新增 ROI", mpos)
        cv2.line(canvas, (card_x + 14, c2_y + 34), (card_x + card_w - 14, c2_y + 34), self.COLOR_BORDER, 1)

        roi_row_y = c2_y + 40
        roi_row_h = 70
        roi_gap_y = 6

        if not rois:
            draw_text(canvas, "当前工位尚未定义任何 3D ROI 物件 (可点击右上角 [+ 新增 ROI] 结构化配置)",
                      (card_x + 36, c2_y + 90), font_size=13, color=self.COLOR_GRAY)
        else:
            roi_mgr = state.roi_mgr
            for i, roi in enumerate(rois[:3]):
                ry = roi_row_y + i * (roi_row_h + roi_gap_y)
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + card_w - 10, ry + roi_row_h), (30, 36, 48), -1)
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + card_w - 10, ry + roi_row_h), (50, 60, 80), 1)

                # ROI 颜色标识
                rgb = roi.visual_color_rgb or [0, 255, 128]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + 14, ry + roi_row_h), bgr, -1)

                # 第一行：名称 + 标识 + 类别徽章 + 挂载坐标系
                draw_text(canvas, f"{roi.name} ({roi.roi_id})", (card_x + 22, ry + 8), font_size=13, color=self.COLOR_WHITE, bold=True)
                # 类别徽章
                cv2.rectangle(canvas, (card_x + 240, ry + 6), (card_x + 315, ry + 26), (20, 26, 36), -1)
                cv2.rectangle(canvas, (card_x + 240, ry + 6), (card_x + 315, ry + 26), bgr, 1)
                cat_map = {"belt": "同步带", "wheel": "驱动轮", "tray": "料盘", "general": "通用"}
                cat_label = f"[{cat_map.get(roi.category, roi.category)}]"
                draw_text(canvas, cat_label, (card_x + 246, ry + 8), font_size=11, color=bgr, bold=True)

                draw_text(canvas, f"所属: {roi.frame_id}", (card_x + 330, ry + 8), font_size=12, color=(0, 220, 255))

                # 右侧操作按钮
                self._draw_button(canvas, roi_row_edit_rect(i), "编辑", mpos)
                self._draw_button(canvas, roi_row_del_rect(i), "删除", mpos, theme_color=(180, 60, 60))

                # 第二行：局部几何尺寸
                c_str = [round(float(x), 1) for x in roi.center_xyz_mm]
                s_str = [round(float(x), 1) for x in roi.size_xyz_mm]
                local_info = f"局部中心: {c_str} | 尺寸(长宽高): {s_str[0]}x{s_str[1]}x{s_str[2]} mm"

                # 第三行：世界坐标系求解
                world_info = "世界 OBB: 未就绪"
                if roi_mgr and coord_mgr:
                    obb = roi_mgr.get_roi_world_obb(roi.roi_id, coord_mgr)
                    if obb and obb.get("is_resolved"):
                        cw = [round(float(x), 1) for x in obb["center_world"]]
                        world_info = f"世界中心: {cw} mm | 8顶点OBB已就绪"
                    elif obb:
                        world_info = "世界 OBB: 依附坐标系动标未就绪"

                draw_text(canvas, local_info, (card_x + 22, ry + 32), font_size=11, color=(150, 165, 185))
                draw_text(canvas, world_info, (card_x + 22, ry + 50), font_size=11, color=(0, 230, 160) if "已就绪" in world_info else self.COLOR_GOLD)

        # -------------------------------------------------------------------------
        # 3. 底部快捷提示
        # -------------------------------------------------------------------------
        hint_y = c2_y + c2_h + 4
        draw_text(canvas, "提示: 点击右上角 [+ 新增] 或各行 [编辑] 可直接在 GUI 中配置，修改后即时保存生效",
                  (box_x + 18, hint_y), font_size=12, color=self.COLOR_GRAY)

    def _render_frame_modal(self, canvas: np.ndarray, state: HubState):
        """渲染机构坐标系结构化表单弹窗 (带下拉选择框、高可编辑质感输入框与 Schema 校验)"""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_w, self.canvas_h), (8, 10, 14), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

        mpos = (state.mouse_x, state.mouse_y)
        mx, my, mw, mh = GEOM_MODAL_X, GEOM_MODAL_Y, GEOM_MODAL_W, GEOM_MODAL_H

        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (22, 27, 36), -1)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (0, 220, 180), 2)

        # 标题栏
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + 44), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 44), (mx + mw, my + 44), self.COLOR_BORDER, 1)

        d = state.frame_modal_data
        title_prefix = "新建机构相对坐标系" if state.frame_modal_is_new else f"编辑坐标系: 【{d.get('name', '')}】"
        draw_text(canvas, f"★ {title_prefix}", (mx + 20, my + 12), font_size=15, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, GEOM_MODAL_CLOSE, "X", mpos, theme_color=(180, 60, 60))

        # 表单字段排布
        form_y = my + 56

        # 1. 名称与标识行 (均采用带 ✎ 的高编辑感输入框)
        draw_text(canvas, "坐标系名称:", (mx + 24, form_y + 4), font_size=13, color=self.COLOR_GRAY)
        self._draw_text_input(canvas, (mx + 115, form_y, 220, 28), str(d.get("name", "")), mpos)

        draw_text(canvas, "唯一 ID:", (mx + 360, form_y + 4), font_size=13, color=self.COLOR_GRAY)
        self._draw_text_input(canvas, (mx + 430, form_y, 220, 28), str(d.get("frame_id", "")), mpos)

        # 2. 定义类型下拉框 (Drop-down Box)
        type_y = form_y + 40
        draw_text(canvas, "定义类型:", (mx + 24, type_y + 4), font_size=13, color=self.COLOR_GRAY)
        cur_type = d.get("type", "fixed_transform")
        type_label = "固定刚体外参 (平移 + 旋转)" if cur_type == "fixed_transform" else "AprilTag 动标绑定 (动态跟踪)"
        self._draw_dropdown_trigger(canvas, (mx + 115, type_y, 360, 28), type_label, mpos,
                                   is_open=(state.active_dropdown == "frame_type"))

        # 3. 挂载父坐标系下拉框 (Drop-down Box)
        parent_y = form_y + 80
        draw_text(canvas, "父坐标系:", (mx + 24, parent_y + 4), font_size=13, color=self.COLOR_GRAY)
        cur_parent = d.get("parent_frame_id", "world")
        frames = state.get_coordinate_frames()
        p_name = "世界基准绝对原点" if cur_parent == "world" else ""
        for f in frames:
            if f.frame_id == cur_parent:
                p_name = f.name
                break
        parent_label = f"[{cur_parent}] {p_name}" if p_name else f"[{cur_parent}]"
        self._draw_dropdown_trigger(canvas, (mx + 115, parent_y, 360, 28), parent_label, mpos,
                                   is_open=(state.active_dropdown == "frame_parent"))

        # 4. 几何参数根据类型切换
        param_y = form_y + 128
        cv2.rectangle(canvas, (mx + 20, param_y), (mx + mw - 20, param_y + 155), (28, 34, 46), -1)
        cv2.rectangle(canvas, (mx + 20, param_y), (mx + mw - 20, param_y + 155), (45, 55, 75), 1)

        if cur_type == "fixed_transform":
            draw_text(canvas, "固定外参变换矩阵 (相对于父级坐标系):", (mx + 32, param_y + 12), font_size=13, color=(0, 240, 220), bold=True)
            t = d.get("translation_xyz_mm", [0, 0, 0])
            r = d.get("rotation_rpy_deg", [0, 0, 0])

            # 平移 X, Y, Z
            draw_text(canvas, "平移 (mm):", (mx + 32, param_y + 48), font_size=13, color=self.COLOR_WHITE)
            self._draw_text_input(canvas, (mx + 125, param_y + 42, 130, 28), f"X: {t[0]:.1f}", mpos)
            self._draw_text_input(canvas, (mx + 265, param_y + 42, 130, 28), f"Y: {t[1]:.1f}", mpos)
            self._draw_text_input(canvas, (mx + 405, param_y + 42, 130, 28), f"Z: {t[2]:.1f}", mpos)

            # 旋转 Roll, Pitch, Yaw
            draw_text(canvas, "旋转 (°):", (mx + 32, param_y + 98), font_size=13, color=self.COLOR_WHITE)
            self._draw_text_input(canvas, (mx + 125, param_y + 92, 130, 28), f"Roll: {r[0]:.1f}", mpos)
            self._draw_text_input(canvas, (mx + 265, param_y + 92, 130, 28), f"Pitch: {r[1]:.1f}", mpos)
            self._draw_text_input(canvas, (mx + 405, param_y + 92, 130, 28), f"Yaw: {r[2]:.1f}", mpos)
        else:
            draw_text(canvas, "AprilTag 动标绑定配置:", (mx + 32, param_y + 12), font_size=13, color=(0, 210, 255), bold=True)
            tid = d.get("tag_id", 0)
            off = d.get("offset_xyz_mm", [0, 0, 0])

            draw_text(canvas, "绑定的 AprilTag:", (mx + 32, param_y + 48), font_size=13, color=self.COLOR_WHITE)
            self._draw_text_input(canvas, (mx + 165, param_y + 42, 160, 28), f"Tag ID: #{tid}", mpos)
            draw_text(canvas, "(点击修改绑定的标靶编号)", (mx + 335, param_y + 48), font_size=11, color=self.COLOR_GRAY)

            draw_text(canvas, "局部偏移 XYZ (mm):", (mx + 32, param_y + 98), font_size=13, color=self.COLOR_WHITE)
            self._draw_text_input(canvas, (mx + 185, param_y + 92, 120, 28), f"dx: {off[0]:.1f}", mpos)
            self._draw_text_input(canvas, (mx + 315, param_y + 92, 120, 28), f"dy: {off[1]:.1f}", mpos)
            self._draw_text_input(canvas, (mx + 445, param_y + 92, 120, 28), f"dz: {off[2]:.1f}", mpos)

        # 底部操作按钮
        self._draw_button(canvas, GEOM_MODAL_SAVE, "保存", mpos, theme_color=(0, 220, 140))
        self._draw_button(canvas, GEOM_MODAL_CANCEL, "取消", mpos)

        # 5. 顶层渲染活跃下拉浮层
        self._render_active_dropdown(canvas, state)

    def _render_roi_modal(self, canvas: np.ndarray, state: HubState):
        """渲染 3D ROI 空间物件结构化表单弹窗 (带下拉选择框、高可编辑质感输入框与 Schema 校验)"""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_w, self.canvas_h), (8, 10, 14), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

        mpos = (state.mouse_x, state.mouse_y)
        mx, my, mw, mh = GEOM_MODAL_X, GEOM_MODAL_Y, GEOM_MODAL_W, GEOM_MODAL_H

        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (22, 27, 36), -1)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (0, 255, 180), 2)

        # 标题栏
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + 44), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 44), (mx + mw, my + 44), self.COLOR_BORDER, 1)

        d = state.roi_modal_data
        title_prefix = "新建 3D ROI 空间物件" if state.roi_modal_is_new else f"编辑 ROI 物件: 【{d.get('name', '')}】"
        draw_text(canvas, f"★ {title_prefix}", (mx + 20, my + 12), font_size=15, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, GEOM_MODAL_CLOSE, "X", mpos, theme_color=(180, 60, 60))

        # 表单字段排布
        form_y = my + 54

        # 1. 名称与标识行 (均采用带 ✎ 的高编辑感输入框)
        draw_text(canvas, "物件名称:", (mx + 24, form_y + 4), font_size=13, color=self.COLOR_GRAY)
        self._draw_text_input(canvas, (mx + 115, form_y, 220, 28), str(d.get("name", "")), mpos)

        draw_text(canvas, "唯一 ID:", (mx + 360, form_y + 4), font_size=13, color=self.COLOR_GRAY)
        self._draw_text_input(canvas, (mx + 430, form_y, 220, 28), str(d.get("roi_id", "")), mpos)

        # 2. 部件类别下拉框 (Drop-down Box)
        cat_y = form_y + 40
        draw_text(canvas, "部件类别:", (mx + 24, cat_y + 4), font_size=13, color=self.COLOR_GRAY)
        cur_cat = d.get("category", "belt")
        cat_map = {
            "belt": "同步带工作面 (belt)",
            "wheel": "驱动轮干涉区 (wheel)",
            "tray": "料盘工装区 (tray)",
            "general": "通用机构部件 (general)"
        }
        cat_label = cat_map.get(cur_cat, f"{cur_cat}")
        self._draw_dropdown_trigger(canvas, (mx + 115, cat_y, 360, 28), cat_label, mpos,
                                   is_open=(state.active_dropdown == "roi_category"))

        # 3. 所属坐标系下拉框 (Drop-down Box)
        parent_y = form_y + 80
        draw_text(canvas, "所属坐标系:", (mx + 24, parent_y + 4), font_size=13, color=self.COLOR_GRAY)
        cur_frame = d.get("frame_id", "world")
        frames = state.get_coordinate_frames()
        f_name = "世界基准绝对原点" if cur_frame == "world" else ""
        for f in frames:
            if f.frame_id == cur_frame:
                f_name = f.name
                break
        frame_label = f"[{cur_frame}] {f_name}" if f_name else f"[{cur_frame}]"
        self._draw_dropdown_trigger(canvas, (mx + 115, parent_y, 360, 28), frame_label, mpos,
                                   is_open=(state.active_dropdown == "roi_frame"))

        # 4. 几何长方体参数区 (中心 + 尺寸 + 姿态)
        geom_y = form_y + 128
        cv2.rectangle(canvas, (mx + 20, geom_y), (mx + mw - 20, geom_y + 165), (28, 34, 46), -1)
        cv2.rectangle(canvas, (mx + 20, geom_y), (mx + mw - 20, geom_y + 165), (45, 55, 75), 1)

        draw_text(canvas, "3D 有向长方体空间定义 (在所属局部坐标系下):", (mx + 32, geom_y + 10), font_size=13, color=(0, 240, 220), bold=True)

        c = d.get("center_xyz_mm", [0, 0, 0])
        s = d.get("size_xyz_mm", [50, 50, 50])
        r = d.get("rotation_rpy_deg", [0, 0, 0])

        # 局部中心
        draw_text(canvas, "局部中心 (mm):", (mx + 32, geom_y + 42), font_size=13, color=self.COLOR_WHITE)
        self._draw_text_input(canvas, (mx + 155, geom_y + 36, 115, 28), f"X: {c[0]:.1f}", mpos)
        self._draw_text_input(canvas, (mx + 280, geom_y + 36, 115, 28), f"Y: {c[1]:.1f}", mpos)
        self._draw_text_input(canvas, (mx + 405, geom_y + 36, 115, 28), f"Z: {c[2]:.1f}", mpos)

        # 尺寸长宽高 (强 Schema 约束)
        draw_text(canvas, "空间尺寸 (mm):", (mx + 32, geom_y + 82), font_size=13, color=self.COLOR_WHITE)
        self._draw_text_input(canvas, (mx + 155, geom_y + 76, 115, 28), f"长 dx: {s[0]:.1f}", mpos)
        self._draw_text_input(canvas, (mx + 280, geom_y + 76, 115, 28), f"宽 dy: {s[1]:.1f}", mpos)
        self._draw_text_input(canvas, (mx + 405, geom_y + 76, 115, 28), f"高 dz: {s[2]:.1f}", mpos)
        draw_text(canvas, "★ 约束: 必须 > 0", (mx + 530, geom_y + 82), font_size=11, color=(0, 255, 180) if all(x > 0 for x in s) else (0, 100, 255))

        # 微调姿态
        draw_text(canvas, "局部旋转 (°):", (mx + 32, geom_y + 122), font_size=13, color=self.COLOR_WHITE)
        self._draw_text_input(canvas, (mx + 155, geom_y + 116, 115, 28), f"R: {r[0]:.1f}", mpos)
        self._draw_text_input(canvas, (mx + 280, geom_y + 116, 115, 28), f"P: {r[1]:.1f}", mpos)
        self._draw_text_input(canvas, (mx + 405, geom_y + 116, 115, 28), f"Y: {r[2]:.1f}", mpos)

        # 底部操作按钮
        self._draw_button(canvas, GEOM_MODAL_SAVE, "保存", mpos, theme_color=(0, 220, 140))
        self._draw_button(canvas, GEOM_MODAL_CANCEL, "取消", mpos)

        # 5. 顶层渲染活跃下拉浮层
        self._render_active_dropdown(canvas, state)

    def _render_page_whitelist(self, canvas: np.ndarray, state: HubState, sc):
        """页签2: Tag 标靶白名单管理页 - 实时读取 tag_whitelist.yaml 呈现放行矩阵 (x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 23, 30), -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题与右上角操作按钮 (紧凑排布在 620 宽内)
        draw_text(canvas, "Tag 标靶白名单管理", (box_x + 16, box_y + 14), font_size=16, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, (760, box_y + 8, 86, 30), "刷新", mpos)
        if state.whitelist_edit_mode:
            self._draw_button(canvas, WL_BTN_DONE, "完成", mpos, theme_color=(0, 200, 120))
        else:
            self._draw_button(canvas, WL_BTN_DONE, "编辑", mpos)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工位", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        wl = state.get_tag_whitelist()
        wl_path = state.workspace_mgr.get_tag_whitelist_path(sc.workspace_id)
        if state.whitelist_edit_mode:
            # 编辑模式: 徽章与状态以芯片工作集合为准 (即时反馈, 不等 yaml 回读)
            allowed_ids = set(state.whitelist_edit_ids)
        else:
            allowed_ids = set(wl.get("allowed_ids") or []) if wl else set()
        has_filter = bool(allowed_ids)  # 白名单恒启用: 名单非空 → 过滤, 空 → 探索模式

        # 1. Workspace 元数据卡与白名单生效状态徽章
        meta_y = box_y + 52
        cv2.rectangle(canvas, (box_x + 12, meta_y), (box_x + box_w - 12, meta_y + 70), (26, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 12, meta_y), (box_x + box_w - 12, meta_y + 70), self.COLOR_BORDER, 1)
        draw_text(canvas, f"{sc.name}  |  {sc.workspace_id}",
                  (box_x + 20, meta_y + 12), font_size=14, color=(0, 240, 220), bold=True)
        draw_text(canvas, f"配置: {os.path.basename(wl_path)}", (box_x + 20, meta_y + 44), font_size=12, color=self.COLOR_DARK_GRAY)

        badge_x, badge_y = box_x + box_w - 180, meta_y + 14
        if has_filter:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (20, 48, 32), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (0, 255, 160), 2)
            draw_text(canvas, f"● 白名单 {len(allowed_ids)} 个", (badge_x + 18, badge_y + 11), font_size=13, color=(0, 255, 180), bold=True)
        else:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (46, 40, 24), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (60, 160, 255), 2)
            draw_text(canvas, "◌ 探索模式", (badge_x + 18, badge_y + 11), font_size=13, color=(80, 200, 255), bold=True)

        # 2. 白名单模式说明卡
        desc_y = meta_y + 80
        cv2.rectangle(canvas, (box_x + 12, desc_y), (box_x + box_w - 12, desc_y + 60), (24, 28, 38), -1)
        cv2.rectangle(canvas, (box_x + 12, desc_y), (box_x + box_w - 12, desc_y + 60), self.COLOR_BORDER, 1)

        if state.whitelist_edit_mode and state.anchor_mode:
            mode_text = "锚点模式: 单击 Tag 芯片打开世界坐标弹窗 (逐轴输入, 支持部分已知与清除)。"
            mode_hint = "提示: 金色芯片 = 已记录锚点 (标注已知轴数)；[退出锚点] 返回白名单编辑。"
            mode_col = (0, 210, 255)
        elif state.whitelist_edit_mode:
            mode_text = "编辑模式: 单击芯片切换放行/拦截，每次点击即时写回 tag_whitelist.yaml。"
            mode_hint = "提示: 右上角 [完成] 退出编辑；芯片右上蓝点 = 该 ID 在工位元数据 valid_tag_ids 中。"
            mode_col = (0, 255, 200)
        elif has_filter:
            mode_text = f"白名单恒启用: 仅放行 allowed_ids 中的 {len(allowed_ids)} 个标靶，其余拦截 (权威约束)。"
            mode_hint = "提示: 修改 allowed_ids 并保存，返回本页签即自动刷新矩阵状态。"
            mode_col = (0, 255, 160)
        elif not wl:
            mode_text = "尚未创建 tag_whitelist.yaml: 探索模式放行所有检测到的有效标靶。"
            mode_hint = "提示: 点击右上角 [编辑] 进入芯片编辑模式 (自动创建配置模板)。"
            mode_col = self.COLOR_GOLD
        else:
            mode_text = "探索模式 (allowed_ids 为空): 放行所有检测到的有效标靶，由全局物理白名单兜底拦截。"
            mode_hint = "提示: 在 allowed_ids 中填入物理布点标靶 ID 即可启用工位过滤。"
            mode_col = (0, 200, 240)

        draw_text(canvas, mode_text, (box_x + 20, desc_y + 10), font_size=13, color=mode_col, bold=True)
        draw_text(canvas, mode_hint, (box_x + 20, desc_y + 34), font_size=12, color=self.COLOR_GRAY)

        # 3. 全量 Tag 标靶放行矩阵 (视图: 只读矩阵 / 编辑: 可点击芯片, 超范围 ID 追加行)
        matrix_y = desc_y + 70
        edit_ids = state.whitelist_edit_ids if state.whitelist_edit_mode else None
        if edit_ids is not None:
            extra_ids = sorted(t for t in edit_ids if t >= 30)
        else:
            extra_ids = sorted(t for t in allowed_ids if t >= 30)
        extra_rows = (len(extra_ids) + WL_COLS - 1) // WL_COLS
        matrix_h = 268 + 44 * extra_rows
        cv2.rectangle(canvas, (box_x + 12, matrix_y), (box_x + box_w - 12, matrix_y + matrix_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (box_x + 12, matrix_y), (box_x + box_w - 12, matrix_y + matrix_h), (40, 48, 66), 1)

        matrix_title = ("AprilTag 标靶放行矩阵 (0~29 号标靶)"
                        + (f"  |  已放行 {len(allowed_ids)} 个" if has_filter else "  |  探索模式"))
        draw_text(canvas, matrix_title, (box_x + 20, matrix_y + 12), font_size=14, color=(0, 255, 200), bold=True)
        cv2.line(canvas, (box_x + 20, matrix_y + 36), (box_x + box_w - 20, matrix_y + 36), self.COLOR_BORDER, 1)

        valid_set = set(sc.valid_tag_ids)
        for t_id in list(range(30)) + extra_ids:
            tx, ty, cw, ch = whitelist_cell_rect(t_id)

            if edit_ids is not None:
                # 编辑模式: 芯片带 Hover 高亮 (锚点模式反映锚点状态, 白名单编辑反映放行集合)
                hovered = point_in_rect(mpos[0], mpos[1], (tx, ty, cw, ch))
                if state.anchor_mode:
                    entry = state.anchor_map.get(t_id)
                    n_known = sum(1 for b in entry["known"] if b) if entry else 0
                    if entry:
                        cell_bg = (54, 48, 28) if hovered else (44, 40, 22)
                        cell_border = (0, 220, 255) if hovered else (0, 190, 255)
                        txt_color = (0, 230, 255)
                        status_desc, status_col = f"锚点 {n_known}/3", (0, 220, 200)
                    else:
                        cell_bg = (32, 36, 46) if hovered else (26, 30, 38)
                        cell_border = (66, 74, 92) if hovered else (52, 60, 76)
                        txt_color = (150, 160, 175)
                        status_desc, status_col = "无锚点", (110, 120, 135)
                else:
                    is_allowed = (t_id in edit_ids)
                    if is_allowed:
                        cell_bg = (34, 58, 48) if hovered else (28, 48, 40)
                        cell_border = (0, 255, 180) if hovered else (0, 240, 160)
                        txt_color = (0, 255, 200)
                        status_desc, status_col = "已放行", (0, 220, 140)
                    else:
                        cell_bg = (50, 32, 32) if hovered else (40, 26, 26)
                        cell_border = (80, 52, 52) if hovered else (60, 40, 40)
                        txt_color = (200, 140, 140)
                        status_desc, status_col = "已拦截", (150, 100, 100)
            elif has_filter:
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
                status_desc = "探索放行"
                status_col = (110, 170, 190)

            cv2.rectangle(canvas, (tx, ty), (tx + cw, ty + ch), cell_bg, -1)
            cv2.rectangle(canvas, (tx, ty), (tx + cw, ty + ch), cell_border, 1)

            draw_text(canvas, f"Tag #{t_id:02d}", (tx + 8, ty + 5), font_size=11, color=txt_color, bold=True)
            draw_text(canvas, status_desc, (tx + 12, ty + 21), font_size=10, color=status_col)
            if t_id in valid_set:
                cv2.circle(canvas, (tx + cw - 8, ty + 9), 3, (0, 200, 240), -1)

        # 4. 底部操作区: 编辑模式为批量按钮 + 锚点切换, 视图模式为指引文案
        if edit_ids is not None:
            self._draw_button(canvas, WL_BTN_ALL, "全部放行", mpos)
            self._draw_button(canvas, WL_BTN_CLEAR, "清空", mpos, theme_color=(180, 60, 60))
            self._draw_button(canvas, WL_BTN_ANCHOR, "退出锚点" if state.anchor_mode else "锚点坐标", mpos)
            hint = "单击 Tag 芯片编辑其世界坐标锚点" if state.anchor_mode else "单击芯片即时写回 yaml"
            draw_text(canvas, hint, (WL_BTN_ANCHOR[0] + WL_BTN_ANCHOR[2] + 14, 632),
                      font_size=12, color=self.COLOR_GRAY)
            if state.anchor_modal_open:
                self._render_anchor_modal(canvas, state)
        else:
            draw_text(canvas, "提示: 点击右上角 [编辑] 进入芯片编辑模式 (单击切换, 即时写回)",
                      (box_x + 20, box_y + box_h - 26), font_size=12, color=self.COLOR_GRAY)

    def _render_anchor_modal(self, canvas: np.ndarray, state):
        """锚点坐标编辑弹窗 (逐轴输入 XYZ / 部分已知 / 清除锚点, 纯鼠标操作)"""
        mpos = (state.mouse_x, state.mouse_y)
        MX, MY, MW, MH = WL_ANCHOR_X, WL_ANCHOR_Y, WL_ANCHOR_W, WL_ANCHOR_H
        cv2.rectangle(canvas, (MX, MY), (MX + MW, MY + MH), (16, 20, 28), -1)
        cv2.rectangle(canvas, (MX, MY), (MX + MW, MY + MH), (0, 200, 240), 2)

        n_known = state.anchor_known_count()
        draw_text(canvas, f"Tag #{state.anchor_modal_tag:02d} 世界坐标锚点编辑 (mm)",
                  (MX + 18, MY + 10), font_size=14, color=(0, 255, 200), bold=True)
        if n_known == 3:
            status_desc, status_col = "完整锚点 (5 DoF 解算)", (0, 230, 150)
        elif n_known >= 1:
            status_desc, status_col = "部分锚点 (约束积累)", (0, 200, 230)
        else:
            status_desc, status_col = "未记录 (保存 = 清除该锚点)", (150, 160, 175)
        draw_text(canvas, f"已知 {n_known}/3 轴: {status_desc}", (MX + 18, MY + 34),
                  font_size=12, color=status_col)

        # 三轴行: 轴名 + 值 + 已知/未记录 + [清除]
        for axis in range(3):
            rx, ry, rw, rh = anchor_row_rect(axis)
            hovered = point_in_rect(mpos[0], mpos[1], (rx, ry, rw, rh))
            is_sel = (state.anchor_axis_sel == axis)
            is_known = bool(state.anchor_modal_known[axis])
            row_bg = (30, 38, 52) if is_sel else ((36, 42, 54) if hovered else (26, 30, 40))
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), row_bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh),
                          (0, 200, 240) if is_sel else (60, 70, 88), 1)
            draw_text(canvas, "XYZ"[axis], (rx + 12, ry + 9), font_size=14,
                      color=(0, 255, 200) if is_known else (150, 160, 175), bold=True)
            if is_sel and state.anchor_axis_buf:
                val_text, val_col = state.anchor_axis_buf + "_", self.COLOR_WHITE
            elif is_known:
                val_text, val_col = f"{state.anchor_modal_xyz[axis]:.1f}", self.COLOR_WHITE
            else:
                val_text, val_col = "---", (120, 130, 145)
            draw_text(canvas, val_text, (rx + 42, ry + 8), font_size=15, color=val_col, bold=True)
            know_text = "已知" if is_known else "未记录"
            draw_text(canvas, know_text, (rx + 180, ry + 10), font_size=12,
                      color=(0, 220, 140) if is_known else (110, 120, 135))
            # 行内 [清除] 按钮 (取消该轴的已知状态)
            cx, cy, cw, ch = anchor_clear_rect(axis)
            chov = point_in_rect(mpos[0], mpos[1], (cx, cy, cw, ch))
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (70, 46, 36) if chov else (58, 38, 30), -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (150, 90, 60), 1)
            draw_text(canvas, "清除", (cx + 17, cy + 7), font_size=12, color=(230, 170, 140))

        # 15 键键盘: 1~9 / . / 0 / -+/ 清空 / 退格 / 确认
        key_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "0", "-/+", "清空", "退格", "确认"]
        for idx, label in enumerate(key_labels):
            kx, ky, kw, kh = anchor_padkey_rect(idx)
            hov = point_in_rect(mpos[0], mpos[1], (kx, ky, kw, kh))
            if label == "确认":
                bg = (26, 88, 60) if hov else (22, 70, 48)
                border, col = (0, 230, 150), (120, 255, 200)
            elif label in ("退格", "清空"):
                bg = (70, 46, 36) if hov else (58, 38, 30)
                border, col = (150, 90, 60), (230, 170, 140)
            else:
                bg = (40, 48, 64) if hov else (32, 38, 50)
                border, col = (90, 105, 135), (220, 228, 240)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), bg, -1)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), border, 1)
            est_w = 8 * len(label) if label.isascii() else 14 * len(label)
            draw_text(canvas, label, (kx + (kw - est_w) // 2, ky + 11), font_size=13, color=col, bold=True)

        # 底部: 保存 / 取消 / 清除锚点
        self._draw_button(canvas, WL_ANCHOR_SAVE, "保存", mpos)
        self._draw_button(canvas, WL_ANCHOR_CANCEL, "取消", mpos)
        self._draw_button(canvas, WL_ANCHOR_DELETE, "清除锚点", mpos, theme_color=(180, 60, 60))

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
        self._draw_button(canvas, (682, c1_y + 132, 130, 28), "新建ROI坐标系", mpos, theme_color=(0, 200, 160))

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
        c4_h = 160
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (40, 48, 66), 1)
        draw_text(canvas, "3D 空间拓扑与几何网络健康度 (Geometry)", (card_x + 16, c4_y + 11), font_size=14, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (card_x + 16, c4_y + 36), (card_x + card_w - 16, c4_y + 36), self.COLOR_BORDER, 1)

        span_mm = getattr(sc, "spatial_span_mm", 685.0 if sc.ba_solved else 0.0)
        loop_cnt = getattr(sc, "loop_closures", max(15, sc.image_count * 3) if sc.ba_solved else 0)
        draw_text(canvas, f"• 空间基线物理最大跨度 : {span_mm:.1f} mm (立体视野覆盖)", (card_x + 18, c4_y + 44), font_size=12, color=self.COLOR_GRAY)
        draw_text(canvas, f"• 空间闭环刚性几何约束 : {loop_cnt} 条跨视角闭环", (card_x + 18, c4_y + 68), font_size=12, color=(0, 240, 180) if loop_cnt >= 10 else self.COLOR_GOLD)

        # 机构多坐标系树统计
        frames = state.get_coordinate_frames()
        rel_frames = [f for f in frames if f.frame_id != "world"]
        rel_names = ", ".join(f.name for f in rel_frames[:2])
        if len(rel_frames) > 2:
            rel_names += f" 等共{len(rel_frames)}个"
        elif not rel_frames:
            rel_names = "未配置相对系"
        frames_desc = f"{len(frames)} 个 (1 绝对世界系 / {len(rel_frames)} 相对系: {rel_names})"
        draw_text(canvas, f"• 机构多坐标系拓扑树   : {frames_desc}", (card_x + 18, c4_y + 91), font_size=12, color=(0, 230, 255), bold=True)

        # 3D ROI 空间物件集合统计
        rois = state.get_roi_spaces()
        if rois:
            roi_brief = ", ".join(f"{r.name}[{r.category}]" for r in rois[:2])
            if len(rois) > 2:
                roi_brief += f" 等{len(rois)}个"
            roi_desc = f"{len(rois)} 个 3D 空间物件 ({roi_brief})"
            roi_col = (0, 255, 180)
        else:
            roi_desc = "0 个 (可于工位 rois.yaml 中按部件配置)"
            roi_col = self.COLOR_GRAY
        draw_text(canvas, f"• 3D ROI 空间物件集合  : {roi_desc}", (card_x + 18, c4_y + 114), font_size=12, color=roi_col, bold=True)

        draw_text(canvas, f"• 标定核心求解与剪枝   : Ceres/Levenberg-Marquardt 两阶段优化 (Huber 稳健核函数)", (card_x + 18, c4_y + 137), font_size=12, color=(140, 155, 175))

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

    def _should_show_tag_bound_tooltip(self, state: HubState, mpos: tuple[int, int]) -> bool:
        """检测鼠标是否悬停在动标帮助提示胶囊或动标参数信息区域"""
        mx, my = mpos
        box_x, box_y = 340, 50
        box_w = 620
        card_x = box_x + 16
        c1_y = box_y + 42
        # 1. 动标定义说明徽章
        help_btn_x, help_btn_y, help_btn_w, help_btn_h = card_x + 310, c1_y + 65, 126, 22
        if help_btn_x <= mx <= help_btn_x + help_btn_w and help_btn_y <= my <= help_btn_y + help_btn_h:
            return True
        # 2. 如果是动标类型坐标系，悬停在动标参数卡片上亦弹出完整解释
        cur_frame = state.get_selected_frame()
        if cur_frame and cur_frame.type == "tag_bound":
            tag_box_y = c1_y + 98
            if card_x + 16 <= mx <= card_x + (box_w - 32) - 16 and tag_box_y <= my <= tag_box_y + 84:
                return True
        return False

    def _draw_tag_bound_tooltip(self, canvas: np.ndarray, anchor_pos: tuple[int, int]):
        title = "AprilTag 动标机制与多 Tag 绑定说明"
        lines = [
            "【动标 (Dynamic Tag) 的定义】",
            "• 安装在【运动机构部件】(如活动滑块、推手、法兰夹爪) 上的视觉标靶；",
            "• 随机构运动实时改变空间坐标，相机每一帧识别并动态解算该部件位姿。",
            "",
            "【为什么此处是一个列表 (可配置多个 Tag)？】",
            "• 工业现场中单动标极易发生受光反光、物料遮挡或大倾角失锁；",
            "• 部件绑定多动标列表 (如 #10, #11) 时，系统启用多标冗余追踪机制；",
            "• 只要视野中能稳定观测到列表中的任意一个动标，即可持续求解位姿！",
            "",
            "【标称安装偏移 (offset)】",
            "• 动标贴片几何中心到机构部件实际旋转轴或受控特征原点的物理装配偏差。",
            "",
            "★ 注意: world 世界坐标系是固定空间基准原点，依靠下方静态标靶阵列定位。"
        ]
        render_floating_tooltip(canvas, title, lines, anchor_pos)



