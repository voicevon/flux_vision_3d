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

        # 分割线
        div_y1 = 380
        cv2.line(canvas, (10, div_y1), (330, div_y1), self.COLOR_BORDER, 1)

        # ==== 2. 场景管理常用操作按钮 ====
        draw_text(canvas, "场景管理快捷操作", (16, div_y1 + 8), font_size=15, color=self.COLOR_CYAN, bold=True)

        # 按钮 1: 新建场景 [N]
        btn1_y = div_y1 + 32
        cv2.rectangle(canvas, (10, btn1_y), (165, btn1_y + 36), (22, 32, 44), -1)
        cv2.rectangle(canvas, (10, btn1_y), (165, btn1_y + 36), (0, 200, 240), 1)
        draw_text(canvas, "[+] 新建场景 [N]", (20, btn1_y + 9), font_size=14, color=(0, 240, 220), bold=True)

        # 按钮 2: 修改名称 [R]
        cv2.rectangle(canvas, (175, btn1_y), (330, btn1_y + 36), (30, 30, 42), -1)
        cv2.rectangle(canvas, (175, btn1_y), (330, btn1_y + 36), (80, 180, 255), 1)
        draw_text(canvas, "[R] 修改名称", (200, btn1_y + 9), font_size=14, color=(80, 200, 255), bold=True)

        # 按钮 3: 克隆场景副本 [K]
        btn2_y = btn1_y + 40
        cv2.rectangle(canvas, (10, btn2_y), (165, btn2_y + 34), (26, 26, 38), -1)
        cv2.rectangle(canvas, (10, btn2_y), (165, btn2_y + 34), (160, 120, 240), 1)
        draw_text(canvas, "[K] 克隆场景", (30, btn2_y + 8), font_size=14, color=(180, 140, 255))

        # 按钮 4: 打开目录 [V]
        cv2.rectangle(canvas, (175, btn2_y), (330, btn2_y + 34), (24, 28, 36), -1)
        cv2.rectangle(canvas, (175, btn2_y), (330, btn2_y + 34), self.COLOR_BORDER, 1)
        draw_text(canvas, "[V] 打开目录", (200, btn2_y + 8), font_size=14, color=self.COLOR_GRAY)

        # 分割线
        div_y2 = btn2_y + 42
        cv2.line(canvas, (10, div_y2), (330, div_y2), self.COLOR_BORDER, 1)

        # ==== 3. 核心快捷工作流通道 (与列表平行展示在左侧下方) ====
        draw_text(canvas, "核心快捷工作流通道", (16, div_y2 + 8), font_size=15, color=self.COLOR_WHITE, bold=True)

        wf_items = [
            ("[C] 原地切入相机连拍向导", "空格抓拍，直接采集到当前场景", (0, 255, 180)),
            ("[S] 启动离线 Studio 深度平差", "智能剪枝/重投影质检/立体几何", (0, 220, 255)),
            ("[P] 生效到生产系统 (发布运行地图)", "覆盖全局 config/tags_map.yaml", self.COLOR_GOLD),
        ]

        wfy = div_y2 + 32
        for title, desc, col in wf_items:
            cv2.rectangle(canvas, (10, wfy), (330, wfy + 44), (22, 26, 36), -1)
            cv2.rectangle(canvas, (10, wfy), (330, wfy + 44), (38, 46, 62), 1)
            draw_text(canvas, title, (18, wfy + 4), font_size=13, color=col, bold=True)
            draw_text(canvas, desc, (18, wfy + 24), font_size=12, color=self.COLOR_GRAY)
            wfy += 48

    def _render_center_report_panel(self, canvas: np.ndarray, state: HubState, sc):
        """渲染中间栏：场景综合体检报告与几何健康看板 (x: 340~800, y: 50~670)"""
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
        cv2.rectangle(canvas, (box_x + 16, meta_y), (box_x + box_w - 16, meta_y + 56), (26, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 16, meta_y), (box_x + box_w - 16, meta_y + 56), (0, 180, 220), 1)
        draw_text(canvas, f"当前场景: 【{sc.name}】", (box_x + 26, meta_y + 8), font_size=16, color=(0, 240, 220), bold=True)
        draw_text(canvas, f"物理唯一ID: {sc.scene_id}", (box_x + 26, meta_y + 32), font_size=13, color=self.COLOR_GRAY)

        # 4 大体检与健康指标卡片
        cards = [
            ("1. 采样数据集规模与有效性",
             f"总采集: {sc.image_count} 帧  |  有效参与: {sc.active_image_count} 帧",
             "状态评级: 样本充足 (≥10帧达标)" if sc.image_count >= 10 else "状态评级: 样本偏少 (建议按 [C] 继续采图)",
             (0, 240, 100) if sc.image_count >= 10 else (0, 180, 255)),

            ("2. 空间标靶拓扑与参考基准",
             f"基准原点: Tag #{sc.origin_tag_id}  |  X轴对准: Tag #{sc.x_axis_tag_id}",
             f"已知标靶拓扑已锚定，空间几何关系锁定",
             (0, 220, 255)),

            ("3. 两阶段 BA 平差与重投影精度",
             f"全局 RMSE 误差: {sc.global_rmse_px:.3f} px" if sc.ba_solved else "尚未执行离线平差 (暂无精度数据)",
             "精度评级: 极优 (误差 < 0.20px)" if (sc.ba_solved and sc.global_rmse_px < 0.2) else
             ("精度评级: 良好" if sc.ba_solved else "待平差: 请按 [S] 启动 Studio 计算"),
             (0, 255, 160) if sc.ba_solved else self.COLOR_GRAY),

            ("4. 生产系统生效与运行状态",
             "★ 已生效为全局生产运行地图 (生效中)" if sc.is_published else "草稿沙盒状态 (尚未生效至生产配置)",
             "全局生产路径: config/tags_map.yaml",
             self.COLOR_GOLD if sc.is_published else (140, 150, 165))
        ]

        cy = meta_y + 68
        for title, val_line, sub_line, col in cards:
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 60), (26, 30, 40), -1)
            cv2.rectangle(canvas, (box_x + 16, cy), (box_x + box_w - 16, cy + 60), (38, 44, 58), 1)
            draw_text(canvas, title, (box_x + 28, cy + 6), font_size=14, color=self.COLOR_WHITE, bold=True)
            draw_text(canvas, val_line, (box_x + 28, cy + 25), font_size=13, color=col, bold=True)
            draw_text(canvas, sub_line, (box_x + 28, cy + 42), font_size=12, color=self.COLOR_GRAY)
            cy += 68

        # 重点业务解释板块：解说“什么是生效为生产运行地图”
        exp_y = cy + 4
        cv2.rectangle(canvas, (box_x + 16, exp_y), (box_x + box_w - 16, box_y + box_h - 14), (20, 26, 36), -1)
        cv2.rectangle(canvas, (box_x + 16, exp_y), (box_x + box_w - 16, box_y + box_h - 14), (0, 160, 200), 1)

        draw_text(canvas, "★ 什么是【生效到生产系统 (发布运行地图)】？", (box_x + 26, exp_y + 10), font_size=14, color=self.COLOR_GOLD, bold=True)
        explain_lines = [
            "• 在线定位与生产运行模块，仅读取全局配置 config/tags_map.yaml。",
            "• 多场景管理允许您在独立工况沙盒中采图与平差调优，互不干扰。",
            "• 验证当前场景精度达标后，按 [P] 键即可将当前高精度地图同步发布",
            "  至 config/tags_map.yaml，使整个生产系统即刻获得高精度立体定位！"
        ]
        for idx, line in enumerate(explain_lines):
            draw_text(canvas, line, (box_x + 26, exp_y + 34 + idx * 20), font_size=12, color=self.COLOR_WHITE)

    def _render_right_album_panel(self, canvas: np.ndarray, state: HubState, sc):
        """渲染最右侧栏：采样相册画廊与大图预览视口 (x: 800~1280, y: 50~670)"""
        box_x, box_y, box_w, box_h = 800, 50, 480, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_PANEL, -1)

        # 栏目标题
        draw_text(canvas, f"采样相册 ({len(state.current_images)} 帧)", (box_x + 16, box_y + 14), font_size=17, color=self.COLOR_WHITE, bold=True)

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
            y = box_y + 44

            thumb = state.get_thumbnail(img_path, tw, th)
            if thumb is not None:
                canvas[y:y + th, x:x + tw] = thumb

            border_col = (0, 240, 200) if is_cur else self.COLOR_BORDER
            cv2.rectangle(canvas, (x, y), (x + tw, y + th), border_col, 2 if is_cur else 1)

            base_name = os.path.basename(img_path)
            cv2.rectangle(canvas, (x, y + th - 15), (x + tw, y + th), (10, 10, 14), -1)
            cv2.putText(canvas, base_name[:12], (x + 3, y + th - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 180) if is_cur else self.COLOR_GRAY, 1, cv2.LINE_AA)

            if is_cur:
                cv2.line(canvas, (x + tw // 2 - 4, y + th + 3), (x + tw // 2 + 4, y + th + 3), (0, 255, 200), 2)

        # 2. 单帧照片高画质大图视口 (y: 172~656)
        prev_box_y = box_y + 118
        prev_box_h = box_h - 130
        cv2.rectangle(canvas, (box_x + 16, prev_box_y), (box_x + box_w - 16, prev_box_y + prev_box_h), (18, 21, 28), -1)
        cv2.rectangle(canvas, (box_x + 16, prev_box_y), (box_x + box_w - 16, prev_box_y + prev_box_h), self.COLOR_BORDER, 1)

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=430, max_h=390)

        img_title = f"单帧预览: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 28, prev_box_y + 10), font_size=15, color=self.COLOR_CYAN, bold=True)
        draw_text(canvas, "[按 F 键展开大视口]", (box_x + box_w - 160, prev_box_y + 11), font_size=13, color=(0, 255, 180))

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + 16 + (box_w - 32 - pw) // 2
            py = prev_box_y + 38 + (400 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (50, 56, 72), 1)

        draw_text(canvas, "[← / →] 左右键选片  |  [F] 放大预览占满空间",
                  (box_x + 70, prev_box_y + prev_box_h - 22), font_size=13, color=self.COLOR_GRAY)

    def _render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """全宽自适应大图视口 (按 F 键展开，横跨中间和右侧，x: 340~1280)"""
        box_x, box_y, box_w, box_h = 340, 50, 940, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 20, 26), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)

        if not state.current_images:
            draw_text(canvas, "当前场景无图片", (box_x + 400, box_y + 280), font_size=20, color=self.COLOR_DARK_GRAY)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=910, max_h=520)

        img_title = f"全宽自适应大图预览: {os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 20, box_y + 14), font_size=17, color=(0, 255, 200), bold=True)
        draw_text(canvas, "[按 F 键退出大图，恢复三栏标准看板]", (box_x + box_w - 300, box_y + 14), font_size=14, color=self.COLOR_GOLD)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 48 + (520 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 70, 90), 1)

        draw_text(canvas, "[← / →] 切换相册大图   |   [Space] 空间抓拍   |   [F] 退出全宽预览恢复三栏",
                  (box_x + 230, box_y + box_h - 24), font_size=14, color=self.COLOR_WHITE)

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
            if state.is_toolbox_open:
                draw_text(canvas, "【工具箱已激活】[S] Studio平差  [A] 在线AR验证  [L] 盲测体检  [D] 漏检切片  [T] 标靶图纸  [W] 白名单  [ESC / M] 关闭",
                          (20, 686), font_size=14, color=(0, 255, 200), bold=True)
            elif state.mode == HubState.MODE_CAPTURE:
                draw_text(canvas, "[Space] 抓拍存入当前场景   |   [ESC / C] 退出采图返回三栏看板   |   [S] 立即平差",
                          (20, 686), font_size=15, color=self.COLOR_WHITE)
            else:
                draw_text(canvas, "[M] 工具箱菜单  [N] 新建场景  [R] 改名  [K] 克隆  [↑/↓] 选场景  [Enter] 设活动  [C] 连拍  [S] 平差  [P] 生效生产  [F] 放大",
                          (20, 686), font_size=14, color=self.COLOR_WHITE)

    def _render_toolbox_modal(self, canvas: np.ndarray, state: HubState):
        """渲染中央悬浮置顶的【标定工具箱综合菜单】(880x520)"""
        # 1. 半透明暗色毛玻璃遮罩压暗底层
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_w, self.canvas_h), (8, 10, 14), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # 2. 弹窗主体视口
        modal_w, modal_h = 880, 520
        mx = (self.canvas_w - modal_w) // 2  # 200
        my = (self.canvas_h - modal_h) // 2  # 100

        # 弹窗底板与多层科技线框
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (20, 24, 32), -1)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (0, 200, 240), 2)
        cv2.rectangle(canvas, (mx + 4, my + 4), (mx + modal_w - 4, my + modal_h - 4), (40, 50, 66), 1)

        # 3. 弹窗标题栏 (my ~ my + 54)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + 54), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 54), (mx + modal_w, my + 54), self.COLOR_BORDER, 1)

        cv2.circle(canvas, (mx + 22, my + 27), 6, (0, 240, 220), -1)
        draw_text(canvas, "★ AprilTag 视觉标定综合工具箱总菜单 (Toolbox Hub)", (mx + 36, my + 15),
                  font_size=18, color=self.COLOR_WHITE, bold=True)

        # 右上角 [X] 关闭按钮 (mx + modal_w - 110 ~ mx + modal_w - 20)
        cv2.rectangle(canvas, (mx + modal_w - 106, my + 13), (mx + modal_w - 18, my + 41), (28, 32, 44), -1)
        cv2.rectangle(canvas, (mx + modal_w - 106, my + 13), (mx + modal_w - 18, my + 41), (70, 80, 100), 1)
        draw_text(canvas, "[X] 关闭 (ESC)", (mx + modal_w - 98, my + 18), font_size=12, color=self.COLOR_GRAY)

        # 副标题说明
        draw_text(canvas, "一站式极速唤起系统内所有独立 OpenCV 交互式图形化诊断、平差、图纸生成与在线验证工具：",
                  (mx + 28, my + 66), font_size=13, color=(0, 200, 240))

        # 4. 6 大核心工具交互卡片 (2列 x 3行)
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

            # 绘制工具卡片底板与边框
            cv2.rectangle(canvas, (x, y), (x + cw, y + ch), (26, 31, 42), -1)
            cv2.rectangle(canvas, (x, y), (x + cw, y + ch), (44, 52, 70), 1)

            # 左侧高亮色条
            cv2.rectangle(canvas, (x, y), (x + 4, y + ch), col, -1)

            # 标题 (带快捷键高亮)
            draw_text(canvas, title, (x + 14, y + 10), font_size=15, color=col, bold=True)

            # 两行功能描述 (自动换行排版)
            desc_line1 = desc[:28]
            desc_line2 = desc[28:56]
            draw_text(canvas, desc_line1, (x + 14, y + 36), font_size=12, color=self.COLOR_WHITE)
            if desc_line2:
                draw_text(canvas, desc_line2, (x + 14, y + 56), font_size=12, color=self.COLOR_GRAY)

            # 右下角点击启动徽章
            cv2.rectangle(canvas, (x + cw - 72, y + ch - 22), (x + cw - 8, y + ch - 6), (16, 22, 30), -1)
            cv2.putText(canvas, "CLICK", (x + cw - 62, y + ch - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

        # 5. 弹窗底部操作指引 (y: my + modal_h - 40)
        footer_y = my + modal_h - 40
        cv2.line(canvas, (mx + 20, footer_y), (mx + modal_w - 20, footer_y), (36, 44, 58), 1)
        draw_text(canvas, "★ 操作提示: 鼠标直接点击对应工具卡片，或直接按下键盘快捷键 [S / A / L / D / T / W] 即可秒级拉起！",
                  (mx + 30, footer_y + 12), font_size=13, color=self.COLOR_GOLD)


