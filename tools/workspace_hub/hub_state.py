"""
Workspace Hub 全局状态与数据模型 (HubState)
======================================
管理多 Workspace 列表、当前选定 Workspace 状态、采图连拍与模式流转
"""

import os
import glob
import time
from collections import OrderedDict
import cv2
import numpy as np

from src.calibration.workspace_manager import WorkspaceManager, Workspace


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """兼容 Windows 中文/特殊字符物理路径的鲁棒图像读取 (np.fromfile + cv2.imdecode)"""
    if not os.path.exists(filepath):
        return None
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        if data is None or len(data) == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(filepath: str, img: np.ndarray) -> bool:
    """兼容 Windows 中文/特殊字符物理路径的鲁棒图像写入 (cv2.imencode + tofile)"""
    try:
        ext = os.path.splitext(filepath)[1]
        ok, buf = cv2.imencode(ext, img)
        if ok and buf is not None:
            buf.tofile(filepath)
            return True
        return False
    except Exception:
        return False


class HubState:
    """工作空间中枢 (Workspace Hub) 统一状态与缓存模型"""

    # 核心视图模式 (标准三栏 / 全宽大图 / 纯净数据看板)
    VIEW_STANDARD = "standard"    # 模式1: 标准三栏 (左340, 中460, 右480)
    VIEW_EXPANDED = "expanded"    # 模式2: 全宽大图 (左340, 右940大图铺满)
    VIEW_DASHBOARD = "dashboard"  # 模式3: 纯净健康看板 (左340固定, 右940大体检看板，无相册无预览)

    def __init__(self, workspace_mgr: WorkspaceManager = None, force_mock: bool = False):
        self.workspace_mgr = workspace_mgr or WorkspaceManager()

        self.workspaces: list[Workspace] = []
        self.prod_workspace_id = ""
        self.selected_workspace_idx = 0

        # 当前选中工位的照片列表与大图选中项
        self.current_images: list[str] = []
        self.selected_image_idx = 0
        self.image_strip_offset = 0

        # 内存缩略图与预览图缓存 (有序字典实现 LRU，限制最大 200 张防内存溢出)
        self.thumbnail_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_cache_size = 200

        # 状态 Toast 提示
        self.toast_msg = ""
        self.toast_time = 0.0

        # 当前视图模式 (默认标准三栏，按 F 键或点击顶部 Tab 循环切换)
        self.view_mode = self.VIEW_STANDARD

        # 生产系统生效机制 Help 说明弹层 (按 H 键或点击 [? Help] 呼出)
        self.is_help_modal_open = False

        # 工位卡片右键上下文菜单 (Context Menu) 状态
        self.context_menu_open = False
        self.context_menu_pos = (0, 0)
        self.context_menu_ws_idx = -1

        # 当前鼠标悬停坐标 (用于按钮 Hover 高亮效果)
        self.mouse_x = -1
        self.mouse_y = -1

        # 初始加载工位
        self.refresh_workspaces()

    def refresh_workspaces(self):
        """刷新工位列表与生产工位标识"""
        self.workspaces = self.workspace_mgr.list_workspaces()
        self.prod_workspace_id = self.workspace_mgr.get_production_workspace_id()

        # 确保选中索引不越界
        if not self.workspaces:
            self.selected_workspace_idx = 0
        else:
            self.selected_workspace_idx = max(0, min(self.selected_workspace_idx, len(self.workspaces) - 1))

        self.load_current_workspace_images()

    def get_production_workspace(self) -> Workspace | None:
        """获取当前发布为生产运行的工位"""
        for ws in self.workspaces:
            if ws.workspace_id == self.prod_workspace_id:
                return ws
        for ws in self.workspaces:
            if ws.is_published:
                return ws
        return None

    def get_selected_workspace(self) -> Workspace | None:
        """获取当前高亮选中的工位"""
        if not self.workspaces or self.selected_workspace_idx >= len(self.workspaces):
            return None
        return self.workspaces[self.selected_workspace_idx]

    def select_workspace_by_offset(self, delta: int):
        """按偏移量切换选中的工位卡片"""
        if not self.workspaces:
            return
        new_idx = (self.selected_workspace_idx + delta) % len(self.workspaces)
        if new_idx != self.selected_workspace_idx:
            self.selected_workspace_idx = new_idx
            self.selected_image_idx = 0
            self.image_strip_offset = 0
            self.load_current_workspace_images()

    def publish_selected_to_production(self) -> bool:
        """将当前选中的工位发布为全局生产运行地图"""
        ws = self.get_selected_workspace()
        if not ws:
            self.set_toast("未选中有效工位")
            return False
        if not ws.ba_solved or not os.path.exists(ws.map_path):
            self.set_toast("发布失败: 该工位尚未进行 BA 平差解算或地图文件缺失")
            return False
        res = self.workspace_mgr.publish_to_production(ws.workspace_id)
        ok = res[0] if isinstance(res, (tuple, list)) else bool(res)
        msg = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else ""
        if ok:
            self.refresh_workspaces()
            self.set_toast(f"★ 工位【{ws.name}】已成功发布为全局生产运行地图！")
        else:
            self.set_toast(f"发布失败: {msg or '无法写入全局生产地图文件'}")
        return ok

    def load_current_workspace_images(self):
        """载入当前选中工位的照片列表"""
        ws = self.get_selected_workspace()
        if not ws or not os.path.exists(ws.calib_raw_images_dir):
            self.current_images = []
            return

        imgs = sorted(glob.glob(os.path.join(ws.calib_raw_images_dir, "*.png")))
        self.current_images = imgs
        if self.current_images:
            self.selected_image_idx = max(0, min(self.selected_image_idx, len(self.current_images) - 1))
        else:
            self.selected_image_idx = 0

    def select_image_by_offset(self, delta: int):
        """在缩略图流中左右切换选中的单帧图片"""
        if not self.current_images:
            return
        new_idx = max(0, min(self.selected_image_idx + delta, len(self.current_images) - 1))
        self.selected_image_idx = new_idx
        # 调整横向滚动带偏移量
        if self.selected_image_idx < self.image_strip_offset:
            self.image_strip_offset = self.selected_image_idx
        elif self.selected_image_idx >= self.image_strip_offset + 5:
            self.image_strip_offset = self.selected_image_idx - 4

    def delete_selected_image(self) -> bool:
        """删除当前选中的照片帧（物理安全移除、清理缓存，并自适应指向相邻帧）"""
        if not self.current_images:
            self.set_toast("当前工位相册为空，无照片可删除。")
            return False

        idx = self.selected_image_idx
        if idx < 0 or idx >= len(self.current_images):
            return False

        img_path = self.current_images[idx]
        file_name = os.path.basename(img_path)

        try:
            if os.path.exists(img_path):
                os.remove(img_path)

            # 清理缩略图与预览图缓存
            keys_to_del = [k for k in self.thumbnail_cache if k.startswith(img_path)]
            for k in keys_to_del:
                self.thumbnail_cache.pop(k, None)
            keys_to_del_prev = [k for k in self.preview_cache if k.startswith(img_path)]
            for k in keys_to_del_prev:
                self.preview_cache.pop(k, None)

            # 重新载入相册列表
            self.load_current_workspace_images()

            # 自适应定位相邻图片
            if self.current_images:
                self.selected_image_idx = min(idx, len(self.current_images) - 1)
            else:
                self.selected_image_idx = 0
            self.image_strip_offset = max(0, min(self.selected_image_idx, len(self.current_images) - 4))

            # 同步更新工位对象的 image_count
            ws = self.get_selected_workspace()
            if ws:
                ws.image_count = len(self.current_images)

            self.set_toast(f"已删除照片: {file_name}")
            return True
        except Exception as e:
            self.set_toast(f"删除照片失败: {e}")
            return False

    def get_thumbnail(self, img_path: str, tw: int = 110, th: int = 70) -> np.ndarray | None:
        """获取缩略图 (带 LRU 内存缓存)"""
        if not os.path.exists(img_path):
            return None
        key = f"{img_path}_{tw}_{th}"
        if key in self.thumbnail_cache:
            self.thumbnail_cache.move_to_end(key)
            return self.thumbnail_cache[key]

        bgr = imread_unicode(img_path)
        if bgr is None:
            return None
        thumb = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)

        # 缓存大小控制
        if len(self.thumbnail_cache) >= self.max_cache_size:
            self.thumbnail_cache.popitem(last=False)
        self.thumbnail_cache[key] = thumb
        return thumb

    def get_preview(self, img_path: str, max_w: int = 440, max_h: int = 280) -> np.ndarray | None:
        """获取单帧高清预览图 (等比例缩放)"""
        if not os.path.exists(img_path):
            return None
        key = f"{img_path}_{max_w}_{max_h}"
        if key in self.preview_cache:
            self.preview_cache.move_to_end(key)
            return self.preview_cache[key]

        bgr = imread_unicode(img_path)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        prev = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        if len(self.preview_cache) >= self.max_cache_size:
            self.preview_cache.popitem(last=False)
        self.preview_cache[key] = prev
        return prev

    def save_capture_frame(self, raw_frame: np.ndarray) -> str:
        """将当前相机帧归档至选中的工位沙盒 raw_images 目录"""
        ws = self.get_selected_workspace()
        if not ws:
            return ""

        os.makedirs(ws.calib_raw_images_dir, exist_ok=True)
        # 获取现有帧的最大序号
        existing = glob.glob(os.path.join(ws.calib_raw_images_dir, "view_*.png"))
        max_idx = 0
        for f in existing:
            base = os.path.basename(f)
            num_part = base.replace("view_", "").replace(".png", "")
            if num_part.isdigit():
                max_idx = max(max_idx, int(num_part))

        new_idx = max_idx + 1
        filename = f"view_{new_idx:04d}.png"
        filepath = os.path.join(ws.calib_raw_images_dir, filename)
        imwrite_unicode(filepath, raw_frame)

        # 触发白闪动效
        self.flash_timer = time.time() + 0.08

        # 刷新工位状态
        ws.refresh_stats()
        ws.save_meta()
        self.load_current_workspace_images()
        self.selected_image_idx = len(self.current_images) - 1
        self.set_toast(f"快照保存成功: {filename} (工位累计 {ws.image_count} 帧)")
        return filepath

    def set_toast(self, msg: str, duration: float = 3.0):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    @property
    def expanded_preview_mode(self) -> bool:
        """当处于全宽大图模式时返回 True"""
        return self.view_mode == self.VIEW_EXPANDED

    @expanded_preview_mode.setter
    def expanded_preview_mode(self, val: bool):
        self.view_mode = self.VIEW_EXPANDED if val else self.VIEW_STANDARD

    def set_view_mode(self, mode: str):
        """显式设定指定视图模式 (支持三段式 Tab 点击)"""
        if mode in (self.VIEW_STANDARD, self.VIEW_EXPANDED, self.VIEW_DASHBOARD):
            self.view_mode = mode
            names = {
                self.VIEW_STANDARD: "标准三栏看板",
                self.VIEW_EXPANDED: "全宽大图沉浸",
                self.VIEW_DASHBOARD: "纯净健康大屏 (无相册)",
            }
            self.set_toast(f"已切换视图模式: 【{names[mode]}】 (按 F 键循环切换)")

    def cycle_view_mode(self):
        """按 [F] 键顺次循环切换视图模式: 标准 -> 全宽大图 -> 纯净看板 -> 标准..."""
        modes = [self.VIEW_STANDARD, self.VIEW_EXPANDED, self.VIEW_DASHBOARD]
        curr_idx = modes.index(self.view_mode) if self.view_mode in modes else 0
        next_mode = modes[(curr_idx + 1) % len(modes)]
        self.set_view_mode(next_mode)

    def toggle_expanded_preview(self):
        """切换全宽大图模式与标准看板模式"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            self.set_view_mode(self.VIEW_EXPANDED)

    def toggle_help_modal(self):
        """打开或关闭生产系统发布机制说明弹窗 (按 H 键或点击 [? Help] 切换)"""
        self.is_help_modal_open = not self.is_help_modal_open
        if self.is_help_modal_open:
            self.set_toast("已呼出【生效到生产系统】业务说明窗 (按 ESC/H 关闭)")
        else:
            self.set_toast("已关闭说明窗。")

    def rename_current_workspace(self, new_name: str) -> bool:
        """重命名当前选中的工位显示名称 (支持中文)"""
        ws = self.get_selected_workspace()
        if not ws:
            return False
        clean = new_name.strip()
        if not clean:
            return False
        ok = self.workspace_mgr.rename_workspace(ws.workspace_id, clean)
        if ok:
            ws.name = clean
            self.refresh_workspaces()
            self.set_toast(f"工位名称已成功修改为: 【{clean}】")
        return ok

    def open_context_menu(self, x: int, y: int, ws_idx: int):
        """在指定鼠标坐标处打开工位卡片的右键上下文菜单"""
        if 0 <= ws_idx < len(self.workspaces):
            self.context_menu_open = True
            self.context_menu_pos = (x, y)
            self.context_menu_ws_idx = ws_idx
            self.selected_workspace_idx = ws_idx
            self.load_current_workspace_images()

    def close_context_menu(self):
        """关闭右键上下文菜单"""
        self.context_menu_open = False
        self.context_menu_ws_idx = -1
