"""
Workspace Hub 全局状态与数据模型 (HubState)
======================================
管理多 Workspace 列表、当前选定 Workspace 状态、采图连拍与模式流转
"""

import os
import glob
import json
import time
from collections import OrderedDict
import cv2
import numpy as np

from src.calibration.workspace_manager import WorkspaceManager, Workspace
from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "workspace_hub"


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

    # 右侧动态区页签 (左右两栏布局: 左侧 Workspace 导航固定, 右侧动态内容四页签)
    TAB_CALIB_IMAGES = "tab_calib_images"   # 页签1: 标定相册 (当前工位采样相册)
    TAB_PROD_IMAGES = "tab_prod_images"     # 页签2: 生产相册 (生产基准工位相册)
    TAB_REPORT = "tab_report"               # 页签3: 体检报告 (几何健康大屏)
    TAB_WHITELIST = "tab_whitelist"         # 页签4: Tag 白名单
    # 页签展示顺序: 1 Dashboard / 2 Tag白名单 / 3 标定相册 / 4 生产相册
    TAB_ORDER = (TAB_REPORT, TAB_WHITELIST, TAB_CALIB_IMAGES, TAB_PROD_IMAGES)

    # 视图模式 (全宽大图沉浸预览, 仅在标定相册页签下双击卡片展开)
    VIEW_STANDARD = "standard"    # 标准: 左栏 + 右侧页签内容
    VIEW_EXPANDED = "expanded"    # 全宽大图: 右侧区域整体铺满单帧大图

    # 相册卡片网格规格 (与渲染器保持一致): 3 列 x 3 行 = 每页 9 张大卡片
    GRID_COLS = 3
    GRID_ROWS = 3
    GRID_PAGE = 9

    def __init__(self, workspace_mgr: WorkspaceManager = None, force_mock: bool = False):
        self.workspace_mgr = workspace_mgr or WorkspaceManager()

        self.workspaces: list[Workspace] = []
        self.selected_workspace_idx = 0

        # 当前选中工位的照片列表与卡片网格选中项
        self.current_images: list[str] = []
        self.selected_image_idx = 0
        self.image_grid_offset = 0   # 卡片网格当前页起始索引 (按整行对齐)

        # 生产相册相关状态 (生产运行基准工位相册)
        self.prod_images: list[str] = []
        self.selected_prod_image_idx = 0
        self.prod_grid_offset = 0

        # 内存缩略图与预览图缓存 (有序字典实现 LRU，限制最大 200 张防内存溢出)
        self.thumbnail_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_cache_size = 200

        # 状态 Toast 提示
        self.toast_msg = ""
        self.toast_time = 0.0

        # 当前视图模式 (默认标准; 按 F 键在标定相册页签内进入全宽大图)
        self.view_mode = self.VIEW_STANDARD

        # 右侧动态区当前激活页签 (默认: 体检报告)
        self.active_tab = self.TAB_REPORT

        # Tag 白名单缓存 (按文件 mtime 自动感知外部编辑并刷新)
        self._whitelist_cache: dict = {}
        self._whitelist_cache_mtime: float = -1.0
        self._whitelist_cache_ws: str = ""

        # 工位卡片右键上下文菜单 (Context Menu) 状态
        self.context_menu_open = False
        self.context_menu_pos = (0, 0)
        self.context_menu_ws_idx = -1

        # 生产机制业务说明弹窗状态
        self.is_help_modal_open = False

        # 当前鼠标悬停坐标 (用于按钮 Hover 高亮效果)
        self.mouse_x = -1
        self.mouse_y = -1

        # 左侧激活卡片的持久化 (按 workspace_id 定位, 不受列表排序变化影响)
        self._saved_workspace_id = self._read_saved_workspace_id()
        self._selection_restored = False

        # 初始加载工位
        self.refresh_workspaces()

    # ------------------------------ 左侧激活卡片持久化 ------------------------------
    @staticmethod
    def _read_saved_workspace_id() -> str:
        """读取上次激活的工位 ID (config/gui_settings.json → workspace_hub.hub_state)"""
        try:
            if not os.path.exists(GUI_SETTINGS_FILE):
                return ""
            with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            return str(((root.get(APP_ID) or {}).get("hub_state") or {})
                       .get("selected_workspace_id") or "")
        except Exception as e:
            log.warning(f"读取 Workspace Hub 激活工位失败: {e}")
            return ""

    def save_selected_workspace(self):
        """持久化当前左侧激活的工位卡片 (退出后下次启动自动恢复高亮与相册)"""
        ws = self.get_selected_workspace()
        if ws is None:
            return
        try:
            root = {}
            if os.path.exists(GUI_SETTINGS_FILE):
                try:
                    with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                        root = json.load(f)
                    if not isinstance(root, dict):
                        root = {}
                except Exception:
                    root = {}
            node = root.setdefault(APP_ID, {})
            state = node.get("hub_state")
            if not isinstance(state, dict):
                state = {}
            state["selected_workspace_id"] = str(ws.workspace_id)
            state["selected_workspace_name"] = str(ws.name)
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            node["hub_state"] = state
            os.makedirs(os.path.dirname(GUI_SETTINGS_FILE), exist_ok=True)
            with open(GUI_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存 Workspace Hub 激活工位失败: {e}")

    def select_workspace_at_index(self, idx: int):
        """激活左侧第 idx 张工位卡片 (高亮 → 载入其相册 → 持久化选择)"""
        if not self.workspaces or not (0 <= idx < len(self.workspaces)):
            return
        if idx != self.selected_workspace_idx:
            self.selected_workspace_idx = idx
            self.selected_image_idx = 0
            self.image_grid_offset = 0
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
        self.load_current_workspace_images()
        self.load_prod_images()
        self.save_selected_workspace()
        ws = self.get_selected_workspace()
        if ws:
            self.workspace_mgr.set_active_workspace(ws.workspace_id)

    def refresh_workspaces(self):
        """刷新工位列表"""
        self.workspaces = self.workspace_mgr.list_workspaces()

        # 首次加载: 恢复上次激活的工位卡片 (按 workspace_id 定位, 索引排序变化无影响)
        if not self._selection_restored:
            self._selection_restored = True
            if self._saved_workspace_id:
                for i, ws in enumerate(self.workspaces):
                    if ws.workspace_id == self._saved_workspace_id:
                        self.selected_workspace_idx = i
                        break

        # 确保选中索引不越界
        if not self.workspaces:
            self.selected_workspace_idx = 0
        else:
            self.selected_workspace_idx = max(0, min(self.selected_workspace_idx, len(self.workspaces) - 1))

        self.load_current_workspace_images()
        self.load_prod_images()
        # 列表变动 (新建/克隆/重命名/删除) 后同步落盘激活卡片
        self.save_selected_workspace()

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
            self.image_grid_offset = 0
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            self.load_current_workspace_images()
            self.load_prod_images()
        self.save_selected_workspace()

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

    def _clamp_grid_offset(self, offset: int, total: int) -> int:
        """将网格分页起始索引按整行对齐并夹紧到合法范围"""
        aligned = max(0, (offset // self.GRID_COLS) * self.GRID_COLS)
        max_offset = max(0, ((max(total, 1) - 1) // self.GRID_PAGE) * self.GRID_PAGE)
        return max(0, min(aligned, max_offset))

    def scroll_image_grid(self, delta_rows: int):
        """卡片网格按行滚动 (滚轮/翻页按钮驱动)"""
        self.image_grid_offset = self._clamp_grid_offset(
            self.image_grid_offset + delta_rows * self.GRID_COLS, len(self.current_images))

    def select_image_at_index(self, idx: int):
        """直接选中第 idx 张卡片 (自动翻页使其可见)"""
        if 0 <= idx < len(self.current_images):
            self.selected_image_idx = idx
            self._ensure_image_visible()

    def _ensure_image_visible(self):
        """确保当前选中卡片处于可见页范围内 (自动翻页)"""
        if not self.current_images:
            self.image_grid_offset = 0
            return
        idx = self.selected_image_idx
        start = self.image_grid_offset
        if idx < start:
            self.image_grid_offset = self._clamp_grid_offset(
                (idx // self.GRID_COLS) * self.GRID_COLS, len(self.current_images))
        elif idx >= start + self.GRID_PAGE:
            target_row = max(0, (idx // self.GRID_COLS) - self.GRID_ROWS + 1)
            self.image_grid_offset = self._clamp_grid_offset(
                target_row * self.GRID_COLS, len(self.current_images))

    def select_image_by_offset(self, delta: int):
        """在卡片网格中前后切换选中的单帧图片 (自动翻页跟随)"""
        if not self.current_images:
            return
        new_idx = max(0, min(self.selected_image_idx + delta, len(self.current_images) - 1))
        self.selected_image_idx = new_idx
        self._ensure_image_visible()

    def load_prod_images(self):
        """载入当前选中工位的生产采图相册 (生产相册页签数据源)"""
        ws = self.get_selected_workspace()
        if not ws or not os.path.exists(ws.prod_raw_images_dir):
            self.prod_images = []
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        imgs = []
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(ws.prod_raw_images_dir, ext)))
        self.prod_images = sorted(list(set(imgs)))
        if self.prod_images:
            self.selected_prod_image_idx = max(0, min(self.selected_prod_image_idx, len(self.prod_images) - 1))
        else:
            self.selected_prod_image_idx = 0

    def scroll_prod_grid(self, delta_rows: int):
        """生产相册卡片网格按行滚动"""
        self.prod_grid_offset = self._clamp_grid_offset(
            self.prod_grid_offset + delta_rows * self.GRID_COLS, len(self.prod_images))

    def select_prod_image_at_index(self, idx: int):
        """直接选中生产相册第 idx 张卡片 (自动翻页使其可见)"""
        if 0 <= idx < len(self.prod_images):
            self.selected_prod_image_idx = idx
            self._ensure_prod_visible()

    def _ensure_prod_visible(self):
        """确保当前选中的生产相册卡片处于可见页范围内"""
        if not self.prod_images:
            self.prod_grid_offset = 0
            return
        idx = self.selected_prod_image_idx
        start = self.prod_grid_offset
        if idx < start:
            self.prod_grid_offset = self._clamp_grid_offset(
                (idx // self.GRID_COLS) * self.GRID_COLS, len(self.prod_images))
        elif idx >= start + self.GRID_PAGE:
            target_row = max(0, (idx // self.GRID_COLS) - self.GRID_ROWS + 1)
            self.prod_grid_offset = self._clamp_grid_offset(
                target_row * self.GRID_COLS, len(self.prod_images))

    def select_prod_image_by_offset(self, delta: int):
        """在生产相册卡片网格中前后切换选中的单帧图片"""
        if not self.prod_images:
            return
        new_idx = max(0, min(self.selected_prod_image_idx + delta, len(self.prod_images) - 1))
        self.selected_prod_image_idx = new_idx
        self._ensure_prod_visible()

    def get_tag_whitelist(self) -> dict:
        """读取当前选中工位的 tag_whitelist.yaml (基于 mtime 自动感知外部编辑并刷新缓存)"""
        ws = self.get_selected_workspace()
        if not ws:
            return {}
        path = self.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        if not os.path.exists(path):
            return {}
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return {}

        if (self._whitelist_cache_ws == ws.workspace_id
                and self._whitelist_cache
                and abs(mtime - self._whitelist_cache_mtime) < 1e-6):
            return self._whitelist_cache

        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return {}

        self._whitelist_cache = data
        self._whitelist_cache_mtime = mtime
        self._whitelist_cache_ws = ws.workspace_id
        return data

    def refresh_whitelist_cache(self):
        """强制失效白名单缓存 (外部编辑器保存返回后立即刷新)"""
        self._whitelist_cache = {}
        self._whitelist_cache_mtime = -1.0
        self._whitelist_cache_ws = ""

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
            self._ensure_image_visible()

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
        """显式设定视图模式 (标准页签看板 / 全宽大图沉浸)"""
        if mode in (self.VIEW_STANDARD, self.VIEW_EXPANDED):
            self.view_mode = mode
            names = {
                self.VIEW_STANDARD: "标准页签看板",
                self.VIEW_EXPANDED: "全宽大图沉浸",
            }
            self.set_toast(f"已切换视图模式: 【{names[mode]}】")

    def cycle_view_mode(self):
        """切换视图模式: 标准页签 <-> 全宽大图 (双击卡片触发)"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            self.set_view_mode(self.VIEW_EXPANDED)

    def toggle_expanded_preview(self):
        """切换全宽大图模式与标准看板模式 (全宽大图仅作用于标定相册页签)"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            self.active_tab = self.TAB_CALIB_IMAGES
            self.set_view_mode(self.VIEW_EXPANDED)

    def set_tab(self, tab: str):
        """切换右侧动态区页签 (左栏保持稳定，仅右栏内容动态更新)"""
        if tab not in self.TAB_ORDER:
            return
        if tab == self.active_tab and self.view_mode == self.VIEW_STANDARD:
            return
        # 离开全宽大图沉浸模式
        self.view_mode = self.VIEW_STANDARD
        if tab != self.active_tab:
            self.active_tab = tab
            if tab == self.TAB_PROD_IMAGES:
                self.load_prod_images()
            elif tab == self.TAB_CALIB_IMAGES:
                self.load_current_workspace_images()
            names = {
                self.TAB_CALIB_IMAGES: "标定相册",
                self.TAB_PROD_IMAGES: "生产相册",
                self.TAB_REPORT: "体检报告",
                self.TAB_WHITELIST: "Tag 白名单",
            }
            self.set_toast(f"已切换页签: 【{names[tab]}】")

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

    def update_current_workspace_description(self, new_desc: str) -> bool:
        """更新当前选中工位的备注说明文本 (支持中文单行文本)"""
        ws = self.get_selected_workspace()
        if not ws:
            return False
        clean = str(new_desc).strip()
        ok = self.workspace_mgr.update_workspace_description(ws.workspace_id, clean)
        if ok:
            ws.description = clean
            self.refresh_workspaces()
            self.set_toast(f"工位备注已成功修改为: 【{clean or '无'}】")
        return ok

    def open_context_menu(self, x: int, y: int, ws_idx: int):
        """在指定鼠标坐标处打开工位卡片的右键上下文菜单"""
        if 0 <= ws_idx < len(self.workspaces):
            self.context_menu_open = True
            self.context_menu_pos = (x, y)
            self.context_menu_ws_idx = ws_idx
            self.select_workspace_at_index(ws_idx)

    def close_context_menu(self):
        """关闭右键上下文菜单"""
        self.context_menu_open = False
        self.context_menu_ws_idx = -1

    def toggle_help_modal(self):
        """打开或关闭 Workspace 工位与生产体系说明弹窗 (按 H 键切换)"""
        self.is_help_modal_open = not self.is_help_modal_open
        if self.is_help_modal_open:
            self.set_toast("已呼出【Workspace 工位与生产体系】业务说明窗 (按 ESC/H 关闭)")
        else:
            self.set_toast("已关闭说明窗。")
