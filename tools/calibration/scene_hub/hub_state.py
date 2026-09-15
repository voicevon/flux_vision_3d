"""
Scene Hub 全局状态与数据模型 (HubState)
======================================
管理场景列表、选中项、视口双模切换、内存缩略图 LRU 缓存与白闪动效
"""

import os
import glob
import time
from collections import OrderedDict
import cv2
import numpy as np

from src.calibration.scene_manager import CalibrationSceneManager, CalibrationScene
from src.calibration.camera_streamer import CameraStreamer


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
    """Scene Hub 统一状态与缓存管理器"""

    MODE_INSPECTOR = "inspector"  # 场景画廊与体检看板模式
    MODE_CAPTURE = "capture"      # 原地实时相机取流连拍模式

    def __init__(self, scene_mgr: CalibrationSceneManager = None, force_mock: bool = False):
        self.scene_mgr = scene_mgr or CalibrationSceneManager()
        self.mode = self.MODE_INSPECTOR

        self.scenes: list[CalibrationScene] = []
        self.active_scene_id = ""
        self.selected_scene_idx = 0

        # 当前选中场景的照片列表与大图选中项
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

        # 全宽大图预览模式 (按 F 键切换：全宽占满 vs 并排体检看板)
        self.expanded_preview_mode = False

        # 标定工具箱总菜单弹层是否打开 (按 M 键或点击呼出)
        self.is_toolbox_open = False

        # 生产系统生效机制 Help 说明弹层 (按 H 键或点击 [? Help] 呼出)
        self.is_help_modal_open = False

        # 当前鼠标悬停坐标 (用于按钮 Hover 高亮效果)
        self.mouse_x = -1
        self.mouse_y = -1

        # 空格抓拍白闪动效倒计时
        self.flash_timer = 0.0

        # 相机取流器句柄
        self.camera_streamer = CameraStreamer(force_mock=force_mock)

        # 初始加载场景
        self.refresh_scenes()

    def refresh_scenes(self):
        """刷新场景列表与活动场景标识"""
        self.scenes = self.scene_mgr.list_scenes()
        self.active_scene_id = self.scene_mgr.get_active_scene_id()

        # 确保选中索引不越界
        if not self.scenes:
            self.selected_scene_idx = 0
        else:
            self.selected_scene_idx = max(0, min(self.selected_scene_idx, len(self.scenes) - 1))

        self.load_current_scene_images()

    def get_selected_scene(self) -> CalibrationScene | None:
        """获取当前高亮选中的场景"""
        if not self.scenes or self.selected_scene_idx >= len(self.scenes):
            return None
        return self.scenes[self.selected_scene_idx]

    def select_scene_by_offset(self, delta: int):
        """按偏移量切换选中的场景卡片"""
        if not self.scenes:
            return
        new_idx = (self.selected_scene_idx + delta) % len(self.scenes)
        if new_idx != self.selected_scene_idx:
            self.selected_scene_idx = new_idx
            self.selected_image_idx = 0
            self.image_strip_offset = 0
            self.load_current_scene_images()

    def set_current_as_active(self) -> bool:
        """将当前选中的场景设为全局活动场景"""
        sc = self.get_selected_scene()
        if not sc:
            return False
        ok = self.scene_mgr.set_active_scene(sc.scene_id)
        if ok:
            self.active_scene_id = sc.scene_id
            self.set_toast(f"已将【{sc.scene_id}】设为全局活动场景！")
        return ok

    def load_current_scene_images(self):
        """载入当前选中场景的照片列表"""
        sc = self.get_selected_scene()
        if not sc or not os.path.exists(sc.raw_images_dir):
            self.current_images = []
            return

        imgs = sorted(glob.glob(os.path.join(sc.raw_images_dir, "*.png")))
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
        """将当前相机帧归档至选中的场景沙盒 raw_images 目录"""
        sc = self.get_selected_scene()
        if not sc:
            return ""

        os.makedirs(sc.raw_images_dir, exist_ok=True)
        # 获取现有帧的最大序号
        existing = glob.glob(os.path.join(sc.raw_images_dir, "view_*.png"))
        max_idx = 0
        for f in existing:
            base = os.path.basename(f)
            num_part = base.replace("view_", "").replace(".png", "")
            if num_part.isdigit():
                max_idx = max(max_idx, int(num_part))

        new_idx = max_idx + 1
        filename = f"view_{new_idx:04d}.png"
        filepath = os.path.join(sc.raw_images_dir, filename)
        imwrite_unicode(filepath, raw_frame)

        # 触发白闪动效
        self.flash_timer = time.time() + 0.08

        # 刷新场景状态
        sc.refresh_stats()
        sc.save_meta()
        self.load_current_scene_images()
        self.selected_image_idx = len(self.current_images) - 1
        self.set_toast(f"快照保存成功: {filename} (场景累计 {sc.image_count} 帧)")
        return filepath

    def set_toast(self, msg: str, duration: float = 3.0):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def toggle_expanded_preview(self):
        """切换单帧大图全宽占满/并排体检看板模式"""
        self.expanded_preview_mode = not self.expanded_preview_mode
        mode_desc = "全宽自适应沉浸模式" if self.expanded_preview_mode else "并排体检看板模式"
        self.set_toast(f"已切换预览视图: 【{mode_desc}】 (按 F 键再次切换)")

    def toggle_toolbox(self):
        """打开或关闭标定工具箱综合菜单 (按 M 键切换)"""
        self.is_toolbox_open = not self.is_toolbox_open
        if self.is_toolbox_open:
            self.is_help_modal_open = False
            self.set_toast("已打开标定工具箱总菜单 (按对应字母启动工具，按 ESC/M 关闭)")
        else:
            self.set_toast("已关闭工具箱总菜单，返回场景驾驶舱。")

    def toggle_help_modal(self):
        """打开或关闭生产系统发布机制说明弹窗 (按 H 键或点击 [? Help] 切换)"""
        self.is_help_modal_open = not self.is_help_modal_open
        if self.is_help_modal_open:
            self.is_toolbox_open = False
            self.set_toast("已呼出【生效到生产系统】业务说明窗 (按 ESC/H 关闭)")
        else:
            self.set_toast("已关闭说明窗。")

    def rename_current_scene(self, new_name: str) -> bool:
        """重命名当前选中的场景显示名称 (支持中文)"""
        sc = self.get_selected_scene()
        if not sc:
            return False
        clean = new_name.strip()
        if not clean:
            return False
        ok = self.scene_mgr.rename_scene(sc.scene_id, clean)
        if ok:
            sc.name = clean
            self.refresh_scenes()
            self.set_toast(f"场景名称已成功修改为: 【{clean}】")
        return ok
