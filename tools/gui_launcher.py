#!/usr/bin/env python3
"""
3D 视觉综合控制中心 (Suite Dashboard)
=================================================
基于 1280x830 工业科技大屏，统一调度 flux_vision_3d 视觉系统的所有核心应用：
- 原生 Windows Unicode 窗口标题，杜绝任何乱码
- 窗口拖拽缩放自适应动态分辨率 (Dynamic Native Canvas)，文字 1:1 矢量超采样，无任何拉伸锯齿
- 高端工业冷峻暗色系配色（低饱和度科技冷蓝、工业深青与钛银灰），视觉沉稳专业不刺眼
- 左侧模块化卡片：【核心生产与工况】、【视觉标定与建图流水线】、【自动化测试与系统运维】
- 右侧动态即时说明大屏 (Live Inspector)：自适应扩展宽度，提供详尽的现场工程指南
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")

from src.calibration.scene_manager import CalibrationSceneManager
from tools.env_utils import check_env_status

# 字体缓存
_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}


def get_cached_font(font_size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont:
    """获取缓存的 TrueType 中文字体"""
    key = (font_size, bold)
    if key not in _FONT_CACHE:
        font_paths = [
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simheittc.ttc" if bold else "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], font_size: int = 16,
              color: Tuple[int, int, int] = (240, 240, 240), bold: bool = False):
    """在 OpenCV BGR 图像上绘制高质量抗锯齿矢量文本 (支持中文)"""
    if not text:
        return
    x, y = pos
    if x >= img.shape[1] or y >= img.shape[0]:
        return

    font = get_cached_font(font_size, bold)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    patch_w = tw + 20
    patch_h = th + 14

    rx2 = min(img.shape[1], x + patch_w)
    ry2 = min(img.shape[0], y + patch_h)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if rx2 <= x or ry2 <= y:
        return

    sub_bgr = img[y:ry2, x:rx2]
    sub_rgb = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(sub_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((0, 0), text, font=font, fill=(color[2], color[1], color[0]))
    res_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[y:ry2, x:rx2] = res_bgr


def wrap_text_by_width(text: str, font_size: int, max_width: int, bold: bool = False) -> List[str]:
    """根据像素最大可用宽度对文本进行精确折行（支持中英混合与标点符号）"""
    if not text or max_width <= 30:
        return [text] if text else []

    font = get_cached_font(font_size, bold)
    lines: List[str] = []
    curr_line = ""

    for char in text:
        test_line = curr_line + char
        bbox = font.getbbox(test_line)
        tw = bbox[2] - bbox[0]
        if tw > max_width and curr_line:
            lines.append(curr_line)
            curr_line = char
        else:
            curr_line = test_line

    if curr_line:
        lines.append(curr_line)

    return lines


def draw_multiline_text(img: np.ndarray, text: str, pos: Tuple[int, int], max_width: int,
                        font_size: int = 16, color: Tuple[int, int, int] = (240, 240, 240),
                        line_spacing: int = 4, bold: bool = False, max_lines: int = 99) -> int:
    """按最大像素宽度折行绘制多行文本，返回下一段可用的起始 y 坐标"""
    lines = wrap_text_by_width(text, font_size, max_width, bold=bold)
    if not lines:
        return pos[1]

    x, y = pos
    line_h = font_size + line_spacing
    for idx, line in enumerate(lines[:max_lines]):
        draw_text(img, line, (x, y + idx * line_h), font_size=font_size, color=color, bold=bold)

    return y + min(len(lines), max_lines) * line_h


class ToolCardMeta:
    """工具卡片定义与其详细说明元数据"""

    def __init__(self, key_id: str, shortcut: str, title: str, subtitle: str,
                 category: str, is_gui: bool, command: List[str],
                 tag_color: Tuple[int, int, int],
                 summary: str, details: List[str], inputs: List[str],
                 outputs: List[str], quick_tips: str):
        self.key_id = key_id
        self.shortcut = shortcut
        self.title = title
        self.subtitle = subtitle
        self.category = category
        self.is_gui = is_gui
        self.command = command
        self.tag_color = tag_color
        self.summary = summary
        self.details = details
        self.inputs = inputs
        self.outputs = outputs
        self.quick_tips = quick_tips


def build_tools_catalog() -> List[ToolCardMeta]:
    """构建全系统核心工具目录：12 张卡片，四大功能分组 (A场景→B Tag标定→C示教标定→D生产调试)"""

    COLOR_A = (195, 155, 45)   # A 场景总控  : 琥珀金 (Amber)
    COLOR_B = (65,  175, 160)  # B Tag标定   : 精密工业深青 (Teal)
    COLOR_C = (220, 145,  60)  # C 示教标定  : 暖橙 (Manual Teach)
    COLOR_D = (90,  140, 195)  # D 生产调试  : 钢蓝 (Steel Blue)
    COLOR_E = (80,  190, 115)  # 预留色彩槽
    COLOR_F = (130, 145, 165)  # 预留色彩槽

    catalog = [
        # ===== A — 场景总控 (1张，顶部全宽) =====
        ToolCardMeta(
            key_id="scene_hub",
            shortcut="1",
            title="工况场景管理中枢 (Scene Hub)",
            subtitle="★ 顶层数据总控！沙盒画廊/大图巡检/生产发布",
            category="A — 场景总控",
            is_gui=True,
            command=[sys.executable, "-m", "tools.scene_hub"],
            tag_color=COLOR_A,
            summary="【首位核心中枢】视觉系统的工况沙盒容器与数据总控驾驶舱，连接采集、平差与生产部署。",
            details=[
                "多工况画廊管理：选择、新建、重命名、克隆与独立物理沙盒数据隔离",
                "三大视图模式：标准三栏工作台 / 单帧大图全宽巡检 / 纯净几何健康看板",
                "场景几何健康度体检：动态覆盖率热力、留一盲测残差分布与两阶段平差指标",
                "严格恪守【草稿沙盒隔离、活动场景验证、生产原子发布】工业安全基准"
            ],
            inputs=["data/calibration_scenes/ 工况沙盒目录"],
            outputs=["当前活动场景切换、scene_meta.yaml、一键原子发布到 config/tags_map.yaml"],
            quick_tips="快捷键: [1] 启动 | 中枢内 [⏎] 激活 | [P] 发布生产 | [S] 进Studio"
        ),

        # ===== B — Tag 标定流水线 (6张，2×3) =====
        ToolCardMeta(
            key_id="tag_generator",
            shortcut="2",
            title="AprilTag 标靶图纸生成器",
            subtitle="[G] 一键生成高清PNG与严格1:1 A4 PDF排版标靶",
            category="B — Tag 标定流水线",
            is_gui=False,
            command=[sys.executable, "tools/calibration/generate_apriltags.py"],
            tag_color=COLOR_B,
            summary="【标定流水线 Step 0】在标定现场第一步：物理打印 AprilTag 16h5 标靶图纸，生成高清 PNG 和严格 1:1 比例的 A4 PDF 排版文件。",
            details=[
                "生成 ID 00 ~ 29 共 30 个 AprilTag 16h5 高清独立 PNG 标靶卡片 (800×800)",
                "自动排版为 2 页 A4 PDF 文件（每页 3列×5行 = 15个标靶，标靶物理尺寸 40mm×40mm）",
                "每页顶部配备 100.0mm 物理校验尺，供游标卡尺验证打印比例严格 1:1 无失真",
                "Tag #0 特别标注 SCARA 旋转中心红线，Tag #1 标注世界 +X 参考轴",
                "同时生成 5×6 总览网格图 (apriltags_16h5_all_grid.png) 方便屏幕快速预览"
            ],
            inputs=["系统已安装 reportlab 库 (pip install reportlab)"],
            outputs=["data/apriltags_16h5/ 目录下的 PNG 标靶卡片、A4 PDF 排版文件与总览网格图"],
            quick_tips="快捷键: [2] 或 [G] 启动 (控制台执行) | 运行后请按 100% 实际尺寸打印 PDF，勿选“适应页面”"
        ),

        ToolCardMeta(
            key_id="tag_wizard",
            shortcut="3",
            title="多视角采图向导 (Wizard)",
            subtitle="[C] 角度雷达交互指引/空格极速连拍/自动归档沙盒",
            category="B — Tag 标定流水线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_capture_wizard.py"],
            tag_color=COLOR_B,
            summary="【现场采图助手】专职采图向导：交互式指导相机移动至不同高度与俯仰角，高效采集高质量标定样本。",
            details=[
                "提供雷达式多视角视盘指引（俯视、大俯仰角、大滚转角、高低多层位态）",
                "按 [空格键] 极速无感连拍，样本自动存入当前场景 raw_images/ 目录",
                "实时 AprilTag 检出回显与白闪快门反馈，采图完毕后返回主中枢自动热重载"
            ],
            inputs=["RealSense 深度相机 (或 --mock 仿真)"],
            outputs=["当前活动场景 raw_images/view_*.png 原始高质量未压缩图集"],
            quick_tips="快捷键: [3] 或 [C] 启动 | 采图界面中 [空格] 拍摄归档 | [R] 重置批次 | [ESC] 完成返回"
        ),

        ToolCardMeta(
            key_id="tag_studio",
            shortcut="4",
            title="离线标定工作站 (Studio)",
            subtitle="[S] 多视角审核/两阶段 BA 平差/智能剪枝/质检闭环",
            category="B — Tag 标定流水线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_offline_studio.py"],
            tag_color=COLOR_B,
            summary="【离线标定核心】一站式样本交互审核、高精两阶段 BA 平差求解、智能剪枝与质检闭环。",
            details=[
                "自动装载当前活动沙盒场景，多视角图像九宫格缩略图交互式审核与启闭",
                "两阶段全局平差：Cauchy 鲁棒核粗平差 + MAD 统计自适应清洗 + LM 精平差",
                "智能残差剪枝 (Auto-Prune)：自动迭代剪除反光/微动导致的高残差外点，拓扑安全守门",
                "视网膜级热力覆盖度评估，一键导出 Markdown 格式全面质检体检报告",
                "内置支持工具：标靶图纸生成 (A4 PDF) 与漏检病因切片诊断已全面打通支持"
            ],
            inputs=["当前活动场景 raw_images/", "相机内参 camera_intrinsics.yaml"],
            outputs=["当前场景 tags_map.yaml", "reports/studio_qa_report_*.md 质检报告"],
            quick_tips="快捷键: [4] 或 [S] 启动 | 工作站内 [⏎] 快速求解 | [P] 智能剪枝 | [E] 超精提取 | [R] 导出报告"
        ),

        ToolCardMeta(
            key_id="tag_offline_verifier",
            shortcut="5",
            title="离线精度体检台 (LOO盲测)",
            subtitle="[L] 标定后留一交叉验证/双棱柱对比/残差评级",
            category="B — Tag 标定流水线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_offline_verifier.py"],
            tag_color=COLOR_B,
            summary="【标定验收闭环】科学级精度评估台：执行严格的 Leave-One-Out (LOO) 盲测交叉验证与外参鲁棒性体检。",
            details=[
                "留一交叉验证：轮流屏蔽每一张标定图像作为未知盲测帧，求解相机外参并预测未参与平差的标靶",
                "3D 空间双棱柱虚实位姿对比：直观呈现盲测外参与全局优化外参的空间刚体位移偏差",
                "2D 像平面残差矢量放大图：标注重投影误差方向分布，揭示畸变或单侧光照系统误差",
                "输出严谨的工业放行评级：优秀 (A)、达标 (B) 或 需补拍 (C)"
            ],
            inputs=["当前活动场景样本图集", "当前场景 tags_map.yaml"],
            outputs=["data/tag_calibration_verification/ 诊断报告与残差矢量可视化图"],
            quick_tips="快捷键: [5] 或 [L] 启动 | 体检界面中 [N/P] 切换盲测帧 | [R] 导出评估报告 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="tag_ar_verifier",
            shortcut="6",
            title="在线 AR 虚实融合验收系统",
            subtitle="[A] 3D轴网虚实融合/时域外参滤波锁定/现场验收",
            category="B — Tag 标定流水线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_calibration_verifier.py"],
            tag_color=COLOR_B,
            summary="【车间透视验收】通过虚实融合 AR 盲测直接肉眼检验平差地图的物理精确度。",
            details=[
                "高帧率实时取流，在检测到的 AprilTag 空间位置上虚实融合叠加 3D 彩色坐标轴",
                "在已知标靶基准上虚实融合渲染 3D 虚拟彩色立方体/四棱柱",
                "多帧时域外参滤波锁定：支持按 [L] 键采集 30 帧静止标靶，输出毫米级空间位姿方差",
                "直观检验空间尺度是否严丝合缝，确认是否存在扭曲、漂移或尺度缩放偏差"
            ],
            inputs=["RealSense 深度相机", "当前生产 tags_map.yaml 或场景地图"],
            outputs=["屏幕实时 AR 渲染显示、时域位姿锁定精度统计"],
            quick_tips="快捷键: [6] 或 [A] 启动 | AR界面中 [L] 启动时域锁定 | [M] 切换模型 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="robot_tag_tracker",
            shortcut="7",
            title="机械臂追踪验证 (Tag ID=2)",
            subtitle="[V] 机械臂实时追踪运动标靶/方向跟随/验收",
            category="B — Tag 标定流水线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/robot_tag_tracker.py"],
            tag_color=COLOR_B,
            summary="【Tag 标定 Step 5 追踪验证】Tag 标定闭环验收：机械臂实时追踪传送带上 Tag ID=2 假芦笋，验证标定精度与方向跟随一致性。",
            details=[
                "高帧率取流，持续检测 Tag ID=2 的 16h5 标靶（假芦笋载体）",
                "跟踪标靶 (X, Y, θ) 位姿，机械臂同步执行追踪运动指令",
                "实时比较视觉解算位姿 ↔ 机械臂编码器反馈，输出毫米级追踪误差",
                "验证 T_cam_to_robot 正确性，确保方向跟随无镜像/翻转偏差"
            ],
            inputs=["RealSense 深度相机 + 生产 tags_map.yaml + 机械臂串口 (COM3)"],
            outputs=["追踪误差实时显示、机械臂运动日志、验收判定"],
            quick_tips="快捷键: [7] 或 [V] 启动 | 传送带上放置 Tag ID=2 | [Q] 退出"
        ),

        # ===== C — 示教标定 (1张，与 Tag 标定并列的独立路线) =====
        ToolCardMeta(
            key_id="hand_eye_calibration",
            shortcut="H",
            title="SCARA 示教标定 (接触式)",
            subtitle="[H] Kabsch/SVD 点对刚体配准/极端无Tag场景",
            category="C — 示教标定 (接触式)",
            is_gui=False,
            command=[sys.executable, "tools/calibration/hand_eye_calibration.py"],
            tag_color=COLOR_C,
            summary="【独立路线 · 示教标定】与 Tag 标定并列的接触式方案：示教 4~6 个物理对应点对，Kabsch/SVD 求解相机→SCARA 变换。",
            details=[
                "经典 Kabsch / Horn / Umeyama SVD 最小二乘刚体配准",
                "操作员示教 N 个点对 (相机坐标 ↔ SCARA 基座坐标, N≥3, 推荐 4~6)",
                "自动计算 R、t 与 RMSE，一键回写 config.yaml T_cam_to_scara",
                "适用：Tag 因反光/遮挡/极端角度无法识别时的保底方案"
            ],
            inputs=["操作员手动示教的点对坐标"],
            outputs=["config.yaml 更新、终端打印 RMSE 与配准质量"],
            quick_tips="快捷键: [H] 启动 (控制台) | 推荐 4~6 个非共面点对"
        ),

        # ===== D — 生产调试 (4张，相机/仿真/抓取/诊断) =====
        ToolCardMeta(
            key_id="d435_live",
            shortcut="8",
            title="RealSense 深度相机与探针",
            subtitle="[D] 物理高帧率取流/毫米级深度探针/单帧快照",
            category="D — 生产调试",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py"],
            tag_color=COLOR_D,
            summary="【现场感知总览】RealSense 物理深度相机的综合查看器与交互式深度测量探针。",
            details=[
                "实时获取 1280x720 RGB 与精准对齐的深度流",
                "鼠标悬停任意像素点，实时探针读取毫米级 (X, Y, Z) 空间坐标",
                "支持深度热力图着色 (JET/TURBO) 与直方图动态均衡增强",
                "按 [S] 键一键保存工业快照 (RGB + Depth + 点云 PLY)"
            ],
            inputs=["Intel RealSense 深度相机 USB 3.0 物理相机"],
            outputs=["data/snapshots/ 单帧高质量工业多模态快照"],
            quick_tips="快捷键: [8] 或 [D] 启动 | 查看器内 [S] 存快照 | [M] 切换热力着色 | [D] 测距探针 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="sim_sandbox",
            shortcut="9",
            title="仿真模拟与离线快照验证",
            subtitle="[M] --mock 纯软件相机仿真 / 历史工业快照位姿解算",
            category="D — 生产调试",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py", "--mock"],
            tag_color=COLOR_D,
            summary="【脱机仿真沙盒】无硬件时的开发与调试利器：涵盖纯软件仿真相机与历史快照抓取算法验证。",
            details=[
                "生成合成渐变深度场与模拟测试 AprilTag 标靶纹理，模拟真实 30FPS 视频流与探针交互",
                "支持算法离线验证：从 data/snapshots/ 快速加载真实历史工业快照，验证芦笋抓取解算",
                "适合在离线工位、出差环境或算法调优期间进行全流程无硬件联调",
                "快照验证命令：python tools/find_top_asparagus.py --snapshot latest"
            ],
            inputs=["纯软件数学合成场 或 data/snapshots/ 历史已采集工业快照"],
            outputs=["模拟工业快照至 data/snapshots/ 或控制台算法解算结果"],
            quick_tips="快捷键: [9] 或 [M] 启动仿真查看器 | 离线快照测试在控制台执行对应参数命令"
        ),

        # ===== D — 生产调试 (续，芦笋抓取) =====
        ToolCardMeta(
            key_id="asparagus_live",
            shortcut="0",
            title="芦笋抓取位姿解算 (实时生产)",
            subtitle="[F] 硬件相机抓拍解算顶层芦笋/输出 G-code",
            category="D — 生产调试",
            is_gui=False,
            command=[sys.executable, "tools/find_top_asparagus.py"],
            tag_color=COLOR_D,
            summary="【核心生产算法】调用物理相机抓拍一帧并解算最上层芦笋空间位姿，输出抓取指令。",
            details=[
                "自动拉起 RealSense 物理相机完成自动曝光对齐与单帧捕获",
                "3D 表面法向量与空间骨架线拟合，精确定位顶层可抓取芦笋",
                "将相机坐标系位姿通过生产标定矩阵转换为 SCARA 机械臂基坐标系",
                "直接生成控制 SCARA 机械臂抓取的标准 G-code 指令与 JSON 协议"
            ],
            inputs=["RealSense 硬件相机", "config/camera_intrinsics.yaml", "config/tags_map.yaml"],
            outputs=["终端打印机械臂 G-code 指令、JSON 抓取坐标与调试渲染图"],
            quick_tips="快捷键: [0] 或 [F] 启动 | 独立控制台视窗执行，打印抓取坐标后按任意键退出。"
        ),

        # ===== D — 生产调试 (续，系统诊断) =====
        ToolCardMeta(
            key_id="sys_diagnose_tests",
            shortcut="-",
            title="系统环境深度诊断与测试套件",
            subtitle="[T] 驱动与依赖诊断 / 85+ 项自动化 CI/CD 全量测试",
            category="D — 生产调试",
            is_gui=False,
            command=[sys.executable, "tools/cli_menu.py", "--diagnose"],
            tag_color=COLOR_D,
            summary="【系统健康与质量守门】全面检查系统环境依赖，并提供工程全量自动化测试套件。",
            details=[
                "全面检查 Python、OpenCV、NumPy C-API 及 RealSense USB 3.0 驱动就绪状态",
                "排查 yaml、PIL、matplotlib、scipy 等工业科学计算包环境版本",
                "全量测试执行命令：python -m unittest discover -s tests -p \"test_*.py\"",
                "涵盖数学平差 (BA)、图论连通拓扑、外参盲测体检与 UI 状态机，保障发布质量"
            ],
            inputs=["系统底层环境注册表与 tests/ 全量测试框架"],
            outputs=["控制台输出清晰的逐项绿勾诊断报告与全工程测试矩阵"],
            quick_tips="快捷键: [T] 启动环境深度诊断 | 遇到红叉时依提示执行 pip 修复命令"
        ),

        ToolCardMeta(
            key_id="pip_install",
            shortcut="P",
            title="安装/更新项目依赖",
            subtitle="[P] pip install -r requirements.txt",
            category="D — 生产调试",
            is_gui=False,
            command=[sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            tag_color=COLOR_D,
            summary="【环境维护】一键安装或更新项目全部 Python 依赖包，确保与 requirements.txt 严格对齐。",
            details=[
                "安装/更新 requirements.txt 中声明的全部依赖",
                "自动处理 numpy、opencv、pyrealsense2、pyyaml 等核心包",
                "离线/网络环境均可运行，失败时控制台会提示缺失源"
            ],
            inputs=["requirements.txt 文件"],
            outputs=["pip 安装进度与版本锁定结果"],
            quick_tips="快捷键: [P] 启动 (控制台) | 首次克隆项目后必执行"
        ),

        ToolCardMeta(
            key_id="open_cmd",
            shortcut="X",
            title="项目根目录命令行 (CMD)",
            subtitle="[X] 打开 CMD，工作目录自动定位到 flux_vision_3d",
            category="D — 生产调试",
            is_gui=False,
            command=["cmd.exe", "/k", "cd /d " + os.path.dirname(os.path.dirname(os.path.abspath(__file__)))],
            tag_color=COLOR_D,
            summary="【开发快速入口】一键打开 CMD 并自动 cd 到项目根目录，省去手动路径粘贴。",
            details=[
                "Windows CMD 自动定位到 flux_vision_3d 项目根目录",
                "保持窗口打开 (/k)，可连续执行 python / pytest / git 等命令",
                "配合 requirements.txt 依赖安装、调试脚本运行等高频操作"
            ],
            inputs=["无"],
            outputs=["新 CMD 窗口，路径已在项目根"],
            quick_tips="快捷键: [X] 启动 | CMD 窗口会保持打开等待输入"
        ),
    ]
    return catalog


class GuiLauncherApp:
    """3D Vision 统一 GUI 控制中心主应用"""

    # 沉稳专业的高级暗色系工业调色板 (克制、低饱和、冷峻科技)
    COLOR_BG = (15, 17, 21)             # 钛黑背景 (#0F1115)
    COLOR_CARD_BG = (22, 26, 33)        # 碳灰底色 (#161A21)
    COLOR_CARD_HOVER = (30, 38, 50)     # 悬停轻提亮
    COLOR_CARD_SEL = (28, 44, 58)       # 选中微蓝底色
    COLOR_BORDER = (38, 46, 58)         # 沉稳边框线
    COLOR_BORDER_HOVER = (0, 220, 180)  # 悬停微光冷青
    COLOR_BORDER_SEL = (0, 240, 200)    # 选中发光青冷线
    COLOR_TEXT_TITLE = (242, 245, 248)  # 纯白冷色
    COLOR_TEXT_SUB = (155, 170, 185)    # 冷银灰副标题
    COLOR_TEXT_MUTED = (115, 130, 145)  # 辅助提示暗灰
    COLOR_ACCENT = (0, 210, 180)        # 科技主强调色 (冰魄冷青)
    COLOR_GOLD = (210, 175, 60)         # 关键生产资产点缀金

    def __init__(self, settings_file: Optional[str] = None):
        self._settings_file = settings_file or GUI_SETTINGS_FILE
        self._is_active = True
        self.canvas_w = 1280
        self.canvas_h = 1000
        # 窗口内部 key 标识使用纯英文，通过 Windows API 设定中文标题杜绝乱码
        self.window_name = "flux_vision_3d_suite_dashboard"
        self._running = True
        self._hwnd = None
        self._last_zoom_action = 0.0
        self._force_ctrl_pressed = False
        self._need_save = False
        self._last_resize_time = 0.0

        self.scene_mgr = CalibrationSceneManager()
        self.tools = build_tools_catalog()
        self.selected_tool_idx = -1    # 初始无选中，键盘/点击才激活焦点
        self.hover_tool_idx = -1

        # 子工具前台运行与暗化挂起态
        self.is_subtool_running: bool = False
        self.running_tool_meta: Optional[ToolCardMeta] = None

        # 视口与真矢量放大镜缩放控制 (基准 1280x830)
        self._base_w   = 1280
        self._base_h   = 1000
        self.scale_pct = 100   # 缩放百分比 (50% ~ 200%)

        # 加载上次记忆的用户偏好设置 (自动恢复缩放与窗口尺寸)
        self._load_settings()

        self.mouse_x = -1
        self.mouse_y = -1
        if self.scale_pct != 100 or self.canvas_w != self._base_w or self.canvas_h != self._base_h:
            self.toast_msg = f"已自动恢复偏好设置：放大镜 {self.scale_pct}%，视窗 {self.canvas_w}×{self.canvas_h} (按 Ctrl+0 可随时复位)"
        else:
            self.toast_msg = "欢迎使用 flux_vision_3d 工业视觉控制中心！按数字键或点击卡片进入工况中枢。"
        self.toast_time = time.time() + 4.5

        # 系统状态缓存
        self.system_status = {}
        self.refresh_system_status()

    def _load_settings(self):
        """从配置文件读取上次记忆的缩放比例与窗口尺寸"""
        target_file = getattr(self, "_settings_file", GUI_SETTINGS_FILE)
        if os.path.exists(target_file):
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "scale_pct" in data:
                    self.scale_pct = max(50, min(200, int(data["scale_pct"])))
                s = self.scale_pct / 100.0
                saved_w = data.get("canvas_w")
                saved_h = data.get("canvas_h")
                if saved_w and saved_h and int(saved_w) >= 480 and int(saved_h) >= 270:
                    self.canvas_w = int(saved_w)
                    self.canvas_h = int(saved_h)
                else:
                    self.canvas_w = max(640, int(self._base_w * s))
                    self.canvas_h = max(360, int(self._base_h * s))
            except Exception:
                pass

    def _save_settings(self):
        """持久化保存当前缩放比例与窗口尺寸到目标配置文件"""
        if not getattr(self, "_is_active", False):
            return
        try:
            target_file = getattr(self, "_settings_file", GUI_SETTINGS_FILE)
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            payload = {
                "scale_pct": self.scale_pct,
                "canvas_w": self.canvas_w,
                "canvas_h": self.canvas_h,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def set_toast(self, msg: str, duration: float = 3.5):
        """设置底部提示消息"""
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def refresh_system_status(self):
        """刷新底层系统与场景状态"""
        try:
            self.scene_mgr.refresh_scenes()
            self.system_status = check_env_status(force_refresh=True)
        except Exception:
            self.system_status = {}

    def _apply_zoom(self, delta_pct: int, reset: bool = False):
        """执行全局真矢量放大镜缩放：卡片尺寸、字号、间距等比矢量缩放，并自动持久化记忆"""
        if reset:
            self.scale_pct = 100
        else:
            self.scale_pct = max(50, min(200, self.scale_pct + delta_pct))

        s = self.scale_pct / 100.0
        rec_w = max(640, int(self._base_w * s))
        rec_h = max(360, int(self._base_h * s))
        self.canvas_w = rec_w
        self.canvas_h = rec_h
        try:
            cv2.resizeWindow(self.window_name, rec_w, rec_h)
        except Exception:
            pass
        self._save_settings()
        self.set_toast(f"矢量放大镜: {self.scale_pct}%  (已自动记忆大小，Ctrl+0 复位)", duration=2.2)

    def _poll_hardware_zoom(self):
        """利用 Win32 原生 GetAsyncKeyState 硬件物理按键探测，彻底绕过中文输入法拦截"""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            u32 = ctypes.windll.user32
            # 当窗口获得焦点时才响应物理热键，防止在其他程序中误触
            fg_hwnd = u32.GetForegroundWindow()
            if self._hwnd and fg_hwnd != self._hwnd:
                return

            now = time.time()
            if now - self._last_zoom_action < 0.18:  # 180ms 防抖冷却，按住平滑递增，单点不跳级
                return

            # 检测 Ctrl 物理按压状态 (VK_CONTROL = 0x11)
            ctrl_pressed = bool(u32.GetAsyncKeyState(0x11) & 0x8000)
            if not ctrl_pressed:
                return

            # 放大: 主键盘 VK_OEM_PLUS (0xBB, 187) 或小键盘 VK_ADD (0x6B, 107)
            zoom_in = bool((u32.GetAsyncKeyState(0xBB) & 0x8000) or (u32.GetAsyncKeyState(0x6B) & 0x8000))
            # 缩小: 主键盘 VK_OEM_MINUS (0xBD, 189) 或小键盘 VK_SUBTRACT (0x6D, 109)
            zoom_out = bool((u32.GetAsyncKeyState(0xBD) & 0x8000) or (u32.GetAsyncKeyState(0x6D) & 0x8000))
            # 复位: 主键盘 '0' (0x30, 48) 或小键盘 '0' (0x60, 96)
            zoom_reset = bool((u32.GetAsyncKeyState(0x30) & 0x8000) or (u32.GetAsyncKeyState(0x60) & 0x8000))

            if zoom_in:
                self._apply_zoom(+10)
                self._last_zoom_action = now
            elif zoom_out:
                self._apply_zoom(-10)
                self._last_zoom_action = now
            elif zoom_reset:
                self._apply_zoom(0, reset=True)
                self._last_zoom_action = now
        except Exception:
            pass

    def _present_canvas(self):
        """在当前物理窗口分辨率下原生呈现矢量画布 (零位图拉伸，零锯齿)"""
        canvas = self._render_canvas()
        cv2.imshow(self.window_name, canvas)

    def run(self):
        """主事件循环 (带 Windows 原生标题 Unicode 注入与全屏真矢量动态排版重绘)"""
        import atexit
        atexit.register(self._save_settings)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_w, self.canvas_h)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        # 解决 Windows 标题栏乱码：使用原生 Win32 Unicode API 注入中文标题
        if sys.platform == "win32":
            try:
                import ctypes
                self._hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
                if self._hwnd:
                    ctypes.windll.user32.SetWindowTextW(self._hwnd, "flux_vision_3d | 3D 视觉综合控制中心 (Suite Dashboard)")
            except Exception:
                self._hwnd = None

        # 首次呈现并确保窗口尺寸精准生效
        self._present_canvas()
        try:
            cv2.resizeWindow(self.window_name, self.canvas_w, self.canvas_h)
        except Exception:
            pass

        while self._running:
            # 0. 窗口关闭检测：若用户直接点击右上角红叉 [X]，安全退出并保存偏好
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            # 1. 硬件级按键轮询 (绕过中文输入法对加减号的拦截)
            self._poll_hardware_zoom()

            # 2. 动态检测窗口实际物理大小 (支持用户手动拖拽拉伸窗口边框，原生矢量重绘)
            rect = cv2.getWindowImageRect(self.window_name)
            if rect and len(rect) >= 4:
                cur_w, cur_h = rect[2], rect[3]
                if cur_w >= 480 and cur_h >= 270:
                    if cur_w != self.canvas_w or cur_h != self.canvas_h:
                        self.canvas_w = cur_w
                        self.canvas_h = cur_h
                        self._need_save = True
                        self._last_resize_time = time.time()

            # 拖拽边框防抖保存：尺寸静止 0.35s 后自动落盘持久化
            if self._need_save and (time.time() - self._last_resize_time > 0.35):
                self._save_settings()
                self._need_save = False

            # 3. 呈现真矢量画布 (无任何 cv2.resize 插值，字形完美)
            self._present_canvas()

            # 4. OpenCV 事件捕获通道
            raw_key = cv2.waitKeyEx(20)
            if raw_key == -1:
                continue

            # 处理退出
            if raw_key in (27, ord('q'), ord('Q')):
                break

            # 键盘选择与启动分发
            self._handle_keyboard(raw_key)

        # 退出前持久化保存最终视口偏好
        self._save_settings()
        cv2.destroyAllWindows()

    def _on_mouse(self, event, x, y, flags, param):
        """鼠标移动、点击与滚轮缩放事件 (与物理坐标 1:1 原生对齐)"""
        if self.is_subtool_running:
            return  # 子应用运行期间，主视窗处于安全挂起待命态，屏蔽一切鼠标操作

        self.mouse_x = x
        self.mouse_y = y

        # ── 1. Ctrl + 鼠标滚轮缩放 (最顺手的放大镜交互) ─────────────────────
        if event == 10:  # cv2.EVENT_MOUSEWHEEL
            ctrl_pressed = False
            if getattr(self, "_force_ctrl_pressed", False):
                ctrl_pressed = True
            elif sys.platform == "win32":
                try:
                    import ctypes
                    ctrl_pressed = bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
                except Exception:
                    pass
            if ctrl_pressed:
                if flags > 0:
                    self._apply_zoom(+10)
                else:
                    self._apply_zoom(-10)
                return

        # 检测鼠标悬停在哪个卡片上（根据当前 scale 坐标直接命中检测）
        card_idx = self._hit_test_cards(x, y)
        self.hover_tool_idx = card_idx

        s = self.scale_pct / 100.0
        # 鼠标左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            # 顶部右上角退出按钮 (自适应右对齐)
            bw = int(125 * s)
            bh = int(34 * s)
            bx = self.canvas_w - bw - int(15 * s)
            by = int(10 * s)
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._running = False
                return

            # 右下角“一键启动当前选中工具”按钮 (动态自适应区域)
            FW = int(370 * s) * 2 + int(12 * s)
            split_x = int(15 * s) + FW + int(15 * s)
            px = split_x + int(15 * s)
            py = int(66 * s)
            pw = max(int(360 * s), self.canvas_w - px - int(20 * s))
            ph = max(int(450 * s), self.canvas_h - int(50 * s) - py - int(15 * s))
            # 点击右侧面板底部启动按钮 —— 已移除按钮，不再需要点击检测

            # 点击左侧卡片
            if card_idx != -1:
                self.selected_tool_idx = card_idx
                self._launch_tool(self.tools[card_idx])
                return
            else:
                # 点击非卡片区域复位选中状态，返回系统大屏总览
                self.selected_tool_idx = -1

        # 鼠标双击直接启动
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if card_idx != -1:
                self.selected_tool_idx = card_idx
                self._launch_tool(self.tools[card_idx])

    def _handle_keyboard(self, raw_key: int):
        """键盘快捷键响应 (6分组: row0 A全宽, row1-3 B 2×3, row4 C 1张, row5 D 2张, row6 E+F 2张)"""
        if self.is_subtool_running:
            return  # 子应用运行期间，主视窗处于安全挂起待命态，屏蔽一切按键操作

        if raw_key == 8:  # Backspace 键复位选中状态
            self.selected_tool_idx = -1
            self.hover_tool_idx = -1
            return

        if raw_key in (13, 10):  # 回车
            if 0 <= self.selected_tool_idx < len(self.tools):
                self._launch_tool(self.tools[self.selected_tool_idx])
            return

        # 方向键：将卡片索引映射到 (row, col) 坐标后导航
        # row 0: idx 0 (A)
        # row 1-3: idx 1-6 (B 2×3)
        # row 4: idx 7 (C, 只有 col 0)
        # row 5-7: idx 8-13 (D 2×3, 全部两列)
        def idx_to_rc(i: int) -> Tuple[int, int]:
            if i < 0:
                return (0, 0)
            if i == 0:
                return (0, 0)
            if 1 <= i <= 6:   # B 区
                b = i - 1
                return (b // 2 + 1, b % 2)
            if i == 7:        # C 区
                return (4, 0)
            if 8 <= i <= 13:  # D 区 2×3
                d = i - 8
                return (d // 2 + 5, d % 2)
            return (7, 0)

        def rc_to_idx(r: int, c: int) -> int:
            if r == 0:
                return 0
            if 1 <= r <= 3:   # B 区
                base_b = (r - 1) * 2
                return min(1 + base_b + c, 13)
            if r == 4:        # C 区只有 col 0
                return 7
            if 5 <= r <= 7:   # D 区
                base_d = (r - 5) * 2
                return min(8 + base_d + c, 13)
            return 13

        row, col = idx_to_rc(self.selected_tool_idx)

        if raw_key in (2490368, 65362, 38):    # 上
            if row > 0:
                row -= 1
                col = 0 if row == 0 else col
                if row == 4 and col >= 1:  # C 区只有 col 0
                    col = 0
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        if raw_key in (2621440, 65364, 40):    # 下
            if row < 7:
                row += 1
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        if raw_key in (2424832, 65361, 37):    # 左
            if row > 0 and col > 0:
                col = 0
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        if raw_key in (2555904, 65363, 39):    # 右
            if row > 0 and col < 1:
                if row != 4:  # C 区只有 col 0
                    col = 1
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        # ── 检测 Ctrl 键状态 (全面兼容 Windows 各种键盘布局与笔记本) ────────────
        ctrl_held = False
        if getattr(self, "_force_ctrl_pressed", False):
            ctrl_held = True
        elif sys.platform == "win32":
            try:
                import ctypes
                u32 = ctypes.windll.user32
                # VK_CONTROL = 0x11, 同步检测当前线程消息队列与物理硬件实时状态
                ctrl_held = bool((u32.GetKeyState(0x11) & 0x8000) or (u32.GetAsyncKeyState(0x11) & 0x8000))
            except Exception:
                pass

        key_byte = (raw_key & 0xFF)
        key_word = (raw_key & 0xFFFF)
        key_char = chr(key_byte).lower() if key_byte < 128 else ""

        # ── 笔记本电脑键盘全面兼容 (支持 =/+ 放大, -/_ 缩小, 0 复位) ───────────────
        # 1. 放大 (Zoom In):
        #    - 字符: '=' (笔记本主键盘等号键未按Shift) 或 '+' (笔记本按了Shift)
        #    - 键码: 主键盘 VK_OEM_PLUS (187) 或小键盘 VK_ADD (107)
        is_zoom_in = (
            key_char in ('=', '+') or
            key_byte in (ord('='), ord('+'), 187, 107) or
            key_word in (ord('='), ord('+'), 187, 107)
        )

        # 2. 缩小 (Zoom Out):
        #    - 字符: '-' (笔记本主键盘减号键未按Shift) 或 '_' (笔记本按了Shift即下划线)
        #    - 键码: 主键盘 VK_OEM_MINUS (189) 或小键盘 VK_SUBTRACT (109) 或 Ctrl+- 控制字符 31
        is_zoom_out = (
            key_char in ('-', '_') or
            key_byte in (ord('-'), ord('_'), 189, 109, 31) or
            key_word in (ord('-'), ord('_'), 189, 109, 31)
        )

        # 3. 复位 100% (Reset):
        #    - 字符: '0'
        #    - 键码: VK_0 (48) 或小键盘 VK_NUMPAD0 (96)
        is_zoom_reset = (
            key_char == '0' or
            key_byte in (ord('0'), 48, 96) or
            key_word in (ord('0'), 48, 96)
        )

        # 触发缩放：按住 Ctrl 组合键，或小键盘独占 +/- 键
        if (ctrl_held and (is_zoom_in or is_zoom_out or is_zoom_reset)) or (key_byte in (107, 109)):
            if is_zoom_in or key_byte == 107:
                self._apply_zoom(+10)
            elif is_zoom_out or key_byte == 109:
                self._apply_zoom(-10)
            elif is_zoom_reset:
                self._apply_zoom(0, reset=True)
            return

        # 若按住 Ctrl 且未命中缩放，拦截避免误触普通单键快捷键
        if ctrl_held:
            return

        # 数字 / 字母快捷键（与新卡片顺序严格对齐）
        key_char = chr(raw_key & 0xFF).lower() if (raw_key & 0xFF) < 128 else ""

        shortcut_map = {
            '1': "scene_hub",            # A
            '2': "tag_generator",        # B Step 0
            '3': "tag_wizard",           # B Step 1
            '4': "tag_studio",           # B Step 2
            '5': "tag_offline_verifier", # B Step 3
            '6': "tag_ar_verifier",      # B Step 4
            '7': "robot_tag_tracker",    # B Step 5 追踪验证
            '8': "d435_live",            # D 感知层
            '9': "sim_sandbox",          # D 感知层
            '0': "asparagus_live",       # E 生产执行
            # 直觉字母快捷键
            'v': "robot_tag_tracker",
            'g': "tag_generator",
            'c': "tag_wizard",
            's': "tag_studio",
            'l': "tag_offline_verifier",
            'a': "tag_ar_verifier",
            'h': "hand_eye_calibration",
            'd': "d435_live",
            'm': "sim_sandbox",
            'f': "asparagus_live",
            't': "sys_diagnose_tests",
            'p': "pip_install",
            'x': "open_cmd",
        }

        if key_char in shortcut_map:
            target_id = shortcut_map[key_char]
            for idx, tool in enumerate(self.tools):
                if tool.key_id == target_id:
                    self.selected_tool_idx = idx
                    self.hover_tool_idx = idx
                    self._launch_tool(tool)
                    return

    def _get_card_rect(self, idx: int) -> Tuple[int, int, int, int]:
        """返回第 idx 张卡片的 (x, y, w, h)，与渲染布局严格保持一致

        布局 (7行，6分组):
          row 0   A 场景总控 (全宽, 1张)
          rows 1-3  B Tag 标定流水线 (2×3 = 6张)
          row 4   C 示教标定 (1张, 左列)
          row 5   D 感知层 (1×2 = 2张)
          row 6   E 生产执行 + F 系统运维 (1×2 = 2张)
        """
        s = self.scale_pct / 100.0
        LH = max(14, int(20 * s))
        CH = max(40, int(70 * s))
        CW = max(200, int(370 * s))
        SX = max(6, int(12 * s))
        SY = max(4, int(8 * s))
        GY = max(16, int(40 * s))
        X0 = max(8, int(15 * s))
        Y0 = max(40, int(66 * s))
        FW = CW * 2 + SX

        if idx == 0:          # A: 顶部全宽
            return X0, Y0 + LH, FW, CH

        if 1 <= idx <= 6:     # B: 2×3 (3行)
            b = idx - 1
            base_y = Y0 + LH + CH + GY + LH
            return X0 + (b % 2) * (CW + SX), base_y + (b // 2) * (CH + SY), CW, CH

        if idx == 7:          # C: 示教标定 (左列)
            base_y = Y0 + LH + CH + GY + LH + 3 * (CH + SY) + GY + LH
            return X0, base_y, CW, CH

        if 8 <= idx <= 13:    # D: 生产调试 (2×3 = 6张)
            d = idx - 8
            base_y = Y0 + LH + CH + GY + LH + 3 * (CH + SY) + GY + LH + CH + GY + LH
            return X0 + (d % 2) * (CW + SX), base_y + (d // 2) * (CH + SY), CW, CH

        return 0, 0, 0, 0

    def _hit_test_cards(self, x: int, y: int) -> int:
        """鼠标命中测试：委托给 _get_card_rect，与渲染位置严格一致"""
        for idx in range(len(self.tools)):
            cx, cy, cw, ch = self._get_card_rect(idx)
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                return idx
        return -1

    def _launch_tool(self, tool: ToolCardMeta):
        """执行启动子工具或测试 (支持全屏暗化蒙版与控制权移交挂起浮岛)"""
        self._save_settings()  # 立即落盘记忆当前大小与比例
        self.is_subtool_running = True
        self.running_tool_meta = tool
        self.set_toast(f"已移交控制权，正在拉起: 【{tool.title}】...")

        # 立即在主窗口渲染暗化挂起画布并强制上屏
        self._present_canvas()
        cv2.waitKey(40)

        cmd = tool.command
        try:
            if tool.is_gui:
                res = subprocess.run(cmd)
                self.set_toast(f"【{tool.title}】已安全返回，控制中心已重新就绪。")
            else:
                if sys.platform == "win32":
                    full_cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd])
                    wrapper_cmd = f'cmd.exe /c "{full_cmd_str} & echo. & echo [完成] 请按任意键返回控制中心... & pause > nul"'
                    res = subprocess.run(wrapper_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
                    self.set_toast(f"【{tool.title}】执行完毕，控制中心已重新就绪。")
                else:
                    res = subprocess.run(cmd)
                    self.set_toast(f"【{tool.title}】执行完毕，控制中心已重新就绪。")
        except Exception as e:
            self.set_toast(f"启动失败: {e}", duration=5.0)
        finally:
            self.is_subtool_running = False
            self.running_tool_meta = None
            self.refresh_system_status()
            self._present_canvas()

    # ========================== 核心渲染逻辑 ==========================

    def _render_canvas(self) -> np.ndarray:
        """渲染完整画布 (在当前窗口实际物理分辨率下进行真·矢量绘制，无任何位图拉伸与锯齿)"""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        s = self.scale_pct / 100.0
        CW = max(200, int(370 * s))
        SX = max(6, int(12 * s))
        X0 = max(8, int(15 * s))
        FW = CW * 2 + SX
        split_x = X0 + FW + max(8, int(15 * s))

        # 1. 顶部监控与状态栏
        self._render_top_bar(canvas)

        # 2. 中间主分割线
        top_h = max(36, int(54 * s))
        footer_h = max(34, int(50 * s))
        cv2.line(canvas, (split_x, top_h), (split_x, self.canvas_h - footer_h), self.COLOR_BORDER, 1)

        # 3. 左侧工具网格区
        self._render_tools_grid(canvas)

        # 4. 右侧实时说明大屏 (有卡片高亮/选中时渲染 Inspector，否则渲染默认环境与硬件总览大屏)
        cur_idx = self.hover_tool_idx if self.hover_tool_idx != -1 else self.selected_tool_idx
        if 0 <= cur_idx < len(self.tools):
            self._render_inspector_panel(canvas, self.tools[cur_idx], split_x)
        else:
            self._render_default_overview_panel(canvas, split_x)

        # 5. 底部状态与快捷键指引栏
        self._render_footer(canvas)

        # 6. 子工具运行挂起态：整体深度暗化并叠加控制权移交模态浮岛
        if self.is_subtool_running and self.running_tool_meta:
            self._render_suspended_modal(canvas, self.running_tool_meta)

        return canvas

    def _render_top_bar(self, canvas: np.ndarray):
        """渲染顶部全局标题与当前生产工况胶囊 (真矢量自适应排布，去除底层环境芯片)"""
        s = self.scale_pct / 100.0
        top_h = max(36, int(54 * s))
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, top_h), (17, 20, 26), -1)
        cv2.line(canvas, (0, top_h), (self.canvas_w, top_h), self.COLOR_BORDER, 1)

        # 标题与微光状态灯
        c_x, c_y = max(12, int(24 * s)), top_h // 2
        cv2.circle(canvas, (c_x, c_y), max(2, int(5 * s)), self.COLOR_ACCENT, -1)
        cv2.circle(canvas, (c_x, c_y), max(4, int(8 * s)), self.COLOR_ACCENT, 1)
        draw_text(canvas, "FLUX VISION 3D", (max(20, int(40 * s)), max(5, int(10 * s))),
                  font_size=max(11, int(15 * s)), color=self.COLOR_ACCENT, bold=True)
        draw_text(canvas, "工业视觉综合控制中心", (max(20, int(40 * s)), max(18, int(28 * s))),
                  font_size=max(9, int(12 * s)), color=(150, 170, 185))

        # 当前活动生产场景胶囊 (紧随标题之后，居中/醒目呈现)
        act_sc = self.scene_mgr.get_active_scene()
        capsule_x = max(200, int(330 * s))
        bw = max(80, int(125 * s))
        bh = max(24, int(34 * s))
        bx = self.canvas_w - bw - max(8, int(15 * s))
        by = max(6, int(10 * s))

        capsule_w = max(int(180 * s), min(int(460 * s), bx - capsule_x - max(12, int(20 * s))))
        if capsule_w > max(120, int(160 * s)):
            cap_y1, cap_y2 = max(6, int(10 * s)), max(26, int(44 * s))
            if act_sc:
                status_tag = "★ 生产环境" if act_sc.is_published else ("已平差" if act_sc.ba_solved else "沙盒草稿")
                tag_col = self.COLOR_GOLD if act_sc.is_published else ((0, 210, 160) if act_sc.ba_solved else (135, 165, 195))
                bg_col = (26, 28, 38) if act_sc.is_published else (20, 26, 34)
                border_col = (85, 70, 30) if act_sc.is_published else (45, 60, 78)
                cv2.rectangle(canvas, (capsule_x, cap_y1), (capsule_x + capsule_w, cap_y2), bg_col, -1)
                cv2.rectangle(canvas, (capsule_x, cap_y1), (capsule_x + capsule_w, cap_y2), border_col, 1)
                cv2.circle(canvas, (capsule_x + max(8, int(14 * s)), (cap_y1 + cap_y2) // 2), max(2, int(4 * s)), tag_col, -1)
                sc_title = f"当前工况: 【{act_sc.name}】 ({status_tag})"
                draw_text(canvas, sc_title, (capsule_x + max(14, int(24 * s)), max(8, int(17 * s))),
                          font_size=max(10, int(13 * s)), color=tag_col, bold=True)
            else:
                cv2.rectangle(canvas, (capsule_x, cap_y1), (capsule_x + capsule_w, cap_y2), (20, 24, 30), -1)
                cv2.rectangle(canvas, (capsule_x, cap_y1), (capsule_x + capsule_w, cap_y2), (38, 46, 56), 1)
                cv2.circle(canvas, (capsule_x + max(8, int(14 * s)), (cap_y1 + cap_y2) // 2), max(2, int(4 * s)), (120, 130, 140), -1)
                draw_text(canvas, "当前工况: 【未选定场景】 (请进入场景总控选择)", (capsule_x + max(14, int(24 * s)), max(8, int(17 * s))),
                          font_size=max(10, int(13 * s)), color=(140, 150, 160))

        # 右上角 [X] 退出按钮 (自适应靠右)
        is_hover_exit = (bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh)
        exit_bg = (48, 22, 24) if is_hover_exit else (32, 20, 22)
        exit_border = (210, 60, 60) if is_hover_exit else (95, 36, 40)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_bg, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_border, 2 if is_hover_exit else 1)
        draw_text(canvas, "[X] 退出 [ESC]", (bx + max(6, int(12 * s)), by + max(4, int(9 * s))),
                  font_size=max(10, int(13 * s)), color=(220, 170, 170), bold=True)

    def _render_tools_grid(self, canvas: np.ndarray):
        """渲染左侧工具卡片网格 (真矢量自适应缩放，无杂乱左侧竖线)"""
        s = self.scale_pct / 100.0
        LH = max(14, int(20 * s))
        CW = max(200, int(370 * s))
        SX = max(6, int(12 * s))
        X0 = max(8, int(15 * s))
        FW = CW * 2 + SX

        HEADER_TEXT = (192, 206, 222)
        group_headers = [
            (self._get_card_rect(0)[1] - LH,  FW, "A  场景总控",                        (195, 155,  45)),
            (self._get_card_rect(1)[1] - LH,  FW, "B  Tag 标定流水线 (AprilTag)",        ( 65, 175, 160)),
            (self._get_card_rect(7)[1] - LH,  CW, "C  示教标定 (接触式 · SVD)",          (220, 145,  60)),
            (self._get_card_rect(8)[1] - LH,  FW, "D  生产调试 (感知/仿真/抓取/诊断)",   ( 90, 140, 195)),
        ]
        for hy, hw, label, accent in group_headers:
            cv2.rectangle(canvas, (X0, hy), (X0 + hw, hy + LH - max(1, int(2 * s))), (18, 22, 30), -1)
            cv2.rectangle(canvas, (X0, hy), (X0 + max(2, int(4 * s)), hy + LH - max(1, int(2 * s))), accent, -1)
            draw_text(canvas, label, (X0 + max(6, int(10 * s)), hy + max(2, int(3 * s))),
                      font_size=max(9, int(12 * s)), color=HEADER_TEXT, bold=False)

        # 卡片渲染
        for idx, tool in enumerate(self.tools):
            cx, cy, cw, ch = self._get_card_rect(idx)

            is_selected = (self.selected_tool_idx >= 0 and idx == self.selected_tool_idx)
            is_hover    = (idx == self.hover_tool_idx)

            if is_selected:
                card_bg, card_border, border_th = self.COLOR_CARD_SEL, self.COLOR_BORDER_SEL, 2
            elif is_hover:
                card_bg, card_border, border_th = self.COLOR_CARD_HOVER, self.COLOR_BORDER_HOVER, 1
            else:
                card_bg, card_border, border_th = self.COLOR_CARD_BG, self.COLOR_BORDER, 1

            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), card_bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), card_border, border_th)

            # 快捷键徽章
            badge_w = max(18, int(28 * s))
            badge_h = max(14, int(20 * s))
            cv2.rectangle(canvas, (cx + max(6, int(12 * s)), cy + max(6, int(10 * s))),
                          (cx + max(6, int(12 * s)) + badge_w, cy + max(6, int(10 * s)) + badge_h), (16, 20, 26), -1)
            cv2.rectangle(canvas, (cx + max(6, int(12 * s)), cy + max(6, int(10 * s))),
                          (cx + max(6, int(12 * s)) + badge_w, cy + max(6, int(10 * s)) + badge_h), (45, 58, 72), 1)
            draw_text(canvas, tool.shortcut, (cx + max(8, int(16 * s)), cy + max(6, int(12 * s))),
                      font_size=max(9, int(12 * s)), color=self.COLOR_TEXT_TITLE, bold=True)

            # 标题与副标题
            title_col = self.COLOR_TEXT_TITLE if (is_selected or is_hover) else (205, 215, 225)
            draw_text(canvas, tool.title, (cx + max(10, int(18 * s)) + badge_w, cy + max(6, int(10 * s))),
                      font_size=max(10, int(14 * s)), color=title_col, bold=True)

            sub_col = (170, 185, 195) if is_hover else self.COLOR_TEXT_MUTED
            max_sub = 52 if cw > int(500 * s) else 24
            draw_text(canvas, tool.subtitle[:max_sub], (cx + max(8, int(14 * s)), cy + max(20, int(36 * s))),
                      font_size=max(9, int(12 * s)), color=sub_col)

            # 运行模式微标
            mode_text = "GUI" if tool.is_gui else "CMD"
            mode_color = (0, 190, 150) if tool.is_gui else (135, 150, 170)
            cv2.putText(canvas, mode_text, (cx + cw - max(30, int(45 * s)), cy + max(14, int(24 * s))),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.24, 0.32 * s), mode_color, 1, cv2.LINE_AA)

    def _render_inspector_panel(self, canvas: np.ndarray, tool: ToolCardMeta, split_x: int):
        """渲染右侧动态即时说明大屏 (自适应全宽与全高，1:1 矢量清晰无模糊，全文本自适应折行)"""
        s = self.scale_pct / 100.0
        px = split_x + max(8, int(15 * s))
        py = max(40, int(66 * s))
        pw = max(int(360 * s), self.canvas_w - px - max(10, int(20 * s)))
        ph = max(int(450 * s), self.canvas_h - max(30, int(50 * s)) - py - max(8, int(15 * s)))

        # 大屏底板
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 22, 28), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (34, 44, 58), 1)

        # 快捷键徽章与主标题（无装饰条，直接贴顶）
        b_w, b_h = max(24, int(36 * s)), max(20, int(32 * s))
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), (14, 18, 24), -1)
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), tool.tag_color, 1)
        draw_text(canvas, f"[{tool.shortcut}]", (px + max(10, int(20 * s)), py + max(8, int(18 * s))),
                  font_size=max(10, int(14 * s)), color=tool.tag_color, bold=True)

        draw_text(canvas, tool.title, (px + max(36, int(62 * s)), py + max(4, int(10 * s))),
                  font_size=max(12, int(18 * s)), color=self.COLOR_TEXT_TITLE, bold=True)
        mode_str = "原生 GUI 视窗" if tool.is_gui else "控制台"
        draw_text(canvas, f"{tool.category}  |  {mode_str}",
                  (px + max(36, int(62 * s)), py + max(20, int(34 * s))), font_size=max(9, int(12 * s)), color=self.COLOR_TEXT_SUB)

        avail_w = pw - max(24, int(42 * s))
        text_x = px + max(10, int(20 * s))
        curr_y = py + max(44, int(64 * s))
        avail_bottom_y = py + ph - max(8, int(12 * s))

        # 1. 核心概述 (Summary) - 支持自适应折行
        draw_text(canvas, "【功能定位与现场痛点】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(23 * s))
        curr_y = draw_multiline_text(canvas, tool.summary, (text_x, curr_y),
                                     max_width=avail_w, font_size=max(10, int(13 * s)),
                                     color=(215, 225, 235), line_spacing=max(3, int(5 * s)))
        curr_y += max(12, int(16 * s))

        # 2. 详细特性清单 (Details) - 支持每项条目自适应折行
        draw_text(canvas, "【工程要点与执行逻辑】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(23 * s))
        bullet_icon_x = px + max(12, int(22 * s))
        bullet_text_x = px + max(20, int(34 * s))
        bullet_w = pw - (bullet_text_x - px) - max(12, int(20 * s))

        for d in tool.details:
            if curr_y > avail_bottom_y - max(30, int(50 * s)):
                break
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 190, 160), -1)
            curr_y = draw_multiline_text(canvas, d, (bullet_text_x, curr_y),
                                         max_width=bullet_w, font_size=max(9, int(12 * s)),
                                         color=(195, 208, 220), line_spacing=max(2, int(4 * s)))
            curr_y += max(3, int(5 * s))
        curr_y += max(4, int(6 * s))

        # 3. 前置依赖与输入 (Inputs)
        if curr_y < avail_bottom_y - max(60, int(90 * s)):
            draw_text(canvas, "【前置条件与输入依赖】", (px + max(8, int(16 * s)), curr_y),
                      font_size=max(10, int(13 * s)), color=(140, 180, 220), bold=True)
            curr_y += max(15, int(21 * s))
            for inp in tool.inputs:
                if curr_y > avail_bottom_y - max(40, int(60 * s)):
                    break
                cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
                curr_y = draw_multiline_text(canvas, inp, (bullet_text_x, curr_y),
                                             max_width=bullet_w, font_size=max(9, int(12 * s)),
                                             color=(185, 200, 215), line_spacing=max(2, int(4 * s)))
                curr_y += max(3, int(5 * s))
            curr_y += max(4, int(6 * s))

        # 4. 输出产物 (Outputs)
        if curr_y < avail_bottom_y - max(50, int(70 * s)):
            draw_text(canvas, "【输出产物与持久化路径】", (px + max(8, int(16 * s)), curr_y),
                      font_size=max(10, int(13 * s)), color=(120, 200, 180), bold=True)
            curr_y += max(15, int(21 * s))
            for out in tool.outputs:
                if curr_y > avail_bottom_y - max(30, int(45 * s)):
                    break
                cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (100, 180, 160), -1)
                curr_y = draw_multiline_text(canvas, out, (bullet_text_x, curr_y),
                                             max_width=bullet_w, font_size=max(9, int(12 * s)),
                                             color=(180, 215, 205), line_spacing=max(2, int(4 * s)))
                curr_y += max(3, int(5 * s))
            curr_y += max(4, int(6 * s))

        # 5. 操作提示 (Quick Tips) - 动态自适应卡片框
        if curr_y < avail_bottom_y - max(20, int(30 * s)):
            tip_font_size = max(9, int(12 * s))
            tip_w = pw - max(16, int(32 * s))
            tip_lines = wrap_text_by_width(tool.quick_tips, tip_font_size, tip_w - max(16, int(24 * s)))
            line_h = tip_font_size + max(2, int(4 * s))
            tip_h = len(tip_lines) * line_h + max(8, int(12 * s))

            if curr_y + tip_h <= avail_bottom_y:
                tip_box_x = px + max(8, int(16 * s))
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + tip_h), (22, 28, 36), -1)
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + tip_h), (36, 48, 62), 1)
                draw_multiline_text(canvas, tool.quick_tips, (tip_box_x + max(8, int(12 * s)), curr_y + max(4, int(6 * s))),
                                    max_width=tip_w - max(16, int(24 * s)), font_size=tip_font_size,
                                    color=self.COLOR_TEXT_SUB, line_spacing=max(2, int(4 * s)))

    def _render_default_overview_panel(self, canvas: np.ndarray, split_x: int):
        """当焦点未在任何工具卡片上时，在右侧渲染系统环境、库依赖与硬件健康状态大屏总览"""
        s = self.scale_pct / 100.0
        px = split_x + max(8, int(15 * s))
        py = max(40, int(66 * s))
        pw = max(int(360 * s), self.canvas_w - px - max(10, int(20 * s)))
        ph = max(int(450 * s), self.canvas_h - max(30, int(50 * s)) - py - max(8, int(15 * s)))

        # 大屏底板
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 22, 28), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (34, 44, 58), 1)

        # 标题徽标与文字（无装饰条，直接贴顶）
        b_w, b_h = max(24, int(36 * s)), max(20, int(32 * s))
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), (14, 18, 24), -1)
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), self.COLOR_ACCENT, 1)
        draw_text(canvas, "[i]", (px + max(12, int(24 * s)), py + max(8, int(18 * s))),
                  font_size=max(10, int(14 * s)), color=self.COLOR_ACCENT, bold=True)

        draw_text(canvas, "系统运行环境与硬件健康总览", (px + max(36, int(62 * s)), py + max(4, int(10 * s))),
                  font_size=max(12, int(18 * s)), color=self.COLOR_TEXT_TITLE, bold=True)

        # 数据提取
        st = self.system_status
        py_ver = sys.version.split()[0]
        cv_ver = st.get("cv_version", cv2.__version__)
        rs_tuple = st.get("realsense", (False, "未检测", False))
        rs_ok = rs_tuple[2] if len(rs_tuple) > 2 else False
        rs_msg = rs_tuple[1] if len(rs_tuple) > 1 else str(rs_tuple)
        snaps_c = st.get("snapshot_count", 0)
        act_sc = self.scene_mgr.get_active_scene()

        # 正文排版
        curr_y = py + max(44, int(64 * s))
        avail_bottom_y = py + ph - max(8, int(12 * s))
        bullet_icon_x = px + max(12, int(22 * s))
        bullet_text_x = px + max(22, int(36 * s))
        bullet_w = pw - (bullet_text_x - px) - max(12, int(20 * s))

        # ── 模块 1：底层环境与科学计算库依赖 ──────────────────────────────
        draw_text(canvas, "【底层运行环境与科学计算依赖】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(24 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 210, 160), -1)
        curr_y = draw_multiline_text(canvas, f"Python 解释器: v{py_ver}  ({sys.executable})",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(210, 225, 238), line_spacing=max(2, int(4 * s)))
        curr_y += max(3, int(5 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 210, 160), -1)
        curr_y = draw_multiline_text(canvas, f"核心视觉库: OpenCV v{cv_ver}  |  矩阵计算: NumPy v{np.__version__}",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(200, 215, 228), line_spacing=max(2, int(4 * s)))
        curr_y += max(3, int(5 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 210, 160), -1)
        curr_y = draw_multiline_text(canvas, f"工程工作空间: {PROJECT_ROOT}",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(170, 185, 200), line_spacing=max(2, int(4 * s)))
        curr_y += max(12, int(16 * s))

        # ── 模块 2：感知硬件与数据资产 ────────────────────────────────────
        draw_text(canvas, "【感知层硬件与数据资产】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=(140, 180, 220), bold=True)
        curr_y += max(16, int(24 * s))

        rs_dot_col = (0, 210, 160) if rs_ok else (135, 145, 160)
        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), rs_dot_col, -1)
        curr_y = draw_multiline_text(canvas, f"RealSense 深度相机: {rs_msg}",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(210, 225, 240) if rs_ok else (160, 175, 185),
                                     line_spacing=max(2, int(4 * s)))
        curr_y += max(3, int(5 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
        curr_y = draw_multiline_text(canvas, f"现场工业快照库: 已归档 {snaps_c} 帧 (RGB+Depth+PLY，位于 data/snapshots/)",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(185, 200, 215), line_spacing=max(2, int(4 * s)))
        curr_y += max(12, int(16 * s))

        # ── 模块 3：当前生产场景与三维标靶地图 ──────────────────────────────
        draw_text(canvas, "【当前生产场景与标靶地图资产】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_GOLD, bold=True)
        curr_y += max(16, int(24 * s))

        if act_sc:
            status_tag = "★ 生产环境 (已正式发布)" if act_sc.is_published else ("已求解全局平差 (BA Solved)" if act_sc.ba_solved else "沙盒草稿 (Sandbox)")
            sc_color = self.COLOR_GOLD if act_sc.is_published else ((0, 210, 160) if act_sc.ba_solved else (160, 180, 200))
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), sc_color, -1)
            curr_y = draw_multiline_text(canvas, f"活动工况场景: 【{act_sc.name}】 ({status_tag})",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=sc_color, bold=True,
                                         line_spacing=max(2, int(4 * s)))
            curr_y += max(3, int(5 * s))

            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (180, 160, 100), -1)
            curr_y = draw_multiline_text(canvas, f"场景标定样本: 已采集 {act_sc.image_count} 帧原始图集",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=(190, 205, 220), line_spacing=max(2, int(4 * s)))
            curr_y += max(3, int(5 * s))

            map_status = "已生成 tags_map.yaml (坐标系对齐已锁定)" if act_sc.ba_solved else "未求解 (需执行两阶段平差)"
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (180, 160, 100), -1)
            curr_y = draw_multiline_text(canvas, f"标靶地图状态: {map_status}",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=(190, 205, 220), line_spacing=max(2, int(4 * s)))
        else:
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (120, 130, 140), -1)
            curr_y = draw_multiline_text(canvas, "活动工况场景: 未选定场景 (请点击 [1] 场景总控中心新建或切换)",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=(150, 160, 170), line_spacing=max(2, int(4 * s)))
        curr_y += max(12, int(16 * s))

        # ── 模块 4：控制中心快捷操作指南 ──────────────────────────────────
        if curr_y < avail_bottom_y - max(20, int(30 * s)):
            tip_w = pw - max(16, int(32 * s))
            guide_text = "操作小贴士: 鼠标悬停左侧任意卡片即可即时查阅该模块的工程定位与执行逻辑；按键盘数字键或双击卡片直接拉起对应工具。"
            tip_font_size = max(9, int(12 * s))
            tip_lines = wrap_text_by_width(guide_text, tip_font_size, tip_w - max(16, int(24 * s)))
            line_h = tip_font_size + max(2, int(4 * s))
            box_h = len(tip_lines) * line_h + max(8, int(12 * s))
            if curr_y + box_h <= avail_bottom_y:
                tip_box_x = px + max(8, int(16 * s))
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + box_h), (22, 28, 36), -1)
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + box_h), (36, 48, 62), 1)
                draw_multiline_text(canvas, guide_text, (tip_box_x + max(8, int(12 * s)), curr_y + max(4, int(6 * s))),
                                    max_width=tip_w - max(16, int(24 * s)), font_size=tip_font_size,
                                    color=self.COLOR_TEXT_SUB, line_spacing=max(2, int(4 * s)))

    def _render_footer(self, canvas: np.ndarray):
        """渲染底部状态反馈与快捷键指引栏 (自适应贴底)"""
        s = self.scale_pct / 100.0
        footer_h = max(34, int(50 * s))
        fy = self.canvas_h - footer_h
        cv2.rectangle(canvas, (0, fy), (self.canvas_w, self.canvas_h), (13, 15, 19), -1)
        cv2.line(canvas, (0, fy), (self.canvas_w, fy), self.COLOR_BORDER, 1)

        # 右侧公司/版权标识 (右对齐)
        f_size = max(9, int(13 * s))
        corp_text = "山东卷积分公司 · 2026 年 9 月"
        text_w = max(200, int(360 * s))
        text_x = max(int(500 * s), self.canvas_w - text_w)
        draw_text(canvas, corp_text, (text_x, fy + max(10, int(16 * s))),
                  font_size=f_size, color=self.COLOR_TEXT_MUTED)

    def _render_suspended_modal(self, canvas: np.ndarray, tool: ToolCardMeta):
        """当子工具/控制台在前台运行时，将主界面整体冷黑深度暗化并呈现挂起模态提示框"""
        s = self.scale_pct / 100.0

        # 1. 全局画面深度暗化 (降至 ~20% 亮度，呈现沉静只读休眠态)
        canvas[:] = (canvas.astype(np.float32) * 0.20).astype(np.uint8)

        # 2. 居中模态卡片几何尺寸
        cx, cy = self.canvas_w // 2, self.canvas_h // 2
        mw = max(int(460 * s), min(int(650 * s), self.canvas_w - 40))
        mh = max(int(170 * s), min(int(230 * s), self.canvas_h - 40))
        x1 = cx - mw // 2
        y1 = cy - mh // 2
        x2 = x1 + mw
        y2 = y1 + mh

        # 模态浮岛背景与外阴影/双层边框
        cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (10, 14, 20), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 26, 36), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 170), 2)

        # 顶栏装饰条 (冷青指示带)
        head_h = max(28, int(42 * s))
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + head_h), (26, 34, 48), -1)
        cv2.line(canvas, (x1, y1 + head_h), (x2, y1 + head_h), (45, 65, 90), 1)

        # 顶栏标题
        cv2.circle(canvas, (x1 + max(12, int(20 * s)), y1 + head_h // 2), max(3, int(5 * s)), (0, 220, 180), -1)
        draw_text(canvas, "控制权已移交 · 控制中心安全待命挂起", (x1 + max(22, int(34 * s)), y1 + max(6, int(10 * s))),
                  font_size=max(11, int(15 * s)), color=(0, 230, 190), bold=True)

        # 正文内容排版
        content_y = y1 + head_h + max(12, int(16 * s))
        text_x = x1 + max(16, int(24 * s))

        # 当前运行子工具
        mode_label = "独立 GUI 视窗" if tool.is_gui else "交互式控制台"
        running_title = f"当前前台运行: 【{tool.title}】 ({mode_label})"
        draw_text(canvas, running_title, (text_x, content_y),
                  font_size=max(11, int(15 * s)), color=self.COLOR_TEXT_TITLE, bold=True)
        content_y += max(20, int(28 * s))

        # 说明项 1：安全待命
        cv2.circle(canvas, (text_x + max(4, int(6 * s)), content_y + max(6, int(8 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
        draw_text(canvas, "主视窗已进入后台只读待命模式，已自动屏蔽鼠标与键盘交互。",
                  (text_x + max(12, int(18 * s)), content_y), font_size=max(9, int(12 * s)), color=(185, 200, 215))
        content_y += max(16, int(22 * s))

        # 说明项 2：唤醒指引
        cv2.circle(canvas, (text_x + max(4, int(6 * s)), content_y + max(6, int(8 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
        draw_text(canvas, "请在前台子应用中完成操作；关闭子视窗后，控制中心将自动唤醒并刷新状态。",
                  (text_x + max(12, int(18 * s)), content_y), font_size=max(9, int(12 * s)), color=(185, 200, 215))

        # 底部提示小胶囊
        bot_bar_h = max(24, int(32 * s))
        bot_y1 = y2 - bot_bar_h - max(6, int(10 * s))
        cv2.rectangle(canvas, (text_x, bot_y1), (x2 - max(16, int(24 * s)), bot_y1 + bot_bar_h), (14, 18, 24), -1)
        cv2.rectangle(canvas, (text_x, bot_y1), (x2 - max(16, int(24 * s)), bot_y1 + bot_bar_h), (35, 48, 65), 1)
        draw_text(canvas, "状态: 独占通道就绪 · 等待前台子应用退出信号...",
                  (text_x + max(10, int(16 * s)), bot_y1 + max(5, int(8 * s))),
                  font_size=max(8, int(11 * s)), color=(140, 160, 180))


def main():
    parser = argparse.ArgumentParser(description="3D 视觉综合控制中心 (Suite Dashboard)")
    args = parser.parse_args()

    app = GuiLauncherApp()
    app.run()


if __name__ == "__main__":
    main()
