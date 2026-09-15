#!/usr/bin/env python3
"""
3D 视觉综合控制中心 (3D Vision Suite GUI Launcher)
=================================================
基于 1280x720 工业科技大屏，统一调度 flux_vision_3d 视觉系统的所有核心应用：
- 原生 Windows Unicode 窗口标题，杜绝任何乱码
- 窗口拖拽缩放自适应动态分辨率 (Dynamic Native Canvas)，文字 1:1 矢量超采样，无任何拉伸锯齿
- 高端工业冷峻暗色系配色（低饱和度科技冷蓝、工业深青与钛银灰），视觉沉稳专业不刺眼
- 左侧模块化卡片：【核心生产与工况】、【视觉标定与建图流水线】、【自动化测试与系统运维】
- 右侧动态即时说明大屏 (Live Inspector)：自适应扩展宽度，提供详尽的现场工程指南
"""

import os
import sys
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

from src.calibration.scene_manager import CalibrationSceneManager
from tools.env_utils import check_env_status

# 字体缓存
_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}


def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], font_size: int = 16,
              color: Tuple[int, int, int] = (240, 240, 240), bold: bool = False):
    """在 OpenCV BGR 图像上绘制高质量抗锯齿矢量文本 (支持中文)"""
    if not text:
        return
    x, y = pos
    if x >= img.shape[1] or y >= img.shape[0]:
        return

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

    font = _FONT_CACHE[key]
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
    """构建全系统核心工具目录：9 张卡片，五大功能分组 (A场景 → B标定链 → C感知层 → D生产执行 → E系统运维)"""

    COLOR_A = (195, 155, 45)   # A 场景总控  : 琥珀金 (Amber)
    COLOR_B = (65,  175, 160)  # B 标定流水线: 精密工业深青 (Teal)
    COLOR_C = (90,  140, 195)  # C 感知层    : 钢蓝 (Steel Blue)
    COLOR_D = (80,  190, 115)  # D 生产执行  : 活力绿 (Production Green)
    COLOR_E = (130, 145, 165)  # E 系统运维  : 沉稳钛银灰 (Titanium Gray)

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
            quick_tips="快捷键: [1] 启动 | 中枢内 [↑/↓] 选场景 | [⏎] 激活 | [P] 发布生产 | [S] 进Studio"
        ),

        # ===== B — 标定流水线 (4张，2×2) =====
        ToolCardMeta(
            key_id="tag_wizard",
            shortcut="2",
            title="多视角采图向导 (Wizard)",
            subtitle="[C] 角度雷达交互指引/空格极速连拍/自动归档沙盒",
            category="B — 标定流水线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_capture_wizard.py"],
            tag_color=COLOR_B,
            summary="【现场采图助手】专职采图向导：交互式指导相机移动至不同高度与俯仰角，高效采集高质量标定样本。",
            details=[
                "提供雷达式多视角视盘指引（俯视、大俯仰角、大滚转角、高低多层位态）",
                "按 [空格键] 极速无感连拍，样本自动存入当前场景 raw_images/ 目录",
                "实时 AprilTag 检出回显与白闪快门反馈，采图完毕后返回主中枢自动热重载"
            ],
            inputs=["RealSense D435 相机 (或 --mock 仿真)"],
            outputs=["当前活动场景 raw_images/view_*.png 原始高质量未压缩图集"],
            quick_tips="快捷键: [2] 或 [C] 启动 | 采图界面中 [空格] 拍摄归档 | [R] 重置批次 | [ESC] 完成返回"
        ),

        ToolCardMeta(
            key_id="tag_studio",
            shortcut="3",
            title="离线标定工作站 (Studio)",
            subtitle="[S] 多视角审核/两阶段 BA 平差/智能剪枝/质检闭环",
            category="B — 标定流水线",
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
            quick_tips="快捷键: [3] 或 [S] 启动 | 工作站内 [⏎] 快速求解 | [P] 智能剪枝 | [E] 超精提取 | [R] 导出报告"
        ),

        ToolCardMeta(
            key_id="tag_offline_verifier",
            shortcut="4",
            title="离线精度体检台 (LOO盲测)",
            subtitle="[L] 标定后留一交叉验证/双棱柱对比/残差评级",
            category="B — 标定流水线",
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
            quick_tips="快捷键: [4] 或 [L] 启动 | 体检界面中 [N/P] 切换盲测帧 | [R] 导出评估报告 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="tag_ar_verifier",
            shortcut="5",
            title="在线 AR 虚实融合验收系统",
            subtitle="[A] 3D轴网虚实融合/时域外参滤波锁定/现场验收",
            category="B — 标定流水线",
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
            inputs=["D435 实时相机", "当前生产 tags_map.yaml 或场景地图"],
            outputs=["屏幕实时 AR 渲染显示、时域位姿锁定精度统计"],
            quick_tips="快捷键: [5] 或 [A] 启动 | AR界面中 [L] 启动时域锁定 | [M] 切换模型 | [ESC] 退出"
        ),

        # ===== C — 感知层 (2张，实/虚镜像对) =====
        ToolCardMeta(
            key_id="d435_live",
            shortcut="6",
            title="D435 实时相机与深度探针",
            subtitle="[D] 物理高帧率取流/毫米级深度探针/单帧快照",
            category="C — 感知层",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py"],
            tag_color=COLOR_C,
            summary="【现场感知总览】RealSense D435 物理相机的综合查看器与交互式深度测量探针。",
            details=[
                "实时获取 1280x720 RGB 与精准对齐的深度流",
                "鼠标悬停任意像素点，实时探针读取毫米级 (X, Y, Z) 空间坐标",
                "支持深度热力图着色 (JET/TURBO) 与直方图动态均衡增强",
                "按 [S] 键一键保存工业快照 (RGB + Depth + 点云 PLY)"
            ],
            inputs=["Intel RealSense D435 USB 3.0 物理相机"],
            outputs=["data/snapshots/ 单帧高质量工业多模态快照"],
            quick_tips="快捷键: [6] 或 [D] 启动 | 查看器内 [S] 存快照 | [M] 切换热力着色 | [D] 测距探针 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="sim_sandbox",
            shortcut="7",
            title="仿真模拟与离线快照验证",
            subtitle="[M] --mock 纯软件相机仿真 / 历史工业快照位姿解算",
            category="C — 感知层",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py", "--mock"],
            tag_color=COLOR_C,
            summary="【脱机仿真沙盒】无硬件时的开发与调试利器：涵盖纯软件仿真相机与历史快照抓取算法验证。",
            details=[
                "生成合成渐变深度场与模拟测试 AprilTag 标靶纹理，模拟真实 30FPS 视频流与探针交互",
                "支持算法离线验证：从 data/snapshots/ 快速加载真实历史工业快照，验证芦笋抓取解算",
                "适合在离线工位、出差环境或算法调优期间进行全流程无硬件联调",
                "快照验证命令：python tools/find_top_asparagus.py --snapshot latest"
            ],
            inputs=["纯软件数学合成场 或 data/snapshots/ 历史已采集工业快照"],
            outputs=["模拟工业快照至 data/snapshots/ 或控制台算法解算结果"],
            quick_tips="快捷键: [7] 或 [M] 启动仿真查看器 | 离线快照测试在控制台执行对应参数命令"
        ),

        # ===== D — 生产执行 (1张) =====
        ToolCardMeta(
            key_id="asparagus_live",
            shortcut="8",
            title="芦笋抓取位姿解算 (实时生产)",
            subtitle="[F] 硬件相机抓拍解算顶层芦笋/输出 G-code",
            category="D — 生产执行",
            is_gui=False,
            command=[sys.executable, "tools/find_top_asparagus.py"],
            tag_color=COLOR_D,
            summary="【核心生产算法】调用物理相机抓拍一帧并解算最上层芦笋空间位姿，输出抓取指令。",
            details=[
                "自动拉起 D435 物理相机完成自动曝光对齐与单帧捕获",
                "3D 表面法向量与空间骨架线拟合，精确定位顶层可抓取芦笋",
                "将相机坐标系位姿通过生产标定矩阵转换为 SCARA 机械臂基坐标系",
                "直接生成控制 SCARA 机械臂抓取的标准 G-code 指令与 JSON 协议"
            ],
            inputs=["D435 硬件相机", "config/camera_intrinsics.yaml", "config/tags_map.yaml"],
            outputs=["终端打印机械臂 G-code 指令、JSON 抓取坐标与调试渲染图"],
            quick_tips="快捷键: [8] 或 [F] 启动 | 独立控制台视窗执行，打印抓取坐标后按任意键退出。"
        ),

        # ===== E — 系统运维 (1张) =====
        ToolCardMeta(
            key_id="sys_diagnose_tests",
            shortcut="9",
            title="系统环境深度诊断与测试套件",
            subtitle="[T] 驱动与依赖诊断 / 85+ 项自动化 CI/CD 全量测试",
            category="E — 系统运维",
            is_gui=False,
            command=[sys.executable, "tools/cli_menu.py", "--diagnose"],
            tag_color=COLOR_E,
            summary="【系统健康与质量守门】全面检查系统环境依赖，并提供工程全量自动化测试套件。",
            details=[
                "全面检查 Python、OpenCV、NumPy C-API 及 RealSense USB 3.0 驱动就绪状态",
                "排查 yaml、PIL、matplotlib、scipy 等工业科学计算包环境版本",
                "全量测试执行命令：python -m unittest discover -s tests -p \"test_*.py\"",
                "涵盖数学平差 (BA)、图论连通拓扑、外参盲测体检与 UI 状态机，保障发布质量"
            ],
            inputs=["系统底层环境注册表与 tests/ 全量测试框架"],
            outputs=["控制台输出清晰的逐项绿勾诊断报告与全工程测试矩阵"],
            quick_tips="快捷键: [9] 或 [T] 启动环境深度诊断 | 遇到红叉时依提示执行 pip 修复命令"
        ),
    ]
    return catalog


    # 统一三大专区主色系 (低饱和专业工业冷色)
    COLOR_SANDBOX = (65, 175, 160)  # 工况沙盒与离线标定: 精密工业深青 (Precision Teal)
    COLOR_PROD = (85, 145, 215)     # 核心在线生产类: 典雅科技冷蓝 (Slate Blue)
    COLOR_CI = (130, 145, 165)      # 仿真演练与运维测试类: 沉稳钛银冷灰 (Titanium Gray)

    catalog = [
        # ================= 专区一：工况沙盒与离线标定 (统一深青精密色系) =================
        ToolCardMeta(
            key_id="scene_hub",
            shortcut="1",
            title="工况场景管理中枢 (Scene Hub)",
            subtitle="★ 顶层数据总控！沙盒画廊/大图巡检/生产发布",
            category="工况与离线",
            is_gui=True,
            command=[sys.executable, "-m", "tools.scene_hub"],
            tag_color=COLOR_SANDBOX,
            summary="【首位核心中枢】视觉系统的工况沙盒容器与数据总控驾驶舱，连接采集、平差与生产部署。",
            details=[
                "多工况画廊管理：选择、新建、重命名、克隆与独立物理沙盒数据隔离",
                "三大视图模式：标准三栏工作台 / 单帧大图全宽巡检 / 纯净几何健康看板",
                "场景几何健康度体检：动态覆盖率热力、留一盲测残差分布与两阶段平差指标",
                "严格恪守【草稿沙盒隔离、活动场景验证、生产原子发布】工业安全基准"
            ],
            inputs=["data/calibration_scenes/ 工况沙盒目录"],
            outputs=["当前活动场景切换、scene_meta.yaml、一键原子发布到 config/tags_map.yaml"],
            quick_tips="快捷键: [1] 或 [⏎] 启动 | 中枢内按 [↑/↓] 选场景 | [⏎] 激活 | [P] 发布生产 | [S] 进Studio"
        ),

        ToolCardMeta(
            key_id="tag_studio",
            shortcut="2",
            title="离线标定工作站 (Studio)",
            subtitle="[S] 多视角审核/两阶段 BA 平差/智能剪枝/质检闭环",
            category="工况与离线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_offline_studio.py"],
            tag_color=COLOR_SANDBOX,
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
            quick_tips="快捷键: [2] 或 [S] 启动 | 工作站内 [⏎] 快速求解 | [P] 智能剪枝 | [E] 超精提取 | [R] 导出报告"
        ),

        ToolCardMeta(
            key_id="tag_wizard",
            shortcut="3",
            title="多视角采图向导 (Wizard)",
            subtitle="[C] 角度雷达交互指引/空格极速连拍/自动归档沙盒",
            category="工况与离线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_capture_wizard.py"],
            tag_color=COLOR_SANDBOX,
            summary="【现场采图助手】专职采图向导：交互式指导相机移动至不同高度与俯仰角，高效采集高质量标定样本。",
            details=[
                "提供雷达式多视角视盘指引（俯视、大俯仰角、大滚转角、高低多层位态）",
                "按 [空格键] 极速无感连拍，样本自动存入当前场景 raw_images/ 目录",
                "实时 AprilTag 检出回显与白闪快门反馈，采图完毕后返回主中枢自动热重载"
            ],
            inputs=["RealSense D435 相机 (或 --mock 仿真)"],
            outputs=["当前活动场景 raw_images/view_*.png 原始高质量未压缩图集"],
            quick_tips="快捷键: [3] 或 [C] 启动 | 采图界面中 [空格] 拍摄归档 | [R] 重置批次 | [ESC] 完成采图返回"
        ),

        ToolCardMeta(
            key_id="tag_offline_verifier",
            shortcut="4",
            title="离线精度体检台 (LOO盲测)",
            subtitle="[L] 标定后留一交叉验证/双棱柱对比/残差评级",
            category="工况与离线",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_offline_verifier.py"],
            tag_color=COLOR_SANDBOX,
            summary="【标定验收闭环】科学级精度评估台：执行严格的 Leave-One-Out (LOO) 盲测交叉验证与外参鲁棒性体检。",
            details=[
                "留一交叉验证：轮流屏蔽每一张标定图像作为未知盲测帧，求解相机外参并预测未参与平差的标靶",
                "3D 空间双棱柱虚实位姿对比：直观呈现盲测外参与全局优化外参的空间刚体位移偏差",
                "2D 像平面残差矢量放大图：标注重投影误差方向分布，揭示畸变或单侧光照系统误差",
                "输出严谨的工业放行评级：优秀 (A)、达标 (B) 或 需补拍 (C)"
            ],
            inputs=["当前活动场景样本图集", "当前场景 tags_map.yaml"],
            outputs=["data/tag_calibration_verification/ 诊断报告与残差矢量可视化图"],
            quick_tips="快捷键: [4] 或 [L] 启动 | 体检界面中 [N/P] 切换盲测帧 | [R] 导出评估报告 | [ESC] 退出"
        ),

        # ================= 专区二：在线生产与现场作业 (统一科技冷蓝色系) =================
        ToolCardMeta(
            key_id="d435_live",
            shortcut="5",
            title="D435 实时相机与深度探针",
            subtitle="[D] 物理高帧率取流/毫米级深度探针/单帧快照",
            category="在线生产",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py"],
            tag_color=COLOR_PROD,
            summary="【现场感知总览】RealSense D435 物理相机的综合查看器与交互式深度测量探针。",
            details=[
                "实时获取 1280x720 RGB 与精准对齐的深度流",
                "鼠标悬停任意像素点，实时探针读取毫米级 (X, Y, Z) 空间坐标",
                "支持深度热力图着色 (JET/TURBO) 与直方图动态均衡增强",
                "按 [S] 键一键保存工业快照 (RGB + Depth + 点云 PLY)"
            ],
            inputs=["Intel RealSense D435 USB 3.0 物理相机"],
            outputs=["data/snapshots/ 单帧高质量工业多模态快照"],
            quick_tips="快捷键: [5] 或 [D] 启动 | 查看器内 [S] 存快照 | [M] 切换热力着色 | [D] 测距探针 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="tag_ar_verifier",
            shortcut="6",
            title="在线 AR 虚实融合验收系统",
            subtitle="[A] 3D轴网虚实融合/时域外参滤波锁定/现场验收",
            category="在线生产",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_calibration_verifier.py"],
            tag_color=COLOR_PROD,
            summary="【车间透视验收】通过虚实融合 AR 盲测直接肉眼检验平差地图的物理精确度。",
            details=[
                "高帧率实时取流，在检测到的 AprilTag 空间位置上虚实融合叠加 3D 彩色坐标轴",
                "在已知标靶基准上虚实融合渲染 3D 虚拟彩色立方体/四棱柱",
                "多帧时域外参滤波锁定：支持按 [L] 键采集 30 帧静止标靶，输出毫米级空间位姿方差",
                "直观检验空间尺度是否严丝合缝，确认是否存在扭曲、漂移或尺度缩放偏差"
            ],
            inputs=["D435 实时相机", "当前生产 tags_map.yaml 或场景地图"],
            outputs=["屏幕实时 AR 渲染显示、时域位姿锁定精度统计"],
            quick_tips="快捷键: [6] 或 [A] 启动 | AR界面中 [L] 启动时域锁定 | [M] 切换模型 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="asparagus_live",
            shortcut="7",
            title="芦笋抓取位姿解算 (实时生产)",
            subtitle="[F] 硬件相机抓拍解算顶层芦笋/输出 G-code",
            category="在线生产",
            is_gui=False,
            command=[sys.executable, "tools/find_top_asparagus.py"],
            tag_color=COLOR_PROD,
            summary="【核心生产算法】调用物理相机抓拍一帧并解算最上层芦笋空间位姿，输出抓取指令。",
            details=[
                "自动拉起 D435 物理相机完成自动曝光对齐与单帧捕获",
                "3D 表面法向量与空间骨架线拟合，精确定位顶层可抓取芦笋",
                "将相机坐标系位姿通过生产标定矩阵转换为 SCARA 机械臂基坐标系",
                "直接生成控制 SCARA 机械臂抓取的标准 G-code 指令与 JSON 协议"
            ],
            inputs=["D435 硬件相机", "config/camera_intrinsics.yaml", "config/tags_map.yaml"],
            outputs=["终端打印机械臂 G-code 指令、JSON 抓取坐标与调试渲染图"],
            quick_tips="快捷键: [7] 或 [F] 启动 | 独立控制台视窗执行，打印抓取坐标后按任意键退出。"
        ),

        # ================= 专区三：仿真演练与运维测试 (统一钛银冷灰色系) =================
        ToolCardMeta(
            key_id="sim_sandbox",
            shortcut="8",
            title="仿真模拟与离线快照验证",
            subtitle="[M] --mock 纯软件相机仿真 / 历史工业快照位姿解算",
            category="仿真与运维",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py", "--mock"],
            tag_color=COLOR_CI,
            summary="【脱机仿真沙盒】无硬件时的开发与调试利器：涵盖纯软件仿真相机与历史快照抓取算法验证。",
            details=[
                "生成合成渐变深度场与模拟测试 AprilTag 标靶纹理，模拟真实 30FPS 视频流与探针交互",
                "支持算法离线验证：从 data/snapshots/ 快速加载真实历史工业快照，验证芦笋抓取解算",
                "适合在离线工位、出差环境或算法调优期间进行全流程无硬件联调",
                "快照验证命令：python tools/find_top_asparagus.py --snapshot latest"
            ],
            inputs=["纯软件数学合成场 或 data/snapshots/ 历史已采集工业快照"],
            outputs=["模拟工业快照至 data/snapshots/ 或控制台算法解算结果"],
            quick_tips="快捷键: [8] 或 [M] 启动仿真查看器 | 离线快照测试在控制台执行对应参数命令"
        ),

        ToolCardMeta(
            key_id="sys_diagnose_tests",
            shortcut="9",
            title="系统环境深度诊断与测试套件",
            subtitle="[T] 驱动与依赖诊断 / 85+ 项自动化 CI/CD 全量测试",
            category="仿真与运维",
            is_gui=False,
            command=[sys.executable, "tools/cli_menu.py", "--diagnose"],
            tag_color=COLOR_CI,
            summary="【系统健康与质量守门】全面检查系统环境依赖，并提供工程全量自动化测试套件。",
            details=[
                "全面检查 Python、OpenCV、NumPy C-API 及 RealSense USB 3.0 驱动就绪状态",
                "排查 yaml、PIL、matplotlib、scipy 等工业科学计算包环境版本",
                "全量测试执行命令：python -m unittest discover -s tests -p \"test_*.py\"",
                "涵盖数学平差 (BA)、图论连通拓扑、外参盲测体检与 UI 状态机，保障发布质量"
            ],
            inputs=["系统底层环境注册表与 tests/ 全量测试框架"],
            outputs=["控制台输出清晰的逐项绿勾诊断报告与全工程测试矩阵"],
            quick_tips="快捷键: [9] 或 [T] 启动环境深度诊断 | 遇到红叉时依提示执行 pip 修复命令"
        ),
    ]
    return catalog
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

    def __init__(self):
        self.canvas_w = 1280
        self.canvas_h = 720
        # 窗口内部 key 标识使用纯英文，通过 Windows API 设定中文标题杜绝乱码
        self.window_name = "flux_vision_3d_suite_dashboard"
        self._running = True
        self._hwnd = None
        self._last_zoom_action = 0.0
        self._force_ctrl_pressed = False

        self.scene_mgr = CalibrationSceneManager()
        self.tools = build_tools_catalog()
        self.selected_tool_idx = -1    # 初始无选中，键盘/点击才激活焦点
        self.hover_tool_idx = -1

        # 视口与真矢量放大镜缩放控制 (基准 1280x720)
        self._base_w   = 1280
        self._base_h   = 720
        self.scale_pct = 100   # 缩放百分比 (50% ~ 200%)

        self.mouse_x = -1
        self.mouse_y = -1
        self.toast_msg = "欢迎使用 flux_vision_3d 工业视觉控制中心！按 [1~9] 或点击卡片进入工况中枢。"
        self.toast_time = time.time() + 4.0

        # 系统状态缓存
        self.system_status = {}
        self.refresh_system_status()

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
        """执行全局真矢量放大镜缩放：卡片尺寸、字号、间距等比矢量缩放，无位图拉伸锯齿"""
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
        self.set_toast(f"矢量放大镜: {self.scale_pct}%  (Ctrl +/- 或 滚轮缩放, Ctrl+0 复位)", duration=2.2)

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

        while self._running:
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

        cv2.destroyAllWindows()

    def _on_mouse(self, event, x, y, flags, param):
        """鼠标移动、点击与滚轮缩放事件 (与物理坐标 1:1 原生对齐)"""
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
            btn_h = int(38 * s)
            btn_y = py + ph - btn_h - int(12 * s)
            if px + int(16 * s) <= x <= px + pw - int(16 * s) and btn_y <= y <= btn_y + btn_h:
                active = self.hover_tool_idx if self.hover_tool_idx != -1 else self.selected_tool_idx
                if 0 <= active < len(self.tools):
                    self._launch_tool(self.tools[active])
                return

            # 点击左侧卡片
            if card_idx != -1:
                self.selected_tool_idx = card_idx
                self._launch_tool(self.tools[card_idx])
                return

        # 鼠标双击直接启动
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if card_idx != -1:
                self.selected_tool_idx = card_idx
                self._launch_tool(self.tools[card_idx])

    def _handle_keyboard(self, raw_key: int):
        """键盘快捷键响应 (五分组布局: row0全宽A, row1-2为B4张2×2, row3为C2张, row4为D+E各1张)"""
        if raw_key in (13, 10):  # 回车
            if 0 <= self.selected_tool_idx < len(self.tools):
                self._launch_tool(self.tools[self.selected_tool_idx])
            return

        # 方向键：将卡片索引映射到 (row, col) 坐标后导航
        # row=0 → card 0 (全宽，col固定=0)
        # row=1 → cards 1,2  row=2 → cards 3,4
        # row=3 → cards 5,6  row=4 → cards 7,8
        def idx_to_rc(i: int) -> Tuple[int, int]:
            if i < 0:
                return (0, 0)   # 未选中时默认从第一张开始导航
            return (0, 0) if i == 0 else ((i - 1) // 2 + 1, (i - 1) % 2)

        def rc_to_idx(r: int, c: int) -> int:
            if r == 0:
                return 0
            i = (r - 1) * 2 + 1 + c
            return min(i, 8)  # clamp 至最后一张

        row, col = idx_to_rc(self.selected_tool_idx)

        if raw_key in (2490368, 65362, 38):    # 上
            if row > 0:
                row -= 1
                col = 0 if row == 0 else col
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        if raw_key in (2621440, 65364, 40):    # 下
            if row < 4:
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
            '2': "tag_wizard",           # B
            '3': "tag_studio",           # B
            '4': "tag_offline_verifier", # B
            '5': "tag_ar_verifier",      # B
            '6': "d435_live",            # C
            '7': "sim_sandbox",          # C
            '8': "asparagus_live",       # D
            '9': "sys_diagnose_tests",   # E
            # 直觉字母快捷键
            's': "tag_studio",
            'c': "tag_wizard",
            'l': "tag_offline_verifier",
            'a': "tag_ar_verifier",
            'd': "d435_live",
            'm': "sim_sandbox",
            'f': "asparagus_live",
            't': "sys_diagnose_tests",
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
        """返回第 idx 张卡片的 (x, y, w, h)，与渲染布局严格保持一致 (真矢量缩放联动)

        布局参数 (五分组):
          LH   分组标题条高度
          CH   卡片高度
          CW   卡片宽度
          SX   组内列间距
          SY   组内行间距
          GY   组间额外间距 (= 5 × SY)
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

        if 1 <= idx <= 4:     # B: 2×2
            b = idx - 1
            base_y = Y0 + LH + CH + GY + LH
            return X0 + (b % 2) * (CW + SX), base_y + (b // 2) * (CH + SY), CW, CH

        if 5 <= idx <= 6:     # C: 1行×2列
            c = idx - 5
            base_y = Y0 + LH + CH + GY + LH + 2 * (CH + SY) + GY + LH
            return X0 + c * (CW + SX), base_y, CW, CH

        if 7 <= idx <= 8:     # D + E: 1行×2列
            de = idx - 7
            base_y = Y0 + LH + CH + GY + LH + 2 * (CH + SY) + GY + LH + CH + GY + LH
            return X0 + de * (CW + SX), base_y, CW, CH

        return 0, 0, 0, 0

    def _hit_test_cards(self, x: int, y: int) -> int:
        """鼠标命中测试：委托给 _get_card_rect，与渲染位置严格一致"""
        for idx in range(len(self.tools)):
            cx, cy, cw, ch = self._get_card_rect(idx)
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                return idx
        return -1

    def _launch_tool(self, tool: ToolCardMeta):
        """执行启动子工具或测试"""
        self.set_toast(f"正在拉起: 【{tool.title}】...")

        self._present_canvas()
        cv2.waitKey(20)

        cmd = tool.command
        try:
            if tool.is_gui:
                res = subprocess.run(cmd)
                self.set_toast(f"【{tool.title}】已退出，系统状态已刷新。")
            else:
                if sys.platform == "win32":
                    full_cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd])
                    wrapper_cmd = f'cmd.exe /c "{full_cmd_str} & echo. & echo [完成] 请按任意键返回控制中心... & pause > nul"'
                    res = subprocess.run(wrapper_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
                    self.set_toast(f"【{tool.title}】执行完毕，控制台已返回。")
                else:
                    res = subprocess.run(cmd)
                    self.set_toast(f"【{tool.title}】执行完毕。")
        except Exception as e:
            self.set_toast(f"启动失败: {e}", duration=5.0)

        self.refresh_system_status()

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

        # 4. 右侧实时说明大屏 (仅在有卡片高亮/选中时渲染)
        cur_idx = self.hover_tool_idx if self.hover_tool_idx != -1 else self.selected_tool_idx
        if 0 <= cur_idx < len(self.tools):
            self._render_inspector_panel(canvas, self.tools[cur_idx], split_x)

        # 5. 底部状态与快捷键指引栏
        self._render_footer(canvas)

        return canvas

    def _render_top_bar(self, canvas: np.ndarray):
        """渲染顶部硬件与系统状态常驻监控栏 (真矢量自适应排布)"""
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

        # 状态探针芯片组 (自适应排布)
        st = self.system_status
        py_ver = sys.version.split()[0]
        cv_ver = st.get("cv_version", cv2.__version__)
        rs_ok = st.get("has_realsense", False)
        rs_str = "D435就绪" if rs_ok else "D435未就绪"
        rs_color = (0, 210, 160) if rs_ok else (130, 140, 150)

        chip_x = max(130, int(230 * s))
        chip_w = max(260, int(460 * s))
        chip_y1, chip_y2 = max(6, int(12 * s)), max(24, int(42 * s))
        cv2.rectangle(canvas, (chip_x, chip_y1), (chip_x + chip_w, chip_y2), (22, 27, 36), -1)
        cv2.rectangle(canvas, (chip_x, chip_y1), (chip_x + chip_w, chip_y2), (36, 46, 60), 1)

        chip_font = max(9, int(12 * s))
        cv2.circle(canvas, (chip_x + max(6, int(14 * s)), top_h // 2), max(2, int(4 * s)), (0, 210, 160), -1)
        draw_text(canvas, f"Py {py_ver}", (chip_x + max(12, int(24 * s)), max(8, int(18 * s))),
                  font_size=chip_font, color=(190, 205, 220))

        sep1 = chip_x + int(chip_w * 0.22)
        cv2.line(canvas, (sep1, max(8, int(16 * s))), (sep1, max(20, int(38 * s))), (40, 50, 65), 1)
        draw_text(canvas, f"CV {cv_ver}", (sep1 + max(6, int(10 * s)), max(8, int(18 * s))),
                  font_size=chip_font, color=(190, 205, 220))

        sep2 = chip_x + int(chip_w * 0.48)
        cv2.line(canvas, (sep2, max(8, int(16 * s))), (sep2, max(20, int(38 * s))), (40, 50, 65), 1)
        cv2.circle(canvas, (sep2 + max(6, int(12 * s)), top_h // 2), max(2, int(4 * s)), rs_color, -1)
        draw_text(canvas, rs_str, (sep2 + max(12, int(22 * s)), max(8, int(18 * s))),
                  font_size=chip_font, color=rs_color)

        sep3 = chip_x + int(chip_w * 0.74)
        cv2.line(canvas, (sep3, max(8, int(16 * s))), (sep3, max(20, int(38 * s))), (40, 50, 65), 1)
        snaps_c = st.get("snapshot_count", 0)
        draw_text(canvas, f"快照:{snaps_c}", (sep3 + max(6, int(10 * s)), max(8, int(18 * s))),
                  font_size=chip_font, color=(190, 205, 220))

        # 当前活动场景胶囊
        act_sc = self.scene_mgr.get_active_scene()
        capsule_x = chip_x + chip_w + max(8, int(15 * s))
        capsule_w = max(int(160 * s), min(int(360 * s), self.canvas_w - capsule_x - int(150 * s)))

        if capsule_x + capsule_w < self.canvas_w - int(140 * s):
            if act_sc:
                status_tag = "★生产" if act_sc.is_published else ("已平差" if act_sc.ba_solved else "沙盒")
                tag_col = self.COLOR_GOLD if act_sc.is_published else ((0, 210, 160) if act_sc.ba_solved else (135, 165, 195))
                cv2.rectangle(canvas, (capsule_x, max(6, int(10 * s))), (capsule_x + capsule_w, max(24, int(44 * s))), (22, 28, 38), -1)
                cv2.rectangle(canvas, (capsule_x, max(6, int(10 * s))), (capsule_x + capsule_w, max(24, int(44 * s))), (45, 60, 78), 1)
                draw_text(canvas, f"【{act_sc.name}】({status_tag})", (capsule_x + max(6, int(10 * s)), max(8, int(17 * s))),
                          font_size=max(10, int(13 * s)), color=tag_col, bold=True)
            else:
                cv2.rectangle(canvas, (capsule_x, max(6, int(10 * s))), (capsule_x + capsule_w, max(24, int(44 * s))), (22, 25, 32), -1)
                cv2.rectangle(canvas, (capsule_x, max(6, int(10 * s))), (capsule_x + capsule_w, max(24, int(44 * s))), (40, 48, 60), 1)
                draw_text(canvas, "未选定场景", (capsule_x + max(6, int(10 * s)), max(8, int(17 * s))),
                          font_size=max(10, int(13 * s)), color=(140, 150, 160))

        # 右上角 [X] 退出按钮 (自适应靠右)
        bw = max(80, int(125 * s))
        bh = max(24, int(34 * s))
        bx = self.canvas_w - bw - max(8, int(15 * s))
        by = max(6, int(10 * s))
        is_hover_exit = (bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh)
        exit_bg = (48, 22, 24) if is_hover_exit else (32, 20, 22)
        exit_border = (210, 60, 60) if is_hover_exit else (95, 36, 40)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_bg, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_border, 2 if is_hover_exit else 1)
        draw_text(canvas, "[X] 退出 [ESC]", (bx + max(6, int(12 * s)), by + max(4, int(9 * s))),
                  font_size=max(10, int(13 * s)), color=(220, 170, 170), bold=True)

    def _render_tools_grid(self, canvas: np.ndarray):
        """渲染左侧工具卡片网格 (真矢量自适应缩放)"""
        s = self.scale_pct / 100.0
        LH = max(14, int(20 * s))
        CW = max(200, int(370 * s))
        SX = max(6, int(12 * s))
        X0 = max(8, int(15 * s))
        FW = CW * 2 + SX

        HEADER_TEXT = (192, 206, 222)
        group_headers = [
            (self._get_card_rect(0)[1] - LH, FW, "A  场景总控",                (195, 155,  45)),
            (self._get_card_rect(1)[1] - LH, FW, "B  标定流水线",              ( 65, 175, 160)),
            (self._get_card_rect(5)[1] - LH, FW, "C  感知层",                  ( 90, 140, 195)),
            (self._get_card_rect(7)[1] - LH, FW, "D  生产执行  ·  E  系统运维", (140, 150, 165)),
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

            # 左侧分组色条
            cv2.rectangle(canvas, (cx, cy), (cx + max(2, int(4 * s)), cy + ch), tool.tag_color, -1)

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
        """渲染右侧动态即时说明大屏 (自适应全宽与全高，1:1 矢量清晰无模糊)"""
        s = self.scale_pct / 100.0
        px = split_x + max(8, int(15 * s))
        py = max(40, int(66 * s))
        pw = max(int(360 * s), self.canvas_w - px - max(10, int(20 * s)))
        ph = max(int(450 * s), self.canvas_h - max(30, int(50 * s)) - py - max(8, int(15 * s)))

        # 大屏底板
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 22, 28), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (34, 44, 58), 1)

        # 头部标题带
        head_h = max(40, int(64 * s))
        cv2.rectangle(canvas, (px, py), (px + pw, py + head_h), (22, 27, 36), -1)
        cv2.line(canvas, (px, py + head_h), (px + pw, py + head_h), (40, 52, 68), 1)

        # 快捷键徽章与主标题
        b_w, b_h = max(24, int(36 * s)), max(20, int(32 * s))
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(16 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(16 * s)) + b_h), (14, 18, 24), -1)
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(16 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(16 * s)) + b_h), tool.tag_color, 1)
        draw_text(canvas, f"[{tool.shortcut}]", (px + max(10, int(20 * s)), py + max(12, int(22 * s))),
                  font_size=max(10, int(14 * s)), color=tool.tag_color, bold=True)

        draw_text(canvas, tool.title, (px + max(36, int(62 * s)), py + max(8, int(14 * s))),
                  font_size=max(12, int(18 * s)), color=self.COLOR_TEXT_TITLE, bold=True)
        mode_str = "原生 GUI 视窗" if tool.is_gui else "控制台"
        draw_text(canvas, f"类别: {tool.category}   |   模式: {mode_str}",
                  (px + max(36, int(62 * s)), py + max(24, int(40 * s))), font_size=max(9, int(12 * s)), color=self.COLOR_TEXT_SUB)

        # 核心概述 (Summary)
        curr_y = py + max(48, int(78 * s))
        draw_text(canvas, "【功能定位与现场痛点】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(24 * s))
        draw_text(canvas, tool.summary, (px + max(10, int(20 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=(215, 225, 235))
        curr_y += max(22, int(36 * s))

        # 详细特性清单 (Details)
        draw_text(canvas, "【工程要点与执行逻辑】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(24 * s))
        for d in tool.details:
            cv2.circle(canvas, (px + max(12, int(24 * s)), curr_y + max(4, int(8 * s))), max(2, int(3 * s)), (0, 190, 160), -1)
            draw_text(canvas, d, (px + max(18, int(34 * s)), curr_y),
                      font_size=max(9, int(12 * s)), color=(195, 208, 220))
            curr_y += max(14, int(22 * s))
        curr_y += max(6, int(10 * s))

        # 前置依赖与输入 (Inputs)
        draw_text(canvas, "【前置条件与输入依赖】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=(140, 180, 220), bold=True)
        curr_y += max(14, int(22 * s))
        for inp in tool.inputs:
            cv2.circle(canvas, (px + max(12, int(24 * s)), curr_y + max(4, int(8 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
            draw_text(canvas, inp, (px + max(18, int(34 * s)), curr_y),
                      font_size=max(9, int(12 * s)), color=(185, 200, 215))
            curr_y += max(13, int(20 * s))
        curr_y += max(6, int(10 * s))

        # 输出产物 (Outputs)
        draw_text(canvas, "【输出产物与持久化路径】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=(120, 200, 180), bold=True)
        curr_y += max(14, int(22 * s))
        for out in tool.outputs:
            cv2.circle(canvas, (px + max(12, int(24 * s)), curr_y + max(4, int(8 * s))), max(2, int(3 * s)), (100, 180, 160), -1)
            draw_text(canvas, out, (px + max(18, int(34 * s)), curr_y),
                      font_size=max(9, int(12 * s)), color=(180, 215, 205))
            curr_y += max(13, int(20 * s))
        curr_y += max(8, int(12 * s))

        # 操作提示 (Quick Tips)
        tip_h = max(24, int(36 * s))
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), curr_y), (px + pw - max(8, int(16 * s)), curr_y + tip_h), (22, 28, 36), -1)
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), curr_y), (px + pw - max(8, int(16 * s)), curr_y + tip_h), (36, 48, 62), 1)
        draw_text(canvas, tool.quick_tips, (px + max(12, int(24 * s)), curr_y + max(4, int(9 * s))),
                  font_size=max(9, int(12 * s)), color=self.COLOR_TEXT_SUB)

        # 底部醒目启动卡片按钮
        btn_h = max(26, int(38 * s))
        btn_y = py + ph - btn_h - max(8, int(12 * s))
        is_hover_btn = (px + max(8, int(16 * s)) <= self.mouse_x <= px + pw - max(8, int(16 * s)) and btn_y <= self.mouse_y <= btn_y + btn_h)
        btn_bg = (0, 190, 145) if is_hover_btn else (0, 155, 120)
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), btn_y), (px + pw - max(8, int(16 * s)), btn_y + btn_h), btn_bg, -1)
        btn_text = f"▶ 立即启动: 【{tool.title}】 (回车 ⏎ 或 双击卡片)"
        draw_text(canvas, btn_text, (px + max(20, int(35 * s)), btn_y + max(6, int(10 * s))),
                  font_size=max(10, int(14 * s)), color=(10, 18, 22), bold=True)

    def _render_footer(self, canvas: np.ndarray):
        """渲染底部状态反馈与快捷键指引栏 (自适应贴底)"""
        s = self.scale_pct / 100.0
        footer_h = max(34, int(50 * s))
        fy = self.canvas_h - footer_h
        cv2.rectangle(canvas, (0, fy), (self.canvas_w, self.canvas_h), (13, 15, 19), -1)
        cv2.line(canvas, (0, fy), (self.canvas_w, fy), self.COLOR_BORDER, 1)

        # 1. 动态 Toast 消息反馈
        now = time.time()
        f_size = max(9, int(13 * s))
        if self.toast_time > now:
            draw_text(canvas, f"[系统提示] {self.toast_msg}", (max(10, int(20 * s)), fy + max(8, int(14 * s))),
                      font_size=max(10, int(14 * s)), color=(0, 220, 180), bold=True)
        else:
            hint_txt = f"[1~9] 快速启动  [↑/↓/←/→] 浏览  [Ctrl +/- 或 滚轮] 放大镜({self.scale_pct}%)  [Ctrl+0] 复位  [⏎] 启动  [ESC] 退出"
            draw_text(canvas, hint_txt, (max(10, int(20 * s)), fy + max(10, int(16 * s))),
                      font_size=f_size, color=(160, 175, 190))

        # 2. 右侧系统时钟与运行提示 (右对齐)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        clock_w = max(200, int(360 * s))
        clock_x = max(int(500 * s), self.canvas_w - clock_w)
        draw_text(canvas, f"{time_str} | FLUX VISION 3D", (clock_x, fy + max(10, int(16 * s))),
                  font_size=f_size, color=self.COLOR_TEXT_MUTED)


def main():
    parser = argparse.ArgumentParser(description="3D 视觉综合控制中心 (3D Vision Suite GUI Launcher)")
    args = parser.parse_args()

    app = GuiLauncherApp()
    app.run()


if __name__ == "__main__":
    main()
