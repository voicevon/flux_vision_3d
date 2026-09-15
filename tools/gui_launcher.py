#!/usr/bin/env python3
"""
3D 视觉综合控制中心 (3D Vision Suite GUI Launcher)
=================================================
基于 1280x720 深色工业科技大屏，统一调度 flux_vision_3d 视觉系统的所有核心应用：
- 顶部硬件与系统状态探针常驻监控
- 左侧模块化科技卡片：【核心生产与工况】、【视觉标定与建图流水线】、【自动化测试与系统运维】
- 右侧动态即时说明大屏 (Live Inspector)：鼠标悬停即刻展开详细工业说明书、依赖、输出产物与操作指南
- 统一支持鼠标一键点击启动与键盘全局快捷键直达
- 智能双模分发：GUI 工具无缝平滑拉起，终端测试与诊断带专属控制台视窗绝不吞日志
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
from tools.cli_menu import check_env_status

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
    """构建全系统工具与应用的权威目录与深度中文说明"""
    catalog = [
        # ================= 核心生产与工况沙盒 =================
        ToolCardMeta(
            key_id="d435_live",
            shortcut="1",
            title="D435 实时相机与探针",
            subtitle="物理硬件高帧率取流与深度像素交互探针",
            category="生产与工况",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py"],
            tag_color=(0, 255, 180),
            summary="RealSense D435 物理相机的综合查看器与交互式深度测量探针。",
            details=[
                "实时获取 1280x720 RGB 与精准对齐的深度流",
                "鼠标悬停任意像素点，实时探针读取毫米级 (X, Y, Z) 空间坐标",
                "支持深度热力图着色 (JET/TURBO) 与直方图动态均衡增强",
                "按 [S] 键一键保存工业快照 (RGB + Depth + 点云 PLY)"
            ],
            inputs=["Intel RealSense D435 USB 3.0 物理相机"],
            outputs=["data/snapshots/ 单帧高质量工业多模态快照"],
            quick_tips="快捷键: [S] 存快照 | [M] 切换热力着色 | [D] 深度距离探针 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="scene_hub",
            shortcut="2",
            title="工况与场景管理中枢",
            subtitle="★ 一级核心总控！沙盒画廊/大图巡检/健康看板",
            category="生产与工况",
            is_gui=True,
            command=[sys.executable, "-m", "tools.scene_hub"],
            tag_color=(0, 240, 220),
            summary="视觉系统的工况沙盒容器与数据总控驾驶舱，连接采集、平差与生产部署。",
            details=[
                "多工况画廊管理：选择、新建、重命名、克隆与独立物理沙盒隔离",
                "三大视图模式：标准三栏工作台 / 单帧大图全宽巡检 / 纯净几何健康看板",
                "场景几何健康度体检：动态覆盖率热力、留一盲测残差分布与两阶段平差指标",
                "严格恪守【草稿沙盒隔离、活动场景验证、生产原子发布】安全基准"
            ],
            inputs=["data/calibration_scenes/ 工况沙盒目录"],
            outputs=["当前活动场景切换、scene_meta.yaml、一键原子发布到 config/tags_map.yaml"],
            quick_tips="快捷键: [↑/↓] 选场景 | [⏎] 设为活动 | [P] 生效生产 | [F] 切换视图 | [S] 进Studio"
        ),

        ToolCardMeta(
            key_id="asparagus_live",
            shortcut="3",
            title="芦笋单帧解算 (实时相机)",
            subtitle="D435 单帧采集解算最顶层抓取位姿与 G-code",
            category="生产与工况",
            is_gui=False,
            command=[sys.executable, "tools/find_top_asparagus.py"],
            tag_color=(0, 210, 255),
            summary="核心生产算法：调用真实相机抓拍一帧并解算最上层芦笋空间位姿。",
            details=[
                "自动拉起 D435 物理相机完成自动曝光对齐与单帧捕获",
                "3D 表面法向量与空间骨架线拟合，精确定位顶层可抓取芦笋",
                "将相机坐标系位姿通过生产标定矩阵转换为 SCARA 机械臂基坐标系",
                "直接生成控制 SCARA 机械臂抓取的标准 G-code 指令与 JSON 协议"
            ],
            inputs=["D435 硬件相机", "config/camera_intrinsics.yaml", "config/tags_map.yaml"],
            outputs=["终端打印机械臂 G-code 指令、JSON 抓取坐标与调试渲染图"],
            quick_tips="执行模式: 独立控制台黑窗执行，打印抓取坐标后按任意键退出。"
        ),

        ToolCardMeta(
            key_id="asparagus_snapshot",
            shortcut="4",
            title="芦笋位姿解算 (离线快照)",
            subtitle="自动装载本地最新工业快照验证抓取算法",
            category="生产与工况",
            is_gui=False,
            command=[sys.executable, "tools/find_top_asparagus.py", "--snapshot", "latest"],
            tag_color=(100, 200, 255),
            summary="无硬件时的算法调试利器：从 snapshots 快速加载真实快照验证抓取解算。",
            details=[
                "无需连接物理相机，自动检索 data/snapshots/ 中最新拍摄的快照",
                "加载对齐后的 RGB-D 深度阵列，纯离线执行芦笋分割与位姿估计",
                "快速回归算法参数改动对实际工业样本的解算精度影响"
            ],
            inputs=["data/snapshots/ 历史已采集快照", "生产外参矩阵"],
            outputs=["终端打印位姿解算结果与测试诊断日志"],
            quick_tips="适合用于算法参数调优、离线回归与演示验证。"
        ),

        ToolCardMeta(
            key_id="d435_mock",
            shortcut="6",
            title="D435 仿真模拟查看器",
            subtitle="--mock 纯软件仿真模式，无需物理相机硬件",
            category="生产与工况",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py", "--mock"],
            tag_color=(180, 180, 220),
            summary="纯软件仿真查看器，支持在无任何物理相机时进行完整的 GUI 功能验证。",
            details=[
                "生成合成渐变深度场与模拟测试 AprilTag 标靶纹理",
                "模拟真实 30FPS 视频流推送与深度像素交互计算",
                "完全保留实时探针、热力图着色与快照导出等全部交互功能"
            ],
            inputs=["无硬件依赖 (纯软件数学合成)"],
            outputs=["模拟工业快照至 data/snapshots/"],
            quick_tips="快捷键与真实相机模式完全一致，适合开发调试与脱机测试。"
        ),

        # ================= 标定与空间建图流水线 =================
        ToolCardMeta(
            key_id="tag_studio",
            shortcut="S",
            title="离线标定工作站 (Studio)",
            subtitle="多视角审核/两阶段 BA 平差/残差剪枝/全局质检",
            category="视觉标定",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_offline_studio.py"],
            tag_color=(0, 255, 140),
            summary="标定流水线核心工作站：一站式样本交互审核、高精两阶段 BA 平差求解与质检闭环。",
            details=[
                "自动装载当前活动沙盒场景，多视角图像九宫格缩略图交互式审核与启闭",
                "两阶段全局平差：Cauchy 鲁棒核粗平差 + MAD 统计自适应清洗 + Levenberg-Marquardt 精平差",
                "智能残差剪枝 (Auto-Prune)：自动迭代剪除反光/微动导致的高残差外点，拓扑安全守门",
                "视网膜级热力覆盖度评估，一键导出 Markdown 格式全面质检体检报告"
            ],
            inputs=["当前活动场景 raw_images/", "相机内参 camera_intrinsics.yaml"],
            outputs=["当前场景 tags_map.yaml", "reports/studio_qa_report_*.md"],
            quick_tips="快捷键: [⏎] 快速求解 | [P] 智能剪枝 | [E] 全局超精提取 | [R] 导出报告"
        ),

        ToolCardMeta(
            key_id="tag_wizard",
            shortcut="C",
            title="交互式多视角采图向导",
            subtitle="视盘俯仰角度交互式指引，空格一键连拍归档",
            category="视觉标定",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_capture_wizard.py"],
            tag_color=(0, 220, 255),
            summary="专职采图向导：交互式指导相机移动至不同高度与俯仰角，高效采集高质量标定样本。",
            details=[
                "提供雷达式多视角视盘指引（俯视、大俯仰角、大滚转角、高低多层位态）",
                "按 [空格键] 极速无感连拍，样本自动存入当前场景 raw_images/ 目录",
                "实时 AprilTag 实时检出回显与白闪快门反馈，采图完毕后返回主中枢自动热重载"
            ],
            inputs=["RealSense D435 相机 (或 --mock 仿真)"],
            outputs=["当前活动场景 raw_images/view_*.png 原始高质量未压缩图集"],
            quick_tips="快捷键: [空格] 拍摄归档 | [R] 重置批次 | [ESC] 完成采图并返回"
        ),

        ToolCardMeta(
            key_id="tag_ar_verifier",
            shortcut="A",
            title="在线 AR 虚实融合验收系统",
            subtitle="相机实时取流/3D轴网虚实融合/时域外参滤波",
            category="视觉标定",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_calibration_verifier.py"],
            tag_color=(255, 200, 0),
            summary="现场终极验收工具：通过虚实融合 AR 盲测直接肉眼检验平差地图的物理精确度。",
            details=[
                "高帧率实时取流，在检测到的 AprilTag 空间位置上虚实融合叠加 3D 彩色坐标轴",
                "在已知标靶基准上虚实融合渲染 3D 虚拟彩色立方体/四棱柱",
                "多帧时域外参滤波锁定：支持按 [L] 键采集 30 帧静止标靶，输出毫米级空间位姿方差",
                "直观检验空间尺度是否严丝合缝，确认是否存在扭曲、漂移或尺度缩放偏差"
            ],
            inputs=["D435 实时相机", "当前生产 tags_map.yaml 或场景地图"],
            outputs=["屏幕实时 AR 渲染显示、时域位姿锁定精度统计"],
            quick_tips="快捷键: [L] 启动定点静态位姿锁定 | [M] 切换立方体/轴线模式 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="tag_map_builder",
            shortcut="M",
            title="纯计算空间建图求解器",
            subtitle="命令行静默求解：图论连通建模 + 两阶段 BA 平差",
            category="视觉标定",
            is_gui=False,
            command=[sys.executable, "tools/calibration/tag_map_builder.py", "--active"],
            tag_color=(120, 240, 100),
            summary="纯算法命令行建图工具：构建全局共视拓扑图，两阶段 BA 静默平差求解。",
            details=[
                "分析当前活动场景内所有采图帧的 Tag 观测，构建多视角共视因子图",
                "基于超图连通性分解最大连通分量，计算最优初始空间位姿骨架",
                "运行两阶段非线性最小二乘优化，终端实时滚动输出重投影 RMSE 收敛曲线"
            ],
            inputs=["当前活动场景 raw_images/", "内参配置"],
            outputs=["当前场景 tags_map.yaml 空间立体地图"],
            quick_tips="执行模式: 弹出独立控制台视窗，求解完成后显示最终 RMSE 并等待按键关闭。"
        ),

        ToolCardMeta(
            key_id="tag_offline_verifier",
            shortcut="L",
            title="离线精度体检台 (LOO盲测)",
            subtitle="全量留一交叉验证/双棱柱对比/残差矢量评级",
            category="视觉标定",
            is_gui=True,
            command=[sys.executable, "tools/calibration/tag_offline_verifier.py"],
            tag_color=(255, 160, 60),
            summary="科学级精度评估台：执行严格的 Leave-One-Out (LOO) 盲测交叉验证与外参鲁棒性体检。",
            details=[
                "留一交叉验证：轮流屏蔽每一张标定图像作为未知盲测帧，求解相机外参并预测未参与平差的标靶",
                "3D 空间双棱柱虚实位姿对比：直观呈现盲测外参与全局优化外参的空间刚体位移偏差",
                "2D 像平面残差矢量放大图：标注重投影误差方向分布，揭示畸变或单侧光照系统误差",
                "输出严谨的工业放行评级：优秀 (A)、达标 (B) 或 需补拍 (C)"
            ],
            inputs=["当前活动场景样本图集", "当前场景 tags_map.yaml"],
            outputs=["data/tag_calibration_verification/ 诊断报告与残差矢量可视化图"],
            quick_tips="快捷键: [N/P] 切换盲测帧 | [R] 导出详细盲测评估报告 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="tag_generator",
            shortcut="T",
            title="AprilTag 标靶图纸生成",
            subtitle="生成 0~29 号矢量标靶与 1:1 A4 打印排版 PDF",
            category="视觉标定",
            is_gui=False,
            command=[sys.executable, "tools/calibration/generate_apriltags.py"],
            tag_color=(160, 140, 240),
            summary="标定前期标靶制作工具：生成工业高精矢量标靶与实际物理尺寸 1:1 排版图纸。",
            details=[
                "生成 tag16h5 字典族 0~29 号超高清无损矢量标靶 PNG",
                "生成严丝合缝的 1:1 A4 打印排版 PDF 文件，标注明晰毫米级刻度与裁剪对齐参考线",
                "支持标准 80mm / 100mm 标靶尺寸，确保在普通打印机上直接输出即用"
            ],
            inputs=["config.yaml 标靶尺寸配置"],
            outputs=["data/apriltags_16h5/ 矢量 PNG 图片与 PDF 打印图纸"],
            quick_tips="执行模式: 生成完成后自动弹出输出目录供查看与打印。"
        ),

        ToolCardMeta(
            key_id="tag_diagnose",
            shortcut="D",
            title="标靶漏检病因切片诊断",
            subtitle="16级阈值网格+CLAHE，深度分析淘汰候选四边形",
            category="视觉标定",
            is_gui=False,
            command=[sys.executable, "tools/calibration/diagnose_tag_frame.py", "--active"],
            tag_color=(220, 120, 160),
            summary="恶劣工况排障利器：深入分析真图候选四边形轮廓，定位漏检真实物理病因。",
            details=[
                "16 级全局/局部阈值遍历 + 自适应双尺度 CLAHE 动态直方图重映射",
                "提取所有候选四边形几何轮廓，分析因面积过小、长宽比失真、对比度过低被拒原因",
                "输出单帧候选四边形切片拼图，协助车间调优补光角度与相机曝光增益"
            ],
            inputs=["当前场景原始图像"],
            outputs=["终端诊断病因切片输出与轮廓标注可视化图"],
            quick_tips="执行模式: 在独立控制台视窗中运行，给出清晰的参数调优指导。"
        ),

        # ================= 自动化测试与系统运维 =================
        ToolCardMeta(
            key_id="test_pipeline",
            shortcut="F1",
            title="视觉管线端到端脱机仿真测试",
            subtitle="运行全流程仿真测试 (test_mock_pipeline.py)",
            category="测试运维",
            is_gui=False,
            command=[sys.executable, "-m", "unittest", "tests/test_mock_pipeline.py"],
            tag_color=(0, 240, 140),
            summary="端到端仿真测试：脱机验证从相机取流、深度点云提取、标靶建图到位姿输出的全流程。",
            details=[
                "无需任何真实硬件，自动构建合成空间场与数学模型",
                "验证整个数据流动管线的鲁棒性，保障核心代码重构无破坏性错误"
            ],
            inputs=["tests/ 单元测试框架"],
            outputs=["控制台输出端到端执行结果与耗时报告"],
            quick_tips="执行模式: 弹出独立控制台窗口显示测试进度，绿色 OK 即代表通过。"
        ),

        ToolCardMeta(
            key_id="test_real_snapshots",
            shortcut="F2",
            title="真实工业快照全量回归测试",
            subtitle="20 组真实工业快照 100% 验收算法正确性",
            category="测试运维",
            is_gui=False,
            command=[sys.executable, "-m", "unittest", "tests/test_real_snapshot.py"],
            tag_color=(0, 220, 200),
            summary="真实工业样本基准测试：使用车间采集的真实芦笋快照库验证抓取算法召回率。",
            details=[
                "覆盖复杂重叠、反光、暗光等 20 组真实工业工况快照",
                "验证顶层芦笋识别准确率、空间法向拟合与机械臂位姿输出稳定性",
                "作为发布生产前不可逾越的算法质量红线"
            ],
            inputs=["data/snapshots/ 工业真实数据集"],
            outputs=["控制台输出 20 组样本测试详细结果矩阵"],
            quick_tips="执行模式: 弹出控制台窗口，执行时间约 3~5 秒，检验算法对真图的适应度。"
        ),

        ToolCardMeta(
            key_id="test_all_suite",
            shortcut="F3",
            title="运行工程全量测试套件",
            subtitle="85+ 项单元测试与集成测试全量自动化执行",
            category="测试运维",
            is_gui=False,
            command=[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            tag_color=(255, 200, 50),
            summary="全工程自动化 CI/CD 测试套件：一键运行全部 85 项自动化单元与集成测试。",
            details=[
                "涵盖数学平差优化器 (BA)、图论连通度拓扑、外参盲测体检引擎",
                "测试场景状态机、数据模型沙盒隔离、UI 渲染引擎与相机流",
                "全绿通过即确保整个工程处于最高健康就绪状态"
            ],
            inputs=["工程 tests/ 全量测试代码"],
            outputs=["全量测试覆盖报告与错误断言排查"],
            quick_tips="执行模式: 独立控制台视窗执行，耗时约 20 秒，全量回归验证。"
        ),

        ToolCardMeta(
            key_id="sys_diagnose",
            shortcut="F4",
            title="系统驱动与运行环境深度诊断",
            subtitle="Python/OpenCV/NumPy/RealSense 驱动与依赖诊断",
            category="测试运维",
            is_gui=False,
            command=[sys.executable, "tools/cli_menu.py", "--diagnose"],
            tag_color=(180, 160, 220),
            summary="环境与依赖排查工具：全面检查操作系统、Python 运行时、OpenCV 与 RealSense 驱动状态。",
            details=[
                "检查 pyrealsense2 驱动库与 USB 3.0 设备枚举状态",
                "检查 OpenCV GUI 窗口系统与 NumPy C-API 兼容性",
                "排查 yaml、PIL、matplotlib、scipy 等工业科学计算包是否就绪"
            ],
            inputs=["系统底层环境与驱动注册表"],
            outputs=["控制台输出清晰的逐项绿勾/红叉体检报告"],
            quick_tips="执行模式: 独立控制台输出诊断，遇到依赖缺失时给出精准 pip 安装命令。"
        ),
    ]
    return catalog


class GuiLauncherApp:
    """3D Vision 统一 GUI 控制中心主应用"""

    COLOR_BG = (14, 16, 20)           # 沉稳深色背景
    COLOR_CARD_BG = (22, 26, 34)      # 卡片底色
    COLOR_CARD_HOVER = (32, 40, 52)   # 卡片悬停底色
    COLOR_BORDER = (40, 50, 66)       # 边框线
    COLOR_BORDER_HOVER = (0, 255, 180)# 悬停荧光边框
    COLOR_BORDER_SEL = (0, 240, 220)  # 选中高亮边框
    COLOR_TEXT_TITLE = (245, 248, 250)
    COLOR_TEXT_MUTED = (140, 155, 170)
    COLOR_TEXT_HINT = (100, 115, 130)
    COLOR_ACCENT = (0, 255, 200)      # 主题青绿
    COLOR_GOLD = (255, 200, 50)       # 金黄色

    def __init__(self):
        self.canvas_w = 1280
        self.canvas_h = 720
        self.window_name = "flux_vision_3d | 3D 视觉综合控制中心 (Suite Dashboard)"
        self._running = True

        self.scene_mgr = CalibrationSceneManager()
        self.tools = build_tools_catalog()
        self.selected_tool_idx = 1     # 默认选中第 2 个 (Scene Hub)
        self.hover_tool_idx = 1

        self.mouse_x = -1
        self.mouse_y = -1
        self.toast_msg = "欢迎使用 flux_vision_3d 工业视觉综合控制中心！按 [2] 或点击卡片进入场景中枢。"
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

    def run(self):
        """主事件循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_w, self.canvas_h)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while self._running:
            canvas = self._render_canvas()
            cv2.imshow(self.window_name, canvas)

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
        """鼠标移动与点击事件"""
        self.mouse_x = x
        self.mouse_y = y

        # 检测鼠标悬停在哪个卡片上
        card_idx = self._hit_test_cards(x, y)
        if card_idx != -1:
            self.hover_tool_idx = card_idx

        # 鼠标左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            # 顶部右上角退出按钮 (x: 1140~1265, y: 10~44)
            if 1140 <= x <= 1265 and 10 <= y <= 44:
                self._running = False
                return

            # 右下角“一键启动当前选中工具”按钮 (x: 800~1260, y: 615~658)
            if 800 <= x <= 1260 and 615 <= y <= 658:
                self._launch_tool(self.tools[self.selected_tool_idx])
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
        """键盘快捷键响应"""
        # 回车键启动当前选中项
        if raw_key in (13, 10):
            self._launch_tool(self.tools[self.selected_tool_idx])
            return

        # 方向键选择
        # 上: 65362 / 2490368 / 38 (Windows)
        # 下: 65364 / 2621440 / 40
        # 左: 65361 / 2424832 / 37
        # 右: 65363 / 2555904 / 39
        if raw_key in (2490368, 65362, 38):  # 上
            if self.selected_tool_idx >= 2:
                self.selected_tool_idx -= 2
                self.hover_tool_idx = self.selected_tool_idx
            return
        elif raw_key in (2621440, 65364, 40):  # 下
            if self.selected_tool_idx + 2 < len(self.tools):
                self.selected_tool_idx += 2
                self.hover_tool_idx = self.selected_tool_idx
            return
        elif raw_key in (2424832, 65361, 37):  # 左
            if self.selected_tool_idx % 2 == 1:
                self.selected_tool_idx -= 1
                self.hover_tool_idx = self.selected_tool_idx
            return
        elif raw_key in (2555904, 65363, 39):  # 右
            if self.selected_tool_idx % 2 == 0 and self.selected_tool_idx + 1 < len(self.tools):
                self.selected_tool_idx += 1
                self.hover_tool_idx = self.selected_tool_idx
            return

        # 字母 / 数字快捷键直接匹配
        key_char = chr(raw_key & 0xFF).lower() if (raw_key & 0xFF) < 128 else ""

        # 特殊快捷键映射
        shortcut_map = {
            '1': "d435_live",
            '2': "scene_hub",
            '3': "asparagus_live",
            '4': "asparagus_snapshot",
            '6': "d435_mock",
            's': "tag_studio",
            'c': "tag_wizard",
            'a': "tag_ar_verifier",
            'm': "tag_map_builder",
            'l': "tag_offline_verifier",
            't': "tag_generator",
            'd': "tag_diagnose",
        }

        if key_char in shortcut_map:
            target_id = shortcut_map[key_char]
            for idx, tool in enumerate(self.tools):
                if tool.key_id == target_id:
                    self.selected_tool_idx = idx
                    self.hover_tool_idx = idx
                    self._launch_tool(tool)
                    return

    def _hit_test_cards(self, x: int, y: int) -> int:
        """测试鼠标坐标命中了哪张卡片"""
        if not (15 <= x <= 775 and 64 <= y <= 665):
            return -1

        # 卡片网格参数
        start_x, start_y = 15, 66
        card_w, card_h = 370, 70
        spacing_x, spacing_y = 12, 8
        cols = 2

        # 遍历卡片判定
        for idx in range(len(self.tools)):
            col = idx % cols
            row = idx // cols
            cx = start_x + col * (card_w + spacing_x)
            cy = start_y + row * (card_h + spacing_y)
            if cx <= x <= cx + card_w and cy <= y <= cy + card_h:
                return idx

        return -1

    def _launch_tool(self, tool: ToolCardMeta):
        """执行启动子工具或测试"""
        self.set_toast(f"正在拉起: 【{tool.title}】...")

        # 刷新画布显示 Toast
        canvas = self._render_canvas()
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(20)

        cmd = tool.command
        try:
            if tool.is_gui:
                # GUI 工具：正常子进程调用，阻塞等待其运行结束
                res = subprocess.run(cmd)
                self.set_toast(f"【{tool.title}】已退出 (退出码: {res.returncode})，系统状态已刷新。")
            else:
                # 终端/纯计算/测试工具：在 Windows 下开辟独立控制台窗口运行，绝不吞日志
                if sys.platform == "win32":
                    # 使用 cmd.exe /c "command && pause" 保证用户看得到测试输出
                    full_cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd])
                    wrapper_cmd = f'cmd.exe /c "{full_cmd_str} & echo. & echo [完成] 请按任意键返回控制中心... & pause > nul"'
                    res = subprocess.run(wrapper_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
                    self.set_toast(f"【{tool.title}】执行完毕，控制台已返回。")
                else:
                    res = subprocess.run(cmd)
                    self.set_toast(f"【{tool.title}】执行完毕 (退出码: {res.returncode})。")
        except Exception as e:
            self.set_toast(f"启动失败: {e}", duration=5.0)

        # 运行结束后刷新状态与数据
        self.refresh_system_status()

    # ========================== 核心渲染逻辑 ==========================

    def _render_canvas(self) -> np.ndarray:
        """渲染 1280x720 完整画布"""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        # 1. 顶部监控与状态栏 (y: 0~54)
        self._render_top_bar(canvas)

        # 2. 中间主分割线
        cv2.line(canvas, (780, 54), (780, 670), self.COLOR_BORDER, 1)

        # 3. 左侧工具网格区 (x: 0~780, y: 54~670)
        self._render_tools_grid(canvas)

        # 4. 右侧实时说明与大屏视窗 (x: 780~1280, y: 54~670)
        cur_idx = self.hover_tool_idx if self.hover_tool_idx != -1 else self.selected_tool_idx
        cur_tool = self.tools[cur_idx] if 0 <= cur_idx < len(self.tools) else self.tools[0]
        self._render_inspector_panel(canvas, cur_tool)

        # 5. 底部状态与快捷键指引栏 (y: 670~720)
        self._render_footer(canvas)

        return canvas

    def _render_top_bar(self, canvas: np.ndarray):
        """渲染顶部硬件与系统状态常驻监控栏"""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 54), (16, 20, 26), -1)
        cv2.line(canvas, (0, 54), (self.canvas_w, 54), self.COLOR_BORDER, 1)

        # 标题与状态灯
        cv2.circle(canvas, (24, 27), 6, self.COLOR_ACCENT, -1)
        cv2.circle(canvas, (24, 27), 9, self.COLOR_ACCENT, 1)
        draw_text(canvas, "FLUX VISION 3D", (40, 10), font_size=15, color=self.COLOR_ACCENT, bold=True)
        draw_text(canvas, "工业视觉综合控制中心", (40, 28), font_size=12, color=(160, 180, 195))

        # 状态探针芯片组 (x: 230 ~ 750)
        st = self.system_status
        py_ver = sys.version.split()[0]
        cv_ver = st.get("cv_version", cv2.__version__)
        rs_str = "D435驱动就绪" if st.get("has_realsense", False) else "D435驱动未就绪"
        rs_color = (0, 255, 140) if st.get("has_realsense", False) else (140, 150, 160)

        chip_x = 230
        cv2.rectangle(canvas, (chip_x, 12), (chip_x + 500, 42), (24, 30, 40), -1)
        cv2.rectangle(canvas, (chip_x, 12), (chip_x + 500, 42), (40, 52, 68), 1)

        # 状态小点与文字
        cv2.circle(canvas, (chip_x + 14, 27), 4, (0, 240, 160), -1)
        draw_text(canvas, f"Py {py_ver}", (chip_x + 24, 18), font_size=12, color=(200, 215, 230))

        cv2.line(canvas, (chip_x + 105, 16), (chip_x + 105, 38), (45, 58, 75), 1)
        draw_text(canvas, f"OpenCV {cv_ver}", (chip_x + 115, 18), font_size=12, color=(200, 215, 230))

        cv2.line(canvas, (chip_x + 235, 16), (chip_x + 235, 38), (45, 58, 75), 1)
        cv2.circle(canvas, (chip_x + 248, 27), 4, rs_color, -1)
        draw_text(canvas, rs_str, (chip_x + 258, 18), font_size=12, color=rs_color)

        cv2.line(canvas, (chip_x + 365, 16), (chip_x + 365, 38), (45, 58, 75), 1)
        snaps_c = st.get("snapshot_count", 0)
        draw_text(canvas, f"快照: {snaps_c}帧", (chip_x + 375, 18), font_size=12, color=(200, 215, 230))

        # 当前活动场景胶囊 (x: 740 ~ 1120)
        act_sc = self.scene_mgr.get_active_scene()
        capsule_x = 745
        capsule_w = 380
        if act_sc:
            status_tag = "★已发布生产" if act_sc.is_published else ("已平差" if act_sc.ba_solved else "草稿沙盒")
            tag_col = (0, 255, 255) if act_sc.is_published else ((0, 255, 140) if act_sc.ba_solved else (140, 180, 220))
            cv2.rectangle(canvas, (capsule_x, 10), (capsule_x + capsule_w, 44), (24, 32, 44), -1)
            cv2.rectangle(canvas, (capsule_x, 10), (capsule_x + capsule_w, 44), (0, 200, 180), 1)
            draw_text(canvas, f"当前沙盒: 【{act_sc.name}】 ({status_tag})", (capsule_x + 12, 17),
                      font_size=13, color=tag_col, bold=True)
        else:
            cv2.rectangle(canvas, (capsule_x, 10), (capsule_x + capsule_w, 44), (28, 30, 36), -1)
            cv2.rectangle(canvas, (capsule_x, 10), (capsule_x + capsule_w, 44), (50, 56, 68), 1)
            draw_text(canvas, "当前沙盒: 未选定活动场景", (capsule_x + 12, 17), font_size=13, color=(160, 170, 180))

        # 右上角 [X] 退出按钮 (x: 1140 ~ 1265, y: 10 ~ 44)
        bx, by, bw, bh = 1140, 10, 125, 34
        is_hover_exit = (bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh)
        exit_bg = (54, 24, 24) if is_hover_exit else (36, 20, 22)
        exit_border = (240, 60, 60) if is_hover_exit else (120, 40, 44)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_bg, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_border, 2 if is_hover_exit else 1)
        draw_text(canvas, "[X] 退出 [ESC]", (bx + 14, by + 9), font_size=13, color=(240, 180, 180), bold=True)

    def _render_tools_grid(self, canvas: np.ndarray):
        """渲染左侧工具卡片网格 (2 列排布)"""
        start_x, start_y = 15, 66
        card_w, card_h = 370, 70
        spacing_x, spacing_y = 12, 8
        cols = 2

        for idx, tool in enumerate(self.tools):
            col = idx % cols
            row = idx // cols
            cx = start_x + col * (card_w + spacing_x)
            cy = start_y + row * (card_h + spacing_y)

            is_selected = (idx == self.selected_tool_idx)
            is_hover = (idx == self.hover_tool_idx)

            # 底色与线框计算
            if is_selected:
                card_bg = (34, 46, 60)
                card_border = (0, 240, 220)
                border_th = 2
            elif is_hover:
                card_bg = self.COLOR_CARD_HOVER
                card_border = self.COLOR_BORDER_HOVER
                border_th = 1
            else:
                card_bg = self.COLOR_CARD_BG
                card_border = self.COLOR_BORDER
                border_th = 1

            cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), card_bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), card_border, border_th)

            # 左侧主题色条
            cv2.rectangle(canvas, (cx, cy), (cx + 5, cy + card_h), tool.tag_color, -1)

            # 快捷键小徽章 (Badge)
            badge_w = 28 if len(tool.shortcut) <= 2 else 34
            cv2.rectangle(canvas, (cx + 12, cy + 10), (cx + 12 + badge_w, cy + 30), (16, 22, 30), -1)
            cv2.rectangle(canvas, (cx + 12, cy + 10), (cx + 12 + badge_w, cy + 30), tool.tag_color, 1)
            draw_text(canvas, tool.shortcut, (cx + 16, cy + 12), font_size=12, color=tool.tag_color, bold=True)

            # 标题与副标题
            title_col = self.COLOR_TEXT_TITLE if (is_selected or is_hover) else (215, 225, 235)
            draw_text(canvas, tool.title, (cx + 18 + badge_w, cy + 10), font_size=14, color=title_col, bold=True)

            sub_col = (180, 195, 205) if is_hover else self.COLOR_TEXT_MUTED
            draw_text(canvas, tool.subtitle[:24], (cx + 14, cy + 36), font_size=12, color=sub_col)

            # 运行模式标记 (GUI 视窗 / 控制台)
            mode_text = "GUI 交互" if tool.is_gui else "控制台"
            mode_color = (0, 200, 140) if tool.is_gui else (160, 170, 200)
            cv2.putText(canvas, mode_text, (cx + card_w - 56, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, mode_color, 1, cv2.LINE_AA)

    def _render_inspector_panel(self, canvas: np.ndarray, tool: ToolCardMeta):
        """渲染右侧动态即时说明大屏 (Live Inspector Panel)"""
        px, py = 790, 66
        pw, ph = 480, 595

        # 大屏底板
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 22, 30), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (36, 46, 62), 1)

        # 头部标题带
        cv2.rectangle(canvas, (px, py), (px + pw, py + 64), (22, 28, 38), -1)
        cv2.line(canvas, (px, py + 64), (px + pw, py + 64), (45, 58, 76), 1)

        # 快捷键徽章与主标题
        cv2.rectangle(canvas, (px + 16, py + 16), (px + 52, py + 48), (14, 18, 24), -1)
        cv2.rectangle(canvas, (px + 16, py + 16), (px + 52, py + 48), tool.tag_color, 2)
        draw_text(canvas, f"[{tool.shortcut}]", (px + 20, py + 22), font_size=14, color=tool.tag_color, bold=True)

        draw_text(canvas, tool.title, (px + 62, py + 14), font_size=18, color=self.COLOR_TEXT_TITLE, bold=True)
        draw_text(canvas, f"类别: {tool.category}  |  模式: {'原生 OpenCV GUI' if tool.is_gui else '独立 Console 控制台'}",
                  (px + 62, py + 40), font_size=12, color=tool.tag_color)

        # 核心概述 (Summary)
        curr_y = py + 78
        draw_text(canvas, "【功能定位与现场痛点】", (px + 16, curr_y), font_size=13, color=self.COLOR_ACCENT, bold=True)
        curr_y += 24
        draw_text(canvas, tool.summary, (px + 20, curr_y), font_size=13, color=(220, 230, 240))
        curr_y += 36

        # 详细特性清单 (Details)
        draw_text(canvas, "【工程要点与执行逻辑】", (px + 16, curr_y), font_size=13, color=self.COLOR_ACCENT, bold=True)
        curr_y += 24
        for d in tool.details:
            cv2.circle(canvas, (px + 24, curr_y + 8), 3, tool.tag_color, -1)
            draw_text(canvas, d, (px + 34, curr_y), font_size=12, color=(200, 212, 225))
            curr_y += 22
        curr_y += 10

        # 前置依赖与输入 (Inputs)
        draw_text(canvas, "【前置条件与输入依赖】", (px + 16, curr_y), font_size=13, color=self.COLOR_GOLD, bold=True)
        curr_y += 22
        for inp in tool.inputs:
            cv2.circle(canvas, (px + 24, curr_y + 8), 3, self.COLOR_GOLD, -1)
            draw_text(canvas, inp, (px + 34, curr_y), font_size=12, color=(225, 215, 195))
            curr_y += 20
        curr_y += 10

        # 输出产物 (Outputs)
        draw_text(canvas, "【输出产物与持久化路径】", (px + 16, curr_y), font_size=13, color=(0, 240, 160), bold=True)
        curr_y += 22
        for out in tool.outputs:
            cv2.circle(canvas, (px + 24, curr_y + 8), 3, (0, 240, 160), -1)
            draw_text(canvas, out, (px + 34, curr_y), font_size=12, color=(190, 235, 215))
            curr_y += 20
        curr_y += 12

        # 操作提示 (Quick Tips)
        cv2.rectangle(canvas, (px + 16, curr_y), (px + pw - 16, curr_y + 36), (22, 28, 38), -1)
        cv2.rectangle(canvas, (px + 16, curr_y), (px + pw - 16, curr_y + 36), (40, 52, 70), 1)
        draw_text(canvas, tool.quick_tips, (px + 24, curr_y + 9), font_size=12, color=self.COLOR_ACCENT)

        # 底部醒目启动卡片按钮 (x: px+16 ~ px+pw-16, y: py+ph-52 ~ py+ph-12)
        btn_y = py + ph - 50
        btn_h = 38
        is_hover_btn = (px + 16 <= self.mouse_x <= px + pw - 16 and btn_y <= self.mouse_y <= btn_y + btn_h)
        btn_bg = (0, 220, 160) if is_hover_btn else (0, 180, 130)
        cv2.rectangle(canvas, (px + 16, btn_y), (px + pw - 16, btn_y + btn_h), btn_bg, -1)
        btn_text = f"▶ 立即启动: 【{tool.title}】 (回车 ⏎ 或 双击卡片)"
        draw_text(canvas, btn_text, (px + 45, btn_y + 10), font_size=14, color=(10, 20, 25), bold=True)

    def _render_footer(self, canvas: np.ndarray):
        """渲染底部状态反馈与快捷键指引栏"""
        fy = 670
        cv2.rectangle(canvas, (0, fy), (self.canvas_w, self.canvas_h), (12, 14, 18), -1)
        cv2.line(canvas, (0, fy), (self.canvas_w, fy), self.COLOR_BORDER, 1)

        # 1. 动态 Toast 消息反馈
        now = time.time()
        if self.toast_time > now:
            draw_text(canvas, f"[中枢状态] {self.toast_msg}", (20, fy + 14),
                      font_size=14, color=(0, 255, 200), bold=True)
        else:
            draw_text(canvas, "[1~6] 生产核心  [S/C/A/M/L/T/D] 标定流水线  [↑/↓/←/→] 浏览说明  [⏎/双击] 启动  [ESC] 退出",
                      (20, fy + 16), font_size=13, color=(170, 185, 200))

        # 2. 右侧系统时钟与运行提示
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        draw_text(canvas, f"系统时间: {time_str} | FLUX VISION 3D", (self.canvas_w - 380, fy + 16),
                  font_size=13, color=self.COLOR_TEXT_MUTED)


def main():
    parser = argparse.ArgumentParser(description="3D 视觉综合控制中心 (3D Vision Suite GUI Launcher)")
    args = parser.parse_args()

    app = GuiLauncherApp()
    app.run()


if __name__ == "__main__":
    main()
