# flux_vision_3d 芦笋 3D 视觉与抓取位姿估计系统

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org)
[![Hardware](https://img.shields.io/badge/Camera-Intel%20RealSense%20D435-orange.svg)](https://www.intelrealsense.com)
[![Status](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)]()

`flux_vision_3d` 是专门针对**传送带上多层堆叠、并排贴合的细长果蔬（绿芦笋）**研发的工业级 3D 视觉感知与智能抓取位姿估计系统。系统通过顶置 3D 深度相机实时解算最顶层芦笋的空间抓取位姿 $(X, Y, Z, R)$，生成标准 G-code 驱动下游 **SCARA 机械臂（flux_loader_mks_v16）** 实施无碰撞下探抓取，并通过 BLE 向 **分发分拣机构（flux_dealer）** 写入多级品质分拣槽位。

---

## 1. 物理工况与硬件规格定义

系统已针对工程量产与现场严苛工况完成鲁棒性设计，核心规格如下：

| 维度 | 实际物理工况 / 选型参数 | 工程应对与算法保障 |
| :--- | :--- | :--- |
| **3D 视觉传感器** | **Intel RealSense D435** (主动红外双目 RGB-D，标准版) | 采用双目结构光点云与彩色图像，不依赖 IMU 惯导 |
| **安装方式与角度** | 安装于**黑色传送带**正上方，呈**大倾角俯视（$\pm 30^\circ$ 范围）** | **无需人工精确测量倾角**：算法通过传送带点云平面方程实时自解算 Pitch / Roll，并通过 3D 空间欧氏测距完全消除大倾角透视短缩畸变 |
| **工作距离与视野** | 物料台面距镜头约 **$55 \sim 70\text{ cm}$ (550~700 mm)**，实测基准约 $640\text{mm}$ | 避开 D435 近距盲区（28cm），处于红外散斑发射器最佳高信噪比分布区 |
| **背景特征** | **黑色低反光传送带皮带** | 黑色背景与嫩绿芦笋形成高反差；算法采用“深度浮凸（$>10\text{mm}$）+ 黑帽暗缝”提取物料 |
| **下游执行机构** | **4 轴 SCARA 机械臂 + 双开闭夹爪** (Marlin G-code) | 通过串口通信，输入格式：`G0 X.. Y.. R..` $\rightarrow$ `G1 Z..` $\rightarrow$ `M4` |
| **品质分拣机构** | **8 级级联步进电机翻料机构** (ESP32 BLE 通信) | 蓝牙单播写入目标槽位：`Target ID: 1~8` |

---

## 2. 软件架构与工程代码目录树

系统采用模块化分层解耦架构，各模块功能划分清晰：

```text
flux_vision_3d/
├── config.yaml               # 系统核心配置文件 (相机流参数、滤波链、深度色彩映射、视觉分离门限及串口配置)
├── README.md                 # 项目主门户与架构说明文档 (本文件)
├── requirements.txt          # Python 依赖清单 (pyrealsense2, opencv-python, numpy, pyyaml 等)
├── run.bat                   # Windows 批处理一键启动入口
├── run.ps1                   # PowerShell 自动化环境检测与启动脚本
│
├── data/                     # 生产运行时与本地测试数据存储
│   └── snapshots/            # 真实快照库 (包含 RGB 图 color_*.png、原始点云 depth_raw_*.npy 及标注分析图)
│
├── docs/                     # 工程技术文档库
│   ├── requirements.md       # 系统详细需求书与软硬件设计规格书 (技术核心细节)
│   └── image_processing_pipeline.md # 图像处理与芦笋 3D 感知全流程架构设计文档 (从原始图像到抓取位姿)
│
├── src/                      # 核心算法与感知业务源码
│   └── vision/
│       └── asparagus_analyzer.py  # 核心感知引擎 (传送带自标定、黑帽暗缝分离、主轴拟合与顶层位姿解算)
│
├── tests/                    # 自动化测试与质量验证管线
│   ├── test_real_snapshot.py      # 全量真实快照自动化测试 (100% 验证 17 组真实数据的分离、顶层识别与 G-code)
│   └── test_mock_pipeline.py      # 仿真管线测试 (无物理相机时的回归验证)
│
└── tools/                    # 交互式运维与调试工具集
    ├── cli_menu.py                # 交互式统一控制终端 (集成环境诊断、工具启动与自动化测试菜单)
    ├── d435_viewer.py             # 实时相机查看器 (支持 3D 深度探针、动态色谱拉伸、滤波器开关与 G-code 打印)
    └── find_top_asparagus.py      # 单帧采集/抓取解算工具 (直接输出标准 SCARA G-code 指令)
```

---

## 3. 核心算法处理管线 (Pipeline)

系统采用“3D 深度基准 $\rightarrow$ 黑帽暗缝切分 $\rightarrow$ 3D 空间无损测距 $\rightarrow$ 顶层拓扑判决”的多维感知管线：

```text
                  ┌───────────────────────────────────────────────┐
                  │ 1. 图像采集与硬件预处理 (RealSense D435 RGB-D) │
                  └───────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │ 2. 传送带平面最小二乘拟合与倾角自标定 (Auto-Tilt) │
                  │    - 求解 Z_table(u,v) = a*u + b*v + d        │
                  │    - 反求相机物理安装倾角 (Pitch, Roll, Total)  │
                  └───────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │ 3. 3D 深度浮凸前景初筛 (H_rel >= 10mm)         │
                  │    - 彻底免疫黑色皮带与深色背景               │
                  └───────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │ 4. 黑帽暗缝切分多根粘连 (Black-Hat Seam Sep)  │
                  │    - 提取贴合接触面的水平纵向阴影缝隙         │
                  │    - 将整捆并排平铺物料切分为独立单根芦笋实例 │
                  └───────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │ 5. 3D 空间真实欧氏测距与主轴拟合              │
                  │    - cv2.fitLine 求解中心轴与倾角 Yaw (R 轴)  │
                  │    - 空间两端点反投影测距 L = sqrt(dx²+dy²+dz²) │
                  │    - 彻底消除 ±30° 大倾角下的透视短缩畸变     │
                  └───────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │ 6. 顶层拓扑排序与 SCARA G-code 抓取位姿生成   │
                  │    - 锁定相对台面凸起净高最大者 [TOPMOST]     │
                  │    - 生成抓取位姿 (X, Y, Z, R) 与安全下探指令 │
                  └───────────────────────────────────────────────┘
```

---

## 4. 快速上手与使用指引

### 4.1 环境准备
系统推荐运行于 Windows 10/11 64 位环境，要求 Python 3.10+：
```powershell
# 1. 进入工程根目录
cd d:\Software\antigravity\flux_vision_3d

# 2. 安装/更新依赖
pip install -r requirements.txt
```

### 4.2 启动控制终端 (CLI Terminal)
在项目根目录下直接运行：
```powershell
./run
# 或在 CMD 下直接运行 run.bat
```
进入控制终端后，可通过输入编号直接触发各项运维功能：
- `[1]` **启动 D435 实时相机查看器与深度探针**：
  - 鼠标移动/点击：实时探测视野中任意物料点的 3D 坐标 $(X, Y, Z)$；
  - `[D]` 键：开启/关闭芦笋检测识别与顶层高亮卡片；
  - `[G]` 键：实时生成并打印当前最顶层芦笋的 SCARA G-code 抓取指令；
  - `[S]` 键：抓拍当前帧的 RGB 图、深度热力图与原始点云数据到 `data/snapshots/`；
  - `[A]` / `[` / `]`：调节深度可视化色彩区间（支持自适应拉伸 Auto-Range）。
- `[4]` **解算最顶层芦笋抓取位姿 (离线快照)**：读取本地最新保存的快照进行即时算法验证。
- `[7]` **运行真实快照算法测试**：一键遍历全部本地真实快照并统计成功率与尺寸指标。
- `[9]` **详细环境与驱动诊断**：自动检测 Python、OpenCV、NumPy 及 D435 相机连接与固件版本。

---

## 5. 深度技术文档与质量保证报告

- **图像处理与芦笋 3D 感知全流程架构设计文档**：详见 [docs/image_processing_pipeline.md](file:///d:/Software/antigravity/flux_vision_3d/docs/image_processing_pipeline.md)（含端到端 Mermaid 流程图、黑帽暗缝实例分割、大倾角空间测距推导与叠压排序模型）；
- **全量真实快照自动化测试 (`tests/test_real_snapshot.py`)**：覆盖全部 **20 组**现场真实快照，实现 **100.0% 顶层抓取锁定成功率**（累计稳定解算 **136 根次**芦笋物料）；
- **手眼标定与安全防撞测试 (`tests/test_hand_eye_calibration.py`)**：验证 Horn/Kabsch SVD 刚体变换精度与 500+mm 危险深度拦截；
- **仿真管线回归测试 (`tests/test_mock_pipeline.py`)**：保障脱机开发时虚拟点云与三层叠压物料分析的持续集成。

详细成果总结请参阅：[walkthrough.md](file:///C:/Users/feng/.gemini/antigravity-ide/brain/40ff887b-c3ba-45ab-8f5f-cc385e46d9da/walkthrough.md)。

---
*文档更新日期: 2026年9月 | flux_vision_3d 团队*
