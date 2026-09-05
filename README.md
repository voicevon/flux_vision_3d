# flux_vision_3d (芦笋 3D 视觉分拣系统)

基于 **Intel RealSense D435 RGB-D 3D 深度相机** 的果蔬堆叠识别与抓取位姿估计系统，作为分拣系统的核心大脑（Vision Subsystem）。

## 项目定位

本工程负责感知料槽/传送带上的多层堆叠芦笋，解决细长果蔬边缘飞点、高光反光空洞与遮挡判顶难题，解算最顶层无遮挡芦笋的空间位姿 $(X, Y, Z, R)$，通过串口发送 G-code 驱动下游 **SCARA 机械臂（flux_loader_mks_v16）** 抓取，并通过 BLE 向 **分发机构（flux_dealer）** 写入分拣槽位。

## 详细技术规格与需求文档

完整系统需求、硬件工况、抗噪算法管线（2D 掩膜腐蚀脊线提取 + 拓扑遮挡有向图 Occlusion DAG）及分阶段交付里程碑，请参阅：
* 📄 **[需求与设计规格书 (docs/requirements.md)](docs/requirements.md)**

## 快速概览

* **工况配置**：D435 俯视垂直安装，工作高度约 70cm，工作视野 50~60cm；
* **核心算法**：
  1. `pyrealsense2` 硬件级空间/时间滤波 + 彩色流硬件对齐 (`align_to_color`)；
  2. 2D 实例分割（YOLOv8-seg / 传统自适应轮廓）；
  3. 掩膜形态学腐蚀（Erosion -6px）提取芦笋中心轴脊线，有效滤除边缘飞点与粘连；
  4. 脊线前 15% 分位数统计滤波获取绝对顶面深度 $Z_{\text{top}}$，免疫湿润表面反光黑洞；
  5. 遮挡有向无环图（Occlusion DAG）拓扑分析，提取入度为 0（In-degree = 0）的最顶层目标；
  6. 求解空间抓取点 $(X, Y, Z)$ 与姿态角 $R$，生成 SCARA G-code 抓取指令。

---
*由 Antigravity 辅助构建与维护*
