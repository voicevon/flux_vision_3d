# AprilTag 观测样本轻量级 OpenCV 交互审核画板设计规范

## 1. 背景与目标
在多标靶 3D 空间建图与 BA 平差流水线中，工程师需要剔除大倾角透视严重失真、模糊或边缘畸变的劣质观测。此前虽然设计了结构化 YAML 清单（`tag_observations.yaml`），但需在千行文本中定位并手动修改 `keep: false`，存在视窗割裂和容易误删关键桥梁的痛点。

本设计旨在通过零外部依赖的 OpenCV 原生窗口，提供一个即开即用、所见即所得的交互式审核画板（`tools/tag_manifest_reviewer.py`）。

---

## 2. 交互与核心功能设计

### 2.1 窗口与画布
- **窗口命名**：`[AprilTag 观测审核画板] (点击标靶剔除/恢复 | A/D翻页 | S保存 | Q退出)`
- **顶部半透明状态栏**：
  - 左侧：当前图像索引与文件名，如 `[3/15] view_0003.png`
  - 中间：当前帧标靶统计，如 `[观测: 4个 | 保留: 3 | 剔除: 1]`
  - 右侧：**实时共视拓扑安全指示灯**：
    - 亮绿：`[拓扑状态: 健康 PASS]`
    - 红色爆闪告警：`[严重警告: Tag #29 已断网失联！请恢复关键桥梁]`
  - 底部操作指引提示。

### 2.2 标靶双态渲染 (Dual-State Rendering)
- **保留状态 (`keep: true`)**：
  - 绿色外多边形轮廓线（厚度 2）+ 4 个彩色顶点；
  - 工业标牌：`Tag ID` + `Cell: WxH px`；
  - 20mm 天蓝半透明 3D 实心正四棱柱 Z 轴。
- **剔除状态 (`keep: false`)**：
  - 红色外多边形轮廓线（厚度 2）；
  - 标靶内部覆盖一层红色半透明蒙版（alpha=0.35）；
  - 标靶中心打上明显的红色对角大叉号（$\times$）；
  - 标牌文字变更为：`Tag ID [已剔除 EXCLUDED]`，隐藏 3D 棱柱。

### 2.3 鼠标交互机制
- 使用 `cv2.setMouseCallback` 监听 `EVENT_LBUTTONDOWN`；
- 获取点击像素坐标 $(x, y)$，通过 `cv2.pointPolygonTest` 依次测试当前帧所有标靶的多边形；
- 命中时，立即将该标靶的 `keep` 字段在 `True` 与 `False` 之间翻转；
- 翻转后立即在内存中调用 `TagMapBuilder.validate_covisibility()`，实现拓扑状态毫秒级实时刷新。

### 2.4 键盘快捷键体系
- `A` / `←` / `P`：上一张图片；
- `D` / `→` / `N`：下一张图片；
- `R`：重置当前图片所有标靶为初始状态；
- `S`：即时保存写回 `tag_observations.yaml`；
- `Q` / `ESC` / `Space`：保存并退出，返回控制台。

---

## 3. 架构集成

1. **新建独立模块**：`tools/tag_manifest_reviewer.py`；
2. **控制台联动**：
   - 在 `tools/tag_map_builder.py` 交互菜单选项 `[2]` 中直接拉起图形审核器；
   - 在 `tools/cli_menu.py` 标定专区选项 `[O]` 或 `[3]` 中支持一键启动；
   - 支持命令行直接执行：`python tools/tag_manifest_reviewer.py`。
