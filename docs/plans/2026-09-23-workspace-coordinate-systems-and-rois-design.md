# Workspace 多坐标系与 ROI 空间设计方案

## 1. 概述与背景

在工业立体视觉系统（`flux_vision_3d`）中，每个工位（`Workspace`）除了基于全局 AprilTag 锚点优化出的绝对世界坐标系外，现场机械构件往往存在相对运动或局部装配基准（例如：固定机架、驱动轮系、可移动同步带/滑块等）。此外，视觉算法（如点云截取、芦笋/工件定位、防碰撞检测等）需要关注特定的 3D 物理空间区域（即 ROI 空间）。

本方案在工位沙盒架构下引入：
1. **多坐标系拓扑体系（Coordinate Frame Tree）**：以唯一的绝对世界坐标系（`world`）为根，支持按需配置任意数量的相对坐标系（通常 1~2 个），支持基于固定刚体变换外参或 AprilTag 动标检测绑定的混合定义方式。
2. **ROI 空间物件集合（ROI Space Collection）**：每个工位下可定义多个挂载于指定坐标系（世界系或相对系）的 3D 有向长方体空间（OBB），代表具体的物理部件（同步带、皮带、驱动轮、托盘等），并提供点云裁剪过滤与 3D 渲染支持。

---

## 2. 总体架构与数据模型

### 2.1 文件沙盒解耦

每个 Workspace 的物理根目录下新增两个完全解耦的独立资产配置文件：
* `frames.yaml`：声明本工位的坐标系树（世界系 + 各级相对坐标系）。
* `rois.yaml`：声明本工位的 ROI 空间物件集合。

在工位新建、克隆与备份时，这两个资产与现有的 `tags_map.yaml`、`anchor_tags.yaml` 保持一致的沙盒生命周期管理。

### 2.2 `frames.yaml` 规范

```yaml
version: "1.0"
workspace_id: "20260915_bench_default"
active_frame_id: "world"

frames:
  # 1. 绝对世界坐标系 (唯一根节点，parent 为空)
  - frame_id: "world"
    name: "绝对世界坐标系"
    parent_frame_id: null
    type: "world"
    description: "工位绝对基准坐标系，由 AprilTag 锚点优化得出"

  # 2. 相对坐标系 (固定外参类型: 刚体平移与欧拉角旋转)
  - frame_id: "frame_conveyor"
    name: "主输送机坐标系"
    parent_frame_id: "world"
    type: "fixed_transform"
    transform:
      translation_xyz_mm: [150.0, 0.0, 50.0]
      rotation_rpy_deg: [0.0, 0.0, 15.0]

  # 3. 相对坐标系 (动标绑定类型: 绑定 AprilTag 并在其基础上叠加偏移)
  - frame_id: "frame_belt_slider"
    name: "同步带滑块坐标系"
    parent_frame_id: "world"
    type: "tag_bound"
    tag_binding:
      tag_id: 12
      offset_xyz_mm: [0.0, 0.0, 10.0]
      offset_rpy_deg: [0.0, 0.0, 0.0]
```

### 2.3 `rois.yaml` 规范

```yaml
version: "1.0"
workspace_id: "20260915_bench_default"

rois:
  # 物件 1: 同步带工作面 (挂载在同步带滑块坐标系)
  - roi_id: "roi_sync_belt_01"
    name: "1号同步带工作面"
    frame_id: "frame_belt_slider"
    category: "belt"                # belt, wheel, tray, general
    enabled: true
    geometry:
      type: "box"                   # 3D OBB
      center_xyz_mm: [0.0, 100.0, 20.0]
      size_xyz_mm: [60.0, 300.0, 15.0]
      rotation_rpy_deg: [0.0, 0.0, 0.0]
    visual_color_rgb: [0, 255, 128]
    description: "同步带上表面主物料通过区"

  # 物件 2: 驱动轮干涉区 (挂载在主输送机坐标系)
  - roi_id: "roi_drive_wheel"
    name: "主驱动轮干涉区"
    frame_id: "frame_conveyor"
    category: "wheel"
    enabled: true
    geometry:
      type: "box"
      center_xyz_mm: [50.0, -20.0, 0.0]
      size_xyz_mm: [80.0, 80.0, 40.0]
      rotation_rpy_deg: [0.0, 90.0, 0.0]
    visual_color_rgb: [255, 165, 0]
    description: "驱动轮轮体空间"
```

---

## 3. 核心计算与服务层设计

系统在 `src/calibration/` 下提供底层引擎支持：

### 3.1 `CoordinateTreeManager`
* **矩阵级联计算**：将任意挂载在父坐标系或 AprilTag 上的相对坐标系递归解算为相对于世界坐标系的 $4 \times 4$ 齐次变换矩阵 $T_{world \to frame} \in SE(3)$。
* **点/位姿转换**：
  * `get_transform(src_frame, dst_frame) -> np.ndarray`
  * `transform_points(points, src_frame, dst_frame) -> np.ndarray`
* **异常防御**：对循环引用进行拓扑检查，对未在 `tags_map.yaml` 中检出的 Tag 动标给予警告并标记状态，避免阻断系统。

### 3.2 `RoiSpaceManager`
* **世界 OBB 求解**：基于局部中心、尺寸与旋转，结合所属坐标系的 $T_{world \to frame}$，计算输出 8 个角点的绝对世界坐标（$8 \times 3$ 顶点）及主轴方向向量。
* **点云高效裁剪判定**：提供 `filter_point_cloud(points_world, roi_id, keep_inside=True)`，将点云转换到 ROI 本地坐标系下进行极速 AABB 区间判定。

---

## 4. 业务与界面集成设计

### 4.1 Workspace 数据模型扩展
* 在 `Workspace` 类中增加 `frames_path` 和 `rois_path` 路径属性。
* 在 `create_workspace()` 时自动写入默认模板配置。
* 在 `clone_workspace()` 时深度拷贝对应文件。

### 4.2 Workspace Hub 看板与专属页签
* **专属页签**：在顶部 Header 设立专属 Tab `[ 坐标系&ROI ]`，集中呈现多坐标系拓扑树与 3D ROI 空间集合；
* **联动指示**：在 Dashboard「3D 空间拓扑与几何网络健康度」板块中同步显示坐标系与 ROI 数量概要。

---

## 5. 结构化 GUI 表单编辑器与强 Schema 校验设计

为了避免手动编辑 YAML 引起的语法错误、悬空父级或非法数值，Hub 内部提供基于强 Schema 约束的弹窗表单体系：

### 5.1 坐标系编辑/新增表单 (`FrameModal`)
* **父坐标系受控单选**：仅提供当前工位已有合法坐标系胶囊供单选，并自动剔除自身与子节点，杜绝拓扑环路与拼写错误；
* **类型单选切换**：`[固定外参]` 与 `[动标绑定]`；
* **数值规范化录入**：平移、旋转或 Tag ID 与局部偏移量，均通过受控输入并校验；
* **保存写穿**：调用 `CoordinateTreeManager.save()` 安全写回 `frames.yaml`。

### 5.2 3D ROI 空间物件编辑/新增表单 (`RoiModal`)
* **所属坐标系受控单选**：直接点击已有坐标系胶囊挂载；
* **部件类别快捷徽章**：`[同步带 belt]`、`[驱动轮 wheel]`、`[料盘 tray]`、`[通用 general]`；
* **强 Schema 约束**：长宽高尺寸必须严格 $> 0$，非法字符与非正数实时拦截；
* **世界位姿实时求解与反馈**：弹窗内实时显示世界中心与 8 顶点 OBB 就绪状态；
* **保存写穿**：调用 `RoiSpaceManager.save()` 安全写回 `rois.yaml`。

