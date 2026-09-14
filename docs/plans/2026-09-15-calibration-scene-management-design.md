# 采样场景与批次分组管理系统技术设计方案 (Calibration Scene Management System Design)

## 1. 概述与背景

在机器视觉与多标靶空间几何建图中，一次标定流程包含“图像采集 -> 观测审核 -> BA 平差求解 -> 空间地图导出 -> 盲测体检验证”的完整闭环。在实际生产与调试过程中，标定需求呈现高度的场景化与批次化特征（如不同的安装机台、相机高低视角、光照工况测试、机械臂不同架次等）。

目前系统中采图路径、审核清单与导出的三维地图固定单一，导致换工况时必须清空或覆盖历史数据，多场景无法并存、无法切换比对、亦无法回溯。

本设计提出**高内聚自包含场景包模式（Self-Contained Scene Bundle）**，结合**日期前缀与自定义别名命名机制**，全面实现标定场景的分组、隔离、切换、发布与历史归档。

---

## 2. 目录结构与数据模型设计

### 2.1 物理目录架构
所有标定场景集中存放在 `data/calibration_scenes/` 目录下（已被 `.gitignore` 保护）：

```text
flux_vision_3d/
├── data/
│   └── calibration_scenes/                 # 场景总库沙盒
│       ├── .active_scene                  # 纯文本标记，记录当前活动场景目录名
│       │
│       ├── 20260915_bench_default/        # 自动迁移历史数据生成的默认场景
│       │   ├── scene_meta.yaml            # 场景自描述元数据
│       │   ├── raw_images/                # 原始无标注高清采图 (view_0001.png ...)
│       │   ├── tag_observations.yaml      # 审核清单 (角点清单与保留/剔除决策)
│       │   ├── tags_map.yaml              # 本场景平差解算出的专属 3D 几何地图
│       │   ├── tags_map.yaml.bak          # 地图历史备份
│       │   ├── visualized/                # Quiver 残差场矢量图与标注图切片
│       │   └── reports/                   # 质检报告与盲测体检报告
│       │
│       └── 20260915_shelf_high/           # 用户新建的高位场景
│           ├── scene_meta.yaml
│           ├── raw_images/
│           ├── tag_observations.yaml
│           ├── tags_map.yaml
│           ├── visualized/
│           └── reports/
│
├── config/
│   ├── config.yaml                        # 全局配置 (记录 active_scene 字段)
│   └── tags_map.yaml                      # 生产环境运行地图 (由当前激活场景一键发布)
```

### 2.2 场景元数据定义 (`scene_meta.yaml`)

```yaml
scene_id: "20260915_shelf_high"           # 场景唯一标识符 (目录名)
name: "shelf_high"                        # 操作员自定义别名
description: "SCARA 高位俯视视角标定"     # 场景描述
created_at: "2026-09-15 02:10:00"         # 创建时间
updated_at: "2026-09-15 02:25:30"         # 最后修改时间
camera_serial: "843112071234"             # 采集相机硬件序列号
valid_tag_ids: [0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
origin_tag_id: 0
x_axis_tag_id: 28
status:
  image_count: 22                         # 采集图像总数
  active_image_count: 21                  # 有效保留图像数
  ba_solved: true                         # 是否已完成平差计算
  global_rmse_px: 0.185                   # 当前地图全局重投影均方根误差 (px)
  is_published: true                      # 是否已发布为生产全局地图 (config/tags_map.yaml)
```

### 2.3 历史数据平滑向前兼容 (Auto-Migration)
系统启动或载入场景时进行自动检测：
1. 若 `data/calibration_scenes/` 不存在，且旧路径 `data/tag_calibration_images/` 存在有效数据：
   - 自动创建 `data/calibration_scenes/20260915_legacy_default/`；
   - 将原图像平滑迁移至 `raw_images/`；
   - 迁移 `tag_observations.yaml`、`visualized/` 及 `reports/`；
   - 若 `config/tags_map.yaml` 存在，复制至场景内并生成初始 `scene_meta.yaml`；
   - 将 `.active_scene` 写入 `20260915_legacy_default`。
2. 现有单测与开发环境完全无损过渡。

---

## 3. 核心软件架构与组件

### 3.1 场景管理器 (`CalibrationSceneManager`)
在 `src/calibration/scene_manager.py` 实现统一的管理器组件：

- **类定义**：`class CalibrationSceneManager`
- **核心数据结构**：`class CalibrationScene`（封装单场景所有路径属性与状态读取）
- **核心方法**：
  - `list_scenes() -> List[CalibrationScene]`：遍历场景沙盒，按创建时间降序排序返回场景列表；
  - `get_active_scene() -> CalibrationScene`：解析 `.active_scene`，返回当前活动场景对象；若无场景则自动初始化一个默认场景；
  - `set_active_scene(scene_id: str) -> bool`：切换当前活动场景；
  - `create_scene(alias: str, description: str = "") -> CalibrationScene`：生成标准化目录名 `YYYYMMDD_<alias>`，创建骨架目录与初始元数据；
  - `clone_scene(src_scene_id: str, new_alias: str) -> CalibrationScene`：无损克隆已有场景作为对照实验；
  - `publish_to_production(scene_id: Optional[str] = None) -> bool`：将场景中的 `tags_map.yaml` 复制发布为 `config/tags_map.yaml`，并更新 `config.yaml` 中的 `calibration.active_scene` 字段；
  - `delete_scene(scene_id: str) -> bool`：安全删除废弃场景（禁止删除正在活动的场景）。

### 3.2 CLI 终端菜单集成 (`tools/cli_menu.py`)
1. **标定子菜单顶部动态状态卡片**：
   - 展示：`【当前活动场景】: 20260915_bench_a (22帧 | RMSE: 0.18px | ★已发布为生产地图)`
   - 目录直观提示：`data/calibration_scenes/20260915_bench_a/`
2. **新增 `[0]` 场景管理专属菜单**：
   - `[L] 列表浏览与对比所有场景`
   - `[S] 快速切换活动场景`
   - `[N] 新建标定采样场景`
   - `[P] 一键将当前场景发布为生产配置`
3. **下游工具链路联动**：
   - 工序 2 (采图向导): 自动写入当前活动场景的 `raw_images/`；
   - 工序 S (离线 Studio): 自动加载当前活动场景的数据集、审核清单和地图；
   - 工序 3 (超精提取)、5 (建图求解)、6 (离线体检): 全部动态对齐当前活动场景。

### 3.3 离线 Studio 集成 (`tools/calibration/tag_offline_studio.py`)
1. **顶部状态栏场景提示**：显示当前正在编辑审核的场景标识；
2. **保存与发布快捷指令**：
   - `Ctrl+S` / 菜单保存：持久化当前场景的 `tags_map.yaml` 和 `scene_meta.yaml`；
   - 新增 `Shift+P` 快捷键：一键弹窗确认并发布到全局 `config/tags_map.yaml`；
3. **一键清空守门保护**：清空操作仅清空当前场景内的观测清单和本场景地图，对其他场景形成绝对沙盒物理隔离。

---

## 4. 生产环境零破损设计

生产实时检测与抓取系统（如 `asparagus_analyzer.py`、`tag_localizer.py`）无需任何侵入式重构：
- 依然默认加载 `config/tags_map.yaml`；
- `config.yaml` 记录 `active_scene` 仅作为版本追溯元数据；
- 发布机制保证生产环境永远使用的是经过确认的、有据可查的场景平差成果。

---

## 5. 验证规划

1. **单元测试 (`tests/test_scene_manager.py`)**：
   - 场景创建、枚举、重名校验、目录生成；
   - 活动场景的切换与持久化；
   - 场景克隆与独立性校验（修改克隆体不影响本体）；
   - 历史数据自动迁移（Auto-migration）正确性；
   - 发布至生产环境（`config/tags_map.yaml` 同步）断言。
2. **端到端工作流验证**：
   - 模拟新建 `20260915_test_exp` -> 存入测试帧 -> 运行 Studio 求解 -> 发布为生产地图 -> 检查 `config/tags_map.yaml` 成功对齐。
