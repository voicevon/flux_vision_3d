# 离线精度体检工具实现计划

## 目标

构建独立的 **离线标定精度体检工具** (`tag_offline_verifier.py`)，对已完成 BA 求解的 `tags_map.yaml` 进行逐帧 Leave-One-Out 盲测批量评估，输出 Markdown 综合评审报告与可视化分析图。

**纯离线批处理**：不依赖相机硬件，仅需 `config/tags_map.yaml` + `data/tag_calibration_images/` 中的采集样本。

---

## 核心能力矩阵

| 能力 | BA 内置报告 (已有) | 本工具 (新建) |
|:---|:---:|:---:|
| 全局 RMSE 统计 | ✅ (求解副产物) | ✅ (独立验证) |
| **Leave-One-Out 盲测** | ❌ | ✅ **核心价值** |
| **Per-Tag 偏差热力图统计** | ❌ | ✅ |
| **自动标记需回审坏帧** | ❌ | ✅ |
| 逐帧 AR 棱柱 + 残差矢量渲染 | Quiver 箭头 | ✅ 叠加盲测棱柱 |
| Markdown 综合评审报告 | 简版 | ✅ 完整版含 Pass/Fail |

---

## 算法设计

### 1. Leave-One-Out 盲测流程

对每张采集图像执行以下循环：

```
for each image:
    detect all tags → detected_tags = {tid: corners_2d}
    for each tag_i in detected_tags:
        solver_tags = detected_tags - {tag_i}     # 排除当前盲测目标
        if len(solver_tags) >= 2:                  # 至少需要 2 个 tag 约束 PnP
            camera_pose = solvePnP(solver_tags, tags_map)
            projected_corners = projectPoints(tags_map[tag_i], camera_pose)
            loo_error_px = mean(||projected - observed||)
            loo_error_mm = loo_error_px * depth / fx
            record(image, tag_i, loo_error_px, loo_error_mm)
```

### 2. Per-Tag 系统性偏差分析

对每个 Tag 汇总其在所有图像中的盲测误差：
- 中位数、均值、最大值、标准差
- 若某 Tag 的中位误差显著高于全局中位数的 2 倍 → 标记为 **系统性偏差嫌疑**（可能 BA 地图坐标不准）

### 3. 坏帧自动标记

若某帧的平均盲测误差 > 全局中位数 + 2.5σ → 标记为建议回审帧，输出到报告中。

### 4. 综合 Pass/Fail 判定标准

| 指标 | PASS | ACCEPTABLE | FAIL |
|:---|:---|:---|:---|
| 全局盲测中位误差 | ≤ 1.5 px | ≤ 3.0 px | > 3.0 px |
| 盲测空间中位偏差 | ≤ 1.0 mm | ≤ 2.5 mm | > 2.5 mm |
| 系统性偏差 Tag 数 | 0 个 | ≤ 2 个 | > 2 个 |

---

## Proposed Changes

### Calibration 工具套件

#### [NEW] [`tag_offline_verifier.py`](file:///d:/Software/antigravity/flux_vision_3d/tools/calibration/tag_offline_verifier.py)

核心离线体检引擎类 `TagOfflineVerifier`，主要方法：

| 方法 | 职责 |
|:---|:---|
| `__init__` | 加载 `tags_map.yaml`、相机内参、检测器初始化 |
| `run_full_verification` | 主入口：扫描全部采集图 → 逐帧检测 → LOO 盲测 → 汇总统计 |
| `_verify_single_frame` | 对单帧执行全标靶 LOO 盲测循环 |
| `_solve_camera_pose` | 给定一组 tag-corners 对，执行超定 PnP 求解相机位姿 |
| `_compute_loo_error` | 计算单个盲测目标的重投影像元误差与空间毫米误差 |
| `_aggregate_statistics` | Per-Tag / Per-Frame 汇总统计、系统性偏差检测、坏帧标记 |
| `_render_verification_frame` | 渲染单帧体检可视化图 (AR 棱柱 + 盲测残差矢量 + 误差标牌) |
| `_export_markdown_report` | 输出完整 Markdown 精度体检报告 |

复用策略：
- 相机内参加载 → 复用 `resolve_camera_intrinsics`
- 标靶地图加载/世界坐标变换 → 复用 `TagCalibrationVerifier` 中的模式 (直接 YAML 读取)
- 标靶检测 → 复用 `TagMapBuilder.detect_tags` (双路融合检测)
- 3D 棱柱渲染 → 复用 `TagMapBuilder.render_tag_3d_axes`

文件输出：
- `data/tag_calibration_verification/offline_verification_report.md` — 综合评审报告
- `data/tag_calibration_images/visualized/*_loo_verify.png` — 逐帧盲测可视化图

---

### CLI 菜单集成

#### [MODIFY] [`cli_menu.py`](file:///d:/Software/antigravity/flux_vision_3d/tools/cli_menu.py)

- 在标定子菜单中新增选项 `[P]` — "离线精度体检 (Leave-One-Out 盲测批量验证)"
- 调用 `tag_offline_verifier.py` 的 `main()`

---

## 验证计划

### 自动验证
- 工具运行后检查 `offline_verification_report.md` 是否成功生成
- 检查可视化图像目录 `*_loo_verify.png` 输出

### 手动验证
- 在现有 22 帧采集样本上运行，确认 LOO 盲测结果与在线验证器的单帧盲测结果一致性

---

## Open Questions

> [!IMPORTANT]
> **检测器选择**：离线体检应使用与 BA 求解时相同的检测器（`TagMapBuilder` 的双路融合），还是使用超精提取器（`TagSuperExtractor` 的 5 路 + CONTOUR）？建议使用 `TagMapBuilder` 保持一致，避免因检测器差异引入额外变量。

> [!NOTE]
> **观测清单 vs 原图重检测**：是直接使用 `tag_observations.yaml` 清单中已存储的角点坐标（已审核过滤），还是对原图重新检测？建议使用清单数据——它经过人工审核过滤和亚像素精修，与 BA 求解的输入完全一致，验证结论更有参考价值。
