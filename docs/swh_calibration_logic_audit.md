# 标定体系深度审查报告 (Calibration Logic Audit)

> **文档定位**：对 `flux_vision_3d` 标定全链路（多视角采图 → 双路检测 → 亚像素精修 → 共视守门员 → 两阶段 BA → 基线尺度 → 世界系对齐 → 在线 PnP）的方法学与逻辑缺陷审查结论，含证据、影响与修复建议。
> **审查日期**：2026-09-14
> **审查方式**：源码通读 + 落盘产物（`config/tags_map.yaml`）交叉验证
> **适用读者**：算法开发、标定实施、架构评审、现场部署

### 审查覆盖范围

| 层 | 文件 |
| :--- | :--- |
| 领域模型 | `src/calibration/ba_optimizer.py`、`covisibility_graph.py`、`manifest_repository.py`、`offline_engine.py` |
| 运行时 | `src/vision/tag_localizer.py`、`src/utils/config_guard.py` |
| 工具链 | `tools/calibration/tag_map_builder.py`、`tag_super_extractor.py`、`tag_calibration_verifier.py`、`tag_offline_verifier.py`、`tag_manifest_reviewer.py`、`tag_capture_wizard.py` |
| 配置与产物 | `config.yaml`、`config/tags_map.yaml` |

---

## 一、总体结论

标定体系（多视角 → 双路检测 → 亚像素精修 → 共视守门员 → 两阶段 BA → 基线尺度 → 世界系对齐 → 在线 PnP）**框架是正确且工程化程度很高的**，尤其共视连通性守门员、IPPE 翻转消歧、MAD 粗差清洗这几处设计确实解决了实打实的工业问题。

但存在 **若干会导致"名义精度"与实际输出不一致的逻辑缺陷**，其中 2 项已达"会输出错误数据却无任何报错"的级别。

---

## 二、P0 —— 会让结果错误且静默通过的问题

### 1. 世界 X 轴对齐实际从未生效（设计与实物/配置自相矛盾）

`align_to_scara_world` 用 `Tag0 → Tag1` 的连线定义世界 +X：

```python
# src/calibration/ba_optimizer.py:717-720
if origin_tag_id in tag_poses and x_align_tag_id in tag_poses:
    vec_x = tag_poses[x_align_tag_id][:3, 3] - p_origin
    yaw_rad = math.atan2(vec_x[1], vec_x[0])
    print(f"[+] 坐标系 X 轴对齐旋转角: {-math.degrees(yaw_rad):.2f}°")
```

而 `x_align_tag_id` 默认是 **1**（`tools/calibration/tag_map_builder.py:869`，CLI 调用未传参，取默认）。

**但 Tag 1 根本不存在**：

- 白名单 `valid_tag_ids: [0, 18..29]`（`config.yaml:62`），检测入口会硬过滤掉一切非白名单 ID（`tag_map_builder.py:170`）；
- 实际落盘地图 `config/tags_map.yaml` 的 keys 实测为 `[0, 18, 19, …, 29]`，无 Tag 1。

于是逻辑走进 `else` 分支：`yaw_rad` 保持 0、**不打印任何警告**，X 轴对齐被静默跳过。同时 `base_static_id` 退化为 `min(all_detected_tags) = 0`：

```python
# src/calibration/ba_optimizer.py:120-121
base_static_id = x_align_tag_id if x_align_tag_id in all_detected_tags else min(all_detected_tags)
tag_poses_init = {base_static_id: np.eye(4, dtype=np.float64)}
```

**实证**：落盘地图中 `tags[0].transform_matrix` 正是单位阵——确凿证明基准被固定在 Tag 0、且对齐是空操作。

**后果**：世界系 X 方向完全等于 Tag 0 标签自身的印刷朝向。若现场张贴 Tag 0 时有几度转角，所有 `robot_x` / `robot_y` 会整体旋转，**下游机械臂每次抓取都偏**，而地图本身"看起来完全正常"。这属于把标定精度押在一个未受控的物理安装角上。

**修复方向**：要么把 `x_axis_id` 改为实际存在的标靶（如 18），要么改用"全部静止标靶的最小二乘拟合主方向"定义 X，并在基准标靶缺失时**硬失败**而非静默降级。

### 2. 地图 schema 与消费方不一致（导致 Tag 0 被误当静态标靶）

实测当前 `config/tags_map.yaml` 顶层只有 `['marker_size_mm', 'tags', 'rmse_px']`，每个标靶只有 `transform_matrix` / `position_mm`。而 `align_to_scara_world` 本应写入 `rpy_deg` / `is_origin` / `is_dynamic_yaw`，`optimize()` 本应写入 `rmse_reprojection_px` / `baseline_gauge` / `origin_tag_id` / `x_axis_align_tag_id`（`ba_optimizer.py:466-479, 746-752`）。

**即：当前地图不是当前优化器产出的**（连 RMSE 字段名都是 `rmse_px` 而非代码读取的 `rmse_reprojection_px`）。

**直接危害**：在线定位器靠该字段排除会随 J1 旋转的 Tag 0：

```python
# src/vision/tag_localizer.py:110
if tag_id in static_tags_map and not static_tags_map[tag_id].get("is_dynamic_yaw", False):
```

字段缺失 → `.get(..., False)` → **Tag 0 被当作静止标靶参与 PnP 解相机位姿**。而 Tag 0 按设计是动态的（`align_to_scara_world` 里 `is_dynamic_yaw = (t_id == origin_tag_id)`）。一旦 J1 转动且 Tag 0 入镜，外参会突变。

**修复方向**：统一 schema（生产者/消费者共用一份定义 + 加载时字段校验），并在加载地图时若缺 `is_dynamic_yaw` 直接告警而非默认放行。

---

## 三、P1 —— 方法学层面的实质缺陷

### 3. 世界系 Z 轴（竖直方向）没有任何物理约束

对齐只做了绕 Z 的 yaw：

```python
# src/calibration/ba_optimizer.py:722-728
R_align = np.array([
    [cos_y, -sin_y, 0.0],
    [sin_y,  cos_y, 0.0],
    [0.0,    0.0,   1.0]
], dtype=np.float64)
```

由于基准标靶被固定为 `np.eye(4)`，**世界 Z 轴 = 基准标签法向**。系统从未用重力/多点平面去校验或校正它。若张贴面有 pitch/roll 倾斜，全局 Z 误差会随平面坐标**线性增长**（远端漂移大），且下游 `Z_robot` 直接继承该误差。

**修复方向**：用 ≥3 个静止标靶中心做最小二乘平面拟合，将世界 XY 平面调到该平面，或引入倾角传感器/已知水平面先验。

### 4. 两阶段 BA 全程使用 Cauchy 鲁棒核 → 最终解不是最小二乘解

```python
# src/calibration/ba_optimizer.py:315-326
res_stage1 = least_squares(
    monitor1.wrap_residuals(residuals_func, obs_weights, None), x0,
    method='trf',
    loss='cauchy',
    f_scale=1.5,
```

阶段二（号称"微容差精平差"）**仍然 `loss='cauchy', f_scale=1.0`**（`ba_optimizer.py:383-394`）。鲁棒核的标准用法是"用它识别/剔除离群 → 最后用线性损失收敛到真最小二乘解"。全程 Cauchy 意味着最终位姿是有偏估计：残差越大被压得越狠，**系统性偏差被"吃掉"而不是被修正**。

另外文档与代码不符：`docs/algorithm_pipeline.md:289` 宣称 `ftol=1e-9, gtol=1e-9`、200+ 步，实际是 `ftol=xtol=gtol=1e-5, max_nfev=200`。所谓"极限深层收敛"被夸大。

**修复方向**：阶段二改用 `loss='linear'`（或 `soft_l1` 收尾），并把文档改为实际值。

### 5. 绝对尺度：双重机制 + 默认关闭 + 依赖未校验假设

尺度来源有三个，互相重叠：

1. `obj_points` 用名义 `marker_size_mm`（默认硬编码 50.0）；
2. BA 内把基线做成软约束，**权重硬编码 5.0**：

```python
# src/calibration/ba_optimizer.py:243-249
if baseline_pair is not None:
    id_a, id_b, real_dist_mm = baseline_pair
    ...
        residuals.append((dist_est - real_dist_mm) * 5.0)
```

3. 平差后**又**做一次刚性整体缩放 `apply_baseline_scale`（`ba_optimizer.py:651-693`），把距离强行拉成精确值。

第 2 步与第 3 步功能重复且会互相拉扯：既然第 3 步会精确重标定尺度，第 2 步那个 5.0 权重（相对像素残差约 1.0 的量级，实际权重 25 倍）只会**扭曲几何形状**去迁就基线。

更关键的是 **`--baseline_pair` 默认 None**（`tag_map_builder.py:870`），CLI 调用也不传 → 实际投产路径下该机制完全关闭，**绝对尺度 100% 依赖"打印出来正好 50.0mm"这一未验证假设**。而标靶是 A4 纸打印 + 粘贴，热胀/打印机缩放 1~3% 完全正常，直接换算成几百毫米处的数毫米到十几毫米系统误差。

**修复方向**：固定使用一次现场实测基线作为唯一定标手段；把 `marker_size` 提入 `config.yaml` 并强制与实测一致；BA 内的基线软约束与后处理缩放二者只保留其一。

### 6. 角点提取方法全链路不统一（文档宣称已根治的缺陷仍在运行链路）

`tag_super_extractor` 与 `tag_manifest_reviewer` 已改用 CONTOUR：

```python
# tools/calibration/tag_super_extractor.py:107
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
```

但以下位置**仍是 cornerSubPix / SUBPIX**：

- 建图器自身检测 `tag_map_builder.py:93,216`（`build_map_from_images` 备用通道 + `export_manifest` 也走它）
- 离线体检引擎 `src/calibration/offline_engine.py:70,124`
- 在线 AR 验证器 `tag_calibration_verifier.py:120`
- **运行时定位器 `src/vision/tag_localizer.py:52`**
- 采图向导、诊断工具

而 `docs/CHANGELOG.md` 明确指出 cornerSubPix 会让角点被"单向梯度吸附滑移 8~10 像素"。于是：**离线建图用高质量角点，在线运行却用被污染的角点**——标定再准，运行时外参仍带该漂移。更重要的是，`offline_engine` 在 `source=images` 复检时也用 SUBPIX，导致**"体检结论"与"建图输入"用了两套不同几何**，体检结果不可信。

**修复方向**：把角点提取抽成 `src/` 下唯一实现，全链路复用。

### 7. 检测参数不一致（体检引擎硬编码，不读配置）

`offline_engine._build_detectors` 把阈值写死为 `5.5 / 2.5 / 0.55 / 0.45`（`offline_engine.py:60-78`），而 `tag_map_builder` 是从 `config.yaml:calibration.tag_detection` 读的。用户在白名单管理里调过的反差参数，**在体检/Studio 路径下不生效**，出现"建图能检测、体检检测不到"的假报错。

### 8. 3σ 不确定度的量纲可疑

```python
# src/calibration/ba_optimizer.py:510-511
JTJ_reg = JTJ + np.eye(JTJ.shape[0]) * 1e-6
cov = np.linalg.pinv(JTJ_reg) * (sigma_res_px ** 2)
```

三个问题：

- 优化时用了 `x_scale='jac'`（`ba_optimizer.py:320`），此时 `res_stage2.jac` 是否仍在原始参数空间**需要核实**；若为缩放空间，则 `cov` 的物理单位不是 mm²，报告的 `±x.x mm` 无意义；
- 残差在优化中被 `sqrt(w)` 加权（`ba_optimizer.py:240`），而 `sigma_res_px` 用的是**未加权** RMSE，量纲不匹配；
- 被固定的基准标靶被赋予 `0.0` 不确定度（`ba_optimizer.py:502`），而它恰恰是把误差传播给全网的那个，属于**系统性低估**。

报告里给出的"3σ 空间不确定度"因此只能当作相对指标，不能当计量结论。

### 9. LOO 盲测允许单标靶解位姿

```python
# src/calibration/offline_engine.py:390-401
if len(known_tags) < 2:
    return []
results = []
for blind_tid in sorted(known_tags.keys()):
    solver_pairs = [(tid, c) for tid, c in known_tags.items() if tid != blind_tid]
```

一帧只检出 2 个标靶时，求解相机位姿只用 **1 个平面标靶（4 点）**——平面 PnP 在此是**欠约束/双解**问题（正是 problem #1 里被消歧的那个二义性）。这类 LOO 结果会随机跳变，却被计入"中位误差/通过率"统计，污染体检结论。应要求 `solver_count >= 2`。

### 10. 守门员只拦"断网"，不拦"被降权到断网"

粗差在阶段二被降权而非剔除：

```python
# src/calibration/ba_optimizer.py:229-230
if active_outliers and (f, t_id) in active_outliers:
    w *= 0.01  # 离群项强力压制
```

若被判离群的恰好是某条 `critical_bridges`（仅单图支撑）观测，则该桥梁**事实上失效**，但连通性守门员是在优化**之前**跑的（`ba_optimizer.py:111`），不会发现。`critical_bridges` 本身也只是打印提示（`ba_optimizer.py:116-117`），从不拦截。建议：若某桥梁的唯一观测被标为离群，应硬失败要求人工恢复。

---

## 四、P2 —— 一致性与可维护性

| # | 问题 | 证据 |
| :-: | :--- | :--- |
| 11 | 标靶名义尺寸 50mm 硬编码在 8 处，采图向导却用 40.0，且 `config.yaml` 无此字段 | `tag_map_builder.py:41`、`tag_capture_wizard.py:85` |
| 12 | 当前落盘地图缺少 `origin_tag_id` / `x_axis_align_tag_id` / `baseline_gauge` / `calibrated_images_count`，且 RMSE 键名与代码读取名不一致 | 实测 YAML keys vs `ba_optimizer.py:466-477` |
| 13 | `dist_coeffs` 恒为全零，但用的是 1080P 广角彩色流（未做去畸变），边缘径向畸变直接成为重投残差下限——这正是实测 4.14px 与标准图 0.032px 差距的合理主因 | `config_guard.py:53`、`ba_optimizer.py:46` |
| 14 | 诊断报告固定文件名 `ba_precision_diagnostic_report.md`，每次覆盖，无历史可追溯 | `ba_optimizer.py:599` |
| 15 | 基准标靶固定为 `np.eye(4)` 且不参与优化，其自身观测噪声直接注入全网，无软先验约束 | `ba_optimizer.py:121` |
| 16 | `origin_tag_id` 缺失时仅 `WARN` 后继续（`p_origin=zeros`），会把错误原点当成有效地图导出 | `ba_optimizer.py:713-714` |

---

## 五、修复优先级建议

1. **立即（正确性）**：修 #1（X 轴对齐）与 #2（地图 schema / `is_dynamic_yaw`）——这两条直接决定机械臂坐标是否可信。
2. **短期（精度）**：#5 尺度定标（强制基线 + 实测边长）、#4 阶段二改线性损失、#6 角点提取统一。
3. **中期**：#3 世界 Z 轴平面校正、#8 不确定度量纲、#9 LOO 约束、#10 桥梁保护。
4. **长期**：把 `marker_size_mm`、检测阈值、角点方法、地图 schema 全部收敛为单一来源（`config.yaml` + 一个 `src/calibration/schema.py`），消除 8 处硬编码漂移。

---

## 六、后续可执行的验证动作

建议补两个回归测试，把 P0 缺陷固化为可复现用例：

1. **Tag 1 缺失时对齐静默跳过**：构造仅含 Tag 0 与 18~29 的观测集，断言 `align_to_scara_world` 应在 `x_align_tag_id` 缺失时抛出/至少显式告警，而非静默 `yaw_rad = 0`。
2. **地图缺 `is_dynamic_yaw` 导致 Tag 0 入 PnP**：加载当前 `config/tags_map.yaml`，断言 `TagLocalizer` 不应把 Tag 0 纳入静止标靶集合（或在加载期即校验字段完整性）。
