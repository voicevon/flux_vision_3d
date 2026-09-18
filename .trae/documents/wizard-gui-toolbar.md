# 采图向导 GUI 化改造：启动不开相机 + Tracker 同款工具栏

## Context (背景)

现状：`tools/calibration/tag_capture_wizard.py`（AprilTag 多视角采图向导）在 `__init__` 末尾**立即启动相机**（`_init_realsense()`，固定 1920x1080@8fps 回退链），主循环是纯键盘 cv2 循环，无鼠标、无工具栏、无相机类型/分辨率选择，退出靠 Q/ESC。

用户需求：
1. 启动向导时**不打开相机**，只加载 UI；
2. 顶部一排工具栏与 Robot 在线跟踪（tools/tracker）**左半部分一致**：`[相机类型 ▼] [分辨率 ▼] [开启]`；
3. 工具栏**最右侧**放 `[退出 X]`；
4. 用户点 `[开启]` 才打开相机，进入预览（AprilTag 检测叠加逻辑照旧）。

方案：复用 tracker 的成熟模式——GUI 先行 + `GuiWindowManager` 矢量窗口 + 每帧重建 buttons 命中表 + `config/gui_settings.json` 持久化。参照文件：`tools/tracker/app.py`（L216-315 持久化与相机开关、L871-948 鼠标分发、L951-1067 主循环）、`tools/tracker/renderer.py`（L41-126 hover/下拉、L127-257 工具栏、L701-736 画布合成）。

## 改动文件

| 文件 | 动作 |
|---|---|
| `tools/calibration/tag_capture_wizard.py` | 改造（主控制器：延迟开相机 + 鼠标/工具栏交互 + 持久化） |
| `tools/calibration/wizard_renderer.py` | 新建（工具栏/画布/Toast 渲染，照抄 tracker renderer 模式） |
| `tests/test_wizard_renderer.py` | 新建（渲染冒烟 + 持久化 + 相机开关状态机测试） |

不改动启动点：`gui_launcher.py` 卡片、`cli_menu.py` 的 `run_tag_capture_wizard` 入口脚本与参数（`--dir` / `--mock`）不变。

## A. tag_capture_wizard.py 改造

1. **`__init__` 移除自动开相机**：
   - 删除末尾 `if not self.mock_mode: self._init_realsense() else enter_mock` 分支及 `_init_realsense()` 方法；
   - 新增薄相机状态（不导入 tracker 的 CameraController——其 fps 策略与向导现状不完全一致，且避免访问私有 `_srv`）：`camera_type="realsense"`、`resolution="1920x1080"`（延续现状 1080P 优先）、`camera_options` / `resolution_options`（与 tracker 同款选项）、`frame_w/frame_h`、`pipeline_running=False`；取流仍走已有 `self._cam_srv`；
   - `--mock` 仅作为「开启」时的取流后端（点开启 → `_cam_srv.enter_mock()`），启动时不再自动进入仿真；
   - `color_sensor` / `actual_stream_desc` 改为开启成功后从 `_cam_srv` 读取（曝光快捷键 [ ]/[E] 依赖 color_sensor，未开启时 Toast 提示「相机未开启」）。
2. **相机开关方法**（借鉴 `tracker/app.py` `_toggle_camera` L294-315）：
   - 关：`_cam_srv.stop()`，`pipeline_running=False`，Toast「相机已关闭」；
   - 开：realsense → `start_realsense(w, h, fps=8 if w>1280 else 15, fallbacks=((w,h,8),), mock_fallback=False)`；usb → `start_usb(w, h)`；`--mock` → `enter_mock()`；失败 Toast 报错且**不静默降级 Mock**（现场采图不能误采仿真帧，改掉现状 `mock_fallback=True`）；
   - `_select_camera_type` / `_change_resolution`（借鉴 app.py L268-292）：运行中先停再按新选择重启。
3. **持久化**（借鉴 app.py `_load_viewer_state`/`_save_viewer_state` L216-265）：`APP_ID="tag_capture_wizard"`，字段 `camera_type` / `resolution` 写入 `config/gui_settings.json`；启动时恢复（合法性校验同 tracker）。
4. **`get_frame` 修正**：仅 `pipeline_running` 时 `read_frame`，未开启返回 `None` —— 杜绝现状「未开流也回退 Mock 帧」的缺陷；未开启时主循环画占位画布。
5. **`run()` 主循环重构**（借鉴 app.py L951-1067）：
   - `GuiWindowManager(app_id="tag_capture_wizard")`；`setup_window("tag_capture_wizard", mouse_callback=self._on_mouse)`（key 纯 ASCII）；`set_unicode_title("AprilTag 采图向导 | flux_vision_3d")`；
   - 相机开启：`read_frame` → 现有检测叠加逻辑不动（画在帧上）→ `renderer.compose_canvas(frame)`；帧 `None` 时占位「取流中...」；
   - 相机关闭：`renderer.make_canvas()` 占位画布「相机未开启 | 请选择相机类型和分辨率，然后点击 [开启]」（与 tracker 占位文案同款）；
   - `draw_toolbar` + 下拉浮层 + Toast → `imshow` → `waitKeyEx` → `win_mgr.poll_events`（红叉/ESC 退出、窗口缩放自适应）；
   - 鼠标：`_on_mouse`（MOVE→`renderer.on_mouse_move`；LBUTTONDOWN→`hit_test`→`_handle_action`；点空白收起下拉）；
   - `_handle_action` 按钮分发（与 tracker 同名同义，app.py L884-948）：`TOGGLE_CAM_DD` / `TOGGLE_RES_DD` / `DD_CAM_i` / `DD_RES_i` / `TOGGLE_CAMERA` / `QUIT`；
   - 键盘快捷键**全部保留**（Space 拍摄 / Tab 预设 / [ ] 曝光 / E 自动曝光 / I K 对比度 / A 3D / W 白名单 / S 存参 / C 清空），新增 `X` 退出，`Q`/`ESC` 退出不变；
   - `finally`：相机 stop + `destroyAllWindows`；首帧 `force_window_focus` 保留（若 GuiWindowManager 已接管激活则去掉）；
   - 启动横幅 print 更新（GUI 先行说明 + 快捷键表，print 用法符合 P2 约定的菜单/横幅场景）。

## B. wizard_renderer.py（新文件，~180 行）

`WizardRenderer(wizard)`，模式照抄 `tools/tracker/renderer.py`，颜色直接取 `GuiTheme` 常量（与 common.py 的别名方式一致）：
- `make_canvas` / `compose_canvas`：窗口物理尺寸真矢量底板 + 帧等比 contain 缩放居中（1:1 像素对齐，鼠标零偏移，renderer.py L715-736 同款）；
- `_is_hover` / `_hover_text`（`GuiTheme.BTN_BEHAVIOR` 单源）；
- `_draw_dropdown_button` / `_render_dropdown_popup`（buttons 命中表每帧重建）；
- `draw_toolbar` **单排**：`TOOLBAR_H = 44`（tracker 双排为 78，向导只需一排）；布局居左：`[相机类型▼ 150px] [分辨率▼ 110px] [开启/关闭 70px]`（x=8 起，gap=6，与 tracker 第一排同款三态/悬停样式），最右 `[退出 X 90px]`（tw-90..tw-8）；
- `draw_toast`：沿用向导现有底部居中样式，改为绘制在最终 canvas 上（相机关闭时也能显示）；
- 文字一律走 `src/utils/text_rendering.py`（中文渲染约定）。

## C. tests/test_wizard_renderer.py（新文件）

- 工具栏命中表：6 类按钮 id 存在、`hit_test` 命中正确；
- 渲染冒烟：构造假 wizard（monkeypatch win_mgr canvas 尺寸），`make_canvas` / `compose_canvas` / `draw_toolbar` 不抛异常、输出尺寸正确；
- 持久化往返：`_save_viewer_state` / `_load_viewer_state`（临时 settings 文件）；
- 相机开关状态机：`--mock` 下 `_toggle_camera` 开/关翻转 `pipeline_running`（参考 test_mock_pipeline.py 既有模式）。

## 验证

1. `ruff check tools/calibration/tag_capture_wizard.py tools/calibration/wizard_renderer.py tests/test_wizard_renderer.py`
2. `python -m pytest tests/ -q`（112 现有 + 新增全绿）
3. 手工冒烟：`python tools/calibration/tag_capture_wizard.py --mock`
   - 启动即显示「相机未开启」占位 + 工具栏（**相机不启动**）；
   - 点 [开启] → Mock 流预览 + Tag 检测叠加，[关闭] 回占位；
   - 下拉切换相机类型/分辨率；点 [退出 X]、按 X、ESC 均干净退出；
   - 重启后下拉选择被记忆（gui_settings.json）。
