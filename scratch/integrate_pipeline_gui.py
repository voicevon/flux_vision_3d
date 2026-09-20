import os

file_path = r"d:\Software\antigravity\flux_vision_3d\tools\asparagus_offline.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 增加 pipeline 导入
import_marker = "from src.vision.asparagus_analyzer import AsparagusAnalyzer"
new_import = """from src.vision.asparagus_analyzer import AsparagusAnalyzer, AsparagusTarget
from src.vision.pipelines import PipelineRegistry, BaseAsparagusPipeline, PipelineResult, PipelineStep"""

assert import_marker in code, "import_marker not found"
code = code.replace(import_marker, new_import)

# 2. 在 __init__ 中集成 Pipeline 状态
old_init_pipeline = """        # 视口显示图层切换 (0: 1.前景, 1: 距离场, 2: 峰脊线, 3: 2.骨架, 4: 3.位姿) - 纯显示切换，一次性全量解算
        self.active_view_mode = 4
        self.stage_vis = [None, None, None, None, None]"""

new_init_pipeline = """        # 多技术路线感知架构 (Pipeline Architecture)
        self.pipeline_key = "ridge_tracing"
        self.pipeline: Optional[BaseAsparagusPipeline] = None
        self.pipeline_result: Optional[PipelineResult] = None
        self.active_step_key = "stage3_poses"
        self._init_pipeline()"""

assert old_init_pipeline in code, "old_init_pipeline not found"
code = code.replace(old_init_pipeline, new_init_pipeline)

# 3. 增加 _init_pipeline 与 switch_pipeline 方法，并更新 run_analyze
old_analyzer_builder = """    def _build_analyzer(self, img_w: int, img_h: int) -> AsparagusAnalyzer:
        \"\"\"构建分析器: 内参取 config.yaml 并按快照实际分辨率等比缩放\"\"\"
        intr = self.sys_cfg["intrinsics"]
        if intr:
            fx, fy, cx, cy, cfg_w, cfg_h = intr
            if cfg_w > 0 and cfg_h > 0 and (cfg_w != img_w or cfg_h != img_h):
                fx, cx = fx * img_w / cfg_w, cx * img_w / cfg_w
                fy, cy = fy * img_h / cfg_h, cy * img_h / cfg_h
        else:
            fx, fy, cx, cy = 909.12, 907.46, 647.46, 377.51   # 640x480 缺省内参
        analyzer = AsparagusAnalyzer(fx=fx, fy=fy, cx=cx, cy=cy)
        analyzer.set_tag_localizer(self.tag_localizer)
        analyzer.set_hand_eye_matrix(self.sys_cfg["t_cam_to_scara"])
        return analyzer"""

new_analyzer_builder = """    def _get_scaled_intrinsics(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        \"\"\"内参取 config.yaml 并按快照实际分辨率等比缩放\"\"\"
        intr = self.sys_cfg["intrinsics"]
        if intr:
            fx, fy, cx, cy, cfg_w, cfg_h = intr
            if cfg_w > 0 and cfg_h > 0 and (cfg_w != img_w or cfg_h != img_h):
                fx, cx = fx * img_w / cfg_w, cx * img_w / cfg_w
                fy, cy = fy * img_h / cfg_h, cy * img_h / cfg_h
        else:
            fx, fy, cx, cy = 909.12, 907.46, 647.46, 377.51
        return fx, fy, cx, cy

    def _init_pipeline(self):
        \"\"\"初始化当前选中的算法流水线\"\"\"
        fx, fy, cx, cy = self._get_scaled_intrinsics(1920, 1080)
        self.pipeline = PipelineRegistry.create(self.pipeline_key, fx=fx, fy=fy, cx=cx, cy=cy)
        if self.pipeline:
            steps = self.pipeline.get_steps()
            if steps:
                self.active_step_key = steps[-1].key  # 默认聚焦最终结果步

    def switch_pipeline(self, pipeline_key: str):
        \"\"\"切换感知算法技术路线 (秒级插拔切换，自动重算与自适应刷新)\"\"\"
        if pipeline_key == self.pipeline_key and self.pipeline is not None:
            return
        self.pipeline_key = pipeline_key
        self._init_pipeline()
        pipe_name = dict(PipelineRegistry.list_options()).get(pipeline_key, pipeline_key)
        self.set_toast(f"已切换算法路线: 【{pipe_name}】")
        if 0 <= self.sel_idx < len(self.samples):
            self.run_analyze()

    def _build_analyzer(self, img_w: int, img_h: int) -> AsparagusAnalyzer:
        \"\"\"向后兼容的分析器构造\"\"\"
        fx, fy, cx, cy = self._get_scaled_intrinsics(img_w, img_h)
        analyzer = AsparagusAnalyzer(fx=fx, fy=fy, cx=cx, cy=cy)
        analyzer.set_tag_localizer(self.tag_localizer)
        analyzer.set_hand_eye_matrix(self.sys_cfg["t_cam_to_scara"])
        return analyzer"""

assert old_analyzer_builder in code, "old_analyzer_builder not found"
code = code.replace(old_analyzer_builder, new_analyzer_builder)

# 4. 替换 run_analyze 和步骤切换方法
old_run_methods = """        try:
            analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            # 无论当前在看哪个视图，后台一次性全量解算完所有算法步骤！
            self.targets = analyzer.analyze(color, depth, stages=(True, True, True))
            # 缓存 5 步视觉过程：[1.前景, 距离场, 峰脊线, 2.骨架, 3.位姿]
            self.stage_vis = [
                analyzer.vis_stage1,
                getattr(analyzer, "vis_dist", None),
                getattr(analyzer, "vis_peaks", None),
                analyzer.vis_stage2,
                analyzer.vis_stage3
            ]
            self.sel_target = 0
            self._apply_view_mode()
        except Exception as exc:
            log.exception("样本解算异常")
            self.error = f"解算异常: {exc}"
            self.set_toast(self.error, True)
            return

        if self.targets:
            top = self.targets[self.sel_target]
            if self.mode == "3d":
                self.gcode_text = top.generate_gcode(safe_z=self.sys_cfg["safe_z"],
                                                     drop_x=self.sys_cfg["drop_x"],
                                                     drop_y=self.sys_cfg["drop_y"])
            self.set_toast(f"识别定位完成: 提取前 {len(self.targets)} 位优选目标", duration=2.2)
        else:
            self.set_toast("未检出符合规格的芦笋目标", duration=2.2)

    def _select_view_mode(self, mode_idx: int):
        \"\"\"
        独立切换视口显示的算法图层 (0: 1.前景, 1: 距离场, 2: 峰脊线, 3: 2.骨架, 4: 3.位姿)
        纯粹切换视口显示内容，完全独立 (individual)；右侧所有识别结果和文本数据始终保持展示！
        \"\"\"
        self.active_view_mode = mode_idx
        if self.stage_vis[mode_idx] is None and (0 <= self.sel_idx < len(self.samples)):
            self.run_analyze()
            return
        self._apply_view_mode()

    def _apply_view_mode(self):
        \"\"\"刷新视口显示的图像内容\"\"\"
        labels = [
            "1.前景物料 (ExG + 传送带 ROI)",
            "CV算法: 欧氏距离变换场 (Distance Transform 半径能量)",
            "CV算法: 垂向极大值峰脊线 (Transverse NMS Ridge Peaks)",
            "2.中轴骨架与单体验证 (Spine & Ridge Tracing)",
            "3.位姿定位与顶层锁定 (Top 3 抓取目标)"
        ]
        idx = max(0, min(self.active_view_mode, len(labels) - 1))
        if self.stage_vis[idx] is not None:
            self.vis_img = self.stage_vis[idx]
            self.set_toast(f"视口显示: {labels[idx]}", duration=1.5)
        elif 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                self.vis_img = color.copy()"""

new_run_methods = """        try:
            # 更新内参
            fx, fy, cx, cy = self._get_scaled_intrinsics(color.shape[1], color.shape[0])
            if self.pipeline:
                self.pipeline.update_intrinsics(fx, fy, cx, cy)
            else:
                self._init_pipeline()

            # 三级标定降级链解算外参
            dummy_analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            frame_transform, frame_calib_source = dummy_analyzer._resolve_calibration(color)
            plane_coeff = dummy_analyzer.fit_table_plane(depth) if depth is not None else None

            # 执行当前技术路线 Pipeline
            self.pipeline_result = self.pipeline.run(
                color_bgr=color,
                depth_mm=depth,
                plane_coeff=plane_coeff,
                frame_transform=frame_transform,
                frame_calib_source=frame_calib_source,
                nominal_z_mm=float(plane_coeff[2]) if (plane_coeff is not None and abs(plane_coeff[2]) > 300) else 640.0
            )
            self.targets = self.pipeline_result.targets
            self.sel_target = 0

            # 确保当前步骤 key 有效
            steps = self.pipeline.get_steps()
            step_keys = [s.key for s in steps]
            if self.active_step_key not in step_keys and step_keys:
                self.active_step_key = step_keys[-1]

            self._apply_active_step()
        except Exception as exc:
            log.exception("流水线执行异常")
            self.error = f"执行异常: {exc}"
            self.set_toast(self.error, True)
            return

        if self.targets:
            top = self.targets[self.sel_target]
            if self.mode == "3d":
                self.gcode_text = top.generate_gcode(safe_z=self.sys_cfg["safe_z"],
                                                     drop_x=self.sys_cfg["drop_x"],
                                                     drop_y=self.sys_cfg["drop_y"])
            self.set_toast(f"解算完成: 检出 {len(self.targets)} 个目标 (耗时: {self.pipeline_result.elapsed_ms}ms)", duration=2.2)
        else:
            self.set_toast(f"未检出符合规格目标 (耗时: {self.pipeline_result.elapsed_ms}ms)", duration=2.2)

    def _select_step(self, step_key: str):
        \"\"\"切换当前算法的中间步骤视图 (完全独立，右侧数据常驻)\"\"\"
        self.active_step_key = step_key
        if self.pipeline_result is None and (0 <= self.sel_idx < len(self.samples)):
            self.run_analyze()
            return
        self._apply_active_step()

    def _apply_active_step(self):
        \"\"\"刷新视口显示的图像内容\"\"\"
        if self.pipeline_result and self.active_step_key in self.pipeline_result.step_snapshots:
            img = self.pipeline_result.step_snapshots[self.active_step_key]
            if img is not None:
                self.vis_img = img
                step_obj = next((s for s in self.pipeline.get_steps() if s.key == self.active_step_key), None)
                if step_obj:
                    self.set_toast(f"视口显示: {step_obj.name} ({step_obj.description})", duration=1.8)
                return
        if 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                self.vis_img = color.copy()"""

assert old_run_methods in code, "old_run_methods not found"
code = code.replace(old_run_methods, new_run_methods)

# 5. 更新 _select_target
old_sel_t = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                self.stage_vis[4] = self.vis_img
                self.active_view_mode = 4"""

new_sel_t = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                if self.pipeline_result:
                    self.pipeline_result.step_snapshots[self.active_step_key] = self.vis_img"""

assert old_sel_t in code, "old_sel_t not found"
code = code.replace(old_sel_t, new_sel_t)

# 6. 更新 render() 中的顶部选择器布局与自适应药丸
old_toolbar_layout = """        # 视口算法流程全透明分段选择器：[ 1.前景 | 距离场 | 峰脊线 | 2.骨架 | 3.位姿 ]
        view_names = ["1.前景", "距离场", "峰脊线", "2.骨架", "3.位姿"]
        pill_w = int(60 * m["s"])
        for v_i in (4, 3, 2, 1, 0):
            bx -= pill_w + int(4 * m["s"])
            pill_rect = (bx, int(14 * m["s"]), bx + pill_w, int(14 * m["s"]) + m["btn_h"])
            self._draw_view_pill(canvas, pill_rect, view_names[v_i], is_active=(self.active_view_mode == v_i))
            self._buttons.append((pill_rect, ("set_view_mode", v_i)))

        # 分段选择器左侧提示标签：“显示:”
        bx -= int(38 * m["s"])
        draw_text(canvas, "显示:", (bx, int(22 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)

        # 最左侧：工位地图选择下拉按钮
        sc_w = int(175 * m["s"])
        bx -= sc_w + int(10 * m["s"])
        self._workspace_rect = (bx, int(14 * m["s"]), bx + sc_w, int(14 * m["s"]) + m["btn_h"])
        is_sc_open = (self.active_dropdown == "WORKSPACE_DROPDOWN")
        self._draw_dropdown_button(canvas, self._workspace_rect, f"地图: {self.current_workspace_name}", is_open=is_sc_open)
        self._buttons.append((self._workspace_rect, ("toggle_dd", "WORKSPACE_DROPDOWN")))"""

new_toolbar_layout = """        # 动态自适应步骤药丸：由当前算法 pipeline.get_steps() 动态决定！
        steps = self.pipeline.get_steps() if self.pipeline else []
        pill_w = int(58 * m["s"])
        for s_obj in reversed(steps):
            bx -= pill_w + int(4 * m["s"])
            pill_rect = (bx, int(14 * m["s"]), bx + pill_w, int(14 * m["s"]) + m["btn_h"])
            self._draw_view_pill(canvas, pill_rect, s_obj.name, is_active=(self.active_step_key == s_obj.key))
            self._buttons.append((pill_rect, ("set_step", s_obj.key)))

        # 步骤药丸左侧小标签：“显示:”
        if steps:
            bx -= int(38 * m["s"])
            draw_text(canvas, "显示:", (bx, int(22 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)

        # 算法路线下拉选择按钮
        pipe_w = int(185 * m["s"])
        bx -= pipe_w + int(8 * m["s"])
        self._pipeline_rect = (bx, int(14 * m["s"]), bx + pipe_w, int(14 * m["s"]) + m["btn_h"])
        is_pipe_open = (self.active_dropdown == "PIPELINE_DROPDOWN")
        curr_pipe_name = dict(PipelineRegistry.list_options()).get(self.pipeline_key, self.pipeline_key)
        # 简化名称显示
        short_pipe_name = curr_pipe_name.split(":")[0] if ":" in curr_pipe_name else curr_pipe_name
        self._draw_dropdown_button(canvas, self._pipeline_rect, f"算法: {short_pipe_name}", is_open=is_pipe_open)
        self._buttons.append((self._pipeline_rect, ("toggle_dd", "PIPELINE_DROPDOWN")))

        # 工位地图选择下拉按钮
        sc_w = int(160 * m["s"])
        bx -= sc_w + int(8 * m["s"])
        self._workspace_rect = (bx, int(14 * m["s"]), bx + sc_w, int(14 * m["s"]) + m["btn_h"])
        is_sc_open = (self.active_dropdown == "WORKSPACE_DROPDOWN")
        self._draw_dropdown_button(canvas, self._workspace_rect, f"地图: {self.current_workspace_name}", is_open=is_sc_open)
        self._buttons.append((self._workspace_rect, ("toggle_dd", "WORKSPACE_DROPDOWN")))"""

assert old_toolbar_layout in code, "old_toolbar_layout not found"
code = code.replace(old_toolbar_layout, new_toolbar_layout)

# 7. 更新下拉浮层弹出与点击处理
old_popup_part = """        # 置顶渲染下拉弹出菜单 (覆盖在所有内容最上层)
        if self.active_dropdown == "WORKSPACE_DROPDOWN" and self._workspace_rect:
            self._render_dropdown_popup(canvas, self._workspace_rect, self.workspace_options, self.current_workspace_id)"""

new_popup_part = """        # 置顶渲染下拉弹出菜单 (覆盖在所有内容最上层)
        if self.active_dropdown == "WORKSPACE_DROPDOWN" and self._workspace_rect:
            self._render_dropdown_popup(canvas, self._workspace_rect, self.workspace_options, self.current_workspace_id)
        elif self.active_dropdown == "PIPELINE_DROPDOWN" and getattr(self, "_pipeline_rect", None):
            self._render_dropdown_popup(canvas, self._pipeline_rect, PipelineRegistry.list_options(), self.pipeline_key)"""

assert old_popup_part in code, "old_popup_part not found"
code = code.replace(old_popup_part, new_popup_part)

# 8. 更新点击命中处理
old_hit_dispatch = """            # 按钮点击
            hit = self.hit_test(x, y)
            if hit:
                act_type, act_val = hit
                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                elif act_type == "set_view_mode":
                    self._select_view_mode(act_val)
                return"""

new_hit_dispatch = """            # 下拉浮层项点击
            if self.active_dropdown and self._dd_items:
                for rect, key in self._dd_items:
                    if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                        dd_type = self.active_dropdown
                        self.active_dropdown = None
                        if dd_type == "WORKSPACE_DROPDOWN":
                            self.switch_workspace(key)
                        elif dd_type == "PIPELINE_DROPDOWN":
                            self.switch_pipeline(key)
                        return
                self.active_dropdown = None

            # 按钮点击
            hit = self.hit_test(x, y)
            if hit:
                act_type, act_val = hit
                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                elif act_type == "set_step":
                    self._select_step(act_val)
                return"""

# 在原代码中寻找点击处理
click_idx = code.find("# 4. 常规左键点击事件")
assert click_idx != -1, "click_idx not found"
sub_code = code[click_idx:]
old_hit_search = """            # 优先判定置顶下拉浮层点击
            if self.active_dropdown and self._dd_items:
                for rect, key in self._dd_items:
                    if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                        self.active_dropdown = None
                        self.switch_workspace(key)
                        return
                self.active_dropdown = None

            # 按钮点击
            hit = self.hit_test(x, y)
            if hit:
                act_type, act_val = hit
                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                elif act_type == "set_view_mode":
                    self._select_view_mode(act_val)
                return"""

assert old_hit_search in sub_code, "old_hit_search not found"
code = code.replace(old_hit_search, new_hit_dispatch)

# 9. 更新右侧结果面板标题显示耗时 Benchmark
old_panel_title = """    def _draw_result_panel(self, canvas, m, rect):
        y = self._panel_bg(canvas, rect, f"识别定位结果 (前3位: {len(self.targets)})")"""

new_panel_title = """    def _draw_result_panel(self, canvas, m, rect):
        perf_tag = f" · {self.pipeline_result.elapsed_ms:.0f}ms" if self.pipeline_result else ""
        y = self._panel_bg(canvas, rect, f"识别结果 (前3位){perf_tag}")"""

assert old_panel_title in code, "old_panel_title not found"
code = code.replace(old_panel_title, new_panel_title)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Integrated pluggable pipeline architecture into tools/asparagus_offline.py successfully!")
