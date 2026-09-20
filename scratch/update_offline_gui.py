import os

file_path = r"d:\Software\antigravity\flux_vision_3d\tools\asparagus_offline.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 在 __init__ 增加 stage 相关状态
old_init_part = """        self.vis_img = None          # 标注可视化 (原始分辨率)
        self.mode = "3d"             # "3d" | "2d"
        self.error = ""              # 当前样本加载/解算错误
        self.gcode_text = ""

        # 视口交互控制器: 参照 Spatial Mapping Studio 实现滚轮放大缩小与平移"""

new_init_part = """        self.vis_img = None          # 标注可视化 (原始分辨率)
        self.mode = "3d"             # "3d" | "2d"
        self.error = ""              # 当前样本加载/解算错误
        self.gcode_text = ""

        # 三阶段透明流水线状态 (1.前景, 2.骨架, 3.位姿)
        self.stage_checked = [True, True, True]
        self.stage_vis = [None, None, None]
        self.active_stage_view = 2

        # 视口交互控制器: 参照 Spatial Mapping Studio 实现滚轮放大缩小与平移"""

assert old_init_part in code, "old_init_part not found"
code = code.replace(old_init_part, new_init_part)

# 2. 修改 _select_sample 清空 stage_vis
old_sel_part = """        self.sel_idx = idx
        self._keep_selection_visible(idx)
        self.targets, self.vis_img, self.gcode_text = [], None, ""
        self.sel_target, self.error = 0, ""
        self.viewport.reset()"""

new_sel_part = """        self.sel_idx = idx
        self._keep_selection_visible(idx)
        self.targets, self.vis_img, self.gcode_text = [], None, ""
        self.stage_vis = [None, None, None]
        self.sel_target, self.error = 0, ""
        self.viewport.reset()"""

assert old_sel_part in code, "old_sel_part not found"
code = code.replace(old_sel_part, new_sel_part)

# 3. 修改 run_analyze 与增加阶段控制方法
old_run_part = """        try:
            analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            self.targets = analyzer.analyze(color, depth)
            self.sel_target = 0
            self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=self.sel_target)
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
            self.set_toast("未检出符合规格的芦笋目标", duration=2.2)"""

new_run_part = """        try:
            analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            self.targets = analyzer.analyze(color, depth, stages=tuple(self.stage_checked))
            self.stage_vis = [analyzer.vis_stage1, analyzer.vis_stage2, analyzer.vis_stage3]
            self.sel_target = 0
            self._update_stage_display()
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
            if self.stage_checked[2]:
                self.set_toast("未检出符合规格的芦笋目标", duration=2.2)
            else:
                self.set_toast(f"流水线已执行至阶段 {self.active_stage_view + 1}", duration=2.0)

    def _toggle_stage(self, stage_idx: int):
        \"\"\"切换三阶段流水线 Checkbox\"\"\"
        new_val = not self.stage_checked[stage_idx]
        self.stage_checked[stage_idx] = new_val

        # 阶段依赖约束：后阶段依赖前阶段
        if stage_idx == 2 and new_val:
            self.stage_checked[0] = True
            self.stage_checked[1] = True
        elif stage_idx == 1 and new_val:
            self.stage_checked[0] = True
        elif stage_idx == 0 and not new_val:
            self.stage_checked[1] = False
            self.stage_checked[2] = False
        elif stage_idx == 1 and not new_val:
            self.stage_checked[2] = False

        # 确定激活视图
        if self.stage_checked[stage_idx]:
            self.active_stage_view = stage_idx
        else:
            if self.stage_checked[2]:
                self.active_stage_view = 2
            elif self.stage_checked[1]:
                self.active_stage_view = 1
            elif self.stage_checked[0]:
                self.active_stage_view = 0
            else:
                self.active_stage_view = -1

        # 若已有缓存直接显示，否则重新计算
        if self.active_stage_view >= 0 and self.stage_vis[self.active_stage_view] is not None:
            self._update_stage_display()
        elif 0 <= self.sel_idx < len(self.samples):
            self.run_analyze()

    def _update_stage_display(self):
        \"\"\"刷新当前选定阶段的视图\"\"\"
        stage_names = ["1.前景物料提取 (ExG+ROI)", "2.独立中轴脊线提取 (Ridge Tracing)", "3.位姿解算与顶层锁定 (Top 3)"]
        if self.active_stage_view >= 0 and self.stage_vis[self.active_stage_view] is not None:
            self.vis_img = self.stage_vis[self.active_stage_view]
            self.set_toast(f"视图切换: {stage_names[self.active_stage_view]}", duration=1.5)
        elif 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                self.vis_img = color.copy()"""

assert old_run_part in code, "old_run_part not found"
code = code.replace(old_run_part, new_run_part)

# 4. 在 _select_target 中支持阶段 3 更新
old_target_part = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)"""

new_target_part = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                self.stage_vis[2] = self.vis_img
                self.active_stage_view = 2"""

assert old_target_part in code, "old_target_part not found"
code = code.replace(old_target_part, new_target_part)

# 5. 增加 _draw_checkbox 并修改 render 布局
old_render_buttons = """        # 标题栏右侧按钮组 (从右向左布局)
        btn_w = int(112 * m["s"])
        bx = W - m["L"]
        buttons = [
            ("退出 [X]", "exit", True),
            ("导出G-code [E]", "export", bool(self.gcode_text)),
            ("识别定位 [空格]", "analyze", bool(self.samples and self.sel_idx >= 0)),
        ]
        for label, _act, enabled in buttons:
            bx -= btn_w + int(8 * m["s"])
            self._draw_button(canvas, (bx, int(14 * m["s"]), bx + btn_w, int(14 * m["s"]) + m["btn_h"]),
                              label, enabled=enabled)

        # 按钮组最左侧：工位地图选择下拉按钮
        sc_w = int(185 * m["s"])
        bx -= sc_w + int(8 * m["s"])
        self._workspace_rect = (bx, int(14 * m["s"]), bx + sc_w, int(14 * m["s"]) + m["btn_h"])
        is_sc_open = (self.active_dropdown == "WORKSPACE_DROPDOWN")
        self._draw_dropdown_button(canvas, self._workspace_rect, f"地图: {self.current_workspace_name}", is_open=is_sc_open)
        self._buttons.append((self._workspace_rect, ("toggle_dd", "WORKSPACE_DROPDOWN")))"""

new_render_buttons = """        # 标题栏右侧按钮组 (从右向左布局)
        btn_w = int(112 * m["s"])
        bx = W - m["L"]
        buttons = [
            ("退出 [X]", "exit", True),
            ("导出G-code [E]", "export", bool(self.gcode_text)),
            ("识别定位 [空格]", "analyze", bool(self.samples and self.sel_idx >= 0)),
        ]
        for label, _act, enabled in buttons:
            bx -= btn_w + int(8 * m["s"])
            self._draw_button(canvas, (bx, int(14 * m["s"]), bx + btn_w, int(14 * m["s"]) + m["btn_h"]),
                              label, enabled=enabled)

        # 3 个阶段 Checkbox 控件：3.位姿、2.骨架、1.前景 (从右向左排在【识别定位】按钮左侧)
        stage_names = ["1.前景", "2.骨架", "3.位姿"]
        cb_w = int(74 * m["s"])
        for s_i in (2, 1, 0):
            bx -= cb_w + int(6 * m["s"])
            cb_rect = (bx, int(14 * m["s"]), bx + cb_w, int(14 * m["s"]) + m["btn_h"])
            self._draw_checkbox(canvas, cb_rect, stage_names[s_i],
                                checked=self.stage_checked[s_i],
                                is_active=(self.active_stage_view == s_i))
            self._buttons.append((cb_rect, ("toggle_stage", s_i)))

        # 最左侧：工位地图选择下拉按钮
        sc_w = int(175 * m["s"])
        bx -= sc_w + int(10 * m["s"])
        self._workspace_rect = (bx, int(14 * m["s"]), bx + sc_w, int(14 * m["s"]) + m["btn_h"])
        is_sc_open = (self.active_dropdown == "WORKSPACE_DROPDOWN")
        self._draw_dropdown_button(canvas, self._workspace_rect, f"地图: {self.current_workspace_name}", is_open=is_sc_open)
        self._buttons.append((self._workspace_rect, ("toggle_dd", "WORKSPACE_DROPDOWN")))"""

assert old_render_buttons in code, "old_render_buttons not found"
code = code.replace(old_render_buttons, new_render_buttons)

# 6. 在 _draw_button 附近定义 _draw_checkbox
old_draw_btn = """        if enabled:
            self._buttons.append((rect, ("btn", label)))"""

new_draw_btn = """        if enabled:
            self._buttons.append((rect, ("btn", label)))

    def _draw_checkbox(self, canvas, rect, label, checked: bool, is_active: bool = False):
        \"\"\"三阶段流水线专属复选框 (支持选中状态、激活聚焦与悬停反馈)\"\"\"
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        hover = x1 <= mx <= x2 and y1 <= my <= y2
        m = self._metrics()

        # 背景与边框
        if is_active:
            bg = (42, 48, 56)
            border = (0, 235, 120)
        elif hover:
            bg = (34, 38, 44)
            border = GuiTheme.BORDER_HOVER
        else:
            bg = (26, 29, 34)
            border = (60, 66, 74)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if (is_active or hover) else 1)

        # 复选框小方格
        box_size = int(13 * m["s"])
        box_x = x1 + int(7 * m["s"])
        box_y = y1 + ((y2 - y1) - box_size) // 2
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_size, box_y + box_size), (90, 100, 115), 1)

        if checked:
            # 勾选态填充
            fill_col = (0, 220, 110) if is_active else (0, 170, 90)
            cv2.rectangle(canvas, (box_x + 2, box_y + 2), (box_x + box_size - 2, box_y + box_size - 2), fill_col, -1)
            # 对勾
            pts = np.array([
                [box_x + 2, box_y + box_size // 2],
                [box_x + box_size // 2 - 1, box_y + box_size - 3],
                [box_x + box_size - 2, box_y + 3]
            ], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (255, 255, 255), 1)

        # 文字标签
        text_x = box_x + box_size + int(5 * m["s"])
        text_col = (255, 255, 255) if is_active else ((220, 220, 220) if checked else (140, 140, 140))
        draw_text(canvas, label, (text_x, y1 + int(8 * m["s"])), m["fs_sub"], text_col, bold=is_active)"""

assert old_draw_btn in code, "old_draw_btn not found"
code = code.replace(old_draw_btn, new_draw_btn)

# 7. 在 _on_mouse 的 hit 判定增加 toggle_stage
old_hit = """                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                return"""

new_hit = """                if act_type == "toggle_dd":
                    self.active_dropdown = None if self.active_dropdown == act_val else act_val
                elif act_type == "btn":
                    self._on_button(act_val)
                elif act_type == "toggle_stage":
                    self._toggle_stage(act_val)
                return"""

assert old_hit in code, "old_hit not found"
code = code.replace(old_hit, new_hit)

# 8. 状态栏快捷键提示更新
old_status_help = """        draw_text(canvas, "[↑↓] 切换样本  ·  [空格] 识别定位  ·  [ESC]/[X] 退出  ·  Ctrl+滚轮缩放",
                  (W - int(410 * m["s"]), yb), m["fs_sub"], GuiTheme.TEXT_MUTED)"""

new_status_help = """        draw_text(canvas, "[↑↓] 样本  ·  [空格] 识别定位  ·  右键拖拽  ·  滚轮无级缩放  ·  双击复位",
                  (W - int(460 * m["s"]), yb), m["fs_sub"], GuiTheme.TEXT_MUTED)"""

if old_status_help in code:
    code = code.replace(old_status_help, new_status_help)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated tools/asparagus_offline.py successfully!")
