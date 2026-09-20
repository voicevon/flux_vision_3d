import os

file_path = r"d:\Software\antigravity\flux_vision_3d\tools\asparagus_offline.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 替换 __init__ 中的状态属性
old_init = """        # 三阶段透明流水线状态 (1.前景, 2.骨架, 3.位姿)
        self.stage_checked = [True, True, True]
        self.stage_vis = [None, None, None]
        self.active_stage_view = 2"""

new_init = """        # 视口显示图层切换 (0: 1.前景, 1: 2.骨架, 2: 3.位姿) - 纯显示切换，各阶段计算全部一次性完成
        self.active_view_mode = 2
        self.stage_vis = [None, None, None]"""

assert old_init in code, "old_init not found"
code = code.replace(old_init, new_init)

# 2. 替换 run_analyze 与 stage 切换逻辑
old_run = """    def run_analyze(self):
        \"\"\"对当前选中的样本执行【识别定位】(提取排名前三位芦笋、详细尺寸、方向、高度与高亮)\"\"\"
        if not (0 <= self.sel_idx < len(self.samples)):
            self.set_toast("请先在左侧列表中选择一张样本照片")
            return

        sample = self.samples[self.sel_idx]
        color = cv2.imread(sample["png"])
        if color is None:
            self.error = "彩色图读取失败"
            self.set_toast(self.error, True)
            return

        depth = None
        if sample["depth"]:
            try:
                depth = np.load(sample["depth"], allow_pickle=False)
                if depth.shape[:2] != color.shape[:2]:   # 深度与彩色分辨率不一致 → 对齐彩色尺寸
                    depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            except Exception as exc:
                log.warning("深度加载失败 (%s): %s", sample["depth"], exc)
                depth = None
        self.mode = "3d" if depth is not None else "2d"

        try:
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

new_run = """    def run_analyze(self):
        \"\"\"对当前选中的样本执行【识别定位】(一次性完成前景、骨架、位姿全部计算，结果数据全展示)\"\"\"
        if not (0 <= self.sel_idx < len(self.samples)):
            self.set_toast("请先在左侧列表中选择一张样本照片")
            return

        sample = self.samples[self.sel_idx]
        color = cv2.imread(sample["png"])
        if color is None:
            self.error = "彩色图读取失败"
            self.set_toast(self.error, True)
            return

        depth = None
        if sample["depth"]:
            try:
                depth = np.load(sample["depth"], allow_pickle=False)
                if depth.shape[:2] != color.shape[:2]:   # 深度与彩色分辨率不一致 → 对齐彩色尺寸
                    depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            except Exception as exc:
                log.warning("深度加载失败 (%s): %s", sample["depth"], exc)
                depth = None
        self.mode = "3d" if depth is not None else "2d"

        try:
            analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            # 无论选择哪种显示视图，后台一次性无条件解算全部 3 个阶段！
            self.targets = analyzer.analyze(color, depth, stages=(True, True, True))
            self.stage_vis = [analyzer.vis_stage1, analyzer.vis_stage2, analyzer.vis_stage3]
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
        独立切换视口显示的图层 (0: 1.前景, 1: 2.骨架, 2: 3.位姿)
        纯粹切换视口显示内容，完全独立 (individual)；右侧所有识别结果和文本数据始终保持展示！
        \"\"\"
        self.active_view_mode = mode_idx
        # 若当前样本尚未解算，自动执行一次性解算
        if self.stage_vis[mode_idx] is None and (0 <= self.sel_idx < len(self.samples)):
            self.run_analyze()
            return
        self._apply_view_mode()

    def _apply_view_mode(self):
        \"\"\"刷新视口显示的图像内容\"\"\"
        labels = ["1.前景物料 (ExG + 传送带 ROI)", "2.独立骨架 (Ridge Tracing 脊线)", "3.位姿定位 (Top 3 抓取目标)"]
        if self.stage_vis[self.active_view_mode] is not None:
            self.vis_img = self.stage_vis[self.active_view_mode]
            self.set_toast(f"视口显示: {labels[self.active_view_mode]}", duration=1.5)
        elif 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                self.vis_img = color.copy()"""

assert old_run in code, "old_run not found"
code = code.replace(old_run, new_run)

# 3. 替换 _select_target 中 stage_vis[2] 的逻辑
old_sel_target = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                self.stage_vis[2] = self.vis_img
                self.active_stage_view = 2"""

new_sel_target = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                self.stage_vis[2] = self.vis_img
                self.active_view_mode = 2"""

assert old_sel_target in code, "old_sel_target not found"
code = code.replace(old_sel_target, new_sel_target)

# 4. 替换 _draw_checkbox 为舒适的 _draw_segmented_tabs
old_draw_cb = """    def _draw_checkbox(self, canvas, rect, label, checked: bool, is_active: bool = False):
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

new_draw_cb = """    def _draw_view_pill(self, canvas, rect, label, is_active: bool):
        \"\"\"舒适高质感分段视图切换药丸 (纯显示层切换，互斥单选，视觉反馈鲜明)\"\"\"
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        hover = x1 <= mx <= x2 and y1 <= my <= y2
        m = self._metrics()

        if is_active:
            bg = (32, 68, 48)            # 沉稳翡翠绿底色
            border = (0, 235, 120)        # 鲜亮高光绿边框
            text_col = (255, 255, 255)    # 纯白加粗文字
        elif hover:
            bg = (34, 38, 46)
            border = (110, 130, 155)
            text_col = (235, 240, 245)
        else:
            bg = (24, 28, 34)
            border = (52, 58, 68)
            text_col = (165, 175, 185)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if is_active else (1 if hover else 1))

        # 激活项左侧绘制一个发光小圆点
        (tw, th), _ = measure_text(label, font_size=m["fs_sub"])
        if is_active:
            dot_x = x1 + int(10 * m["s"])
            dot_y = y1 + (y2 - y1) // 2
            cv2.circle(canvas, (dot_x, dot_y), int(3.5 * m["s"]), (0, 235, 120), -1)
            cv2.circle(canvas, (dot_x, dot_y), int(5 * m["s"]), (0, 235, 120), 1)
            tx = dot_x + int(8 * m["s"])
        else:
            tx = x1 + ((x2 - x1) - tw) // 2

        ty = y1 + ((y2 - y1) - th) // 2
        draw_text(canvas, label, (tx, ty), m["fs_sub"], text_col, bold=is_active)"""

assert old_draw_cb in code, "old_draw_cb not found"
code = code.replace(old_draw_cb, new_draw_cb)

# 5. 替换 render 中的 Checkbox 组为舒适的分段药丸切换器
old_render_cbs = """        # 3 个阶段 Checkbox 控件：3.位姿、2.骨架、1.前景 (从右向左排在【识别定位】按钮左侧)
        stage_names = ["1.前景", "2.骨架", "3.位姿"]
        cb_w = int(74 * m["s"])
        for s_i in (2, 1, 0):
            bx -= cb_w + int(6 * m["s"])
            cb_rect = (bx, int(14 * m["s"]), bx + cb_w, int(14 * m["s"]) + m["btn_h"])
            self._draw_checkbox(canvas, cb_rect, stage_names[s_i],
                                checked=self.stage_checked[s_i],
                                is_active=(self.active_stage_view == s_i))
            self._buttons.append((cb_rect, ("toggle_stage", s_i)))"""

new_render_cbs = """        # 视口显示层切换分段选择器：[ 1.前景 | 2.骨架 | 3.位姿 ] (纯显示，完全独立)
        view_names = ["1.前景", "2.骨架", "3.位姿"]
        pill_w = int(76 * m["s"])
        for v_i in (2, 1, 0):
            bx -= pill_w + int(6 * m["s"])
            pill_rect = (bx, int(14 * m["s"]), bx + pill_w, int(14 * m["s"]) + m["btn_h"])
            self._draw_view_pill(canvas, pill_rect, view_names[v_i], is_active=(self.active_view_mode == v_i))
            self._buttons.append((pill_rect, ("set_view_mode", v_i)))

        # 分段选择器左侧优雅提示标签：“显示:”
        bx -= int(42 * m["s"])
        draw_text(canvas, "显示:", (bx, int(22 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)"""

assert old_render_cbs in code, "old_render_cbs not found"
code = code.replace(old_render_cbs, new_render_cbs)

# 6. 替换 _on_mouse 中的 toggle_stage
old_hit = """                elif act_type == "toggle_stage":
                    self._toggle_stage(act_val)"""

new_hit = """                elif act_type == "set_view_mode":
                    self._select_view_mode(act_val)"""

assert old_hit in code, "old_hit not found"
code = code.replace(old_hit, new_hit)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated view selector logic in tools/asparagus_offline.py successfully!")
