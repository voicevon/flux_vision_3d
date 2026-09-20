import os

file_path = r"d:\Software\antigravity\flux_vision_3d\tools\asparagus_offline.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. 更新 __init__ 中的视图变量
old_init_part = """        # 视口显示图层切换 (0: 1.前景, 1: 2.骨架, 2: 3.位姿) - 纯显示切换，各阶段计算全部一次性完成
        self.active_view_mode = 2
        self.stage_vis = [None, None, None]"""

new_init_part = """        # 视口显示图层切换 (0: 1.前景, 1: 距离场, 2: 峰脊线, 3: 2.骨架, 4: 3.位姿) - 纯显示切换，一次性全量解算
        self.active_view_mode = 4
        self.stage_vis = [None, None, None, None, None]"""

assert old_init_part in code, "old_init_part not found"
code = code.replace(old_init_part, new_init_part)

# 2. 更新 _select_sample 重置 stage_vis
old_reset_vis = """        self.stage_vis = [None, None, None]"""
new_reset_vis = """        self.stage_vis = [None, None, None, None, None]"""
assert old_reset_vis in code, "old_reset_vis not found"
code = code.replace(old_reset_vis, new_reset_vis)

# 3. 更新 run_analyze 和 _select_view_mode
old_run_part = """        try:
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

new_run_part = """        try:
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

assert old_run_part in code, "old_run_part not found"
code = code.replace(old_run_part, new_run_part)

# 4. 更新 _select_target 中的索引为 4 (3.位姿)
old_upd_vis = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                self.stage_vis[2] = self.vis_img
                self.active_view_mode = 2"""

new_upd_vis = """                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                self.stage_vis[4] = self.vis_img
                self.active_view_mode = 4"""

assert old_upd_vis in code, "old_upd_vis not found"
code = code.replace(old_upd_vis, new_upd_vis)

# 5. 更新 _metrics() 中的面板宽度：左栏减半 (140), 右栏减至 2/3 (236)
old_metrics = """            "list_w": int(280 * s),      # 左侧样本列表宽
            "right_w": int(352 * s),     # 右侧结果面板宽
            "header_h": int(52 * s),
            "bottom_h": int(46 * s),
            "row_h": int(44 * s),        # 样本/结果行高"""

new_metrics = """            "list_w": int(140 * s),      # 左侧样本列表宽 (按用户要求缩减至一半)
            "right_w": int(236 * s),     # 右侧结果面板宽 (按用户要求缩减至 2/3)
            "header_h": int(52 * s),
            "bottom_h": int(46 * s),
            "row_h": int(32 * s),        # 样本行高 (更紧凑一屏浏览更多)"""

assert old_metrics in code, "old_metrics not found"
code = code.replace(old_metrics, new_metrics)

# 6. 更新 _draw_sample_list 适配 140px 紧凑宽度
old_sample_list = """            name = smp["name"]
            if len(name) > 26:
                name = name[:12] + "..." + name[-11:]
            draw_text(canvas, name, (x1 + int(10 * m["s"]), ry1 + int(5 * m["s"])),
                      m["fs_small"], GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB)
            tag = "3D 成对" if smp["depth"] else "仅 2D"
            tag_col = GuiTheme.OK if smp["depth"] else GuiTheme.WARN
            draw_text(canvas, tag, (x1 + int(10 * m["s"]), ry1 + row_h - int(18 * m["s"])),
                      m["fs_small"], tag_col)
            self._sample_rows.append(((x1 + 2, ry1, x2 - 2, ry2), idx))"""

new_sample_list = """            raw_name = smp["name"]
            short_name = raw_name.replace(".png", "").replace("view_", "")
            if len(short_name) > 10:
                short_name = short_name[-10:]
            
            draw_text(canvas, short_name, (x1 + int(8 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB, bold=is_sel)
            
            tag = "3D" if smp["depth"] else "2D"
            tag_col = GuiTheme.OK if smp["depth"] else GuiTheme.TEXT_MUTED
            draw_text(canvas, tag, (x2 - int(24 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], tag_col, bold=is_sel)
            self._sample_rows.append(((x1 + 2, ry1, x2 - 2, ry2), idx))"""

assert old_sample_list in code, "old_sample_list not found"
code = code.replace(old_sample_list, new_sample_list)

# 7. 更新 _draw_result_panel 适配 236px 紧凑宽度
old_result_render = """            badge = "【#1 最优推荐】" if is_top else f"【#{t.id} 候选目标】"
            badge_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else GuiTheme.TEXT_SUB)
            draw_text(canvas, f"{badge}  直径 D:{t.diam_mm}mm  |  长 L:{t.length_mm}mm",
                      (x1 + int(10 * m["s"]), ry1 + int(6 * m["s"])), m["fs_small"], badge_col, bold=is_sel)

            h_str = f"+{t.rel_height_mm}mm" if t.rel_height_mm > 0 else "--"
            draw_text(canvas, f"方向:{t.yaw_deg}°  |  台面凸起高度:{h_str}  (深度Z:{t.grip_z}mm)",
                      (x1 + int(10 * m["s"]), ry1 + int(24 * m["s"])), m["fs_small"],
                      GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB)

            draw_text(canvas, f"机械臂 SCARA: ({t.robot_x}, {t.robot_y}, {t.robot_z}) R:{t.robot_r}°",
                      (x1 + int(10 * m["s"]), ry1 + int(42 * m["s"])), m["fs_small"],
                      GuiTheme.ACCENT if is_sel else GuiTheme.TEXT_MUTED)"""

new_result_render = """            badge = "#1最优" if is_top else f"#{t.id}候选"
            badge_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else GuiTheme.TEXT_SUB)
            draw_text(canvas, f"{badge} D:{t.diam_mm} L:{int(t.length_mm)}mm",
                      (x1 + int(8 * m["s"]), ry1 + int(5 * m["s"])), m["fs_small"], badge_col, bold=True)

            h_str = f"+{t.rel_height_mm}mm" if t.rel_height_mm > 0 else (f"Z:{int(t.grip_z)}" if t.grip_z > 0 else "--")
            draw_text(canvas, f"方向:{t.yaw_deg}° 凸起:{h_str}",
                      (x1 + int(8 * m["s"]), ry1 + int(23 * m["s"])), m["fs_small"],
                      GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB)

            draw_text(canvas, f"S:({int(t.robot_x)},{int(t.robot_y)},{int(t.robot_z)}) R:{int(t.robot_r)}°",
                      (x1 + int(8 * m["s"]), ry1 + int(41 * m["s"])), m["fs_small"],
                      GuiTheme.ACCENT if is_sel else GuiTheme.TEXT_MUTED)"""

assert old_result_render in code, "old_result_render not found"
code = code.replace(old_result_render, new_result_render)

# 8. 更新 render() 中的分段药丸组为 5 个视图
old_view_pills = """        # 视口显示层切换分段选择器：[ 1.前景 | 2.骨架 | 3.位姿 ] (纯显示，完全独立)
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

new_view_pills = """        # 视口算法流程全透明分段选择器：[ 1.前景 | 距离场 | 峰脊线 | 2.骨架 | 3.位姿 ]
        view_names = ["1.前景", "距离场", "峰脊线", "2.骨架", "3.位姿"]
        pill_w = int(60 * m["s"])
        for v_i in (4, 3, 2, 1, 0):
            bx -= pill_w + int(4 * m["s"])
            pill_rect = (bx, int(14 * m["s"]), bx + pill_w, int(14 * m["s"]) + m["btn_h"])
            self._draw_view_pill(canvas, pill_rect, view_names[v_i], is_active=(self.active_view_mode == v_i))
            self._buttons.append((pill_rect, ("set_view_mode", v_i)))

        # 分段选择器左侧提示标签：“显示:”
        bx -= int(38 * m["s"])
        draw_text(canvas, "显示:", (bx, int(22 * m["s"])), m["fs_sub"], GuiTheme.TEXT_MUTED)"""

assert old_view_pills in code, "old_view_pills not found"
code = code.replace(old_view_pills, new_view_pills)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated asparagus_offline.py with 5-stage CV pipeline views and adjusted panel widths!")
