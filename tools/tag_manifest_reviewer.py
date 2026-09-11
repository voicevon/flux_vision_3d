#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 观测样本轻量级 OpenCV 交互审核画板 (Tag Manifest Reviewer)
==================================================================
核心特性：
  1. 全 GUI 视觉交互操作栏 (Toolbar)：
     - 底部常驻美观的 GUI 按钮组，鼠标直接点击即可完成全部操作：
       [◀ 上一张 (A)] [下一张 (D) ▶] [🔍 重新识别本图] [🔄 全量重扫全部] [↺ 复位 (R)] [💾 保存 (S)] [🚪 保存退出 (Q)]
  2. 智能增量重新扫描识别 (Rescan & Discover New Tags)：
     - 实时调用最新双路检测算法与 18~29 号 12 枚白名单；
     - 增量发掘新标靶，智能继承保留已有人工剔除/保留标记 (keep)，绝不误抹除工作成果。
  3. 鼠标直接单击标靶翻转状态：
     - 点击任意标靶多边形内部，毫秒级翻转【保留 (绿+3D棱柱)】与【剔除 (红+蒙版+大叉)】。
  4. 毫秒级拓扑连通性安全守门员：
     - 顶栏实时计算并显示共视连通健康度与红绿灯警报，防呆阻断孤立断网。
  5. 键鼠无缝协同：
     - 鼠标点击按钮与键盘快捷键 (A/D/R/S/Q) 双通道支持。
"""

import os
import sys
import time
import copy
import yaml
import argparse
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from tools.tag_map_builder import TagMapBuilder
except ImportError:
    TagMapBuilder = None


class TagManifestReviewer:
    def __init__(self, 
                 manifest_path: str = "data/tag_calibration_images/tag_observations.yaml",
                 builder: Optional[Any] = None):
        """
        初始化交互式审核画板
        :param manifest_path: tag_observations.yaml 文件路径
        :param builder: TagMapBuilder 实例（用于连通性校验与 3D 渲染）
        """
        self.manifest_path = manifest_path
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"未找到观测清单文件: {self.manifest_path}，请先执行扫描导出！")

        self.builder = builder if builder is not None else TagMapBuilder(marker_size_mm=50.0)
        self.window_name = "AprilTag Observations Reviewer (GUI Button Toolbar & Interactive Review)"

        # 加载清单数据
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.raw_manifest = yaml.safe_load(f) or {}

        # 组织有序的图像列表
        self.image_keys = sorted(list(self.raw_manifest.get("images", {}).keys()))
        if not self.image_keys:
            raise ValueError("观测清单中未包含任何图像观测数据！")

        # 备份初始状态以供单帧复位 (R 键)
        self.initial_states = {}
        for k in self.image_keys:
            self.initial_states[k] = [
                copy.deepcopy(obs.get("keep", True)) 
                for obs in self.raw_manifest["images"][k].get("observations", [])
            ]

        self.current_idx = 0
        self.has_unsaved_changes = False
        self.display_img = None
        self.last_covis_report = {"is_valid": True, "message": "", "all_tags": []}

        # GUI 按钮热区列表: [(btn_action_id, (x1, y1, x2, y2), label, is_enabled)]
        self.gui_action_buttons = []
        self.mouse_hover_pos = (-1, -1)

        # Toast 通知
        self.toast_msg = ""
        self.toast_time = 0.0

        # 首次计算拓扑健康度
        self.update_topology()

    def set_toast(self, msg: str):
        self.toast_msg = msg
        self.toast_time = time.time()

    def update_topology(self):
        """实时重算当前全部保留观测下的共视连通性状态"""
        frame_detections = []
        for k in self.image_keys:
            tags_in_frame = {}
            for obs in self.raw_manifest["images"][k].get("observations", []):
                if obs.get("keep", True):
                    tid = int(obs["tag_id"])
                    corners = np.array(obs["corners"], dtype=np.float64)
                    tags_in_frame[tid] = corners
            if len(tags_in_frame) >= 2:
                frame_detections.append(tags_in_frame)

        self.last_covis_report = self.builder.validate_covisibility(frame_detections)

    def rescan_current_frame(self):
        """重新使用最新双路检测器与白名单扫描识别当前图像，增量补充新 Tag"""
        curr_key = self.image_keys[self.current_idx]
        img_info = self.raw_manifest["images"][curr_key]
        raw_path = img_info.get("image_path", "")

        raw_img = cv2.imread(raw_path)
        if raw_img is None:
            self.set_toast(f"无法读取图片: {raw_path}")
            return

        # 运行最新检测器
        detected = self.builder.detect_tags(raw_img)
        existing_obs = img_info.get("observations", [])
        existing_map = {int(obs["tag_id"]): obs for obs in existing_obs}

        added_count = 0
        updated_obs = []

        for tid in sorted(detected.keys()):
            corners = detected[tid]
            metrics = self.builder.compute_tag_metrics(corners)
            if tid in existing_map:
                # 继承已有人工标记，仅更新坐标
                old_entry = existing_map[tid]
                old_entry["corners"] = [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                old_entry["cell_size_px"] = metrics["cell_size_px"]
                old_entry["center_px"] = metrics["center_px"]
                old_entry["area_px"] = metrics["area_px"]
                updated_obs.append(old_entry)
            else:
                # 新发现的标靶！默认加入
                new_entry = {
                    "tag_id": int(tid),
                    "keep": True,
                    "cell_size_px": metrics["cell_size_px"],
                    "center_px": metrics["center_px"],
                    "area_px": metrics["area_px"],
                    "note": "全新重新扫描识别检出",
                    "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                }
                updated_obs.append(new_entry)
                added_count += 1

        img_info["observations"] = updated_obs
        img_info["detected_count"] = len(updated_obs)
        self.has_unsaved_changes = True
        self.update_topology()
        self.render_current_frame()

        if added_count > 0:
            msg = f"本图识别完成: 共 {len(updated_obs)} 个标靶 (+新增 {added_count} 个)"
        else:
            msg = f"本图识别完成: 检出 {len(updated_obs)} 个标靶 (已是最新)"
        self.set_toast(msg)
        print(f"[RESCAN] [{curr_key}] {msg}")

    def rescan_all_frames(self):
        """全量重新扫描数据集中所有图像并更新清单"""
        total_found = 0
        total_new = 0

        for k in self.image_keys:
            img_info = self.raw_manifest["images"][k]
            raw_path = img_info.get("image_path", "")
            raw_img = cv2.imread(raw_path)
            if raw_img is None:
                continue

            detected = self.builder.detect_tags(raw_img)
            existing_obs = img_info.get("observations", [])
            existing_map = {int(obs["tag_id"]): obs for obs in existing_obs}

            updated_obs = []
            for tid in sorted(detected.keys()):
                corners = detected[tid]
                metrics = self.builder.compute_tag_metrics(corners)
                if tid in existing_map:
                    old_entry = existing_map[tid]
                    old_entry["corners"] = [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                    old_entry["cell_size_px"] = metrics["cell_size_px"]
                    old_entry["center_px"] = metrics["center_px"]
                    old_entry["area_px"] = metrics["area_px"]
                    updated_obs.append(old_entry)
                else:
                    new_entry = {
                        "tag_id": int(tid),
                        "keep": True,
                        "cell_size_px": metrics["cell_size_px"],
                        "center_px": metrics["center_px"],
                        "area_px": metrics["area_px"],
                        "note": "全量重扫检出",
                        "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                    }
                    updated_obs.append(new_entry)
                    total_new += 1
                total_found += 1

            img_info["observations"] = updated_obs
            img_info["detected_count"] = len(updated_obs)

        self.has_unsaved_changes = True
        self.update_topology()
        self.render_current_frame()
        self.set_toast(f"全量重扫完成: 累计检出 {total_found} 次观测 (+新增 {total_new} 项)")
        print(f"[RESCAN ALL] 全量重扫完成: 累计 {total_found} 次观测，新增 {total_new} 项标靶！")

    def on_mouse_event(self, event, x, y, flags, param):
        """鼠标交互处理：点击按钮区或点击标靶多边形"""
        self.mouse_hover_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. 优先检查是否点击了底部的 GUI 按钮
            for btn_id, (bx1, by1, bx2, by2), label, is_enabled in self.gui_action_buttons:
                if bx1 <= x <= bx2 and by1 <= y <= by2 and is_enabled:
                    if btn_id == "PREV":
                        if self.current_idx > 0:
                            self.current_idx -= 1
                            self.render_current_frame()
                    elif btn_id == "NEXT":
                        if self.current_idx < len(self.image_keys) - 1:
                            self.current_idx += 1
                            self.render_current_frame()
                    elif btn_id == "RESCAN_CURR":
                        self.rescan_current_frame()
                    elif btn_id == "RESCAN_ALL":
                        self.rescan_all_frames()
                    elif btn_id == "RESET":
                        curr_key = self.image_keys[self.current_idx]
                        if curr_key in self.initial_states:
                            init_keeps = self.initial_states[curr_key]
                            observations = self.raw_manifest["images"][curr_key].get("observations", [])
                            for i, keep_val in enumerate(init_keeps):
                                if i < len(observations):
                                    observations[i]["keep"] = keep_val
                            self.has_unsaved_changes = True
                            self.update_topology()
                            self.render_current_frame()
                            self.set_toast("已恢复为初始加载状态")
                    elif btn_id == "SAVE":
                        self.save_changes()
                        self.render_current_frame()
                        self.set_toast("清单已即时保存 (tag_observations.yaml)")
                    elif btn_id == "EXIT":
                        if self.has_unsaved_changes:
                            self.save_changes()
                        self.is_running = False
                    return

            # 2. 检查是否点击了画面中的标靶多边形内部
            curr_key = self.image_keys[self.current_idx]
            observations = self.raw_manifest["images"][curr_key].get("observations", [])
            
            hit_idx = -1
            for idx in reversed(range(len(observations))):
                obs = observations[idx]
                corners = np.array(obs["corners"], dtype=np.float32)
                dist = cv2.pointPolygonTest(corners, (float(x), float(y)), False)
                if dist >= 0:
                    hit_idx = idx
                    break

            if hit_idx != -1:
                target_obs = observations[hit_idx]
                old_keep = target_obs.get("keep", True)
                new_keep = not old_keep
                target_obs["keep"] = new_keep
                self.has_unsaved_changes = True

                status_str = "【保留】" if new_keep else "【剔除】"
                print(f"[CLICK] [{curr_key}] Tag #{target_obs['tag_id']} 状态切换为: {status_str}")
                self.set_toast(f"Tag #{target_obs['tag_id']} -> {status_str}")

                # 毫秒级重算拓扑
                self.update_topology()
                self.render_current_frame()

    def on_mouse_click(self, event, x, y, flags, param):
        """兼容别名回调"""
        self.on_mouse_event(event, x, y, flags, param)

    def render_current_frame(self):
        """绘制当前帧画面，融合标靶双态、顶部状态栏与底部 GUI 按钮栏"""
        curr_key = self.image_keys[self.current_idx]
        img_info = self.raw_manifest["images"][curr_key]
        raw_path = img_info.get("image_path", "")

        raw_img = cv2.imread(raw_path)
        if raw_img is None:
            raw_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(raw_img, f"Image not found: {raw_path}", (100, 540),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        disp = raw_img.copy()
        h, w = disp.shape[:2]
        observations = img_info.get("observations", [])

        num_kept = 0
        num_excl = 0
        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

        # 1. 渲染所有标靶
        for obs in observations:
            tid = int(obs["tag_id"])
            keep = obs.get("keep", True)
            corners = np.array(obs["corners"], dtype=np.float32)
            pts_int = corners.astype(np.int32)
            cx = int(np.mean(corners[:, 0]))
            min_y = int(np.min(corners[:, 1]))
            badge_x = max(10, cx - 60)
            badge_y = max(70, min_y - 14)

            cell_size = obs.get("cell_size_px", [0, 0])

            if keep:
                num_kept += 1
                # 绿色边框
                cv2.polylines(disp, [pts_int], True, (0, 255, 0), 2, cv2.LINE_AA)
                # 4 顶点彩色圆点
                for pt_idx, pt in enumerate(pts_int):
                    cv2.circle(disp, tuple(pt), 5, dot_colors[pt_idx], -1)

                # 绿色标牌
                tag_text = f"Tag {tid}" + (" [ORIGIN]" if tid == 0 else "")
                cell_text = f"Cell: {cell_size[0]}x{cell_size[1]}px"
                cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 116, badge_y + 8), (20, 20, 20), -1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 116, badge_y + 8), (0, 255, 255), 1)
                cv2.putText(disp, tag_text, (badge_x, badge_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp, cell_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

                # 3D 正四棱柱 Z 轴 (实心方柱体)
                try:
                    self.builder.render_tag_3d_axes(disp, corners, tid)
                except Exception:
                    pass

            else:
                num_excl += 1
                # 红色边框
                cv2.polylines(disp, [pts_int], True, (0, 0, 255), 3, cv2.LINE_AA)
                
                # 半透明红色蒙版
                overlay = disp.copy()
                cv2.fillPoly(overlay, [pts_int], (0, 0, 220))
                cv2.addWeighted(overlay, 0.38, disp, 0.62, 0, disp)

                # 红色对角大叉号
                cv2.line(disp, tuple(pts_int[0]), tuple(pts_int[2]), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(disp, tuple(pts_int[1]), tuple(pts_int[3]), (0, 0, 255), 3, cv2.LINE_AA)

                # 红色醒目标牌
                tag_text = f"Tag {tid} [EXCLUDED]"
                cv2.rectangle(disp, (badge_x - 6, badge_y - 25), (badge_x + 140, badge_y + 5), (15, 15, 15), -1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 25), (badge_x + 140, badge_y + 5), (0, 0, 255), 2)
                cv2.putText(disp, tag_text, (badge_x, badge_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 80, 255), 2, cv2.LINE_AA)

        # 2. 顶部半透明状态栏 (高 58px)
        bar_overlay = disp.copy()
        cv2.rectangle(bar_overlay, (0, 0), (w, 58), (15, 15, 15), -1)
        cv2.addWeighted(bar_overlay, 0.85, disp, 0.15, 0, disp)
        cv2.line(disp, (0, 58), (w, 58), (70, 70, 70), 1)

        save_indicator = " [*有未保存修改*]" if self.has_unsaved_changes else ""
        left_info = f"[{self.current_idx + 1}/{len(self.image_keys)}] {curr_key} | 观测:{len(observations)} (保留:{num_kept}, 剔除:{num_excl}){save_indicator}"
        cv2.putText(disp, left_info, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        # 拓扑连通性安全指示灯
        if self.last_covis_report["is_valid"]:
            topo_str = f"[TOPOLOGY: PASS 连通网健康 ({len(self.last_covis_report['all_tags'])} Tags 处于连通图)]"
            topo_color = (0, 255, 0)
        else:
            unconn = self.last_covis_report.get("unconnected_tags", [])
            topo_str = f"[TOPOLOGY ALERT 严重断网告警: Tag {unconn} 已失联!]"
            topo_color = (0, 0, 255)
        
        cv2.putText(disp, topo_str, (w - 680, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, topo_color, 2, cv2.LINE_AA)

        # 顶栏第二行：简易提示
        tips_text = "操作说明: 直接鼠标左键单击画面中标靶切换【保留/剔除】; 底部提供了完整的图形操作按钮"
        cv2.putText(disp, tips_text, (15, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        # 3. 底部全新 GUI 操作按钮栏 (高 50px)
        tb_h = 52
        tb_y1 = h - tb_h
        tb_overlay = disp.copy()
        cv2.rectangle(tb_overlay, (0, tb_y1), (w, h), (18, 18, 18), -1)
        cv2.addWeighted(tb_overlay, 0.88, disp, 0.12, 0, disp)
        cv2.line(disp, (0, tb_y1), (w, tb_y1), (80, 80, 80), 1)

        # 规划 7 个工业级 GUI 操作按钮
        btn_defs = [
            ("PREV", "[< 上一张 (A)]", 130, self.current_idx > 0, (60, 60, 60), (0, 180, 220)),
            ("NEXT", "[下一张 (D) >]", 130, self.current_idx < len(self.image_keys) - 1, (60, 60, 60), (0, 180, 220)),
            ("RESCAN_CURR", "[重新识别本图]", 145, True, (40, 80, 140), (80, 160, 255)),
            ("RESCAN_ALL", "[全量重扫全部]", 145, True, (30, 90, 100), (60, 200, 220)),
            ("RESET", "[复位当前帧 (R)]", 145, True, (80, 60, 40), (220, 160, 60)),
            ("SAVE", "[即时保存清单 (S)]", 160, True, (0, 100, 40), (0, 230, 100)),
            ("EXIT", "[保存并退出 (Q)]", 145, True, (100, 30, 30), (240, 80, 80)),
        ]

        self.gui_action_buttons.clear()
        bx = 15
        by1 = tb_y1 + 8
        by2 = tb_y1 + 42

        mx, my = self.mouse_hover_pos

        for action_id, label, bw, enabled, normal_bg, active_border in btn_defs:
            bx2 = bx + bw
            # 悬停检测
            is_hover = (enabled and (bx <= mx <= bx2 and by1 <= my <= by2))
            
            if not enabled:
                bg_col = (35, 35, 35)
                border_col = (60, 60, 60)
                txt_col = (100, 100, 100)
            elif is_hover:
                # 悬停高亮
                bg_col = tuple(min(255, c + 35) for c in normal_bg)
                border_col = (255, 255, 255)
                txt_col = (255, 255, 255)
            else:
                bg_col = normal_bg
                border_col = active_border
                txt_col = (235, 235, 235)

            # 绘制按钮矩形
            cv2.rectangle(disp, (bx, by1), (bx2, by2), bg_col, -1)
            cv2.rectangle(disp, (bx, by1), (bx2, by2), border_col, 2 if is_hover else 1)
            
            # 文字居中绘制
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            tx = bx + (bw - tw) // 2
            ty = by1 + (by2 - by1 + th) // 2
            cv2.putText(disp, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, txt_col, 1, cv2.LINE_AA)

            self.gui_action_buttons.append((action_id, (bx, by1, bx2, by2), label, enabled))
            bx += bw + 12

        # 4. Toast 临时浮层通知
        if time.time() - self.toast_time < 2.2 and self.toast_msg:
            (tw, th), _ = cv2.getTextSize(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            toast_x = (w - tw) // 2
            toast_y = h - 100
            cv2.rectangle(disp, (toast_x - 16, toast_y - 28), (toast_x + tw + 16, toast_y + 10), (0, 0, 0), -1)
            cv2.rectangle(disp, (toast_x - 16, toast_y - 28), (toast_x + tw + 16, toast_y + 10), (0, 230, 255), 2)
            cv2.putText(disp, self.toast_msg, (toast_x, toast_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        self.display_img = disp
        cv2.imshow(self.window_name, self.display_img)

    def save_changes(self):
        """将内存中的修改写回 tag_observations.yaml"""
        total_obs = 0
        total_kept = 0
        total_excl = 0
        for k in self.image_keys:
            for obs in self.raw_manifest["images"][k].get("observations", []):
                total_obs += 1
                if obs.get("keep", True):
                    total_kept += 1
                else:
                    total_excl += 1

        self.raw_manifest["summary"]["total_observations"] = total_obs
        self.raw_manifest["summary"]["total_kept"] = total_kept
        self.raw_manifest["summary"]["total_excluded"] = total_excl
        self.raw_manifest["summary"]["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_comments = (
            "# ==============================================================================\n"
            "# AprilTag 离线建图与 BA 平差观测数据清单 (Observations Manifest)\n"
            "# (已由交互式审核画板 tag_manifest_reviewer.py 同步更新)\n"
            "# ==============================================================================\n\n"
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(header_comments)
            yaml.dump(self.raw_manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        self.has_unsaved_changes = False
        print(f"[SAVE] 审核修改已成功保存至: {self.manifest_path}")
        print(f"       累计统计: 保留 {total_kept} 次，已剔除 {total_excl} 次")

    def run(self):
        """启动 OpenCV 交互事件循环"""
        self.is_running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)

        print("\n" + "=" * 68)
        print("   【AprilTag 观测样本全 GUI 交互审核画板已启动】")
        print("   * 鼠标左键：单击标靶翻转剔除/保留；点击底部按钮直接执行操作")
        print("   * GUI 按钮：[◀ 上一张] [下一张 ▶] [重新识别本图] [全量重扫] [复位] [保存] [退出]")
        print("   * 快捷键  ：A/D 翻页 | R 复位 | S 保存 | Q/ESC 退出")
        print("=" * 68 + "\n")

        self.render_current_frame()

        while self.is_running:
            key = cv2.waitKey(25) & 0xFF
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # 翻页: A / 左箭头
            if key in (ord('a'), ord('A'), 81):
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.render_current_frame()
                else:
                    self.set_toast("已经是第一张图片")
                    self.render_current_frame()

            # 翻页: D / 右箭头
            elif key in (ord('d'), ord('D'), 83):
                if self.current_idx < len(self.image_keys) - 1:
                    self.current_idx += 1
                    self.render_current_frame()
                else:
                    self.set_toast("已经是最后一张图片")
                    self.render_current_frame()

            # 重新识别当前帧: F / 扫描
            elif key in (ord('f'), ord('F')):
                self.rescan_current_frame()

            # 复位本帧: R
            elif key in (ord('r'), ord('R')):
                curr_key = self.image_keys[self.current_idx]
                if curr_key in self.initial_states:
                    init_keeps = self.initial_states[curr_key]
                    observations = self.raw_manifest["images"][curr_key].get("observations", [])
                    for i, keep_val in enumerate(init_keeps):
                        if i < len(observations):
                            observations[i]["keep"] = keep_val
                    self.has_unsaved_changes = True
                    self.update_topology()
                    self.render_current_frame()
                    self.set_toast(f"[{curr_key}] 已恢复为初始加载状态")

            # 保存: S
            elif key in (ord('s'), ord('S')):
                self.save_changes()
                self.render_current_frame()
                self.set_toast("清单已即时保存 (tag_observations.yaml)")

            # 退出: Q / ESC
            elif key in (ord('q'), ord('Q'), 27):
                if self.has_unsaved_changes:
                    self.save_changes()
                break

        cv2.destroyAllWindows()
        print("[EXIT] 交互审核完成，窗口已安全关闭。")


def main():
    parser = argparse.ArgumentParser(description="AprilTag 观测样本交互式审核画板")
    parser.add_argument("--manifest", type=str, default="data/tag_calibration_images/tag_observations.yaml",
                        help="观测清单 tag_observations.yaml 路径")
    args = parser.parse_args()

    reviewer = TagManifestReviewer(manifest_path=args.manifest)
    reviewer.run()


if __name__ == "__main__":
    main()
