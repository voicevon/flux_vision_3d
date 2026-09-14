#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 离线标定综合工作站单元测试 (test_tag_offline_studio.py)
============================================================
覆盖测试用例：
  1. Studio 初始化与领域模型装配校验；
  2. 帧序列资产列表筛选过滤 (All / Warning / Excluded)；
  3. 单帧状态翻转 (保留 ⇋ 剔除) 与缓存同步；
  4. 单标靶状态翻转 (Keep / Exclude) 与当前帧残差重算；
  5. 三栏式 GUI 渲染流水线无异常闭环；
  6. 鼠标事件与按钮点击分发；
  7. 全景质检报告导出与 Markdown 文件完整性校验。
"""

import os
import sys
import glob
import shutil
import tempfile
import unittest
import numpy as np
import cv2
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.calibration.tag_offline_studio import TagOfflineStudio


class TestTagOfflineStudio(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # 生成 3 张模拟测试图片
        for i in range(1, 4):
            img = np.full((1080, 1920, 3), 40 + i * 10, dtype=np.uint8)
            cv2.imwrite(os.path.join(self.image_dir, f"view_{i:04d}.png"), img)

        self.map_path = os.path.join(self.temp_dir, "test_tags_map.yaml")
        # 复制或写入一个基础测试地图
        real_map = os.path.join(PROJECT_ROOT, "config", "tags_map.yaml")
        if os.path.exists(real_map):
            shutil.copy(real_map, self.map_path)

        self.manifest_path = os.path.join(self.temp_dir, "test_tag_observations.yaml")
        init_manifest = {
            "version": "2.0_test",
            "tag_family": "DICT_APRILTAG_16h5",
            "marker_size_mm": 50.0,
            "summary": {"total_images": 3, "total_observations": 3, "total_kept": 3},
            "images": {
                f"view_{i:04d}.png": {
                    "file_name": f"view_{i:04d}.png",
                    "image_path": os.path.join(self.image_dir, f"view_{i:04d}.png"),
                    "detected_count": 1,
                    "enabled": True,
                    "observations": [
                        {
                            "tag_id": i,
                            "keep": True,
                            "corners": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
                        }
                    ]
                }
                for i in range(1, 4)
            }
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(init_manifest, f)

        self.studio = TagOfflineStudio(
            map_path=self.map_path,
            image_dir=self.image_dir,
            marker_size_mm=50.0,
            win_w=1920,
            win_h=1080,
            manifest_path=self.manifest_path
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试 Studio 初始化与资产发现"""
        self.assertEqual(len(self.studio.image_files), 3, "应扫描到 3 张测试采图")
        self.assertEqual(self.studio.current_img_idx, 0, "默认初始选中第 0 帧")
        self.assertIsNotNone(self.studio.engine)
        self.assertIsNotNone(self.studio.manifest_repo)
        self.assertIsNotNone(self.studio.optimizer)
        self.assertIsNotNone(self.studio.reporter)

    def test_filter_modes(self):
        """测试左栏列表在 All / Warning / Excluded 模式下的索引筛选"""
        indices_all = self.studio._get_filtered_indices()
        self.assertEqual(len(indices_all), 3)

        # 手动剔除第一帧
        bname = os.path.basename(self.studio.image_files[0])
        self.studio.frame_metrics_cache[bname]["is_excluded"] = True

        self.studio.filter_mode = "excluded"
        indices_excl = self.studio._get_filtered_indices()
        self.assertEqual(len(indices_excl), 1)
        self.assertEqual(indices_excl[0], 0)

        # 恢复全部模式
        self.studio.filter_mode = "all"
        self.assertEqual(len(self.studio._get_filtered_indices()), 3)

    def test_toggle_frame_exclusion(self):
        """测试单帧状态翻转"""
        bname = os.path.basename(self.studio.image_files[0])
        self.assertFalse(self.studio.frame_metrics_cache[bname]["is_excluded"])

        self.studio.current_img_idx = 0
        self.studio.toggle_current_frame_exclusion()
        self.assertTrue(self.studio.frame_metrics_cache[bname]["is_excluded"])

        # 再次翻转恢复
        self.studio.toggle_current_frame_exclusion()
        self.assertFalse(self.studio.frame_metrics_cache[bname]["is_excluded"])

    def test_toggle_tag_exclusion(self):
        """测试单帧内单标靶状态翻转"""
        bname = os.path.basename(self.studio.image_files[0])
        # 伪造一个观测
        self.studio.manifest_data.setdefault("images", {})[bname] = {
            "observations": [{"tag_id": 18, "corners": [[0, 0], [10, 0], [10, 10], [0, 10]], "keep": True}]
        }
        self.studio.current_img_idx = 0
        self.studio.toggle_tag_exclusion_in_current_frame(18)

        obs = self.studio.get_observations_for_image(bname)
        self.assertEqual(len(obs), 1)
        self.assertFalse(obs[0]["keep"])


    def test_gui_render_pipeline(self):
        """测试完整三栏 GUI 渲染流水线无异常 (含 Tag 叠加与残差矢量)"""
        bname = os.path.basename(self.studio.image_files[0])
        # 伪造带实际观测角点的帧数据
        self.studio.manifest_data.setdefault("images", {})[bname] = {
            "observations": [{
                "tag_id": 0,
                "corners": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]],
                "keep": True
            }]
        }
        self.studio.refresh_all_frame_metrics()

        canvas = np.zeros((self.studio.win_h, self.studio.win_w, 3), dtype=np.uint8)
        self.studio.is_ba_running = True
        self.studio.set_toast("测试单元运行中")

        # 执行全景渲染
        self.studio.render(canvas)

        self.assertGreater(len(self.studio.gui_buttons), 0, "应成功注册 GUI 交互按钮")
        self.assertGreater(canvas.shape[0], 0)
        self.assertGreater(canvas.shape[1], 0)

        # 直接测试 visualizer.draw_reprojection_vectors
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        obs_pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
        proj_pts = np.array([[12, 11], [25, 23]], dtype=np.float32)
        self.studio.visualizer.draw_reprojection_vectors(test_img, obs_pts, proj_pts, scale_factor=20.0)
        self.assertIsNotNone(test_img)

    def test_button_click_events(self):
        """测试鼠标点击事件分发"""
        self.studio.is_running = True
        # 模拟点击退出按钮
        self.studio._handle_button_click("EXIT", "EXIT", 0, 0)
        self.assertFalse(self.studio.is_running, "点击 EXIT 应将 is_running 置为 False")

        # 模拟点击选帧
        self.studio._handle_button_click("SELECT_FRAME_2", 2, 0, 0)
        self.assertEqual(self.studio.current_img_idx, 2, "应成功切换选定帧索引至 2")

    def test_export_verification_report(self):
        """测试全景质检报告导出"""
        self.studio.export_verification_report()
        report_dir = os.path.join(PROJECT_ROOT, "data", "tag_calibration_verification")
        reports = glob.glob(os.path.join(report_dir, "studio_qa_report_*.md"))
        self.assertGreater(len(reports), 0, "应成功生成质检报告 Markdown 文件")

    def test_viewport_zoom_and_pan(self):
        """测试视口区分区域滚轮 (左栏列表滚动 vs 中间画布缩放) 与平移重置"""
        # 1. 鼠标在左栏 (x=100, y=200): 滚轮只影响 scroll_offset
        self.assertEqual(self.studio.scroll_offset, 0)
        self.assertEqual(self.studio.zoom_level, 1.0)
        self.studio._on_mouse(cv2.EVENT_MOUSEWHEEL, mx=100, my=200, flags=-1, param=None)
        self.assertGreater(self.studio.scroll_offset, 0, "左栏滚轮应增加列表偏移")
        self.assertEqual(self.studio.zoom_level, 1.0, "左栏滚轮不应影响画布缩放")

        # 2. 鼠标在中间画布 (x=800, y=500): 滚轮只放大画布图像
        old_scroll = self.studio.scroll_offset
        self.studio._on_mouse(cv2.EVENT_MOUSEWHEEL, mx=800, my=500, flags=1, param=None)
        self.assertGreater(self.studio.zoom_level, 1.0, "中间画布滚轮向上应放大图像")
        self.assertEqual(self.studio.scroll_offset, old_scroll, "中间画布滚轮不应影响左栏列表")

        # 3. 鼠标右键在中间画布按住并拖拽
        old_pan_x = self.studio.pan_offset_x
        old_pan_y = self.studio.pan_offset_y
        self.studio._on_mouse(cv2.EVENT_RBUTTONDOWN, mx=800, my=500, flags=0, param=None)
        self.assertTrue(self.studio.is_panning)
        self.studio._on_mouse(cv2.EVENT_MOUSEMOVE, mx=830, my=520, flags=0, param=None)
        self.studio._on_mouse(cv2.EVENT_RBUTTONUP, mx=830, my=520, flags=0, param=None)
        self.assertFalse(self.studio.is_panning)
        self.assertEqual(self.studio.pan_offset_x, old_pan_x + 30.0)
        self.assertEqual(self.studio.pan_offset_y, old_pan_y + 20.0)


        # 4. 双击中间画布重置缩放与平移
        self.studio._on_mouse(cv2.EVENT_LBUTTONDBLCLK, mx=800, my=500, flags=0, param=None)
        self.assertEqual(self.studio.zoom_level, 1.0, "双击应重置缩放至 1.0x")
        self.assertEqual(self.studio.pan_offset_x, 0.0, "双击应重置平移偏置")
        self.assertEqual(self.studio.pan_offset_y, 0.0)

        # 5. 放大到 3.0x 下渲染画布无异常
        self.studio.zoom_level = 3.0
        canvas = np.zeros((self.studio.win_h, self.studio.win_w, 3), dtype=np.uint8)
        self.studio.render(canvas)
        self.assertGreater(canvas.shape[0], 0)

    def test_dropdowns_and_sorting(self):
        """测试下拉菜单展开、项选择、外部点击收起及按残差降序排序"""
        # 1. 点击展开 FILTER_DROPDOWN
        self.assertIsNone(self.studio.active_dropdown)
        self.studio._handle_button_click("TOGGLE_FILTER_DROPDOWN", "FILTER_DROPDOWN", 0, 0)
        self.assertEqual(self.studio.active_dropdown, "FILTER_DROPDOWN")

        # 2. 点击外部收起
        self.studio._on_mouse(cv2.EVENT_LBUTTONDOWN, mx=999, my=999, flags=0, param=None)
        self.assertIsNone(self.studio.active_dropdown, "点击外部应收起下拉菜单")

        # 3. 选择排序方式: 按残差降序 (err_desc)
        # 为 3 个图像伪造不同的残差
        f0 = os.path.basename(self.studio.image_files[0])
        f1 = os.path.basename(self.studio.image_files[1])
        f2 = os.path.basename(self.studio.image_files[2])
        self.studio.frame_metrics_cache[f0]["mean_err"] = 0.12
        self.studio.frame_metrics_cache[f1]["mean_err"] = 0.88  # 最高残差
        self.studio.frame_metrics_cache[f2]["mean_err"] = 0.45

        self.studio._handle_button_click("DD_SELECT_SORT_DROPDOWN_err_desc", ("SORT_DROPDOWN", "err_desc"), 0, 0)
        self.assertEqual(self.studio.sort_mode, "err_desc")
        self.assertIsNone(self.studio.active_dropdown)

        sorted_indices = self.studio._get_filtered_indices()
        self.assertEqual(sorted_indices[0], 1, "残差最高 (0.88px) 的 view_0002 应排在第 0 位")
        self.assertEqual(sorted_indices[1], 2, "残差次高 (0.45px) 的 view_0003 应排在第 1 位")
        self.assertEqual(sorted_indices[2], 0, "残差最低 (0.12px) 的 view_0001 应排在第 2 位")

    def test_view_mode_and_ba_progress(self):
        """测试视口双独立正交模式 (BA 理论与单帧实测) 与全局平差进度条渲染"""
        # 1. 测试默认双 3D 模式
        self.assertEqual(self.studio.ba_view_mode, "3d")
        self.assertEqual(self.studio.obs_view_mode, "3d")
        self.assertEqual(self.studio.right_bar_w, 180, "右侧栏应成功瘦身为 180px")

        # 2. 测试切换 BA 与 OBS 独立下拉框
        for ba_m in ["3d", "2d", "off"]:
            self.studio._handle_button_click(f"DD_SELECT_BA_VIEW_DROPDOWN_{ba_m}", ("BA_VIEW_DROPDOWN", ba_m), 0, 0)
            self.assertEqual(self.studio.ba_view_mode, ba_m)

        for obs_m in ["3d", "2d", "off"]:
            self.studio._handle_button_click(f"DD_SELECT_OBS_VIEW_DROPDOWN_{obs_m}", ("OBS_VIEW_DROPDOWN", obs_m), 0, 0)
            self.assertEqual(self.studio.obs_view_mode, obs_m)

        canvas = np.zeros((self.studio.win_h, self.studio.win_w, 3), dtype=np.uint8)
        self.studio.render(canvas)
        self.assertGreater(canvas.shape[0], 0)

        # 3. 测试带有大阶段+子阶段双进度条的渲染
        self.studio.is_ba_running = True
        self.studio.ba_progress = 0.65
        self.studio.ba_stage_text = "阶段 3/4: 两阶段 Cauchy 平差求解中..."
        self.studio.ba_sub_progress = 0.42
        self.studio.ba_sub_text = "[微容差深度平差] 轮次 #14/35 | 实时 RMSE: 0.198 px"
        canvas = np.zeros((self.studio.win_h, self.studio.win_w, 3), dtype=np.uint8)
        self.studio.render(canvas)
        self.assertGreater(canvas.shape[0], 0)

    def test_canvas_click_tag_toggle(self):
        """测试在中间视口图片上直接点击 Tag 触发剔除(打叉)与恢复"""
        bname = os.path.basename(self.studio.image_files[0])
        self.studio.current_img_idx = 0

        # 为测试帧注入确定性的观测标靶
        self.studio.manifest_data.setdefault("images", {})[bname] = {
            "observations": [{
                "tag_id": 99,
                "corners": [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
                "keep": True
            }]
        }
        self.studio.frame_metrics_cache[bname]["observations"] = self.studio.get_observations_for_image(bname)

        obs_list = self.studio.get_observations_for_image(bname)
        self.assertEqual(len(obs_list), 1)
        self.assertTrue(obs_list[0].get("keep", True), "默认初始应为保留状态")

        # 触发翻转
        self.studio.toggle_tag_exclusion_in_current_frame(99)
        obs_after = self.studio.get_observations_for_image(bname)
        self.assertFalse(obs_after[0]["keep"], "翻转后应变为剔除(打红叉)状态")

        # 再次翻转恢复
        self.studio.toggle_tag_exclusion_in_current_frame(99)
        obs_restored = self.studio.get_observations_for_image(bname)
        self.assertTrue(obs_restored[0]["keep"], "再次点击应恢复保留状态")

    def test_super_extract_and_persistence(self):
        """测试单帧超精重提取的规范字段生成与磁盘文件持久化"""
        bname = os.path.basename(self.studio.image_files[0])
        self.studio.current_img_idx = 0

        # Mock super_extractor 返回包含多字段的测试标靶
        fake_corners = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
        mock_result = {
            10: {
                "tag_id": 10,
                "corners": fake_corners,
                "channel": "CLAHE_8x8",
                "metrics": {
                    "cell_size_px": [16, 16],
                    "center_px": [150.0, 150.0],
                    "area_px": 10000.0
                }
            },
            11: {
                "tag_id": 11,
                "corners": fake_corners + 100,
                "channel": "RAW",
                "metrics": {
                    "cell_size_px": [16, 16],
                    "center_px": [250.0, 250.0],
                    "area_px": 10000.0
                }
            }
        }
        self.studio.data_mgr.super_extractor.extract_from_image = lambda path: mock_result

        # 执行超精重提取
        ret_bname, ret_count = self.studio.super_extract_current_frame()
        self.assertEqual(ret_bname, bname)
        self.assertEqual(ret_count, 2)

        # 检查内存中的 manifest 条目规范完整性
        img_entry = self.studio.manifest_data["images"][bname]
        self.assertEqual(img_entry["detected_count"], 2)
        self.assertTrue(img_entry["enabled"])
        self.assertIn("observations", img_entry)
        self.assertEqual(len(img_entry["observations"]), 2)
        self.assertEqual(img_entry["observations"][0]["channel"], "CLAHE_8x8")
        self.assertIn("cell_size_px", img_entry["observations"][0])

        # 核心持久化验证：新建一个 StudioDataManager 从磁盘加载 manifest_path
        from tools.calibration.studio.studio_state import StudioDataManager
        new_mgr = StudioDataManager(
            map_path=self.map_path,
            image_dir=self.image_dir,
            manifest_path=self.studio.manifest_path,
            engine=self.studio.engine,
            marker_size_mm=50.0
        )
        loaded_obs = new_mgr.get_observations_for_image(bname)
        self.assertEqual(len(loaded_obs), 2, "重新加载后超精提取的标靶数量必须100%保持，不可丢失")
        self.assertEqual(loaded_obs[0]["tag_id"], 10)
        self.assertEqual(loaded_obs[1]["tag_id"], 11)

    def test_reset_map(self):
        """测试地图一键复位：内存清空、地图文件备份与重写为空"""
        self.assertGreater(len(self.studio.tags_map_data.get("tags", {})), 0, "初始应装载了地图")

        # 执行地图复位
        success = self.studio.reset_map()
        self.assertTrue(success)
        self.assertEqual(len(self.studio.tags_map_data.get("tags", {})), 0, "复位后 tags 字典应为空")
        self.assertEqual(len(self.studio.engine.tags_map.get("tags", {})), 0, "引擎内绑定的地图也应同步清空")

        # 验证 .bak 备份文件存在
        bak_file = self.studio.map_path + ".bak"
        self.assertTrue(os.path.exists(bak_file), "复位前应自动生成地图 .bak 备份")

    def test_reset_all_keep_status(self):
        """测试一键复位所有观测保留状态"""
        bname = os.path.basename(self.studio.image_files[0])
        # 先剔除该帧并剔除某个 tag
        self.studio.toggle_image_exclusion(bname)
        self.assertTrue(self.studio.is_image_excluded(bname))

        # 执行一键复位
        restored_cnt = self.studio.reset_all_keep_status()
        self.assertFalse(self.studio.is_image_excluded(bname), "一键复位后帧应恢复为有效保留")

        # 从磁盘重新验证
        import yaml
        with open(self.studio.manifest_path, "r", encoding="utf-8") as f:
            disk_data = yaml.safe_load(f)
        self.assertTrue(disk_data["images"][bname]["enabled"])
        self.assertFalse(disk_data["images"][bname]["excluded"])

    def test_super_extract_all_frames(self):
        """测试全局全量超精重提取：旧角点与空间观测全部清空，全量重新超精拟合与落盘"""
        # Mock super_extractor 返回包含 2 个标靶的字典
        mock_result = {
            10: {
                "corners": np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]),
                "channel": "CLAHE_8x8",
                "metrics": {"cell_size_px": [25, 25], "center_px": [150.0, 150.0], "area_px": 10000.0}
            },
            20: {
                "corners": np.array([[300.0, 300.0], [400.0, 300.0], [400.0, 400.0], [300.0, 400.0]]),
                "channel": "BICUBIC_2X",
                "metrics": {"cell_size_px": [30, 30], "center_px": [350.0, 350.0], "area_px": 10000.0}
            }
        }
        self.studio.data_mgr.super_extractor.extract_from_image = lambda path: mock_result

        # 执行全局全量超精提取
        progress_records = []
        def on_prog(cur, total, bname, count):
            progress_records.append((cur, total, bname, count))

        total_frames, total_tags = self.studio.super_extract_all_frames(progress_callback=on_prog)
        self.assertEqual(total_frames, 3, "应处理全部 3 帧图片")
        self.assertEqual(total_tags, 6, "3 帧每帧 2 个标靶，总计 6 个标靶")
        self.assertEqual(len(progress_records), 3, "回调应触发 3 次")

        # 校验内存中的每帧标靶状态
        for bname in self.studio.manifest_data["images"]:
            obs = self.studio.manifest_data["images"][bname]["observations"]
            self.assertEqual(len(obs), 2, "旧标靶位置应被完全清空并重建为 2 个新标靶")
            self.assertTrue(obs[0]["keep"], "新提取的标靶应默认启用为有效保留")
            self.assertTrue(obs[1]["keep"])

        # 校验磁盘持久化落盘
        import yaml
        with open(self.studio.manifest_path, "r", encoding="utf-8") as f:
            disk_manifest = yaml.safe_load(f)
        for bname in disk_manifest["images"]:
            self.assertEqual(len(disk_manifest["images"][bname]["observations"]), 2)
            self.assertTrue(disk_manifest["images"][bname]["enabled"])

    def test_frame_diagnostics_and_gate_status(self):
        """测试单帧病因切片诊断算法、物理指标与准入门限/拓扑评估"""
        diag = self.studio.data_mgr.diagnose_frame(0)
        self.assertIn("contrast_rms", diag)
        self.assertIn("laplacian_var", diag)
        self.assertIn("mean_intensity", diag)
        self.assertIn("false_rejections_count", diag)
        self.assertIn("missing_projected_tags", diag)

        # 检查 StudioDataManager 的门限与拓扑字段
        self.assertIn(self.studio.gate_status, ["PASS", "ACCEPTABLE", "REVIEW"])
        self.assertIn("is_valid", self.studio.topology_status)
        self.assertIsInstance(self.studio.global_median_mm, float)

    def test_diagnostics_ui_and_viewport_rendering(self):
        """测试切换至病因诊断切片视图下的 GUI 渲染与动作按钮注册"""
        self.studio.toggle_frame_diagnostics()
        self.assertTrue(self.studio.show_frame_diagnostics)

        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.studio.render(canvas)

        # 校验注册的按钮中应包含 DIAGNOSE_FRAME 与 LAUNCH_AR
        btn_ids = [btn[0] for btn in self.studio.gui_buttons]
        self.assertIn("DIAGNOSE_FRAME", btn_ids)
        self.assertIn("LAUNCH_AR", btn_ids)

        # 模拟点击 DIAGNOSE_FRAME 切回常规视图
        self.studio._handle_button_click("DIAGNOSE_FRAME", "", 0, 0)
        self.assertFalse(self.studio.show_frame_diagnostics)

    def test_manifest_snapshot_and_restore(self):
        """测试审核清单快照深拷贝与一键撤销回滚"""
        # 1. 制作快照
        snap = self.studio.data_mgr.create_manifest_snapshot()
        self.assertIsNotNone(snap)
        orig_keep = self.studio.manifest_data["images"]["view_0001.png"]["observations"][0]["keep"]

        # 2. 模拟修改: 剔除一个标靶
        self.studio.manifest_data["images"]["view_0001.png"]["observations"][0]["keep"] = not orig_keep

        # 3. 恢复快照
        succ = self.studio.data_mgr.restore_manifest_snapshot()
        self.assertTrue(succ)
        self.assertEqual(
            self.studio.manifest_data["images"]["view_0001.png"]["observations"][0]["keep"],
            orig_keep,
            "恢复快照后标靶保留状态应精确还原"
        )

    def test_find_worst_prunable_topology_guard(self):
        """测试最大离差标靶寻找与拓扑安全守门员"""
        prunable = self.studio.data_mgr.find_worst_prunable_observations(top_k=5)
        self.assertIsInstance(prunable, list)

    def test_auto_prune_ba_and_settlement_ui(self):
        """测试自动迭代剪枝平差调度生命周期、结算数据包与UI交互"""
        fake_opt_res = {
            "final_rmse": 0.28,
            "final_tag_poses_aligned": {
                0: np.eye(4),
                18: np.eye(4),
                19: np.eye(4)
            }
        }
        call_count = [0]
        def fake_solve(*args, **kwargs):
            self.studio.data_mgr.global_rmse = 0.85 if call_count[0] == 0 else 0.35
            self.studio.data_mgr.global_median_mm = 1.60 if call_count[0] == 0 else 0.90
            return True, {"final_rmse": 0.35}, "收敛成功"
        self.studio.ba_runner._execute_ba_solve = fake_solve

        def fake_find(top_k=2):
            call_count[0] += 1
            if call_count[0] == 1:
                return [("view_0001.png", 0, 1.25, 2.5)]
            return []
        self.studio.data_mgr.find_worst_prunable_observations = fake_find

        self.studio.data_mgr.global_rmse = 0.85
        self.studio.data_mgr.global_median_mm = 1.60

        started = self.studio.start_auto_prune_ba()
        self.assertTrue(started)

        if self.studio.ba_runner.ba_thread:
            self.studio.ba_runner.ba_thread.join(timeout=5.0)

        res = self.studio.ba_runner.poll_result()
        self.assertIsNotNone(res)

        settle = self.studio.prune_settlement_data
        self.assertIsNotNone(settle)
        self.assertIn("initial_rmse", settle)
        self.assertIn("final_rmse", settle)
        self.assertIn("total_pruned_count", settle)

        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.studio.render(canvas)
        btn_ids = [btn[0] for btn in self.studio.gui_buttons]
        self.assertIn("ACCEPT_PRUNE", btn_ids)
        self.assertIn("UNDO_PRUNE", btn_ids)

        self.studio.undo_prune_results()
        self.assertIsNone(self.studio.prune_settlement_data)


if __name__ == "__main__":
    unittest.main()

