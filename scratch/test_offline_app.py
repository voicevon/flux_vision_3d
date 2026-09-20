import os
import cv2
import numpy as np
from tools.asparagus_offline import AsparagusOfflineApp

def test_offline_app():
    print("=== 开始端到端测试 AsparagusOfflineApp ===")
    app = AsparagusOfflineApp()
    print(f"当前工位: {app.current_workspace_name} (ID: {app.current_workspace_id})")
    print(f"样本目录: {app.sample_dir}")
    print(f"载入样本数: {len(app.samples)}")
    assert len(app.samples) > 0, "未载入任何样本!"

    # 1. 验证三阶段 Checkbox 初始状态
    print(f"阶段勾选初始状态: {app.stage_checked}, 激活视图: {app.active_stage_view}")
    assert app.stage_checked == [True, True, True]

    # 2. 执行识别解算
    print("--- 执行 run_analyze ---")
    app.run_analyze()
    print(f"检出目标数量: {len(app.targets)}")
    assert len(app.targets) <= 3, "目标数量超过排名前三位限额!"
    for t in app.targets:
        print(f"  Target #{t.id}: 直径 D={t.diam_mm}mm, 长度 L={t.length_mm}mm, 朝向 R={t.yaw_deg}deg, 凸起 H={t.rel_height_mm}mm")
        print(f"    SCARA 位姿: ({t.robot_x}, {t.robot_y}, {t.robot_z}) R={t.robot_r}deg")
        assert 6.0 <= t.diam_mm <= 48.0, f"直径 {t.diam_mm}mm 超出范围!"

    # 3. 验证三阶段可视化图生成
    assert app.stage_vis[0] is not None, "阶段 1 可视化图为空!"
    assert app.stage_vis[1] is not None, "阶段 2 可视化图为空!"
    assert app.stage_vis[2] is not None, "阶段 3 可视化图为空!"
    print(f"阶段 1 图尺寸: {app.stage_vis[0].shape}")
    print(f"阶段 2 图尺寸: {app.stage_vis[1].shape}")
    print(f"阶段 3 图尺寸: {app.stage_vis[2].shape}")

    # 4. 测试点击 Checkbox 切换视图
    print("--- 测试 Checkbox 视图切换 ---")
    app._toggle_stage(0) # 取消勾选阶段 1 -> 级联取消阶段 2, 3
    print(f"取消勾选阶段 1 后的勾选状态: {app.stage_checked}")
    assert app.stage_checked == [False, False, False]

    app._toggle_stage(2) # 勾选阶段 3 -> 级联勾选阶段 1, 2
    print(f"勾选阶段 3 后的勾选状态: {app.stage_checked}")
    assert app.stage_checked == [True, True, True]
    assert app.active_stage_view == 2

    # 单独切换查看阶段 1 视图
    app.active_stage_view = 0
    app._update_stage_display()
    assert app.vis_img is app.stage_vis[0]

    # 渲染当前 GUI 画布并保存
    app.active_stage_view = 2 # 切换回阶段 3
    app._update_stage_display()
    canvas = app.render()
    out_img_path = r"scratch/gui_preview.png"
    cv2.imwrite(out_img_path, canvas)
    print(f"完整 GUI 界面已渲染并保存至: {out_img_path}")

    # 5. 测试视口缩放与平移
    app.viewport.zoom_at(600, 400, True, (300, 60, 800, 600))
    assert app.viewport.zoom_level > 1.0, "缩放失败!"
    app.viewport.reset()
    assert app.viewport.zoom_level == 1.0, "视口复位失败!"

    print("=== 所有自动化测试全部通过 (ALL PASSED) ===")

if __name__ == "__main__":
    test_offline_app()
