import cv2
from tools.asparagus_offline import AsparagusOfflineApp

app = AsparagusOfflineApp()
m = app._metrics()
print(f"新布局宽度: 左栏={m['list_w']}px (原280), 右栏={m['right_w']}px (原352), 视口有效宽={app.win_mgr.canvas_w - m['list_w'] - m['right_w']}px")

app.run_analyze()
assert len(app.targets) > 0, "Targets is empty!"

views = [
    (0, "gui_5v_1_foreground.png"),
    (1, "gui_5v_2_distance.png"),
    (2, "gui_5v_3_peaks.png"),
    (3, "gui_5v_4_spines.png"),
    (4, "gui_5v_5_poses.png"),
]

for v_idx, fname in views:
    app._select_view_mode(v_idx)
    canvas = app.render()
    out_p = f"scratch/{fname}"
    cv2.imwrite(out_p, canvas)
    print(f"已生成截图: {out_p}")

print("=== 5 阶段视图测试全部成功 ===")
