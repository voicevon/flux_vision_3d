"""
SCARA 机械臂手眼标定工具 (Eye-to-Hand Calibration CLI)
======================================================
适用工况：
  - 相机大倾角固定俯视传送带 (Eye-to-Hand)
  - SCARA 机械臂固定安装于机台基座
  - 目标：求解相机坐标系到 SCARA 基座坐标系的 4x4 齐次变换矩阵 T_cam_to_scara

标定原理：
  采用经典 Kabsch / Horn / Umeyama 算法（基于 SVD 的最小二乘点对刚体配准），
  利用操作员示教的 N 个物理对应点对 (N >= 3, 推荐 4~6 个点)，求解最优旋转 R 与平移 t，
  计算均方根误差 (RMSE)，并一键回写更新 config.yaml。
"""

import os
import sys
import yaml
import numpy as np
from typing import List, Tuple, Optional

# 解决 Windows 控制台中文输出编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def compute_rigid_transform_svd(pts_cam: np.ndarray, pts_robot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    使用 SVD 求解最优刚体变换 R, t (使得 R * pts_cam + t 逼近 pts_robot)
    :param pts_cam: 相机坐标系下的点集 (N x 3)
    :param pts_robot: 机械臂基座坐标系下的点集 (N x 3)
    :return: (R: 3x3 旋转矩阵, t: 3x1 平移向量, rmse: 均方根误差 mm)
    """
    assert pts_cam.shape == pts_robot.shape, "相机点集与机械臂点集维度必须完全一致"
    n = pts_cam.shape[0]
    if n < 3:
        raise ValueError(f"刚体变换求解至少需要 3 个非共线点对，当前仅有 {n} 个点")

    # 1. 计算质心
    centroid_cam = np.mean(pts_cam, axis=0)
    centroid_robot = np.mean(pts_robot, axis=0)

    # 2. 去中心化
    a_centered = pts_cam - centroid_cam
    b_centered = pts_robot - centroid_robot

    # 3. 计算协方差矩阵 H
    h_mat = a_centered.T @ b_centered

    # 4. SVD 分解
    u, s, vt = np.linalg.svd(h_mat)
    r_mat = vt.T @ u.T

    # 5. 处理反射歧义 (确保 det(R) == +1)
    if np.linalg.det(r_mat) < 0:
        vt[-1, :] *= -1
        r_mat = vt.T @ u.T

    # 6. 计算平移向量
    t_vec = centroid_robot - r_mat @ centroid_cam

    # 7. 计算均方根误差 (RMSE)
    transformed_cam = (r_mat @ pts_cam.T).T + t_vec
    errors = np.linalg.norm(transformed_cam - pts_robot, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return r_mat, t_vec, rmse


def save_matrix_to_config(t_matrix: np.ndarray, config_path: str = "config.yaml") -> bool:
    """将解算出的 4x4 齐次变换矩阵回写至 config.yaml"""
    if not os.path.exists(config_path):
        print(f"[ERROR] 找不到配置文件: {config_path}")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "calibration" not in cfg:
            cfg["calibration"] = {}

        # 转换为浮点列表 (保留 5 位小数提高精度)
        mat_list = []
        for row in t_matrix:
            mat_list.append([round(float(val), 5) for val in row])

        cfg["calibration"]["t_cam_to_scara"] = mat_list

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"[OK] 标定矩阵已成功写入配置文件: {config_path}")
        return True
    except Exception as e:
        print(f"[ERROR] 写入 config.yaml 失败: {e}")
        return False


def run_demo_calibration(config_path: str = "config.yaml"):
    """
    运行仿真自检测试 (使用预设典型相机大倾角俯视坐标点对)
    """
    print("\n" + "=" * 70)
    print("      SCARA 手眼标定系统 (Horn SVD 刚体变换仿真自检与精度评测)")
    print("=" * 70)

    # 模拟一组带有真实物理倾角的坐标点对 (相机斜俯视 640mm 工作台)
    # 真实旋转: 绕 X 倾斜 15 度，绕 Y 倾斜 10 度，平移 (180, 320, -500)
    pts_cam = np.array([
        [-120.5, -80.2, 532.0],
        [110.2, -75.4, 545.1],
        [-105.8, 95.6, 528.4],
        [115.3, 88.7, 541.2],
        [2.1, 5.3, 536.0]
    ])

    # 模拟真实机械臂示教点 (加 0.3mm 测量白噪声)
    np.random.seed(42)
    noise = np.random.normal(0, 0.25, pts_cam.shape)
    # 真实预设标定矩阵
    r_ground_truth = np.array([
        [0.9848, 0.0302, 0.1710],
        [0.0000, 0.9848, -0.1736],
        [-0.1736, 0.1710, 0.9698]
    ])
    t_ground_truth = np.array([185.0, 310.0, -480.0])
    pts_robot = (r_ground_truth @ pts_cam.T).T + t_ground_truth + noise

    print(f"参与配准的点对数量: {len(pts_cam)} 点")
    print("-" * 70)
    print(f"{'标定点':<8}{'相机测量 (Xc, Yc, Zc) mm':<32}{'机械臂示教 (Xr, Yr, Zr) mm'}")
    print("-" * 70)
    for i in range(len(pts_cam)):
        c_str = f"({pts_cam[i][0]:.1f}, {pts_cam[i][1]:.1f}, {pts_cam[i][2]:.1f})"
        r_str = f"({pts_robot[i][0]:.1f}, {pts_robot[i][1]:.1f}, {pts_robot[i][2]:.1f})"
        print(f"点 #{i+1:<5}{c_str:<32}{r_str}")
    print("-" * 70)

    r_mat, t_vec, rmse = compute_rigid_transform_svd(pts_cam, pts_robot)

    # 组装 4x4 齐次矩阵
    t_4x4 = np.eye(4)
    t_4x4[:3, :3] = r_mat
    t_4x4[:3, 3] = t_vec

    print(f"\n[SVD 解算成功] 标定均方根重投影误差 (RMSE): {rmse:.3f} mm")
    if rmse < 1.0:
        print("[评价] 精度极高 (RMSE < 1.0mm)，完全达到工业夹持装配标准！")
    elif rmse < 3.0:
        print("[评价] 精度良好 (RMSE < 3.0mm)，满足普通分拣需求。")
    else:
        print("[警告] 误差偏大 (RMSE >= 3.0mm)，建议复查示教点触碰精度。")

    print("\n齐次变换矩阵 T_cam_to_scara (4x4):")
    for row in t_4x4:
        print("  [ " + ", ".join(f"{val:10.5f}" for val in row) + " ]")

    print("\n" + "=" * 70)
    return t_4x4, rmse


def run_interactive_calibration(config_path: str = "config.yaml"):
    """
    工业现场交互式标定向导
    引导操作员逐点录入相机坐标与机械臂读数
    """
    print("\n" + "=" * 72)
    print("      SCARA 机械臂手眼标定向导 (Eye-to-Hand 物理接触配准)")
    print("=" * 72)
    print("操作步骤：")
    print("  1. 在传送带上选定 3~6 个特征点（可用白色马克笔点在皮带上，或放置小标志块）；")
    print("  2. 在视觉工具（如 d435_viewer.py）中查看该点的相机空间坐标 (X_c, Y_c, Z_c) mm；")
    print("  3. 示教 SCARA 机械臂末端精确接触该特征点，读取机械臂当前示教坐标 (X_r, Y_r, Z_r) mm；")
    print("  4. 依次录入各个点对，系统将自动利用 SVD 闭式解算刚体变换并回写 config.yaml。")
    print("-" * 72)

    pts_cam_list = []
    pts_robot_list = []

    while True:
        pt_idx = len(pts_cam_list) + 1
        print(f"\n>>> 录入第 #{pt_idx} 个标定点 (输入 'q' 结束录入并开始解算, 'c' 清空重来):")
        
        c_input = input(f"  请输入点 #{pt_idx} 相机坐标 [Xc, Yc, Zc] (以逗号或空格分隔): ").strip()
        if c_input.lower() == 'q':
            break
        if c_input.lower() == 'c':
            pts_cam_list.clear()
            pts_robot_list.clear()
            print("  [INFO] 已清空所有点。")
            continue

        try:
            c_vals = [float(x) for x in c_input.replace(",", " ").split()]
            if len(c_vals) != 3:
                print("  [!] 输入格式错误，需要恰好 3 个数值 (X, Y, Z)")
                continue
        except ValueError:
            print("  [!] 数值解析失败，请重新输入")
            continue

        r_input = input(f"  请输入点 #{pt_idx} 机械臂坐标 [Xr, Yr, Zr] (以逗号或空格分隔): ").strip()
        try:
            r_vals = [float(x) for x in r_input.replace(",", " ").split()]
            if len(r_vals) != 3:
                print("  [!] 输入格式错误，需要恰好 3 个数值 (X, Y, Z)")
                continue
        except ValueError:
            print("  [!] 数值解析失败，请重新输入")
            continue

        pts_cam_list.append(c_vals)
        pts_robot_list.append(r_vals)
        print(f"  [OK] 已记录点 #{pt_idx}: 相机={c_vals} -> 机械臂={r_vals}")

        if len(pts_cam_list) >= 4:
            ans = input("  当前已录入 4 个及以上点，是否立即开始计算? (y/n/继续输入): ").strip().lower()
            if ans == 'y':
                break

    if len(pts_cam_list) < 3:
        print("\n[!] 标定点不足 3 个，无法计算变换矩阵。标定退出。")
        return

    pts_cam = np.array(pts_cam_list, dtype=float)
    pts_robot = np.array(pts_robot_list, dtype=float)

    try:
        r_mat, t_vec, rmse = compute_rigid_transform_svd(pts_cam, pts_robot)
        t_4x4 = np.eye(4)
        t_4x4[:3, :3] = r_mat
        t_4x4[:3, 3] = t_vec

        print("\n" + "=" * 70)
        print(f"[标定解算成功] 均方根误差 (RMSE): {rmse:.3f} mm")
        print("4x4 齐次变换矩阵 T_cam_to_scara:")
        for row in t_4x4:
            print("  [ " + ", ".join(f"{val:10.5f}" for val in row) + " ]")
        print("=" * 70)

        confirm = input("\n是否将此标定矩阵保存并更新至 config.yaml? (y/n): ").strip().lower()
        if confirm == 'y':
            save_matrix_to_config(t_4x4, config_path)
            print("[SUCCESS] 手眼标定已完成，视觉系统后续将直接输出机械臂世界坐标！\n")
        else:
            print("[INFO] 用户取消保存。")

    except Exception as e:
        print(f"[ERROR] 标定计算失败: {e}")


def main():
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo_calibration(cfg_path)
        return

    print("\n请选择标定操作模式:")
    print("  [1] 工业现场交互式多点触碰标定向导 (推荐实机调试使用)")
    print("  [2] 运行 SVD 算法仿真精度验证与 Demo 测试")
    print("  [Q] 退出")
    choice = input("\n请输入选项 [1/2/Q]: ").strip().upper()

    if choice == "1":
        run_interactive_calibration(cfg_path)
    elif choice == "2":
        run_demo_calibration(cfg_path)
    else:
        print("已退出。")


if __name__ == "__main__":
    main()
