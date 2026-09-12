import numpy as np
import cv2
from tools.tag_map_builder import TagMapBuilder

builder = TagMapBuilder()
f_dets, v_frames, _ = builder.load_observations_manifest("data/tag_calibration_images/tag_observations.yaml")

# 运行当前的初值生成流程
all_detected_tags = set()
for tags in f_dets:
    all_detected_tags.update(tags.keys())

base_static_id = 1 if 1 in all_detected_tags else min(all_detected_tags)
tag_poses_init = {base_static_id: np.eye(4, dtype=np.float64)}
camera_poses_init = {}

changed = True
while changed:
    changed = False
    for f_idx, tags in enumerate(f_dets):
        if f_idx not in camera_poses_init:
            for t_id, corners in tags.items():
                if t_id in tag_poses_init:
                    succ, rvec, tvec = builder.solve_single_tag_pnp(corners)
                    if succ:
                        T_c_t = builder.rvec_tvec_to_matrix(rvec, tvec)
                        T_w_c = tag_poses_init[t_id] @ np.linalg.inv(T_c_t)
                        camera_poses_init[f_idx] = T_w_c
                        changed = True
                        break
        if f_idx in camera_poses_init:
            T_w_c = camera_poses_init[f_idx]
            for t_id, corners in tags.items():
                if t_id not in tag_poses_init:
                    succ, rvec, tvec = builder.solve_single_tag_pnp(corners)
                    if succ:
                        T_c_t = builder.rvec_tvec_to_matrix(rvec, tvec)
                        T_w_t = T_w_c @ T_c_t
                        tag_poses_init[t_id] = T_w_t
                        changed = True

print(f"Base tag: {base_static_id}")
print(f"Initialized tags: {len(tag_poses_init)} / {len(all_detected_tags)}")
print(f"Initialized cameras: {len(camera_poses_init)} / {len(f_dets)}")

# 检查每个帧上每个 tag 的重投影误差
tag_errors = {}
frame_errors = {}
for f_idx in camera_poses_init:
    T_w_c = camera_poses_init[f_idx]
    T_c_w = np.linalg.inv(T_w_c)
    tags = f_dets[f_idx]
    f_errs = []
    for t_id, corners in tags.items():
        if t_id in tag_poses_init:
            T_w_t = tag_poses_init[t_id]
            T_c_t = T_c_w @ T_w_t
            rv, tv = builder.matrix_to_rvec_tvec(T_c_t)
            proj, _ = cv2.projectPoints(builder.obj_points, rv, tv, builder.camera_matrix, builder.dist_coeffs)
            proj = proj.reshape((4, 2))
            diff = np.linalg.norm(proj - corners, axis=1) # 4 个角点的误差
            mean_e = np.mean(diff)
            f_errs.append(mean_e)
            tag_errors.setdefault(t_id, []).append((v_frames[f_idx], mean_e))
    frame_errors[v_frames[f_idx]] = np.mean(f_errs) if f_errs else 0.0

print("\n--- Frame Average Errors (Initial) ---")
for f_name, err in sorted(frame_errors.items(), key=lambda x: x[1], reverse=True):
    print(f"  {f_name:15s}: {err:7.2f} px")

print("\n--- Tag Average Errors (Initial) ---")
for t_id in sorted(tag_errors.keys()):
    errs = [e for _, e in tag_errors[t_id]]
    max_frame, max_e = max(tag_errors[t_id], key=lambda x: x[1])
    print(f"  Tag #{t_id:2d}: Mean={np.mean(errs):6.2f} px, Max={max_e:6.2f} px in {max_frame}")
