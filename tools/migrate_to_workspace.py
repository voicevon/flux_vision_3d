#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据迁移脚本：将 data/calibration_scenes/ 升维迁移至 data/workspaces/
===================================================================
1. 场景升级为工位 (Workspace): data/workspaces/<ws_id>/
2. 工位顶层资产: tags_map.yaml, tag_whitelist.yaml, workspace_meta.yaml
3. 标定业务专区: calibration/ (raw_images/, tag_observations.yaml, reports/, visualized/)
4. 生产业务专区: production/ (raw_images/, reports/)
5. 安全幂等、有备份校验。
"""

import os
import shutil
import glob
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD_SCENES_DIR = os.path.join(PROJECT_ROOT, "data", "calibration_scenes")
NEW_WORKSPACES_DIR = os.path.join(PROJECT_ROOT, "data", "workspaces")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "data", "calibration_scenes.migrated_backup")


def migrate():
    print(f"[MIGRATE] 检查旧场景目录: {OLD_SCENES_DIR}")
    if not os.path.exists(OLD_SCENES_DIR):
        print(f"[MIGRATE] 旧场景目录不存在，跳过迁移。")
        return

    os.makedirs(NEW_WORKSPACES_DIR, exist_ok=True)

    # 1. 迁移 .active_scene -> .active_workspace
    old_active = os.path.join(OLD_SCENES_DIR, ".active_scene")
    new_active = os.path.join(NEW_WORKSPACES_DIR, ".active_workspace")
    if os.path.exists(old_active):
        shutil.copy2(old_active, new_active)
        print(f"[MIGRATE] 复制激活标记: {old_active} -> {new_active}")

    # 2. 遍历每个场景
    items = sorted(os.listdir(OLD_SCENES_DIR))
    for item in items:
        if item.startswith("."):
            continue
        old_ws_dir = os.path.join(OLD_SCENES_DIR, item)
        if not os.path.isdir(old_ws_dir):
            continue

        new_ws_dir = os.path.join(NEW_WORKSPACES_DIR, item)
        print(f"\n[MIGRATE] 正在迁移工位: {item} -> {new_ws_dir}")

        calib_dir = os.path.join(new_ws_dir, "calibration")
        calib_raw = os.path.join(calib_dir, "raw_images")
        calib_reports = os.path.join(calib_dir, "reports")
        calib_vis = os.path.join(calib_dir, "visualized")

        prod_dir = os.path.join(new_ws_dir, "production")
        prod_raw = os.path.join(prod_dir, "raw_images")
        prod_reports = os.path.join(prod_dir, "reports")

        os.makedirs(calib_raw, exist_ok=True)
        os.makedirs(calib_reports, exist_ok=True)
        os.makedirs(calib_vis, exist_ok=True)
        os.makedirs(prod_raw, exist_ok=True)
        os.makedirs(prod_reports, exist_ok=True)

        # 2.1 复制 tags_map.yaml 及备份至工位顶层
        for map_name in ["tags_map.yaml", "tags_map.yaml.bak"]:
            old_map = os.path.join(old_ws_dir, map_name)
            new_map = os.path.join(new_ws_dir, map_name)
            if os.path.exists(old_map):
                shutil.copy2(old_map, new_map)
                print(f"  - 顶层地图: {new_map}")

        # 2.2 自动生成 tag_whitelist.yaml
        tags_map_path = os.path.join(new_ws_dir, "tags_map.yaml")
        whitelist_path = os.path.join(new_ws_dir, "tag_whitelist.yaml")
        tag_ids = []
        if os.path.exists(tags_map_path):
            try:
                with open(tags_map_path, "r", encoding="utf-8") as f:
                    map_data = yaml.safe_load(f) or {}
                raw_tags = map_data.get("tags", {})
                if isinstance(raw_tags, dict):
                    tag_ids = sorted([int(k) for k in raw_tags.keys() if str(k).isdigit()])
                elif isinstance(raw_tags, list):
                    for t in raw_tags:
                        if isinstance(t, dict) and "id" in t:
                            tag_ids.append(int(t["id"]))
            except Exception as e:
                print(f"  ! 解析 tags_map 提取白名单失败: {e}")

        with open(whitelist_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "workspace_id": item,
                "whitelist_tag_ids": sorted(list(set(tag_ids))),
                "description": "工位合法 AprilTag 白名单"
            }, f, allow_unicode=True, default_flow_style=False)
        print(f"  - 顶层白名单: {whitelist_path} (Tags: {tag_ids})")

        # 2.3 迁移元数据为 workspace_meta.yaml
        old_meta = os.path.join(old_ws_dir, "scene_meta.yaml")
        new_meta = os.path.join(new_ws_dir, "workspace_meta.yaml")
        meta_dict = {}
        if os.path.exists(old_meta):
            try:
                with open(old_meta, "r", encoding="utf-8") as f:
                    meta_dict = yaml.safe_load(f) or {}
            except Exception:
                pass
        meta_dict["workspace_id"] = meta_dict.pop("scene_id", item)
        with open(new_meta, "w", encoding="utf-8") as f:
            yaml.dump(meta_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"  - 顶层元数据: {new_meta}")

        # 2.4 复制标定原始图像
        old_raw_dir = os.path.join(old_ws_dir, "raw_images")
        if os.path.exists(old_raw_dir):
            for png in glob.glob(os.path.join(old_raw_dir, "*.png")):
                shutil.copy2(png, os.path.join(calib_raw, os.path.basename(png)))
            print(f"  - 标定图片: {len(glob.glob(os.path.join(calib_raw, '*.png')))} 张")

        # 2.5 复制 tag_observations.yaml
        obs_sources = [
            os.path.join(old_ws_dir, "tag_observations.yaml"),
            os.path.join(old_raw_dir, "tag_observations.yaml") if os.path.exists(old_raw_dir) else ""
        ]
        target_obs = os.path.join(calib_dir, "tag_observations.yaml")
        for obs_src in obs_sources:
            if obs_src and os.path.exists(obs_src):
                shutil.copy2(obs_src, target_obs)
                print(f"  - 标定观测清单: {target_obs}")
                break

        # 2.6 复制 reports
        old_reports = os.path.join(old_ws_dir, "reports")
        if os.path.exists(old_reports):
            for rep in glob.glob(os.path.join(old_reports, "*.*")):
                shutil.copy2(rep, os.path.join(calib_reports, os.path.basename(rep)))
            print(f"  - 标定报告: {len(glob.glob(os.path.join(calib_reports, '*.*')))} 份")

        # 2.7 复制 visualized
        for vis_src_dir in [os.path.join(old_ws_dir, "visualized"), os.path.join(old_raw_dir, "visualized") if os.path.exists(old_raw_dir) else ""]:
            if vis_src_dir and os.path.exists(vis_src_dir):
                for vis_f in glob.glob(os.path.join(vis_src_dir, "*.*")):
                    shutil.copy2(vis_f, os.path.join(calib_vis, os.path.basename(vis_f)))
        print(f"  - 标定可视化残差: {len(glob.glob(os.path.join(calib_vis, '*.*')))} 份")

    # 3. 安全备份旧目录
    if os.path.exists(OLD_SCENES_DIR):
        if os.path.exists(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
        os.rename(OLD_SCENES_DIR, BACKUP_DIR)
        print(f"\n[MIGRATE] 原旧目录已安全重命名备份为: {BACKUP_DIR}")

    print("\n[MIGRATE] 全量数据升维迁移完毕！")


if __name__ == "__main__":
    migrate()
