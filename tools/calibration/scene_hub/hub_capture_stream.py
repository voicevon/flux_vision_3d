"""
Scene Hub 原地采图取流控制器 (HubCaptureStream)
==============================================
负责实时检测 AprilTag 16h5 标靶、绘制高精引导线与状态叠加层、处理白闪反馈
"""

import time
import cv2
import numpy as np
from tools.calibration.scene_hub.hub_state import HubState


class HubCaptureStream:
    """原地取流与实时 AprilTag 跟踪辅助器"""

    def __init__(self):
        # 标靶字典 16h5
        self.tag_family = cv2.aruco.DICT_APRILTAG_16h5
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.tag_family)
        
        # 实时检测参数 (轻量级平衡)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 4.0
        params.minOtsuStdDev = 0.5
        params.minMarkerPerimeterRate = 0.01
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    def render_stream_viewport(self, state: HubState, target_w: int, target_h: int) -> np.ndarray:
        """
        拉取最新相机帧，渲染 AprilTag 实时标注与浮动指示，缩放到指定尺寸
        """
        ok, raw_frame = state.camera_streamer.read()
        if not ok or raw_frame is None:
            # 相机无信号提示卡
            canvas = np.full((target_h, target_w, 3), 20, dtype=np.uint8)
            cv2.putText(canvas, "[CAMERA OFFLINE] 等待视频流接入...", (target_w // 4, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return canvas

        disp_frame = raw_frame.copy()

        # 执行实时标靶检测
        gray = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        tag_count = len(ids) if ids is not None else 0

        # 绘制检测框与角点准星
        if tag_count > 0:
            flat_ids = np.ravel(ids)
            for i in range(tag_count):
                c = corners[i][0]
                tag_id = int(flat_ids[i])
                pts = c.astype(np.int32)

                # 绿色多边形边框
                cv2.polylines(disp_frame, [pts], isClosed=True, color=(0, 240, 100), thickness=2, lineType=cv2.LINE_AA)
                # 起始角点微红圆圈
                cv2.circle(disp_frame, tuple(pts[0]), 4, (0, 100, 255), -1, lineType=cv2.LINE_AA)

                # 浮动 ID 徽章
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                label = f"#{tag_id}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(disp_frame, (cx - 4, cy - lh - 4), (cx + lw + 4, cy + 4), (0, 40, 0), -1)
                cv2.rectangle(disp_frame, (cx - 4, cy - lh - 4), (cx + lw + 4, cy + 4), (0, 220, 80), 1)
                cv2.putText(disp_frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 顶部半透明 HUD 状态指示横幅
        sc = state.get_selected_scene()
        scene_id = sc.scene_id if sc else "Unknown"
        hud_h = 42
        hud_overlay = disp_frame[:hud_h, :].copy()
        cv2.rectangle(hud_overlay, (0, 0), (disp_frame.shape[1], hud_h), (12, 16, 20), -1)
        cv2.addWeighted(hud_overlay, 0.82, disp_frame[:hud_h, :], 0.18, 0, disp_frame[:hud_h, :])

        # 状态文字
        stream_mode = "MOCK" if state.camera_streamer.is_mock else "ONLINE"
        fps_text = f"{state.camera_streamer.fps:.1f} FPS"
        cam_tag = f"[CAM: {stream_mode} | {fps_text}]"
        cv2.putText(disp_frame, f"LIVE CAPTURE -> [{scene_id}]", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(disp_frame, f"TAGS DETECTED: {tag_count}   {cam_tag}", (disp_frame.shape[1] - 400, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 100) if tag_count > 0 else (120, 120, 120), 1, cv2.LINE_AA)

        # 底部操作指引浮层
        fh, fw = disp_frame.shape[:2]
        foot_h = 36
        foot_overlay = disp_frame[fh - foot_h:, :].copy()
        cv2.rectangle(foot_overlay, (0, 0), (fw, foot_h), (10, 12, 16), -1)
        cv2.addWeighted(foot_overlay, 0.85, disp_frame[fh - foot_h:, :], 0.15, 0, disp_frame[fh - foot_h:, :])
        cv2.putText(disp_frame, "[Space] 空格抓拍当前帧   |   [ESC / C] 完成采图并返回场景看板   |   [S] 立即启动Studio深度平差",
                    (20, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 240, 220), 1, cv2.LINE_AA)

        # 空格抓拍 50ms 屏幕柔和白闪动效
        now = time.time()
        if state.flash_timer > now:
            alpha = min(0.65, (state.flash_timer - now) / 0.08)
            white_layer = np.full_like(disp_frame, 255)
            cv2.addWeighted(white_layer, alpha, disp_frame, 1.0 - alpha, 0, disp_frame)

        # 缩放到目标视口尺寸
        if disp_frame.shape[1] != target_w or disp_frame.shape[0] != target_h:
            disp_frame = cv2.resize(disp_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return disp_frame
