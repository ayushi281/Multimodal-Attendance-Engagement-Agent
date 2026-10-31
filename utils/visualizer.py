"""
HUD & Frame Visualizer
Renders modern real-time heads-up display (HUD), face bounding boxes,
gaze direction projection rays, EAR / MAR gauges, and engagement metrics overlay.
"""

import math
import cv2
import numpy as np

# Color Palette (BGR)
COLOR_GREEN = (46, 204, 113)
COLOR_AMBER = (52, 152, 219) # Orange/Amber
COLOR_RED = (75, 75, 231)
COLOR_CYAN = (235, 206, 135)
COLOR_PURPLE = (204, 102, 153)
COLOR_DARK_BG = (25, 25, 30)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (140, 140, 140)

def draw_hud(frame, visual_metrics, audio_metrics, participant_info=None, liveness_msg="Liveness: Verified"):
    """
    Overlays a futuristic, high-tech telemetry HUD on top of the webcam frame.
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()
    
    # 1. Telemetry Top Bar (Glassmorphic semi-transparent banner)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    
    # Title / Mode
    cv2.putText(canvas, "MULTIMODAL AGENT HUD", (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 2)
    
    # Participant Tag
    pname = participant_info.get("name", "Unknown Guest") if participant_info else "Unregistered / Guest"
    pid = participant_info.get("id", "N/A") if participant_info else "N/A"
    cv2.putText(canvas, f"ID: {pname} [{pid}]", (290, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    # Liveness indicator badge
    live_color = COLOR_GREEN if "Verified" in liveness_msg else COLOR_RED
    cv2.circle(canvas, (w - 180, 28), 7, live_color, -1)
    cv2.putText(canvas, liveness_msg, (w - 165, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    if not visual_metrics.get("detected"):
        # Absent warning banner
        cv2.rectangle(canvas, (w // 4, h // 2 - 40), (3 * w // 4, h // 2 + 40), (0, 0, 140), -1)
        cv2.putText(canvas, "NO ATTENDEE DETECTED (ABSENT)", (w // 4 + 20, h // 2 + 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_WHITE, 2)
        return canvas

    score = visual_metrics.get("attention_score", 0.0)
    status = visual_metrics.get("status", "Unknown")
    ear = visual_metrics.get("ear", 0.0)
    head_pose = visual_metrics.get("head_pose", {})
    yaw = head_pose.get("yaw", 0.0)
    pitch = head_pose.get("pitch", 0.0)

    # Pick status color
    if score >= 75.0:
        accent_color = COLOR_GREEN
    elif score >= 50.0:
        accent_color = (0, 165, 255) # Orange
    else:
        accent_color = COLOR_RED

    # 2. Draw 3D Gaze Orientation Ray (if rvec and tvec available)
    rvec = visual_metrics.get("rvec")
    tvec = visual_metrics.get("tvec")
    if rvec is not None and tvec is not None:
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Project a 100mm 3D ray out of nose
        nose_end_point3D = np.array([[0.0, 0.0, 450.0]], dtype=np.float64)
        nose_tip_point3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        p2, _ = cv2.projectPoints(nose_end_point3D, rvec, tvec, camera_matrix, dist_coeffs)
        p1, _ = cv2.projectPoints(nose_tip_point3D, rvec, tvec, camera_matrix, dist_coeffs)

        pt1 = (int(p1[0][0][0]), int(p1[0][0][1]))
        pt2 = (int(p2[0][0][0]), int(p2[0][0][1]))

        # Draw Gaze Arrow
        cv2.arrowedLine(canvas, pt1, pt2, (255, 255, 0), 2, tipLength=0.25)
        cv2.circle(canvas, pt1, 4, (0, 255, 255), -1)

    # 3. Bottom-Left Engagement HUD Widget
    card_w, card_h = 240, 140
    card_x, card_y = 15, h - card_h - 15

    sub_overlay = canvas.copy()
    cv2.rectangle(sub_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_DARK_BG, -1)
    cv2.addWeighted(sub_overlay, 0.8, canvas, 0.2, 0, canvas)
    cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), accent_color, 1)

    # Score Gauge Bar
    cv2.putText(canvas, f"ATTENTION: {int(score)}%", (card_x + 12, card_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, accent_color, 1)
    bar_w = 210
    fill_w = int((score / 100.0) * bar_w)
    cv2.rectangle(canvas, (card_x + 12, card_y + 35), (card_x + 12 + bar_w, card_y + 45), (60, 60, 60), -1)
    cv2.rectangle(canvas, (card_x + 12, card_y + 35), (card_x + 12 + fill_w, card_y + 45), accent_color, -1)

    # Status & Metrics
    cv2.putText(canvas, f"State: {status}", (card_x + 12, card_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1)
    cv2.putText(canvas, f"EAR: {ear:.2f} | Blinks: {visual_metrics.get('blink_count', 0)}", (card_x + 12, card_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)
    cv2.putText(canvas, f"Pose: Y:{yaw:+.0f}° P:{pitch:+.0f}°", (card_x + 12, card_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)

    # 4. Bottom-Right Audio Participation Widget
    aud_w, aud_h = 210, 80
    aud_x, aud_y = w - aud_w - 15, h - aud_h - 15

    aud_overlay = canvas.copy()
    cv2.rectangle(aud_overlay, (aud_x, aud_y), (aud_x + aud_w, aud_y + aud_h), COLOR_DARK_BG, -1)
    cv2.addWeighted(aud_overlay, 0.8, canvas, 0.2, 0, canvas)
    cv2.rectangle(canvas, (aud_x, aud_y), (aud_x + aud_w, aud_y + aud_h), (120, 120, 120), 1)

    is_speaking = audio_metrics.get("is_speaking", False)
    speaking_sec = audio_metrics.get("speaking_duration_sec", 0.0)
    vol_pct = audio_metrics.get("volume_pct", 0.0)
    mic_status = "🎙️ SPEAKING" if is_speaking else "🤫 SILENT (Listening)"
    mic_color = COLOR_GREEN if is_speaking else COLOR_GRAY

    cv2.putText(canvas, mic_status, (aud_x + 12, aud_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mic_color, 1)
    # Mic level bar
    aud_fill = int((vol_pct / 100.0) * 180)
    cv2.rectangle(canvas, (aud_x + 12, aud_y + 35), (aud_x + 12 + 180, aud_y + 44), (60, 60, 60), -1)
    cv2.rectangle(canvas, (aud_x + 12, aud_y + 35), (aud_x + 12 + aud_fill, aud_y + 44), (255, 180, 0), -1)
    cv2.putText(canvas, f"Vocal Time: {speaking_sec}s", (aud_x + 12, aud_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    return canvas
