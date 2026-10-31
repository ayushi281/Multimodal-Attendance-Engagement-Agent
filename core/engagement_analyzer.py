"""
Engagement Analyzer Module
Calculates multimodal visual engagement metrics using MediaPipe Face Mesh:
- Eye Aspect Ratio (EAR) for blink rate & drowsiness detection
- Head Pose Estimation (Yaw, Pitch, Roll) using 3D-to-2D solvePnP for gaze & distraction tracking
- Mouth Aspect Ratio (MAR) for yawning / speech detection
- Composite Attention Score (0 - 100%)
"""

import math
import numpy as np
import cv2

class EngagementAnalyzer:
    def __init__(self, ear_threshold=0.21, drowsy_frames_thresh=15, pose_yaw_thresh=25.0, pose_pitch_thresh=20.0):
        self.ear_threshold = ear_threshold
        self.drowsy_frames_thresh = drowsy_frames_thresh
        self.pose_yaw_thresh = pose_yaw_thresh
        self.pose_pitch_thresh = pose_pitch_thresh
        
        self.drowsy_counter = 0
        self.total_blinks = 0
        self._eye_closed_prev = False
        
        # 3D generic facial model points for Head Pose (solvePnP)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # MediaPipe landmark indices for model points
        # Nose: 1, Chin: 152, Left eye corner: 33, Right eye corner: 263, Left mouth: 61, Right mouth: 291
        self.pose_landmark_indices = [1, 152, 33, 263, 61, 291]
        
        # Eye landmarks (MediaPipe Face Mesh)
        # Left eye: [362, 385, 387, 263, 373, 380]
        # Right eye: [33, 160, 158, 133, 153, 144]
        self.left_eye_indices = [362, 385, 387, 263, 373, 380]
        self.right_eye_indices = [33, 160, 158, 133, 153, 144]
        
        # Mouth landmarks: 61 (left corner), 291 (right corner), 13 (upper lip), 14 (lower lip)
        self.mouth_indices = [61, 291, 13, 14]

    def _calc_ear(self, eye_pts):
        """Compute Eye Aspect Ratio (EAR) given 6 (x,y) eye coordinates."""
        # Vertical distances
        v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
        v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
        # Horizontal distance
        h = np.linalg.norm(eye_pts[0] - eye_pts[3])
        if h == 0:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def _calc_mar(self, mouth_pts):
        """Compute Mouth Aspect Ratio (MAR) for yawning detection."""
        # mouth_pts: [left, right, upper, lower]
        horizontal = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
        vertical = np.linalg.norm(mouth_pts[2] - mouth_pts[3])
        if horizontal == 0:
            return 0.0
        return float(vertical / horizontal)

    def _estimate_head_pose(self, landmarks, frame_w, frame_h):
        """Estimate Yaw, Pitch, Roll angles using solvePnP."""
        image_points = []
        for idx in self.pose_landmark_indices:
            lm = landmarks[idx]
            image_points.append([lm.x * frame_w, lm.y * frame_h])
        image_points = np.array(image_points, dtype=np.float64)

        # Camera calibration matrix approximation
        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0, None, None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        # Sy = sqrt(R00*R00 + R10*R10)
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0

        # Convert to degrees
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)

        return pitch, yaw, roll, rotation_vector, translation_vector

    def analyze_frame_landmarks(self, face_landmarks, frame_w, frame_h):
        """
        Processes normalized MediaPipe landmarks and returns a structured dictionary
        of engagement and attentiveness metrics.
        """
        if not face_landmarks:
            return {
                "detected": False,
                "attention_score": 0.0,
                "status": "Absent",
                "is_looking_screen": False,
                "is_drowsy": False,
                "is_yawning": False,
                "ear": 0.0,
                "mar": 0.0,
                "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "blink_count": self.total_blinks
            }

        # 1. Calculate EAR for both eyes
        left_eye_pts = np.array([[face_landmarks[i].x * frame_w, face_landmarks[i].y * frame_h] for i in self.left_eye_indices])
        right_eye_pts = np.array([[face_landmarks[i].x * frame_w, face_landmarks[i].y * frame_h] for i in self.right_eye_indices])
        
        left_ear = self._calc_ear(left_eye_pts)
        right_ear = self._calc_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # Blink and drowsiness detection logic
        eye_closed = avg_ear < self.ear_threshold
        if eye_closed:
            self.drowsy_counter += 1
        else:
            if self._eye_closed_prev and self.drowsy_counter < self.drowsy_frames_thresh:
                self.total_blinks += 1
            self.drowsy_counter = 0
        self._eye_closed_prev = eye_closed

        is_drowsy = self.drowsy_counter >= self.drowsy_frames_thresh

        # 2. Calculate MAR for mouth opening / yawning
        mouth_pts = np.array([[face_landmarks[i].x * frame_w, face_landmarks[i].y * frame_h] for i in self.mouth_indices])
        mar = self._calc_mar(mouth_pts)
        is_yawning = mar > 0.65

        # 3. Head Pose Estimation
        pitch, yaw, roll, rvec, tvec = self._estimate_head_pose(face_landmarks, frame_w, frame_h)
        
        # Check if looking at screen
        is_looking_screen = (abs(yaw) <= self.pose_yaw_thresh) and (abs(pitch) <= self.pose_pitch_thresh)

        # 4. Compute Composite Attention Score (0 - 100)
        # Base: 100 points
        score = 100.0

        # Pose penalty: proportional to deviation beyond thresholds
        yaw_dev = max(0.0, abs(yaw) - self.pose_yaw_thresh)
        pitch_dev = max(0.0, abs(pitch) - self.pose_pitch_thresh)
        pose_penalty = min(60.0, (yaw_dev * 1.5) + (pitch_dev * 1.5))
        score -= pose_penalty

        # Drowsiness penalty
        if is_drowsy:
            score -= 50.0
        elif eye_closed:
            score -= 15.0

        # Yawning penalty
        if is_yawning:
            score -= 20.0

        attention_score = float(np.clip(score, 5.0, 100.0))

        # Determine human-readable status
        if is_drowsy:
            status = "Drowsy / Eyes Closed"
        elif not is_looking_screen:
            if yaw > self.pose_yaw_thresh:
                status = "Looking Left"
            elif yaw < -self.pose_yaw_thresh:
                status = "Looking Right"
            elif pitch > self.pose_pitch_thresh:
                status = "Looking Down"
            else:
                status = "Looking Up / Away"
        elif is_yawning:
            status = "Fatigued / Yawning"
        elif attention_score >= 80.0:
            status = "Highly Attentive"
        else:
            status = "Moderately Attentive"

        return {
            "detected": True,
            "attention_score": round(attention_score, 1),
            "status": status,
            "is_looking_screen": is_looking_screen,
            "is_drowsy": is_drowsy,
            "is_yawning": is_yawning,
            "ear": round(avg_ear, 3),
            "mar": round(mar, 3),
            "head_pose": {
                "pitch": round(pitch, 1),
                "yaw": round(yaw, 1),
                "roll": round(roll, 1)
            },
            "blink_count": self.total_blinks,
            "rvec": rvec,
            "tvec": tvec
        }
