"""
Face Recognizer & Biometric Attendance Module
Uses MediaPipe Face Mesh landmarks to generate normalized facial biometric signatures,
match against enrolled profiles, and verify liveness (anti-spoofing via micro-motions & blinks).
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import cv2

class FaceRecognizer:
    def __init__(self, profiles_dir="data/profiles", similarity_thresh=0.82):
        self.profiles_dir = profiles_dir
        self.similarity_thresh = similarity_thresh
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Selected key facial landmarks across face mesh for robust biometric signature:
        # Contour: 10, 152, 234, 454
        # Eyes: 33, 133, 362, 263, 159, 145, 386, 374
        # Eyebrows: 70, 107, 336, 300
        # Nose: 1, 2, 98, 327, 168
        # Lips: 61, 291, 0, 17, 13, 14
        # Cheeks & Jaw: 58, 288, 172, 397, 132, 361
        self.key_indices = [
            10, 152, 234, 454, 33, 133, 362, 263, 159, 145, 386, 374,
            70, 107, 336, 300, 1, 2, 98, 327, 168, 61, 291, 0, 17, 13, 14,
            58, 288, 172, 397, 132, 361, 197, 195, 5, 4, 45, 275
        ]
        
        # In-memory profile cache
        self.profiles = {}
        self.load_profiles()
        
        # Temporal buffers for liveness tracking per session
        self.recent_poses = []
        self.recent_ears = []

    def extract_signature(self, landmarks, frame_w, frame_h):
        """
        Generates an affine-invariant, normalized facial geometric signature vector.
        Translates origin to nose tip (index 1) and scales by inter-ocular distance.
        """
        if not landmarks:
            return None

        # Reference points
        nose = np.array([landmarks[1].x * frame_w, landmarks[1].y * frame_h, landmarks[1].z * frame_w])
        left_eye = np.array([landmarks[33].x * frame_w, landmarks[33].y * frame_h, landmarks[33].z * frame_w])
        right_eye = np.array([landmarks[263].x * frame_w, landmarks[263].y * frame_h, landmarks[263].z * frame_w])
        
        iod = np.linalg.norm(left_eye - right_eye)
        if iod < 1e-4:
            iod = 1.0

        coords = []
        for idx in self.key_indices:
            lm = landmarks[idx]
            pt = np.array([lm.x * frame_w, lm.y * frame_h, lm.z * frame_w])
            # Center relative to nose and normalize by inter-ocular distance
            norm_pt = (pt - nose) / iod
            coords.extend(norm_pt.tolist())

        vec = np.array(coords, dtype=np.float32)
        # Unit normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.tolist()

    def register_profile(self, name, participant_id, role, email, landmarks, frame, frame_w, frame_h):
        """
        Saves a new participant profile with their facial signature and avatar snapshot.
        """
        signature = self.extract_signature(landmarks, frame_w, frame_h)
        if signature is None:
            return False, "Could not extract facial landmarks."

        # Crop face for avatar
        h, w = frame.shape[:2]
        xs = [int(landmarks[idx].x * w) for idx in self.key_indices]
        ys = [int(landmarks[idx].y * h) for idx in self.key_indices]
        min_x, max_x = max(0, min(xs) - 40), min(w, max(xs) + 40)
        min_y, max_y = max(0, min(ys) - 60), min(h, max(ys) + 40)
        
        avatar_crop = frame[min_y:max_y, min_x:max_x] if max_x > min_x and max_y > min_y else frame
        
        clean_id = participant_id.strip().replace(" ", "_")
        avatar_filename = f"{clean_id}_avatar.jpg"
        avatar_path = os.path.join(self.profiles_dir, avatar_filename)
        cv2.imwrite(avatar_path, avatar_crop)

        profile_data = {
            "id": clean_id,
            "name": name.strip(),
            "role": role.strip(),
            "email": email.strip(),
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avatar_path": avatar_path,
            "signature": signature
        }

        profile_json_path = os.path.join(self.profiles_dir, f"{clean_id}.json")
        with open(profile_json_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        self.profiles[clean_id] = profile_data
        return True, f"Successfully enrolled {name} (ID: {clean_id})"

    def load_profiles(self):
        """Loads all saved JSON profiles from profiles_dir."""
        self.profiles = {}
        if not os.path.exists(self.profiles_dir):
            return
        for fname in os.listdir(self.profiles_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.profiles_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.profiles[data["id"]] = data
                except Exception as e:
                    print(f"Error loading profile {fname}: {e}")

    def delete_profile(self, participant_id):
        """Removes a profile and associated avatar."""
        if participant_id in self.profiles:
            avatar = self.profiles[participant_id].get("avatar_path")
            if avatar and os.path.exists(avatar):
                try:
                    os.remove(avatar)
                except Exception:
                    pass
            json_path = os.path.join(self.profiles_dir, f"{participant_id}.json")
            if os.path.exists(json_path):
                os.remove(json_path)
            del self.profiles[participant_id]
            return True
        return False

    def identify(self, landmarks, frame_w, frame_h):
        """
        Compares detected face with enrolled profiles using cosine similarity.
        Returns (best_match_profile, similarity_score).
        """
        if not self.profiles or not landmarks:
            return None, 0.0

        current_sig = np.array(self.extract_signature(landmarks, frame_w, frame_h), dtype=np.float32)
        
        best_match = None
        best_sim = -1.0

        for pid, prof in self.profiles.items():
            enrolled_sig = np.array(prof["signature"], dtype=np.float32)
            # Cosine similarity
            sim = np.dot(current_sig, enrolled_sig)
            if sim > best_sim:
                best_sim = float(sim)
                best_match = prof

        if best_sim >= self.similarity_thresh:
            return best_match, best_sim
        else:
            return None, best_sim

    def verify_liveness(self, current_ear, current_yaw, current_pitch, window_size=20):
        """
        Anti-spoofing liveness check.
        Requires subtle micro-movements in pose or EAR changes (natural blinks)
        over the sliding window. A printed photo or still image shows near-zero variance.
        """
        self.recent_ears.append(current_ear)
        self.recent_poses.append((current_yaw, current_pitch))
        
        if len(self.recent_ears) > window_size:
            self.recent_ears.pop(0)
        if len(self.recent_poses) > window_size:
            self.recent_poses.pop(0)

        if len(self.recent_ears) < 8:
            return True, "Calibrating liveness..."

        ear_std = float(np.std(self.recent_ears))
        pose_arr = np.array(self.recent_poses)
        pose_std = float(np.std(pose_arr))

        # If completely static (photo on screen / paper)
        is_live = (ear_std > 0.006) or (pose_std > 0.35)
        status_msg = "Liveness: Verified Real Person" if is_live else "Liveness: Suspicious Static Feed"
        return is_live, status_msg
