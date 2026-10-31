"""
Multimodal Fusion Engine
Fuses Visual Attention (Head pose, EAR, gaze), Acoustic Participation (VAD, speaking ratio),
and Interactive Activity into a unified Multimodal Engagement Index (MEI).
Maintains rolling timeline history, attendance registry, and detects engagement drop events.
"""

import time
from datetime import datetime
import pandas as pd
import numpy as np

class MultimodalFusionEngine:
    def __init__(self, visual_weight=0.70, audio_weight=0.20, interaction_weight=0.10):
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.interaction_weight = interaction_weight
        
        # Temporal log buffer
        self.timeline_records = []
        
        # Attendance tracking map: {participant_id: dict_record}
        self.attendance_records = {}
        
        # Engagement drop detection window
        self.recent_scores = []
        self.last_intervention_time = 0
        self.intervention_cooldown_sec = 25  # minimum seconds between AI nudges

    def register_attendance(self, participant_id, name, role="Student", liveness_verified=True):
        """Marks attendance for a detected participant."""
        now_dt = datetime.now()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        if participant_id not in self.attendance_records:
            self.attendance_records[participant_id] = {
                "id": participant_id,
                "name": name,
                "role": role,
                "check_in_time": now_str,
                "last_seen_time": now_str,
                "first_seen_epoch": time.time(),
                "last_seen_epoch": time.time(),
                "total_seconds_present": 1,
                "engagement_scores": [85.0],
                "drowsy_incidents": 0,
                "liveness_verified": liveness_verified,
                "status": "Checked In"
            }
        else:
            rec = self.attendance_records[participant_id]
            rec["last_seen_time"] = now_str
            rec["last_seen_epoch"] = time.time()
            rec["total_seconds_present"] += 1
            if liveness_verified:
                rec["liveness_verified"] = True

    def record_tick(self, visual_metrics, audio_metrics, participant_info=None, interaction_score=80.0):
        """
        Ingests 1 step/frame of metrics across visual, audio, and interaction streams.
        Returns the fused Multimodal Engagement Index (MEI) and any event flags.
        """
        now = time.time()
        
        # 1. Visual Score (0 - 100)
        vis_score = visual_metrics.get("attention_score", 0.0) if visual_metrics.get("detected") else 0.0
        
        # 2. Audio Contribution
        # Speaking actively provides an engagement boost, but silence while looking attentively is also engaged
        vol_pct = audio_metrics.get("volume_pct", 0.0)
        is_speaking = audio_metrics.get("is_speaking", False)
        if is_speaking:
            aud_score = min(100.0, 75.0 + vol_pct * 0.25)
        else:
            # Attentive listening corresponds to high auditory attention
            aud_score = 70.0

        # 3. Multimodal Engagement Index (MEI)
        fused_mei = (
            (vis_score * self.visual_weight) +
            (aud_score * self.audio_weight) +
            (interaction_score * self.interaction_weight)
        )
        fused_mei = float(np.clip(fused_mei, 0.0, 100.0))
        
        # Participant handling
        pid = participant_info.get("id", "Unknown") if participant_info else "Unknown"
        pname = participant_info.get("name", "Guest / Unenrolled") if participant_info else "Guest / Unenrolled"
        
        if pid != "Unknown" and visual_metrics.get("detected"):
            self.register_attendance(pid, pname)
            if pid in self.attendance_records:
                self.attendance_records[pid]["engagement_scores"].append(fused_mei)
                if visual_metrics.get("is_drowsy"):
                    self.attendance_records[pid]["drowsy_incidents"] += 1

        # Event tagging
        event_tag = "Normal"
        if not visual_metrics.get("detected"):
            event_tag = "Absent"
        elif visual_metrics.get("is_drowsy"):
            event_tag = "Drowsy"
        elif not visual_metrics.get("is_looking_screen"):
            event_tag = "Distracted"
        elif is_speaking:
            event_tag = "Speaking"

        # Record timeline entry
        record = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "epoch": now,
            "participant_id": pid,
            "participant_name": pname,
            "visual_score": round(vis_score, 1),
            "audio_volume": round(vol_pct, 1),
            "fused_mei": round(fused_mei, 1),
            "status": visual_metrics.get("status", "Absent"),
            "event": event_tag,
            "is_speaking": is_speaking,
            "ear": visual_metrics.get("ear", 0.0),
            "yaw": visual_metrics.get("head_pose", {}).get("yaw", 0.0),
            "pitch": visual_metrics.get("head_pose", {}).get("pitch", 0.0)
        }
        self.timeline_records.append(record)

        # Check for engagement drop trigger
        self.recent_scores.append(fused_mei)
        if len(self.recent_scores) > 30:  # approx 3-5 seconds of frames
            self.recent_scores.pop(0)

        trigger_nudge = False
        nudge_reason = ""
        avg_recent = float(np.mean(self.recent_scores)) if self.recent_scores else 100.0

        if (now - self.last_intervention_time) > self.intervention_cooldown_sec:
            if visual_metrics.get("is_drowsy"):
                trigger_nudge = True
                nudge_reason = "Prolonged eye closure / severe drowsiness detected"
                self.last_intervention_time = now
            elif avg_recent < 50.0 and visual_metrics.get("detected"):
                trigger_nudge = True
                nudge_reason = f"Attention dropped to {round(avg_recent)}% (looking away or disengaged)"
                self.last_intervention_time = now
            elif not visual_metrics.get("detected") and len(self.timeline_records) > 40:
                # Check if absent for last 20 frames
                last_few_detected = [r.get("visual_score", 0) > 0 for r in self.timeline_records[-20:]]
                if not any(last_few_detected):
                    trigger_nudge = True
                    nudge_reason = "Participant absent from camera view"
                    self.last_intervention_time = now

        return {
            "fused_mei": round(fused_mei, 1),
            "visual_score": round(vis_score, 1),
            "audio_score": round(aud_score, 1),
            "event_tag": event_tag,
            "trigger_nudge": trigger_nudge,
            "nudge_reason": nudge_reason,
            "record": record
        }

    def get_timeline_df(self):
        """Converts timeline records into a Pandas DataFrame."""
        if not self.timeline_records:
            return pd.DataFrame(columns=["timestamp", "visual_score", "audio_volume", "fused_mei", "status", "event"])
        return pd.DataFrame(self.timeline_records)

    def get_attendance_df(self):
        """Converts registered attendance records into a structured summary DataFrame."""
        rows = []
        for pid, data in self.attendance_records.items():
            scores = data.get("engagement_scores", [0])
            avg_score = float(np.mean(scores)) if scores else 0.0
            rows.append({
                "Participant ID": pid,
                "Name": data["name"],
                "Role": data["role"],
                "Check-In Time": data["check_in_time"],
                "Last Seen": data["last_seen_time"],
                "Present Duration (s)": data["total_seconds_present"],
                "Avg Engagement (MEI)": round(avg_score, 1),
                "Drowsy Alerts": data["drowsy_incidents"],
                "Liveness Verified": "✅ Verified" if data["liveness_verified"] else "⚠️ Unverified",
                "Attendance Status": "Present" if avg_score >= 40.0 else "Needs Attention"
            })
        if not rows:
            return pd.DataFrame(columns=["Participant ID", "Name", "Role", "Check-In Time", "Last Seen", "Present Duration (s)", "Avg Engagement (MEI)", "Drowsy Alerts", "Liveness Verified", "Attendance Status"])
        return pd.DataFrame(rows)

    def clear(self):
        """Resets session timeline and attendance."""
        self.timeline_records = []
        self.attendance_records = {}
        self.recent_scores = []
        self.last_intervention_time = 0
