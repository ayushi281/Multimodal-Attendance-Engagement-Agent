"""
Automated Pipeline Integration Tests
Verifies that all core engines (Engagement, Face Recognition, Audio, Fusion, GenAI, Visualizer)
execute correctly and cohesively.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2

from core.engagement_analyzer import EngagementAnalyzer
from core.face_recognizer import FaceRecognizer
from core.audio_analyzer import AudioAnalyzer
from core.multimodal_fusion import MultimodalFusionEngine
from core.genai_agent import GenAIAgent
from utils.visualizer import draw_hud
from utils.export_helpers import export_attendance_csv, save_session_report

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def create_mock_face_mesh():
    """Generates 468 mock normalized facial landmarks."""
    landmarks = []
    for i in range(468):
        # Center landmarks roughly in the middle of frame
        landmarks.append(MockLandmark(0.5 + 0.1 * np.sin(i), 0.5 + 0.1 * np.cos(i), 0.0))
    # Customize key landmarks
    landmarks[1] = MockLandmark(0.5, 0.5, 0.0)    # Nose
    landmarks[152] = MockLandmark(0.5, 0.7, -0.05) # Chin
    landmarks[33] = MockLandmark(0.4, 0.45, -0.02) # Left eye
    landmarks[263] = MockLandmark(0.6, 0.45, -0.02) # Right eye
    landmarks[61] = MockLandmark(0.45, 0.6, -0.02)  # Mouth left
    landmarks[291] = MockLandmark(0.55, 0.6, -0.02) # Mouth right
    return landmarks

def test_pipeline():
    print("🚀 Starting Multimodal Pipeline Verification...")

    # 1. Test Engagement Analyzer
    print("Testing EngagementAnalyzer...")
    analyzer = EngagementAnalyzer()
    mock_lms = create_mock_face_mesh()
    vis_res = analyzer.analyze_frame_landmarks(mock_lms, 640, 480)
    assert vis_res["detected"] is True, "Face should be detected"
    assert "attention_score" in vis_res, "Attention score missing"
    assert "head_pose" in vis_res, "Head pose missing"
    print(f"✅ EngagementAnalyzer passed! Score: {vis_res['attention_score']}%, Status: {vis_res['status']}")

    # 2. Test Face Recognizer
    print("Testing FaceRecognizer...")
    recognizer = FaceRecognizer(profiles_dir="data/profiles")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reg_ok, reg_msg = recognizer.register_profile(
        name="Test Student",
        participant_id="TEST-001",
        role="Student",
        email="test@ai.edu",
        landmarks=mock_lms,
        frame=dummy_frame,
        frame_w=640,
        frame_h=480
    )
    assert reg_ok is True, f"Registration failed: {reg_msg}"
    matched_prof, sim = recognizer.identify(mock_lms, 640, 480)
    assert matched_prof is not None, "Should identify the newly enrolled profile"
    assert matched_prof["id"] == "TEST-001", "ID mismatch"
    # Clean up test profile
    recognizer.delete_profile("TEST-001")
    print(f"✅ FaceRecognizer passed! Similarity: {round(sim, 3)}")

    # 3. Test Audio Analyzer
    print("Testing AudioAnalyzer...")
    audio = AudioAnalyzer()
    # Test simulated/fallback mode safely
    aud_metrics = audio.get_metrics()
    assert "volume_rms" in aud_metrics
    assert "is_speaking" in aud_metrics
    print(f"✅ AudioAnalyzer passed! Metrics: {aud_metrics}")

    # 4. Test Multimodal Fusion Engine
    print("Testing MultimodalFusionEngine...")
    fusion = MultimodalFusionEngine()
    dummy_person = {"id": "DEMO-1", "name": "Ayush Demo", "role": "Student"}
    f_res = fusion.record_tick(vis_res, aud_metrics, dummy_person)
    assert "fused_mei" in f_res, "Fused MEI score missing"
    attend_df = fusion.get_attendance_df()
    assert len(attend_df) > 0, "Attendance record not saved"
    print(f"✅ MultimodalFusionEngine passed! Fused MEI: {f_res['fused_mei']}%")

    # 5. Test GenAI Cognitive Agent
    print("Testing GenAIAgent...")
    agent = GenAIAgent(provider="local")
    nudge = agent.generate_intervention_nudge("Ayush", "Eyes closed", 35.0)
    assert len(nudge) > 10, "Nudge generation failed"
    print(f"✅ GenAI Intervention Nudge: {nudge}")

    quiz = agent.generate_pop_quiz("Deep Learning")
    assert "question" in quiz and len(quiz["options"]) == 4, "Pop-quiz format invalid"
    print(f"✅ GenAI Pop-Quiz: {quiz['question']}")

    summary_stats = {
        "session_duration_sec": 120,
        "total_participants": 2,
        "avg_mei": 84.5,
        "avg_visual_score": 86.0,
        "total_speaking_sec": 24.0,
        "total_drowsy_incidents": 1,
        "total_distracted_incidents": 2
    }
    report = agent.generate_session_report(summary_stats)
    assert "Multimodal Session Intelligence" in report, "Report generation failed"
    print("✅ GenAI Post-Mortem Report generated successfully!")

    # 6. Test Visualizer & Exporters
    print("Testing Visualizer HUD...")
    hud_img = draw_hud(dummy_frame, vis_res, aud_metrics, dummy_person)
    assert hud_img.shape == dummy_frame.shape, "HUD frame shape mismatch"
    print("✅ Visualizer HUD passed!")

    print("Testing Exporters...")
    csv_file = export_attendance_csv(attend_df)
    assert os.path.exists(csv_file), "CSV export missing"
    rep_file = save_session_report(report)
    assert os.path.exists(rep_file), "Report export missing"
    print(f"✅ Exporters passed! CSV: {csv_file}, Report: {rep_file}")

    print("\n🎉 ALL PIPELINE INTEGRATION TESTS PASSED 100%!")

if __name__ == "__main__":
    test_pipeline()
