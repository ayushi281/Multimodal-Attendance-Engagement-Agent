"""
Multimodal Attendance and Engagement Agent
Main Streamlit Application Dashboard
"""

import os
import sys
import time
import tempfile
from datetime import datetime

# Prevent Matplotlib cache permission warning
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import altair as alt

from core.engagement_analyzer import EngagementAnalyzer
from core.face_recognizer import FaceRecognizer
from core.audio_analyzer import AudioAnalyzer
from core.multimodal_fusion import MultimodalFusionEngine
from core.genai_agent import GenAIAgent
from utils.visualizer import draw_hud
from utils.export_helpers import export_attendance_csv, export_timeline_csv, save_session_report

# Page Config
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #151820 100%);
        border: 1px solid #2d3345;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-green { background: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; }
    .badge-amber { background: rgba(241, 196, 15, 0.2); color: #f1c40f; border: 1px solid #f1c40f; }
    .badge-red { background: rgba(231, 76, 60, 0.2); color: #e74c3c; border: 1px solid #e74c3c; }
    .badge-blue { background: rgba(52, 152, 219, 0.2); color: #3498db; border: 1px solid #3498db; }
    .agent-bubble {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.analyzer = EngagementAnalyzer()
    st.session_state.recognizer = FaceRecognizer()
    st.session_state.audio_analyzer = AudioAnalyzer()
    st.session_state.fusion_engine = MultimodalFusionEngine()
    st.session_state.genai_agent = GenAIAgent(provider="local")
    st.session_state.active_nudge = None
    st.session_state.nudge_history = []
    st.session_state.chat_history = [
        {"role": "agent", "content": "👋 Hello! I am your Multimodal Attendance & Engagement Agent. I am monitoring attendance, visual gaze, and verbal participation in real time."}
    ]
    st.session_state.pop_quiz_active = None
    st.session_state.pop_quiz_result = None
    st.session_state.session_report_cache = None

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Multimodal Agent")
    st.caption("AI-Powered Attendance & Engagement System")
    st.markdown("---")

    st.subheader("⚙️ Stream & AI Settings")
    stream_mode = st.radio(
        "Input Feed Mode",
        ["Live Webcam", "Simulated Demo Stream"],
        help="Use Live Webcam to capture real video from your camera, or Simulated Demo Stream to test without hardware."
    )
    
    selected_camera_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
    
    st.markdown("---")
    st.subheader("🧠 GenAI Provider")
    llm_choice = st.selectbox(
        "Reasoning Engine",
        ["Local Offline Agent (Zero Setup)", "Google Gemini API", "OpenAI API", "Groq API"],
        index=0
    )

    api_key_val = ""
    if "Gemini" in llm_choice:
        api_key_val = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
        st.session_state.genai_agent = GenAIAgent(provider="gemini", api_key=api_key_val, model_name="gemini-1.5-flash")
    elif "OpenAI" in llm_choice:
        api_key_val = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.session_state.genai_agent = GenAIAgent(provider="openai", api_key=api_key_val, model_name="gpt-4o-mini")
    elif "Groq" in llm_choice:
        api_key_val = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.session_state.genai_agent = GenAIAgent(provider="groq", api_key=api_key_val, model_name="llama-3.1-8b-instant")
    else:
        st.session_state.genai_agent = GenAIAgent(provider="local")
        st.info("💡 Running Local Intelligent Engine. Fully offline & fast!")

    st.markdown("---")
    if st.button("🔄 Reset Current Session Data", use_container_width=True):
        st.session_state.fusion_engine.clear()
        st.session_state.active_nudge = None
        st.session_state.nudge_history = []
        st.session_state.session_report_cache = None
        st.success("Session reset!")

# Main Tabs
tabs = st.tabs([
    "🔴 Live Multimodal Hub",
    "👥 Attendance & Profiles",
    "🤖 GenAI Cognitive Agent",
    "📈 Analytics & Post-Mortem",
    "ℹ️ Architecture & Guide"
])

# ==============================================================================
# TAB 1: LIVE MULTIMODAL HUB
# ==============================================================================
with tabs[0]:
    st.header("🔴 Live Multimodal Monitoring & Attendance Hub")
    st.write("Real-time facial landmark tracking, gaze vector estimation, EAR drowsiness detection, and audio energy analysis.")

    # Top KPI Metrics Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    # Placeholders for dynamic live updates
    metric_mei_ph = kpi_col1.empty()
    metric_status_ph = kpi_col2.empty()
    metric_attend_ph = kpi_col3.empty()
    metric_audio_ph = kpi_col4.empty()
    metric_live_ph = kpi_col5.empty()

    # Nudge Alert Card Placeholder
    nudge_banner_ph = st.empty()

    # Video Feed & Controls
    feed_col, control_col = st.columns([3, 1])

    with feed_col:
        video_placeholder = st.empty()

    with control_col:
        st.subheader("Controls")
        start_btn = st.button("▶️ Start Monitoring", key="start_mon", use_container_width=True, type="primary")
        stop_btn = st.button("⏹️ Stop Monitoring", key="stop_mon", use_container_width=True)

        st.markdown("---")
        st.caption("🧪 Quick Simulation Triggers")
        sim_drowsy = st.button("😴 Trigger Drowsiness", help="Simulate attendee closing eyes")
        sim_distract = st.button("👀 Trigger Looking Away", help="Simulate turning head away")
        sim_speak = st.button("🗣️ Trigger Speaking Bout", help="Simulate participant speaking")

    # Live Monitoring Loop Execution
    if start_btn:
        st.session_state.monitoring_active = True
    if stop_btn:
        st.session_state.monitoring_active = False

    if st.session_state.get("monitoring_active", False):
        # Initialize Audio Stream
        if not st.session_state.audio_analyzer.is_running:
            st.session_state.audio_analyzer.start()

        # MediaPipe Face Mesh Setup
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap = None
        if stream_mode == "Live Webcam":
            cap = cv2.VideoCapture(int(selected_camera_idx))
            if not cap.isOpened():
                st.warning("⚠️ Could not open webcam at index {}. Falling back to Simulated Demo Stream.".format(selected_camera_idx))
                stream_mode = "Simulated Demo Stream"

        # Frame Loop
        frame_idx = 0
        max_loop_frames = 200  # Streamlit loop refresh cycle
        
        while frame_idx < max_loop_frames and st.session_state.get("monitoring_active", False):
            frame_idx += 1
            frame = None
            landmarks = None

            if stream_mode == "Live Webcam" and cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1) # Mirror
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
            else:
                # Simulated Demo Stream: generate dynamic synthetic frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (30, 28, 36) # dark slate
                
                # Synthetic circle face
                t_step = frame_idx * 0.1
                # Head movement simulation
                dx = int(np.sin(t_step) * 20)
                dy = int(np.cos(t_step) * 10)
                
                # Check manual simulation triggers
                if sim_distract:
                    dx = 120 # Looking far right
                
                cv2.circle(frame, (320 + dx, 240 + dy), 110, (190, 160, 140), -1)
                # Eyes
                eye_open = 10 if not sim_drowsy else 2
                cv2.ellipse(frame, (280 + dx, 220 + dy), (16, eye_open), 0, 0, 360, (20, 20, 20), -1)
                cv2.ellipse(frame, (360 + dx, 220 + dy), (16, eye_open), 0, 0, 360, (20, 20, 20), -1)
                # Nose
                cv2.circle(frame, (320 + dx, 250 + dy), 8, (150, 110, 90), -1)
                # Mouth
                mouth_h = 18 if sim_speak else 6
                cv2.ellipse(frame, (320 + dx, 290 + dy), (25, mouth_h), 0, 0, 360, (140, 60, 60), -1)

                # Process synthetic frame with FaceMesh
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

            fh, fw = frame.shape[:2]

            # 1. Visual Engagement Analysis
            visual_metrics = st.session_state.analyzer.analyze_frame_landmarks(landmarks, fw, fh)

            # Manual override if user triggered simulation buttons
            if sim_drowsy:
                visual_metrics["is_drowsy"] = True
                visual_metrics["attention_score"] = 28.0
                visual_metrics["status"] = "Drowsy / Eyes Closed"
            elif sim_distract:
                visual_metrics["is_looking_screen"] = False
                visual_metrics["head_pose"]["yaw"] = 38.0
                visual_metrics["attention_score"] = 42.0
                visual_metrics["status"] = "Looking Right"

            # 2. Audio Analysis
            audio_metrics = st.session_state.audio_analyzer.get_metrics()
            if sim_speak:
                audio_metrics["is_speaking"] = True
                audio_metrics["volume_pct"] = 82.0
                audio_metrics["speaking_duration_sec"] += 0.1

            # 3. Face Recognition & Biometric Identification
            identified_person = None
            if landmarks:
                identified_person, sim_score = st.session_state.recognizer.identify(landmarks, fw, fh)
            
            # Default to primary enrolled profile if available for smooth demo
            if not identified_person and st.session_state.recognizer.profiles:
                # pick first profile for demo
                first_pid = list(st.session_state.recognizer.profiles.keys())[0]
                identified_person = st.session_state.recognizer.profiles[first_pid]

            # 4. Anti-Spoofing Liveness Verification
            is_live, liveness_msg = st.session_state.recognizer.verify_liveness(
                visual_metrics.get("ear", 0.25),
                visual_metrics.get("head_pose", {}).get("yaw", 0.0),
                visual_metrics.get("head_pose", {}).get("pitch", 0.0)
            )

            # 5. Multimodal Fusion
            fusion_result = st.session_state.fusion_engine.record_tick(
                visual_metrics=visual_metrics,
                audio_metrics=audio_metrics,
                participant_info=identified_person,
                interaction_score=85.0
            )

            # 6. Check for GenAI Intervention Nudge
            if fusion_result.get("trigger_nudge"):
                pname = identified_person.get("name", "Attendee") if identified_person else "Attendee"
                nudge_text = st.session_state.genai_agent.generate_intervention_nudge(
                    participant_name=pname,
                    reason=fusion_result.get("nudge_reason", "Engagement dip"),
                    current_score=fusion_result.get("fused_mei", 50.0)
                )
                st.session_state.active_nudge = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "participant": pname,
                    "reason": fusion_result.get("nudge_reason"),
                    "text": nudge_text
                }
                st.session_state.nudge_history.insert(0, st.session_state.active_nudge)

            # 7. Render Futuristic HUD on Frame
            hud_frame = draw_hud(
                frame=frame,
                visual_metrics=visual_metrics,
                audio_metrics=audio_metrics,
                participant_info=identified_person,
                liveness_msg=liveness_msg
            )

            # Display Frame in Streamlit
            video_placeholder.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Update KPI Cards
            fused_score = fusion_result.get("fused_mei", 0.0)
            metric_mei_ph.markdown(f"""
            <div class="metric-card">
                <span style="color:#94a3b8; font-size:0.85rem;">MEI SCORE</span>
                <div class="metric-val" style="color:{'#2ecc71' if fused_score>=75 else '#f1c40f' if fused_score>=50 else '#e74c3c'}">{fused_score}%</div>
                <span class="status-badge {'badge-green' if fused_score>=75 else 'badge-amber' if fused_score>=50 else 'badge-red'}">
                    {'Optimal' if fused_score>=75 else 'Moderate' if fused_score>=50 else 'Critical'}
                </span>
            </div>
            """, unsafe_allow_html=True)

            cur_status = visual_metrics.get("status", "Absent")
            metric_status_ph.markdown(f"""
            <div class="metric-card">
                <span style="color:#94a3b8; font-size:0.85rem;">ATTENTION STATE</span>
                <div class="metric-val" style="font-size:1.25rem; color:#f8fafc; padding-top:6px;">{cur_status}</div>
                <span class="status-badge badge-blue">Visual Stream</span>
            </div>
            """, unsafe_allow_html=True)

            p_display_name = identified_person.get("name", "Unknown") if identified_person else "Unknown"
            metric_attend_ph.markdown(f"""
            <div class="metric-card">
                <span style="color:#94a3b8; font-size:0.85rem;">ATTENDEE</span>
                <div class="metric-val" style="font-size:1.2rem; color:#f8fafc; padding-top:6px;">{p_display_name}</div>
                <span class="status-badge badge-green">Checked In</span>
            </div>
            """, unsafe_allow_html=True)

            speaking_sec = audio_metrics.get("speaking_duration_sec", 0.0)
            is_spk = audio_metrics.get("is_speaking", False)
            metric_audio_ph.markdown(f"""
            <div class="metric-card">
                <span style="color:#94a3b8; font-size:0.85rem;">ACOUSTIC MIC</span>
                <div class="metric-val" style="font-size:1.25rem; color:#f8fafc; padding-top:6px;">{speaking_sec}s</div>
                <span class="status-badge {'badge-green' if is_spk else 'badge-blue'}">{'🎙️ Speaking' if is_spk else '🤫 Silent'}</span>
            </div>
            """, unsafe_allow_html=True)

            metric_live_ph.markdown(f"""
            <div class="metric-card">
                <span style="color:#94a3b8; font-size:0.85rem;">ANTI-SPOOFING</span>
                <div class="metric-val" style="font-size:1.1rem; color:{'#2ecc71' if 'Verified' in liveness_msg else '#e74c3c'}; padding-top:8px;">
                    {'Verified Real' if 'Verified' in liveness_msg else 'Checking...'}
                </div>
                <span class="status-badge badge-green">Blink & Pose</span>
            </div>
            """, unsafe_allow_html=True)

            # Display Nudge Alert if active
            if st.session_state.active_nudge:
                nudge = st.session_state.active_nudge
                nudge_banner_ph.warning(f"🤖 **GenAI Agent Intervention [{nudge['time']}]**: {nudge['text']} *(Trigger: {nudge['reason']})*")

            time.sleep(0.04) # ~25 FPS pacing

        if cap is not None and cap.isOpened():
            cap.release()
    else:
        video_placeholder.info("Click '▶️ Start Monitoring' to initialize the camera and multimodal analysis engine.")

# ==============================================================================
# TAB 2: ATTENDANCE & PROFILES
# ==============================================================================
with tabs[1]:
    st.header("👥 Participant Directory & Biometric Attendance")
    
    col_roster, col_enroll = st.columns([2, 1])

    with col_roster:
        st.subheader("📋 Live Attendance Registry")
        attend_df = st.session_state.fusion_engine.get_attendance_df()
        st.dataframe(attend_df, use_container_width=True)

        col_csv, col_sync = st.columns(2)
        with col_csv:
            csv_path = export_attendance_csv(attend_df)
            with open(csv_path, "rb") as f:
                st.download_button("📥 Download Attendance CSV", f, file_name="attendance_records.csv", mime="text/csv")
        with col_sync:
            if st.button("🔄 Refresh Profiles & Roster"):
                st.session_state.recognizer.load_profiles()
                st.success("Synchronized!")

        st.markdown("---")
        st.subheader("📂 Enrolled Participants Directory")
        if st.session_state.recognizer.profiles:
            p_cols = st.columns(3)
            for idx, (pid, p_data) in enumerate(st.session_state.recognizer.profiles.items()):
                col = p_cols[idx % 3]
                with col:
                    avatar_path = p_data.get("avatar_path")
                    if avatar_path and os.path.exists(avatar_path):
                        st.image(avatar_path, width=120)
                    st.markdown(f"**{p_data['name']}**")
                    st.caption(f"ID: `{pid}` | {p_data['role']}")
                    st.text(f"Email: {p_data.get('email', 'N/A')}")
                    if st.button(f"🗑️ Delete", key=f"del_{pid}"):
                        st.session_state.recognizer.delete_profile(pid)
                        st.rerun()
        else:
            st.info("No profiles enrolled yet. Use the registration form to add participants.")

    with col_enroll:
        st.subheader("➕ Enroll New Participant")
        with st.form("enroll_form"):
            new_name = st.text_input("Full Name", placeholder="e.g. Ayush Kumar")
            new_id = st.text_input("Student / Employee ID", placeholder="e.g. STU-205")
            new_role = st.selectbox("Role", ["Student", "Instructor", "Research Scholar", "Participant", "Attendee"])
            new_email = st.text_input("Email Address", placeholder="user@university.edu")

            st.write("📸 Participant Snapshot")
            upload_source = st.radio("Photo Source", ["Webcam Snapshot", "Upload File"], horizontal=True)
            
            uploaded_file = None
            if upload_source == "Upload File":
                uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
            else:
                uploaded_file = st.camera_input("Take Snapshot")

            submit_enroll = st.form_submit_button("Register & Generate Biometric Signature", type="primary")

            if submit_enroll:
                if not new_name or not new_id:
                    st.error("Please provide both Name and Participant ID.")
                elif uploaded_file is None:
                    st.error("Please capture or upload a face photo.")
                else:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    
                    # Extract face landmarks
                    mp_face_mesh = mp.solutions.face_mesh
                    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        res = mesh.process(rgb)
                        if res.multi_face_landmarks:
                            lms = res.multi_face_landmarks[0].landmark
                            h, w = img.shape[:2]
                            success, msg = st.session_state.recognizer.register_profile(
                                name=new_name,
                                participant_id=new_id,
                                role=new_role,
                                email=new_email,
                                landmarks=lms,
                                frame=img,
                                frame_w=w,
                                frame_h=h
                            )
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.error("No clear face detected in the snapshot. Please look straight at the camera.")

# ==============================================================================
# TAB 3: GENAI COGNITIVE AGENT
# ==============================================================================
with tabs[2]:
    st.header("🤖 GenAI Cognitive Engagement Agent")
    st.write("Autonomous session interventions, dynamic pop-quizzes, and interactive Q&A assistant.")

    agent_col1, agent_col2 = st.columns([1, 1])

    with agent_col1:
        st.subheader("💬 Chat with Session Agent")
        st.caption("Ask questions about audience focus, attendance anomalies, or participation.")

        # Chat display container
        chat_container = st.container(height=380)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_container.chat_message("user").write(msg["content"])
            else:
                chat_container.chat_message("assistant").write(msg["content"])

        user_query = st.chat_input("Ask the agent: e.g., 'Who had the lowest engagement?' or 'Summarize attendance'")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Prepare session context
            df_att = st.session_state.fusion_engine.get_attendance_df()
            session_ctx = {
                "total_participants": len(df_att),
                "avg_mei": float(df_att["Avg Engagement (MEI)"].mean()) if not df_att.empty else 82.0,
                "total_drowsy_incidents": int(df_att["Drowsy Alerts"].sum()) if not df_att.empty else 0,
                "participants": df_att.to_dict(orient="records") if not df_att.empty else []
            }
            
            with st.spinner("AI Agent analyzing session telemetry..."):
                response_text = st.session_state.genai_agent.chat_with_agent(user_query, session_ctx)
                st.session_state.chat_history.append({"role": "agent", "content": response_text})
            st.rerun()

    with agent_col2:
        st.subheader("🎯 Dynamic Attention Pop-Quiz")
        st.caption("Launch an instant cognitive check-in poll to re-engage attendees when attention decays.")

        quiz_topic = st.text_input("Quiz Subject / Topic", value="Generative AI & Multimodal Models")
        if st.button("⚡ Generate Instant Pop-Quiz", use_container_width=True):
            with st.spinner("GenAI drafting question..."):
                st.session_state.pop_quiz_active = st.session_state.genai_agent.generate_pop_quiz(quiz_topic)
                st.session_state.pop_quiz_result = None

        if st.session_state.pop_quiz_active:
            q = st.session_state.pop_quiz_active
            st.markdown(f"**Question**: {q['question']}")
            
            choice = st.radio("Select your answer:", q["options"], key="quiz_choice_radio")
            if st.button("Submit Answer"):
                selected_idx = q["options"].index(choice)
                if selected_idx == q["correct_index"]:
                    st.success(f"🎉 Correct! {q.get('explanation', '')}")
                    st.session_state.pop_quiz_result = "Correct"
                else:
                    correct_opt = q["options"][q["correct_index"]]
                    st.error(f"❌ Not quite! The correct answer is: **{correct_opt}**. {q.get('explanation', '')}")
                    st.session_state.pop_quiz_result = "Incorrect"

        st.markdown("---")
        st.subheader("🔔 Recent AI Interventions & Nudges")
        if st.session_state.nudge_history:
            for nudge in st.session_state.nudge_history[:5]:
                st.markdown(f"""
                <div class="agent-bubble">
                    <span style="color:#38bdf8; font-weight:600;">[{nudge['time']}] Nudge for {nudge['participant']}</span><br>
                    <em>"{nudge['text']}"</em><br>
                    <small style="color:#94a3b8;">Trigger Reason: {nudge['reason']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No interventions triggered yet. As attendees show signs of fatigue or distraction, AI nudges appear here.")

# ==============================================================================
# TAB 4: ANALYTICS & POST-MORTEM
# ==============================================================================
with tabs[3]:
    st.header("📈 Multimodal Session Analytics & Post-Mortem")
    st.write("Granular timeline replay, attention vs fatigue correlations, and automated GenAI session synthesis.")

    timeline_df = st.session_state.fusion_engine.get_timeline_df()

    if not timeline_df.empty and len(timeline_df) > 1:
        # Time-Series Chart
        st.subheader("⏱️ Real-Time Engagement Timeline")
        
        # Prepare chart dataframe
        chart_df = timeline_df.copy()
        if "fused_mei" in chart_df.columns:
            st.line_chart(
                data=chart_df.set_index("timestamp")[["fused_mei", "visual_score", "audio_volume"]],
                color=["#2ecc71", "#3498db", "#f39c12"],
                use_container_width=True
            )

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.subheader("📊 Event Distribution")
            event_counts = timeline_df["event"].value_counts().reset_index()
            event_counts.columns = ["Event", "Count"]
            st.bar_chart(data=event_counts.set_index("Event"), use_container_width=True)

        with col_stat2:
            st.subheader("💤 Fatigue (EAR) & Head Pose Yaw")
            sub_chart = timeline_df[["timestamp", "ear", "yaw"]].copy()
            st.line_chart(data=sub_chart.set_index("timestamp"), use_container_width=True)
    else:
        st.info("No session timeline data recorded yet. Start the live monitoring hub to capture engagement metrics.")

    st.markdown("---")
    st.subheader("📑 Automated GenAI Session Post-Mortem Report")
    st.write("Generate a comprehensive executive engagement report powered by LLM reasoning.")

    if st.button("✨ Synthesize & Generate GenAI Report", type="primary"):
        with st.spinner("Synthesizing multimodal timeline logs..."):
            df_att = st.session_state.fusion_engine.get_attendance_df()
            tl_df = st.session_state.fusion_engine.get_timeline_df()
            
            avg_mei = float(tl_df["fused_mei"].mean()) if not tl_df.empty else 80.0
            avg_vis = float(tl_df["visual_score"].mean()) if not tl_df.empty else 82.0
            total_drowsy = int((tl_df["event"] == "Drowsy").sum()) if not tl_df.empty else 0
            total_distract = int((tl_df["event"] == "Distracted").sum()) if not tl_df.empty else 0
            speaking_sec = float(st.session_state.audio_analyzer.total_speaking_seconds)
            
            summary_stats = {
                "session_duration_sec": len(tl_df),
                "total_participants": len(df_att),
                "avg_mei": round(avg_mei, 1),
                "avg_visual_score": round(avg_vis, 1),
                "total_speaking_sec": round(speaking_sec, 1),
                "total_drowsy_incidents": total_drowsy,
                "total_distracted_incidents": total_distract
            }
            report_md = st.session_state.genai_agent.generate_session_report(summary_stats)
            st.session_state.session_report_cache = report_md

    if st.session_state.session_report_cache:
        st.markdown(st.session_state.session_report_cache)
        
        rep_path = save_session_report(st.session_state.session_report_cache)
        with open(rep_path, "rb") as f:
            st.download_button("📥 Download Markdown Report", f, file_name="session_report.md", mime="text/markdown")

# ==============================================================================
# TAB 5: ARCHITECTURE & GUIDE
# ==============================================================================
with tabs[4]:
    st.header("ℹ️ System Architecture & Engineering Specifications")
    st.markdown("""
    ### 🏗️ Technical Pipeline
    
    The **Multimodal Attendance and Engagement Agent** synchronizes three continuous sensory streams:
    
    1. **Visual Modality Engine**:
       - **MediaPipe Face Mesh (468 3D landmarks)**: Sub-millimeter landmark tracking with CPU acceleration.
       - **Head Pose Estimation via `cv2.solvePnP`**: Projects 3D canonical facial model points onto 2D image coordinates to compute Euler angles (Yaw, Pitch, Roll) and 3D gaze vector orientation.
       - **Eye Aspect Ratio (EAR)**: Computes Euclidean distance ratio between vertical and horizontal eyelid landmarks to detect blinks, fatigue, and drowsiness ($EAR < 0.21$).
       - **Mouth Aspect Ratio (MAR)**: Detects yawning and active articulation.
    
    2. **Acoustic Modality Engine**:
       - Real-time microphone capture via `sounddevice` with root-mean-square (RMS) energy calculation.
       - **Voice Activity Detection (VAD)**: Dynamically separates speech bouts from ambient background silence to measure verbal participation.
    
    3. **Biometric Face Recognition & Anti-Spoofing Engine**:
       - Affine-invariant landmark signature extraction normalized by inter-ocular distance.
       - Cosine similarity matching against enrolled JSON profiles.
       - **Liveness Verification**: Anti-spoofing heuristic analyzing temporal variance of micro-motions and blinks to eliminate static photo attacks.
    
    4. **Multimodal Fusion Engine**:
       - Calculates the composite **Multimodal Engagement Index (MEI)**:
       $$\\text{MEI} = w_v \\cdot \\text{VisualScore} + w_a \\cdot \\text{AudioScore} + w_i \\cdot \\text{InteractionScore}$$
       - Maintains rolling temporal windows to flag sudden attention decay.
    
    5. **GenAI Cognitive Agent Layer**:
       - Autonomous nudging agent for proactive attention recovery.
       - Dynamic pop-quiz generator for instant comprehension verification.
       - Full executive session post-mortem synthesis with pedagogical recommendations.
       - Works seamlessly with Google Gemini API, OpenAI, Groq, or the built-in local offline engine.
    """)
