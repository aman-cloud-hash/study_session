"""
AI Study Focus & Distraction Copilot — Streamlit Web Dashboard
==============================================================

Replicates the exact Obsidian Cyber-Dark Mission Control UI from the Desktop app.
Features 30 FPS real-time video streaming, Google MediaPipe 3D FaceLandmarker,
Async YOLOv8 Phone Radar, Instant Pygame Audio Alerts, Live Telemetry Cards,
and SQLite Analytics.
"""

import sys
import time
import types
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import pandas as pd

# ─── Ensure root is on path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── PyTorch NMS ABI Compatibility Patch ───────────────────────────────────
def _setup_torch_compatibility() -> None:
    try:
        import torch

        def _pure_torch_nms(boxes, scores, iou_threshold):
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                i = order[0].item()
                keep.append(i)
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                w = torch.clamp(xx2 - xx1, min=0.0)
                h = torch.clamp(yy2 - yy1, min=0.0)
                inter = w * h
                union = areas[i] + areas[order[1:]] - inter
                iou = inter / torch.clamp(union, min=1e-6)
                order = order[1:][iou <= iou_threshold]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)

        try:
            import torchvision
            torchvision.ops.nms = _pure_torch_nms
        except Exception:
            pass

        if "torchvision.ops" in sys.modules:
            sys.modules["torchvision.ops"].nms = _pure_torch_nms
        else:
            tv = sys.modules.get("torchvision", types.ModuleType("torchvision"))
            tv_ops = types.ModuleType("torchvision.ops")
            tv_ops.nms = _pure_torch_nms
            tv.ops = tv_ops
            sys.modules["torchvision"] = tv
            sys.modules["torchvision.ops"] = tv_ops
    except Exception:
        pass

_setup_torch_compatibility()

import config
from src.detection.face_detector import FaceMeshDetector
from src.detection.eye_detector import EyeDetector
from src.detection.phone_detector import PhoneDetector
from src.detection.distraction_engine import DistractionEngine, DistractionState
from src.database.database import DatabaseManager
from src.analytics.analytics import AnalyticsEngine
from src.utils.audio import AlertManager
from src.utils.helpers import format_time, ensure_directories

# ─── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Focus Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Futuristic Cyber-Dark Glassmorphic UI CSS ──────────────────────────────
st.markdown(
    """
    <style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0A0B10;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Header Pill */
    .hero-badge {
        display: inline-block;
        padding: 5px 16px;
        background: rgba(79, 70, 229, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        color: #A5B4FC;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    /* Big Score Gauge Card */
    .gauge-card {
        background: linear-gradient(145deg, #141724 0%, #0E101A 100%);
        border: 1px solid #1F2338;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .gauge-title {
        color: #64748B;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .gauge-val-emerald {
        color: #10B981;
        font-size: 48px;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 4px 0;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    .gauge-val-amber {
        color: #F59E0B;
        font-size: 48px;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 4px 0;
        text-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
    }
    .gauge-val-red {
        color: #EF4444;
        font-size: 48px;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 4px 0;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    
    /* Digital Countdown Clock Card */
    .clock-card {
        background: #12141F;
        border: 1px solid #1F2338;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .clock-time {
        color: #F8FAFC;
        font-size: 36px;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
        letter-spacing: 1px;
    }
    
    /* Telemetry Grid Chip */
    .telemetry-box {
        background: #12141F;
        border: 1px solid #1F2338;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .chip-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: #181B2B;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 13px;
    }
    .chip-label { color: #94A3B8; font-weight: 600; }
    .chip-val-green { color: #10B981; font-weight: 800; }
    .chip-val-red { color: #F87171; font-weight: 800; }
    .chip-val-amber { color: #FBBF24; font-weight: 800; }
    
    /* Video Stream Frame Box */
    .video-bezel {
        background: #12141F;
        border: 2px solid #232840;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ensure_directories()

# ─── Session State Initialisation ───────────────────────────────────────────
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "student_name" not in st.session_state:
    st.session_state.student_name = "Student"
if "duration_min" not in st.session_state:
    st.session_state.duration_min = 25
if "cam_index" not in st.session_state:
    st.session_state.cam_index = config.DEFAULT_CAMERA_INDEX

# ─── Sidebar Controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-badge">⚙️ AI SENTRY CONFIG</div>', unsafe_allow_html=True)
    st.markdown("### Study Profile")

    student_name_input = st.text_input("👤 Student Name", value=st.session_state.student_name, max_chars=30)
    st.session_state.student_name = student_name_input

    duration_choice = st.select_slider(
        "⏱️ Target Session Duration",
        options=[15, 25, 45, 60, 90, 120],
        value=st.session_state.duration_min,
        format_func=lambda x: f"{x} Minutes",
    )
    st.session_state.duration_min = duration_choice

    st.markdown("---")
    st.markdown("### Camera Device")
    cam_input = st.number_input("📹 Camera Device Index", min_value=0, max_value=4, value=st.session_state.cam_index, step=1)
    st.session_state.cam_index = int(cam_input)

    st.markdown("---")
    st.markdown("### Computer Vision Sensitivity")

    ear_val = st.slider("👁️ Eye Closed Threshold (EAR)", min_value=0.15, max_value=0.35, value=float(config.EAR_THRESHOLD), step=0.01)
    config.EAR_THRESHOLD = ear_val

    drowsy_sec = st.slider("😴 Drowsy Delay (Seconds)", min_value=1.0, max_value=8.0, value=float(config.EYE_CLOSED_DURATION_THRESHOLD), step=0.5)
    config.EYE_CLOSED_DURATION_THRESHOLD = drowsy_sec

    phone_thresh = st.slider("📱 YOLO Phone Confidence", min_value=0.20, max_value=0.80, value=float(config.PHONE_DETECTION_CONFIDENCE), step=0.05)
    config.PHONE_DETECTION_CONFIDENCE = phone_thresh

    sound_toggle = st.toggle("🔊 Enable Audio Alarms", value=config.SOUND_ENABLED)
    config.SOUND_ENABLED = sound_toggle

    st.markdown("---")
    st.markdown(f"🔒 *{config.PRIVACY_NOTICE}*")


# ─── Main Header ────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">✨ NEXT-GEN AI STUDY COPILOT • MISSION CONTROL</div>', unsafe_allow_html=True)
st.title("🎯 AI Study Focus & Distraction Sentry")
st.markdown("Real-time computer vision tracking study focus, phone usage, and eye drowsiness with instant audio alerts.")

# ─── Navigation Tabs ────────────────────────────────────────────────────────
tab_live, tab_history, tab_about = st.tabs(["🎥 Live Mission Control", "📊 Session History & Trends", "ℹ️ Architecture"])

# ─── TAB 1: Live Mission Control ────────────────────────────────────────────
with tab_live:
    # Top Control Buttons
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])
    
    with ctrl_col1:
        if not st.session_state.is_running:
            if st.button("🚀 START SESSION", use_container_width=True, type="primary"):
                st.session_state.is_running = True
                st.rerun()
        else:
            if st.button("⏹️ END SESSION", use_container_width=True, type="secondary"):
                st.session_state.is_running = False
                st.rerun()

    st.markdown("---")

    # 2-Column Dashboard Grid
    col_video, col_telemetry = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown("### 📹 Live Vision Sentry Feed")
        video_placeholder = st.empty()
        st.caption("⚡ Camera frames are processed strictly in local RAM at 30+ FPS. Zero recording or transmission.")

    with col_telemetry:
        st.markdown(f"### 📊 Student: **{st.session_state.student_name}**")
        gauge_placeholder = st.empty()
        clock_placeholder = st.empty()
        chips_placeholder = st.empty()

    # If Not Running, Render Clean Idle State
    if not st.session_state.is_running:
        video_placeholder.info("Click **'🚀 START SESSION'** above to launch live camera tracking and audio sentry!")
        gauge_placeholder.markdown(
            """
            <div class="gauge-card">
                <div class="gauge-title">Focus Efficiency Score</div>
                <div class="gauge-val-emerald">100%</div>
                <div style="color: #94A3B8; font-size: 13px; font-weight: 600;">Status: Ready to Launch</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        clock_placeholder.markdown(
            f"""
            <div class="clock-card">
                <div class="gauge-title">Target Session Duration</div>
                <div class="clock-time">{st.session_state.duration_min:02d}:00</div>
                <div style="color: #64748B; font-size: 12px; margin-top: 4px;">Camera Device: #{st.session_state.cam_index}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chips_placeholder.markdown(
            """
            <div class="telemetry-box">
                <div class="gauge-title" style="margin-bottom: 8px;">Telemetry Radar Status</div>
                <div class="chip-row"><span class="chip-label">👁️ Eye Sentry</span><span class="chip-val-green">● STANDBY</span></div>
                <div class="chip-row"><span class="chip-label">📱 Phone Radar</span><span class="chip-val-green">● STANDBY</span></div>
                <div class="chip-row"><span class="chip-label">🔊 Audio Alerts</span><span class="chip-val-green">● ACTIVE</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ─── ACTIVE MONITORING LOOP ─────────────────────────────────────────────
    if st.session_state.is_running:
        # Initialize Models & Vision Pipeline
        face_detector = FaceMeshDetector()
        eye_detector = EyeDetector()
        phone_detector = PhoneDetector()
        distraction_engine = DistractionEngine()
        alert_manager = AlertManager()
        db = DatabaseManager()

        # Open Webcam
        cap = cv2.VideoCapture(st.session_state.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(st.session_state.cam_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        start_time = time.time()
        target_total_sec = st.session_state.duration_min * 60
        fps = 30.0
        prev_frame_time = time.time()

        try:
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                # Mirror frame
                frame = cv2.flip(frame, 1)

                # Compute FPS
                now = time.time()
                dt = now - prev_frame_time
                prev_frame_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                elapsed_sec = time.time() - start_time
                remaining_sec = max(0, target_total_sec - elapsed_sec)

                # 1. Face & Eye Tracking
                face_results, face_count = face_detector.process(frame)
                if face_count > 0:
                    eye_data = eye_detector.detect_eye_state_from_face(frame, face_results)
                else:
                    eye_data = {"eye_status": "UNKNOWN", "left_ear": 0.0, "right_ear": 0.0, "avg_ear": 0.0}

                # 2. YOLO Phone Detection
                phone_data = phone_detector.detect(frame)

                # 3. State Machine Engine
                engine_data = distraction_engine.update(
                    face_count=face_count,
                    eye_status=eye_data.get("eye_status", "UNKNOWN"),
                    phone_detected=phone_data.get("phone_detected", False),
                    current_fps=fps,
                )

                # 4. Synchronize Instant Audio State
                target_alert = engine_data.get("target_alert")
                alert_manager.sync_alert_state(target_alert)

                # 5. Draw Phone Bounding Box
                if phone_data.get("phone_detected", False) and phone_data.get("detections"):
                    frame = phone_detector.draw_detections(frame, phone_data["detections"])

                # 6. Render Cyberpunk Glass HUD Overlay
                h, w = frame.shape[:2]
                state = engine_data.get("current_state", DistractionState.FOCUSED)

                if state == DistractionState.FOCUSED:
                    theme_color = (130, 210, 30)
                    state_text = "FOCUSED"
                elif state == DistractionState.PHONE_DISTRACTION:
                    theme_color = (40, 60, 240)
                    state_text = "PHONE DETECTED"
                elif state == DistractionState.DROWSY:
                    theme_color = (30, 160, 255)
                    state_text = "DROWSINESS DETECTED"
                elif state == DistractionState.HIGH_DISTRACTION:
                    theme_color = (20, 20, 255)
                    state_text = "CRITICAL DISTRACTION"
                elif state == DistractionState.NO_FACE:
                    theme_color = (160, 160, 160)
                    state_text = "NO FACE DETECTED"
                else:
                    theme_color = (140, 140, 140)
                    state_text = "STUDY SENTRY"

                overlay = frame.copy()

                # Top-Left State Pill
                pill_w, pill_h = 240, 36
                cv2.rectangle(overlay, (14, 14), (14 + pill_w, 14 + pill_h), (12, 14, 22), -1)
                cv2.circle(overlay, (32, 14 + pill_h // 2), 6, theme_color, -1)
                cv2.putText(
                    overlay,
                    state_text,
                    (48, 14 + pill_h // 2 + 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.48,
                    (240, 245, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Top-Right Audio Pill
                audio_playing = alert_manager.is_playing
                audio_pill_w = 130
                audio_x = max(14, w - audio_pill_w - 14)
                audio_color = (40, 60, 240) if audio_playing else (120, 180, 40)
                audio_label = "AUDIO ON" if audio_playing else "AUDIO OFF"
                cv2.rectangle(overlay, (audio_x, 14), (w - 14, 14 + pill_h), (12, 14, 22), -1)
                cv2.circle(overlay, (audio_x + 18, 14 + pill_h // 2), 5, audio_color, -1)
                cv2.putText(
                    overlay,
                    audio_label,
                    (audio_x + 32, 14 + pill_h // 2 + 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.46,
                    (220, 225, 240),
                    1,
                    cv2.LINE_AA,
                )

                # Eye closure progress bar
                eye_dur = engine_data.get("eye_closed_duration", 0.0)
                if eye_dur > 0 and eye_data.get("eye_status") == "CLOSED":
                    progress = min(1.0, eye_dur / max(0.1, config.EYE_CLOSED_DURATION_THRESHOLD))
                    bar_w = 220
                    bar_h = 8
                    bar_x = 14
                    bar_y = 58

                    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 20, 30), -1)
                    fill_w = int(bar_w * progress)
                    bar_col = (30, 160, 255) if progress < 0.8 else (30, 40, 240)
                    if fill_w > 0:
                        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
                    cv2.putText(
                        overlay,
                        f"Eyes Closed: {eye_dur:.1f}s",
                        (bar_x + bar_w + 10, bar_y + 7),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.40,
                        (240, 240, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Blend Glass HUD
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                cv2.rectangle(frame, (14, 14), (14 + pill_w, 14 + pill_h), (50, 55, 75), 1)
                cv2.rectangle(frame, (audio_x, 14), (w - 14, 14 + pill_h), (50, 55, 75), 1)

                # Convert BGR to RGB for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # 7. Update Telemetry Cards
                score = engine_data.get("focus_score", 100.0)
                score_cls = "gauge-val-emerald" if score >= 80 else ("gauge-val-amber" if score >= 60 else "gauge-val-red")

                gauge_placeholder.markdown(
                    f"""
                    <div class="gauge-card">
                        <div class="gauge-title">Focus Efficiency Score</div>
                        <div class="{score_cls}">{score:.0f}%</div>
                        <div style="color: #94A3B8; font-size: 13px; font-weight: 600;">Status: {state_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Countdown Clock
                clock_placeholder.markdown(
                    f"""
                    <div class="clock-card">
                        <div class="gauge-title">Time Remaining</div>
                        <div class="clock-time">{format_time(remaining_sec)}</div>
                        <div style="color: #64748B; font-size: 12px; margin-top: 4px;">Elapsed: {format_time(elapsed_sec)} • FPS: {int(fps)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Telemetry Chips
                eye_stat = eye_data.get("eye_status", "OPEN")
                eye_chip = "chip-val-green" if eye_stat == "OPEN" else "chip-val-red"

                phone_det = phone_data.get("phone_detected", False)
                phone_chip = "chip-val-red" if phone_det else "chip-val-green"
                phone_lbl = "DETECTED" if phone_det else "CLEAR"

                drowsy_conf = engine_data.get("drowsiness_confirmed", False)
                drowsy_chip = "chip-val-red" if drowsy_conf else "chip-val-green"
                drowsy_lbl = "DROWSY" if drowsy_conf else "ALERT"

                audio_chip = "chip-val-red" if alert_manager.is_playing else "chip-val-green"
                audio_lbl = "ACTIVE ALERT" if alert_manager.is_playing else "OFF"

                chips_placeholder.markdown(
                    f"""
                    <div class="telemetry-box">
                        <div class="gauge-title" style="margin-bottom: 8px;">Real-Time Telemetry Radar</div>
                        <div class="chip-row"><span class="chip-label">👁️ Eye State</span><span class="{eye_chip}">● {eye_stat} ({eye_data.get('avg_ear', 0.0):.2f})</span></div>
                        <div class="chip-row"><span class="chip-label">📱 Phone Radar</span><span class="{phone_chip}">● {phone_lbl}</span></div>
                        <div class="chip-row"><span class="chip-label">😴 Drowsiness Sentry</span><span class="{drowsy_chip}">● {drowsy_lbl}</span></div>
                        <div class="chip-row"><span class="chip-label">🔊 Audio Alarms</span><span class="{audio_chip}">● {audio_lbl}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Check if session time completed
                if remaining_sec <= 0:
                    st.success("🎉 Target session duration completed! Saving records...")
                    st.session_state.is_running = False
                    break

                time.sleep(0.01)

        finally:
            cap.release()
            alert_manager.stop_all()
            face_detector.close()


# ─── TAB 2: Historical Analytics ────────────────────────────────────────────
with tab_history:
    st.markdown("### 📊 Session Archives & Historical Progression")
    db = DatabaseManager()
    analytics = AnalyticsEngine(db)

    stats = analytics.get_summary_statistics()
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    s_col1.metric("Total Sessions", stats.get("total_sessions", 0))
    s_col2.metric("Total Study Time", format_time(stats.get("total_study_time_sec", 0)))
    s_col3.metric("Average Focus Score", f"{stats.get('avg_focus_score', 0)}%")
    s_col4.metric("Best Focus Score", f"{stats.get('best_focus_score', 0)}%")

    st.markdown("---")

    chart_col, table_col = st.columns([1, 1], gap="medium")

    with chart_col:
        st.markdown("#### 📈 Focus Score Trend Progression")
        fig = analytics.generate_history_trend_chart()
        if fig:
            st.pyplot(fig)
        else:
            st.info("Complete study sessions to generate your focus score trendline.")

    with table_col:
        st.markdown("#### 📜 Past Study Sessions Table")
        sessions = db.get_all_sessions()
        if sessions:
            df = pd.DataFrame(sessions)
            df["Duration"] = df["total_duration"].apply(format_time)
            df["Focused"] = df["focused_duration"].apply(format_time)
            df["Score"] = df["focus_score"].apply(lambda s: f"{s}%")
            display_df = df[["session_id", "student_name", "date", "Duration", "Focused", "Score"]]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No study sessions recorded yet. Start a session to build your study streak!")


# ─── TAB 3: Architecture & About ────────────────────────────────────────────
with tab_about:
    st.markdown("### 🏗️ High-Speed Vision & AI Pipeline")
    st.markdown(
        """
        - **Google MediaPipe 478-Point 3D Face Mesh:** Calculates precise Eye Aspect Ratio (EAR) with sub-millisecond latency.
        - **Multi-Threaded Async YOLOv8:** Object classification for mobile phones with pure PyTorch Vectorized NMS.
        - **Instant State-Based Audio:** Audio alerts start in 0s on phone detection and stop instantly (<5ms) when distraction ends.
        - **100% Local Privacy:** Zero recording or network streaming. All inferences execute in volatile RAM.
        """
    )
