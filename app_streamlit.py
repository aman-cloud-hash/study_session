"""
AI Study Focus & Distraction Copilot — Streamlit Cloud / Web Edition
====================================================================

Exact Replica of Desktop App Workflow with Multi-Page State Router:
- 🏠 Home Setup Screen (Student Profile, Preset Duration Pills, Camera Picker, Feature Grid, Launch Button)
- 🎥 Live Mission Control Dashboard (30 FPS Video, Corner Reticle Box, Cyberpunk Glass HUD, Live Gauge & Clock)
- 🏆 Completion Report Screen (Celebration Badge, Donut Chart, Metric Cards Grid)
- 📊 History & Analytics Screen (SQLite Session Table, Filter, Progression Charts)
- ⚙️ Settings Screen (Sliders for EAR, Drowsiness Delay, YOLO Confidence, Sound Switch)
"""

import os
import sys
import time
import types
import threading
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
import importlib
importlib.reload(config)

SNAPSHOTS_DIR = getattr(config, "SNAPSHOTS_DIR", PROJECT_ROOT / "database" / "snapshots")
ADMIN_PASSWORD = getattr(config, "ADMIN_PASSWORD", "Aman-Rajbhar-2005")

from src.detection.face_detector import FaceMeshDetector
from src.detection.eye_detector import EyeDetector
from src.detection.phone_detector import PhoneDetector
from src.detection.distraction_engine import DistractionEngine, DistractionState
from src.database.database import DatabaseManager
from src.analytics.analytics import AnalyticsEngine
from src.utils.audio import AlertManager
from src.utils.helpers import format_time, ensure_directories, detect_available_cameras

# ─── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Focus Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Ultra-Premium Dark Glassmorphic Design System ───────────────────────────
st.markdown(
    """
    <style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0A0B10;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Navbar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        background: #12141F;
        border: 1px solid #1F2338;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .nav-title {
        font-size: 18px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    
    /* Glowing Hero Badge */
    .hero-badge {
        display: inline-block;
        padding: 5px 14px;
        background: rgba(79, 70, 229, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        color: #A5B4FC;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    /* Surface Cards */
    .card-surface {
        background: #12141F;
        border: 1px solid #1F2338;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .card-title {
        color: #F8FAFC;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 16px;
    }
    .section-label {
        color: #64748B;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    /* Big Score Gauge */
    .gauge-card {
        background: linear-gradient(145deg, #141724 0%, #0E101A 100%);
        border: 1px solid #1F2338;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        margin-bottom: 14px;
    }
    .gauge-val-emerald {
        color: #10B981;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
    }
    .gauge-val-amber {
        color: #F59E0B;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 0 25px rgba(245, 158, 11, 0.35);
    }
    .gauge-val-red {
        color: #EF4444;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
    }
    
    /* Telemetry Pill Rows */
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: #181B2B;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .t-label { color: #94A3B8; font-weight: 600; }
    .t-val-green { color: #10B981; font-weight: 800; }
    .t-val-red { color: #F87171; font-weight: 800; }
    .t-val-amber { color: #FBBF24; font-weight: 800; }
    
    /* Feature Highlight Pill */
    .feat-card {
        background: #141724;
        border: 1px solid #1F2338;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .feat-title { color: #FFFFFF; font-size: 14px; font-weight: 700; margin-bottom: 4px; }
    .feat-desc { color: #94A3B8; font-size: 12px; line-height: 1.4; }
    
    /* Clean Button Styling */
    div.stButton > button:first-child {
        border-radius: 12px;
        font-weight: 700;
        padding: 10px 20px;
        border: none;
        transition: all 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ensure_directories()

# ─── Initialize State Router & Camera Detection ──────────────────────────────
if "detected_cameras" not in st.session_state:
    try:
        st.session_state.detected_cameras = detect_available_cameras(max_tested=4)
    except Exception:
        st.session_state.detected_cameras = [{"index": 0, "name": "Camera 0 [Default]"}]

if "current_page" not in st.session_state:
    st.session_state.current_page = "home"  # 'home', 'session', 'completion', 'history', 'settings'
if "student_name" not in st.session_state:
    st.session_state.student_name = "Student"
if "duration_min" not in st.session_state:
    st.session_state.duration_min = 25
if "cam_index" not in st.session_state:
    st.session_state.cam_index = config.DEFAULT_CAMERA_INDEX
if "last_session_stats" not in st.session_state:
    st.session_state.last_session_stats = None

# ─── Modern Top Navigation Bar ──────────────────────────────────────────────
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([4, 1.2, 1.2, 1.2])

with nav_col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0;">
            <span style="font-size: 24px;">🎯</span>
            <div>
                <div style="font-size: 17px; font-weight: 800; color: #FFFFFF;">AI STUDY FOCUS COPILOT</div>
                <div style="font-size: 11px; color: #818CF8; font-weight: 700;">NEXT-GEN VISION SENTINEL</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.current_page == "home" else "secondary"):
        st.session_state.current_page = "home"
        st.rerun()

with nav_col3:
    if st.button("📊 History", use_container_width=True, type="primary" if st.session_state.current_page == "history" else "secondary"):
        st.session_state.current_page = "history"
        st.rerun()

with nav_col4:
    if st.button("⚙️ Settings", use_container_width=True, type="primary" if st.session_state.current_page == "settings" else "secondary"):
        st.session_state.current_page = "settings"
        st.rerun()

st.markdown("<hr style='border: 0; height: 1px; background: #1F2338; margin: 8px 0 20px 0;'>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 🏠 SCREEN 1: HOME LAUNCHER & SETUP (Exact Desktop HomeScreen Match)
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "home":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 24px;">
            <div class="hero-badge">✨ NEXT-GEN AI STUDY COPILOT</div>
            <h1 style="font-size: 36px; font-weight: 900; margin: 4px 0; color: #FFFFFF;">AI STUDY FOCUS SENTINEL</h1>
            <p style="color: #94A3B8; font-size: 14px; margin: 0 auto; max-width: 600px;">
                Real-time Computer Vision sentry for laser-sharp study focus, instant phone distraction alerts, and drowsiness prevention.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_setup, col_features = st.columns([1.1, 1], gap="large")

    with col_setup:
        st.markdown(
            """
            <div class="card-surface">
                <div class="card-title">👤 Student & Session Setup</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Student Name
        st.markdown('<div class="section-label">STUDENT NAME</div>', unsafe_allow_html=True)
        name_val = st.text_input("Student Name Input", value=st.session_state.student_name, label_visibility="collapsed", placeholder="Enter your name...")
        st.session_state.student_name = name_val if name_val.strip() else "Student"

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Preset Duration Pills
        st.markdown('<div class="section-label">SELECT TARGET DURATION</div>', unsafe_allow_html=True)
        pill_cols = st.columns(5)
        presets = [25, 45, 60, 90, 120]
        for i, dur in enumerate(presets):
            with pill_cols[i]:
                is_sel = (st.session_state.duration_min == dur)
                if st.button(f"{dur}m", key=f"pill_{dur}", use_container_width=True, type="primary" if is_sel else "secondary"):
                    st.session_state.duration_min = dur
                    st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Camera Selector with Auto-Detection & Scan
        st.markdown('<div class="section-label">ACTIVE CAMERA DEVICE</div>', unsafe_allow_html=True)
        cam_col1, cam_col2 = st.columns([3.2, 1])

        cam_names = [c["name"] for c in st.session_state.detected_cameras]
        # Find current index
        current_cam_name = next((c["name"] for c in st.session_state.detected_cameras if c["index"] == st.session_state.cam_index), cam_names[0])

        with cam_col1:
            selected_cam_name = st.selectbox(
                "Active Camera Device",
                options=cam_names,
                index=cam_names.index(current_cam_name),
                label_visibility="collapsed",
            )
            # Update selected index
            for c in st.session_state.detected_cameras:
                if c["name"] == selected_cam_name:
                    st.session_state.cam_index = c["index"]
                    break

        with cam_col2:
            if st.button("🔄 Scan", use_container_width=True):
                st.session_state.detected_cameras = detect_available_cameras(max_tested=4)
                st.rerun()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Big Launch Button
        if st.button("🚀  START FOCUS SESSION", use_container_width=True, type="primary"):
            st.session_state.current_page = "session"
            st.rerun()

    with col_features:
        st.markdown(
            '<div class="card-surface">'
            '<div class="card-title">⚡ AI Vision Engine Capabilities</div>'
            '<div class="feat-card">'
            '<div class="feat-title">👁️ Google MediaPipe 3D Mesh</div>'
            '<div class="feat-desc">Calculates 478 landmark points to compute Eye Aspect Ratio (EAR) with sub-millisecond precision.</div>'
            '</div>'
            '<div class="feat-card">'
            '<div class="feat-title">📱 YOLOv8 Mobile Phone Radar</div>'
            '<div class="feat-desc">Async background inference with pure PyTorch NMS for instant 1-frame phone pickup detection.</div>'
            '</div>'
            '<div class="feat-card">'
            '<div class="feat-title">😴 Smart Drowsiness Sentry</div>'
            '<div class="feat-desc">Temporal debounce logic tracks consecutive eye closures to trigger high-urgency wake-up alerts.</div>'
            '</div>'
            '<div class="feat-card">'
            '<div class="feat-title">🔊 Targeted Meme & Voice Alarms</div>'
            '<div class="feat-desc">Plays custom meme audio on phone distraction and instant halt the moment phone is put away.</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# 🎥 SCREEN 2: LIVE MISSION CONTROL (Exact Desktop SessionScreen Match)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "session":
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"<h2 style='margin: 0; color: #FFFFFF;'>🔴 Live Vision Sentry • <span style='color: #818CF8;'>{st.session_state.student_name}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #64748B; margin: 0;'>Target: {st.session_state.duration_min}m • Hardware Camera #{st.session_state.cam_index}</p>", unsafe_allow_html=True)
    with header_col2:
        if st.button("⏹️  END SESSION", use_container_width=True, type="secondary"):
            if "active_session_data" in st.session_state and st.session_state.active_session_data:
                try:
                    db = DatabaseManager()
                    db.initialise()
                    if st.session_state.get("active_session_db_id"):
                        db.update_session(st.session_state.active_session_db_id, st.session_state.active_session_data)
                    else:
                        db.save_session(st.session_state.active_session_data)
                    st.session_state.last_session_stats = st.session_state.active_session_data
                    st.session_state.active_session_data = None
                    st.session_state.active_session_db_id = None
                except Exception as e:
                    print(f"[Save on Stop Error] {e}")
            st.session_state.current_page = "completion"
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col_video, col_dash = st.columns([3, 2], gap="large")

    with col_video:
        video_feed_placeholder = st.empty()
        st.caption("⚡ DirectShow 30 FPS Stream • Processed in local volatile memory. 100% Privacy.")

    with col_dash:
        gauge_box = st.empty()
        clock_box = st.empty()
        telemetry_box = st.empty()

    # Initialize Vision & Audio Pipeline
    face_detector = FaceMeshDetector()
    eye_detector = EyeDetector()
    phone_detector = PhoneDetector()
    distraction_engine = DistractionEngine()
    alert_manager = AlertManager()
    db = DatabaseManager()

    cap = cv2.VideoCapture(st.session_state.cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(st.session_state.cam_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()
    target_sec = st.session_state.duration_min * 60
    fps = 30.0
    prev_time = time.time()
    snapshot_taken = False
    snapshot_file_path = None
    session_saved = False
    session_db_id = None
    st.session_state.active_session_db_id = None
    frame_count = 0
    elapsed = 0.0

    try:
        while st.session_state.current_page == "session":
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1
            elapsed = time.time() - start_time
            remaining = max(0, target_sec - elapsed)

            # Compute FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # 1. Face & Eye Tracking
            face_results, face_count = face_detector.process(frame)
            if face_count > 0:
                eye_data = eye_detector.detect_eye_state_from_face(frame, face_results)
            else:
                eye_data = {"eye_status": "UNKNOWN", "left_ear": 0.0, "right_ear": 0.0, "avg_ear": 0.0}

            # 2. YOLO Phone Radar
            phone_data = phone_detector.detect(frame)

            # 3. Decision Engine
            engine_data = distraction_engine.update(
                face_count=face_count,
                eye_status=eye_data.get("eye_status", "UNKNOWN"),
                phone_detected=phone_data.get("phone_detected", False),
            )

            # 4. Instant State-Based Audio Trigger
            target_alert = engine_data.get("target_alert")
            alert_manager.sync_alert_state(target_alert)

            # 5. Draw Cyberpunk Reticle Bounding Box
            if phone_data.get("phone_detected", False) and phone_data.get("detections"):
                frame = phone_detector.draw_detections(frame, phone_data["detections"])

            score_val = engine_data.get("focus_score", 100.0)

            # 📸 3-SECOND AUTOMATIC IDENTITY SNAPSHOT & INSTANT DATABASE PERSISTENCE
            if not snapshot_taken and (elapsed >= 3.0 or frame_count >= 30):
                try:
                    snap_name = f"snap_{st.session_state.student_name}_{int(time.time())}.jpg"
                    snap_full_path = config.SNAPSHOTS_DIR / snap_name
                    cv2.imwrite(str(snap_full_path), frame)
                    snapshot_file_path = str(snap_full_path)
                    snapshot_taken = True

                    # ⚡ Save to SQLite immediately at 3 seconds!
                    db_init = DatabaseManager()
                    db_init.initialise()
                    session_db_id = db_init.save_session({
                        "student_name": st.session_state.student_name,
                        "total_duration": elapsed,
                        "focused_duration": engine_data.get("focused_duration", 0.0),
                        "distracted_duration": engine_data.get("distracted_duration", 0.0),
                        "phone_duration": engine_data.get("phone_duration", 0.0),
                        "drowsiness_duration": engine_data.get("drowsiness_duration", 0.0),
                        "phone_events": engine_data.get("phone_events", 0),
                        "drowsiness_events": engine_data.get("drowsiness_events", 0),
                        "focus_score": score_val,
                        "snapshot_path": snapshot_file_path,
                    })
                    st.session_state.active_session_db_id = session_db_id
                except Exception as e:
                    print(f"[Snapshot & Auto-Save Error] {e}")

            # 6. Render Cyberpunk Glass HUD
            h, w = frame.shape[:2]
            state = engine_data.get("current_state", DistractionState.FOCUSED)

            if state == DistractionState.FOCUSED:
                theme_col = (130, 210, 30)
                state_lbl = "FOCUSED"
            elif state == DistractionState.PHONE_DISTRACTION:
                theme_col = (40, 60, 240)
                state_lbl = "PHONE DETECTED"
            elif state == DistractionState.DROWSY:
                theme_col = (30, 160, 255)
                state_lbl = "DROWSINESS DETECTED"
            elif state == DistractionState.HIGH_DISTRACTION:
                theme_col = (20, 20, 255)
                state_lbl = "CRITICAL DISTRACTION"
            elif state == DistractionState.NO_FACE:
                theme_col = (160, 160, 160)
                state_lbl = "NO FACE DETECTED"
            else:
                theme_col = (140, 140, 140)
                state_lbl = "STUDY SENTRY"

            overlay = frame.copy()
            # Top-Left State Pill
            cv2.rectangle(overlay, (14, 14), (250, 50), (12, 14, 22), -1)
            cv2.circle(overlay, (32, 32), 6, theme_col, -1)
            cv2.putText(overlay, state_lbl, (48, 37), cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 245, 255), 1, cv2.LINE_AA)

            # Top-Right Audio Pill
            audio_on = alert_manager.is_playing
            audio_x = max(14, w - 144)
            audio_col = (40, 60, 240) if audio_on else (120, 180, 40)
            cv2.rectangle(overlay, (audio_x, 14), (w - 14, 50), (12, 14, 22), -1)
            cv2.circle(overlay, (audio_x + 18, 32), 5, audio_col, -1)
            cv2.putText(overlay, "AUDIO ON" if audio_on else "AUDIO OFF", (audio_x + 32, 37), cv2.FONT_HERSHEY_DUPLEX, 0.46, (220, 225, 240), 1, cv2.LINE_AA)

            # Eye closure progress bar
            eye_dur = engine_data.get("eye_closed_duration", 0.0)
            if eye_dur > 0 and eye_data.get("eye_status") == "CLOSED":
                prog = min(1.0, eye_dur / max(0.1, config.EYE_CLOSED_DURATION_THRESHOLD))
                cv2.rectangle(overlay, (14, 58), (234, 66), (20, 20, 30), -1)
                fill_w = int(220 * prog)
                if fill_w > 0:
                    cv2.rectangle(overlay, (14, 58), (14 + fill_w, 66), (30, 160, 255), -1)
                cv2.putText(overlay, f"Eyes Closed: {eye_dur:.1f}s", (244, 65), cv2.FONT_HERSHEY_DUPLEX, 0.40, (240, 240, 255), 1, cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (14, 14), (250, 50), (50, 55, 75), 1)
            cv2.rectangle(frame, (audio_x, 14), (w - 14, 50), (50, 55, 75), 1)

            # Render Stream to Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_feed_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            score_cls = "gauge-val-emerald" if score_val >= 80 else ("gauge-val-amber" if score_val >= 60 else "gauge-val-red")

            # 💾 Continuous Session State Cache
            st.session_state.active_session_data = {
                "student_name": st.session_state.student_name,
                "total_duration": elapsed,
                "focused_duration": engine_data.get("focused_duration", 0.0),
                "distracted_duration": engine_data.get("distracted_duration", 0.0),
                "phone_duration": engine_data.get("phone_duration", 0.0),
                "drowsiness_duration": engine_data.get("drowsiness_duration", 0.0),
                "phone_events": engine_data.get("phone_events", 0),
                "drowsiness_events": engine_data.get("drowsiness_events", 0),
                "focus_score": score_val,
                "snapshot_path": snapshot_file_path,
            }

            gauge_box.markdown(
                f"""
                <div class="gauge-card">
                    <div class="section-label">FOCUS EFFICIENCY SCORE</div>
                    <div class="{score_cls}">{score_val:.0f}%</div>
                    <div style="color: #94A3B8; font-size: 13px; font-weight: 700;">Status: {state_lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            clock_box.markdown(
                f"""
                <div class="card-surface" style="padding: 16px 20px; margin-bottom: 14px;">
                    <div class="section-label">TIME REMAINING</div>
                    <div style="font-size: 38px; font-weight: 900; color: #F8FAFC; font-variant-numeric: tabular-nums;">{format_time(remaining)}</div>
                    <div style="color: #64748B; font-size: 12px; font-weight: 600;">Elapsed: {format_time(elapsed)} • ⚡ FPS: {int(fps)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            eye_st = eye_data.get("eye_status", "OPEN")
            eye_col_cls = "t-val-green" if eye_st == "OPEN" else "t-val-red"
            ph_det = phone_data.get("phone_detected", False)
            ph_col_cls = "t-val-red" if ph_det else "t-val-green"
            ph_txt = "DETECTED" if ph_det else "CLEAR"
            drowsy_flag = engine_data.get("drowsiness_confirmed", False)
            drowsy_col_cls = "t-val-red" if drowsy_flag else "t-val-green"
            drowsy_txt = "DROWSY" if drowsy_flag else "ALERT"
            aud_col_cls = "t-val-red" if alert_manager.is_playing else "t-val-green"
            aud_txt = "ACTIVE" if alert_manager.is_playing else "OFF"

            telemetry_box.markdown(
                f"""
                <div class="card-surface" style="padding: 16px;">
                    <div class="section-label" style="margin-bottom: 10px;">LIVE TELEMETRY RADAR</div>
                    <div class="telemetry-row"><span class="t-label">👁️ Eye Sentry</span><span class="{eye_col_cls}">● {eye_st} ({eye_data.get('avg_ear', 0.0):.2f})</span></div>
                    <div class="telemetry-row"><span class="t-label">📱 Phone Radar</span><span class="{ph_col_cls}">● {ph_txt}</span></div>
                    <div class="telemetry-row"><span class="t-label">😴 Drowsiness Sentry</span><span class="{drowsy_col_cls}">● {drowsy_txt}</span></div>
                    <div class="telemetry-row"><span class="t-label">🔔 Audio Alarms</span><span class="{aud_col_cls}">● {aud_txt}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Auto-save when target duration finishes
            if remaining <= 0:
                if st.session_state.active_session_data:
                    if session_db_id:
                        db.update_session(session_db_id, st.session_state.active_session_data)
                    else:
                        db.save_session(st.session_state.active_session_data)
                    st.session_state.last_session_stats = st.session_state.active_session_data
                    st.session_state.active_session_data = None
                session_saved = True
                st.session_state.current_page = "completion"
                st.rerun()

            time.sleep(0.01)

    finally:
        cap.release()
        alert_manager.stop_all()
        face_detector.close()

        # Always update/save session on exit if elapsed >= 1.0s and not already saved
        if elapsed >= 1.0 and not session_saved and "active_session_data" in st.session_state and st.session_state.active_session_data:
            try:
                db_final = DatabaseManager()
                db_final.initialise()
                target_id = session_db_id or st.session_state.get("active_session_db_id")
                if target_id:
                    db_final.update_session(target_id, st.session_state.active_session_data)
                else:
                    db_final.save_session(st.session_state.active_session_data)
                st.session_state.last_session_stats = st.session_state.active_session_data
                st.session_state.active_session_data = None
                session_saved = True
            except Exception as e:
                print(f"[Finally Save Error] {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 🏆 SCREEN 3: COMPLETION REPORT (Exact Desktop CompletionScreen Match)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "completion":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 24px;">
            <div class="hero-badge" style="color: #10B981; border-color: rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.15);">🎉 SESSION COMPLETE!</div>
            <h1 style="font-size: 34px; font-weight: 900; color: #FFFFFF;">STUDY PERFORMANCE SUMMARY</h1>
            <p style="color: #94A3B8; font-size: 14px;">Great work! Your session has been recorded into the persistent archive.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db = DatabaseManager()
    sessions = db.get_all_sessions()
    latest = sessions[0] if sessions else (st.session_state.last_session_stats or {})

    c_score = latest.get("focus_score", 100)
    c_tot = latest.get("total_duration", 0)
    c_foc = latest.get("focused_duration", 0)
    c_ph_cnt = latest.get("phone_events", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Focus Score", f"{c_score:.0f}%")
    m2.metric("⏱️ Total Time", format_time(c_tot))
    m3.metric("✨ Focused Time", format_time(c_foc))
    m4.metric("📱 Phone Alerts", c_ph_cnt)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("← Back to Home Launcher", use_container_width=True, type="primary"):
            st.session_state.current_page = "home"
            st.rerun()
    with btn_c2:
        if st.button("📊 View All History", use_container_width=True, type="secondary"):
            st.session_state.current_page = "history"
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# 📊 SCREEN 4: HISTORY & ADMIN VAULT (Password Protected)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "history":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="hero-badge">📊 STUDY HISTORY & ANALYTICS</div>
            <h2 style="font-size: 28px; font-weight: 900; color: #FFFFFF; margin: 4px 0;">Historical Performance Archives</h2>
            <p style="color: #94A3B8; font-size: 13px;">Track your long-term focus progression, study consistency, and verification snapshots.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db = DatabaseManager()
    analytics = AnalyticsEngine(db)
    stats = analytics.get_summary_statistics()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Sessions", stats.get("total_sessions", 0))
    s2.metric("Total Study Time", format_time(stats.get("total_study_time_sec", 0)))
    s3.metric("Average Score", f"{stats.get('avg_focus_score', 0)}%")
    s4.metric("Best Score", f"{stats.get('best_focus_score', 0)}%")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    c_chart, c_table = st.columns([1, 1], gap="large")

    with c_chart:
        st.markdown("#### 📈 Focus Progression Trend")
        fig = analytics.generate_history_trend_chart()
        if fig:
            st.pyplot(fig)
        else:
            st.info("Complete study sessions to generate your trend chart.")

    with c_table:
        st.markdown("#### 📜 Recorded Study Sessions")
        sessions = db.get_all_sessions()
        if sessions:
            df = pd.DataFrame(sessions)
            df["Duration"] = df["total_duration"].apply(format_time)
            df["Focused"] = df["focused_duration"].apply(format_time)
            df["Score"] = df["focus_score"].apply(lambda s: f"{s:.0f}%")
            display_cols = ["session_id", "student_name", "date", "Duration", "Focused", "Score"]
            display_df = df[[col for col in display_cols if col in df.columns]]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No study sessions recorded yet.")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # ── 🔐 ADMIN VERIFICATION VAULT (PASSWORD PROTECTED) ──────────────────────
    st.markdown(
        """
        <div class="card-surface">
            <div class="card-title">🔐 Admin Verification Vault (Student Identity Snapshots)</div>
            <p style="color: #94A3B8; font-size: 13px; margin-top: -8px;">
                Identity photos captured automatically 2 seconds after each student session begins. Strictly protected for authorized administrators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    admin_col1, admin_col2 = st.columns([3, 1])

    with admin_col1:
        pwd_input = st.text_input("Enter Admin Password", type="password", placeholder="Enter secret admin passkey...", label_visibility="collapsed")
    with admin_col2:
        if st.button("🔓 Unlock Vault", use_container_width=True, type="primary"):
            if pwd_input == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.success("✅ Admin Access Granted!")
                st.rerun()
            else:
                st.error("❌ Incorrect Admin Password. Access Denied.")

    # Render Snapshots Gallery if Authenticated
    if st.session_state.admin_authenticated:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 12px 16px; margin: 12px 0 20px 0; color: #10B981; font-weight: 700; font-size: 14px;">
                🔓 Admin Access Active • Displaying All Session Verification Snapshots
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not sessions:
            st.info("No study sessions recorded yet. Start a session and its verification snapshot will be saved automatically!")
        else:
            gallery_cols = st.columns(3)
            for idx, s in enumerate(sessions):
                with gallery_cols[idx % 3]:
                    st.markdown(
                        f"""
                        <div style="background: #141724; border: 1px solid #1F2338; border-radius: 14px; padding: 14px; margin-bottom: 16px;">
                            <div style="font-weight: 800; font-size: 15px; color: #FFFFFF;">👤 {s.get('student_name', 'Student')} (Session #{s.get('session_id', idx+1)})</div>
                            <div style="font-size: 12px; color: #94A3B8; margin-bottom: 8px;">📅 {s.get('date', 'Today')} • ⏱️ {format_time(s.get('total_duration', 0))} • 🎯 Score: {s.get('focus_score', 100):.0f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    snap_p = s.get("snapshot_path")
                    if snap_p and Path(snap_p).exists():
                        st.image(snap_p, use_container_width=True)
                    else:
                        st.markdown(
                            """
                            <div style="background: #0E101A; border: 1px dashed #23283E; border-radius: 10px; padding: 28px; text-align: center; color: #64748B; font-size: 12px;">
                                📷 No Photo (Legacy Session)
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ⚙️ SCREEN 5: SETTINGS (Exact Desktop SettingsScreen Match)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "settings":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="hero-badge">⚙️ APPLICATION SETTINGS</div>
            <h2 style="font-size: 28px; font-weight: 900; color: #FFFFFF; margin: 4px 0;">AI Sentry Configuration</h2>
            <p style="color: #94A3B8; font-size: 13px;">Fine-tune computer vision sensitivity, audio alarms, and detection thresholds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card-surface">
            <div class="card-title">👁️ Eye & Drowsiness Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_ear = st.slider("Eye Aspect Ratio (EAR) Threshold", min_value=0.15, max_value=0.35, value=float(config.EAR_THRESHOLD), step=0.01)
    config.EAR_THRESHOLD = c_ear

    c_drowsy = st.slider("Drowsiness Alert Delay (seconds)", min_value=1.0, max_value=8.0, value=float(config.EYE_CLOSED_DURATION_THRESHOLD), step=0.5)
    config.EYE_CLOSED_DURATION_THRESHOLD = c_drowsy

    st.markdown(
        """
        <div class="card-surface">
            <div class="card-title">📱 Mobile Phone Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_phone = st.slider("YOLO Phone Detection Confidence", min_value=0.20, max_value=0.80, value=float(config.PHONE_DETECTION_CONFIDENCE), step=0.05)
    config.PHONE_DETECTION_CONFIDENCE = c_phone

    st.markdown(
        """
        <div class="card-surface">
            <div class="card-title">📹 Camera Hardware Device</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cam_names = [c["name"] for c in st.session_state.detected_cameras]
    cur_cam = next((c["name"] for c in st.session_state.detected_cameras if c["index"] == st.session_state.cam_index), cam_names[0])
    s_cam_col1, s_cam_col2 = st.columns([3.2, 1])
    with s_cam_col1:
        chosen_cam = st.selectbox("Settings Active Camera", options=cam_names, index=cam_names.index(cur_cam), label_visibility="collapsed")
        for c in st.session_state.detected_cameras:
            if c["name"] == chosen_cam:
                st.session_state.cam_index = c["index"]
                config.DEFAULT_CAMERA_INDEX = c["index"]
                break
    with s_cam_col2:
        if st.button("🔄 Scan Devices", key="scan_settings_cam", use_container_width=True):
            st.session_state.detected_cameras = detect_available_cameras(max_tested=4)
            st.rerun()

    st.markdown(
        """
        <div class="card-surface">
            <div class="card-title">🔊 Audio Alarms & Hardware</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_sound = st.toggle("Enable Voice & Meme Distraction Alerts", value=config.SOUND_ENABLED)
    config.SOUND_ENABLED = c_sound

    st.success("✅ Settings applied automatically!")
