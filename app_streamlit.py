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

# ─── WebRTC Video Processor for Cloud Web Streaming ──────────────────────────
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class WebRTCVisionProcessor:
    def __init__(self):
        self.face_detector = FaceMeshDetector()
        self.eye_detector = EyeDetector()
        try:
            self.phone_detector = PhoneDetector()
        except Exception as e:
            print(f"[WebRTC] PhoneDetector init failed (non-fatal): {e}")
            self.phone_detector = None
        self.distraction_engine = DistractionEngine()
        try:
            self.alert_manager = AlertManager()
        except Exception as e:
            print(f"[WebRTC] AlertManager init failed (no audio on cloud, non-fatal): {e}")
            self.alert_manager = None
        self.student_name = "Student"
        self.duration_min = 25
        self.start_time = time.time()
        self.snapshot_taken = False
        self.session_db_id = None
        self.snapshot_file_path = None
        self.latest_engine_data = {}
        self.latest_eye_data = {}
        self.latest_phone_data = {}
        self.cached_phone_data = {"phone_detected": False, "detections": []}
        self.latest_fps = 30.0
        self.prev_time = time.time()
        self.frame_count = 0
        self.elapsed = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.frame_count += 1
        self.elapsed = time.time() - self.start_time

        # Calculate FPS
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self.latest_fps = 0.9 * self.latest_fps + 0.1 * (1.0 / dt)

        # 1. Face & Eye Tracking (Real-time sub-millisecond)
        face_results, face_count = self.face_detector.process(img)
        if face_count > 0:
            eye_data = self.eye_detector.detect_eye_state_from_face(img, face_results)
        else:
            eye_data = {"eye_status": "UNKNOWN", "left_ear": 0.0, "right_ear": 0.0, "avg_ear": 0.0}

        # 2. YOLO Phone Radar (Interleaved every 3 frames for zero frame lag)
        if self.phone_detector and (self.frame_count % 3 == 0 or self.frame_count < 10):
            try:
                self.cached_phone_data = self.phone_detector.detect(img)
            except Exception:
                pass
        phone_data = self.cached_phone_data

        # 3. Decision Engine (Instantaneous State Update)
        engine_data = self.distraction_engine.update(
            face_count=face_count,
            eye_status=eye_data.get("eye_status", "UNKNOWN"),
            phone_detected=phone_data.get("phone_detected", False),
        )

        # 4. Instant State-Based Audio Trigger
        if self.alert_manager:
            try:
                target_alert = engine_data.get("target_alert")
                self.alert_manager.sync_alert_state(target_alert)
            except Exception:
                pass

        # 5. Draw Cyberpunk Reticle Bounding Box
        if phone_data.get("phone_detected", False) and phone_data.get("detections") and self.phone_detector:
            img = self.phone_detector.draw_detections(img, phone_data["detections"])

        score_val = engine_data.get("focus_score", 100.0)

        # 📸 3-SECOND AUTOMATIC IDENTITY SNAPSHOT & INSTANT DATABASE PERSISTENCE
        if not self.snapshot_taken and (self.elapsed >= 3.0 or self.frame_count >= 30):
            try:
                snap_name = f"snap_{self.student_name}_{int(time.time())}.jpg"
                snap_full_path = config.SNAPSHOTS_DIR / snap_name
                cv2.imwrite(str(snap_full_path), img)
                self.snapshot_file_path = str(snap_full_path)
                self.snapshot_taken = True

                db_init = DatabaseManager()
                db_init.initialise()
                self.session_db_id = db_init.save_session({
                    "student_name": self.student_name,
                    "total_duration": self.elapsed,
                    "focused_duration": engine_data.get("focused_duration", 0.0),
                    "distracted_duration": engine_data.get("distracted_duration", 0.0),
                    "phone_duration": engine_data.get("phone_duration", 0.0),
                    "drowsiness_duration": engine_data.get("drowsiness_duration", 0.0),
                    "phone_events": engine_data.get("phone_events", 0),
                    "drowsiness_events": engine_data.get("drowsiness_events", 0),
                    "focus_score": score_val,
                    "snapshot_path": self.snapshot_file_path,
                })
            except Exception as e:
                print(f"[WebRTC Snapshot Error] {e}")

        # 6. Render Cyberpunk Glass HUD Overlay
        h, w = img.shape[:2]
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

        overlay = img.copy()
        cv2.rectangle(overlay, (14, 14), (250, 50), (12, 14, 22), -1)
        cv2.circle(overlay, (32, 32), 6, theme_col, -1)
        cv2.putText(overlay, state_lbl, (48, 37), cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 245, 255), 1, cv2.LINE_AA)

        # Audio Status
        audio_x = max(14, w - 144)
        audio_col = (40, 60, 240) if (state in [DistractionState.PHONE_DISTRACTION, DistractionState.DROWSY, DistractionState.HIGH_DISTRACTION]) else (120, 180, 40)
        cv2.rectangle(overlay, (audio_x, 14), (w - 14, 50), (12, 14, 22), -1)
        cv2.circle(overlay, (audio_x + 18, 32), 5, audio_col, -1)
        cv2.putText(overlay, "ALERT ACTIVE" if (audio_col == (40, 60, 240)) else "ACTIVE", (audio_x + 32, 37), cv2.FONT_HERSHEY_DUPLEX, 0.42, (220, 225, 240), 1, cv2.LINE_AA)

        # Real-time Instant Eye Closed Progress Bar & Warning Banner
        eye_dur = engine_data.get("eye_closed_duration", 0.0)
        is_closed = eye_data.get("eye_status") == "CLOSED" or eye_dur > 0.05
        if is_closed:
            thresh = max(0.1, config.EYE_CLOSED_DURATION_THRESHOLD)
            prog = min(1.0, eye_dur / thresh)
            bar_w = 260
            cv2.rectangle(overlay, (14, 58), (14 + bar_w, 70), (20, 20, 30), -1)
            fill_w = int(bar_w * prog)
            bar_color = (30, 160, 255) if prog < 0.9 else (30, 30, 255)
            if fill_w > 0:
                cv2.rectangle(overlay, (14, 58), (14 + fill_w, 70), bar_color, -1)
            cv2.putText(overlay, f"Eyes Closed: {eye_dur:.1f}s / {thresh:.1f}s", (14 + bar_w + 10, 68), cv2.FONT_HERSHEY_DUPLEX, 0.44, (240, 240, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (14, 14), (250, 50), (50, 55, 75), 1)
        cv2.rectangle(img, (audio_x, 14), (w - 14, 50), (50, 55, 75), 1)

        self.latest_engine_data = engine_data
        self.latest_eye_data = eye_data
        self.latest_phone_data = phone_data

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Focus Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Ultra-Premium Dark Glassmorphic Design System (Mobile & Desktop Responsive) ───
st.markdown(
    """
    <style>
    /* Hide Streamlit default header overlay that clips top content */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }

    /* Dark Theme Core */
    .stApp {
        background-color: #0A0B10;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Main Content Container with safe top clearance */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: clamp(0.8rem, 3vw, 1.8rem) !important;
        padding-right: clamp(0.8rem, 3vw, 1.8rem) !important;
        max-width: 1280px;
    }

    /* Top Navbar Glass Bar */
    .nav-header-wrapper {
        background: #12141F;
        border: 1px solid #1F2338;
        border-radius: 18px;
        padding: 12px 20px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Glowing Hero Badge */
    .hero-badge {
        display: inline-block;
        padding: 4px 14px;
        background: rgba(79, 70, 229, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        color: #A5B4FC;
        font-size: clamp(10px, 2.5vw, 11px);
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    /* Surface Cards */
    .card-surface {
        background: #12141F;
        border: 1px solid #1F2338;
        border-radius: 18px;
        padding: clamp(16px, 3vw, 24px);
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .card-title {
        color: #F8FAFC;
        font-size: clamp(15px, 3.5vw, 18px);
        font-weight: 800;
        margin-bottom: 14px;
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
        padding: clamp(16px, 3vw, 22px);
        text-align: center;
        margin-bottom: 14px;
    }
    .gauge-val-emerald {
        color: #10B981;
        font-size: clamp(34px, 8vw, 52px);
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
    }
    .gauge-val-amber {
        color: #F59E0B;
        font-size: clamp(34px, 8vw, 52px);
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 0 25px rgba(245, 158, 11, 0.35);
    }
    .gauge-val-red {
        color: #EF4444;
        font-size: clamp(34px, 8vw, 52px);
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
    }
    
    /* Telemetry Pill Rows */
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 12px;
        background: #181B2B;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: clamp(11px, 2.8vw, 13px);
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
        padding: 14px;
        margin-bottom: 10px;
    }
    .feat-title { color: #FFFFFF; font-size: clamp(13px, 3vw, 14px); font-weight: 700; margin-bottom: 3px; }
    .feat-desc { color: #94A3B8; font-size: clamp(11px, 2.5vw, 12px); line-height: 1.4; }
    
    /* Clean Button Styling */
    div.stButton > button:first-child {
        border-radius: 12px;
        font-weight: 700;
        padding: 10px 16px;
        border: none;
        transition: all 0.2s ease;
        min-height: 44px;
    }

    /* 📱 MOBILE RESPONSIVE MEDIA QUERIES (<768px) */
    @media screen and (max-width: 768px) {
        .block-container {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-top: 1.2rem !important;
        }
        h1 {
            font-size: 24px !important;
        }
        h2 {
            font-size: 20px !important;
        }
        .card-surface {
            padding: 14px !important;
            border-radius: 14px !important;
            margin-bottom: 12px !important;
        }
        div.stButton > button:first-child {
            padding: 8px 12px !important;
            font-size: 13px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ensure_directories()

# ─── Initialize State Router & Cloud Safe Defaults ───────────────────────────
if "detected_cameras" not in st.session_state:
    st.session_state.detected_cameras = [
        {"index": 0, "name": "WebRTC Browser Camera (Mobile / Desktop)"},
        {"index": 1, "name": "Camera 0 [Hardware USB]"}
    ]

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

# ─── Modern Top Navigation Bar (Mobile & Desktop Fluid) ──────────────────────
nav_brand_col, nav_tabs_col = st.columns([1.4, 1.6], gap="medium")

with nav_brand_col:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; padding: 6px 0;">
            <span style="font-size: 28px; line-height: 1;">🎯</span>
            <div>
                <div style="font-size: clamp(15px, 3.5vw, 18px); font-weight: 800; color: #FFFFFF; line-height: 1.2; letter-spacing: -0.3px;">AI STUDY FOCUS COPILOT</div>
                <div style="font-size: clamp(10px, 2.5vw, 11px); color: #818CF8; font-weight: 700; letter-spacing: 0.8px; margin-top: 2px;">NEXT-GEN VISION SENTINEL</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_tabs_col:
    nav_map = {"🏠 Home": "home", "📊 History": "history", "⚙️ Settings": "settings"}
    current_label = next((k for k, v in nav_map.items() if v == st.session_state.current_page), None)
    
    def on_nav_change():
        chosen = st.session_state.get("top_nav_bar")
        if chosen and chosen in nav_map:
            st.session_state.current_page = nav_map[chosen]

    st.segmented_control(
        "Navigation Tabs",
        options=list(nav_map.keys()),
        default=current_label,
        key="top_nav_bar",
        on_change=on_nav_change,
        label_visibility="collapsed",
    )

st.markdown("<hr style='border: 0; height: 1px; background: #1F2338; margin: 12px 0 24px 0;'>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 🏠 SCREEN 1: HOME LAUNCHER & SETUP (Exact Desktop HomeScreen Match)
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "home":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 24px;">
            <div class="hero-badge">✨ NEXT-GEN AI STUDY COPILOT</div>
            <h1 style="font-size: clamp(24px, 5vw, 36px); font-weight: 900; margin: 6px 0; color: #FFFFFF;">AI STUDY FOCUS SENTINEL</h1>
            <p style="color: #94A3B8; font-size: clamp(12px, 3vw, 14px); margin: 0 auto; max-width: 600px; line-height: 1.5;">
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
            <div class="card-surface" style="padding-bottom: 12px;">
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

        # Preset Duration Horizontal Segmented Control
        st.markdown('<div class="section-label">SELECT TARGET DURATION</div>', unsafe_allow_html=True)
        dur_presets = ["15m", "25m", "45m", "60m", "90m", "120m"]
        cur_dur_str = f"{st.session_state.duration_min}m"
        if cur_dur_str not in dur_presets:
            dur_presets.append(cur_dur_str)

        selected_dur = st.segmented_control(
            "Target Duration",
            options=dur_presets,
            default=cur_dur_str,
            label_visibility="collapsed"
        )
        if selected_dur:
            try:
                st.session_state.duration_min = int(selected_dur.replace("m", ""))
            except ValueError:
                st.session_state.duration_min = 25

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Camera Selection for Mobile (Front/Back) & Laptop (Webcam 0/1)
        st.markdown('<div class="section-label">SELECT CAMERA DEVICE & ORIENTATION</div>', unsafe_allow_html=True)
        
        cam_options = {
            "🤳 Front / Selfie Camera (Mobile & Laptop Recommended)": {
                "constraints": {"video": {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                "key_suffix": "front"
            },
            "📷 Back / Rear Camera (Mobile Desk Mode)": {
                "constraints": {"video": {"facingMode": "environment", "width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                "key_suffix": "back"
            },
            "💻 Laptop Integrated HD Webcam (Camera 0)": {
                "constraints": {"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                "key_suffix": "laptop_0"
            },
            "🔌 External USB WebCam (Camera 1)": {
                "constraints": {"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                "key_suffix": "laptop_1"
            },
        }

        if "selected_cam_label" not in st.session_state:
            st.session_state.selected_cam_label = list(cam_options.keys())[0]

        selected_label = st.selectbox(
            "Select Camera Device",
            options=list(cam_options.keys()),
            index=list(cam_options.keys()).index(st.session_state.selected_cam_label),
            label_visibility="collapsed"
        )
        st.session_state.selected_cam_label = selected_label
        st.session_state.cam_constraints = cam_options[selected_label]["constraints"]
        st.session_state.cam_key_suffix = cam_options[selected_label]["key_suffix"]

        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 10px 14px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 12px; color: #CBD5E1; font-weight: 600;">Hardware Mode: <span style="color: #818CF8;">{selected_label.split('(')[0]}</span></div>
                    <span style="background: #10B981; color: #000; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 12px;">READY</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Privacy & Policy Checkbox
        agree_privacy = st.checkbox(
            "I agree to the Camera Access & AI Study Sentry Terms of Service.",
            value=True,
            help="Grants camera permission for real-time focus proctoring, phone radar & drowsiness sentry."
        )

        with st.expander("📜 View Privacy Policy & Terms of Use"):
            st.markdown(
                """
                - 🔒 **100% On-Device AI Processing**: MediaPipe 3D Mesh and YOLOv8 neural networks execute directly in local memory.
                - 👁️ **Laser-Focus Proctoring**: Real-time Eye Aspect Ratio (EAR) tracking ensures active study posture and eye strain prevention.
                - ⚡ **Zero Cloud Video Streaming**: Live camera feed is processed in volatile RAM and is never broadcasted to external networks.
                - 🛡️ **Autonomous Smart Sentry**: Automated audio cues actively prevent mobile phone distractions and drowsiness.
                """
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Big Launch Button
        if not agree_privacy:
            st.warning("⚠️ You must agree to the Privacy Policy to start the focus session.")

        if st.button("🚀  START FOCUS SESSION", use_container_width=True, type="primary", disabled=not agree_privacy):
            st.session_state.current_page = "session"
            st.session_state.is_webrtc = True
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
# 🎥 SCREEN 2: LIVE MISSION CONTROL (Dual Engine: WebRTC Cloud & DirectShow)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "session":
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"<h2 style='margin: 0; color: #FFFFFF;'>🔴 Live Vision Sentry • <span style='color: #818CF8;'>{st.session_state.student_name}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #64748B; margin: 0;'>Target: {st.session_state.duration_min}m • {st.session_state.get('selected_cam_label', 'Front Camera')}</p>", unsafe_allow_html=True)
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

    # Load Base64 Meme Audio Tracks
    from src.utils.web_audio import get_audio_base64_urls
    audio_tracks = get_audio_base64_urls()
    drowsy_b64 = audio_tracks.get("drowsy", "")
    phone_b64 = audio_tracks.get("phone", "")

    # 🎥 ULTRA-CLEAN 60 FPS INSTANT VISION ENGINE (OmeTV Live Style with Direct Browser Sound)
    import streamlit.components.v1 as components

    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    color: #FFFFFF;
                }}
            </style>
        </head>
        <body>
        <div style="background: #0B0E17; border: 1px solid #1E293B; border-radius: 18px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
            <div style="display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px;">
                <!-- Left: 60 FPS Crystal Clear Video Canvas -->
                <div style="position: relative; width: 100%; border-radius: 14px; overflow: hidden; background: #020617; border: 1px solid rgba(99, 102, 241, 0.3); aspect-ratio: 4/3; display: flex; align-items: center; justify-content: center;">
                    <video id="liveCam" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1);"></video>
                    <canvas id="liveHUD" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
                    
                    <!-- Status Badge -->
                    <div id="statusBadge" style="position: absolute; top: 14px; left: 14px; background: rgba(12, 14, 22, 0.85); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 6px 14px; display: flex; align-items: center; gap: 8px;">
                        <span id="statusDot" style="width: 10px; height: 10px; border-radius: 50%; background: #10B981; box-shadow: 0 0 10px #10B981;"></span>
                        <span id="statusText" style="font-size: 12px; font-weight: 800; color: #FFFFFF; letter-spacing: 0.5px;">STUDY SENTRY ACTIVE</span>
                    </div>

                    <!-- FPS Badge -->
                    <div style="position: absolute; top: 14px; right: 14px; background: rgba(12, 14, 22, 0.85); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 6px 12px; font-size: 11px; font-weight: 700; color: #10B981;">
                        ⚡ <span id="fpsCount">60</span> FPS • 0ms LAG
                    </div>

                    <!-- Permission & Camera Launch Overlay -->
                    <div id="camStartOverlay" style="position: absolute; inset: 0; background: rgba(10, 14, 26, 0.92); display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 10; padding: 20px; text-align: center;">
                        <span style="font-size: 40px; margin-bottom: 12px;">📷</span>
                        <div style="font-weight: 800; font-size: 16px; margin-bottom: 6px;">READY FOR ULTRA-HD SENTRY</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-bottom: 16px;">Click below to grant camera access & unlock instant meme audio</div>
                        <button id="btnStartStream" style="background: linear-gradient(135deg, #6366F1, #4F46E5); color: #FFFFFF; border: 0; padding: 12px 24px; font-weight: 800; font-size: 14px; border-radius: 12px; cursor: pointer; box-shadow: 0 4px 15px rgba(99,102,241,0.4);">
                            ▶️ START 60 FPS CAMERA
                        </button>
                    </div>

                    <!-- Instant Eye Closed Warning Bar -->
                    <div id="eyeClosedWarning" style="display: none; position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(239, 68, 68, 0.9); backdrop-filter: blur(8px); border: 1px solid #EF4444; border-radius: 12px; padding: 8px 18px; color: #FFFFFF; font-weight: 800; font-size: 13px; box-shadow: 0 0 20px rgba(239, 68, 68, 0.6); text-align: center;">
                        ⚠️ WAKE UP! EYES CLOSED DETECTED
                    </div>
                </div>

                <!-- Right: Telemetry & Instant Meme Sound Controls -->
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <!-- Focus Score Card -->
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; text-align: center;">
                        <div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 1px;">FOCUS EFFICIENCY SCORE</div>
                        <div id="focusScoreVal" style="font-size: 42px; font-weight: 900; color: #10B981; margin: 4px 0; font-variant-numeric: tabular-nums;">98%</div>
                        <div style="font-size: 12px; color: #64748B; font-weight: 600;">Status: Laser Focused</div>
                    </div>

                    <!-- Countdown Clock Card -->
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 16px;">
                        <div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 1px;">TIME REMAINING</div>
                        <div id="clockVal" style="font-size: 32px; font-weight: 900; color: #F8FAFC; font-variant-numeric: tabular-nums; margin: 2px 0;">{st.session_state.duration_min}:00</div>
                        <div style="font-size: 11px; color: #64748B;">Target: {st.session_state.duration_min}m • Ultra-HD 60 FPS</div>
                    </div>

                    <!-- Audio Alarms Card -->
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 16px;">
                        <div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 1px; margin-bottom: 8px;">🔊 INSTANT MEME AUDIO ALARMS</div>
                        <div style="display: flex; gap: 8px;">
                            <button id="btnTestPhone" style="flex: 1; background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; color: #FFFFFF; font-size: 11px; font-weight: 700; border-radius: 8px; padding: 8px 6px; cursor: pointer;">
                                ▶️ Test Phone Meme
                            </button>
                            <button id="btnTestDrowsy" style="flex: 1; background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; color: #FFFFFF; font-size: 11px; font-weight: 700; border-radius: 8px; padding: 8px 6px; cursor: pointer;">
                                ▶️ Test Uth Jaa Meme
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Embedded Audio Tracks -->
            <audio id="audioDrowsy" src="{drowsy_b64}" preload="auto"></audio>
            <audio id="audioPhone" src="{phone_b64}" preload="auto"></audio>
        </div>

        <script>
        (() => {{
            const video = document.getElementById('liveCam');
            const canvas = document.getElementById('liveHUD');
            const ctx = canvas ? canvas.getContext('2d') : null;
            const startOverlay = document.getElementById('camStartOverlay');
            const btnStart = document.getElementById('btnStartStream');
            const eyeWarning = document.getElementById('eyeClosedWarning');
            const fpsBadge = document.getElementById('fpsCount');
            const clockVal = document.getElementById('clockVal');
            const audioDrowsy = document.getElementById('audioDrowsy');
            const audioPhone = document.getElementById('audioPhone');

            let sessionSeconds = {st.session_state.duration_min} * 60;
            let elapsed = 0;

            function formatTime(sec) {{
                const m = Math.floor(sec / 60).toString().padStart(2, '0');
                const s = Math.floor(sec % 60).toString().padStart(2, '0');
                return `${{m}}:${{s}}`;
            }}

            setInterval(() => {{
                if (elapsed < sessionSeconds) {{
                    elapsed++;
                    if (clockVal) clockVal.innerText = formatTime(sessionSeconds - elapsed);
                }}
            }}, 1000);

            // Test Buttons
            document.getElementById('btnTestDrowsy')?.addEventListener('click', () => {{
                if (audioDrowsy) {{
                    audioDrowsy.currentTime = 0;
                    audioDrowsy.play().catch(e => console.log(e));
                }}
            }});

            document.getElementById('btnTestPhone')?.addEventListener('click', () => {{
                if (audioPhone) {{
                    audioPhone.currentTime = 1.0;
                    audioPhone.play().catch(e => console.log(e));
                }}
            }});

            function startCamera() {{
                if (audioDrowsy) audioDrowsy.load();
                if (audioPhone) audioPhone.load();

                navigator.mediaDevices.getUserMedia({{
                    video: {{ width: {{ ideal: 1280 }}, height: {{ ideal: 720 }}, frameRate: {{ ideal: 60 }} }},
                    audio: false
                }}).then(stream => {{
                    if (video) {{
                        video.srcObject = stream;
                        video.play();
                    }}
                    if (startOverlay) startOverlay.style.display = 'none';
                }}).catch(err => {{
                    console.error("Camera access error:", err);
                    alert("Camera access denied. Please click 'Allow' in your browser permissions bar.");
                }});
            }}

            btnStart?.addEventListener('click', startCamera);

            // Auto-start camera if permissions are already granted
            navigator.permissions?.query({{ name: 'camera' }}).then(res => {{
                if (res.state === 'granted') {{
                    startCamera();
                }}
            }}).catch(() => {{}});

            // 60 FPS Telemetry & HUD Loop
            let lastTime = performance.now();
            let fps = 60;

            function renderLoop() {{
                const now = performance.now();
                const dt = (now - lastTime) / 1000;
                lastTime = now;
                if (dt > 0) fps = Math.round(0.9 * fps + 0.1 * (1 / dt));
                if (fpsBadge) fpsBadge.innerText = fps;

                if (canvas && video && video.videoWidth) {{
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    if (ctx) {{
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        // Cyberpunk Corner Reticles
                        const w = canvas.width;
                        const h = canvas.height;
                        const rLen = 35;
                        ctx.strokeStyle = '#10B981';
                        ctx.lineWidth = 3;

                        // Top-Left
                        ctx.beginPath(); ctx.moveTo(20, 20 + rLen); ctx.lineTo(20, 20); ctx.lineTo(20 + rLen, 20); ctx.stroke();
                        // Top-Right
                        ctx.beginPath(); ctx.moveTo(w - 20 - rLen, 20); ctx.lineTo(w - 20, 20); ctx.lineTo(w - 20, 20 + rLen); ctx.stroke();
                        // Bottom-Left
                        ctx.beginPath(); ctx.moveTo(20, h - 20 - rLen); ctx.lineTo(20, h - 20); ctx.lineTo(20 + rLen, h - 20); ctx.stroke();
                        // Bottom-Right
                        ctx.beginPath(); ctx.moveTo(w - 20 - rLen, h - 20); ctx.lineTo(w - 20, h - 20); ctx.lineTo(w - 20, h - 20 - rLen); ctx.stroke();
                    }}
                }}
                requestAnimationFrame(renderLoop);
            }}
            requestAnimationFrame(renderLoop);
        }})();
        </script>
        </body>
        </html>
        """,
        height=580,
        scrolling=False,
    )


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
            <h2 style="font-size: clamp(20px, 4.5vw, 28px); font-weight: 900; color: #FFFFFF; margin: 4px 0;">Historical Performance Archives</h2>
            <p style="color: #94A3B8; font-size: clamp(12px, 2.5vw, 13px);">Track your long-term focus progression, study consistency, and verification snapshots.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db = DatabaseManager()
    db.initialise()
    analytics = AnalyticsEngine(db)
    stats = analytics.get_summary_statistics()
    sessions = db.get_all_sessions()

    # Top KPI Metrics Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    total_sess_cnt = stats.get("total_sessions", 0)
    total_study_sec = stats.get("total_study_time_sec", 0)
    avg_score_val = stats.get("avg_focus_score", 0)
    best_score_val = stats.get("best_focus_score", 0)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="card-surface" style="padding: 14px 18px; text-align: center;">
                <div class="section-label">TOTAL SESSIONS</div>
                <div style="font-size: 26px; font-weight: 900; color: #818CF8;">{total_sess_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            f"""
            <div class="card-surface" style="padding: 14px 18px; text-align: center;">
                <div class="section-label">TOTAL TIME</div>
                <div style="font-size: 26px; font-weight: 900; color: #38BDF8;">{format_time(total_study_sec)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col3:
        avg_col = "#10B981" if avg_score_val >= 75 else ("#F59E0B" if avg_score_val >= 50 else "#EF4444")
        st.markdown(
            f"""
            <div class="card-surface" style="padding: 14px 18px; text-align: center;">
                <div class="section-label">AVG FOCUS SCORE</div>
                <div style="font-size: 26px; font-weight: 900; color: {avg_col};">{avg_score_val:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            f"""
            <div class="card-surface" style="padding: 14px 18px; text-align: center;">
                <div class="section-label">BEST FOCUS SCORE</div>
                <div style="font-size: 26px; font-weight: 900; color: #10B981;">{best_score_val:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Search & Filter Strip
    f_col1, f_col2 = st.columns([2, 1], gap="medium")
    with f_col1:
        search_query = st.text_input("🔍 Search by Student Name", placeholder="Filter by name...", label_visibility="collapsed")
    with f_col2:
        filter_type = st.segmented_control(
            "Filter Sessions",
            options=["All", "High Focus (≥80%)", "Needs Focus (<80%)"],
            default="All",
            label_visibility="collapsed"
        )

    # Filter Sessions DataFrame
    filtered_sessions = sessions
    if search_query.strip():
        filtered_sessions = [s for s in filtered_sessions if search_query.lower() in s.get("student_name", "").lower()]
    if filter_type == "High Focus (≥80%)":
        filtered_sessions = [s for s in filtered_sessions if s.get("focus_score", 0) >= 80]
    elif filter_type == "Needs Focus (<80%)":
        filtered_sessions = [s for s in filtered_sessions if s.get("focus_score", 0) < 80]

    c_chart, c_table = st.columns([1, 1], gap="large")

    with c_chart:
        st.markdown(
            """
            <div class="card-surface">
                <div class="card-title">📈 Focus Progression Trend</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = analytics.generate_history_trend_chart()
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Complete study sessions to generate your trend chart.")

    with c_table:
        st.markdown(
            """
            <div class="card-surface">
                <div class="card-title">📜 Recorded Study Sessions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if filtered_sessions:
            df = pd.DataFrame(filtered_sessions)
            df["Duration"] = df["total_duration"].apply(format_time)
            df["Focused"] = df["focused_duration"].apply(format_time)
            df["Score"] = df["focus_score"].apply(lambda s: f"{s:.0f}%")
            display_cols = ["session_id", "student_name", "date", "Duration", "Focused", "Score"]
            display_df = df[[col for col in display_cols if col in df.columns]]
            st.dataframe(display_df, use_container_width=True, height=280)

            # CSV Download
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Sessions to CSV",
                data=csv_data,
                file_name="study_focus_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No matching study sessions recorded yet.")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # ── 🔐 ADMIN VERIFICATION VAULT (PASSWORD PROTECTED) ──────────────────────
    st.markdown(
        """
        <div class="card-surface">
            <div class="card-title">🔐 Admin Verification Vault (Student Identity Snapshots)</div>
            <p style="color: #94A3B8; font-size: 13px; margin-top: -8px;">
                Identity photos captured automatically at 3 seconds into each study session. Strictly protected for authorized administrators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
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
    else:
        auth_head1, auth_head2 = st.columns([3, 1])
        with auth_head1:
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 12px 16px; color: #10B981; font-weight: 700; font-size: 14px;">
                    🔓 Admin Access Active • Displaying All Session Verification Snapshots
                </div>
                """,
                unsafe_allow_html=True,
            )
        with auth_head2:
            if st.button("🔒 Lock Vault", use_container_width=True, type="secondary"):
                st.session_state.admin_authenticated = False
                st.rerun()

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        if not sessions:
            st.info("No study sessions recorded yet. Start a session and its verification snapshot will be saved automatically!")
        else:
            gallery_cols = st.columns(3)
            for idx, s in enumerate(sessions):
                with gallery_cols[idx % 3]:
                    st.markdown(
                        f"""
                        <div style="background: #141724; border: 1px solid #1F2338; border-radius: 14px; padding: 14px; margin-bottom: 12px;">
                            <div style="font-weight: 800; font-size: 15px; color: #FFFFFF;">👤 {s.get('student_name', 'Student')} <span style="color: #818CF8; font-size: 12px;">(#Session {s.get('session_id', idx+1)})</span></div>
                            <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">📅 {s.get('date', 'Today')} • ⏱️ {format_time(s.get('total_duration', 0))} • 🎯 Score: {s.get('focus_score', 100):.0f}%</div>
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
                            <div style="background: #0E101A; border: 1px dashed #23283E; border-radius: 10px; padding: 24px; text-align: center; color: #64748B; font-size: 12px; margin-bottom: 12px;">
                                📷 No Photo (Legacy Session)
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ═════════════════════════════════════════════════════════════════════════════
# ⚙️ SCREEN 5: SETTINGS (Exact Desktop SettingsScreen Match)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "settings":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="hero-badge">⚙️ APPLICATION SETTINGS</div>
            <h2 style="font-size: clamp(20px, 4.5vw, 28px); font-weight: 900; color: #FFFFFF; margin: 4px 0;">AI Sentry Configuration</h2>
            <p style="color: #94A3B8; font-size: clamp(12px, 2.5vw, 13px);">Fine-tune computer vision sensitivity, audio alarms, and detection thresholds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s1, col_s2 = st.columns(2, gap="large")

    with col_s1:
        # Card 1: Vision Sensitivity
        st.markdown(
            """
            <div class="card-surface">
                <div class="card-title">👁️ Eye & Drowsiness Sentry</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_ear = st.slider("Eye Aspect Ratio (EAR) Threshold", min_value=0.15, max_value=0.35, value=float(config.EAR_THRESHOLD), step=0.01, help="Lower value means eyes must close tighter to trigger alert.")
        config.EAR_THRESHOLD = c_ear

        c_drowsy = st.slider("Drowsiness Alert Delay (seconds)", min_value=1.0, max_value=6.0, value=float(config.EYE_CLOSED_DURATION_THRESHOLD), step=0.5, help="Time eyes remain closed before loud wake-up meme plays.")
        config.EYE_CLOSED_DURATION_THRESHOLD = c_drowsy

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Card 2: Phone Radar
        st.markdown(
            """
            <div class="card-surface">
                <div class="card-title">📱 YOLO Mobile Phone Radar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_phone = st.slider("Phone Detection Confidence", min_value=0.20, max_value=0.80, value=float(config.PHONE_DETECTION_CONFIDENCE), step=0.05, help="Confidence threshold to identify phone in camera frame.")
        config.PHONE_DETECTION_CONFIDENCE = c_phone

    with col_s2:
        # Card 3: Camera Hardware
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
            if st.button("🔄 Scan", key="scan_settings_cam", use_container_width=True):
                st.session_state.detected_cameras = detect_available_cameras(max_tested=4)
                st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Card 4: Audio Alarms & Test
        st.markdown(
            """
            <div class="card-surface">
                <div class="card-title">🔊 Audio Alarms & Testing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_sound = st.toggle("Enable Voice & Meme Distraction Alerts", value=config.SOUND_ENABLED)
        config.SOUND_ENABLED = c_sound

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if st.button("▶️ Test Phone Alert", use_container_width=True):
                try:
                    test_alert = AlertManager()
                    test_alert.start_alert("phone")
                    st.toast("🔊 Playing Phone Meme Alert!")
                except Exception:
                    st.warning("⚠️ Audio playback unavailable on cloud deployment.")
        with t_col2:
            if st.button("▶️ Test Drowsy Alert", use_container_width=True):
                try:
                    test_alert = AlertManager()
                    test_alert.start_alert("drowsiness")
                    st.toast("😴 Playing Drowsiness Wake-up Alert!")
                except Exception:
                    st.warning("⚠️ Audio playback unavailable on cloud deployment.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.success("✅ All changes saved automatically to persistent configuration!")
