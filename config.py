"""
AI Study Focus & Distraction Detector — Configuration
=====================================================

All application-wide constants and default settings live here.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_ROOT / "audio"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"
SNAPSHOTS_DIR = DATABASE_DIR / "snapshots"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATABASE_PATH = DATABASE_DIR / "study_focus.db"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

ADMIN_PASSWORD = "Aman-Rajbhar-2005"

# Audio alert source files
DROWSINESS_ALERT_AUDIO = AUDIO_DIR / "uth jaa - EXTRA UNGLI WAALA (128k).mp3"
PHONE_ALERT_AUDIO = AUDIO_DIR / "Rakh Teri Maa Ki Rakh Meme Template [7yF953fKPWA].mp3"
PHONE_ALERT_START_OFFSET_SEC = 1.0  # Starts from 0:01 sec

CUSTOM_ALERT_AUDIO = DROWSINESS_ALERT_AUDIO

# ─── Camera ───────────────────────────────────────────────────────────────────

DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ─── Eye Detection & Drowsiness ───────────────────────────────────────────────

EAR_THRESHOLD = 0.26
EYE_BLINK_THRESHOLD = 0.35

# Continuous duration (seconds) eyes must remain closed before Drowsiness Alert
EYE_CLOSED_DURATION_THRESHOLD = 3.0  # Exactly 3.0s as configured

# Debounce: Seconds of continuous OPEN eyes required to reset closed timer (ignores 1-frame noise)
EYE_OPEN_DEBOUNCE_SEC = 0.20  # Fast 0.20s recovery

FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# ─── Phone & Remote Detection ────────────────────────────────────────────────

PHONE_CLASS_ID = 67       # COCO cell phone
REMOTE_CLASS_ID = 65      # COCO remote

PHONE_CLASS_NAME = "cell phone"

# Strict threshold (0.60) to detect ONLY actual mobile phones and reject bare hands/fingers
PHONE_DETECTION_CONFIDENCE = 0.60

# Max allowed skin ratio in bounding box (if >45% is skin, it's fingers/palm, NOT a phone)
PHONE_MAX_SKIN_RATIO = 0.45

# Minimum confidence to recognize and suppress remotes
REMOTE_SUPPRESSION_CONFIDENCE = 0.50

# Require 2 consecutive frames to confirm phone detection (filters out hand movements/flicker)
PHONE_CONFIRMATION_FRAMES = 2

# Lost frame persistence: keep phone active for ~0.4s to prevent rapid audio chatter
PHONE_LOST_FRAMES = 10

# Aspect ratio bounds for real smartphones (standard phone aspect ratios: 1.3 - 2.2)
PHONE_MIN_ASPECT_RATIO = 1.0
PHONE_MAX_ASPECT_RATIO = 2.8

# Area bounds relative to total frame area (prevents tiny slivers/fingers)
PHONE_MIN_AREA_RATIO = 0.010
PHONE_MAX_AREA_RATIO = 0.90

# Minimum physical pixel constraints for webcam (prevents narrow finger crops)
PHONE_MIN_DIMENSION_PX = 40
PHONE_MIN_LONGEST_SIDE_PX = 70

# ─── Distraction Engine ──────────────────────────────────────────────────────

FOCUS_SCORE_WEIGHTS = {
    "phone_penalty_per_second": 0.5,
    "drowsiness_penalty_per_second": 0.3,
    "phone_event_penalty": 2.0,
    "drowsiness_event_penalty": 1.5,
}

# ─── Audio Alert System ──────────────────────────────────────────────────────

SOUND_ENABLED = True

SOUND_DROWSINESS = "drowsiness_alert.wav"
SOUND_PHONE = "phone_alert.wav"
SOUND_HIGH_DISTRACTION = "high_distraction_alert.wav"
SOUND_SESSION_COMPLETE = "session_complete.wav"

AUDIO_ALERT_RETRIGGER_COOLDOWN = 0.5

# ─── Session Defaults ────────────────────────────────────────────────────────

DEFAULT_SESSION_MINUTES = 25
MIN_SESSION_MINUTES = 1
MAX_SESSION_MINUTES = 480

# ─── Detection Interval ──────────────────────────────────────────────────────

# Run YOLO inference every frame (1 = instant responsiveness)
YOLO_DETECTION_INTERVAL = 1


# ─── Debug Logging ───────────────────────────────────────────────────────────

DEBUG_LOGGING = True

# ─── UI / Theme ──────────────────────────────────────────────────────────────

APP_NAME = "AI Study Focus"
APP_VERSION = "2.1.0"

DEFAULT_THEME = "dark"
DEFAULT_COLOR_THEME = "blue"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

# ─── Privacy ──────────────────────────────────────────────────────────────────

PRIVACY_NOTICE = (
    "🔒 Camera processing happens locally. "
    "No video is uploaded or stored."
)
