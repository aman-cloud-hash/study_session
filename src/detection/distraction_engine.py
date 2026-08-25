"""
Intelligent Distraction Decision Engine & State Machine
======================================================

Strict timestamp-based temporal tracking, debounce filtering, and priority-based state machine.
Computes the exact active alert target for synchronized audio control.
"""

import time
import config
from src.utils.helpers import clamp


class DistractionState:
    FOCUSED = "FOCUSED"
    DROWSY = "DROWSY"
    PHONE_DISTRACTION = "PHONE DISTRACTION"
    HIGH_DISTRACTION = "HIGH DISTRACTION"
    NO_FACE = "NO FACE DETECTED"
    MULTIPLE_FACES = "MULTIPLE FACES DETECTED"
    UNKNOWN = "INITIALIZING"


class DistractionEngine:
    """
    Central state machine determining distraction state and active audio alert.

    Priority hierarchy:
    1. HIGH DISTRACTION (Phone + Eyes Closed >= 3s)
    2. PHONE DISTRACTION (Phone detected immediately)
    3. DROWSY (Eyes Closed >= 3s continuously)
    4. FOCUSED (Eyes Open, Phone not detected)
    """

    def __init__(
        self,
        eye_closed_threshold_sec: float = config.EYE_CLOSED_DURATION_THRESHOLD,
        eye_open_debounce_sec: float = config.EYE_OPEN_DEBOUNCE_SEC,
    ) -> None:
        self.eye_closed_threshold_sec = eye_closed_threshold_sec
        self.eye_open_debounce_sec = eye_open_debounce_sec

        # State tracking
        self.current_state = DistractionState.UNKNOWN
        self.last_update_ts = time.time()

        # Cumulative durations (in seconds)
        self.total_tracked_time = 0.0
        self.focused_duration = 0.0
        self.distracted_duration = 0.0
        self.phone_duration = 0.0
        self.drowsiness_duration = 0.0

        # Exact timestamp-based eye tracking with debounce
        self.eye_closed_start_time: float | None = None
        self._eye_open_start_time: float | None = None
        self.drowsiness_confirmed = False

        # Event tracking
        self._in_drowsy_event = False
        self._in_phone_event = False
        self.drowsiness_events = 0
        self.phone_events = 0

        # Target alert to synchronize with AlertManager
        self.target_alert: str | None = None

    def reset(self) -> None:
        """Reset all metrics."""
        self.current_state = DistractionState.UNKNOWN
        self.last_update_ts = time.time()
        self.total_tracked_time = 0.0
        self.focused_duration = 0.0
        self.distracted_duration = 0.0
        self.phone_duration = 0.0
        self.drowsiness_duration = 0.0

        self.eye_closed_start_time = None
        self._eye_open_start_time = None
        self.drowsiness_confirmed = False
        self._in_drowsy_event = False
        self._in_phone_event = False

        self.drowsiness_events = 0
        self.phone_events = 0
        self.target_alert = None

    def update(
        self,
        face_count: int,
        eye_status: str,
        phone_detected: bool,
        current_ts: float | None = None,
    ) -> dict:
        """
        Evaluate frame signals and update temporal state machine.
        """
        now = current_ts if current_ts is not None else time.time()
        dt = max(0.0, min(1.0, now - self.last_update_ts))
        self.last_update_ts = now
        self.total_tracked_time += dt

        # ── 1. Timestamp-Based Eye Closed & Drowsiness Tracking with Debounce ──
        is_eyes_closed = (eye_status == "CLOSED")

        if is_eyes_closed and face_count > 0:
            self._eye_open_start_time = None

            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
                if config.DEBUG_LOGGING:
                    self._log("Eyes closed — Observation timer started")

            closed_duration = now - self.eye_closed_start_time

            if closed_duration >= self.eye_closed_threshold_sec:
                if not self.drowsiness_confirmed:
                    self.drowsiness_confirmed = True
                    self.drowsiness_events += 1
                    self._in_drowsy_event = True
                    if config.DEBUG_LOGGING:
                        self._log(f"Eyes closed for {closed_duration:.2f}s — 😴 DROWSINESS CONFIRMED")
                self.drowsiness_duration += dt
            else:
                self.drowsiness_confirmed = False
        else:
            # Eyes observed OPEN: require continuous debounce window before wiping closed timer
            if self._eye_open_start_time is None:
                self._eye_open_start_time = now

            open_duration = now - self._eye_open_start_time

            if open_duration >= self.eye_open_debounce_sec:
                if self.eye_closed_start_time is not None:
                    if config.DEBUG_LOGGING:
                        self._log("Eyes opened — Observation timer reset")
                self.eye_closed_start_time = None
                self.drowsiness_confirmed = False
                self._in_drowsy_event = False

        # ── 2. Phone Tracking ──────────────────────────────────────────────
        if phone_detected:
            self.phone_duration += dt
            if not self._in_phone_event:
                self._in_phone_event = True
                self.phone_events += 1
        else:
            self._in_phone_event = False

        # ── 3. State Machine & Strict Alert Priority ───────────────────────
        if phone_detected and self.drowsiness_confirmed:
            # Priority 1: High Distraction (Phone + Eyes Closed)
            self.current_state = DistractionState.HIGH_DISTRACTION
            self.target_alert = "high_distraction"
            self.distracted_duration += dt
        elif phone_detected:
            # Priority 2: Phone Distraction (Instant even if phone covers face)
            self.current_state = DistractionState.PHONE_DISTRACTION
            self.target_alert = "phone"
            self.distracted_duration += dt
        elif face_count == 0:
            # Priority 3: No Face Detected
            self.current_state = DistractionState.NO_FACE
            self.target_alert = None
            self.distracted_duration += dt
        elif face_count > 1:
            # Priority 4: Multiple Faces
            self.current_state = DistractionState.MULTIPLE_FACES
            self.target_alert = None
            self.distracted_duration += dt
        elif self.drowsiness_confirmed:
            # Priority 5: Drowsiness (Confirmed after 3s)
            self.current_state = DistractionState.DROWSY
            self.target_alert = "drowsiness"
            self.distracted_duration += dt
        else:
            # Priority 6: Focused (All clear)
            self.current_state = DistractionState.FOCUSED
            self.target_alert = None
            self.focused_duration += dt

        # ── 4. Focus Score ─────────────────────────────────────────────────
        focus_score = self.calculate_focus_score()

        return {
            "current_state": self.current_state,
            "target_alert": self.target_alert,
            "drowsiness_confirmed": self.drowsiness_confirmed,
            "eye_closed_duration": (now - self.eye_closed_start_time) if self.eye_closed_start_time else 0.0,
            "focus_score": round(focus_score, 1),
            "focused_duration": self.focused_duration,
            "distracted_duration": self.distracted_duration,
            "phone_duration": self.phone_duration,
            "drowsiness_duration": self.drowsiness_duration,
            "phone_events": self.phone_events,
            "drowsiness_events": self.drowsiness_events,
        }

    def calculate_focus_score(self) -> float:
        """Calculate focus percentage."""
        if self.total_tracked_time <= 0:
            return 100.0

        base_score = (self.focused_duration / self.total_tracked_time) * 100.0
        event_penalty = (self.phone_events * 0.5) + (self.drowsiness_events * 0.5)
        final_score = base_score - event_penalty
        return clamp(final_score, 0.0, 100.0)

    @staticmethod
    def _log(msg: str) -> None:
        now_str = time.strftime("%H:%M:%S")
        print(f"[{now_str}] [DistractionEngine] {msg}")
