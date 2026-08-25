"""
Real-Time Session Monitoring Engine
===================================

Runs webcam capture, computer vision detection pipelines,
state machine synchronization, and audio alerts on a dedicated background thread.
"""

import time
import threading
import cv2
import numpy as np
import config
from src.detection.face_detector import FaceMeshDetector
from src.detection.eye_detector import EyeDetector
from src.detection.phone_detector import PhoneDetector
from src.detection.distraction_engine import DistractionEngine, DistractionState
from src.utils.timer import SessionTimer
from src.utils.audio import AlertManager


class SessionMonitor:
    """
    Coordinates real-time vision pipelines with background execution.
    """

    def __init__(
        self,
        student_name: str,
        duration_seconds: float,
        on_frame_update=None,
        on_status_update=None,
        on_session_complete=None,
        camera_index: int = config.DEFAULT_CAMERA_INDEX,
    ) -> None:
        self.student_name = student_name
        self.duration_seconds = duration_seconds
        self.camera_index = camera_index

        # Callbacks
        self.on_frame_update = on_frame_update
        self.on_status_update = on_status_update
        self.on_session_complete = on_session_complete

        # Core Components
        self.timer = SessionTimer(duration_seconds)
        self.alert_manager = AlertManager()
        self.face_detector = FaceMeshDetector()
        self.eye_detector = EyeDetector()
        self.phone_detector = PhoneDetector()
        self.engine = DistractionEngine()

        # Threading & Control
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._is_running = False

        # Metrics
        self.fps = 0.0
        self.last_frame: np.ndarray | None = None
        self.start_timestamp: str = ""
        self.end_timestamp: str = ""

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._is_running:
            return

        self._is_running = True
        self._stop_event.clear()
        self._pause_event.clear()
        self.start_timestamp = time.strftime("%H:%M:%S")
        self.timer.start()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Pause the monitoring session and stop any active audio."""
        self._pause_event.set()
        self.timer.pause()
        self.alert_manager.stop_all()

    def resume(self) -> None:
        """Resume monitoring after pause."""
        self._pause_event.clear()
        self.timer.resume()

    def stop(self) -> dict:
        """Stop monitoring, release webcam, stop all audio, and return session summary."""
        self._stop_event.set()
        self._pause_event.clear()
        self._is_running = False
        self.timer.stop()
        self.alert_manager.stop_all()
        self.end_timestamp = time.strftime("%H:%M:%S")

        # Build session summary
        summary = {
            "student_name": self.student_name,
            "date": time.strftime("%Y-%m-%d"),
            "start_time": self.start_timestamp,
            "end_time": self.end_timestamp,
            "total_duration": round(self.timer.elapsed, 1),
            "focused_duration": round(self.engine.focused_duration, 1),
            "distracted_duration": round(self.engine.distracted_duration, 1),
            "phone_duration": round(self.engine.phone_duration, 1),
            "drowsiness_duration": round(self.engine.drowsiness_duration, 1),
            "phone_events": self.engine.phone_events,
            "drowsiness_events": self.engine.drowsiness_events,
            "focus_score": round(self.engine.calculate_focus_score(), 1),
        }
        return summary

    def _run_loop(self) -> None:
        """Main worker loop."""
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        frame_count = 0
        fps_start = time.time()
        fps_frames = 0
        phone_result = {"phone_detected": False, "detections": []}

        try:
            while not self._stop_event.is_set():
                # Check timer completion
                if self.timer.is_finished:
                    self.alert_manager.sync_alert_state("session_complete")
                    if self.on_session_complete:
                        summary = self.stop()
                        self.on_session_complete(summary)
                    break

                # Handle pause
                if self._pause_event.is_set():
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(
                        frame, "Camera Initializing / Offline...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )

                # Flip horizontally for natural mirror view
                frame = cv2.flip(frame, 1)
                frame_count += 1
                fps_frames += 1

                # Calculate real FPS
                if time.time() - fps_start >= 1.0:
                    self.fps = fps_frames / (time.time() - fps_start)
                    fps_frames = 0
                    fps_start = time.time()

                # 1. Face & Eye Detection
                results_payload, face_count = self.face_detector.process(frame)
                eye_data = {"eye_status": "UNKNOWN", "avg_ear": 0.0}

                if face_count > 0:
                    eye_data = self.eye_detector.detect_eye_state_from_face(
                        frame, results_payload
                    )
                    frame = self.eye_detector.draw_eye_annotations(frame, eye_data)

                # 2. YOLO Phone Detection (every N frames for responsiveness)
                if frame_count % config.YOLO_DETECTION_INTERVAL == 0:
                    phone_result = self.phone_detector.detect(frame)

                if phone_result.get("phone_detected") and phone_result.get("detections"):
                    frame = self.phone_detector.draw_detections(
                        frame, phone_result["detections"]
                    )

                # 3. Update Distraction Engine
                engine_data = self.engine.update(
                    face_count=face_count,
                    eye_status=eye_data["eye_status"],
                    phone_detected=phone_result["phone_detected"],
                )

                # 4. Synchronize Alert Audio State (State-Based & Instant Stop)
                target_alert = engine_data["target_alert"]
                self.alert_manager.sync_alert_state(target_alert)

                # 5. Render HUD Overlay
                annotated_frame = self._render_hud_overlay(
                    frame, engine_data, eye_data, face_count, self.fps, self.alert_manager.is_playing
                )
                self.last_frame = annotated_frame

                # 6. Dispatch updates to GUI
                if self.on_frame_update:
                    self.on_frame_update(annotated_frame)

                if self.on_status_update:
                    status_payload = {
                        "remaining_time": self.timer.remaining,
                        "elapsed_time": self.timer.elapsed,
                        "progress": self.timer.progress,
                        "current_state": engine_data["current_state"],
                        "eye_status": eye_data["eye_status"],
                        "ear_val": eye_data.get("avg_ear", 0.0),
                        "drowsiness_confirmed": engine_data["drowsiness_confirmed"],
                        "eye_closed_sec": engine_data["eye_closed_duration"],
                        "phone_detected": phone_result["phone_detected"],
                        "audio_active": self.alert_manager.is_playing,
                        "focus_score": engine_data["focus_score"],
                        "distracted_duration": engine_data["distracted_duration"],
                        "phone_events": engine_data["phone_events"],
                        "drowsiness_events": engine_data["drowsiness_events"],
                        "fps": self.fps,
                    }
                    self.on_status_update(status_payload)

                time.sleep(0.015)

        finally:
            cap.release()
            self.alert_manager.stop_all()
            self.face_detector.close()

    def _render_hud_overlay(
        self,
        frame: np.ndarray,
        engine_data: dict,
        eye_data: dict,
        face_count: int,
        fps: float,
        audio_playing: bool,
    ) -> np.ndarray:
        """Render modern glassmorphism HUD badges directly on video stream."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        state = engine_data["current_state"]

        if state == DistractionState.FOCUSED:
            theme_color = (130, 210, 30)   # Vivid Emerald
            state_text = "FOCUSED"
        elif state == DistractionState.PHONE_DISTRACTION:
            theme_color = (40, 60, 240)    # Crimson Red
            state_text = "PHONE DETECTED"
        elif state == DistractionState.DROWSY:
            theme_color = (30, 160, 255)   # Amber Orange
            state_text = "DROWSINESS DETECTED"
        elif state == DistractionState.HIGH_DISTRACTION:
            theme_color = (20, 20, 255)    # Deep Crimson
            state_text = "CRITICAL DISTRACTION"
        elif state == DistractionState.NO_FACE:
            theme_color = (160, 160, 160)  # Slate
            state_text = "NO FACE DETECTED"
        elif state == DistractionState.MULTIPLE_FACES:
            theme_color = (30, 140, 255)   # Amber
            state_text = "MULTIPLE FACES"
        else:
            theme_color = (140, 140, 140)
            state_text = "INITIALIZING"

        # Create overlay for smooth alpha transparency
        overlay = annotated.copy()

        # Top-Left State Pill
        pill_w, pill_h = 240, 36
        cv2.rectangle(overlay, (14, 14), (14 + pill_w, 14 + pill_h), (12, 14, 22), -1)
        # Dot indicator
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
        audio_pill_w = 130
        audio_x = w - audio_pill_w - 14
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

        # Blend overlay (85% opacity for sleek glass feel)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Crisp Borders
        cv2.rectangle(annotated, (14, 14), (14 + pill_w, 14 + pill_h), (50, 55, 75), 1)
        cv2.rectangle(annotated, (audio_x, 14), (w - 14, 14 + pill_h), (50, 55, 75), 1)

        # Eye Closure Warning Timer Bar
        eye_dur = engine_data.get("eye_closed_duration", 0.0)
        if eye_dur > 0 and eye_data.get("eye_status") == "CLOSED":
            progress = min(1.0, eye_dur / max(0.1, config.EYE_CLOSED_DURATION_THRESHOLD))
            bar_w = 220
            bar_h = 8
            bar_x = 14
            bar_y = 58

            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 20, 30), -1)
            fill_w = int(bar_w * progress)
            bar_col = (30, 160, 255) if progress < 0.8 else (30, 40, 240)
            if fill_w > 0:
                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 85), 1)

            cv2.putText(
                annotated,
                f"Eyes Closed: {eye_dur:.1f}s",
                (bar_x + bar_w + 10, bar_y + 7),
                cv2.FONT_HERSHEY_DUPLEX,
                0.40,
                (240, 240, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
