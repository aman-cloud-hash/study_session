"""
Live Session Monitoring Screen (Cyberpunk Dark Mission Control)
==============================================================

Displays real-time webcam feed, dynamic status cards, glowing focus score,
countdown timer, and responsive pause/stop controls.
"""

import customtkinter as ctk
import cv2
from PIL import Image
import config
from src.monitoring.session_monitor import SessionMonitor
from src.utils.helpers import format_time


class SessionScreen(ctk.CTkFrame):
    """
    High-tech AI Monitoring Dashboard.
    """

    def __init__(self, parent: ctk.CTkFrame, controller) -> None:
        super().__init__(parent, fg_color="#0A0B10")
        self.controller = controller
        self.monitor: SessionMonitor | None = None
        self._is_paused = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ── Left Column: Video Feed & Controls ─────────────────────────────
        self.left_panel = ctk.CTkFrame(
            self,
            fg_color="#12141F",
            corner_radius=16,
            border_width=1,
            border_color="#1F2338",
        )
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Video Header
        header = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))

        # Live Indicator Pill
        live_pill = ctk.CTkFrame(header, fg_color="#181D2E", corner_radius=12, border_width=1, border_color="#2D3756")
        live_pill.pack(side="left")
        ctk.CTkLabel(
            live_pill,
            text=" 🔴 LIVE VISION SENTRY ",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F87171",
        ).pack(padx=10, pady=3)

        # FPS Pill
        self.fps_pill = ctk.CTkFrame(header, fg_color="#181D2E", corner_radius=12, border_width=1, border_color="#2D3756")
        self.fps_pill.pack(side="right")
        self.fps_lbl = ctk.CTkLabel(
            self.fps_pill,
            text="⚡ FPS: --",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#60A5FA",
        )
        self.fps_lbl.pack(padx=10, pady=3)

        # Video Container
        self.video_container = ctk.CTkFrame(
            self.left_panel,
            fg_color="#08090E",
            corner_radius=12,
            border_width=1,
            border_color="#1E2338",
        )
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=18, pady=6)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_container,
            text="Initializing High-Speed Vision Stream...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#64748B",
        )
        self.video_label.grid(row=0, column=0)

        # Control Bar
        ctrl_bar = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        ctrl_bar.grid(row=2, column=0, sticky="ew", padx=18, pady=(8, 16))

        self.pause_btn = ctk.CTkButton(
            ctrl_bar,
            text="⏸️  Pause Session",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            corner_radius=10,
            fg_color="#1E2338",
            hover_color="#2B324D",
            command=self._toggle_pause,
        )
        self.pause_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.stop_btn = ctk.CTkButton(
            ctrl_bar,
            text="⏹️  End Session",
            fg_color="#DC2626",
            hover_color="#B91C1C",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            corner_radius=10,
            command=self._on_stop,
        )
        self.stop_btn.pack(side="right", fill="x", expand=True, padx=(8, 0))

        # ── Right Column: Telemetry & Metrics Cards ───────────────────────
        self.right_panel = ctk.CTkScrollableFrame(
            self,
            fg_color="#12141F",
            corner_radius=16,
            border_width=1,
            border_color="#1F2338",
        )
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)

        # Student ID Header
        user_header = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        user_header.pack(fill="x", padx=6, pady=(4, 10))

        self.student_lbl = ctk.CTkLabel(
            user_header,
            text="👤 Student: ---",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#F8FAFC",
        )
        self.student_lbl.pack(side="left")

        # Focus Score Hero Card
        score_card = ctk.CTkFrame(
            self.right_panel,
            fg_color="#181B2B",
            corner_radius=14,
            border_width=1,
            border_color="#2A2F45",
        )
        score_card.pack(fill="x", padx=4, pady=6)

        ctk.CTkLabel(
            score_card,
            text="FOCUS EFFICIENCY SCORE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B",
        ).pack(anchor="w", padx=16, pady=(12, 0))

        score_row = ctk.CTkFrame(score_card, fg_color="transparent")
        score_row.pack(fill="x", padx=16, pady=(4, 12))

        self.score_lbl = ctk.CTkLabel(
            score_row,
            text="100%",
            font=ctk.CTkFont(size=38, weight="bold"),
            text_color="#10B981",
        )
        self.score_lbl.pack(side="left")

        self.state_badge = ctk.CTkLabel(
            score_row,
            text="FOCUSED",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#10B981",
        )
        self.state_badge.pack(side="right", pady=8)

        # Countdown Timer Card
        timer_card = ctk.CTkFrame(
            self.right_panel,
            fg_color="#181B2B",
            corner_radius=14,
            border_width=1,
            border_color="#2A2F45",
        )
        timer_card.pack(fill="x", padx=4, pady=6)

        timer_top = ctk.CTkFrame(timer_card, fg_color="transparent")
        timer_top.pack(fill="x", padx=16, pady=(12, 2))

        ctk.CTkLabel(
            timer_top,
            text="TIME REMAINING",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B",
        ).pack(side="left")

        self.elapsed_lbl = ctk.CTkLabel(
            timer_top,
            text="Elapsed: 00:00:00",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8",
        )
        self.elapsed_lbl.pack(side="right")

        self.remaining_lbl = ctk.CTkLabel(
            timer_card,
            text="00:00:00",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="#6366F1",
        )
        self.remaining_lbl.pack(anchor="w", padx=16, pady=2)

        self.progress_bar = ctk.CTkProgressBar(
            timer_card, height=8, corner_radius=4, fg_color="#0F111A", progress_color="#4F46E5"
        )
        self.progress_bar.pack(fill="x", padx=16, pady=(4, 14))
        self.progress_bar.set(0.0)

        # Live Sentry Indicators Card
        grid_card = ctk.CTkFrame(
            self.right_panel,
            fg_color="#181B2B",
            corner_radius=14,
            border_width=1,
            border_color="#2A2F45",
        )
        grid_card.pack(fill="x", padx=4, pady=6)

        ctk.CTkLabel(
            grid_card,
            text="REAL-TIME TELEMETRY",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B",
        ).pack(anchor="w", padx=16, pady=(12, 8))

        grid = ctk.CTkFrame(grid_card, fg_color="transparent")
        grid.pack(fill="x", padx=14, pady=(0, 12))
        grid.grid_columnconfigure((0, 1), weight=1)

        # Eye State
        ctk.CTkLabel(grid, text="👁️ Eye State:", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.eye_lbl = ctk.CTkLabel(
            grid, text="● OPEN", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10B981"
        )
        self.eye_lbl.grid(row=0, column=1, sticky="e", pady=4)

        # Drowsiness Sentry
        ctk.CTkLabel(grid, text="😴 Drowsiness:", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.drowsy_status_lbl = ctk.CTkLabel(
            grid, text="● NORMAL", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10B981"
        )
        self.drowsy_status_lbl.grid(row=1, column=1, sticky="e", pady=4)

        # Mobile Phone Radar
        ctk.CTkLabel(grid, text="📱 Phone Radar:", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.phone_lbl = ctk.CTkLabel(
            grid, text="● CLEAR", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10B981"
        )
        self.phone_lbl.grid(row=2, column=1, sticky="e", pady=4)

        # Audio Speaker
        ctk.CTkLabel(grid, text="🔊 Audio Alert:", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.audio_lbl = ctk.CTkLabel(
            grid, text="● SILENT", font=ctk.CTkFont(size=12, weight="bold"), text_color="#64748B"
        )
        self.audio_lbl.grid(row=3, column=1, sticky="e", pady=4)

        # Distraction Breakdown
        stats_card = ctk.CTkFrame(
            self.right_panel,
            fg_color="#181B2B",
            corner_radius=14,
            border_width=1,
            border_color="#2A2F45",
        )
        stats_card.pack(fill="x", padx=4, pady=6)

        ctk.CTkLabel(
            stats_card,
            text="SESSION VIOLATIONS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B",
        ).pack(anchor="w", padx=16, pady=(12, 8))

        sgrid = ctk.CTkFrame(stats_card, fg_color="transparent")
        sgrid.pack(fill="x", padx=14, pady=(0, 14))
        sgrid.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(sgrid, text="Distracted Time:", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(
            row=0, column=0, sticky="w", pady=3
        )
        self.dist_time_lbl = ctk.CTkLabel(sgrid, text="00:00:00", font=ctk.CTkFont(size=12, weight="bold"), text_color="#F8FAFC")
        self.dist_time_lbl.grid(row=0, column=1, sticky="e", pady=3)

        ctk.CTkLabel(sgrid, text="Phone Violations:", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(
            row=1, column=0, sticky="w", pady=3
        )
        self.phone_evt_lbl = ctk.CTkLabel(sgrid, text="0", font=ctk.CTkFont(size=12, weight="bold"), text_color="#F8FAFC")
        self.phone_evt_lbl.grid(row=1, column=1, sticky="e", pady=3)

        ctk.CTkLabel(sgrid, text="Drowsy Violations:", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(
            row=2, column=0, sticky="w", pady=3
        )
        self.drowsy_evt_lbl = ctk.CTkLabel(sgrid, text="0", font=ctk.CTkFont(size=12, weight="bold"), text_color="#F8FAFC")
        self.drowsy_evt_lbl.grid(row=2, column=1, sticky="e", pady=3)

    # ── Session Lifecycle ─────────────────────────────────────────────────

    def start_session(self, student_name: str, duration_seconds: float, camera_index: int = None) -> None:
        """Initialize and start the live monitoring session."""
        if camera_index is None:
            camera_index = config.DEFAULT_CAMERA_INDEX

        self.student_lbl.configure(text=f"👤 Student: {student_name}")
        self._is_paused = False
        self.pause_btn.configure(text="⏸️  Pause Session")

        self.monitor = SessionMonitor(
            student_name=student_name,
            duration_seconds=duration_seconds,
            camera_index=camera_index,
            on_frame_update=self._on_frame_update,
            on_status_update=self._on_status_update,
            on_session_complete=self._on_session_complete,
        )
        self.monitor.start()

    def _toggle_pause(self) -> None:
        if not self.monitor:
            return
        if self._is_paused:
            self.monitor.resume()
            self._is_paused = False
            self.pause_btn.configure(text="⏸️  Pause Session")
        else:
            self.monitor.pause()
            self._is_paused = True
            self.pause_btn.configure(text="▶️  Resume Session")

    def _on_stop(self) -> None:
        if self.monitor:
            m = self.monitor
            self.monitor = None
            summary = m.stop()
            self._on_session_complete(summary)

    # ── Callbacks (Dispatched safely to Tk main thread) ───────────────────

    def _on_frame_update(self, frame_bgr) -> None:
        """Render webcam frame into CTkLabel."""
        try:
            disp_w = 640
            disp_h = 480
            frame_resized = cv2.resize(frame_bgr, (disp_w, disp_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(disp_w, disp_h))

            self.after(0, lambda: self.video_label.configure(image=imgtk, text=""))
        except Exception:
            pass

    def _on_status_update(self, status: dict) -> None:
        """Update metrics cards."""
        self.after(0, lambda: self._apply_status_update(status))

    def _apply_status_update(self, s: dict) -> None:
        # Remaining & Elapsed
        self.remaining_lbl.configure(text=format_time(s["remaining_time"]))
        self.elapsed_lbl.configure(text=f"Elapsed: {format_time(s['elapsed_time'])}")
        self.progress_bar.set(s["progress"])

        # FPS
        self.fps_lbl.configure(text=f"⚡ FPS: {int(s['fps'])}")

        # Current State
        state = s["current_state"]
        color = "#10B981"  # Emerald
        if "DROWSY" in state or "PHONE" in state:
            color = "#F59E0B"  # Amber
        if "HIGH" in state or "MULTIPLE" in state:
            color = "#EF4444"  # Red
        elif "NO FACE" in state:
            color = "#94A3B8"  # Slate

        self.state_badge.configure(text=state, text_color=color)

        # Focus Score
        score = s["focus_score"]
        self.score_lbl.configure(text=f"{score}%")
        if score >= 80:
            self.score_lbl.configure(text_color="#10B981")
        elif score >= 60:
            self.score_lbl.configure(text_color="#F59E0B")
        else:
            self.score_lbl.configure(text_color="#EF4444")

        # Eye Status
        eye = s["eye_status"]
        eye_closed_sec = s.get("eye_closed_sec", 0.0)
        if eye == "CLOSED":
            self.eye_lbl.configure(text=f"● CLOSED ({eye_closed_sec:.1f}s)", text_color="#EF4444")
        else:
            self.eye_lbl.configure(text="● OPEN", text_color="#10B981")

        # Drowsiness Status
        is_drowsy = s.get("drowsiness_confirmed", False)
        self.drowsy_status_lbl.configure(
            text="● DROWSY" if is_drowsy else "● NORMAL",
            text_color="#EF4444" if is_drowsy else "#10B981",
        )

        # Phone Status
        phone_on = s["phone_detected"]
        self.phone_lbl.configure(
            text="● DETECTED" if phone_on else "● CLEAR",
            text_color="#EF4444" if phone_on else "#10B981",
        )

        # Audio Alert Status
        audio_on = s.get("audio_active", False)
        self.audio_lbl.configure(
            text="● ACTIVE ALERT" if audio_on else "● SILENT",
            text_color="#EF4444" if audio_on else "#64748B",
        )

        # Times & Events
        self.dist_time_lbl.configure(text=format_time(s["distracted_duration"]))
        self.phone_evt_lbl.configure(text=str(s["phone_events"]))
        self.drowsy_evt_lbl.configure(text=str(s["drowsiness_events"]))

    def _on_session_complete(self, summary: dict) -> None:
        """Save session to database and transition to Completion screen."""
        self.after(0, lambda: self._handle_completion(summary))

    def _handle_completion(self, summary: dict) -> None:
        try:
            from src.database.database import DatabaseManager
            db = DatabaseManager()
            session_id = db.save_session(summary)
            summary["session_id"] = session_id
            db.close()
        except Exception as e:
            print(f"[SessionScreen] Error saving session to DB: {e}")

        comp_screen = self.controller.frames.get("CompletionScreen")
        if comp_screen:
            comp_screen.load_summary(summary)
            self.controller.show_frame("CompletionScreen")

