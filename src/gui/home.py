"""
Home Screen (Modern Dark Glassmorphism UI)
===========================================

Sleek landing dashboard with student profile configuration,
interactive preset pills, camera device scanner, and AI features overview.
"""

import threading
import customtkinter as ctk
import config
from src.utils.helpers import validate_session_duration, detect_available_cameras


class HomeScreen(ctk.CTkFrame):
    """Modern Launcher & Setup Screen."""

    def __init__(self, parent: ctk.CTkFrame, controller) -> None:
        super().__init__(parent, fg_color="#0D0E15")
        self.controller = controller
        self.available_cameras: list[dict] = [{"index": 0, "name": "Camera 0 [Default]"}]
        self.selected_camera_index: int = config.DEFAULT_CAMERA_INDEX
        self._active_preset_btn = None

        self._build_ui()
        self._scan_cameras_async()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Scrollable Outer Container
        scroll_wrapper = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_wrapper.grid(row=0, column=0, sticky="nsew", padx=30, pady=20)
        scroll_wrapper.grid_columnconfigure(0, weight=1)

        # ── Hero Header ───────────────────────────────────────────────────
        hero = ctk.CTkFrame(scroll_wrapper, fg_color="transparent")
        hero.pack(fill="x", pady=(5, 20))

        # Glowing AI Badge
        badge_frame = ctk.CTkFrame(hero, fg_color="#181B2B", corner_radius=20, border_width=1, border_color="#3B4261")
        badge_frame.pack(pady=(0, 8))
        ctk.CTkLabel(
            badge_frame,
            text=" ✨  NEXT-GEN AI STUDY COPILOT ",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#818CF8",
        ).pack(padx=14, pady=4)

        # Title
        title_lbl = ctk.CTkLabel(
            hero,
            text="AI STUDY FOCUS",
            font=ctk.CTkFont(size=34, weight="bold"),
            text_color="#FFFFFF",
        )
        title_lbl.pack()

        sub_lbl = ctk.CTkLabel(
            hero,
            text="Real-time Computer Vision sentry for laser-sharp focus and distraction prevention.",
            font=ctk.CTkFont(size=14),
            text_color="#94A3B8",
        )
        sub_lbl.pack(pady=(2, 0))

        # ── 2-Column Main Dashboard ───────────────────────────────────────
        content_grid = ctk.CTkFrame(scroll_wrapper, fg_color="transparent")
        content_grid.pack(fill="both", expand=True, pady=5)
        content_grid.grid_columnconfigure((0, 1), weight=1)

        # ── Left Column: Configuration Card ──────────────────────────────
        setup_card = ctk.CTkFrame(
            content_grid,
            fg_color="#141724",
            corner_radius=16,
            border_width=1,
            border_color="#23283E",
        )
        setup_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)

        # Card Title
        ctk.CTkLabel(
            setup_card,
            text="👤 Student & Session Setup",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#F8FAFC",
        ).pack(anchor="w", padx=22, pady=(18, 12))

        # Student Name
        ctk.CTkLabel(
            setup_card,
            text="STUDENT NAME",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B",
        ).pack(anchor="w", padx=22)

        self.name_entry = ctk.CTkEntry(
            setup_card,
            placeholder_text="e.g. Aman / Alex",
            height=42,
            fg_color="#1B1E30",
            border_color="#2D334D",
            corner_radius=10,
            font=ctk.CTkFont(size=14),
        )
        self.name_entry.pack(fill="x", padx=22, pady=(6, 16))

        # Session Duration
        ctk.CTkLabel(
            setup_card,
            text="TARGET DURATION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B",
        ).pack(anchor="w", padx=22)

        # Preset Pills
        pills_row = ctk.CTkFrame(setup_card, fg_color="transparent")
        pills_row.pack(fill="x", padx=22, pady=(6, 8))

        self.preset_buttons = []
        presets = [("25m", 0, 25), ("45m", 0, 45), ("60m", 1, 0), ("90m", 1, 30), ("120m", 2, 0)]
        for h, m in [(0, 25), (0, 45), (1, 0), (1, 30), (2, 0)]:
            pass

        for idx, (label, h, m) in enumerate(presets):
            btn = ctk.CTkButton(
                pills_row,
                text=label,
                width=55,
                height=32,
                corner_radius=8,
                fg_color="#1E2338" if label != "25m" else "#4F46E5",
                hover_color="#4338CA",
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda hh=h, mm=m, lbl=label: self._select_preset(hh, mm, lbl),
            )
            btn.pack(side="left", fill="x", expand=True, padx=2)
            self.preset_buttons.append((label, btn))

        # Custom Time Inputs
        custom_row = ctk.CTkFrame(setup_card, fg_color="transparent")
        custom_row.pack(fill="x", padx=22, pady=(4, 16))

        ctk.CTkLabel(custom_row, text="Hours:", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(side="left", padx=(0, 6))
        self.hours_entry = ctk.CTkEntry(
            custom_row, width=60, height=36, fg_color="#1B1E30", border_color="#2D334D", corner_radius=8, justify="center"
        )
        self.hours_entry.insert(0, "0")
        self.hours_entry.pack(side="left", padx=(0, 16))

        ctk.CTkLabel(custom_row, text="Minutes:", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(side="left", padx=(0, 6))
        self.minutes_entry = ctk.CTkEntry(
            custom_row, width=60, height=36, fg_color="#1B1E30", border_color="#2D334D", corner_radius=8, justify="center"
        )
        self.minutes_entry.insert(0, "25")
        self.minutes_entry.pack(side="left")

        # Camera Selection
        cam_header = ctk.CTkFrame(setup_card, fg_color="transparent")
        cam_header.pack(fill="x", padx=22)
        ctk.CTkLabel(
            cam_header,
            text="CAMERA SOURCE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B",
        ).pack(side="left")

        self.cam_status_lbl = ctk.CTkLabel(
            cam_header, text="Scanning...", font=ctk.CTkFont(size=11), text_color="#6366F1"
        )
        self.cam_status_lbl.pack(side="right")

        cam_row = ctk.CTkFrame(setup_card, fg_color="transparent")
        cam_row.pack(fill="x", padx=22, pady=(6, 20))

        self.camera_dropdown = ctk.CTkOptionMenu(
            cam_row,
            values=["Camera 0 [Default]"],
            height=40,
            fg_color="#1B1E30",
            button_color="#2D334D",
            button_hover_color="#3B4261",
            corner_radius=10,
            font=ctk.CTkFont(size=13),
            command=self._on_camera_selected,
        )
        self.camera_dropdown.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.refresh_cam_btn = ctk.CTkButton(
            cam_row,
            text="🔄 Scan",
            width=76,
            height=40,
            fg_color="#1E2338",
            hover_color="#2A314E",
            corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._scan_cameras_async,
        )
        self.refresh_cam_btn.pack(side="right")

        # ── Right Column: AI Intelligence & Launch Card ───────────────────
        right_card = ctk.CTkFrame(
            content_grid,
            fg_color="#141724",
            corner_radius=16,
            border_width=1,
            border_color="#23283E",
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)

        ctk.CTkLabel(
            right_card,
            text="⚡ AI Sentry Features",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#F8FAFC",
        ).pack(anchor="w", padx=22, pady=(18, 14))

        features = [
            ("👁️", "Face Mesh & Eye Tracking", "478-point landmark tracking with sub-second blink detection"),
            ("📱", "Ultra-Fast Phone Radar", "Instant 1-frame trigger with anti-remote suppression"),
            ("😴", "Drowsiness Warning", "Observation timer alert to prevent falling asleep"),
            ("🔊", "Dynamic Distraction Audio", "Custom voice alert triggers immediately on distraction"),
        ]

        for icon, title, desc in features:
            item_box = ctk.CTkFrame(right_card, fg_color="#181B2D", corner_radius=10)
            item_box.pack(fill="x", padx=22, pady=4)

            ctk.CTkLabel(item_box, text=icon, font=ctk.CTkFont(size=18)).pack(side="left", padx=(12, 8), pady=8)
            text_col = ctk.CTkFrame(item_box, fg_color="transparent")
            text_col.pack(side="left", fill="x", expand=True, pady=6)

            ctk.CTkLabel(
                text_col, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color="#F1F5F9"
            ).pack(anchor="w")
            ctk.CTkLabel(
                text_col, text=desc, font=ctk.CTkFont(size=11), text_color="#94A3B8"
            ).pack(anchor="w")

        # Error label
        self.error_label = ctk.CTkLabel(
            right_card, text="", text_color="#EF4444", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.error_label.pack(pady=(10, 2))

        # Launch Button
        self.start_btn = ctk.CTkButton(
            right_card,
            text="▶   START FOCUS SESSION",
            height=48,
            corner_radius=12,
            fg_color="#4F46E5",
            hover_color="#4338CA",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._on_start,
        )
        self.start_btn.pack(fill="x", padx=22, pady=(4, 18))

        # ── Bottom Action Navigation Bar ──────────────────────────────────
        nav_bar = ctk.CTkFrame(scroll_wrapper, fg_color="transparent")
        nav_bar.pack(fill="x", pady=(15, 5))

        ctk.CTkButton(
            nav_bar,
            text="📊 Session History",
            height=38,
            corner_radius=10,
            fg_color="#181B2B",
            hover_color="#242940",
            font=ctk.CTkFont(size=13),
            command=self._on_history,
        ).pack(side="left", padx=(0, 8), fill="x", expand=True)

        ctk.CTkButton(
            nav_bar,
            text="⚙️ Settings",
            height=38,
            corner_radius=10,
            fg_color="#181B2B",
            hover_color="#242940",
            font=ctk.CTkFont(size=13),
            command=self._on_settings,
        ).pack(side="left", padx=8, fill="x", expand=True)

        ctk.CTkButton(
            nav_bar,
            text="❌ Exit App",
            height=38,
            corner_radius=10,
            fg_color="#1F151B",
            hover_color="#331A26",
            text_color="#F87171",
            font=ctk.CTkFont(size=13),
            command=self._on_exit,
        ).pack(side="left", padx=(8, 0), fill="x", expand=True)

        # Privacy Badge
        ctk.CTkLabel(
            scroll_wrapper,
            text=f"🔒 {config.PRIVACY_NOTICE}",
            font=ctk.CTkFont(size=11),
            text_color="#64748B",
        ).pack(pady=(12, 0))

    # ── Interactive Preset Selection ──────────────────────────────────────

    def _select_preset(self, hours: int, minutes: int, label: str) -> None:
        self.hours_entry.delete(0, "end")
        self.hours_entry.insert(0, str(hours))
        self.minutes_entry.delete(0, "end")
        self.minutes_entry.insert(0, str(minutes))
        self.error_label.configure(text="")

        for btn_label, btn in self.preset_buttons:
            if btn_label == label:
                btn.configure(fg_color="#4F46E5")
            else:
                btn.configure(fg_color="#1E2338")

    # ── Camera Discovery ──────────────────────────────────────────────────

    def _scan_cameras_async(self) -> None:
        self.cam_status_lbl.configure(text="Scanning cameras...", text_color="#6366F1")
        self.refresh_cam_btn.configure(state="disabled")
        threading.Thread(target=self._scan_cameras_worker, daemon=True).start()

    def _scan_cameras_worker(self) -> None:
        cameras = detect_available_cameras(max_tested=6)
        self.after(0, lambda: self._apply_camera_results(cameras))

    def _apply_camera_results(self, cameras: list[dict]) -> None:
        self.available_cameras = cameras
        names = [cam["name"] for cam in cameras]
        self.camera_dropdown.configure(values=names)

        match = next((c for c in cameras if c["index"] == self.selected_camera_index), cameras[0])
        self.selected_camera_index = match["index"]
        self.camera_dropdown.set(match["name"])
        config.DEFAULT_CAMERA_INDEX = self.selected_camera_index

        self.cam_status_lbl.configure(
            text=f"● {len(cameras)} Online",
            text_color="#10B981" if len(cameras) > 0 else "#EF4444",
        )
        self.refresh_cam_btn.configure(state="normal")

    def _on_camera_selected(self, chosen_name: str) -> None:
        for cam in self.available_cameras:
            if cam["name"] == chosen_name:
                self.selected_camera_index = cam["index"]
                config.DEFAULT_CAMERA_INDEX = cam["index"]
                break

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_duration(self) -> tuple[int, int]:
        try:
            h = int(self.hours_entry.get() or "0")
        except ValueError:
            h = 0
        try:
            m = int(self.minutes_entry.get() or "0")
        except ValueError:
            m = 0
        return h, m

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            self.error_label.configure(text="⚠️ Please enter your student name.")
            return

        hours, minutes = self._get_duration()
        valid, msg = validate_session_duration(hours, minutes)
        if not valid:
            self.error_label.configure(text=f"⚠️ {msg}")
            return

        self.error_label.configure(text="")
        total_seconds = (hours * 60 + minutes) * 60

        session_frame = self.controller.frames.get("SessionScreen")
        if session_frame:
            self.controller.show_frame("SessionScreen")
            session_frame.start_session(
                student_name=name,
                duration_seconds=total_seconds,
                camera_index=self.selected_camera_index,
            )

    def _on_history(self) -> None:
        hist_frame = self.controller.frames.get("HistoryScreen")
        if hist_frame:
            hist_frame.refresh()
            self.controller.show_frame("HistoryScreen")

    def _on_settings(self) -> None:
        self.controller.show_frame("SettingsScreen")

    def _on_exit(self) -> None:
        self.controller.destroy()


