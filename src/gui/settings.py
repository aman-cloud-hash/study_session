"""
Application Settings Screen (Modern Dark Dashboard)
===================================================

Configure detection parameters, audio alarms, cameras, and display themes.
"""

import threading
import customtkinter as ctk
import config
from src.utils.helpers import detect_available_cameras


class SettingsScreen(ctk.CTkFrame):
    """
    User preferences and detection parameters configuration screen.
    """

    def __init__(self, parent: ctk.CTkFrame, controller) -> None:
        super().__init__(parent, fg_color="#0A0B10")
        self.controller = controller
        self.available_cameras: list[dict] = [{"index": 0, "name": f"Camera {config.DEFAULT_CAMERA_INDEX} [Default]"}]
        self.selected_camera_index: int = config.DEFAULT_CAMERA_INDEX

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(20, 10))

        title_col = ctk.CTkFrame(header, fg_color="transparent")
        title_col.pack(side="left")

        ctk.CTkLabel(
            title_col, text="⚙️ AI Sentry Settings", font=ctk.CTkFont(size=24, weight="bold"), text_color="#FFFFFF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_col, text="Fine-tune computer vision sensitivity, audio alarms, and hardware sources.", font=ctk.CTkFont(size=12), text_color="#94A3B8"
        ).pack(anchor="w")

        ctk.CTkButton(
            header,
            text="← Back to Home",
            width=130,
            height=36,
            corner_radius=10,
            fg_color="#1E2338",
            hover_color="#2A314E",
            command=lambda: self.controller.show_frame("HomeScreen"),
        ).pack(side="right")

        # ── Settings Container ────────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=1, column=0, sticky="nsew", padx=30, pady=10)
        scroll.grid_columnconfigure(0, weight=1)

        # 1. Eye Detection Thresholds Card
        eye_card = self._create_card(scroll, "👁️ Eye & Drowsiness Detection")

        # EAR Threshold
        self.ear_slider = self._add_slider_setting(
            eye_card,
            label="Eye Aspect Ratio (EAR) Threshold:",
            desc="Landmark ratio threshold below which eye is marked CLOSED.",
            from_=0.15,
            to=0.35,
            default=config.EAR_THRESHOLD,
            format_str="{:.2f}",
        )

        # Drowsiness Duration
        self.drowsy_slider = self._add_slider_setting(
            eye_card,
            label="Drowsiness Alert Delay (seconds):",
            desc="Consecutive seconds eyes remain closed before triggering alert.",
            from_=1.0,
            to=10.0,
            default=config.EYE_CLOSED_DURATION_THRESHOLD,
            format_str="{:.1f}s",
        )

        # 2. Phone Detection Card
        phone_card = self._create_card(scroll, "📱 Mobile Phone Detection")

        self.phone_conf_slider = self._add_slider_setting(
            phone_card,
            label="YOLO Phone Detection Confidence:",
            desc="Minimum model confidence required to register a phone.",
            from_=0.25,
            to=0.90,
            default=config.PHONE_DETECTION_CONFIDENCE,
            format_str="{:.2f}",
        )

        # 3. Audio & Notifications Card
        audio_card = self._create_card(scroll, "🔊 Audio & Distraction Alarms")

        sound_row = ctk.CTkFrame(audio_card, fg_color="transparent")
        sound_row.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(
            sound_row,
            text="Enable Voice / Audio Distraction Alerts:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F8FAFC",
        ).pack(side="left")

        self.sound_switch = ctk.CTkSwitch(
            sound_row,
            text="ON" if config.SOUND_ENABLED else "OFF",
            progress_color="#4F46E5",
        )
        if config.SOUND_ENABLED:
            self.sound_switch.select()
        else:
            self.sound_switch.deselect()
        self.sound_switch.pack(side="right")

        # 4. Camera & Device Card
        cam_card = self._create_card(scroll, "📹 Camera Device Source")
        cam_row = ctk.CTkFrame(cam_card, fg_color="transparent")
        cam_row.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(
            cam_row, text="Active Video Device:", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC"
        ).pack(side="left")

        self.cam_dropdown = ctk.CTkOptionMenu(
            cam_row,
            values=[f"Camera {config.DEFAULT_CAMERA_INDEX} [Default]"],
            height=36,
            fg_color="#181B2B",
            button_color="#2D334D",
            button_hover_color="#3B4261",
            corner_radius=8,
            command=self._on_settings_camera_selected,
        )
        self.cam_dropdown.pack(side="right", padx=(8, 0))

        self.cam_scan_btn = ctk.CTkButton(
            cam_row,
            text="🔄 Scan",
            width=70,
            height=36,
            corner_radius=8,
            fg_color="#1E2338",
            hover_color="#2A314E",
            command=self._scan_cameras_settings,
        )
        self.cam_scan_btn.pack(side="right")
        self._scan_cameras_settings()

        # 5. UI Appearance Theme
        theme_card = self._create_card(scroll, "🎨 Display Theme")
        theme_row = ctk.CTkFrame(theme_card, fg_color="transparent")
        theme_row.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(
            theme_row, text="Appearance Mode:", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC"
        ).pack(side="left")
        self.theme_menu = ctk.CTkOptionMenu(
            theme_row,
            values=["dark", "light"],
            height=36,
            fg_color="#181B2B",
            button_color="#2D334D",
            button_hover_color="#3B4261",
            corner_radius=8,
            command=self._change_theme,
        )
        self.theme_menu.set(config.DEFAULT_THEME)
        self.theme_menu.pack(side="right")

        # ── Save Button ───────────────────────────────────────────────────
        save_btn = ctk.CTkButton(
            scroll,
            text="💾  SAVE & APPLY SETTINGS",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=46,
            corner_radius=12,
            fg_color="#4F46E5",
            hover_color="#4338CA",
            command=self._save_settings,
        )
        save_btn.pack(fill="x", padx=4, pady=(20, 8))

        self.status_lbl = ctk.CTkLabel(
            scroll, text="", text_color="#10B981", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.status_lbl.pack(pady=4)

    def _scan_cameras_settings(self) -> None:
        self.cam_scan_btn.configure(state="disabled")
        threading.Thread(target=self._scan_cameras_worker, daemon=True).start()

    def _scan_cameras_worker(self) -> None:
        cameras = detect_available_cameras(max_tested=6)
        self.after(0, lambda: self._apply_camera_results(cameras))

    def _apply_camera_results(self, cameras: list[dict]) -> None:
        self.available_cameras = cameras
        names = [cam["name"] for cam in cameras]
        self.cam_dropdown.configure(values=names)

        match = next((c for c in cameras if c["index"] == self.selected_camera_index), cameras[0])
        self.selected_camera_index = match["index"]
        self.cam_dropdown.set(match["name"])
        self.cam_scan_btn.configure(state="normal")

    def _on_settings_camera_selected(self, chosen_name: str) -> None:
        for cam in self.available_cameras:
            if cam["name"] == chosen_name:
                self.selected_camera_index = cam["index"]
                config.DEFAULT_CAMERA_INDEX = cam["index"]
                break

    def _create_card(self, parent, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color="#12141F",
            corner_radius=16,
            border_width=1,
            border_color="#1F2338",
        )
        card.pack(fill="x", pady=6, padx=4)
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color="#F8FAFC"
        ).pack(anchor="w", padx=16, pady=(14, 6))
        return card

    def _add_slider_setting(
        self, parent, label: str, desc: str, from_: float, to: float, default: float, format_str: str
    ) -> ctk.CTkSlider:
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", padx=16, pady=8)

        top_row = ctk.CTkFrame(container, fg_color="transparent")
        top_row.pack(fill="x")
        ctk.CTkLabel(
            top_row, text=label, font=ctk.CTkFont(size=12, weight="bold"), text_color="#F1F5F9"
        ).pack(side="left")
        val_lbl = ctk.CTkLabel(
            top_row, text=format_str.format(default), font=ctk.CTkFont(size=12, weight="bold"), text_color="#60A5FA"
        )
        val_lbl.pack(side="right")

        ctk.CTkLabel(
            container, text=desc, font=ctk.CTkFont(size=11), text_color="#64748B"
        ).pack(anchor="w", pady=(1, 6))

        slider = ctk.CTkSlider(
            container,
            from_=from_,
            to=to,
            progress_color="#4F46E5",
            button_color="#818CF8",
            button_hover_color="#A5B4FC",
            command=lambda val: val_lbl.configure(text=format_str.format(val)),
        )
        slider.set(default)
        slider.pack(fill="x")
        return slider

    def _change_theme(self, theme: str) -> None:
        ctk.set_appearance_mode(theme)

    def _save_settings(self) -> None:
        config.EAR_THRESHOLD = float(self.ear_slider.get())
        config.EYE_CLOSED_DURATION_THRESHOLD = float(self.drowsy_slider.get())
        config.PHONE_DETECTION_CONFIDENCE = float(self.phone_conf_slider.get())
        config.SOUND_ENABLED = bool(self.sound_switch.get())
        config.DEFAULT_CAMERA_INDEX = self.selected_camera_index

        self.status_lbl.configure(text="✅ Settings saved and applied successfully!")
        self.after(3000, lambda: self.status_lbl.configure(text=""))


