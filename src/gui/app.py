"""
Main Application Window
=======================

Initialises CustomTkinter root window and coordinates screen transitions.
"""

import customtkinter as ctk
import config


class StudyFocusApp(ctk.CTk):
    """
    Root application window. Holds all screen frames in a central container.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Window setup ──────────────────────────────────────────────────
        self.title(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(1050, 700)

        # Theme
        ctk.set_appearance_mode(config.DEFAULT_THEME)
        ctk.set_default_color_theme(config.DEFAULT_COLOR_THEME)

        # ── Container that holds all screens ─────────────────────────────
        self.container = ctk.CTkFrame(self, fg_color="#0A0B10")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # ── Registry of screens ──────────────────────────────────────────
        self.frames: dict[str, ctk.CTkFrame] = {}

        # Import all screens
        from src.gui.home import HomeScreen
        from src.gui.session import SessionScreen
        from src.gui.completion import CompletionScreen
        from src.gui.history import HistoryScreen
        from src.gui.settings import SettingsScreen

        screen_classes = (
            HomeScreen,
            SessionScreen,
            CompletionScreen,
            HistoryScreen,
            SettingsScreen,
        )

        for ScreenClass in screen_classes:
            name = ScreenClass.__name__
            frame = ScreenClass(parent=self.container, controller=self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Show the home screen first
        self.show_frame("HomeScreen")

        # Clean exit on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Navigation ────────────────────────────────────────────────────────

    def show_frame(self, name: str) -> None:
        """Raise the frame identified by *name* to the top."""
        frame = self.frames.get(name)
        if frame is not None:
            frame.tkraise()

    def _on_close(self) -> None:
        """Ensure any active monitoring thread stops before exit."""
        session_frame = self.frames.get("SessionScreen")
        if session_frame and hasattr(session_frame, "monitor") and session_frame.monitor:
            try:
                session_frame.monitor.stop()
            except Exception:
                pass
        self.destroy()
