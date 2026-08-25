"""
Session Completion Screen (Modern Dark Report Dashboard)
=========================================================

Displays final session metrics, focus score breakdown,
and embedded Matplotlib donut chart upon session completion.
"""

from pathlib import Path
import tkinter.filedialog as fd
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import config
from src.analytics.analytics import AnalyticsEngine
from src.utils.helpers import format_time


class CompletionScreen(ctk.CTkFrame):
    """
    Summary and report screen presented when a session finishes.
    """

    def __init__(self, parent: ctk.CTkFrame, controller) -> None:
        super().__init__(parent, fg_color="#0A0B10")
        self.controller = controller
        self.analytics = AnalyticsEngine()
        self.current_summary: dict = {}
        self.canvas_widget = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        scroll_container = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_container.grid(row=0, column=0, sticky="nsew", padx=25, pady=15)
        scroll_container.grid_columnconfigure((0, 1), weight=1)

        # ── Header Banner ────────────────────────────────────────────────
        header_frame = ctk.CTkFrame(scroll_container, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, pady=(10, 15))

        badge = ctk.CTkFrame(header_frame, fg_color="#181B2B", corner_radius=20, border_width=1, border_color="#3B4261")
        badge.pack(pady=(0, 6))
        ctk.CTkLabel(
            badge,
            text=" 🏆 SESSION COMPLETE ",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#10B981",
        ).pack(padx=14, pady=4)

        self.congrats_lbl = ctk.CTkLabel(
            header_frame,
            text="EXCELLENT FOCUS SESSION!",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="#FFFFFF",
        )
        self.congrats_lbl.pack()

        self.student_name_lbl = ctk.CTkLabel(
            header_frame,
            text="Great job, Student!",
            font=ctk.CTkFont(size=15),
            text_color="#94A3B8",
        )
        self.student_name_lbl.pack(pady=(2, 0))

        # ── Metric Cards Grid ─────────────────────────────────────────────
        metrics_grid = ctk.CTkFrame(scroll_container, fg_color="transparent")
        metrics_grid.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=10)
        metrics_grid.grid_columnconfigure((0, 1), weight=1)

        self.card_score = self._create_metric_card(metrics_grid, 0, 0, "⭐ FINAL SCORE", "---", "#10B981")
        self.card_total = self._create_metric_card(metrics_grid, 0, 1, "⏱️ TOTAL DURATION", "00:00:00")
        self.card_focus = self._create_metric_card(metrics_grid, 1, 0, "✅ FOCUSED TIME", "00:00:00", "#10B981")
        self.card_dist = self._create_metric_card(metrics_grid, 1, 1, "⚠️ DISTRACTED TIME", "00:00:00", "#EF4444")
        self.card_phone = self._create_metric_card(metrics_grid, 2, 0, "📱 PHONE USAGE", "00:00:00")
        self.card_drowsy = self._create_metric_card(metrics_grid, 2, 1, "😴 DROWSINESS TIME", "00:00:00")
        self.card_phone_evt = self._create_metric_card(metrics_grid, 3, 0, "🔔 PHONE ALERTS", "0")
        self.card_drowsy_evt = self._create_metric_card(metrics_grid, 3, 1, "🔔 DROWSY ALERTS", "0")

        # ── Chart Container ───────────────────────────────────────────────
        self.chart_container = ctk.CTkFrame(
            scroll_container,
            fg_color="#12141F",
            corner_radius=16,
            border_width=1,
            border_color="#1F2338",
        )
        self.chart_container.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=10)

        # ── Bottom Action Bar ─────────────────────────────────────────────
        action_bar = ctk.CTkFrame(scroll_container, fg_color="transparent")
        action_bar.grid(row=2, column=0, columnspan=2, pady=(20, 10))

        ctk.CTkButton(
            action_bar,
            text="💾 Save Report",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=42,
            corner_radius=10,
            fg_color="#1E2338",
            hover_color="#2A314E",
            command=self._on_save_report,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            action_bar,
            text="📊 View History",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=42,
            corner_radius=10,
            fg_color="#1E2338",
            hover_color="#2A314E",
            command=self._on_view_history,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            action_bar,
            text="▶ Start New Session",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=42,
            corner_radius=10,
            fg_color="#4F46E5",
            hover_color="#4338CA",
            command=self._on_start_new,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            action_bar,
            text="❌ Exit",
            font=ctk.CTkFont(size=13),
            height=42,
            corner_radius=10,
            fg_color="#1F151B",
            hover_color="#331A26",
            text_color="#F87171",
            command=self._on_exit,
        ).pack(side="left", padx=6)

    def _create_metric_card(
        self, parent, row: int, col: int, title: str, default_val: str, val_color: str = "#F8FAFC"
    ) -> ctk.CTkLabel:
        card = ctk.CTkFrame(
            parent,
            fg_color="#141724",
            corner_radius=14,
            border_width=1,
            border_color="#1F2338",
        )
        card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B"
        ).pack(anchor="w", padx=14, pady=(10, 0))
        val_lbl = ctk.CTkLabel(
            card, text=default_val, font=ctk.CTkFont(size=20, weight="bold"), text_color=val_color
        )
        val_lbl.pack(anchor="w", padx=14, pady=(0, 10))
        return val_lbl

    def load_summary(self, summary: dict) -> None:
        """Populate screen with final session data and render donut chart."""
        self.current_summary = summary
        name = summary.get("student_name", "Student")
        self.student_name_lbl.configure(text=f"Great job, {name}!")

        score = summary.get("focus_score", 0.0)
        self.card_score.configure(text=f"{score}%")
        self.card_total.configure(text=format_time(summary.get("total_duration", 0.0)))
        self.card_focus.configure(text=format_time(summary.get("focused_duration", 0.0)))
        self.card_dist.configure(text=format_time(summary.get("distracted_duration", 0.0)))
        self.card_phone.configure(text=format_time(summary.get("phone_duration", 0.0)))
        self.card_drowsy.configure(text=format_time(summary.get("drowsiness_duration", 0.0)))
        self.card_phone_evt.configure(text=str(summary.get("phone_events", 0)))
        self.card_drowsy_evt.configure(text=str(summary.get("drowsiness_events", 0)))

        # Render Chart
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        fig = self.analytics.generate_single_session_chart(summary)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=10, pady=10)

    def _on_save_report(self) -> None:
        """Save chart & report as PNG file."""
        if not self.current_summary:
            return
        default_name = f"focus_report_{self.current_summary.get('student_name', 'student')}_{self.current_summary.get('date', 'session')}.png"
        path = fd.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            initialdir=str(config.REPORTS_DIR),
            initialfile=default_name,
        )
        if path:
            self.analytics.generate_single_session_chart(self.current_summary, Path(path))

    def _on_view_history(self) -> None:
        self.controller.show_frame("HistoryScreen")

    def _on_start_new(self) -> None:
        self.controller.show_frame("HomeScreen")

    def _on_exit(self) -> None:
        self.controller.destroy()

