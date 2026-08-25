"""
Session History Screen (Modern Dark Dashboard)
==============================================

Displays past study sessions stored in SQLite with aggregate statistics,
scrollable history table, and trend charts.
"""

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.database.database import DatabaseManager
from src.analytics.analytics import AnalyticsEngine
from src.utils.helpers import format_time


class HistoryScreen(ctk.CTkFrame):
    """
    Shows historical study sessions and performance trends.
    """

    def __init__(self, parent: ctk.CTkFrame, controller) -> None:
        super().__init__(parent, fg_color="#0A0B10")
        self.controller = controller
        self.db = DatabaseManager()
        self.analytics = AnalyticsEngine(self.db)

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ── Header Bar ───────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=25, pady=(15, 10))

        title_col = ctk.CTkFrame(header, fg_color="transparent")
        title_col.pack(side="left")

        ctk.CTkLabel(
            title_col, text="📊 Study History & Analytics", font=ctk.CTkFont(size=24, weight="bold"), text_color="#FFFFFF"
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_col, text="Track your long-term focus progression and study milestones.", font=ctk.CTkFont(size=12), text_color="#94A3B8"
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

        # ── Aggregate Stats Cards ─────────────────────────────────────────
        self.stats_frame = ctk.CTkFrame(
            self,
            fg_color="#12141F",
            corner_radius=16,
            border_width=1,
            border_color="#1F2338",
        )
        self.stats_frame.grid(row=1, column=0, sticky="ew", padx=25, pady=6)
        self.stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.stat_sessions = self._add_stat(0, "TOTAL SESSIONS", "0")
        self.stat_time = self._add_stat(1, "TOTAL STUDY TIME", "00:00:00")
        self.stat_avg = self._add_stat(2, "AVERAGE SCORE", "0%", "#60A5FA")
        self.stat_best = self._add_stat(3, "BEST SCORE", "0%", "#10B981")

        # ── Main Content Tabs: Sessions Table & Trend Chart ───────────────
        self.tabview = ctk.CTkTabview(
            self,
            fg_color="#12141F",
            segmented_button_fg_color="#181B2B",
            segmented_button_selected_color="#4F46E5",
            segmented_button_selected_hover_color="#4338CA",
            corner_radius=16,
        )
        self.tabview.grid(row=2, column=0, sticky="nsew", padx=25, pady=10)

        self.tab_table = self.tabview.add("📜 Past Sessions")
        self.tab_chart = self.tabview.add("📈 Focus Score Trend")

        # Table Scrollable Frame
        self.table_scroll = ctk.CTkScrollableFrame(self.tab_table, fg_color="transparent")
        self.table_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Chart Container
        self.chart_container = ctk.CTkFrame(self.tab_chart, fg_color="transparent")
        self.chart_container.pack(fill="both", expand=True, padx=10, pady=10)

    def _add_stat(self, col: int, label: str, default: str, val_col: str = "#F8FAFC") -> ctk.CTkLabel:
        box = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        box.grid(row=0, column=col, padx=12, pady=12)
        ctk.CTkLabel(box, text=label, font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B").pack(anchor="w")
        val_lbl = ctk.CTkLabel(box, text=default, font=ctk.CTkFont(size=19, weight="bold"), text_color=val_col)
        val_lbl.pack(anchor="w", pady=(2, 0))
        return val_lbl

    def refresh(self) -> None:
        """Reload data from SQLite and rebuild tables and charts."""
        stats = self.analytics.get_summary_statistics()
        self.stat_sessions.configure(text=str(stats["total_sessions"]))
        self.stat_time.configure(text=format_time(stats["total_study_time_sec"]))
        self.stat_avg.configure(text=f"{stats['avg_focus_score']}%")
        self.stat_best.configure(text=f"{stats['best_focus_score']}%")

        # Rebuild table
        for widget in self.table_scroll.winfo_children():
            widget.destroy()

        sessions = self.db.get_all_sessions()
        if not sessions:
            ctk.CTkLabel(
                self.table_scroll,
                text="No study sessions recorded yet. Start a session to see history!",
                font=ctk.CTkFont(size=14),
                text_color="#64748B",
            ).pack(pady=40)
            return

        # Table Headers
        headers = ["ID", "Student", "Date", "Duration", "Focused", "Phone", "Drowsy", "Score", "Action"]
        hdr_frame = ctk.CTkFrame(self.table_scroll, fg_color="#181B2B", corner_radius=8)
        hdr_frame.pack(fill="x", pady=(0, 4))
        hdr_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8), weight=1)

        for col, h in enumerate(headers):
            ctk.CTkLabel(
                hdr_frame, text=h, font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8", padx=6, pady=8
            ).grid(row=0, column=col, sticky="w")

        # Table Rows
        for s in sessions:
            row_frame = ctk.CTkFrame(self.table_scroll, fg_color="#141724", corner_radius=8, border_width=1, border_color="#1F2338")
            row_frame.pack(fill="x", pady=2)
            row_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8), weight=1)

            sid = s["session_id"]
            ctk.CTkLabel(row_frame, text=f"#{sid}", font=ctk.CTkFont(size=11), text_color="#94A3B8").grid(row=0, column=0, sticky="w", padx=6)
            ctk.CTkLabel(row_frame, text=str(s["student_name"]), font=ctk.CTkFont(size=11, weight="bold"), text_color="#F8FAFC").grid(row=0, column=1, sticky="w", padx=6)
            ctk.CTkLabel(row_frame, text=str(s["date"]), font=ctk.CTkFont(size=11), text_color="#94A3B8").grid(row=0, column=2, sticky="w", padx=6)
            ctk.CTkLabel(row_frame, text=format_time(s["total_duration"]), font=ctk.CTkFont(size=11), text_color="#F8FAFC").grid(row=0, column=3, sticky="w", padx=6)
            ctk.CTkLabel(row_frame, text=format_time(s["focused_duration"]), font=ctk.CTkFont(size=11), text_color="#10B981").grid(row=0, column=4, sticky="w", padx=6)
            ctk.CTkLabel(row_frame, text=format_time(s["phone_duration"]), font=ctk.CTkFont(size=11), text_color="#F87171").grid(row=0, column=5, sticky="w", padx=6)
            ctk.CTkLabel(row_frame, text=format_time(s["drowsiness_duration"]), font=ctk.CTkFont(size=11), text_color="#FBBF24").grid(row=0, column=6, sticky="w", padx=6)

            score_val = s["focus_score"]
            score_color = "#10B981" if score_val >= 80 else "#F59E0B" if score_val >= 60 else "#EF4444"
            ctk.CTkLabel(row_frame, text=f"{score_val}%", font=ctk.CTkFont(size=12, weight="bold"), text_color=score_color).grid(row=0, column=7, sticky="w", padx=6)

            del_btn = ctk.CTkButton(
                row_frame,
                text="🗑️",
                width=30,
                height=26,
                corner_radius=6,
                fg_color="#1F151B",
                hover_color="#331A26",
                text_color="#F87171",
                command=lambda id_val=sid: self._delete_session(id_val),
            )
            del_btn.grid(row=0, column=8, padx=6, pady=4)

        # Rebuild Chart
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        fig = self.analytics.generate_history_trend_chart()
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

    def _delete_session(self, session_id: int) -> None:
        self.db.delete_session(session_id)
        self.refresh()

