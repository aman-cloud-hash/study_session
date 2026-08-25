"""
Session Analytics & Visualization Module
========================================

Generates summary statistics and Matplotlib visual charts
from SQLite study session records.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Tkinter embedding & file export
import matplotlib.pyplot as plt
import pandas as pd
import config
from src.database.database import DatabaseManager


class AnalyticsEngine:
    """
    Computes aggregate metrics and renders clean dark-themed charts.
    """

    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        self.db = db_manager or DatabaseManager()

    def get_summary_statistics(self) -> dict:
        """
        Calculate overall study statistics across all recorded sessions.
        """
        sessions = self.db.get_all_sessions()
        if not sessions:
            return {
                "total_sessions": 0,
                "total_study_time_sec": 0.0,
                "avg_focus_score": 0.0,
                "best_focus_score": 0.0,
                "total_phone_time_sec": 0.0,
                "total_drowsiness_time_sec": 0.0,
            }

        df = pd.DataFrame(sessions)
        return {
            "total_sessions": len(df),
            "total_study_time_sec": float(df["total_duration"].sum()),
            "avg_focus_score": round(float(df["focus_score"].mean()), 1),
            "best_focus_score": round(float(df["focus_score"].max()), 1),
            "total_phone_time_sec": float(df["phone_duration"].sum()),
            "total_drowsiness_time_sec": float(df["drowsiness_duration"].sum()),
        }

    def generate_single_session_chart(
        self, session_data: dict, output_path: Path | None = None
    ) -> plt.Figure:
        """
        Generate a Focused vs Distracted breakdown donut chart for a single session.
        """
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)
        fig.patch.set_facecolor("#1e1e24")
        ax.set_facecolor("#1e1e24")

        focused = max(0.1, session_data.get("focused_duration", 0.0))
        phone = session_data.get("phone_duration", 0.0)
        drowsy = session_data.get("drowsiness_duration", 0.0)
        other_dist = max(
            0.0, session_data.get("distracted_duration", 0.0) - phone - drowsy
        )

        labels = ["Focused", "Phone Distraction", "Drowsiness", "Other Distraction"]
        values = [focused, phone, drowsy, other_dist]
        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]

        # Filter zero slices
        filtered_labels = []
        filtered_values = []
        filtered_colors = []
        for l, v, c in zip(labels, values, colors):
            if v > 0:
                filtered_labels.append(l)
                filtered_values.append(v)
                filtered_colors.append(c)

        wedges, texts, autotexts = ax.pie(
            filtered_values,
            labels=filtered_labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=filtered_colors,
            wedgeprops={"edgecolor": "#1e1e24", "linewidth": 2, "width": 0.55},
        )
        for t in texts:
            t.set_color("#e0e0e0")
            t.set_fontsize(10)
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontsize(10)
            at.set_weight("bold")

        ax.set_title(
            f"Focus Breakdown — {session_data.get('student_name', 'Student')}",
            fontsize=13,
            color="#ffffff",
            pad=15,
            weight="bold",
        )
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), bbox_inches="tight", dpi=150)

        return fig

    def generate_history_trend_chart(self) -> plt.Figure | None:
        """
        Generate Focus Score Trend chart over historical sessions.
        """
        sessions = self.db.get_all_sessions()
        if not sessions:
            return None

        df = pd.DataFrame(sessions).iloc[::-1].reset_index(drop=True)  # chronological

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
        fig.patch.set_facecolor("#1e1e24")
        ax.set_facecolor("#1e1e24")

        x = range(1, len(df) + 1)
        y = df["focus_score"]

        ax.plot(x, y, marker="o", color="#3498db", linewidth=2.5, markersize=6)
        ax.fill_between(x, y, color="#3498db", alpha=0.2)

        ax.set_title("Focus Score History Trend", fontsize=12, color="#ffffff", pad=10, weight="bold")
        ax.set_xlabel("Session Number", color="#bbbbbb", fontsize=9)
        ax.set_ylabel("Focus Score (%)", color="#bbbbbb", fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.3, color="#555555")
        ax.tick_params(colors="#aaaaaa")

        plt.tight_layout()
        return fig
