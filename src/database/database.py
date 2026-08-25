"""
Database Manager
================

SQLite wrapper for storing and retrieving study-session data.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import config


class DatabaseManager:
    """
    Thin wrapper around SQLite for the ``study_focus.db`` database.

    Usage
    -----
    >>> db = DatabaseManager()
    >>> db.initialise()          # creates tables if needed
    >>> db.save_session({...})
    >>> rows = db.get_all_sessions()
    >>> db.close()
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or config.DATABASE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ── Connection helpers ────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row  # dict-like access
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Schema ────────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """Create the sessions table if it does not exist."""
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name        TEXT    NOT NULL,
                date                TEXT    NOT NULL,
                start_time          TEXT    NOT NULL,
                end_time            TEXT    NOT NULL,
                total_duration      REAL    NOT NULL,
                focused_duration    REAL    NOT NULL DEFAULT 0,
                distracted_duration REAL    NOT NULL DEFAULT 0,
                phone_duration      REAL    NOT NULL DEFAULT 0,
                drowsiness_duration REAL    NOT NULL DEFAULT 0,
                phone_events        INTEGER NOT NULL DEFAULT 0,
                drowsiness_events   INTEGER NOT NULL DEFAULT 0,
                focus_score         REAL    NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()

    # ── CRUD ──────────────────────────────────────────────────────────────

    def save_session(self, data: dict[str, Any]) -> int:
        """
        Insert a new session row and return its ``session_id``.

        Parameters
        ----------
        data : dict
            Must contain keys matching the column names
            (minus ``session_id``, which is auto-generated).
        """
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO sessions (
                student_name, date, start_time, end_time,
                total_duration, focused_duration, distracted_duration,
                phone_duration, drowsiness_duration,
                phone_events, drowsiness_events, focus_score
            ) VALUES (
                :student_name, :date, :start_time, :end_time,
                :total_duration, :focused_duration, :distracted_duration,
                :phone_duration, :drowsiness_duration,
                :phone_events, :drowsiness_events, :focus_score
            );
            """,
            data,
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Return every session as a list of dicts, newest first."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY session_id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_session_by_id(self, session_id: int) -> dict[str, Any] | None:
        """Return a single session dict, or ``None`` if not found."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_session(self, session_id: int) -> None:
        """Delete a session by ID."""
        conn = self._connect()
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    def get_session_count(self) -> int:
        """Return total number of stored sessions."""
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0
