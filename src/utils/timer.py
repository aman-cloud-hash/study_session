"""
Session Timer
=============

Tracks elapsed / remaining time for a study session.
Supports start, pause, resume, stop, and reset.
"""

import time


class SessionTimer:
    """
    A pause-aware countdown timer.

    Parameters
    ----------
    duration_seconds : float
        Total session length in seconds.
    """

    def __init__(self, duration_seconds: float) -> None:
        self.duration = duration_seconds  # total duration (seconds)
        self._elapsed = 0.0               # accumulated elapsed time
        self._start_ts: float | None = None
        self._running = False
        self._finished = False

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin or resume the timer."""
        if self._finished:
            return
        if not self._running:
            self._start_ts = time.time()
            self._running = True

    def pause(self) -> None:
        """Pause the timer, preserving elapsed time."""
        if self._running and self._start_ts is not None:
            self._elapsed += time.time() - self._start_ts
            self._start_ts = None
            self._running = False

    def resume(self) -> None:
        """Alias for start() — resumes after pause."""
        self.start()

    def stop(self) -> None:
        """Stop the timer permanently."""
        self.pause()
        self._finished = True

    def reset(self, duration_seconds: float | None = None) -> None:
        """Reset to initial state, optionally with a new duration."""
        if duration_seconds is not None:
            self.duration = duration_seconds
        self._elapsed = 0.0
        self._start_ts = None
        self._running = False
        self._finished = False

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def elapsed(self) -> float:
        """Total elapsed seconds (accounts for pauses)."""
        if self._running and self._start_ts is not None:
            return self._elapsed + (time.time() - self._start_ts)
        return self._elapsed

    @property
    def remaining(self) -> float:
        """Seconds left until the session ends (≥ 0)."""
        return max(0.0, self.duration - self.elapsed)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_finished(self) -> bool:
        """True when either stopped manually or time ran out."""
        if self._finished:
            return True
        if self.remaining <= 0:
            self._finished = True
            self._running = False
            return True
        return False

    @property
    def progress(self) -> float:
        """Fraction of session completed (0.0 → 1.0)."""
        if self.duration <= 0:
            return 1.0
        return min(1.0, self.elapsed / self.duration)
