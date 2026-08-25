"""
Utility helpers used across the application.
"""

from pathlib import Path
import config


def ensure_directories() -> None:
    """Create all required directories if they do not already exist."""
    directories = [
        config.AUDIO_DIR,
        config.MODELS_DIR,
        config.DATABASE_DIR,
        config.SNAPSHOTS_DIR,
        config.REPORTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Convert a float number of seconds into HH:MM:SS display string.

    Examples
    --------
    >>> format_time(3661.5)
    '01:01:01'
    >>> format_time(59)
    '00:00:59'
    """
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp *value* between *low* and *high* inclusive."""
    return max(low, min(high, value))


def validate_session_duration(hours: int, minutes: int) -> tuple[bool, str]:
    """
    Validate that the session duration is within allowed bounds.

    Returns
    -------
    (is_valid, error_message)
    """
    total_minutes = hours * 60 + minutes
    if total_minutes < config.MIN_SESSION_MINUTES:
        return False, f"Session must be at least {config.MIN_SESSION_MINUTES} minute(s)."
    if total_minutes > config.MAX_SESSION_MINUTES:
        return False, f"Session cannot exceed {config.MAX_SESSION_MINUTES} minutes ({config.MAX_SESSION_MINUTES // 60} hours)."
    return True, ""


def detect_available_cameras(max_tested: int = 4) -> list[dict]:
    """
    Detect available cameras connected to the system.
    Returns list of dicts: [{'index': 0, 'name': 'Camera 0: Integrated Camera'}, ...]
    """
    cameras = []

    # 1. Primary on Windows: pygrabber (instant DirectShow device query, <50ms)
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
        if devices:
            for idx, dev_name in enumerate(devices):
                cameras.append({
                    "index": idx,
                    "name": f"Camera {idx}: {dev_name}"
                })
            return cameras
    except Exception:
        pass

    # 2. Fallback: OpenCV DirectShow probe
    import cv2
    import platform
    is_windows = platform.system() == "Windows"

    for idx in range(max_tested):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if is_windows else cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                res_str = f" ({w}x{h})" if (w > 0 and h > 0) else ""
                name = f"Camera {idx}{res_str}"
                if idx == 0:
                    name += " [Default]"
                cameras.append({"index": idx, "name": name})
            else:
                if cap is not None:
                    cap.release()
        except Exception:
            pass

    if not cameras:
        cameras.append({"index": 0, "name": "Camera 0 [Default]"})

    return cameras

