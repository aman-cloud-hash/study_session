"""
State-Based Alert Manager
=========================

Synchronizes audio alert playback directly with the current distraction state.
Ensures instant start when distraction occurs and INSTANT STOP (<5ms)
as soon as distraction conditions cease.
"""

import time
import threading
from pathlib import Path
import config


class AlertManager:
    """
    State-synchronized audio alert controller.

    Rules:
    - Condition TRUE  → Audio ON
    - Condition FALSE → Audio OFF immediately
    - Condition becomes TRUE again → Audio ON again
    - Maximum active alert at any moment = 1
    """

    def __init__(self) -> None:
        self._initialised = False
        self._enabled = config.SOUND_ENABLED
        self._current_alert: str | None = None
        self._sound_objects: dict[str, object] = {}
        self._channel = None
        self._lock = threading.Lock()

        # Custom alert audio fallback mapping
        drowsiness_mp3 = getattr(config, "DROWSINESS_ALERT_AUDIO", config.CUSTOM_ALERT_AUDIO)
        phone_mp3 = getattr(config, "PHONE_ALERT_AUDIO", config.AUDIO_DIR / "Rakh Teri Maa Ki Rakh Meme Template [7yF953fKPWA].mp3")
        phone_offset = getattr(config, "PHONE_ALERT_START_OFFSET_SEC", 1.0)

        self._sound_paths: dict[str, Path] = {
            "drowsiness": drowsiness_mp3 if drowsiness_mp3.exists() else (config.SOUNDS_DIR / config.SOUND_DROWSINESS),
            "phone": phone_mp3 if phone_mp3.exists() else (config.SOUNDS_DIR / config.SOUND_PHONE),
            "high_distraction": phone_mp3 if phone_mp3.exists() else (config.SOUNDS_DIR / config.SOUND_HIGH_DISTRACTION),
            "session_complete": config.SOUNDS_DIR / config.SOUND_SESSION_COMPLETE,
        }

        self._offsets: dict[str, float] = {
            "drowsiness": 0.0,
            "phone": phone_offset,
            "high_distraction": phone_offset,
            "session_complete": 0.0,
        }

        self._init_mixer()

    def _init_mixer(self) -> None:
        """Initialize pygame mixer & dedicated playback channel."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(8)
            self._channel = pygame.mixer.Channel(0)  # Dedicated alert channel
            self._initialised = True

            # Preload sounds for instant zero-latency playback
            self._preload_sounds()
        except Exception as exc:
            print(f"[AlertManager] Could not initialise audio mixer: {exc}")
            self._initialised = False

    def _load_sound_with_offset(self, path: Path, offset_sec: float = 0.0):
        """Preload a sound object, applying offset trimming if requested."""
        import pygame
        sound = pygame.mixer.Sound(str(path))
        if offset_sec <= 0.0:
            return sound

        # Try slicing sample array with pygame.sndarray
        try:
            import pygame.sndarray
            arr = pygame.sndarray.array(sound)
            init_freq = pygame.mixer.get_init()[0]
            start_sample = int(init_freq * offset_sec)
            if 0 < start_sample < len(arr):
                trimmed_arr = arr[start_sample:]
                return pygame.sndarray.make_sound(trimmed_arr)
        except Exception:
            pass

        # Fallback: slice raw buffer bytes
        try:
            freq, size, channels = pygame.mixer.get_init()
            bytes_per_sample = abs(size) // 8
            frame_size = channels * bytes_per_sample
            raw = sound.get_raw()
            offset_bytes = int(freq * offset_sec) * frame_size
            if 0 < offset_bytes < len(raw):
                return pygame.mixer.Sound(buffer=raw[offset_bytes:])
        except Exception:
            pass

        return sound

    def _preload_sounds(self) -> None:
        """Preload sound files into memory."""
        for category, path in self._sound_paths.items():
            if path and path.exists():
                try:
                    offset = self._offsets.get(category, 0.0)
                    sound = self._load_sound_with_offset(path, offset)
                    self._sound_objects[category] = sound
                except Exception as e:
                    print(f"[AlertManager] Failed to preload '{category}' from {path}: {e}")

    # ── State Synchronization (Called on every frame) ──────────────────────

    def sync_alert_state(self, target_alert: str | None) -> None:
        """
        Synchronize the active audio alert with the current target state.

        Parameters
        ----------
        target_alert : str or None
            'high_distraction', 'phone', 'drowsiness', or None (focused/no alert)
        """
        if not self._enabled or not self._initialised:
            if self._current_alert is not None:
                self.stop_alert()
            return

        with self._lock:
            # If state hasn't changed, keep current playback status
            if target_alert == self._current_alert:
                return

            # State has changed:
            old_alert = self._current_alert

            if target_alert is None:
                # Distraction ceased -> Stop immediately
                self._stop_playback_internal()
                self._current_alert = None
                if config.DEBUG_LOGGING and old_alert:
                    self._log(f"Distraction resolved ({old_alert} ended) -> Alert stopped")
            else:
                # New or switched distraction -> Start target alert
                self._stop_playback_internal()
                self._start_playback_internal(target_alert)
                self._current_alert = target_alert
                if config.DEBUG_LOGGING:
                    self._log(f"Alert started: {target_alert.upper()}")

    def start_alert(self, category: str) -> None:
        """Explicitly start an alert."""
        self.sync_alert_state(category)

    def stop_alert(self) -> None:
        """Explicitly stop any currently playing alert immediately."""
        with self._lock:
            self._stop_playback_internal()
            self._current_alert = None

    def stop_all(self) -> None:
        """Stop all audio."""
        self.stop_alert()

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _start_playback_internal(self, category: str) -> None:
        """Internal playback start."""
        sound = self._sound_objects.get(category)
        if sound is None:
            # Try on-demand load
            path = self._sound_paths.get(category)
            if path and path.exists():
                try:
                    offset = self._offsets.get(category, 0.0)
                    sound = self._load_sound_with_offset(path, offset)
                    self._sound_objects[category] = sound
                except Exception:
                    sound = None

        if sound is not None:
            try:
                # Play on dedicated channel (loops=-1 so audio stays ON as long as condition holds)
                if self._channel:
                    self._channel.play(sound, loops=-1)
                else:
                    sound.play(loops=-1)
            except Exception as e:
                print(f"[AlertManager] Error starting sound for '{category}': {e}")

    def _stop_playback_internal(self) -> None:
        """Internal instant playback stop."""
        try:
            if self._channel and self._channel.get_busy():
                self._channel.stop()
            import pygame
            pygame.mixer.stop()
        except Exception:
            pass

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable alert sounds at runtime."""
        self._enabled = enabled
        if not enabled:
            self.stop_alert()

    @property
    def is_playing(self) -> bool:
        """True if an alert is currently actively playing."""
        if not self._initialised or not self._enabled:
            return False
        if self._channel:
            return self._channel.get_busy()
        return self._current_alert is not None

    @property
    def current_alert(self) -> str | None:
        return self._current_alert

    @staticmethod
    def _log(msg: str) -> None:
        now_str = time.strftime("%H:%M:%S")
        print(f"[{now_str}] [AlertManager] {msg}")


# Alias for backward compatibility
AudioManager = AlertManager
