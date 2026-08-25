"""
Sound Generator Utility
=======================

Generates synthesized clean alert WAV files using Python's standard `wave` and `math` modules.
Ensures the app works out-of-the-box with no missing audio assets.
"""

import math
import struct
import wave
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def generate_tone(
    filename: Path,
    frequencies: list[float],
    durations: list[float],
    volume: float = 0.5,
    sample_rate: int = 44100,
) -> None:
    """Generate a sequence of sine wave tones and save to a .wav file."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    total_frames = []

    for freq, duration in zip(frequencies, durations):
        n_samples = int(sample_rate * duration)
        for i in range(n_samples):
            # Fade in/out envelope to avoid audio clicks
            envelope = 1.0
            attack_samples = int(sample_rate * 0.02)
            release_samples = int(sample_rate * 0.02)
            if i < attack_samples:
                envelope = i / attack_samples
            elif i > n_samples - release_samples:
                envelope = (n_samples - i) / release_samples

            if freq == 0:
                sample_val = 0.0
            else:
                sample_val = math.sin(2.0 * math.pi * freq * (i / sample_rate))

            val = int(sample_val * volume * envelope * 32767.0)
            val = max(-32768, min(32767, val))
            total_frames.append(val)

    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        raw_data = struct.pack(f"<{len(total_frames)}h", *total_frames)
        wav_file.writeframes(raw_data)


def create_all_default_sounds() -> None:
    """Generate all standard alert tones if they don't exist."""
    sounds_dir = config.SOUNDS_DIR
    sounds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Drowsiness alert (Gentle repeating dual beep: 520Hz -> 650Hz)
    drowsy_path = sounds_dir / config.SOUND_DROWSINESS
    if not drowsy_path.exists():
        generate_tone(
            drowsy_path,
            frequencies=[523.25, 0, 659.25, 0, 523.25, 0, 659.25],
            durations=[0.18, 0.08, 0.22, 0.1, 0.18, 0.08, 0.3],
            volume=0.6,
        )

    # 2. Phone alert (Attention alert: 880Hz -> 440Hz pulse)
    phone_path = sounds_dir / config.SOUND_PHONE
    if not phone_path.exists():
        generate_tone(
            phone_path,
            frequencies=[880.0, 0, 784.0, 0, 880.0],
            durations=[0.15, 0.08, 0.15, 0.08, 0.25],
            volume=0.6,
        )

    # 3. High distraction alert (Triple urgency chime: 988Hz -> 880Hz -> 784Hz)
    high_path = sounds_dir / config.SOUND_HIGH_DISTRACTION
    if not high_path.exists():
        generate_tone(
            high_path,
            frequencies=[987.77, 0, 880.0, 0, 783.99, 0, 659.25],
            durations=[0.12, 0.06, 0.12, 0.06, 0.15, 0.06, 0.35],
            volume=0.7,
        )

    # 4. Session complete chime (Pleasant ascending completion chord: C -> E -> G -> C high)
    complete_path = sounds_dir / config.SOUND_SESSION_COMPLETE
    if not complete_path.exists():
        generate_tone(
            complete_path,
            frequencies=[523.25, 659.25, 783.99, 1046.50],
            durations=[0.2, 0.2, 0.25, 0.6],
            volume=0.7,
        )


if __name__ == "__main__":
    create_all_default_sounds()
    print("Default alert sounds generated successfully.")
