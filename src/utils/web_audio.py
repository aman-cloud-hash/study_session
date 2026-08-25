import base64
from pathlib import Path
import config

def get_audio_base64_urls() -> dict[str, str]:
    """Return base64 data URLs for client-side instant meme audio playback."""
    res = {}
    try:
        p_drowsy = config.AUDIO_DIR / "uth jaa - EXTRA UNGLI WAALA (128k).mp3"
        if p_drowsy.exists():
            b64 = base64.b64encode(p_drowsy.read_bytes()).decode("utf-8")
            res["drowsy"] = f"data:audio/mp3;base64,{b64}"
    except Exception as e:
        print(f"[WebAudio] Drowsy audio load error: {e}")

    try:
        p_phone = config.AUDIO_DIR / "Rakh Teri Maa Ki Rakh Meme Template [7yF953fKPWA].mp3"
        if p_phone.exists():
            b64 = base64.b64encode(p_phone.read_bytes()).decode("utf-8")
            res["phone"] = f"data:audio/mp3;base64,{b64}"
    except Exception as e:
        print(f"[WebAudio] Phone audio load error: {e}")

    return res
