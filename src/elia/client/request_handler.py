import io
import requests
from elia.config import Config

def send_audio_and_get_result(wav_bytes: bytes, timeout=60) -> dict:
    """Invia l'audio al server di trascrizione e ritorna il risultato JSON."""
    files = {"audio": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    r = requests.post(Config.ENDPOINT_ASK, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()

def pay_attention() -> dict:
    """Invia una richiesta al server per attivare l'attenzione."""
    r = requests.post(Config.ENDPOINT_ATTENTION, timeout=60)
    r.raise_for_status()
    return r.json()

def get_report_full(timeout=60) -> dict:
    """Richiama l'endpoint del report emotivo completo."""
    r = requests.get(Config.ENDPOINT_REPORT_FULL, timeout=timeout)
    r.raise_for_status()
    return r.json()
