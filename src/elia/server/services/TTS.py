"""
API pubblica:
- tts_create(text: str) -> tuple[bytes, int]
    Ritorna i bytes WAV e il sample rate, pronti da inviare al client.
- tts_play(text: str) -> None
    Riproduce localmente (opzionale) usando sounddevice.
"""

from __future__ import annotations
import asyncio
import os
import tempfile
from typing import Optional, Tuple
import logging

import edge_tts
import sounddevice as sd
import soundfile as sf
from elia.config import Config

logger = logging.getLogger(__name__)

# Parametri di default (personalizzabili)
DEFAULT_PITCH = Config.DEFAULT_PITCH
DEFAULT_RATE = Config.DEFAULT_RATE


# =========================
# Funzioni interne
# =========================
async def _save_async(text: str, out_path: str, voice: str, rate: str, pitch: str) -> None:
    """
    Esegue la sintesi vocale in modalità asincrona con edge-tts
    e salva l’audio su file WAV.
    """
    com = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await com.save(out_path)


def _save_blocking(text: str, out_path: str, voice: str, rate: str, pitch: str) -> None:
    """
    Wrapper per eseguire la sintesi TTS in modo sincrono,
    anche se c’è già un event loop attivo (es. in ambiente async).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        logger.debug("Event loop già attivo → uso nuovo loop dedicato")
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(_save_async(text, out_path, voice, rate, pitch))
        finally:
            new_loop.close()
    else:
        asyncio.run(_save_async(text, out_path, voice, rate, pitch))


# =========================
# API pubblica
# =========================
def tts_create(text: str) -> Tuple[bytes, int]:
    """
    Sintetizza il testo usando la voce definita in Config.TTS_VOICE.
    Ritorna:
        (wav_bytes, sample_rate)
    Solleva eccezioni in caso di errore.
    """
    if not text:
        raise ValueError("tts_create: 'text' non può essere vuoto.")

    voice: Optional[str] = getattr(Config, "TTS_VOICE", None)
    if not voice:
        raise ValueError("tts_create: Config.TTS_VOICE non è impostato.")

    logger.info("🎤 Avvio sintesi vocale | Voice=%s | Text='%s...'", voice, text[:40])

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    try:
        _save_blocking(text, wav_path, voice, DEFAULT_RATE, DEFAULT_PITCH)

        # Carica l’audio e il sample rate
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        sr = sf.info(wav_path).samplerate

        logger.info("✅ Sintesi completata (durata campione ~%.2f sec)", len(wav_bytes) / (sr * 2))
        return wav_bytes, sr

    except Exception:
        logger.exception("❌ Errore durante la sintesi vocale")
        raise
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            logger.warning("Impossibile eliminare il file temporaneo %s", wav_path)


def tts_play(text: str) -> None:
    """
    Riproduce localmente il testo sintetizzato, in modo sincrono,
    usando il device audio di default.
    """
    wav_bytes, sr = tts_create(text)
    import io
    with sf.SoundFile(io.BytesIO(wav_bytes), mode="r") as f:
        data = f.read(dtype="float32", always_2d=False)
    sd.stop()
    sd.play(data, sr)
    sd.wait()
    logger.info("🔊 Riproduzione completata")
