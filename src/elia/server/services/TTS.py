"""
API pubblica:
- tts_create(text: str) -> tuple[bytes, int]
    Ritorna i bytes WAV e il sample rate, pronti da inviare al client.
- tts_play(text: str) -> None
    Riproduce localmente (opzionale) usando sounddevice.
"""

import asyncio
import io
import logging
from typing import Optional, Tuple

import edge_tts
import sounddevice as sd
import soundfile as sf
from elia.config import Config

logger = logging.getLogger(__name__)

DEFAULT_PITCH = Config.DEFAULT_PITCH
DEFAULT_RATE = Config.DEFAULT_RATE


# =========================
# Funzioni interne
# =========================
async def _synthesize_async(text: str, voice: str, rate: str, pitch: str) -> bytes:
    """
    Esegue la sintesi vocale in memoria (senza scrivere su disco).
    Ritorna i bytes WAV.
    """
    com = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    # edge-tts di base salva su file, ma possiamo catturare l’audio in stream
    audio_chunks = []
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


def _synthesize_blocking(text: str, voice: str, rate: str, pitch: str) -> bytes:
    """
    Wrapper sincrono per eseguire la sintesi.
    Usa un nuovo loop se ne esiste già uno attivo.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        logger.debug("Loop attivo → uso loop separato per TTS")
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(_synthesize_async(text, voice, rate, pitch))
        finally:
            new_loop.close()
    else:
        return asyncio.run(_synthesize_async(text, voice, rate, pitch))


# =========================
# API pubblica
# =========================
def tts_create(text: str) -> Tuple[bytes, int]:
    """
    Sintetizza il testo usando la voce definita in Config.TTS_VOICE.
    Ritorna:
        (wav_bytes, sample_rate)
    """
    if not text:
        raise ValueError("tts_create: 'text' non può essere vuoto.")

    voice: Optional[str] = getattr(Config, "TTS_VOICE", None)
    if not voice:
        raise ValueError("tts_create: Config.TTS_VOICE non è impostato.")

    logger.info("🎤 Avvio sintesi vocale | Voice=%s | Text='%s...'", voice, text[:40])

    try:
        wav_bytes = _synthesize_blocking(text, voice, DEFAULT_RATE, DEFAULT_PITCH)

        # Usa soundfile per ricavare info
        with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
            sr = f.samplerate
            duration = f.frames / sr

        logger.info("✅ Sintesi completata (durata ~%.2f sec)", duration)
        return wav_bytes, sr

    except Exception:
        logger.exception("❌ Errore durante la sintesi vocale")
        raise


def tts_play(text: str) -> None:
    """
    Riproduce localmente il testo sintetizzato.
    """
    wav_bytes, sr = tts_create(text)
    with sf.SoundFile(io.BytesIO(wav_bytes), mode="r") as f:
        data = f.read(dtype="float32", always_2d=False)
    sd.stop()
    sd.play(data, sr)
    sd.wait()
    logger.info("🔊 Riproduzione completata")
