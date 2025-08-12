"""
API pubblica: 
- tts_say(text: str) -> tuple[bytes, int]
    Ritorna i bytes WAV e il sample rate, pronti da inviare al client.
- tts_play(text: str) -> None
    (Opzionale) Riproduce localmente usando sounddevice.
"""

from __future__ import annotations
import asyncio
import os
import tempfile
from typing import Optional, Tuple

import edge_tts
import sounddevice as sd
import soundfile as sf
from elia.config import Config

pitch = "-15Hz"
rate = "+10%"

async def _save_async(text: str, out_path: str, voice: str, rate: str, pitch: str) -> None:
    com = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,      # es. "-10%" per rallentare
        pitch=pitch,    # es. "-2st" per tono più grave
    )
    await com.save(out_path)


def _save_blocking(text: str, out_path: str, voice: str, rate: str) -> None:
    """Gestisce anche il caso in cui ci sia già un event loop attivo."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(_save_async(text, out_path, voice, rate, pitch))
        finally:
            new_loop.close()
    else:
        asyncio.run(_save_async(text, out_path, voice, rate, pitch))



def tts_create(text: str) -> Tuple[bytes, int]:
    """
    Sintetizza 'text' usando la voce in Config.TTS_VOICE e RITORNA l'audio.
    Ritorna: (wav_bytes, sample_rate)

    - Usa un file WAV temporaneo e lo carica in memoria.
    - NON riproduce localmente.
    - Solleva eccezioni in caso di errori.
    """
    if not text:
        raise ValueError("tts_say: 'text' non può essere vuoto.")

    voice: Optional[str] = getattr(Config, "TTS_VOICE", None)
    if not voice:
        raise ValueError("tts_say: Config.TTS_VOICE non è impostato.")
  # essenziale/minimale; si può estendere in futuro

    # 1) Sintesi su file temporaneo
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    try:
        _save_blocking(text, wav_path, voice, rate)

        # 2) Leggi i bytes e il sample rate
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        # Usa soundfile.info per recuperare il sample rate senza decodificare tutto
        info = sf.info(wav_path)
        sr = info.samplerate

        return wav_bytes, sr
    finally:
        # 3) Pulizia
        try:
            os.remove(wav_path)
        except OSError:
            pass


# (Opzionale) utility per mantenere la vecchia riproduzione locale, se serve.
def tts_play(text: str) -> None:
    """Riproduce localmente (sincrono) usando il device di default."""
    wav_bytes, sr = tts_create(text)
    # Decodifica dai bytes per la riproduzione
    # soundfile può leggere da bytes via SoundFile(file=io.BytesIO(...)), ma per semplicità:
    import io
    with sf.SoundFile(io.BytesIO(wav_bytes), mode="r") as f:
        data = f.read(dtype="float32", always_2d=False)
    sd.stop()
    sd.play(data, sr)
    sd.wait()
