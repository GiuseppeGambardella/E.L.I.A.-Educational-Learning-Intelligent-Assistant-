import base64
import sounddevice as sd
import soundfile as sf
import io
import logging

logger = logging.getLogger(__name__)

def play_audio(audio):
    """Decodifica audio Base64 e lo riproduce."""
    if not audio:
        logger.warning("⚠️ Nessun audio ricevuto da riprodurre")
        return

    try:
        # Decodifica Base64 in bytes
        audio_bytes = base64.b64decode(audio)

        # Leggi il WAV dai bytes e riproduci
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            data = f.read(dtype="float32", always_2d=False)
            sr = f.samplerate

        sd.stop()
        sd.play(data, sr)
        sd.wait()
        logger.info("🔊 Riproduzione audio completata")

    except Exception as e:
        logger.exception("❌ Errore durante la riproduzione audio")
        print("Errore durante la riproduzione audio:", str(e))


