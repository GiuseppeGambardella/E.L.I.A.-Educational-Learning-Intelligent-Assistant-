import io
import time
import base64
import logging
import sounddevice as sd
import soundfile as sf

from elia.client.EventEmitter import EventEmitter
from elia.client.recorder import record_until_silence
from elia.config import Config
from elia.client.request_handler import send_audio_and_get_result, pay_attention

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

event_emitter = EventEmitter()


def check_attention():
    """Richiama l'endpoint /attention e mostra il messaggio."""
    logger.info("🔎 Avvio check attenzione...")
    try:
        result = pay_attention()
        if result.get("success"):
            logger.info("✅ Attenzione ricevuta con successo")
            print(result.get("message", "Tutto bene?\nTi vedo un po' distratto."))
        else:
            logger.warning("⚠️ Errore nell'attivazione dell'attenzione")
            print("❌ Errore nell'attivazione dell'attenzione:", result.get("error", "motivo sconosciuto"))
    except Exception as e:
        logger.exception("❌ Eccezione in check_attention")
        print("Errore inatteso:", str(e))


def on_wake_word_detected(**kwargs):
    """Si attiva quando viene rilevata la wake word."""
    logger.info("✅ Wake word rilevata: 'Ehi Elia' → avvio registrazione")
    while True:
        try:
            wav_bytes = record_until_silence()
            logger.info("🎙️ Registrazione completata, invio al server per trascrizione...")
            t0 = time.perf_counter()
            result = send_audio_and_get_result(wav_bytes)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"⏱️ Risposta dal server in {dt_ms:.2f} ms")

            if not result.get("success"):
                logger.error(f"❌ Errore risposta server: {result.get('error', 'motivo sconosciuto')}")
                print("❌ Errore nella ricezione della risposta:", result.get("error", "motivo sconosciuto"))
                return

            status = result.get("status", "ok")

            if status == "ok":
                logger.info("✅ Risposta status=ok ricevuta")
                message = result.get("message", "")
                print(f"💬 {message}")
                play_audio(result.get("audio"))
                return

            if status == "clarify":
                logger.info("🔄 Richiesta di chiarimento dal server")
                message = result.get("message", "")
                print(f"💬 {message}")
                play_audio(result.get("audio"))
                continue

            logger.warning(f"⚠️ Status sconosciuto ricevuto: {status}")
            return

        except Exception as e:
            logger.exception("❌ Eccezione in on_wake_word_detected")
            print("Errore inatteso durante la gestione della wake word:", str(e))
            return


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


# Bind degli eventi
event_emitter.on(event_emitter.WORD_DETECTED, on_wake_word_detected)
event_emitter.on(event_emitter.ATTENTION_CHECK, check_attention)
