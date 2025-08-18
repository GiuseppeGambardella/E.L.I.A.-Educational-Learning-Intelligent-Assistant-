import time
import logging
from elia.client.EventEmitter import EventEmitter
from elia.client.recorder import record_until_silence
from elia.client.request_handler import send_audio_and_get_result, pay_attention

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Istanza globale di EventEmitter
event_emitter = EventEmitter()


def check_attention():
    """Richiama l'endpoint /attention e ritorna il messaggio generato."""
    logger.info("🔎 Avvio check attenzione...")
    try:
        result = pay_attention()
        if result.get("success"):
            logger.info("✅ Attenzione ricevuta con successo")
            return {"status": "ok", "message": result.get("message")}
        else:
            logger.warning("⚠️ Errore nell'attivazione dell'attenzione")
            return {"status": "error", "error": result.get("error", "motivo sconosciuto")}
    except Exception as e:
        logger.exception("❌ Eccezione in check_attention")
        return {"status": "error", "error": str(e)}


def on_wake_word_detected(**kwargs):
    """Si attiva quando viene rilevata la wake word."""
    logger.info("✅ Wake word rilevata: 'Ehi Elia' → avvio registrazione")
    try:
        wav_bytes = record_until_silence()

        # Se arriva una tupla (bytes, sr), prendi solo i bytes
        if isinstance(wav_bytes, tuple):
            wav_bytes = wav_bytes[0]

        logger.info("🎙️ Registrazione completata, invio al server per trascrizione...")
        t0 = time.perf_counter()
        result = send_audio_and_get_result(wav_bytes)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"⏱️ Risposta dal server in {dt_ms:.2f} ms")

        if not result.get("success"):
            logger.error(f"❌ Errore risposta server: {result.get('error', 'motivo sconosciuto')}")
            return {"status": "error", "error": result.get("error", "motivo sconosciuto")}

        # Ritorno direttamente il dict che arriva dal server
        return result

    except Exception as e:
        logger.exception("❌ Eccezione in on_wake_word_detected")
        return {"status": "error", "error": str(e)}

# Bind degli eventi
event_emitter.on(event_emitter.WORD_DETECTED, on_wake_word_detected)
event_emitter.on(event_emitter.ATTENTION_CHECK, check_attention)
