from elia.client.EventEmitter import EventEmitter
from elia.client.recorder import record_until_silence
from elia.config import Config
from elia.client.request_handler import send_audio_and_get_result


event_emitter = EventEmitter()

def on_wake_word_detected(**kwargs):
    """Si attiva quando viene rilevata la wake word."""
    print("✅ Wake word trovata: 'Ehi Elia' → inizio registrazione")
    while True:
        wav_bytes = record_until_silence()
        print("🎙️ Registrazione completata, invio al server per trascrizione...")
        result = send_audio_and_get_result(wav_bytes)

        if not result.get("success"):
            print("❌ Errore nella ricezione della risposta:", result.get("error", "motivo sconosciuto"))
            return

        status = result.get("status", "ok")

        if status == "ok":
            print("✅ Risposta ricevuta")
            message = result.get("message", "")
            print(f"💬 {message}")
            return

        if status == "clarify":
            print(f"❓ {result.get('message', 'Non sono sicuro di aver capito perfettamente.')}")
            print("🎙️  Per favore ripeti: sto registrando di nuovo…")
            continue

event_emitter.on(event_emitter.WORD_DETECTED, on_wake_word_detected)
