import io
import time
from elia.client.EventEmitter import EventEmitter
from elia.client.recorder import record_until_silence
from elia.config import Config
from elia.client.request_handler import send_audio_and_get_result
import base64
import sounddevice as sd
import soundfile as sf


event_emitter = EventEmitter()

def on_wake_word_detected(**kwargs):
    """Si attiva quando viene rilevata la wake word."""
    print("✅ Wake word trovata: 'Ehi Elia' → inizio registrazione")
    while True:
        wav_bytes = record_until_silence()
        print("🎙️ Registrazione completata, invio al server per trascrizione...")
        t0 = time.perf_counter()
        result = send_audio_and_get_result(wav_bytes)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"⏱️ {dt_ms:.2f} ms")

        if not result.get("success"):
            print("❌ Errore nella ricezione della risposta:", result.get("error", "motivo sconosciuto"))
            return

        status = result.get("status", "ok")

        if status == "ok":
            print("✅ Risposta ricevuta")
            message = result.get("message", "")
            print(f"💬 {message}")
            play_audio(result.get("audio"))
            return

        if status == "clarify":
            print("✅ Richiesta di ripetizione")
            message = result.get("message", "")
            print(f"💬 {message}")
            play_audio(result.get("audio"))
            continue


def play_audio(audio):
    audio_b64 = audio
    if audio_b64:
        # Decodifica Base64 in bytes
        audio_bytes = base64.b64decode(audio_b64)

        # Leggi il WAV dai bytes e riproduci
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            data = f.read(dtype="float32", always_2d=False)
            sr = f.samplerate
        sd.stop()
        sd.play(data, sr)
    if not audio:
        print("⚠️ Nessun audio da riprodurre (audio è None o vuoto).")
        return
    audio_b64 = audio
    # Decodifica Base64 in bytes
    audio_bytes = base64.b64decode(audio_b64)

    # Leggi il WAV dai bytes e riproduci
    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        data = f.read(dtype="float32", always_2d=False)
        sr = f.samplerate
    sd.stop()
    sd.play(data, sr)
    sd.wait()

event_emitter.on(event_emitter.WORD_DETECTED, on_wake_word_detected)
