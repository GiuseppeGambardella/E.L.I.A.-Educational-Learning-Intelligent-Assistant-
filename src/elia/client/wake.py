import os
import pvporcupine
from pvrecorder import PvRecorder
from elia.config import Config
from elia.client.events import event_emitter
from elia.client.services.audio import play_audio

ACCESS_KEY = Config.PICOVOICE_KEY
KEYWORD_PATH = Config.PICOVOICE_WORD  # .ppn della keyword "Ehi Elia"

if not ACCESS_KEY:
    raise RuntimeError("PICOVOICE_KEY mancante nel .env")
if not KEYWORD_PATH or not os.path.isfile(KEYWORD_PATH):
    raise FileNotFoundError(f"Keyword non trovata: {KEYWORD_PATH}")

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[KEYWORD_PATH],
    model_path=Config.PICOVOICE_PARAMS,
)
rec = PvRecorder(device_index=Config.AUDIO_DEVICE_INDEX, frame_length=porcupine.frame_length)

print("🎤 Di' “Ehi Elia” (CTRL+C per uscire)")
rec.start()
try:
    while True:
        pcm = rec.read()
        if porcupine.process(pcm) >= 0:
            result = event_emitter.emit(event_emitter.WORD_DETECTED)
            if result:
                status = result.get("status")
                if status == "ok":
                    print(f"💬 {result.get('message')}")
                    play_audio(result.get("audio"))
                elif status == "clarify":
                    print(f"💬 {result.get('message')}")
                    play_audio(result.get("audio"))
                    print("🔄 Chiarimento richiesto, sto registrando...")
                    continue
                else:
                    print("⚠️ Errore:", result.get("error"))
except KeyboardInterrupt:
    pass
finally:
    rec.stop(); rec.delete(); porcupine.delete()
