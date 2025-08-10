import requests
from elia.client.EventEmitter import EventEmitter
from elia.client.recorder import record_until_silence
import io
from elia.config import Config
import requests

event_emitter = EventEmitter()

def on_wake_word_detected(**kwargs):
    print("✅ Wake word trovata: 'Ehi Elia' → inizio registrazione")
    wav_bytes = record_until_silence()

    try:
        files = {"audio": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
        r = requests.post(Config.ENDPOINT_TRANSCRIBE, files=files, timeout=60)
        r.raise_for_status()
        result = r.json()

        if result.get("success"):
            print("✅ Messaggio inviato correttamente")
        else:
            print("❌ Errore nella trascrizione:", result.get("error", "motivo sconosciuto"))

    except requests.exceptions.ConnectionError:
        print("❌ Errore: Impossibile connettersi al server di trascrizione")
    except requests.exceptions.Timeout:
        print("❌ Errore: Timeout della richiesta al server di trascrizione")
    except requests.exceptions.HTTPError as e:
        print(f"❌ Errore HTTP: {e.response.status_code} - {e.response.reason}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore nella richiesta: {str(e)}")
    except Exception as e:
        print(f"❌ Errore imprevisto durante la trascrizione: {str(e)}")


event_emitter.on(event_emitter.WORD_DETECTED, on_wake_word_detected)
