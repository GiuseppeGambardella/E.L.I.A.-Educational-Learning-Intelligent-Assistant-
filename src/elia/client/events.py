from elia.client.EventEmitter import EventEmitter

event_emitter = EventEmitter()

def on_wake_word_detected(**kwargs):
    print("✅ Wake word trovata: 'Ehi Elia' dentro evento")

event_emitter.on("wake_word_detected", on_wake_word_detected)
