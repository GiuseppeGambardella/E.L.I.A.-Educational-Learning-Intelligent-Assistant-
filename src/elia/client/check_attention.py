import os
from elia.config import Config
from elia.client.events import event_emitter

try:
    while True:
        tasto = input("Premi 'a' per per testare il richiamo dell'attenzione dell'agente: ")
        if tasto == 'a':
            event_emitter.emit(event_emitter.ATTENTION_CHECK)
except KeyboardInterrupt:
    pass