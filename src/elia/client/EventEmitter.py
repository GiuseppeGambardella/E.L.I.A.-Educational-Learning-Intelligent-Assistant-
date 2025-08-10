# EventEmitter minimale, thread-safe "quanto basta" per callbacks veloci.
from collections import defaultdict
from typing import Callable, Dict, List, Any
import traceback
import datetime

class EventEmitter:
    def __init__(self, log_file="src/elia/client/events.log"):
        self._events = {}
        self._log_file = log_file  # percorso file log, es. "events.log"

    def on(self, event, handler):
        """Registra una funzione da eseguire quando l'evento è emesso."""
        self._events.setdefault(event, []).append(handler)

    def emit(self, event, *args, **kwargs):
        """Esegue tutte le funzioni associate all'evento."""
        # Scrivi nel log se richiesto
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] EVENT: {event} ARGS: {args} KWARGS: {kwargs}\n")

        # Esegui tutti i gestori registrati
        for handler in self._events.get(event, []):
            handler(*args, **kwargs)
