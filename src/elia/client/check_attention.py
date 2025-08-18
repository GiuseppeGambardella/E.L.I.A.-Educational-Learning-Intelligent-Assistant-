from elia.client.events import event_emitter

try:
    while True:
        tasto = input("Premi 'a' per per testare il richiamo dell'attenzione dell'agente: ")
        if tasto == 'a':
            result = event_emitter.emit(event_emitter.ATTENTION_CHECK)
            if result:
                status = result.get("status")
                if status == "ok":
                    print(f"💬 {result.get('message')}")
                elif status == "error":
                    print("⚠️ Errore:", result.get("error"))
                else:
                    print("⚠️ Risposta non riconosciuta:", result)
except KeyboardInterrupt:
    pass