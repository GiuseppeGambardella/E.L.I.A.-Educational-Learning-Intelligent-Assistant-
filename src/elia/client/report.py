from elia.client.events import event_emitter

if __name__ == "__main__":
    print("🚀 Avvio tester report emotivi...")
    while True:
        print("\n" + "🎯" + "=" * 30 + "🎯")
        print("   REPORT EMOTIVI STUDENTI")
        print("🎯" + "=" * 30 + "🎯")
        print("1. 📄 Report Full (analisi completa)")
        print("2. ❌ Esci")
        
        choice = input("\n👉 Scegli (1-2): ").strip()
        
        if choice == "1":
            result = event_emitter.emit(event_emitter.REPORT_FULL)
            if result:
                status = result.get("status")
                if status == "ok":
                    print("\n✅ Report emotivo generato con successo!")
                    print("Report:", result.get("report"))
                    print("Statistiche:", result.get("statistics"))
                else:
                    print("\n❌ Errore nella generazione del report emotivo:", result.get("error"))
        elif choice == "2":
            print("\n👋 Arrivederci!")
            break
        else:
            print("❌ Scelta non valida!")