"""
Script semplice per testare i report emotivi.
Due pulsanti: uno per report small, uno per report full.
"""

import requests
from elia.config import Config

ENDPOINT_REPORT_FULL = Config.ENDPOINT_REPORT_FULL
ENDPOINT_REPORT_SMALL = Config.ENDPOINT_REPORT_SMALL

def report_full():
    """Ottiene il report emotivo completo con analisi LLM"""
    print("\n📄 REPORT FULL - Analisi completa")
    print("=" * 40)
    print("🔄 Generazione in corso...")
    
    try:
        response = requests.get(ENDPOINT_REPORT_FULL)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Report generato!")
            
            # Mostra statistiche
            stats = data.get('statistics', {})
            print(f"\n📊 Totale interazioni: {stats.get('total_interactions', 0)}")
            
            # Mostra report
            print("\n📝 ANALISI DETTAGLIATA:")
            print("-" * 30)
            report_text = data.get('report', 'Nessun report')
            print(report_text)
                
        else:
            print(f"❌ Errore {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Server non raggiungibile")
    except Exception as e:
        print(f"❌ Errore: {e}")


def main():
    """Menu principale con due opzioni"""
    while True:
        print("\n" + "🎯" + "=" * 30 + "🎯")
        print("   REPORT EMOTIVI STUDENTI")
        print("🎯" + "=" * 30 + "🎯")
        print("1. 📄 Report Full (analisi completa)")
        print("2. ❌ Esci")
        
        choice = input("\n👉 Scegli (1-2): ").strip()
        
        if choice == "1":
            report_full()
        elif choice == "2":
            print("\n👋 Arrivederci!")
            break
        else:
            print("❌ Scelta non valida!")


if __name__ == "__main__":
    print("🚀 Avvio tester report emotivi...")
    main()
