"""
Modulo dedicato alla generazione dei report emotivi.
Separato dalla logica del database per mantenere le responsabilità distinte.
"""

import logging
from elia.server.memory.memory import get_all_emotional_data
from elia.server.models.llm import ask_llm
from elia.config import Config

logger = logging.getLogger(__name__)


EMOTIONAL_REPORT_PROMPT = Config.EMOTIONAL_REPORT_PROMPT
ANALYSIS_PROMPT = Config.ANALYSIS_EXPERT_PROMPT

def generate_emotional_report():
    """
    Genera un report emotivo completo basato sui dati memorizzati.
    Utilizza get_all_emotional_data() per recuperare i dati dal database.
    """
    logger.info("🚀 Avvio generazione report emotivo")
    
    try:
        # Recupera i dati dal database
        logger.info("📊 Recupero dati dal database...")
        db_result = get_all_emotional_data()
        
        if db_result["status"] == "empty":
            logger.warning("📭 Database vuoto, impossibile generare report")
            return {"status": "error", "message": "Nessun dato disponibile per il report"}
        
        if db_result["status"] == "error":
            logger.error(f"❌ Errore nel recupero dati: {db_result['message']}")
            return {"status": "error", "message": f"Errore database: {db_result['message']}"}
        
        # Estrai i dati
        data = db_result["data"]
        documents = data["documents"]
        emotional_reports = data["emotional_reports"]
        total_interactions = data["total_interactions"]
        valid_reports = data["valid_emotional_reports"]
        
        logger.info(f"✅ Dati recuperati: {total_interactions} interazioni, {valid_reports} report emotivi")
        
        # Prepara il sample delle interazioni per l'LLM
        logger.info("🔍 Preparazione sample per LLM...")
        data_summary = []
        
        for i, (question, report) in enumerate(zip(documents, emotional_reports)):
            # Limita a primi 50 per non sovraccaricare l'LLM
            if i < 50:
                question_preview = question[:100] + "..." if len(question) > 100 else question
                report_preview = report[:150] + "..." if len(report) > 150 else report
                data_summary.append(f"Domanda: {question_preview} | Report emotivo: {report_preview}")

        logger.info(f"📝 Sample preparato: {len(data_summary)} esempi per LLM")
        
        # Costruisci il prompt per l'LLM
        prompt = f"""
            {EMOTIONAL_REPORT_PROMPT}

            STATISTICHE GENERALI:
            - Totale interazioni: {total_interactions}
            - Report emotivi validi: {valid_reports}

            SAMPLE DELLE INTERAZIONI CON REPORT EMOTIVI:
            {chr(10).join(data_summary)}
            """
        
        logger.info("🤖 Invio prompt a LLM per generazione report...")
        
        # Genera il report con l'LLM
        report = ask_llm(ANALYSIS_PROMPT, prompt)
        
        logger.info("✅ Report generato con successo")
        
        return {
            "status": "success",
            "report": report,
            "statistics": {
                "total_interactions": total_interactions,
                "valid_emotional_reports": valid_reports
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Errore critico nella generazione report: {e}")
        logger.exception("Traceback completo:")
        return {"status": "error", "message": str(e)}

