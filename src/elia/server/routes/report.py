import logging
from flask import Blueprint, jsonify
from elia.server.services.emotional_reports import generate_emotional_report

bp = Blueprint("report", __name__)
logger = logging.getLogger(__name__)


@bp.get("/emotional_report")
def emotional_report_endpoint():
    """
    Endpoint per generare un report emotivo basato sui dati memorizzati.
    """
    logger.info("🚀 Avvio richiesta report emotivo completo")
    try:
        logger.info("📊 Chiamata a generate_emotional_report()...")
        result = generate_emotional_report()
        
        logger.info(f"✅ generate_emotional_report() completato con status: {result.get('status', 'unknown')}")
        
        if result["status"] == "success":
            stats = result.get("statistics", {})
            total_interactions = stats.get("total_interactions", 0)
            sentiment_types = len(stats.get("sentiment_distribution", {}))
            
            logger.info(f"📈 Statistiche generate: {total_interactions} interazioni, {sentiment_types} tipi sentiment")
            logger.info("📝 Report LLM generato con successo")
            
            return jsonify({
                "success": True,
                "report": result["report"],
                "statistics": result["statistics"]
            }), 200
        else:
            logger.warning(f"⚠️ Report fallito: {result.get('message', 'Motivo sconosciuto')}")
            return jsonify({
                "success": False,
                "error": result["message"]
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Errore critico in /emotional_report: {str(e)}")
        logger.exception("Traceback completo:")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500