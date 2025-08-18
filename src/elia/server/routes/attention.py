from flask import Blueprint, jsonify, request
import logging
from elia.server.models import llm

bp = Blueprint("attention", __name__)
logger = logging.getLogger(__name__)

# Prompt costante per l'LLM
ATTENTION_PROMPT = (
    "Comportati come un professore. Lo studente si è distratto, "
    "richiamalo all'attenzione senza essere invasivo. MASSIMO 15 PAROLE. "
    "Non stai spiegando tu, stai soltanto controllando l'attenzione degli studenti, "
    "devi solo richiamarli all'attenzione."
)


@bp.post("/attention")
def attention_endpoint():
    """
    Endpoint per richiamare l'attenzione dello studente.
    Utilizza un LLM per generare un messaggio breve e non invasivo.
    """
    logger.info("📥 Richiesta ricevuta su /attention")

    try:
        # Tracciamo dettagli della richiesta
        client_ip = request.remote_addr
        logger.debug(f"Richiesta da IP: {client_ip}")

        # Invio del prompt all'LLM
        logger.debug("Invio del prompt all'LLM...")
        llm_result = llm.ask_llm(ATTENTION_PROMPT, context=None)

        if not llm_result or not isinstance(llm_result, str):
            logger.warning("⚠️ Risposta LLM vuota o non valida")
            return jsonify({
                "success": False,
                "error": "Risposta LLM non valida",
                "message": "Impossibile generare un messaggio di attenzione."
            }), 502  # Bad Gateway → l'LLM non ha risposto come atteso

        return jsonify({
            "success": True,
            "message": llm_result.strip()
        }), 200

    except Exception as e:
        logger.exception("❌ Errore durante l'elaborazione della richiesta /attention")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Si è verificato un errore nel generare la risposta."
        }), 500
