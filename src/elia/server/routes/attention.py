from flask import Blueprint, jsonify
import logging
from elia.server.models import llm

bp = Blueprint("attention", __name__)
logger = logging.getLogger(__name__)

ATTENTION_PROMPT = (
    "Comportati come un professore. Lo studente si è distratto, "
    "richiamalo all'attenzione senza essere invasivo. MASSIMO 15 PAROLE. "
    "Non stai spiegando tu, stai soltanto controllando l'attenzione degli studenti, "
    "devi solo richiamarli all'attenzione."
)

@bp.post("/attention")
def attention_endpoint():
    logger.info("📥 Richiesta ricevuta su /attention")

    response = {}
    status_code = 200

    try:
        logger.debug(f"Prompt inviato all'LLM: {ATTENTION_PROMPT!r}")
        llm_result = llm.ask_llm(ATTENTION_PROMPT, None)

        logger.info("✅ Risposta ricevuta dall'LLM con successo")

        response = {
            "success": True,
            "message": llm_result
        }

    except Exception as e:
        logger.exception("❌ Errore durante l'elaborazione della richiesta /attention")

        response = {
            "success": False,
            "error": str(e),
            "message": "Si è verificato un errore nel generare la risposta."
        }
        status_code = 500

    return jsonify(response), status_code
