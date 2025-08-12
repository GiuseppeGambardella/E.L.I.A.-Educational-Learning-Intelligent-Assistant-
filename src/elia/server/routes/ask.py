from flask import Blueprint, request, jsonify
import tempfile, os, logging

from elia.server.services.asr import transcribe_wav
from elia.config import Config
from elia.server.models.llm import ask_llm
from elia.server.models import intent_recognition

bp = Blueprint("transcribe", __name__)
logger = logging.getLogger(__name__)

INCLUDE_THRESHOLD = 20  # percentuale

CLARIFY_PROMPT = (
    "Lo studente ha fatto una domanda che tu non hai capito bene. "
    "Scrivi una sola frase, educata e concisa (max 15 parole), che chieda di ripeterla. "
    "Non aggiungere altro."
)

CONTEXT_PROMPT = (
    "Sei un assistente virtuale che aiuta gli studenti, rispondendo alle loro domande.\n"
    "Se il prompt è vuoto, non rispondere.\n"
    "Il tuo nome è Elia, e devi rispondere in maniera adeguata agli studenti in base al loro stato emotivo e al contesto della conversazione.\n"
    "Dato che sei un agente virtuale, non hai accesso a informazioni personali sugli studenti, quindi fai del tuo meglio per fornire risposte utili basate solo sulle informazioni fornite nella conversazione.\n"
    "Cerca di mantenere un tono empatico e di supporto.\n"
    "Se ti vengono fatte domande su argomenti che non conosci, è meglio ammettere la tua ignoranza piuttosto che inventare risposte.\n"
    "Se ti vengono fatte domande personali su di te o su qualsiasi cosa non inerente all'apprendimento, è meglio evitare di rispondere e reindirizzare la conversazione verso l'argomento principale.\n"
    "Devi scrivere tutto rigorosamente in italiano.\n"
    "La risposta deve essere breve e mirata. MASSIMO 120 PAROLE.\n"
    "Avrai delle parole chiave che ti aiuteranno a capire il senso della richiesta:\n"
    "ask_explanation -> Spiegami meglio.\n"
    "ask_definition -> Definisci meglio.\n"
    "ask_example -> Fai un esempio.\n"
    "request_summary -> Fai un riassunto.\n"
    "ask_steps -> Fai un elenco dei passaggi.\n"
    "ask_compare -> Fai un confronto.\n"
    "ask_simplify -> Semplifica.\n"
    "express_emotion -> Lo studente ha dimostrato un'emozione.\n"
    "express_difficulty -> Lo studente ha dimostrato una difficoltà.\n"
    "In base a questi tag, devi riconoscere come fare.\n"
    "Quando non ci sono devi rispondere normalmente.\n"
)

@bp.post("/ask")
def ask_endpoint():
    response = {}
    status_code = 200
    tmp_path = None

    try:
        # Validazione input
        if "audio" not in request.files:
            response = {"success": False, "error": "manca il file 'audio'"}
            status_code = 400
        else:
            f = request.files["audio"]
            if not f.filename:
                response = {"success": False, "error": "nome file vuoto"}
                status_code = 400
            else:
                # Salvataggio temporaneo
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                f.save(tmp_path)

                # ASR
                res = transcribe_wav(tmp_path)
                text = res.get("text", "") or ""
                confidence = res.get("confidence", None)

                # Caso chiarimento (bassa confidenza)
                if confidence is not None and confidence < Config.ASR_CONF_THRESHOLD:
                    # Mantengo l'ordine dei parametri come nel tuo codice originale
                    ask_llm_result = ask_llm(CONTEXT_PROMPT, CLARIFY_PROMPT)
                    response = {
                        "success": True,
                        "status": "clarify",
                        "message": ask_llm_result,
                        "confidence": confidence,
                    }
                else:
                    # Top-3 intenti
                    top3_res = intent_recognition.get_top_three_intents(text)
                    top3_intents = top3_res[0] if isinstance(top3_res, tuple) else top3_res

                    # Tag con score >= 20%
                    tags = [it["label"] for it in top3_intents if it.get("score", 0) * 100 >= INCLUDE_THRESHOLD]
                    normal_prompt = ((" ".join(tags) + " " + text).strip() if tags else text)

                    # Chiamata LLM (mantengo l'ordine dei parametri come nel tuo codice)
                    ask_llm_result = ask_llm(CONTEXT_PROMPT, normal_prompt)

                    response = {
                        "success": True,
                        "status": "ok",
                        "message": ask_llm_result,
                        "confidence": confidence,
                    }

    except Exception as e:
        logger.exception("Errore in /ask")
        response = {"success": False, "error": str(e)}
        status_code = 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return jsonify(response), status_code