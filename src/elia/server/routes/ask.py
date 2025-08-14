from flask import Blueprint, request, jsonify
import tempfile, os, logging, base64

from elia.server.services.asr import transcribe_wav
from elia.config import Config
from elia.server.models.llm import ask_llm
from elia.server.services.TTS import tts_create
from elia.server.services.sentiment_analysis import SentimentAnalyzer

bp = Blueprint("ask", __name__)
logger = logging.getLogger(__name__)

INCLUDE_THRESHOLD = 20

CLARIFY_PROMPT = (
    "Scrivi una sola frase, educata e concisa (MAX 15 PAROLE), che chieda di ripetere la domanda.\n"
    "Non aggiungere altro.\n"
    "Devi essere il piu sintetico possibile.\n"
)

CONTEXT_PROMPT = (
    "Sei un assistente virtuale di nome Elia (Educational Learning Intelligent Assistant) che aiuta gli studenti rispondendo alle loro domande.\n"
    "Quando ti salutano, ricambia il saluto in modo semplice.\n"
    "Rispondi solo in italiano, mantenendo tutti gli accenti corretti.\n"
    "Non usare emoji, solo testo puro.\n"
    "Adotta sempre un tono empatico e di supporto, calibrando la risposta allo stato emotivo dello studente.\n"
    "Rispetta il limite massimo di 120 parole.\n"
    "Sciogli sempre gli acronimi (esempio: d.C. -> dopo Cristo).\n"
    "Non inventare informazioni.\n"
    "Se la domanda contiene errori o imprecisioni, correggili e segnala la correzione nella risposta.\n"
    "Non attingere a dati esterni: usa solo le tue conoscenze interne e il contenuto della domanda.\n"
    "Non devi mai mentire o fornire informazioni false.\n"
    "Non devi usare caratteri volti ad evidenziare parole."
)

@bp.post("/ask")
def ask_endpoint():
    response = {}
    status_code = 200
    in_tmp_path = None

    try:
        if "audio" not in request.files:
            response = {"success": False, "error": "manca il file 'audio'"}
            status_code = 400
        else:
            f = request.files["audio"]
            if not f.filename:
                response = {"success": False, "error": "nome file vuoto"}
                status_code = 400
            else:
                # Salva file audio temporaneo
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                in_tmp_path = tmp.name
                tmp.close()
                f.save(in_tmp_path)

                # Trascrizione audio
                res = transcribe_wav(in_tmp_path)
                text = res.get("text", "") or ""
                confidence = res.get("confidence", None)

                # Copia del CONTEXT_PROMPT originale per non modificarlo globalmente
                base_context = CONTEXT_PROMPT  

                if confidence is not None and confidence < Config.ASR_CONF_THRESHOLD:
                    logger.info("LLM request: low ASR confidence, asking for clarification.")
                    llm_text = ask_llm(base_context, CLARIFY_PROMPT)
                    status = "clarify"
                else:
                    # Parte di intent recognition disabilitata
                    # top3_res = intent_recognition.get_top_three_intents(text)
                    # top3_intents = top3_res[0] if isinstance(top3_res, tuple) else top3_res
                    # tags = [it["label"] for it in top3_intents if it.get("score", 0) * 100 >= INCLUDE_THRESHOLD]
                    tags = []

                    # Analisi del sentiment
                    logger.info("Sentiment analysis in corso...")
                    sentiment_analyzer = SentimentAnalyzer()
                    attitudine = sentiment_analyzer.analyze(text)
                    logger.info(f"Sentiment principale: {attitudine.get('sentiment', '')}")

                    # Prepara il prompt per il LLM
                    normal_prompt = ((" ".join(tags) + " " + text).strip() if tags else text)
                    local_context = base_context + "\nL'attitudine dello studente è: " + attitudine.get("sentiment", "") + ", rispondi di conseguenza."

                    # Richiesta al LLM
                    logger.info("LLM request in corso...")
                    llm_text = ask_llm(local_context, normal_prompt)
                    status = "ok"
                
                # Pulizia risposta LLM
                llm_text = llm_text.replace("*", "")

                # Sintesi vocale della risposta
                audio_bytes, _ = tts_create(llm_text or "Non sono riuscito a capire la domanda, per favore ripeti.")

                # Risposta finale JSON
                response = {
                    "success": True,
                    "status": status,
                    "message": llm_text,
                    "confidence": confidence,
                    "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                }
    except Exception as e:
        logger.exception("Errore in /ask")
        response = {"success": False, "error": str(e)}
        status_code = 500
    finally:
        if in_tmp_path and os.path.exists(in_tmp_path):
            try:
                os.remove(in_tmp_path)
            except OSError:
                pass

    return jsonify(response), status_code