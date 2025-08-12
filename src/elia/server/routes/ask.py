from flask import Blueprint, request, jsonify
import tempfile, os, logging, base64

from elia.server.services.asr import transcribe_wav
from elia.config import Config
from elia.server.models.llm import ask_llm
from elia.server.models import intent_recognition
from elia.server.services.TTS import tts_create

bp = Blueprint("ask", __name__)
logger = logging.getLogger(__name__)

voice = Config.TTS_VOICE

INCLUDE_THRESHOLD = 20

CLARIFY_PROMPT = (
    "Scrivi una sola frase, educata e concisa (MAX 15 PAROLE), che chieda di ripetere la domanda.\n"
    "Non aggiungere altro.\n"
    "Devi essere il piu sintetico possibile.\n"
)

CONTEXT_PROMPT = (
    "Sei un assistente virtuale di nome Elia (Educational Learning Intelligent Assistant) che aiuta gli studenti rispondendo alle loro domande.\n"
    "Rispondi solo in italiano, mantenendo tutti gli accenti corretti.\n"
    "Non usare emoji, solo testo puro.\n"
    "Adotta sempre un tono empatico e di supporto, calibrando la risposta allo stato emotivo dello studente.\n"
    "Rispetta il limite massimo di 120 parole.\n"
    "Sciogli sempre gli acronimi (esempio: d.C. -> dopo Cristo).\n"
    "Non inventare informazioni.\n"
    "Se la domanda contiene errori o imprecisioni, correggili e segnala la correzione nella risposta.\n"
    "Non attingere a dati esterni: usa solo le tue conoscenze interne e il contenuto della domanda.\n"
)

@bp.post("/ask")
def ask_endpoint():
    response = {}
    status_code = 200
    in_tmp_path = None
    tts_tmp_path = None

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
                in_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                in_tmp_path = in_tmp.name
                f.save(in_tmp_path)

                res = transcribe_wav(in_tmp_path)
                text = res.get("text", "") or ""
                confidence = res.get("confidence", None)

                if confidence is not None and confidence < Config.ASR_CONF_THRESHOLD:
                    llm_text = ask_llm(CONTEXT_PROMPT, CLARIFY_PROMPT)
                    status = "clarify"
                else:
                    #top3_res = intent_recognition.get_top_three_intents(text)
                    #top3_intents = top3_res[0] if isinstance(top3_res, tuple) else top3_res
                    #tags = [it["label"] for it in top3_intents if it.get("score", 0) * 100 >= INCLUDE_THRESHOLD]
                    tags=[]
                    normal_prompt = ((" ".join(tags) + " " + text).strip() if tags else text)
                    llm_text = ask_llm(CONTEXT_PROMPT, normal_prompt)
                    llm_text = llm_text.replace("**", "")
                    status = "ok"

                audio_file, _ = tts_create(llm_text)  # usa Config.TTS_VOICE, sintetizza, riproduce e pulisce

                response = {
                    "success": True,
                    "status": status,
                    "message": llm_text,
                    "confidence": confidence,
                    "audio": base64.b64encode(audio_file).decode("utf-8")
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
        if tts_tmp_path and os.path.exists(tts_tmp_path):
            try:
                os.remove(tts_tmp_path)
            except OSError:
                pass

    return jsonify(response), status_code