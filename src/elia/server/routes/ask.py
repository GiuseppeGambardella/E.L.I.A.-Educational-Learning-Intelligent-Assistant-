import time
from flask import Blueprint, request, jsonify
import tempfile, os, logging, base64
from concurrent.futures import ThreadPoolExecutor

from elia.server.services.asr import transcribe_wav
from elia.config import Config
from elia.server.models.llm import ask_llm
from elia.server.services.TTS import tts_create
from elia.server.services.sentiment_analysis import SentimentAnalyzer
from elia.server.memory.memory import search as chroma_search
from elia.server.memory.memory import add_qa

bp = Blueprint("ask", __name__)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=2)

# INCLUDE_THRESHOLD = 20 # sensibilità della classificazione degli intenti, disabilitato
SIMILARITY_THRESHOLD = 0.7  # sensibilità della similarità

CLARIFY_PROMPT = (
    "Comportati come se non avessi capito. Scrivi una sola frase, educata e concisa (MAX 15 PAROLE), che chieda di ripetere.\n"
    "Non aggiungere altro.\n"
    "Devi essere il piu sintetico possibile.\n"
)

CONTEXT_PROMPT = (
    "Sei un assistente virtuale di nome Elia (Educational Learning Intelligent Assistant) che aiuta gli studenti rispondendo alle loro domande.\n"
    "Quando ti salutano, ricambia il saluto in modo semplice.\n"
    "Non ricordare sempre che sei un assistente vocale volto all'educamento, ma rispondi solo quando ti viene espressamente chiesto.\n"
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
            return jsonify({"success": False, "error": "manca il file 'audio'"}), 400

        f = request.files["audio"]
        if not f.filename:
            return jsonify({"success": False, "error": "nome file vuoto"}), 400

        # ========================
        # 1. Salva file audio temporaneo
        # ========================
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        in_tmp_path = tmp.name
        tmp.close()
        f.save(in_tmp_path)

        # ========================
        # 2. Trascrizione audio
        # ========================
        res = transcribe_wav(in_tmp_path)
        text = res.get("text", "") or ""
        confidence = res.get("confidence", None)

        base_context = CONTEXT_PROMPT  

        if confidence is not None and confidence < Config.ASR_CONF_THRESHOLD:
            logger.info("LLM request: low ASR confidence, asking for clarification.")
            llm_text = ask_llm(base_context, CLARIFY_PROMPT)
            status = "clarify"
            similar_qas = []
            attitudine = {"sentiment": ""}
        else:
            # ========================
            # 3. Parallelizza sentiment + ricerca memoria
            # ========================
            sentiment_analyzer = SentimentAnalyzer()
            future_sentiment = executor.submit(sentiment_analyzer.analyze, text)
            future_chroma = executor.submit(chroma_search, text, 1)  # top_k=1

            attitudine = future_sentiment.result()
            similar_qas = future_chroma.result()

            logger.info(f"Sentiment principale: {attitudine.get('sentiment', '')}")

            # Controllo soglia similarità
            if similar_qas and similar_qas[0]["similarità"] >= SIMILARITY_THRESHOLD:
                logger.info(f"Memoria accettata (similarità {similar_qas[0]['similarità']})")
            else:
                logger.info("Nessuna memoria rilevante trovata → contesto vuoto")
                similar_qas = []
            # ========================
            # 4. Prepara prompt LLM
            # ========================
            memoria_context = ""
            for qa in similar_qas:
                memoria_context += f"\nDomanda passata: {qa['domanda_simile']} | Risposta: {qa['risposta_passata']}"

            local_context = (
                base_context
                + "\nL'attitudine dello studente è: "
                + attitudine.get("sentiment", "")
                + ", rispondi di conseguenza."
                + "\nMemoria passata utile (se rilevante):"
                + memoria_context
                + "\n Rispondi in maniera coerente con quello che hai detto prima."
            )

            normal_prompt = text
            logger.info("LLM request in corso...")
            llm_text = ask_llm(local_context, normal_prompt)
            status = "ok"

            if not similar_qas or similar_qas[0]["similarità"] < 1:
                executor.submit(add_qa, text, llm_text)
                logger.info("Avviato salvataggio QA in memoria (thread separato).")
            else:
                logger.info("QA già presente in memoria (similarità=1) → non salvata.")
        # ========================
        # 5. Pulizia + TTS
        # ========================
        llm_text = (llm_text or "").replace("*", "")
        start_time = time.perf_counter()
        audio_bytes, _ = tts_create(llm_text or "Non sono riuscito a capire la domanda, per favore ripeti.")
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        logger.info("TTS completato in %.3f secondi", elapsed)

        # ========================
        # 6. Risposta finale
        # ========================
        response = {
            "success": True,
            "status": status,
            "message": llm_text,
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
