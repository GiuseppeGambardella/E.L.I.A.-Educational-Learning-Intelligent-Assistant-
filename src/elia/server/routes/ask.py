import time
import tempfile
import os
import logging
import base64
from flask import Blueprint, request, jsonify
from concurrent.futures import ThreadPoolExecutor

from elia.server.services.asr import transcribe_bytes
from elia.config import Config
from elia.server.models.llm import ask_llm
from elia.server.services.TTS import tts_create
from elia.server.services.sentiment_analysis import SentimentAnalyzer
from elia.server.memory.memory import search as chroma_search, add_qa

bp = Blueprint("ask", __name__)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=4)
sentiment_analyzer = SentimentAnalyzer()

SIMILARITY_THRESHOLD = Config.SIMILARITY_THRESHOLD

CLARIFY_PROMPT = Config.CLARIFY_PROMPT

CONTEXT_PROMPT = Config.CONTEXT_PROMPT

# ================================
# Helper functions
# ================================

def save_temp_audio(file_storage) -> str:
    """Salva il file audio temporaneamente e restituisce il path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = tmp.name
    tmp.close()
    file_storage.save(path)
    return path

def cleanup_temp(path: str):
    """Elimina il file temporaneo se esiste."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
            logger.debug("File temporaneo %s eliminato", path)
        except OSError:
            logger.warning("Impossibile eliminare il file temporaneo %s", path)

def analyze_context(text: str):
    """Esegue sentiment analysis e ricerca memoria in parallelo."""
    future_sentiment = executor.submit(sentiment_analyzer.analyze, text)
    future_chroma = executor.submit(chroma_search, text, 1)

    attitudine = future_sentiment.result()
    similar_qas = future_chroma.result()

    logger.info(f"Sentiment principale: {attitudine.get('sentiment', '')}")

    if similar_qas and similar_qas[0]["similarità"] >= SIMILARITY_THRESHOLD:
        logger.info(f"Memoria accettata (similarità {similar_qas[0]['similarità']})")
    else:
        logger.info("Nessuna memoria rilevante trovata → contesto vuoto")
        similar_qas = []

    return attitudine, similar_qas

def build_context(base_context: str, attitudine: dict, similar_qas: list) -> str:
    """Costruisce il contesto finale per l'LLM."""
    memoria_context = ""
    for qa in similar_qas:
        memoria_context += f"\nDomanda passata: {qa['domanda_simile']} | Risposta: {qa['risposta_passata']}"

    return (
        base_context
        + "\nL'attitudine dello studente è: "
        + attitudine.get("sentiment", "")
        + ", rispondi di conseguenza."
        + "\nMemoria passata utile (se rilevante):"
        + memoria_context
        + "\n Rispondi in maniera coerente con quello che hai detto prima."
    )

def run_tts(text: str) -> str:
    """Genera audio TTS e restituisce l'audio codificato in base64."""
    text = (text or "").replace("*", "")
    start = time.perf_counter()
    audio_bytes, _ = tts_create(text or "Non sono riuscito a capire la domanda, per favore ripeti.")
    elapsed = time.perf_counter() - start
    logger.info("TTS completato in %.3f secondi", elapsed)
    return base64.b64encode(audio_bytes).decode("utf-8")

# ================================
# Endpoint
# ================================

@bp.post("/ask")
def ask_endpoint():
    in_tmp_path = None
    try:
        # 1. Validazione input
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "manca il file 'audio'"}), 400
        f = request.files["audio"]
        if not f.filename:
            return jsonify({"success": False, "error": "nome file vuoto"}), 400

        # 2. Leggo direttamente i byte
        audio_bytes = f.read()

        # 3. Trascrizione
        res = transcribe_bytes(audio_bytes)
        text = res.get("text", "") or ""
        confidence = res.get("confidence", None)

        base_context = CONTEXT_PROMPT  

        # 4. Scelta: chiarificazione o normale
        if confidence is not None and confidence < Config.ASR_CONF_THRESHOLD:
            logger.info("Confidenza bassa → richiesta chiarimento")
            llm_text = ask_llm(base_context, CLARIFY_PROMPT)
            status = "clarify"

            # Anche qui TTS in parallelo (solo TTS serve)
            future_tts = executor.submit(run_tts, llm_text)
            audio_b64 = future_tts.result()

        else:
            # Sentiment + memoria già in parallelo
            attitudine, similar_qas = analyze_context(text)
            local_context = build_context(base_context, attitudine, similar_qas)

            # Chiamata LLM (bloccante, non parallelizzabile)
            llm_text = ask_llm(local_context, text)
            status = "ok"

            # Lancia subito TTS e QA in parallelo
            future_tts = executor.submit(run_tts, llm_text)
            if not similar_qas or similar_qas[0]["similarità"] < 1:
                executor.submit(add_qa, text, llm_text)

            # Aspetta solo il TTS (QA continua in background)
            audio_b64 = future_tts.result()

        # 5. Risposta finale
        return jsonify({
            "success": True,
            "status": status,
            "message": llm_text,
            "audio": audio_b64,
        }), 200

    except Exception as e:
        logger.exception("Errore in /ask")
        return jsonify({"success": False, "error": str(e)}), 500