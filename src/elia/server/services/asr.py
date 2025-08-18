# =========================
# TRASCRIZIONE CON FASTER-WHISPER (ITALIANO)
# =========================
import math
from faster_whisper import WhisperModel
from elia.config import Config
import torch
from statistics import mean
import logging
import time

logger = logging.getLogger(__name__)

# =========================
# MODELLO GLOBALE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

_model = WhisperModel(
    Config.WHISPER_MODEL or "medium",
    device=device,
    compute_type="auto"
)

logger.info("Whisper model loaded: %s on %s", Config.WHISPER_MODEL or "medium", device)

# =========================
# FUNZIONI DI SUPPORTO
# =========================
def _sigmoid(x: float) -> float:
    """
    Normalizza un logit in [0,1].
    Limita x a [-10,10] per stabilità numerica.
    """
    x = max(min(x, 10.0), -10.0)
    return 1.0 / (1.0 + math.exp(-x))


def _compute_confidence(segments) -> float:
    """
    Calcola una confidenza aggregata a partire dai segmenti.
    - Priorità: probabilità parola-per-parola
    - Fallback: media di avg_logprob dei segmenti
    - Altro segnale: no_speech_prob
    """
    word_probs, avg_logprobs, no_speech = [], [], []

    for seg in segments:
        # 1) parola-per-parola
        if getattr(seg, "words", None):
            for w in seg.words:
                p = getattr(w, "probability", None)
                if p is not None:
                    word_probs.append(float(p))
        # 2) fallback: logprobs segmenti
        if getattr(seg, "avg_logprob", None) is not None:
            avg_logprobs.append(float(seg.avg_logprob))
        # 3) ulteriore segnale: no_speech_prob
        if getattr(seg, "no_speech_prob", None) is not None:
            no_speech.append(float(seg.no_speech_prob))

    if word_probs:
        return float(mean(word_probs))
    if avg_logprobs:
        return float(mean(_sigmoid(x) for x in avg_logprobs))
    if no_speech:
        return float(max(0.0, 1.0 - mean(no_speech)))

    return 0.5  # fallback neutrale

# =========================
# FUNZIONE PRINCIPALE
# =========================
def transcribe_wav(path_wav: str) -> dict:
    """
    Trascrive un file WAV in italiano usando faster-whisper.
    Ritorna dict con:
      - text: trascrizione
      - duration: durata audio (se disponibile)
      - confidence: stima di confidenza [0,1]
    """
    try:
        start = time.perf_counter()

        segments, info = _model.transcribe(
            path_wav,
            language="it",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True  # parola-per-parola
        )

        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()

        # Calcolo confidenza
        conf_words = _compute_confidence(segs)
        lang_prob = getattr(info, "language_probability", None)

        if isinstance(lang_prob, float):
            confidence = 0.8 * conf_words + 0.2 * lang_prob
        else:
            confidence = conf_words

        elapsed = time.perf_counter() - start
        logger.info("Trascrizione completata | file=%s | durata=%.2fs | testo_len=%d",
                    path_wav, elapsed, len(text))

        return {
            "text": text,
            "duration": getattr(info, "duration", None),
            "confidence": float(confidence)
        }

    except Exception as e:
        logger.exception("Errore durante la trascrizione di %s", path_wav)
        return {"text": "", "duration": None, "confidence": 0.0}
