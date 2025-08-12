# trascrizione con faster-whisper (italiano)
import math
from faster_whisper import WhisperModel
from elia.config import Config
import torch
from statistics import mean
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

_model = WhisperModel(
    Config.WHISPER_MODEL or "medium",
    device=device,
    compute_type="auto"
)

logger.info(f"Whisper model loaded: {Config.WHISPER_MODEL or 'medium'} on {device}")

#Funzione di Trascrizione
def transcribe_wav(path_wav: str) -> dict:
    
    segments, info = _model.transcribe(
        path_wav,
        language="it",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True  # parola-per-parola
    )

    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()

    # Calcola confidenza
    conf_words = _compute_confidence(segs)
    lang_prob = getattr(info, "language_probability", None)

    if isinstance(lang_prob, float):
        confidence = 0.8 * conf_words + 0.2 * lang_prob
    else:
        confidence = conf_words

    return {
        "text": text,
        "duration": getattr(info, "duration", None),
        "confidence": float(confidence)
    }


# Normalizzazione valore di soglia
def _sigmoid(x: float) -> float:
    x = max(min(x, 10.0), -10.0)
    return 1.0 / (1.0 + math.exp(-x))


#Calcolo della confidenza
def _compute_confidence(segments) -> float:

    word_probs, avg_logprobs, no_speech = [], [], []

    for seg in segments:
        # 1) parola-per-parola (migliore)
        if getattr(seg, "words", None):
            for w in seg.words:
                p = getattr(w, "probability", None)
                if p is not None:
                    word_probs.append(float(p))
        # 2) fallback: media degli avg_logprob dei segmenti
        if getattr(seg, "avg_logprob", None) is not None:
            avg_logprobs.append(float(seg.avg_logprob))
        # 3) ulteriore segnale: no_speech_prob alto => bassa confidenza
        if getattr(seg, "no_speech_prob", None) is not None:
            no_speech.append(float(seg.no_speech_prob))

    if word_probs:
        return float(mean(word_probs))

    if avg_logprobs:
        return float(mean(_sigmoid(x) for x in avg_logprobs))

    if no_speech:
        return float(max(0.0, 1.0 - mean(no_speech)))

    return 0.5  # neutrale se non abbiamo nulla
