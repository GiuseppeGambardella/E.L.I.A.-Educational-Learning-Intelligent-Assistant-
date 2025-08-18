import io
import math
import time
import logging
from statistics import mean

import torch
from faster_whisper import WhisperModel
from elia.config import Config

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
# SUPPORTO
# =========================
def _sigmoid(x: float) -> float:
    x = max(min(x, 10.0), -10.0)
    return 1.0 / (1.0 + math.exp(-x))


def _compute_confidence(segments) -> float:
    word_probs, avg_logprobs, no_speech = [], [], []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                if getattr(w, "probability", None) is not None:
                    word_probs.append(float(w.probability))
        if getattr(seg, "avg_logprob", None) is not None:
            avg_logprobs.append(float(seg.avg_logprob))
        if getattr(seg, "no_speech_prob", None) is not None:
            no_speech.append(float(seg.no_speech_prob))
    if word_probs:
        return float(mean(word_probs))
    if avg_logprobs:
        return float(mean(_sigmoid(x) for x in avg_logprobs))
    if no_speech:
        return float(max(0.0, 1.0 - mean(no_speech)))
    return 0.5


# =========================
# TRASCRIZIONI
# =========================
def _run_transcription(audio, from_file: bool = True) -> dict:
    try:
        start = time.perf_counter()

        if from_file:
            segments, info = _model.transcribe(
                audio, language="it", beam_size=5, vad_filter=True, word_timestamps=True
            )
        else:
            # audio è bytes → uso un buffer in memoria
            buf = io.BytesIO(audio)
            segments, info = _model.transcribe(
                buf, language="it", beam_size=5, vad_filter=True, word_timestamps=True
            )

        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()

        conf_words = _compute_confidence(segs)
        lang_prob = getattr(info, "language_probability", None)
        confidence = 0.8 * conf_words + 0.2 * lang_prob if isinstance(lang_prob, float) else conf_words

        elapsed = time.perf_counter() - start
        logger.info(
            "Trascrizione completata | durata=%.2fs | conf=%.3f | testo_len=%d",
            elapsed, confidence, len(text)
        )

        return {
            "text": text,
            "duration": getattr(info, "duration", None),
            "confidence": float(confidence),
            "error": None
        }
    except Exception as e:
        logger.exception("Errore durante la trascrizione")
        return {"text": "", "duration": None, "confidence": 0.0, "error": str(e)}


def transcribe_wav(path: str) -> dict:
    """Trascrive un file WAV da path."""
    return _run_transcription(path, from_file=True)


def transcribe_bytes(audio_bytes: bytes) -> dict:
    """Trascrive un audio WAV già in memoria (bytes)."""
    return _run_transcription(audio_bytes, from_file=False)
