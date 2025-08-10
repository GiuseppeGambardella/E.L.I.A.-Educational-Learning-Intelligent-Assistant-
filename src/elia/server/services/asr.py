# trascrizione con faster-whisper (italiano)
from faster_whisper import WhisperModel
from elia.config import Config

_model = WhisperModel(
    Config.WHISPER_MODEL or "small",
    device="cuda",
    compute_type="auto"
)

def transcribe_wav(path_wav: str) -> dict:
    segments, info = _model.transcribe(path_wav, language="it")
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "duration": info.duration}
