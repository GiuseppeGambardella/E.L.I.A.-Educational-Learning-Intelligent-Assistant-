from flask import Blueprint, request, jsonify, current_app
import tempfile, os, datetime
from elia.server.services.asr import transcribe_wav

bp = Blueprint("transcribe", __name__)

print("Transcribe blueprint initialized")

def _transcripts_dir() -> str:
    # cartella configurabile via ENV/Config, fallback a "data/transcripts"
    base = getattr(current_app.config, "TRANSCRIPTS_DIR", None) or "data/transcripts"
    os.makedirs(base, exist_ok=True)
    return base

@bp.post("/transcribe")
def transcribe_endpoint():
    if "audio" not in request.files:
        return jsonify(success=False, error="manca il file 'audio'"), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify(success=False, error="nome file vuoto"), 400

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.save(tmp.name)
    try:
        res = transcribe_wav(tmp.name)   # {"text":..., "duration":...}
        text, duration = res.get("text", ""), res.get("duration")

        # salva SOLO lato server
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _transcripts_dir()
        out_path = os.path.join(out_dir, f"{ts}.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(text.strip() + "\n")

        return jsonify(success=True, id=ts, duration=duration)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    finally:
        try: os.remove(tmp.name)
        except OSError: pass
