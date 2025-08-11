from flask import Blueprint, request, jsonify, current_app
import tempfile, os, datetime
from elia.server.services.asr import transcribe_wav
from elia.config import Config

bp = Blueprint("transcribe", __name__)

cartella_transcripts = "data/transcripts"

print("Transcribe blueprint initialized")

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
        # transcribe_wav deve restituire anche "confidence"
        res = transcribe_wav(tmp.name)
        duration = res.get("duration")
        confidence = res.get("confidence")

        # salva su file lato server
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # decide cosa tornare al client
        if confidence is not None and confidence < Config.ASR_CONF_THRESHOLD:
            return jsonify(
                success=True,
                status="clarify",
                id=ts,
                duration=duration,
                confidence=round(confidence, 3),
                message="Non sono sicuro di aver capito perfettamente. Puoi ripetere? Dal server"
            ), 200

        return jsonify(
            success=True,
            status="ok",
            id=ts,
            duration=duration,
            confidence=None if confidence is None else round(confidence, 3),
        ), 200

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
