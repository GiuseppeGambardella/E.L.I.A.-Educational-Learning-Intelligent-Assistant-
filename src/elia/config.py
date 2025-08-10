from dotenv import load_dotenv
import os
load_dotenv()

print("Loading configuration from environment variables...")

class Config:
    ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = ENV == "development"
    PORT = int(os.getenv("PORT", "5000"))
    PICOVOICE_KEY = os.getenv("PICOVOICE_KEY")
    PICOVOICE_WORD = os.getenv("PICOVOICE_WORD")
    PICOVOICE_PARAMS = os.getenv("PICOVOICE_PARAMS")
    AUDIO_DEVICE_INDEX = int(os.getenv("AUDIO_DEVICE_INDEX", 0))
    ENDPOINT_TRANSCRIBE = os.getenv("ENDPOINT_TRANSCRIBE", "http://localhost:5000/transcribe")
    WHISPER_MODEL = os.getenv("FWHISPER_MODEL", "small")  # Modello ASR Faster-Whisper