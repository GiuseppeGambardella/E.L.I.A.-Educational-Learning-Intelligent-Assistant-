from dotenv import load_dotenv
import os
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading configuration from environment variables...")

class Config:
    ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = ENV == "development"
    PORT = int(os.getenv("PORT", "5000"))
    PICOVOICE_KEY = os.getenv("PICOVOICE_KEY")
    PICOVOICE_WORD = os.getenv("PICOVOICE_WORD")
    PICOVOICE_PARAMS = os.getenv("PICOVOICE_PARAMS")
    AUDIO_DEVICE_INDEX = int(os.getenv("AUDIO_DEVICE_INDEX", 0))
    ENDPOINT_ASK = os.getenv("ENDPOINT_ASK", "http://localhost:5000/ask")
    WHISPER_MODEL = os.getenv("FWHISPER_MODEL", "small")
    ASR_CONF_THRESHOLD = float(os.getenv("ASR_CONF_THRESHOLD", 0.60))
    ASR_MIN_WORDS = int(os.getenv("ASR_MIN_WORDS", 3))
    GEMMA_API_URL = os.getenv("GEMMA_API_URL")
    GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    TTS_VOICE = os.getenv("TTS_VOICE", "it-IT-DiegoNeural")