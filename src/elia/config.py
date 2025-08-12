from dotenv import load_dotenv
import os
import logging

load_dotenv()

def _setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    # Configura il root logger una sola volta
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    logging.getLogger(__name__).debug("Logging initialized at %s", level_name)

_setup_logging()
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
    ENDPOINT_TRANSCRIBE = os.getenv("ENDPOINT_TRANSCRIBE", "http://localhost:5000/ask")
    WHISPER_MODEL = os.getenv("FWHISPER_MODEL", "small")
    ASR_CONF_THRESHOLD = float(os.getenv("ASR_CONF_THRESHOLD", 0.60))
    ASR_MIN_WORDS = int(os.getenv("ASR_MIN_WORDS", 3))
    GEMMA_API_URL = os.getenv("GEMMA_API_URL")
    GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")