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
    ENDPOINT_ATTENTION = os.getenv("ENDPOINT_ATTENTION", "http://localhost:5000/attention")
    CLARIFY_PROMPT = os.getenv("CLARIFY_PROMPT", "Comportati come se non avessi capito. Scrivi una sola frase, educata e concisa (MAX 15 PAROLE), che chieda di ripetere. Non aggiungere altro. Devi essere il piu sintetico possibile.")
    CONTEXT_PROMPT = os.getenv("CONTEXT_PROMPT", "Sei un assistente virtuale di nome Elia (Educational Learning Intelligent Assistant) che aiuta gli studenti rispondendo alle loro domande. Quando ti salutano, ricambia il saluto in modo semplice. Non ricordare sempre che sei un assistente vocale volto all'educamento, ma rispondi solo quando ti viene espressamente chiesto. Rispondi solo in italiano, mantenendo tutti gli accenti corretti. Non usare emoji, solo testo puro. Adotta sempre un tono empatico e di supporto, calibrando la risposta allo stato emotivo dello studente. Rispetta il limite massimo di 120 parole. Sciogli sempre gli acronimi (esempio: d.C. -> dopo Cristo). Non inventare informazioni. Se la domanda contiene errori o imprecisioni, correggili e segnala la correzione nella risposta. Non attingere a dati esterni: usa solo le tue conoscenze interne e il contenuto della domanda. Non devi mai mentire o fornire informazioni false. Non devi usare caratteri volti ad evidenziare parole.")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    DEFAULT_PITCH = os.getenv("DEFAULT_PITCH", "-15Hz")
    DEFAULT_RATE = os.getenv("DEFAULT_RATE", "+10%")