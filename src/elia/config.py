from dotenv import load_dotenv
import os
load_dotenv()

class Config:
    ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = ENV == "development"
    PORT = int(os.getenv("PORT", "5000"))
    PICOVOICE_KEY = os.getenv("PICOVOICE_KEY")
    PICOVOICE_WORD = os.getenv("PICOVOICE_WORD")
    PICOVOICE_PARAMS = os.getenv("PICOVOICE_PARAMS")