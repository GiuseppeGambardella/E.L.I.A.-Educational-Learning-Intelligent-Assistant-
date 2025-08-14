import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "neuraly/bert-base-italian-cased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

logger.info("Starting sentiment analysis component")

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=1,
            device=0 if torch.cuda.is_available() else -1
        )

    def analyze(self, text):
        logger.info("🔍 Sentiment analysis in progress...")

        if not text or not text.strip():
            logger.warning("⚠️ Empty text, cannot analyze sentiment.")
            return {"sentiment": None, "dettaglio": []}

        risultati = self.classifier(text)[0]
        sentiment_principale = risultati[0]['label']

        logger.info(f"✅ Sentiment detected: {sentiment_principale}")

        return {
            "sentiment": sentiment_principale,
            "dettaglio": risultati
        }

logger.info("Sentiment analysis component initialized")
