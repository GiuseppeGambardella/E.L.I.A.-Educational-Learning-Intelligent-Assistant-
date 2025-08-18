import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

# =========================
# COSTANTI MODELLO
# =========================
MODEL_NAME = "neuraly/bert-base-italian-cased-sentiment"

# =========================
# CARICAMENTO MODELLO E TOKENIZER
# =========================
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    logger.info("Modello di sentiment analysis caricato: %s", MODEL_NAME)
except Exception as e:
    logger.exception("Errore durante il caricamento del modello %s", MODEL_NAME)
    raise

# =========================
# CLASSE PRINCIPALE
# =========================
class SentimentAnalyzer:
    """
    Componente per analisi del sentiment in italiano
    basata su BERT (neuraly/bert-base-italian-cased-sentiment).
    """

    def __init__(self):
        """
        Inizializza la pipeline HuggingFace per text-classification.
        Usa GPU se disponibile.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=1,
            device=device
        )
        logger.info("Pipeline HuggingFace inizializzata (device=%s)", "cuda" if device == 0 else "cpu")

    def analyze(self, text: str) -> dict:
        """
        Esegue sentiment analysis su una stringa.
        Ritorna:
            {
              "sentiment": <label principale> | None,
              "dettaglio": <lista risultati della pipeline>
            }
        """
        logger.info("🔍 Avvio analisi sentiment")

        if not text or not text.strip():
            logger.warning("⚠️ Testo vuoto, impossibile analizzare il sentiment")
            return {"sentiment": None, "dettaglio": []}

        try:
            risultati = self.classifier(text)[0]
            sentiment_principale = risultati[0]["label"]
            score = risultati[0]["score"]

            logger.info("✅ Sentiment rilevato: %s (score=%.3f)", sentiment_principale, score)

            return {
                "sentiment": sentiment_principale,
                "dettaglio": risultati
            }

        except Exception as e:
            logger.exception("Errore durante l'analisi del sentiment")
            return {"sentiment": None, "dettaglio": []}


logger.info("Componente di sentiment analysis inizializzato con successo")
