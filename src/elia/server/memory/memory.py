import os, uuid, logging, torch
import chromadb
from sentence_transformers import SentenceTransformer
from elia.server.models.llm import ask_llm
from elia.config import Config

logger = logging.getLogger(__name__)

# ==========================================
# Setup DB persistente
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "elia_memoria"

os.makedirs(DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # ANN veloce
)

# ==========================================
# Lazy loading del modello embeddings
# ==========================================
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Caricamento modello embeddings su %s...", device)
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _embedding_model

# ==========================================
# Funzioni principali
# ==========================================
def add_qa(question: str, answer: str, sentiment: str = None):
    """
    Aggiunge una coppia domanda-risposta al database.
    
    Args:
        question: La domanda dello studente
        answer: La risposta fornita
        sentiment: Il breve report emotivo dell'interazione (non un singolo sentiment)
    """
    try:
        model = get_embedding_model()
        embedding = model.encode(question, convert_to_numpy=True)
        q_id = str(uuid.uuid4())

        # Metadati estesi con sentiment
        metadata = {"answer": answer}
        if sentiment:
            metadata["sentiment"] = sentiment

        collection.add(
            ids=[q_id],
            documents=[question],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        logger.info("QA aggiunta | Domanda: %.80s... | Report emotivo: %.80s...", question, sentiment or "N/A")
        return {"status": "ok", "id": q_id}
    except Exception as e:
        logger.exception("Errore in add_qa")
        return {"status": "error", "message": str(e)}

def search(query: str, top_k: int = 5):
    try:
        model = get_embedding_model()
        query_emb = model.encode(query, convert_to_numpy=True)

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        scores = results.get("distances", [[]])[0]

        out = []
        for doc, meta, score in zip(docs, metas, scores):
            sim = round(1 - score, 3)
            out.append({
                "domanda_simile": doc,
                "risposta_passata": meta.get("answer", ""),
                "similarità": sim
            })

        return out
    except Exception as e:
        logger.exception("Errore in search")
        return []


def get_all_emotional_data():
    """
    Recupera tutte le entry dal database con i relativi sentiment.
    Restituisce i dati strutturati per l'analisi emotiva.
    """
    try:
        logger.info("🗄️ Recupero dati emotivi dal database...")
        
        # Recupera tutti i record dal database
        all_data = collection.get()
        
        if not all_data or not all_data.get('documents'):
            logger.warning("📭 Nessun dato disponibile nel database")
            return {
                "status": "empty",
                "message": "Nessun dato disponibile",
                "data": {
                    "documents": [],
                    "metadatas": [],
                    "emotional_reports": [],
                    "total_interactions": 0,
                    "valid_emotional_reports": 0
                }
            }
        
        documents = all_data.get('documents', [])
        metadatas = all_data.get('metadatas', [])
        
        # Analizza i report emotivi (non più sentiment singoli)
        emotional_reports = []
        valid_reports = 0
        
        for metadata in metadatas:
            report = metadata.get('sentiment', 'Nessun report disponibile')  # 'sentiment' contiene il report
            emotional_reports.append(report)
            if report != 'Nessun report disponibile' and report:
                valid_reports += 1
        
        logger.info(f"📊 Dati recuperati: {len(documents)} documenti, {valid_reports} report emotivi validi")
        
        return {
            "status": "success",
            "data": {
                "documents": documents,
                "metadatas": metadatas,
                "emotional_reports": emotional_reports,
                "total_interactions": len(documents),
                "valid_emotional_reports": valid_reports
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Errore nel recupero dati emotivi: {e}")
        return {
            "status": "error",
            "message": str(e),
            "data": {
                "documents": [],
                "metadatas": [],
                "emotional_reports": [],
                "total_interactions": 0,
                "valid_emotional_reports": 0
            }
        }