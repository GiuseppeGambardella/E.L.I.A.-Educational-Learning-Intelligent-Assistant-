import os, uuid, logging, torch
import chromadb
from sentence_transformers import SentenceTransformer

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
def add_qa(question: str, answer: str):
    try:
        model = get_embedding_model()
        embedding = model.encode(question, convert_to_numpy=True)
        q_id = str(uuid.uuid4())

        collection.add(
            ids=[q_id],
            documents=[question],
            embeddings=[embedding],
            metadatas=[{"answer": answer}]
        )
        logger.info("QA aggiunta | Domanda: %.80s...", question)
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
