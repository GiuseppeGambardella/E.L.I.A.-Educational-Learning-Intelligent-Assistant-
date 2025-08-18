import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import logging

logger = logging.getLogger(__name__)

# ==========================================
# Setup DB persistente e collezione
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "elia_memoria"

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH, exist_ok=True)
    logger.info("Cartella DB creata: %s", DB_PATH)
else:
    logger.info("Cartella DB trovata: %s", DB_PATH)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
logger.info("Collezione caricata: %s", COLLECTION_NAME)

# ==========================================
# Lazy loading del modello embeddings
# ==========================================
_embedding_model = None

def get_embedding_model():
    """Carica il modello embeddings una sola volta (lazy)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Caricamento modello embeddings...")
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
        logger.info("Modello embeddings caricato (%s)", _embedding_model.device)
    return _embedding_model

# ==========================================
# Funzioni principali
# ==========================================
def add_qa(question: str, answer: str):
    """
    Aggiunge una domanda+risposta a Chroma.
    L'ID viene generato automaticamente con UUID.
    """
    try:
        model = get_embedding_model()
        q_id = str(uuid.uuid4())
        collection.add(
            ids=[q_id],
            documents=[question],
            embeddings=[model.encode(question).tolist()],
            metadatas=[{"answer": answer}]
        )
        logger.info("QA aggiunta | Domanda: '%s'", question)
        return {"status": "ok", "id": q_id}
    except Exception as e:
        logger.exception("Errore in add_qa (Domanda='%s')", question)
        return {"status": "error", "message": str(e)}

def search(query: str, top_k: int = 5):
    """
    Cerca domande simili in Chroma.
    Ritorna lista di {domanda_simile, risposta_passata, similarità}.
    """
    try:
        model = get_embedding_model()
        results = collection.query(
            query_embeddings=[model.encode(query).tolist()],
            n_results=top_k
        )

        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        scores = results.get("distances", [[]])[0]

        if docs:
            logger.info("Ricerca completata | Query: '%s' | Risultati: %d", query, len(docs))
            for doc, meta, score in zip(docs, metas, scores):
                sim = round(1 - score, 3)
                logger.debug("Risultato: '%s' (similarità=%.3f)", doc, sim)
                out.append({
                    "domanda_simile": doc,
                    "risposta_passata": meta.get("answer", ""),
                    "similarità": sim
                })
        else:
            logger.info("Ricerca completata | Query: '%s' | Nessun risultato", query)

        return out

    except Exception as e:
        logger.exception("Errore in search (Query='%s')", query)
        return []
