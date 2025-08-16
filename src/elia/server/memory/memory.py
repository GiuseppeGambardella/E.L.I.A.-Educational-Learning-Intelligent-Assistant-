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
    logger.info(f"Cartella DB creata: {DB_PATH}")
else:
    logger.info(f"Cartella DB trovata: {DB_PATH}")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
logger.info(f"Collezione caricata: {COLLECTION_NAME}")

# ==========================================
# Lazy loading del modello embeddings
# ==========================================
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Caricamento modello embeddings...")
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
        logger.info("Modello embeddings caricato")
    return _embedding_model

# ==========================================
# Funzioni principali
# ==========================================
def add_qa(question: str, answer: str):
    """
    Aggiunge una domanda+risposta a Chroma.
    L'ID viene generato automaticamente con UUID.
    """
    model = get_embedding_model()
    q_id = str(uuid.uuid4())
    collection.add(
        ids=[q_id],
        documents=[question],
        embeddings=[model.encode(question).tolist()],
        metadatas=[{"answer": answer}]
    )
    logger.info("QA aggiunta | Domanda: '%s' |", question)
    return {"status": "ok", "id": q_id}


def search(query: str, top_k: int = 5):
    """
    Cerca domande simili in Chroma.
    """
    model = get_embedding_model()
    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=top_k
    )

    out = []
    if results["documents"][0]:
        logger.info("Ricerca completata | Query: '%s' | Risultati trovati: %d",
                    query, len(results["documents"][0]))
        for doc, meta, score in zip(results["documents"][0],
                                    results["metadatas"][0],
                                    results["distances"][0]):
            sim = round(1 - score, 3)
            logger.info(" (similarità %.3f)", sim)
            out.append({
                "domanda_simile": doc,
                "risposta_passata": meta["answer"],
                "similarità": sim
            })
    else:
        logger.info("Ricerca completata | Query: '%s' | Nessun risultato", query)

    return out
