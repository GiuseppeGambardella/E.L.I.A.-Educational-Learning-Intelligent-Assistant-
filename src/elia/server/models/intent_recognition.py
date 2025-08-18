# =========================
# DESCRIZIONE E IMPORT
# =========================
"""
Intent recognition: prova prima i pattern (pattern_entities.jsonl), altrimenti modello ML spaCy.
Carica i pattern e le regole dell'AttributeRuler una sola volta.

Rifattorizzato per eliminare duplicazioni:
  - Funzioni centralizzate per caricamento/sanitizzazione pattern e modello spaCy.
  - Selezione top-N unificata per pattern e modello.
  - Gestione AttributeRuler unificata.
"""
import pathlib
import time
import threading
import json
from typing import Any, List, Dict, Tuple, Optional

import logging
import spacy
from spacy.matcher import Matcher

logger = logging.getLogger(__name__)

# =========================
# CONFIGURAZIONE E PARAMETRI
# =========================
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "server" / "models" / "nlp_model" / "best"
PATTERN_FILE = BASE_DIR / "models" / "nlp" / "pattern_entities.jsonl"

GLOBAL_THRESHOLD = 0.5   # soglia minima assoluta per accettare un intent
MAX_ACTIVE = 3           # numero massimo di intenti attivi
RELATIVE_GAP = 0.12      # gap relativo rispetto al top score
FORCE_MIN_TOP = 3        # numero minimo di intenti restituiti

# Locks / globals (per caricare una sola volta)
_patterns_lock = threading.Lock()
_model_lock = threading.Lock()
_nlp_patterns: Optional[spacy.language.Language] = None
_matcher: Optional[Matcher] = None
_nlp_model: Optional[spacy.language.Language] = None
_loaded_model_time: Optional[float] = None

# =========================
# UTILITY PATTERN MATCHING (LOW-LEVEL)
# =========================
def _load_raw_patterns() -> List[Dict[str, Any]]:
    """
    Carica i pattern dal file JSONL.
    Ogni riga deve contenere {"label":..., "pattern":...}.
    """
    patterns: List[Dict[str, Any]] = []
    if not PATTERN_FILE.exists():
        logger.warning("Pattern file non trovato: %s", PATTERN_FILE)
        return patterns
    with PATTERN_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            obj = json.loads(line)
            patterns.append({"label": obj["label"], "pattern": obj["pattern"]})
    return patterns

def _has_lemma_token(tokens: Any) -> bool:
    """Verifica se il pattern contiene un token con attributo LEMMA."""
    return isinstance(tokens, list) and any(isinstance(t, dict) and any(k.upper() == "LEMMA" for k in t) for t in tokens)

def _needs_lemma(patterns: List[Dict[str, Any]]) -> bool:
    """Ritorna True se almeno un pattern richiede LEMMA."""
    return any(_has_lemma_token(p["pattern"]) for p in patterns)

def _degrade_lemma_token(tok: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converte un token che usa LEMMA in uno basato su LOWER,
    per pipeline che non hanno un lemmatizer.
    """
    lemma_val = tok.get("LEMMA") or tok.get("lemma")
    op = tok.get("OP")
    new_tok: Dict[str, Any] = {}
    if isinstance(lemma_val, dict):
        for k, v in lemma_val.items():
            if k.upper() == "IN":
                new_tok["LOWER"] = {"IN": [str(x).lower() for x in v]}
                break
        if not new_tok:
            new_tok["LOWER"] = str(lemma_val).lower()
    elif isinstance(lemma_val, list):
        new_tok["LOWER"] = {"IN": [str(x).lower() for x in lemma_val]}
    else:
        new_tok["LOWER"] = str(lemma_val).lower()
    if op:
        new_tok["OP"] = op
    return new_tok

def _degrade_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applica la degradazione da LEMMA a LOWER a tutti i pattern se necessario."""
    out: List[Dict[str, Any]] = []
    for p in patterns:
        toks = p["pattern"]
        if isinstance(toks, list):
            nt = []
            for t in toks:
                if isinstance(t, dict) and any(k.upper() == "LEMMA" for k in t):
                    nt.append(_degrade_lemma_token(t))
                else:
                    nt.append(t)
            out.append({"label": p["label"], "pattern": nt})
        else:
            out.append(p)
    return out

def _sanitize_token(tok: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pulisce un token da chiavi non valide.
    Mantiene solo attributi ammessi da spaCy Matcher.
    """
    allowed = {
        "TEXT","LOWER","POS","TAG","DEP","SHAPE","PREFIX","SUFFIX","LENGTH",
        "IS_ALPHA","IS_ASCII","IS_DIGIT","IS_PUNCT","IS_SPACE","IS_STOP",
        "LIKE_NUM","LIKE_EMAIL","LEMMA","OP","IS_TITLE","IS_LOWER","IS_UPPER"
    }
    clean: Dict[str, Any] = {}
    for k, v in tok.items():
        K = k.upper()
        if K == "OP":
            clean["OP"] = v
        elif K in allowed:
            if isinstance(v, dict):
                sub: Dict[str, Any] = {sk.upper(): sv for sk, sv in v.items() if sk.upper() in {"IN","NOT_IN"}}
                if sub:
                    clean[K] = sub
            else:
                clean[K] = v
    return clean

def _sanitize_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applica la sanificazione a tutti i pattern caricati."""
    valid: List[Dict[str, Any]] = []
    for p in patterns:
        toks = p["pattern"]
        if isinstance(toks, list):
            new_list = []
            skip = False
            for tok in toks:
                if not isinstance(tok, dict):
                    skip = True
                    break
                st = _sanitize_token(tok)
                if not st:
                    skip = True
                    break
                new_list.append(st)
            if not skip and new_list:
                valid.append({"label": p["label"], "pattern": new_list})
    return valid

# =========================
# HELPER UNIFICATI
# =========================
def _load_spacy_it_model(require_lemma: bool) -> Tuple[spacy.language.Language, bool]:
    """
    Carica un modello spaCy italiano.
    - Prova 'it_core_news_lg'
    - Se fallisce → blank('it')
    """
    try:
        nlp = spacy.load("it_core_news_lg")
        has_lemma = "lemmatizer" in nlp.pipe_names
        logger.info("Caricato modello it_core_news_lg (lemmatizer=%s)", has_lemma)
    except Exception as e:
        nlp = spacy.blank("it")
        has_lemma = False
        logger.warning("Fallback su blank('it'): %s", e)
    return nlp, has_lemma

def _prepare_patterns_for(nlp: spacy.language.Language, raw_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Se i pattern richiedono lemma ma la pipeline non ha un lemmatizer,
    degrada i token e poi li sanifica.
    """
    patterns = raw_patterns
    if _needs_lemma(patterns) and "lemmatizer" not in nlp.pipe_names:
        patterns = _degrade_patterns(patterns)
    return _sanitize_patterns(patterns)

def _ensure_attribute_ruler_from_jsonl(nlp: spacy.language.Language, patterns: List[Dict[str, Any]]) -> None:
    """
    Carica le regole dell'AttributeRuler a partire dai pattern JSONL.
    Viene eseguito una sola volta per pipeline.
    """
    try:
        if getattr(nlp, "_elia_ruler_loaded", False):
            return
        if "attribute_ruler" in nlp.pipe_names:
            ruler = nlp.get_pipe("attribute_ruler")
        else:
            before = "lemmatizer" if "lemmatizer" in nlp.pipe_names else None
            ruler = nlp.add_pipe("attribute_ruler", before=before)
        added = 0
        for p in patterns:
            try:
                ruler.add([p["pattern"]], attrs={"NORM": str(p["label"])})
                added += 1
            except Exception as e:
                logger.warning("AttributeRuler add fallita per %s: %s", p.get("label"), e)
        nlp._elia_ruler_loaded = True
        if added:
            logger.info("AttributeRuler: caricate %d regole dal JSONL", added)
    except Exception as e:
        logger.warning("AttributeRuler non inizializzato: %s", e)

def _select_top_items(sorted_items: List[Tuple[str, float]], threshold: float, rel_gap: float, max_active: int, force_min_top: int) -> List[Tuple[str, float]]:
    """
    Seleziona i top item con criteri:
      - soglia assoluta
      - gap relativo dal top
      - massimo numero di intenti attivi
      - minimo forzato
    """
    if not sorted_items:
        return []
    top_score = sorted_items[0][1]
    chosen: List[Tuple[str, float]] = []
    for lab, score in sorted_items:
        if score >= threshold and score >= top_score - rel_gap:
            chosen.append((lab, score))
        if len(chosen) >= max_active:
            break
    if len(chosen) < force_min_top:
        used = {l for l, _ in chosen}
        for lab, score in sorted_items:
            if lab in used:
                continue
            chosen.append((lab, score))
            if len(chosen) >= force_min_top or len(chosen) >= max_active:
                break
    if not chosen and sorted_items:
        chosen = [sorted_items[0]]
    return chosen

# =========================
# INIZIALIZZAZIONE PIPELINE PATTERN E MODELLO
# =========================
def load_patterns(reload: bool = False):
    """
    Inizializza la pipeline spaCy e i pattern Matcher.
    Carica i pattern dal JSONL e li aggiunge a spaCy.
    """
    global _nlp_patterns, _matcher
    if _nlp_patterns is not None and not reload:
        return
    with _patterns_lock:
        if _nlp_patterns is not None and not reload:
            return
        try:
            raw = _load_raw_patterns()
            require_lemma = _needs_lemma(raw)
            nlp_pat, has_lemma = _load_spacy_it_model(require_lemma)
            patterns = _prepare_patterns_for(nlp_pat, raw)
            _ensure_attribute_ruler_from_jsonl(nlp_pat, patterns)

            matcher = Matcher(nlp_pat.vocab, validate=True)
            added = 0
            for p in patterns:
                try:
                    matcher.add(p["label"], [p["pattern"]])
                    added += 1
                except Exception as e:
                    logger.warning("Pattern non aggiunto: %s -> %s", p.get("label"), e)

            _nlp_patterns = nlp_pat
            _matcher = matcher
            logger.info("Pattern caricati: %d (lemmatizer=%s)", added, has_lemma)
            logger.debug("Pattern labels caricati: %s", [p["label"] for p in patterns])
        except Exception as e:
            _nlp_patterns = None
            _matcher = None
            logger.exception("Errore caricamento pattern: %s", e)

def load_model_pipeline(reload: bool = False):
    """
    Inizializza la pipeline ML (textcat).
    Se non trova il modello addestra, fallback su blank('it').
    """
    global _nlp_model, _loaded_model_time
    if _nlp_model is not None and not reload:
        return
    with _model_lock:
        if _nlp_model is not None and not reload:
            return
        try:
            if MODEL_DIR.exists():
                _nlp_model = spacy.load(MODEL_DIR)
                logger.info("Modello ML caricato da %s", MODEL_DIR)
            else:
                _nlp_model = spacy.blank("it")
                logger.warning("Directory modello ML inesistente, fallback blank.")
        except FileNotFoundError as fnf_err:
            _nlp_model = spacy.blank("it")
            logger.error("Modello ML non trovato: %s", fnf_err)
        except Exception as ex:
            _nlp_model = spacy.blank("it")
            logger.error("Errore caricamento modello ML: %s", ex)

        raw = _load_raw_patterns()
        patterns = _prepare_patterns_for(_nlp_model, raw)
        _ensure_attribute_ruler_from_jsonl(_nlp_model, patterns)
        _loaded_model_time = time.time()

# =========================
# MATCHING E SCORING
# =========================
def _pattern_hits(text: str) -> List[Tuple[str, str]]:
    """
    Esegue il Matcher sui pattern.
    Ritorna lista di (span_text, label).
    """
    load_patterns()
    doc = _nlp_patterns(text)  # type: ignore
    hits: List[Tuple[str, str]] = []
    if _matcher:
        for mid, start, end in _matcher(doc):  # type: ignore
            label = doc.vocab.strings[mid]
            span = doc[start:end]
            hits.append((span.text, label))
    logger.debug("Pattern hits per '%s': %s", text, hits)
    return hits

def _score_pattern_only(hits: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    """
    Converte le occorrenze in punteggi normalizzati (count/max_count).
    Ordina e seleziona i top item.
    """
    counts: Dict[str, int] = {}
    for _, lab in hits:
        counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return []
    max_c = max(counts.values())
    scored = [(lab, counts[lab] / max_c) for lab in counts]
    scored.sort(key=lambda x: x[1], reverse=True)
    return _select_top_items(scored, threshold=0.0, rel_gap=RELATIVE_GAP, max_active=MAX_ACTIVE, force_min_top=FORCE_MIN_TOP)

def _rank_intents(doc) -> List[Tuple[str, float]]:
    """Ordina gli intenti del modello ML in base al punteggio."""
    return sorted(doc.cats.items(), key=lambda kv: kv[1], reverse=True) if hasattr(doc, "cats") else []

def _select_active(sorted_intents: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Applica la selezione top agli intenti del modello ML."""
    return _select_top_items(sorted_intents, threshold=GLOBAL_THRESHOLD, rel_gap=RELATIVE_GAP, max_active=MAX_ACTIVE, force_min_top=FORCE_MIN_TOP)

# =========================
# API INTERNA: ANALISI E CLASSIFICAZIONE
# =========================
def _analyze_intents(text: str) -> Dict[str, Any]:
    """
    Analizza un testo e ritorna:
      - intenti attivi
      - intenti totali
      - hits di pattern
      - sorgente della decisione (pattern o modello)
    """
    hits = _pattern_hits(text)
    if hits:
        active = _score_pattern_only(hits)
        primary = active[0][0] if active else None
        logger.debug("Intents trovati via pattern: %s", active)
        return {
            "text": text,
            "primary_intent": primary,
            "intents_active": [{"label": l, "score": round(s, 4)} for l, s in active],
            "intents_all": [{"label": l, "score": round(s, 4)} for l, s in active],
            "pattern_hits": [{"text": t, "label": lab} for t, lab in hits],
            "decision_source": "pattern",
            "timestamp": time.time(),
        }
    load_model_pipeline()
    doc = _nlp_model(text)  # type: ignore
    ranked = _rank_intents(doc)
    active = _select_active(ranked)
    primary = active[0][0] if active else None
    logger.debug("Intents trovati via modello: %s", active)
    return {
        "text": text,
        "primary_intent": primary,
        "intents_active": [{"label": l, "score": round(s, 4)} for l, s in active],
        "intents_all": [{"label": l, "score": round(s, 4)} for l, s in ranked],
        "pattern_hits": [],
        "decision_source": "model",
        "timestamp": time.time(),
    }

def _classify_top(text: str, k: int = 3):
    """
    Ritorna i top-k intenti dal testo e la sorgente della decisione.
    """
    res = _analyze_intents(text)
    return res["intents_all"][:k], res.get("decision_source")

# =========================
# API PUBBLICA
# =========================
def get_top_three_intents(text: str):
    """
    API pubblica:
    Ritorna le 3 (o meno) intenzioni più probabili per un testo.
    Output: (lista di {label,score}, sorgente)
    """
    top3, source = _classify_top(text, k=3)
    logger.info("Top-3 intenti per '%s': %s (source=%s)", text, top3, source)
    return top3, source

__all__ = ["get_top_three_intents"]
