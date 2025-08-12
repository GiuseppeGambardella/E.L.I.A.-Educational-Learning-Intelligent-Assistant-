# =========================
# DESCRIZIONE E IMPORT
# =========================
"""
Intent recognition: PRIORITA' JSONL (pattern_entities.jsonl) → se nessun match ⇒ modello ML.
Gestisce pattern rules e modello spaCy multilabel per classificazione intenti.
"""
import pathlib
import time
import threading
import json
from typing import List, Dict, Tuple
import os
import spacy
from spacy.matcher import Matcher
import logging


logging.basicConfig(level=logging.INFO)

# =========================
# CONFIGURAZIONE E PARAMETRI
# =========================
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "server" / "models" / "nlp_model" / "best"
PATTERN_FILE = BASE_DIR / "models" / "nlp" / "pattern_entities.jsonl"
BASE_MODEL = os.getenv("INTENTS_BASE_MODEL", "it_core_news_md")

GLOBAL_THRESHOLD = 0.5
PER_LABEL_THRESHOLDS: Dict[str, float] = {}
MAX_ACTIVE = 3
RELATIVE_GAP = 0.12
FORCE_MIN_TOP = 3

# Locks / globals
_patterns_lock = threading.Lock()
_model_lock = threading.Lock()
_nlp_patterns = None
_matcher: Matcher | None = None
_nlp_model = None
_loaded_patterns_time = None
_loaded_model_time = None
_patterns_keep_lemma = False

# =========================
# UTILITY PATTERN MATCHING
# =========================
def _load_raw_patterns() -> List[Dict]:
    # Carica pattern da file JSONL
    patterns = []
    if not PATTERN_FILE.exists():
        return patterns
    with PATTERN_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            obj = json.loads(line)
            patterns.append({"label": obj["label"], "pattern": obj["pattern"]})
    return patterns

def _has_lemma_token(tokens) -> bool:
    # Verifica se pattern contiene LEMMA
    if not isinstance(tokens, list):
        return False
    return any(isinstance(t, dict) and any(k.upper() == "LEMMA" for k in t) for t in tokens)

def _needs_lemma(patterns: List[Dict]) -> bool:
    # Serve lemmatizzazione?
    return any(_has_lemma_token(p["pattern"]) for p in patterns)

def _degrade_lemma_token(tok: dict) -> dict:
    # Converte LEMMA in LOWER se serve
    lemma_val = tok.get("LEMMA") or tok.get("lemma")
    op = tok.get("OP")
    new_tok: Dict = {}
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

def _degrade_patterns(patterns: List[Dict]) -> List[Dict]:
    # Applica degradazione LEMMA→LOWER su tutti i pattern
    out = []
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

def _sanitize_token(tok: dict) -> dict:
    # Pulisce token pattern per matcher spaCy
    allowed = {
        "TEXT","LOWER","POS","TAG","DEP","SHAPE","PREFIX","SUFFIX","LENGTH",
        "IS_ALPHA","IS_ASCII","IS_DIGIT","IS_PUNCT","IS_SPACE","IS_STOP",
        "LIKE_NUM","LIKE_EMAIL","LEMMA","OP"
    }
    clean = {}
    for k, v in tok.items():
        K = k.upper()
        if K == "OP":
            clean["OP"] = v
        elif K in allowed:
            if isinstance(v, dict):
                sub = {}
                for sk, sv in v.items():
                    if sk.upper() in {"IN","NOT_IN"}:
                        sub[sk.upper()] = sv
                if sub:
                    clean[K] = sub
            else:
                clean[K] = v
    return clean

def _sanitize_patterns(patterns: List[Dict]) -> List[Dict]:
    # Pulisce tutti i pattern
    valid = []
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
# INIZIALIZZAZIONE PIPELINE PATTERN E MODELLO
# =========================
def load_patterns(reload: bool = False):
    # Carica pipeline pattern matcher
    global _nlp_patterns, _matcher, _loaded_patterns_time, _patterns_keep_lemma
    if _nlp_patterns is not None and not reload:
        return
    with _patterns_lock:
        if _nlp_patterns is not None and not reload:
            return
        raw = _load_raw_patterns()
        need_lemma = _needs_lemma(raw)
        keep = os.getenv("INTENTS_KEEP_LEMMA") == "1"
        have_lemma = False
        if need_lemma and keep:
            try:
                nlp_pat = spacy.load(BASE_MODEL, disable=[])
                have_lemma = "lemmatizer" in nlp_pat.pipe_names
            except Exception:
                nlp_pat = spacy.blank("it")
        else:
            nlp_pat = spacy.blank("it")

        if need_lemma and not have_lemma:
            raw = _degrade_patterns(raw)
            _patterns_keep_lemma = False
        else:
            _patterns_keep_lemma = need_lemma and have_lemma

        raw = _sanitize_patterns(raw)

        matcher = Matcher(nlp_pat.vocab, validate=True)
        for p in raw:
            try:
                matcher.add(p["label"], [p["pattern"]])
            except Exception:
            except Exception as e:
                logging.warning(f"[IntentRecognition] Failed to add pattern: label={p.get('label')}, pattern={p.get('pattern')}, error={e}")

        _nlp_patterns = nlp_pat
        _matcher = matcher
        _loaded_patterns_time = time.time()

def load_model_pipeline(reload: bool = False):
    # Carica pipeline modello ML spaCy
    global _nlp_model, _loaded_model_time
    if _nlp_model is not None and not reload:
        return
    with _model_lock:
        if _nlp_model is not None and not reload:
            return
        if MODEL_DIR.exists():
            try:
                _nlp_model = spacy.load(MODEL_DIR)
                logging.info(f"[IntentRecognition] Modello ML caricato da {MODEL_DIR}")
            except Exception:
                _nlp_model = spacy.blank("it")
                logging.warning("[IntentRecognition] Errore nel caricamento modello ML, fallback blank.")
        else:
            _nlp_model = spacy.blank("it")
            logging.warning("[IntentRecognition] Modello ML non trovato, fallback blank.")
        _loaded_model_time = time.time()

# =========================
# MATCHING E SCORING
# =========================
def _pattern_hits(text: str) -> List[Tuple[str,str]]:
    # Trova pattern nel testo
    load_patterns()
    doc = _nlp_patterns(text)
    hits = []
    if _matcher:
        for mid, start, end in _matcher(doc):
            label = doc.vocab.strings[mid]
            span = doc[start:end]
            hits.append((span.text, label))
    return hits

def _score_pattern_only(hits: List[Tuple[str,str]]) -> List[Tuple[str,float]]:
    # Calcola score pattern
    counts: Dict[str,int] = {}
    for _, lab in hits:
        counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return []
    max_c = max(counts.values())
    scored = [(lab, counts[lab] / max_c) for lab in counts]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[0][1]
    active = []
    for lab, sc in scored:
        if sc >= top - RELATIVE_GAP:
            active.append((lab, sc))
        if len(active) >= MAX_ACTIVE:
            break
    # Forza almeno le top FORCE_MIN_TOP
    if len(active) < FORCE_MIN_TOP:
        used = {l for l, _ in active}
        for lab, sc in scored:
            if lab in used:
                continue
            active.append((lab, sc))
            if len(active) >= FORCE_MIN_TOP or len(active) >= MAX_ACTIVE:
                break
    return active

def get_threshold(label: str) -> float:
    # Soglia per label
    return PER_LABEL_THRESHOLDS.get(label, GLOBAL_THRESHOLD)

def _rank_intents(doc) -> List[Tuple[str,float]]:
    # Ordina label per score
    return sorted(doc.cats.items(), key=lambda kv: kv[1], reverse=True) if hasattr(doc, "cats") else []

def _select_active(sorted_intents: List[Tuple[str,float]]) -> List[Tuple[str,float]]:
    # Seleziona label attive (sopra soglia o top-N)
    if not sorted_intents:
        return []
    top = sorted_intents[0][1]
    chosen = []
    for lab, score in sorted_intents:
        if score >= get_threshold(lab) and score >= top - RELATIVE_GAP:
            chosen.append((lab, score))
        if len(chosen) >= MAX_ACTIVE:
            break
    if len(chosen) < FORCE_MIN_TOP:
        used = {l for l,_ in chosen}
        for lab, score in sorted_intents:
            if lab in used:
                continue
            chosen.append((lab, score))
            if len(chosen) >= FORCE_MIN_TOP or len(chosen) >= MAX_ACTIVE:
                break
    if not chosen and sorted_intents:
        chosen = [sorted_intents[0]]
    return chosen

# =========================
# API INTERNA: ANALISI E CLASSIFICAZIONE
# =========================
def _analyze_intents(text: str) -> Dict:
    # Analizza testo e restituisce tutte le info (pattern o modello)
    hits = _pattern_hits(text)
    if hits:
        active = _score_pattern_only(hits)
        primary = active[0][0] if active else None
        return {
            "text": text,
            "primary_intent": primary,
            "intents_active": [{"label": l, "score": round(s,4)} for l,s in active],
            "intents_all": [{"label": l, "score": round(s,4)} for l,s in active],
            "pattern_hits": [{"text": t, "label": lab} for t, lab in hits],
            "decision_source": "pattern",
            "rule_patterns_kept_lemma": _patterns_keep_lemma,
            "patterns_loaded_at": _loaded_patterns_time,
            "model_loaded_at": _loaded_model_time,
            "timestamp": time.time()
        }
    load_model_pipeline()
    doc = _nlp_model(text)
    ranked = _rank_intents(doc)
    active = _select_active(ranked)
    primary = active[0][0] if active else None
    return {
        "text": text,
        "primary_intent": primary,
        "intents_active": [{"label": l, "score": round(s,4)} for l,s in active],
        "intents_all": [{"label": l, "score": round(s,4)} for l,s in ranked],
        "pattern_hits": [],
        "decision_source": "model",
        "rule_patterns_kept_lemma": _patterns_keep_lemma,
        "patterns_loaded_at": _loaded_patterns_time,
        "model_loaded_at": _loaded_model_time,
        "timestamp": time.time()
    }

def _classify_intent(text: str):
    # Restituisce la label principale, score e fonte
    res = _analyze_intents(text)
    label = res.get("primary_intent")
    score = 0.0
    if label:
        for it in res.get("intents_active", []):
            if it["label"] == label:
                score = it["score"]
                break
    return label, score, res.get("decision_source")

def _classify_top(text: str, k: int = 3):
    # Restituisce le top-k label ordinate
    res = _analyze_intents(text)
    return res["intents_all"][:k], res.get("decision_source")

# =========================
# API PUBBLICA: SOLO LA FUNZIONE ESPORTATA
# =========================
def get_top_three_intents(text: str):
    """
    Prende in input una stringa di testo e restituisce le 3 (o meno) intenzioni più probabili.
    Output: lista di dict [{'label': ..., 'score': ...}, ...], decision_source
    """
    top3, source = _classify_top(text, k=3)
    return top3, source

__all__ = ["get_top_three_intents"]
