# =========================
# IMPORT E PARAMETRI BASE
# =========================
import pathlib, random, yaml, spacy, logging
from spacy.training import Example
from spacy.util import minibatch, compounding, fix_random_seed

logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_YAML = BASE_DIR / "models" / "nlp" / "intents.yml"
OUT_DIR = BASE_DIR / "server" / "models" / "nlp_model"
BASE_MODEL = "it_core_news_lg"
EPOCHS = 30
SEED = 42
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 4
THRESHOLD = 0.5

# =========================
# CARICAMENTO DATASET
# =========================
def load_dataset(path):
    """
    Carica il dataset YAML e restituisce:
      - train: lista di (testo, cats)
      - dev: lista di (testo, cats)
      - test: lista di (testo, cats)
      - labels: lista di tutte le label presenti
    
    Ogni elemento è una tupla (text, {label: score}).

    Se il dev set non è presente:
      - se c’è un test set → usa quello come dev
      - altrimenti → splitta il train in 90% train / 10% dev
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    labels = list(data.get("labels") or [])

    def _read(split_name):
        """
        Legge un sottoinsieme (train/dev/test) dal file YAML
        e converte ogni voce in (testo, categorie).
        """
        rows = data.get(split_name, []) or []
        items = []
        for row in rows:
            text = row["text"]
            # inizializza tutte le label a 0.0
            cats = {lab: 0.0 for lab in labels}
            # assegna i valori presenti
            for k, v in row.get("cats", {}).items():
                if k not in labels:
                    raise ValueError(
                        f"Label non ammessa nel set {split_name}: {k}. Valide: {labels}"
                    )
                cats[k] = float(v)
            items.append((text, cats))
        return items

    train = _read("train")
    dev = _read("dev")
    test = _read("test")

    # fallback: usa test come dev se manca
    if not dev and test:
        dev = test
    # fallback: split automatico se manca sia dev che test
    if not dev:
        random.shuffle(train)
        cut = max(1, int(len(train) * 0.9))
        train, dev = train[:cut], train[cut:]

    return train, dev, test, labels

# =========================
# UTILITY: FORMATTAZIONE ESEMPI
# =========================
def make_examples(nlp, items):
    """
    Converte una lista di (testo, cats) in oggetti spaCy `Example`.
    Questo è il formato richiesto da `nlp.update` durante il training.
    """
    return [Example.from_dict(nlp.make_doc(t), {"cats": c}) for t, c in items]

# =========================
# METRICHE DI VALUTAZIONE
# =========================
def macro_metrics(nlp, items, labels, thr=THRESHOLD):
    """
    Calcola precision, recall e F1 in modalità macro (media sulle label).
    
    Parametri:
      - nlp: modello spaCy
      - items: lista (text, cats)
      - labels: lista di label
      - thr: soglia di classificazione (default=0.5)

    Ritorna:
      (precision_macro, recall_macro, f1_macro)
    """
    from collections import defaultdict
    tp=defaultdict(int); fp=defaultdict(int); fn=defaultdict(int)

    # calcola TP, FP, FN
    for text, gold in items:
        doc = nlp(text)
        for lab in labels:
            pred = doc.cats.get(lab,0.0) >= thr
            goldv = gold.get(lab,0.0) >= 0.5
            if pred and goldv: tp[lab]+=1
            elif pred and not goldv: fp[lab]+=1
            elif (not pred) and goldv: fn[lab]+=1

    # calcola metriche per ogni label
    ps=[]; rs=[]; f1s=[]
    for lab in labels:
        p = tp[lab]/(tp[lab]+fp[lab]) if tp[lab]+fp[lab] else 0
        r = tp[lab]/(tp[lab]+fn[lab]) if tp[lab]+fn[lab] else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        ps.append(p); rs.append(r); f1s.append(f1)

    return (sum(ps)/len(ps), sum(rs)/len(rs), sum(f1s)/len(f1s))

def accuracy_metrics(nlp, items, labels, thr=THRESHOLD):
    """
    Calcola due metriche di accuratezza:
      - micro accuracy (a livello di singola label)
      - subset accuracy (a livello di esempio: tutte le label corrette)

    Ritorna:
      (micro_accuracy, subset_accuracy)
    """
    total=0; correct=0
    subset_total=len(items); subset_correct=0

    for text, gold in items:
        doc = nlp(text)
        pred_set=set(); gold_set=set()
        for lab in labels:
            pred = doc.cats.get(lab,0.0) >= thr
            goldv = gold.get(lab,0.0) >= 0.5
            if pred: pred_set.add(lab)
            if goldv: gold_set.add(lab)
            if pred == goldv: correct += 1
            total += 1
        if pred_set == gold_set: subset_correct += 1

    micro_acc = correct/total if total else 0.0
    subset_acc = subset_correct/subset_total if subset_total else 0.0
    return micro_acc, subset_acc

# =========================
# TRAINING E VALIDAZIONE
# =========================
def main():
    """
    Funzione principale di training:
      - Carica dataset YAML
      - Inizializza modello spaCy multilabel
      - Esegue training con early stopping
      - Salva il best model e il modello finale
      - Valuta sul test set se disponibile
    """
    # fissiamo il seed per riproducibilità
    fix_random_seed(SEED)
    random.seed(SEED)

    # carica dataset
    train, dev, test, labels = load_dataset(DATA_YAML)

    # inizializza modello spaCy di base
    nlp = spacy.load(BASE_MODEL)

    # rimuovi eventuali vecchie pipe textcat
    for p in list(nlp.pipe_names):
        if p.startswith("textcat"):
            nlp.remove_pipe(p)

    # aggiungi classificatore multilabel
    textcat = nlp.add_pipe("textcat_multilabel", last=True)
    for lab in labels:
        textcat.add_label(lab)

    # inizializza pesi
    train_examples = make_examples(nlp, train)
    nlp.initialize(lambda: train_examples)

    best_macro_f1 = -1.0
    no_improve = 0
    best_path = OUT_DIR / "best"
    best_path.mkdir(parents=True, exist_ok=True)

    # loop di training
    for epoch in range(1, EPOCHS + 1):
        losses = {}
        random.shuffle(train)

        # minibatch dinamico
        for batch in minibatch(train, size=compounding(4.0, 32.0, 1.5)):
            examples = make_examples(nlp, batch)
            nlp.update(examples, losses=losses, drop=DROPOUT)

        # calcola metriche su dev
        macro_p, macro_r, macro_f1 = macro_metrics(nlp, dev, labels, THRESHOLD)
        acc, subset_acc = accuracy_metrics(nlp, dev, labels, THRESHOLD)
        comp_key = [k for k in losses.keys() if k.startswith("textcat")][0]

        logger.info(
            "Epoch %02d | loss=%.4f | acc=%.3f subset_acc=%.3f | P=%.3f R=%.3f F1=%.3f",
            epoch, losses.get(comp_key,0), acc, subset_acc, macro_p, macro_r, macro_f1
        )

        # early stopping: salva il best
        if macro_f1 > best_macro_f1 + 1e-4:
            best_macro_f1 = macro_f1
            no_improve = 0
            nlp.to_disk(best_path)
            logger.info("  * nuovo best salvato (macro F1=%.3f)", best_macro_f1)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info("  * early stopping attivato")
                break

    # =========================
    # TEST FINALE
    # =========================
    if test:
        nlp = spacy.load(best_path)
        test_macro_p, test_macro_r, test_macro_f1 = macro_metrics(nlp, test, labels, THRESHOLD)
        test_acc, test_subset_acc = accuracy_metrics(nlp, test, labels, THRESHOLD)
        logger.info("Test metrics | Acc=%.3f SubsetAcc=%.3f P=%.3f R=%.3f F1=%.3f",
                    test_acc, test_subset_acc, test_macro_p, test_macro_r, test_macro_f1)

    # =========================
    # SALVATAGGIO MODELLO
    # =========================
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUT_DIR)
    logger.info("Modello finale salvato in %s", OUT_DIR)
    if (best_path / "meta.json").exists():
        logger.info("Miglior checkpoint in %s (macro F1=%.3f)", best_path, best_macro_f1)

if __name__ == "__main__":
    main()
