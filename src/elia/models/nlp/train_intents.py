# =========================
# IMPORT E PARAMETRI BASE
# =========================
import pathlib, random, yaml, spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

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
    Carica il dataset YAML e restituisce train/dev/test/labels.
    Se dev non esiste, lo crea splittando train.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    labels = list(data.get("labels") or [])
    def _read(split_name):
        rows = data.get(split_name, []) or []
        items = []
        for row in rows:
            text = row["text"]
            cats = {lab: 0.0 for lab in labels}
            for k, v in row.get("cats", {}).items():
                if k not in labels_set:
                    raise ValueError(f"Label non ammessa nel set {split_name}: {k}. Valide: {labels}")
                cats[k] = float(v)
            items.append((text, cats))
        return items
    train = _read("train")
    dev = _read("dev")
    test = _read("test")
    if not dev and test:
        dev = test
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
    Converte una lista (text, cats) in spaCy Example.
    """
    return [Example.from_dict(nlp.make_doc(t), {"cats": c}) for t, c in items]

# =========================
# METRICHE DI VALUTAZIONE
# =========================
def macro_metrics(nlp, items, labels, thr=THRESHOLD):
    """
    Calcola macro precision, recall, F1 per tutte le label.
    """
    from collections import defaultdict
    tp=defaultdict(int); fp=defaultdict(int); fn=defaultdict(int)
    for text, gold in items:
        doc = nlp(text)
        for lab in labels:
            pred = doc.cats.get(lab,0.0) >= thr
            goldv = gold.get(lab,0.0) >= 0.5
            if pred and goldv: tp[lab]+=1
            elif pred and not goldv: fp[lab]+=1
            elif (not pred) and goldv: fn[lab]+=1
    ps=[]; rs=[]; f1s=[]
    for lab in labels:
        p = tp[lab]/(tp[lab]+fp[lab]) if tp[lab]+fp[lab] else 0
        r = tp[lab]/(tp[lab]+fn[lab]) if tp[lab]+fn[lab] else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        ps.append(p); rs.append(r); f1s.append(f1)
    return (sum(ps)/len(ps), sum(rs)/len(rs), sum(f1s)/len(f1s))

def accuracy_metrics(nlp, items, labels, thr=THRESHOLD):
    """
    Calcola micro accuracy (label-level) e subset accuracy (esempio-level).
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
    random.seed(SEED)
    train, dev, test, labels = load_dataset(DATA_YAML)

    # Inizializza modello spaCy e pipe multilabel
    nlp = spacy.load(BASE_MODEL)
    for p in list(nlp.pipe_names):
        if p.startswith("textcat"):
            nlp.remove_pipe(p)
    textcat = nlp.add_pipe("textcat_multilabel", last=True)
    for lab in labels:
        textcat.add_label(lab)

    train_examples = make_examples(nlp, train)
    nlp.initialize(lambda: train_examples)

    best_macro_f1 = -1.0
    no_improve = 0
    best_path = OUT_DIR / "best"
    best_path.mkdir(parents=True, exist_ok=True)

    # Loop di training con early stopping
    for epoch in range(1, EPOCHS + 1):
        losses = {}
        random.shuffle(train)
        for batch in minibatch(train, size=compounding(4.0, 32.0, 1.5)):
            examples = make_examples(nlp, batch)
            nlp.update(examples, losses=losses, drop=DROPOUT)

        macro_p, macro_r, macro_f1 = macro_metrics(nlp, dev, labels, THRESHOLD)
        acc, subset_acc = accuracy_metrics(nlp, dev, labels, THRESHOLD)
        comp_key = [k for k in losses.keys() if k.startswith("textcat")][0]

        print(
            f"Epoch {epoch:02d} | loss={losses.get(comp_key,0):.4f} "
            f"| acc={acc:.3f} subset_acc={subset_acc:.3f} "
            f"| P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}"
        )

        # Salva il modello migliore
        if macro_f1 > best_macro_f1 + 1e-4:
            best_macro_f1 = macro_f1
            no_improve = 0
            nlp.to_disk(best_path)
            print("  * nuovo best salvato")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print("  * early stopping")
                break

    # =========================
    # TEST FINALE
    # =========================
    if test:
        nlp = spacy.load(best_path)
        test_macro_p, test_macro_r, test_macro_f1 = macro_metrics(nlp, test, labels, THRESHOLD)
        test_acc, test_subset_acc = accuracy_metrics(nlp, test, labels, THRESHOLD)
        print("\nTest metrics:")
        print(f"Accuracy={test_acc:.3f} SubsetAcc={test_subset_acc:.3f} "
              f"P={test_macro_p:.3f} R={test_macro_r:.3f} F1={test_macro_f1:.3f}")

    # =========================
    # SALVATAGGIO MODELLO
    # =========================
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUT_DIR)
    print(f"Modello finale salvato in {OUT_DIR}")
    if (best_path / "meta.json").exists():
        print(f"Miglior checkpoint in {best_path} (macro F1={best_macro_f1:.3f})")

if __name__ == "__main__":
    main()