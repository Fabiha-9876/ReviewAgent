"""
Build a frozen held-out test set that no classifier has trained on, then
evaluate V3, V4, and V5 on it for a fair head-to-head comparison.

Steps:
  1. Reconstruct each model's training texts deterministically by replaying
     the stratified_cap + 80/10/10 split logic on the same source data with
     seed=42 (matching train_classifier_v3.py).
  2. Compute the set of texts that NO model trained or evaluated on (union
     of all three models' selected_records).
  3. Stratify-sample 5,000 frozen test rows by V5 label from the untrained
     pool. If a class has fewer untrained candidates than the target, take
     all available.
  4. Run V3, V4, V5 inference on the frozen test set.
  5. Report a fair comparison table: same test set, same metrics.

Output:
    data/processed/holdout_frozen/holdout_5k.json
    data/processed/holdout_frozen/v3_v4_v5_comparison.json
    data/processed/holdout_frozen/v3_v4_v5_comparison.csv
"""

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS

V5_RELABEL = Path("data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json")
SOURCES = {
    "v3": Path("data/processed/rrgen_corrected/rrgen_corrected.json"),
    "v4": Path("data/processed/rrgen_corrected_v2/rrgen_corrected_v2.json"),
    "v5": Path("data/processed/rrgen_v5_training.json"),
}
MODELS = {
    "v3": Path("models/stage1_classifier_v3"),
    "v4": Path("models/stage1_classifier_v4"),
    "v5": Path("models/stage1_classifier_v5"),
}
OUT_DIR = Path("data/processed/holdout_frozen")
SEED = 42
HOLDOUT_SEED = 999
MAX_PER_CLASS = 15_000
TARGET_PER_LABEL = 700  # 5 actionable + praise + other → ~5K target


def stratified_cap(records, key_fn, max_per_class, seed=42):
    """Replay the exact logic from train_classifier_v3.py."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[key_fn(r)].append(r)
    out = []
    for cls, rows in by_class.items():
        rng.shuffle(rows)
        out.extend(rows[:max_per_class])
    rng.shuffle(out)
    return out


def reconstruct_training_texts(source_path: Path) -> set[str]:
    """Replay stratified_cap with seed=42 on the source data, return all 'used' texts.

    All rows that survived the cap end up in train/val/test for that model — none
    were excluded from inference, so excluding all of them gives true held-out.
    """
    with open(source_path) as f:
        records = json.load(f)
    label_field = "final_label" if any("final_label" in r for r in records[:5]) else "labels"
    key_fn = (lambda r: r["final_label"]) if label_field == "final_label" \
             else (lambda r: r["labels"][0])
    capped = stratified_cap(records, key_fn, MAX_PER_CLASS, SEED)
    return set(r["text"] for r in capped)


def predict_with_model(model_path: Path, texts: list[str], device: str,
                       batch_size: int = 64) -> tuple[list[str], list[float]]:
    print(f"  loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()
    preds = []
    confs = []
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = texts[i:i+batch_size]
            inputs = tokenizer(chunk, padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            for p in probs:
                idx = int(p.argmax())
                preds.append(LABELS[idx])
                confs.append(float(p[idx]))
            if (i // batch_size) % 20 == 0:
                print(f"    {i + len(chunk):>5,}/{n:,}", flush=True)
    # Free memory
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return preds, confs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\n[1/5] Loading V5-relabeled 215K: {V5_RELABEL}")
    with open(V5_RELABEL) as f:
        all_rows = json.load(f)
    print(f"  {len(all_rows):,} rows")

    print(f"\n[2/5] Reconstructing each model's used texts")
    used_v3 = reconstruct_training_texts(SOURCES["v3"])
    print(f"  V3 used:  {len(used_v3):,}")
    used_v4 = reconstruct_training_texts(SOURCES["v4"])
    print(f"  V4 used:  {len(used_v4):,}")
    used_v5 = reconstruct_training_texts(SOURCES["v5"])
    print(f"  V5 used:  {len(used_v5):,}")
    union = used_v3 | used_v4 | used_v5
    print(f"  UNION:    {len(union):,}  (texts excluded as 'seen by some model')")

    print(f"\n[3/5] Sampling frozen test from untrained pool")
    untrained = [r for r in all_rows if r["text"] not in union]
    print(f"  untrained pool: {len(untrained):,}")

    # Stratify by V5 label
    by_label = defaultdict(list)
    for r in untrained:
        by_label[r["v5_label"]].append(r)
    print(f"  untrained per V5 label:")
    for lbl in LABELS:
        print(f"    {lbl:18s} {len(by_label[lbl]):>7,}")

    rng = random.Random(HOLDOUT_SEED)
    holdout = []
    for lbl in LABELS:
        pool = by_label[lbl]
        rng.shuffle(pool)
        take = min(TARGET_PER_LABEL, len(pool))
        holdout.extend(pool[:take])
    print(f"\n  Frozen holdout: {len(holdout):,} rows")
    final_dist = Counter(r["v5_label"] for r in holdout)
    for lbl in LABELS:
        print(f"    {lbl:18s} {final_dist.get(lbl,0):>5,}")

    # Save the frozen test set
    holdout_path = OUT_DIR / "holdout_5k.json"
    with open(holdout_path, "w") as f:
        json.dump(holdout, f)
    print(f"  saved → {holdout_path}")

    print(f"\n[4/5] Running V3 / V4 / V5 inference on frozen holdout")
    texts = [r["text"] for r in holdout]
    # Ground truth = cleanlab-corrected labels (independent of any single classifier).
    # Using v5_label as truth would be circular for V5 (it's V5's own predictions).
    # V4 trained directly on this distribution (likely advantage); V3 trained on V1
    # corrections; V5 trained on V2 + compat aug. Acknowledged in the report.
    truth = [r["corrected_v2_label"] for r in holdout]

    results = {}
    for name in ["v3", "v4", "v5"]:
        print(f"\n  Model: {name}")
        t0 = time.time()
        preds, confs = predict_with_model(MODELS[name], texts, device)
        dt = time.time() - t0
        results[name] = {"preds": preds, "confs": confs, "inference_sec": dt}
        print(f"  done in {dt:.1f}s")

    print(f"\n[5/5] Computing fair comparison metrics")
    truth_dist = Counter(truth)
    summary = {
        "holdout_size": len(holdout),
        "ground_truth_label": "corrected_v2_label",
        "ground_truth_distribution": dict(truth_dist),
        "stratification_label": "v5_label",
        "per_v5_label_counts": dict(final_dist),
        "caveat": "V4 was trained on the corrected_v2_label distribution and may have a structural advantage. V5 trained on V2-corrected + compat aug. V3 trained on V1-corrected.",
        "models": {},
    }

    print("\n" + "="*70)
    print("FAIR HEAD-TO-HEAD ON FROZEN HOLDOUT")
    print("="*70)
    print(f"Holdout: {len(holdout):,} rows  |  ground truth: corrected_v2_label  |  seed: {HOLDOUT_SEED}")
    print(f"None of these rows appeared in V3, V4, or V5 training.")
    print(f"Note: V4 trained on this label distribution and may have an advantage.\n")

    for name in ["v3", "v4", "v5"]:
        preds = results[name]["preds"]
        rep = classification_report(truth, preds, target_names=LABELS,
                                     output_dict=True, zero_division=0,
                                     labels=LABELS)
        macro = rep["macro avg"]["f1-score"]
        micro = f1_score(truth, preds, average="micro", labels=LABELS, zero_division=0)
        weighted = rep["weighted avg"]["f1-score"]

        summary["models"][name] = {
            "macro_f1": round(macro, 4),
            "micro_f1": round(float(micro), 4),
            "weighted_f1": round(weighted, 4),
            "per_class_f1": {l: round(rep[l]["f1-score"], 3) for l in LABELS},
            "inference_sec": round(results[name]["inference_sec"], 1),
        }

        print(f"\n[{name.upper()}] inference {results[name]['inference_sec']:.1f}s")
        print(f"  Macro F1:    {macro:.4f}")
        print(f"  Micro F1:    {micro:.4f}")
        print(f"  Weighted F1: {weighted:.4f}")
        print(f"  Per-class F1:")
        for l in LABELS:
            print(f"    {l:18s} {rep[l]['f1-score']:.3f}  (sup={int(rep[l]['support'])})")

    # Comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'metric':18s} {'V3':>9} {'V4':>9} {'V5':>9}")
    for metric in ["macro_f1", "micro_f1", "weighted_f1"]:
        row = f"{metric:18s}"
        for m in ["v3", "v4", "v5"]:
            row += f" {summary['models'][m][metric]:>9.4f}"
        print(row)
    print()
    print(f"{'class':18s} {'V3':>9} {'V4':>9} {'V5':>9}")
    for l in LABELS:
        row = f"{l:18s}"
        for m in ["v3", "v4", "v5"]:
            row += f" {summary['models'][m]['per_class_f1'][l]:>9.3f}"
        print(row)

    with open(OUT_DIR / "v3_v4_v5_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    import csv
    with open(OUT_DIR / "v3_v4_v5_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + ["V3", "V4", "V5"])
        for metric in ["macro_f1", "micro_f1", "weighted_f1"]:
            w.writerow([metric] + [summary["models"][m][metric] for m in ["v3","v4","v5"]])
        w.writerow([])
        w.writerow(["per_class_f1", "V3", "V4", "V5"])
        for l in LABELS:
            w.writerow([l] + [summary["models"][m]["per_class_f1"][l] for m in ["v3","v4","v5"]])

    print(f"\nSaved:")
    print(f"  {OUT_DIR/'holdout_5k.json'}")
    print(f"  {OUT_DIR/'v3_v4_v5_comparison.json'}")
    print(f"  {OUT_DIR/'v3_v4_v5_comparison.csv'}")


if __name__ == "__main__":
    main()
