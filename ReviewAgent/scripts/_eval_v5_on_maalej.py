"""Run V5 classifier on Maalej dataset (out-of-distribution check).

Maalej has 5 of V5's 7 classes (no compatibility, no performance).
This script:
  1. Predicts V5 labels on all 5008 Maalej reviews.
  2. For the 5 overlapping classes, computes confusion + F1 against Maalej's labels.
  3. Counts and samples V5's "compatibility" predictions on Maalej (Maalej has no
     compatibility label, so this measures whether V5's compatibility detector
     transfers cross-corpus or whether it's just memorizing synthetic-template
     vocabulary.)

Output: data/processed/ablations/v5_on_maalej.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

LABELS = ["bug_report", "feature_request", "performance", "usability",
          "compatibility", "praise", "other"]
BASE = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent")
MAALEJ_PATH = BASE / "data/raw/maalej/maalej_labeled.json"
MODEL_PATH = BASE / "models/stage1_classifier_v5"
OUT_PATH = BASE / "data/processed/ablations/v5_on_maalej.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading Maalej", file=sys.stderr)
with open(MAALEJ_PATH) as f:
    rows = json.load(f)
texts = [r["text"] for r in rows]
gold = [r["labels"][0] if isinstance(r.get("labels"), list) else r.get("label") for r in rows]
print(f"  {len(texts):,} reviews", file=sys.stderr)

# Load V5
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()

print("Predicting", file=sys.stderr)
t0 = time.time()
preds = []
probs_all = []
B = 64
with torch.inference_mode():
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  padding=True, max_length=256).to(device)
        logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(p.argmax(axis=1).tolist())
        probs_all.append(p)
        if i % 1024 == 0:
            print(f"  {i}/{len(texts)} ({time.time()-t0:.1f}s)", file=sys.stderr)

probs_all = np.concatenate(probs_all, axis=0)
v5_labels = [LABELS[i] for i in preds]
print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

# Distribution
from collections import Counter
v5_dist = Counter(v5_labels)
gold_dist = Counter(gold)
print("\nV5 distribution on Maalej:", file=sys.stderr)
for l in LABELS:
    n = v5_dist.get(l, 0)
    pct = 100 * n / len(texts)
    print(f"  {l:20s} {n:>5d}  ({pct:.1f}%)", file=sys.stderr)
print("\nMaalej gold distribution:", file=sys.stderr)
for l, n in gold_dist.most_common():
    pct = 100 * n / len(texts)
    print(f"  {l:20s} {n:>5d}  ({pct:.1f}%)", file=sys.stderr)

# Confusion on overlapping classes
overlapping = [l for l in LABELS if l in gold_dist]
print(f"\nOverlapping classes: {overlapping}", file=sys.stderr)
from sklearn.metrics import classification_report, confusion_matrix
mask = [g in overlapping and v in overlapping for g, v in zip(gold, v5_labels)]
y_true = [g for g, m in zip(gold, mask) if m]
y_pred = [v for v, m in zip(v5_labels, mask) if m]
print(f"  {len(y_true)} reviews where both are in overlapping label set", file=sys.stderr)
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=overlapping).tolist()

# V5 compatibility predictions on Maalej (Maalej has no compat label,
# so these are "what does V5 think looks like compatibility in OOD data?")
v5_compat_idxs = [i for i, l in enumerate(v5_labels) if l == "compatibility"]
print(f"\nV5 compat predictions on Maalej: {len(v5_compat_idxs)} reviews "
      f"({100*len(v5_compat_idxs)/len(texts):.2f}%)", file=sys.stderr)

# Sample 20 for manual inspection (sorted by confidence)
compat_idx_in_LABELS = LABELS.index("compatibility")
compat_confidences = [(i, float(probs_all[i, compat_idx_in_LABELS])) for i in v5_compat_idxs]
compat_confidences.sort(key=lambda x: -x[1])
sample_size = min(20, len(compat_confidences))
samples = []
for i, conf in compat_confidences[:sample_size]:
    samples.append({
        "text": texts[i][:280],
        "v5_compat_prob": round(conf, 3),
        "maalej_label": gold[i],
    })

out = {
    "n_maalej": len(texts),
    "v5_distribution_on_maalej": dict(v5_dist),
    "maalej_gold_distribution": dict(gold_dist),
    "overlapping_classes": overlapping,
    "n_overlapping_evaluable": len(y_true),
    "v5_vs_maalej_classification_report": {
        k: {sk: round(sv, 3) if isinstance(sv, (int, float)) else sv
            for sk, sv in v.items()} if isinstance(v, dict) else v
        for k, v in report.items()
    },
    "confusion_matrix": {
        "labels": overlapping, "matrix": cm,
        "rows_are_gold_cols_are_pred": True,
    },
    "v5_compat_predictions_on_maalej": {
        "n_predicted_compat": len(v5_compat_idxs),
        "fraction_of_corpus": round(len(v5_compat_idxs) / len(texts), 4),
        "top_20_by_confidence": samples,
    },
}
json.dump(out, open(OUT_PATH, "w"), indent=2)
print(f"\nSaved -> {OUT_PATH}", file=sys.stderr)
print(json.dumps({"summary": {
    "macro_f1_overlapping_5class": round(report.get("macro avg", {}).get("f1-score", 0), 3),
    "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 3),
    "n_v5_compat_on_maalej": len(v5_compat_idxs),
    "compat_rate_pct": round(100 * len(v5_compat_idxs) / len(texts), 2),
}}, indent=2))
