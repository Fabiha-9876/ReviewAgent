"""
Find suspected label errors in the LLM-labeled RRGen set (~215K) using a small
verified subset (~5K MAALEJ) as a ground-truth anchor.

Strategy
--------
1. Train a lightweight "anchor" classifier (TF-IDF + Logistic Regression) on the
   verified 5K. This is intentionally small and fast; swap in RoBERTa later once
   the workflow is validated.
2. Predict 7-class probabilities on every RRGen sample.
3. Run cleanlab.filter.find_label_issues with:
       labels     = LLM-assigned predicted_label  (noisy)
       pred_probs = anchor model's probabilities  (independent)
   Disagreements where the anchor is confident => suspected mislabels.
4. Rank suspects by label_quality_score (lower = more suspect) and export
   top-N to CSV for manual review / volunteer verification.

Usage
-----
    pip install cleanlab
    python3 scripts/cleanlab_find_label_issues.py \
        --verified data/raw/maalej/maalej_labeled.json \
        --noisy    data/processed/rrgen_full_labeled/rrgen_full_labeled.json \
        --out-dir  data/processed/label_issues \
        --top-n    5000

Outputs
-------
    data/processed/label_issues/
        suspects_top5000.csv      top-N ranked for review
        suspects_full.json        full ranked list with scores + anchor probs
        noise_matrix.json         estimated LLM confusion on the 5K
        anchor_report.json        anchor classifier metrics on 5K (CV)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS  # 7-class label order


def load_verified(path: Path) -> tuple[list[str], list[str]]:
    """MAALEJ clean anchor. Each record has 'text' and 'labels' (list, first is primary)."""
    with open(path) as f:
        data = json.load(f)
    texts, labels = [], []
    for r in data:
        lbls = r.get("labels") or []
        if not lbls:
            continue
        label = lbls[0]
        if label not in LABELS:
            continue
        texts.append(r["text"])
        labels.append(label)
    return texts, labels


def load_noisy(path: Path) -> list[dict]:
    """RRGen LLM-labeled. Each record has text, predicted_label, all_confidences, etc."""
    with open(path) as f:
        return json.load(f)


def train_anchor(texts: list[str], labels: list[str]) -> tuple[Pipeline, dict]:
    """TF-IDF + Logistic Regression on verified data. Returns fitted pipeline + CV report."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95,
            sublinear_tf=True, max_features=50_000,
        )),
        ("clf", LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1,
        )),
    ])
    cv_preds = cross_val_predict(pipe, texts, labels, cv=5, n_jobs=-1)
    report = classification_report(labels, cv_preds, output_dict=True, zero_division=0)
    pipe.fit(texts, labels)
    return pipe, report


def predict_probs(pipe: Pipeline, texts: list[str], batch_size: int = 10_000) -> np.ndarray:
    """Predict probabilities in column order LABELS. Batched to keep memory bounded."""
    class_order = list(pipe.named_steps["clf"].classes_)
    reorder = [class_order.index(l) for l in LABELS]
    out = np.empty((len(texts), len(LABELS)), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        probs = pipe.predict_proba(chunk)[:, reorder]
        out[i : i + batch_size] = probs
        print(f"  predicted {i + len(chunk):,}/{len(texts):,}", flush=True)
    return out


def estimate_noise_matrix(verified_labels: list[str], noisy_preds_on_verified: np.ndarray) -> dict:
    """LLM-vs-clean confusion on the verified set, normalized per true class."""
    argmax_noisy = noisy_preds_on_verified.argmax(axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.float64)
    for true_lbl, pred_idx in zip(verified_labels, argmax_noisy):
        matrix[LABELS.index(true_lbl), pred_idx] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return {
        "rows_are_true_labels": True,
        "labels": LABELS,
        "counts": matrix.astype(int).tolist(),
        "normalized": (matrix / row_sums).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True, type=Path)
    ap.add_argument("--noisy", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=5000)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="Require anchor_top_prob - prob_of_llm_label >= this to flag (extra filter).")
    args = ap.parse_args()

    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        print("ERROR: cleanlab not installed. Run: pip install cleanlab", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading verified anchor: {args.verified}")
    v_texts, v_labels = load_verified(args.verified)
    print(f"      {len(v_texts):,} verified samples; distribution: {dict(Counter(v_labels))}")

    print(f"[2/6] Loading noisy set: {args.noisy}")
    noisy = load_noisy(args.noisy)
    n_texts = [r["text"] for r in noisy]
    n_labels_str = [r["predicted_label"] for r in noisy]
    label_to_idx = {l: i for i, l in enumerate(LABELS)}
    n_labels_idx = np.array([label_to_idx[l] for l in n_labels_str], dtype=np.int64)
    print(f"      {len(n_texts):,} noisy samples")

    print("[3/6] Training anchor classifier (TF-IDF + LogReg, 5-fold CV)")
    pipe, cv_report = train_anchor(v_texts, v_labels)
    print(f"      CV macro-F1: {cv_report['macro avg']['f1-score']:.4f}")

    print("[4/6] Predicting probabilities on noisy set")
    pred_probs = predict_probs(pipe, n_texts)

    print("[5/6] Running cleanlab.find_label_issues")
    issue_mask = find_label_issues(
        labels=n_labels_idx,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    quality_scores = get_label_quality_scores(labels=n_labels_idx, pred_probs=pred_probs)
    print(f"      cleanlab flagged {len(issue_mask):,} potential label issues")

    # Optional extra filter on confidence margin
    if args.min_margin > 0:
        top_probs = pred_probs.max(axis=1)
        llm_probs = pred_probs[np.arange(len(n_labels_idx)), n_labels_idx]
        margin = top_probs - llm_probs
        issue_mask = np.array([i for i in issue_mask if margin[i] >= args.min_margin])
        print(f"      after margin>={args.min_margin}: {len(issue_mask):,}")

    print("[6/6] Writing outputs")
    suspects = []
    for rank, idx in enumerate(issue_mask[: args.top_n]):
        r = noisy[idx]
        probs = pred_probs[idx]
        anchor_idx = int(probs.argmax())
        suspects.append({
            "rank": rank + 1,
            "idx_in_noisy": int(idx),
            "text": r["text"],
            "app_id": r.get("app_id"),
            "rating": r.get("rating"),
            "llm_label": n_labels_str[idx],
            "llm_confidence": r.get("confidence"),
            "anchor_label": LABELS[anchor_idx],
            "anchor_confidence": float(probs[anchor_idx]),
            "anchor_prob_of_llm_label": float(probs[n_labels_idx[idx]]),
            "label_quality_score": float(quality_scores[idx]),
            "anchor_probs": {l: float(p) for l, p in zip(LABELS, probs)},
        })

    # CSV (top-N for human review)
    import csv
    csv_path = args.out_dir / f"suspects_top{args.top_n}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "idx_in_noisy", "app_id", "rating",
            "llm_label", "llm_confidence",
            "anchor_label", "anchor_confidence", "anchor_prob_of_llm_label",
            "label_quality_score", "text",
        ])
        for s in suspects:
            w.writerow([
                s["rank"], s["idx_in_noisy"], s["app_id"], s["rating"],
                s["llm_label"], s["llm_confidence"],
                s["anchor_label"], s["anchor_confidence"], s["anchor_prob_of_llm_label"],
                s["label_quality_score"], s["text"],
            ])

    # Full JSON
    with open(args.out_dir / "suspects_full.json", "w") as f:
        json.dump(suspects, f, indent=2)

    # Estimated LLM noise matrix on the verified set
    verified_probs = predict_probs(pipe, v_texts)  # anchor on verified (sanity)
    noise = estimate_noise_matrix(v_labels, verified_probs)
    with open(args.out_dir / "noise_matrix.json", "w") as f:
        json.dump(noise, f, indent=2)

    with open(args.out_dir / "anchor_report.json", "w") as f:
        json.dump(cv_report, f, indent=2)

    # Summary
    flagged_by_llm_label = Counter(s["llm_label"] for s in suspects)
    print("\nSuspects by LLM label (top-N):")
    for lbl, cnt in flagged_by_llm_label.most_common():
        print(f"  {lbl:20s} {cnt:,}")
    print(f"\nWrote: {csv_path}")
    print(f"       {args.out_dir / 'suspects_full.json'}")
    print(f"       {args.out_dir / 'noise_matrix.json'}")


if __name__ == "__main__":
    main()
