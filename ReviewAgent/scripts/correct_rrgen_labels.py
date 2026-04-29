"""
Produce a corrected version of the 215K RRGen labeled dataset by leveraging:
  1. The 5,230 manually verified annotations (ground truth -> always win).
  2. An anchor classifier trained on verified + MAALEJ (used to correct
     high-confidence LLM mistakes on the remaining ~210K).
  3. cleanlab's statistical label-issue detection (for the "should we trust
     the LLM label or the anchor?" decision).

Correction policy per row (in priority order):
  A. If text matches a verified annotation -> use verified label
     (source = "human_verified", confidence = 1.0)
  B. Else if cleanlab flagged the row AND anchor confidence >= min_anchor_conf
     AND anchor prob of the LLM's label <= max_llm_prob
     -> replace with anchor label
     (source = "anchor_corrected")
  C. Else keep the LLM label
     (source = "llm_kept")

Usage:
    python3 scripts/correct_rrgen_labels.py \
        --anchor  data/processed/anchor_combined.json \
        --noisy   data/processed/rrgen_full_labeled/rrgen_full_labeled.json \
        --verified data/processed/verified_annotations.json \
        --out-dir data/processed/rrgen_corrected

Outputs:
    rrgen_corrected.json         Full 215K with final labels + provenance
    correction_stats.json        Counts per source, per-class deltas, noise matrix
    correction_log.csv           Only the rows whose label changed (for audit)
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS


def load_anchor(path: Path):
    with open(path) as f:
        data = json.load(f)
    texts, labels = [], []
    for r in data:
        lbls = r.get("labels") or []
        if not lbls or lbls[0] not in LABELS:
            continue
        texts.append(r["text"])
        labels.append(lbls[0])
    return texts, labels


def train_anchor(texts, labels):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95,
            sublinear_tf=True, max_features=50_000,
        )),
        ("clf", LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1,
        )),
    ])
    pipe.fit(texts, labels)
    return pipe


def predict_probs(pipe, texts, batch=10_000):
    order = list(pipe.named_steps["clf"].classes_)
    reorder = [order.index(l) for l in LABELS]
    out = np.empty((len(texts), len(LABELS)), dtype=np.float32)
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        out[i : i + batch] = pipe.predict_proba(chunk)[:, reorder]
        print(f"  predicted {i + len(chunk):,}/{len(texts):,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, type=Path)
    ap.add_argument("--noisy", required=True, type=Path)
    ap.add_argument("--verified", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-anchor-conf", type=float, default=0.70,
                    help="Required anchor confidence to override LLM label.")
    ap.add_argument("--max-llm-prob", type=float, default=0.20,
                    help="Anchor's probability for the LLM label must be <= this to override.")
    args = ap.parse_args()

    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        print("ERROR: cleanlab not installed. Run: pip install cleanlab", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading anchor: {args.anchor}")
    a_texts, a_labels = load_anchor(args.anchor)
    print(f"      {len(a_texts):,} anchor samples")

    print(f"[2/6] Loading verified annotations: {args.verified}")
    with open(args.verified) as f:
        verified = json.load(f)
    # Build text -> verified-label lookup
    verified_by_text = {}
    for v in verified:
        verified_by_text[v["text"]] = v["labels"][0]
    print(f"      {len(verified_by_text):,} unique verified texts")

    print(f"[3/6] Loading noisy 215K: {args.noisy}")
    with open(args.noisy) as f:
        noisy = json.load(f)
    texts = [r["text"] for r in noisy]
    llm_labels_str = [r["predicted_label"] for r in noisy]
    llm_labels_idx = np.array([LABELS.index(l) for l in llm_labels_str], dtype=np.int64)
    print(f"      {len(noisy):,} samples")

    print("[4/6] Training anchor classifier (TF-IDF + LogReg)")
    pipe = train_anchor(a_texts, a_labels)

    print("[5/6] Predicting probabilities on 215K")
    pred_probs = predict_probs(pipe, texts)

    print("     Running cleanlab to flag likely label issues")
    issue_idx = set(find_label_issues(
        labels=llm_labels_idx, pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    ).tolist())
    qscores = get_label_quality_scores(labels=llm_labels_idx, pred_probs=pred_probs)
    print(f"     cleanlab flagged {len(issue_idx):,} rows")

    print("[6/6] Applying corrections")
    corrected = []
    log_rows = []
    source_counter = Counter()
    transitions = Counter()  # (old_label, new_label) -> count (only when changed)
    per_class_before = Counter(llm_labels_str)
    per_class_after = Counter()

    for i, r in enumerate(noisy):
        text = r["text"]
        anchor_probs = pred_probs[i]
        anchor_idx = int(anchor_probs.argmax())
        anchor_label = LABELS[anchor_idx]
        anchor_conf = float(anchor_probs[anchor_idx])
        anchor_prob_of_llm = float(anchor_probs[llm_labels_idx[i]])

        # Rule A: human verified
        if text in verified_by_text:
            final = verified_by_text[text]
            source = "human_verified"
            final_conf = 1.0
        # Rule B: cleanlab flagged + anchor confident + LLM label has low anchor prob
        elif (
            i in issue_idx
            and anchor_conf >= args.min_anchor_conf
            and anchor_prob_of_llm <= args.max_llm_prob
            and anchor_label != r["predicted_label"]
        ):
            final = anchor_label
            source = "anchor_corrected"
            final_conf = anchor_conf
        # Rule C: keep LLM label
        else:
            final = r["predicted_label"]
            source = "llm_kept"
            final_conf = r.get("confidence", anchor_prob_of_llm)

        source_counter[source] += 1
        per_class_after[final] += 1

        corrected_row = {
            "text": text,
            "rating": r.get("rating"),
            "app_id": r.get("app_id"),
            "timestamp": r.get("timestamp"),
            "original_response": r.get("original_response"),
            "llm_label": r["predicted_label"],
            "llm_confidence": r.get("confidence"),
            "final_label": final,
            "final_confidence": final_conf,
            "source": source,
            "anchor_label": anchor_label,
            "anchor_confidence": anchor_conf,
            "label_quality_score": float(qscores[i]),
        }
        corrected.append(corrected_row)

        if final != r["predicted_label"]:
            transitions[(r["predicted_label"], final)] += 1
            log_rows.append({
                "idx": i,
                "text": text,
                "app_id": r.get("app_id"),
                "llm_label": r["predicted_label"],
                "llm_confidence": r.get("confidence"),
                "final_label": final,
                "anchor_confidence": anchor_conf,
                "label_quality_score": float(qscores[i]),
                "source": source,
            })

    # Write outputs
    with open(args.out_dir / "rrgen_corrected.json", "w") as f:
        json.dump(corrected, f)

    with open(args.out_dir / "correction_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "idx", "text", "app_id", "llm_label", "llm_confidence",
            "final_label", "anchor_confidence", "label_quality_score", "source",
        ])
        w.writeheader()
        w.writerows(log_rows)

    stats = {
        "total_rows": len(corrected),
        "sources": dict(source_counter),
        "total_changed": sum(v for k, v in source_counter.items() if k != "llm_kept"),
        "per_class_before": dict(per_class_before),
        "per_class_after": dict(per_class_after),
        "top_transitions": [
            {"from": a, "to": b, "count": n}
            for (a, b), n in transitions.most_common(30)
        ],
        "thresholds": {
            "min_anchor_conf": args.min_anchor_conf,
            "max_llm_prob": args.max_llm_prob,
        },
        "cleanlab_flagged": len(issue_idx),
    }
    with open(args.out_dir / "correction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Console summary
    print("\n" + "=" * 70)
    print("CORRECTION SUMMARY")
    print("=" * 70)
    for src, n in source_counter.most_common():
        pct = 100 * n / len(corrected)
        print(f"  {src:20s} {n:>7,}  ({pct:5.2f}%)")
    print(f"\nTotal label changes: {stats['total_changed']:,}  ({100*stats['total_changed']/len(corrected):.2f}%)")

    print("\nLabel distribution BEFORE vs AFTER:")
    print(f"  {'class':20s} {'before':>8s} {'after':>8s} {'delta':>8s}")
    for lbl in LABELS:
        b = per_class_before.get(lbl, 0)
        a = per_class_after.get(lbl, 0)
        print(f"  {lbl:20s} {b:>8,} {a:>8,} {a-b:>+8,}")

    print("\nTop 15 transitions (LLM -> corrected):")
    for (a, b), n in transitions.most_common(15):
        print(f"  {a:20s} -> {b:20s}  {n:>5,}")

    print(f"\nOutputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
