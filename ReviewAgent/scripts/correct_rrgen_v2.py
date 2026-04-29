"""
Second-pass label correction on the 215K RRGen using the RoBERTa anchor
(models/anchor_roberta/) instead of TF-IDF.

Pipeline:
  1. Load RoBERTa anchor.
  2. Predict 7-class probabilities on all 215,583 RRGen rows.
  3. Run cleanlab.filter.find_label_issues on (LLM labels, anchor probs).
  4. Apply correction policy:
       A. text in verified set    -> use verified label  (source = human_verified)
       B. cleanlab flagged AND anchor confident AND LLM-prob low
                                  -> use anchor label    (source = anchor_corrected_v2)
       C. else                    -> keep LLM label      (source = llm_kept)
  5. Write corrected dataset + correction log + stats.

Usage:
    python3 scripts/correct_rrgen_v2.py
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS


def load_anchor_model(path: Path, device: str):
    print(f"Loading RoBERTa anchor from {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device)
    model.eval()
    return tokenizer, model


def predict_probs_roberta(tokenizer, model, texts, device, batch_size=32, max_length=256):
    """Predict 7-class sigmoid probabilities for all texts."""
    out = np.empty((len(texts), len(LABELS)), dtype=np.float32)
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = texts[i : i + batch_size]
            inputs = tokenizer(chunk, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            out[i : i + batch_size] = probs
            if (i // batch_size) % 50 == 0:
                pct = 100 * (i + len(chunk)) / n
                print(f"  {i + len(chunk):>7,} / {n:,}  ({pct:5.1f}%)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-model", type=Path,
                    default=Path("models/anchor_roberta"))
    ap.add_argument("--noisy", type=Path,
                    default=Path("data/processed/rrgen_full_labeled/rrgen_full_labeled.json"))
    ap.add_argument("--verified", type=Path,
                    default=Path("data/processed/verified_annotations.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/rrgen_corrected_v2"))
    ap.add_argument("--min-anchor-conf", type=float, default=0.70)
    ap.add_argument("--max-llm-prob", type=float, default=0.20)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        print("ERROR: cleanlab not installed. Run: pip install cleanlab", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print(f"\n[1/4] Loading verified annotations: {args.verified}")
    with open(args.verified) as f:
        verified = json.load(f)
    verified_by_text = {v["text"]: v["labels"][0] for v in verified}
    print(f"      {len(verified_by_text):,} unique verified texts")

    print(f"\n[2/4] Loading noisy 215K: {args.noisy}")
    with open(args.noisy) as f:
        noisy = json.load(f)
    texts = [r["text"] for r in noisy]
    llm_labels_str = [r["predicted_label"] for r in noisy]
    llm_labels_idx = np.array([LABELS.index(l) for l in llm_labels_str], dtype=np.int64)
    print(f"      {len(noisy):,} samples")

    print(f"\n[3/4] Predicting probabilities with RoBERTa anchor")
    tokenizer, model = load_anchor_model(args.anchor_model, device)
    pred_probs = predict_probs_roberta(tokenizer, model, texts, device,
                                        batch_size=args.batch_size)

    # Save raw probs for later reuse
    np.save(args.out_dir / "anchor_probs.npy", pred_probs)
    print(f"      saved anchor_probs.npy ({pred_probs.nbytes / 1e6:.1f} MB)")

    print(f"\n[4/4] Running cleanlab + applying corrections")
    issue_idx = set(find_label_issues(
        labels=llm_labels_idx, pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    ).tolist())
    qscores = get_label_quality_scores(labels=llm_labels_idx, pred_probs=pred_probs)
    print(f"      cleanlab flagged {len(issue_idx):,} rows")

    corrected = []
    log_rows = []
    source_counter = Counter()
    transitions = Counter()
    per_class_before = Counter(llm_labels_str)
    per_class_after = Counter()

    for i, r in enumerate(noisy):
        text = r["text"]
        anchor_probs = pred_probs[i]
        anchor_idx = int(anchor_probs.argmax())
        anchor_label = LABELS[anchor_idx]
        anchor_conf = float(anchor_probs[anchor_idx])
        anchor_prob_of_llm = float(anchor_probs[llm_labels_idx[i]])

        if text in verified_by_text:
            final = verified_by_text[text]
            source = "human_verified"
            final_conf = 1.0
        elif (
            i in issue_idx
            and anchor_conf >= args.min_anchor_conf
            and anchor_prob_of_llm <= args.max_llm_prob
            and anchor_label != r["predicted_label"]
        ):
            final = anchor_label
            source = "anchor_corrected_v2"
            final_conf = anchor_conf
        else:
            final = r["predicted_label"]
            source = "llm_kept"
            final_conf = r.get("confidence", anchor_prob_of_llm)

        source_counter[source] += 1
        per_class_after[final] += 1

        corrected.append({
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
        })
        if final != r["predicted_label"]:
            transitions[(r["predicted_label"], final)] += 1
            log_rows.append({
                "idx": i, "text": text, "app_id": r.get("app_id"),
                "llm_label": r["predicted_label"],
                "llm_confidence": r.get("confidence"),
                "final_label": final,
                "anchor_confidence": anchor_conf,
                "label_quality_score": float(qscores[i]),
                "source": source,
            })

    with open(args.out_dir / "rrgen_corrected_v2.json", "w") as f:
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
        "anchor_model": str(args.anchor_model),
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

    print("\n" + "=" * 70)
    print("CORRECTION V2 SUMMARY (RoBERTa anchor)")
    print("=" * 70)
    for src, n in source_counter.most_common():
        pct = 100 * n / len(corrected)
        print(f"  {src:25s} {n:>7,}  ({pct:5.2f}%)")
    print(f"\nTotal changes: {stats['total_changed']:,}  ({100*stats['total_changed']/len(corrected):.2f}%)")

    print("\nClass distribution BEFORE vs AFTER:")
    print(f"  {'class':20s} {'before':>8s} {'after':>8s} {'delta':>8s}")
    for lbl in LABELS:
        b = per_class_before.get(lbl, 0)
        a = per_class_after.get(lbl, 0)
        print(f"  {lbl:20s} {b:>8,} {a:>8,} {a-b:>+8,}")

    print("\nTop 15 transitions (LLM -> corrected):")
    for (a, b), n in transitions.most_common(15):
        print(f"  {a:20s} -> {b:20s}  {n:>5,}")

    print(f"\nOutputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
