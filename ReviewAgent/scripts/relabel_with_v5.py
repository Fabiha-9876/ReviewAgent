"""
Apply V5 classifier to all 215,583 RRGen rows and produce a unified
dataset with predictions from BOTH V2 and V5, plus the corrected label.

This enables:
  1. Final clean labeled dataset for downstream use (V5 is the production model).
  2. Triple-source validation: V2 (LLM original), V5 (after correction+training),
     and corrected_v2.final_label (cleanlab+anchor pipeline).
  3. Agreement analysis — when do all three agree? when do they disagree?

Output:
    data/processed/rrgen_v5_relabeled/
        rrgen_v5_relabeled.json     215K rows with v2_label, v5_label, corrected_label
        relabel_stats.json          agreement matrix + per-class distributions
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS


def predict(tokenizer, model, texts, device, batch_size=64, max_length=256):
    out_labels = []
    out_confs = []
    out_all = []
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = texts[i : i + batch_size]
            inputs = tokenizer(chunk, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            for p in probs:
                idx = int(p.argmax())
                out_labels.append(LABELS[idx])
                out_confs.append(float(p[idx]))
                out_all.append({l: float(p[j]) for j, l in enumerate(LABELS)})
            if (i // batch_size) % 50 == 0:
                pct = 100 * (i + len(chunk)) / n
                print(f"  {i + len(chunk):>7,} / {n:,}  ({pct:5.1f}%)", flush=True)
    return out_labels, out_confs, out_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=Path("models/stage1_classifier_v5"))
    ap.add_argument("--noisy", type=Path,
                    default=Path("data/processed/rrgen_full_labeled/rrgen_full_labeled.json"))
    ap.add_argument("--corrected-v2", type=Path,
                    default=Path("data/processed/rrgen_corrected_v2/rrgen_corrected_v2.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/rrgen_v5_relabeled"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print(f"\n[1/4] Loading 215K dataset: {args.noisy}")
    with open(args.noisy) as f:
        rows = json.load(f)
    print(f"      {len(rows):,} rows (with V2 LLM labels)")

    print(f"\n[2/4] Loading V2-corrected: {args.corrected_v2}")
    with open(args.corrected_v2) as f:
        v2_corrected = json.load(f)
    # Index by text for fast lookup (texts should match 1:1 in same order)
    if len(v2_corrected) != len(rows):
        print(f"      WARNING: lengths differ ({len(rows)} vs {len(v2_corrected)})")

    print(f"\n[3/4] Loading V5 model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_path)).to(device)
    model.eval()

    print(f"\n[4/4] Running V5 inference on 215K (batch={args.batch_size})")
    texts = [r["text"] for r in rows]
    t0 = time.time()
    v5_labels, v5_confs, v5_all = predict(tokenizer, model, texts, device,
                                           batch_size=args.batch_size)
    dt = time.time() - t0
    print(f"      Done in {dt/60:.1f} min")

    # Build unified dataset
    out = []
    for i, r in enumerate(rows):
        v2_corr = v2_corrected[i] if i < len(v2_corrected) else {}
        out.append({
            "text": r["text"],
            "rating": r.get("rating"),
            "app_id": r.get("app_id"),
            "timestamp": r.get("timestamp"),
            "original_response": r.get("original_response"),

            "v2_label": r["predicted_label"],
            "v2_confidence": r.get("confidence"),

            "v5_label": v5_labels[i],
            "v5_confidence": v5_confs[i],
            "v5_all_confidences": v5_all[i],

            "corrected_v2_label": v2_corr.get("final_label", r["predicted_label"]),
            "corrected_v2_source": v2_corr.get("source", "n/a"),
        })

    with open(args.out_dir / "rrgen_v5_relabeled.json", "w") as f:
        json.dump(out, f)

    # Agreement analysis
    n_v2_v5_agree = sum(1 for r in out if r["v2_label"] == r["v5_label"])
    n_v5_corr_agree = sum(1 for r in out if r["v5_label"] == r["corrected_v2_label"])
    n_all_agree = sum(1 for r in out if r["v2_label"] == r["v5_label"] == r["corrected_v2_label"])

    # Where v2_corrected changed the label, did V5 agree with the change?
    changed = [r for r in out if r["v2_label"] != r["corrected_v2_label"]]
    v5_supports_correction = sum(1 for r in changed if r["v5_label"] == r["corrected_v2_label"])
    v5_disagrees_correction = sum(1 for r in changed if r["v5_label"] == r["v2_label"])
    v5_third_opinion = len(changed) - v5_supports_correction - v5_disagrees_correction

    # Distributions
    v2_dist = Counter(r["v2_label"] for r in out)
    v5_dist = Counter(r["v5_label"] for r in out)
    corr_dist = Counter(r["corrected_v2_label"] for r in out)

    stats = {
        "total_rows": len(out),
        "v5_inference_minutes": dt / 60,
        "agreement": {
            "v2_v5":           {"count": n_v2_v5_agree, "pct": 100 * n_v2_v5_agree / len(out)},
            "v5_corrected":    {"count": n_v5_corr_agree, "pct": 100 * n_v5_corr_agree / len(out)},
            "all_three":       {"count": n_all_agree, "pct": 100 * n_all_agree / len(out)},
        },
        "v2_correction_validation": {
            "rows_where_v2_was_corrected": len(changed),
            "v5_supports_the_correction":  v5_supports_correction,
            "v5_supports_original_v2":     v5_disagrees_correction,
            "v5_third_opinion":            v5_third_opinion,
            "support_rate_pct": 100 * v5_supports_correction / len(changed) if changed else 0,
        },
        "label_distributions": {
            "v2_llm":         dict(v2_dist),
            "v5":             dict(v5_dist),
            "corrected_v2":   dict(corr_dist),
        },
    }
    with open(args.out_dir / "relabel_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Console summary
    print("\n" + "=" * 70)
    print("V5 RELABELING + AGREEMENT ANALYSIS")
    print("=" * 70)
    print(f"V2 vs V5 agree:        {n_v2_v5_agree:>7,} / {len(out):,}  ({100*n_v2_v5_agree/len(out):5.2f}%)")
    print(f"V5 vs corrected_v2:    {n_v5_corr_agree:>7,} / {len(out):,}  ({100*n_v5_corr_agree/len(out):5.2f}%)")
    print(f"All three agree:       {n_all_agree:>7,} / {len(out):,}  ({100*n_all_agree/len(out):5.2f}%)")

    print(f"\nV2 corrections validation (V5 as third opinion):")
    print(f"  rows where V2 corrected: {len(changed):,}")
    print(f"  V5 SUPPORTS correction:  {v5_supports_correction:,}  ({100*v5_supports_correction/len(changed):.2f}%)")
    print(f"  V5 supports orig V2:     {v5_disagrees_correction:,}  ({100*v5_disagrees_correction/len(changed):.2f}%)")
    print(f"  V5 third opinion:        {v5_third_opinion:,}  ({100*v5_third_opinion/len(changed):.2f}%)")

    print(f"\nLabel distribution comparison:")
    print(f"  {'class':20s} {'V2 LLM':>9} {'V5':>9} {'corrected':>9}")
    for lbl in LABELS:
        print(f"  {lbl:20s} {v2_dist.get(lbl,0):>9,} {v5_dist.get(lbl,0):>9,} {corr_dist.get(lbl,0):>9,}")

    print(f"\nOutputs: {args.out_dir}/")


if __name__ == "__main__":
    main()
