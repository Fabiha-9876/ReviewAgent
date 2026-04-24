"""
Label 10K RRGen Reviews Using Trained RoBERTa Classifier
=========================================================

Uses the trained Stage 1 classifier (F1 0.7974) to predict labels
on 10,000 randomly sampled RRGen reviews. Outputs:
  1. JSON with predictions + confidence scores
  2. CSV for easy viewing
  3. Distribution statistics

Usage:
    python3 scripts/label_rrgen_10k.py
    python3 scripts/label_rrgen_10k.py --samples 10000 --batch-size 64
"""

import json
import csv
import sys
import random
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1.classifier import ReviewClassifier, LABELS


def main():
    parser = argparse.ArgumentParser(description="Label RRGen reviews with trained classifier")
    parser.add_argument("--samples", type=int, default=10000, help="Number of reviews to sample")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", type=str, default="models/stage1_classifier",
                        help="Path to trained classifier")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load RRGen data ──────────────────────────────────────────────────
    print("Loading RRGen dataset...")
    with open("data/raw/rrgen/rrgen_reviews.json") as f:
        rrgen_all = json.load(f)
    print(f"  Total RRGen reviews: {len(rrgen_all):,}")

    # Filter out very short reviews (< 10 chars) and very long ones (> 500 chars)
    rrgen_filtered = [r for r in rrgen_all if 10 <= len(r["text"]) <= 500]
    print(f"  After length filter (10-500 chars): {len(rrgen_filtered):,}")

    # Sample 10K
    if len(rrgen_filtered) < args.samples:
        print(f"  Warning: only {len(rrgen_filtered)} reviews available, using all")
        sampled = rrgen_filtered
    else:
        sampled = random.sample(rrgen_filtered, args.samples)
    print(f"  Sampled: {len(sampled):,} reviews")

    # ── Load classifier ──────────────────────────────────────────────────
    print(f"\nLoading classifier from {args.model_path}...")
    classifier = ReviewClassifier.load(args.model_path)
    print(f"  Device: {classifier.device}")
    print(f"  Labels: {LABELS}")

    # ── Run inference in batches ─────────────────────────────────────────
    print(f"\nRunning inference (batch_size={args.batch_size})...")
    all_results = []
    texts = [r["text"] for r in sampled]

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        predictions = classifier.predict(batch)
        all_results.extend(predictions)
        if (i // args.batch_size) % 20 == 0:
            print(f"  Processed {min(i + args.batch_size, len(texts)):,}/{len(texts):,}")

    print(f"  Done. {len(all_results):,} predictions.")

    # ── Build output ─────────────────────────────────────────────────────
    labeled = []
    label_counts = Counter()
    hitl_count = 0
    confidence_sums = {l: 0.0 for l in LABELS}
    confidence_counts = {l: 0 for l in LABELS}

    for review, (pred_labels, confidences) in zip(sampled, all_results):
        needs_hitl = classifier.needs_hitl(confidences)
        if needs_hitl:
            hitl_count += 1

        primary_label = pred_labels[0]
        max_conf = max(confidences.values())

        record = {
            "text": review["text"],
            "rating": review["rating"],
            "app_id": review.get("app_id", ""),
            "timestamp": review.get("timestamp", ""),
            "predicted_labels": pred_labels,
            "primary_label": primary_label,
            "confidence": round(max_conf, 4),
            "all_confidences": {k: round(v, 4) for k, v in confidences.items()},
            "needs_hitl": needs_hitl,
            "has_response": bool(review.get("response")),
            "original_response": review.get("response", ""),
            "source": "rrgen",
        }
        labeled.append(record)

        for pl in pred_labels:
            label_counts[pl] += 1
            confidence_sums[pl] += confidences[pl]
            confidence_counts[pl] += 1

    # ── Statistics ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)
    for label in LABELS:
        count = label_counts[label]
        pct = count / len(labeled) * 100
        avg_conf = confidence_sums[label] / confidence_counts[label] if confidence_counts[label] > 0 else 0
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)  avg_conf={avg_conf:.3f}")

    print(f"\n  Total reviews:     {len(labeled):,}")
    print(f"  Needs HITL review: {hitl_count:,} ({hitl_count/len(labeled)*100:.1f}%)")
    print(f"  High confidence:   {len(labeled) - hitl_count:,} ({(len(labeled)-hitl_count)/len(labeled)*100:.1f}%)")

    # Confidence distribution
    confs = [r["confidence"] for r in labeled]
    print(f"\n  Confidence stats:")
    print(f"    Mean:   {np.mean(confs):.4f}")
    print(f"    Median: {np.median(confs):.4f}")
    print(f"    Min:    {np.min(confs):.4f}")
    print(f"    Max:    {np.max(confs):.4f}")
    print(f"    <0.5:   {sum(1 for c in confs if c < 0.5)}")
    print(f"    <0.7:   {sum(1 for c in confs if c < 0.7)}")
    print(f"    >=0.9:  {sum(1 for c in confs if c >= 0.9)}")

    # ── Save JSON ────────────────────────────────────────────────────────
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "rrgen_10k_labeled.json"
    with open(json_path, "w") as f:
        json.dump(labeled, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = output_dir / "rrgen_10k_labeled.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "text", "rating", "primary_label", "all_labels",
            "confidence", "needs_hitl", "app_id", "timestamp",
            "bug_report", "feature_request", "performance",
            "usability", "compatibility", "praise", "other"
        ])
        for i, r in enumerate(labeled, 1):
            writer.writerow([
                i,
                r["text"],
                r["rating"],
                r["primary_label"],
                "|".join(r["predicted_labels"]),
                r["confidence"],
                r["needs_hitl"],
                r["app_id"],
                r["timestamp"],
                *[r["all_confidences"].get(l, 0) for l in LABELS],
            ])
    print(f"Saved CSV:  {csv_path}")

    # ── Save summary stats ───────────────────────────────────────────────
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(labeled),
        "model_path": args.model_path,
        "seed": args.seed,
        "label_distribution": {l: label_counts[l] for l in LABELS},
        "hitl_needed": hitl_count,
        "hitl_percentage": round(hitl_count / len(labeled) * 100, 2),
        "confidence_stats": {
            "mean": round(float(np.mean(confs)), 4),
            "median": round(float(np.median(confs)), 4),
            "min": round(float(np.min(confs)), 4),
            "max": round(float(np.max(confs)), 4),
        },
    }
    stats_path = output_dir / "rrgen_10k_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
