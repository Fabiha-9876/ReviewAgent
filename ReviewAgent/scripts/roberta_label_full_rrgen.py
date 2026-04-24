"""
RoBERTa-Based Labeling of FULL RRGen Dataset (310K)
=====================================================

Labels ALL 310K RRGen reviews using trained RoBERTa classifier.
No keyword filtering — uses the complete raw dataset.

Usage:
    python3 -u scripts/roberta_label_full_rrgen.py

    # Resume if interrupted
    python3 -u scripts/roberta_label_full_rrgen.py --resume

Output:
    data/processed/rrgen_full_labeled/
        rrgen_full_labeled.json       — all 310K labeled reviews
        rrgen_full_stats.json         — distribution + stats
        rrgen_full_checkpoint.json    — for resuming
"""

import json
import sys
import random
import argparse
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1.classifier import ReviewClassifier, LABELS


def main():
    parser = argparse.ArgumentParser(description="Label full 310K RRGen with RoBERTa")
    parser.add_argument("--model-path", default="models/stage1_classifier",
                        help="Path to trained RoBERTa classifier")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--checkpoint-every", type=int, default=10000,
                        help="Save checkpoint every N reviews (default: 10000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path("data/processed/rrgen_full_labeled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load RRGen ───────────────────────────────────────────────────────
    print("Loading RRGen dataset...")
    with open("data/raw/rrgen/rrgen_reviews.json") as f:
        rrgen_all = json.load(f)
    print(f"  Total RRGen reviews: {len(rrgen_all):,}")

    # ── Filter only by length (keep everything else) ─────────────────────
    print("\nFiltering by length (min 5 chars)...")
    reviews = []
    seen_texts = set()
    skipped_short = 0
    skipped_dup = 0

    for r in rrgen_all:
        text = r.get("text", "").strip()
        if len(text) < 5:
            skipped_short += 1
            continue
        if text in seen_texts:
            skipped_dup += 1
            continue
        seen_texts.add(text)
        reviews.append(r)

    print(f"  Kept: {len(reviews):,}")
    print(f"  Skipped (too short): {skipped_short:,}")
    print(f"  Skipped (duplicate): {skipped_dup:,}")

    # ── Resume support ───────────────────────────────────────────────────
    start_idx = 0
    results = []

    checkpoint_path = output_dir / "rrgen_full_checkpoint.json"
    if args.resume and checkpoint_path.exists():
        print("\nResuming from checkpoint...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        results = checkpoint["results"]
        start_idx = checkpoint["next_idx"]
        print(f"  Loaded {len(results):,} already labeled, starting from index {start_idx:,}")

    # ── Load classifier ──────────────────────────────────────────────────
    print(f"\nLoading RoBERTa classifier from {args.model_path}...")
    classifier = ReviewClassifier.load(args.model_path)
    print(f"  Device: {classifier.device}")

    # ── Label all reviews ────────────────────────────────────────────────
    total = len(reviews)
    remaining = total - start_idx
    print(f"\nLabeling {remaining:,} reviews (total: {total:,}, batch_size: {args.batch_size})...")

    start_time = time.time()
    batch_size = args.batch_size

    for i in range(start_idx, total, batch_size):
        batch_reviews = reviews[i:i + batch_size]
        batch_texts = [r["text"] for r in batch_reviews]

        predictions = classifier.predict(batch_texts)

        for review, (pred_labels, confidences) in zip(batch_reviews, predictions):
            primary_label = pred_labels[0] if pred_labels else "other"
            max_conf = max(confidences.values()) if confidences else 0

            results.append({
                "text": review["text"],
                "rating": review.get("rating", 0),
                "app_id": review.get("app_id", ""),
                "timestamp": review.get("timestamp", ""),
                "original_response": review.get("response", ""),
                "predicted_label": primary_label,
                "confidence": round(max_conf, 4),
                "all_confidences": {k: round(v, 4) for k, v in confidences.items()},
                "needs_hitl": classifier.needs_hitl(confidences),
                "source": "rrgen",
            })

        done = min(i + batch_size, total)

        # Progress update every 640 reviews
        if done % (batch_size * 10) == 0 or done == total:
            elapsed = time.time() - start_time
            rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            eta_sec = (total - done) / rate if rate > 0 else 0
            eta_min = eta_sec / 60
            print(f"    {done:,}/{total:,} done ({rate:.0f}/sec, ETA: {eta_min:.0f} min)")

        # Save checkpoint
        if done % args.checkpoint_every == 0 and done > start_idx:
            checkpoint_data = {
                "next_idx": done,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)
            print(f"    [checkpoint saved at {done:,}]")

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed/60:.1f} min ({len(results)/(elapsed):.0f} reviews/sec)")

    # ── Statistics ────────────────────────────────────────────────────────
    label_counts = Counter(r["predicted_label"] for r in results)
    hitl_count = sum(1 for r in results if r.get("needs_hitl"))
    confidences = [r["confidence"] for r in results]

    print("\n" + "=" * 60)
    print(f"FULL RRGEN LABELING RESULTS ({len(results):,} reviews)")
    print("=" * 60)
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = count / len(results) * 100 if results else 0
        print(f"  {label:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n  Total labeled:     {len(results):,}")
    print(f"  Needs HITL:        {hitl_count:,} ({hitl_count/len(results)*100:.1f}%)")
    print(f"  High confidence:   {len(results)-hitl_count:,} ({(len(results)-hitl_count)/len(results)*100:.1f}%)")

    print(f"\n  Confidence stats:")
    print(f"    Mean:   {np.mean(confidences):.4f}")
    print(f"    Median: {np.median(confidences):.4f}")
    print(f"    >=0.8:  {sum(1 for c in confidences if c >= 0.8):,}")
    print(f"    >=0.9:  {sum(1 for c in confidences if c >= 0.9):,}")
    print(f"    <0.5:   {sum(1 for c in confidences if c < 0.5):,}")

    # ── Save final output ────────────────────────────────────────────────
    out_path = output_dir / "rrgen_full_labeled.json"
    print(f"\n  Saving {len(results):,} results to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    stats = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "total_rrgen_raw": len(rrgen_all),
        "skipped_short": skipped_short,
        "skipped_duplicate": skipped_dup,
        "total_labeled": len(results),
        "needs_hitl": hitl_count,
        "hitl_percentage": round(hitl_count / len(results) * 100, 2),
        "label_distribution": {l: label_counts.get(l, 0) for l in LABELS},
        "confidence_mean": round(float(np.mean(confidences)), 4),
        "confidence_median": round(float(np.median(confidences)), 4),
    }
    stats_path = output_dir / "rrgen_full_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # ── Clean up checkpoint ──────────────────────────────────────────────
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Removed checkpoint file")

    print(f"\n{'='*60}")
    print(f"DONE! Full RRGen dataset labeled.")
    print(f"  Total: {len(results):,} reviews")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
