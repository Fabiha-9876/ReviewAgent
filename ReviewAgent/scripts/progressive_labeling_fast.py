"""
Fast Progressive Labeling (No Retraining Between Rounds)
=========================================================

Uses the existing trained classifier to label RRGen reviews in batches.
Retraining is done ONCE at the end (optional), not between rounds.

This is much faster on CPU (~2 min per 10K instead of ~3 hours).

Strategy:
  Round 1: Label 10K  → cumulative  ~10K
  Round 2: Label 10K  → cumulative  ~20K
  Round 3: Label 10K  → cumulative  ~30K
  Round 4+: Label 30K → cumulative  ~60K, 90K, ...

After all rounds: optionally retrain on high-confidence labels.

Usage:
    python3 scripts/progressive_labeling_fast.py --target 10k
    python3 scripts/progressive_labeling_fast.py --target 20k
    python3 scripts/progressive_labeling_fast.py --target 30k
    python3 scripts/progressive_labeling_fast.py --target all   # runs 10k → 20k → 30k
    python3 scripts/progressive_labeling_fast.py --retrain      # retrain on cumulative data
"""

import json
import csv
import sys
import random
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1.classifier import ReviewClassifier, LABELS


# ── Keyword lists for seeding underrepresented categories ────────────────────

PERFORMANCE_KEYWORDS = [
    "slow", "lag", "laggy", "lagging", "battery", "drain", "draining",
    "heat", "heats", "heating", "hot", "overheat", "overheating",
    "memory", "ram", "freeze", "freezing", "hung", "loading",
    "takes forever", "takes long", "takes too long", "so slow",
    "super slow", "very slow", "extremely slow", "sluggish",
    "unresponsive", "cpu", "resource", "heavy", "power consumption",
    "battery life", "eats battery", "kills battery", "battery hog",
]

COMPATIBILITY_KEYWORDS = [
    "doesn't work on", "not working on", "doesn't support",
    "not compatible", "incompatible", "won't work on",
    "blank screen on", "crashes on my", "broken on",
    "android 14", "android 15", "android 13", "android 12",
    "ios 17", "ios 18", "ios 16", "samsung", "pixel", "xiaomi",
    "oneplus", "huawei", "oppo", "ipad", "tablet",
    "iphone 15", "iphone 14", "iphone 13",
    "galaxy s2", "galaxy s23", "galaxy s24", "galaxy a",
    "my device", "my phone", "specific device", "older phone",
    "not optimized", "screen size", "resolution",
]


def keyword_seed(pool, existing_texts):
    """Find real performance & compatibility reviews via keywords."""
    available = [r for r in pool if r["text"] not in existing_texts]

    perf = [r for r in available if any(kw in r["text"].lower() for kw in PERFORMANCE_KEYWORDS)]
    compat = [r for r in available if any(kw in r["text"].lower() for kw in COMPATIBILITY_KEYWORDS)]

    # Remove overlap
    compat_texts = {r["text"] for r in compat}
    perf = [r for r in perf if r["text"] not in compat_texts]

    perf_sample = random.sample(perf, min(150, len(perf)))
    compat_sample = random.sample(compat, min(100, len(compat)))

    seeded = []
    for r in perf_sample:
        seeded.append({
            "text": r["text"], "rating": r["rating"],
            "app_id": r.get("app_id", ""), "timestamp": r.get("timestamp", ""),
            "labels": ["performance"], "confidence": 0.0,
            "source": "keyword_seed", "needs_verification": True,
        })
    for r in compat_sample:
        seeded.append({
            "text": r["text"], "rating": r["rating"],
            "app_id": r.get("app_id", ""), "timestamp": r.get("timestamp", ""),
            "labels": ["compatibility"], "confidence": 0.0,
            "source": "keyword_seed", "needs_verification": True,
        })

    print(f"  Keyword seeded: {len(perf_sample)} performance + {len(compat_sample)} compatibility")
    return seeded


def predict_batch(classifier, reviews, conf_threshold):
    """Run classifier on reviews, split by confidence."""
    texts = [r["text"] for r in reviews]
    all_results = []

    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        all_results.extend(classifier.predict(batch))
        done = min(i + 64, len(texts))
        if done % 2000 < 64:
            print(f"    {done:,}/{len(texts):,}")

    high, low = [], []
    for review, (pred_labels, confidences) in zip(reviews, all_results):
        max_conf = max(confidences.values())
        record = {
            "text": review["text"], "rating": review["rating"],
            "app_id": review.get("app_id", ""), "timestamp": review.get("timestamp", ""),
            "labels": pred_labels, "confidence": round(max_conf, 4),
            "all_confidences": {k: round(v, 4) for k, v in confidences.items()},
            "source": "model_predicted",
            "needs_verification": max_conf < conf_threshold,
        }
        (high if max_conf >= conf_threshold else low).append(record)

    return high, low


def print_distribution(data, title="Distribution"):
    counts = Counter()
    for r in data:
        for l in r["labels"]:
            counts[l] += 1
    print(f"\n  {title}:")
    for label in LABELS:
        c = counts[label]
        pct = c / len(data) * 100 if data else 0
        print(f"    {label:20s}: {c:6d} ({pct:5.1f}%)")
    print(f"    {'TOTAL':20s}: {len(data):6d}")
    return counts


def run_target(target, rrgen_all, classifier, conf_threshold, base_dir):
    """Run labeling up to a target size."""

    targets = {
        "10k": {"size": 10000, "batch": 10000},
        "20k": {"size": 20000, "batch": 10000},
        "30k": {"size": 30000, "batch": 10000},
    }

    if target not in targets:
        print(f"Unknown target: {target}. Use: 10k, 20k, 30k, or all")
        return

    cfg = targets[target]
    output_dir = base_dir / target
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TARGET: {target} labeled reviews")
    print(f"{'='*70}")

    # Load previous cumulative data
    prev_targets = [t for t in ["10k", "20k", "30k"] if t != target]
    cumulative = []
    for prev in ["10k", "20k", "30k"]:
        if prev == target:
            break
        prev_file = base_dir / prev / f"labeled_{prev}.json"
        if prev_file.exists():
            with open(prev_file) as f:
                cumulative = json.load(f)
            print(f"  Loaded previous: {prev} ({len(cumulative):,} reviews)")

    # For 10k, also load synthetic seed data
    if not cumulative:
        seed_path = Path("data/raw/sample_reviews.json")
        if seed_path.exists():
            with open(seed_path) as f:
                synthetic = json.load(f)
            for r in synthetic:
                cumulative.append({
                    "text": r["text"], "rating": r["rating"],
                    "app_id": r.get("app_id", ""), "timestamp": r.get("timestamp", ""),
                    "labels": r["labels"], "confidence": 1.0,
                    "source": r.get("source", "synthetic"),
                    "needs_verification": False,
                })
            print(f"  Loaded seed data: {len(cumulative):,} synthetic reviews")

    existing_texts = {r["text"] for r in cumulative}

    # Filter pool
    pool = [r for r in rrgen_all if 10 <= len(r["text"]) <= 500 and r["text"] not in existing_texts]
    print(f"  Available pool: {len(pool):,}")

    # Keyword seed (only for first round)
    seeded = []
    if target == "10k":
        seeded = keyword_seed(pool, existing_texts)
        seeded_texts = {r["text"] for r in seeded}
        pool = [r for r in pool if r["text"] not in seeded_texts]

    # Sample batch
    batch_size = cfg["batch"]
    batch = random.sample(pool, min(batch_size, len(pool)))
    print(f"  Sampled batch: {len(batch):,}")

    # Predict
    print(f"  Predicting...")
    high, low = predict_batch(classifier, batch, conf_threshold)
    print(f"  High confidence (>={conf_threshold}): {len(high):,}")
    print(f"  Low confidence: {len(low):,}")
    print(f"  Acceptance rate: {len(high)/len(batch)*100:.1f}%")

    # Combine
    new_labels = seeded + high
    cumulative = cumulative + new_labels

    print_distribution(cumulative, f"Cumulative ({target})")

    # Source breakdown
    src_counts = Counter(r.get("source", "unknown") for r in cumulative)
    print(f"\n  Sources:")
    for s, c in src_counts.most_common():
        print(f"    {s:20s}: {c:6d}")

    # Confidence stats
    confs = [r["confidence"] for r in cumulative if r["confidence"] > 0]
    if confs:
        print(f"\n  Confidence: mean={np.mean(confs):.3f} median={np.median(confs):.3f} min={np.min(confs):.3f}")

    # ── Save ─────────────────────────────────────────────────────────────
    # Cumulative
    with open(output_dir / f"labeled_{target}.json", "w") as f:
        json.dump(cumulative, f, indent=2)

    # New labels only
    with open(output_dir / f"new_labels_{target}.json", "w") as f:
        json.dump(new_labels, f, indent=2)

    # Low confidence (for optional HITL)
    with open(output_dir / f"low_confidence_{target}.json", "w") as f:
        json.dump(low, f, indent=2)

    # CSV
    with open(output_dir / f"labeled_{target}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "rating", "labels", "confidence", "source", "needs_verification"])
        for i, r in enumerate(cumulative, 1):
            writer.writerow([i, r["text"], r["rating"], "|".join(r["labels"]),
                             r["confidence"], r.get("source", ""), r.get("needs_verification", "")])

    # Stats
    dist = Counter()
    for r in cumulative:
        for l in r["labels"]:
            dist[l] += 1

    stats = {
        "target": target,
        "timestamp": datetime.now().isoformat(),
        "cumulative_total": len(cumulative),
        "new_labels": len(new_labels),
        "high_confidence": len(high),
        "low_confidence": len(low),
        "seeded": len(seeded),
        "acceptance_rate": round(len(high) / len(batch) * 100, 2),
        "conf_threshold": conf_threshold,
        "label_distribution": {l: dist[l] for l in LABELS},
        "source_distribution": dict(src_counts),
    }
    with open(output_dir / f"stats_{target}.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved to: {output_dir}/")
    print(f"  Cumulative total: {len(cumulative):,}")
    return cumulative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="10k",
                        help="Target: 10k, 20k, 30k, or all")
    parser.add_argument("--conf-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="models/stage1_classifier")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load RRGen once
    print("Loading RRGen dataset...")
    with open("data/raw/rrgen/rrgen_reviews.json") as f:
        rrgen_all = json.load(f)
    print(f"  Total: {len(rrgen_all):,}")

    # Load classifier once
    print(f"Loading classifier from {args.model_path}...")
    classifier = ReviewClassifier.load(args.model_path)
    print(f"  Device: {classifier.device}")

    base_dir = Path("data/processed/progressive")
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.target == "all":
        for t in ["10k", "20k", "30k"]:
            run_target(t, rrgen_all, classifier, args.conf_threshold, base_dir)
    else:
        run_target(args.target, rrgen_all, classifier, args.conf_threshold, base_dir)

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
