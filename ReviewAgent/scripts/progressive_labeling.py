"""
Progressive Semi-Supervised Labeling Pipeline
===============================================

Iteratively expands labeled dataset using trained RoBERTa classifier:

  Round 1:  0 → 10K   (seed + high-confidence from 10K unlabeled)
  Round 2: 10K → 20K  (retrain → predict next 10K → take high-conf)
  Round 3: 20K → 30K  (retrain → predict next 10K → take high-conf)
  Round 4+: +30K each (retrain → predict next 30K → take high-conf)

Before starting: seeds performance & compatibility categories via
keyword filtering (the classifier has near-zero recall on these).

Usage:
    python3 scripts/progressive_labeling.py --round 1
    python3 scripts/progressive_labeling.py --round 2
    python3 scripts/progressive_labeling.py --round all
    python3 scripts/progressive_labeling.py --round 1 --conf-threshold 0.85

Output per round:
    data/processed/round_N/
        labeled_NNk.json        — all labeled data (cumulative)
        new_labels_NNk.json     — only new labels from this round
        stats_NNk.json          — distribution, confidence, metrics
        model_round_N/          — retrained model checkpoint
"""

import json
import csv
import sys
import random
import argparse
import shutil
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1.classifier import ReviewClassifier, LABELS


# ══════════════════════════════════════════════════════════════════════════════
# Keyword seeding for underrepresented categories
# ══════════════════════════════════════════════════════════════════════════════

PERFORMANCE_KEYWORDS = [
    "slow", "lag", "laggy", "lagging", "battery", "drain", "draining",
    "heat", "heats", "heating", "hot", "overheat", "overheating",
    "memory", "ram", "freeze", "freezing", "hung", "loading",
    "takes forever", "takes long", "takes too long", "so slow",
    "super slow", "very slow", "extremely slow", "painfully slow",
    "sluggish", "unresponsive", "cpu", "resource", "heavy",
    "power consumption", "battery life", "eats battery",
    "kills battery", "battery hog", "performance",
]

COMPATIBILITY_KEYWORDS = [
    "doesn't work on", "not working on", "doesn't support",
    "not compatible", "incompatible", "won't work on",
    "blank screen on", "crashes on my", "broken on",
    "android 14", "android 15", "android 13", "android 12",
    "ios 17", "ios 18", "ios 16",
    "samsung", "pixel", "xiaomi", "oneplus", "huawei", "oppo",
    "ipad", "tablet", "iphone 15", "iphone 14", "iphone 13",
    "galaxy s2", "galaxy s23", "galaxy s24", "galaxy a",
    "my device", "my phone", "specific device", "older phone",
    "not optimized", "screen size", "resolution",
]


def keyword_filter(reviews, keywords):
    """Find reviews matching any keyword (case-insensitive)."""
    results = []
    for r in reviews:
        text_lower = r["text"].lower()
        if any(kw in text_lower for kw in keywords):
            results.append(r)
    return results


def seed_underrepresented(rrgen_all, existing_texts):
    """Find real performance & compatibility reviews via keywords."""
    # Filter out reviews already in training set
    pool = [r for r in rrgen_all if r["text"] not in existing_texts]

    perf_candidates = keyword_filter(pool, PERFORMANCE_KEYWORDS)
    compat_candidates = keyword_filter(pool, COMPATIBILITY_KEYWORDS)

    # Remove overlap (if a review matches both, assign to the better fit)
    compat_texts = {r["text"] for r in compat_candidates}
    perf_only = [r for r in perf_candidates if r["text"] not in compat_texts]

    print(f"\n  Keyword seeding:")
    print(f"    Performance candidates: {len(perf_only)}")
    print(f"    Compatibility candidates: {len(compat_candidates)}")

    # Sample up to 150 performance + 100 compatibility
    perf_sample = random.sample(perf_only, min(150, len(perf_only)))
    compat_sample = random.sample(compat_candidates, min(100, len(compat_candidates)))

    seeded = []
    for r in perf_sample:
        seeded.append({
            "text": r["text"],
            "rating": r["rating"],
            "app_id": r.get("app_id", ""),
            "timestamp": r.get("timestamp", ""),
            "labels": ["performance"],
            "confidence": 0.0,  # keyword-assigned, not model-predicted
            "source": "keyword_seed",
            "needs_verification": True,
        })
    for r in compat_sample:
        seeded.append({
            "text": r["text"],
            "rating": r["rating"],
            "app_id": r.get("app_id", ""),
            "timestamp": r.get("timestamp", ""),
            "labels": ["compatibility"],
            "confidence": 0.0,
            "source": "keyword_seed",
            "needs_verification": True,
        })

    print(f"    Seeded: {len(perf_sample)} performance + {len(compat_sample)} compatibility")
    return seeded


# ══════════════════════════════════════════════════════════════════════════════
# Core labeling function
# ══════════════════════════════════════════════════════════════════════════════

def label_batch(classifier, reviews, conf_threshold=0.80):
    """Predict labels for a batch, return high-confidence ones."""
    texts = [r["text"] for r in reviews]
    all_results = []

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        predictions = classifier.predict(batch)
        all_results.extend(predictions)
        if (i // batch_size) % 25 == 0 and i > 0:
            print(f"    Predicted {min(i + batch_size, len(texts)):,}/{len(texts):,}")

    high_conf = []
    low_conf = []

    for review, (pred_labels, confidences) in zip(reviews, all_results):
        max_conf = max(confidences.values())
        record = {
            "text": review["text"],
            "rating": review["rating"],
            "app_id": review.get("app_id", ""),
            "timestamp": review.get("timestamp", ""),
            "labels": pred_labels,
            "confidence": round(max_conf, 4),
            "all_confidences": {k: round(v, 4) for k, v in confidences.items()},
            "source": "model_predicted",
            "needs_verification": max_conf < conf_threshold,
        }
        if max_conf >= conf_threshold:
            high_conf.append(record)
        else:
            low_conf.append(record)

    return high_conf, low_conf


# ══════════════════════════════════════════════════════════════════════════════
# Training function
# ══════════════════════════════════════════════════════════════════════════════

def retrain_classifier(labeled_data, round_num, output_dir, epochs=3):
    """Retrain the classifier on the cumulative labeled dataset."""
    print(f"\n  Retraining classifier (Round {round_num}, {len(labeled_data):,} samples, {epochs} epochs)...")

    texts = [r["text"] for r in labeled_data]
    # Convert labels to multi-hot (floats required for BCEWithLogitsLoss)
    label_vectors = []
    for r in labeled_data:
        vec = [0.0] * len(LABELS)
        for lbl in r["labels"]:
            if lbl in LABELS:
                vec[LABELS.index(lbl)] = 1.0
        # If no valid label, mark as "other"
        if sum(vec) == 0:
            vec[LABELS.index("other")] = 1.0
        label_vectors.append(vec)

    # 90/10 split for training (we want max training data in self-training)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_vectors, test_size=0.1, random_state=42
    )

    model_path = str(output_dir / f"model_round_{round_num}")

    classifier = ReviewClassifier(model_name_or_path="roberta-base")
    metrics = classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=model_path,
        epochs=epochs,
        batch_size=16,
    )

    # Evaluate
    predictions = classifier.predict(val_texts)
    y_pred = []
    for pred_labels, _ in predictions:
        vec = [0] * len(LABELS)
        for lbl in pred_labels:
            if lbl in LABELS:
                vec[LABELS.index(lbl)] = 1
        if sum(vec) == 0:
            vec[LABELS.index("other")] = 1
        y_pred.append(vec)

    y_pred_arr = np.array(y_pred)
    y_true_arr = np.array(val_labels)

    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    f1_micro = f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)

    print(f"  Validation F1 macro: {f1_macro:.4f}")
    print(f"  Validation F1 micro: {f1_micro:.4f}")
    print(f"  Model saved: {model_path}")

    return classifier, model_path, {"f1_macro": f1_macro, "f1_micro": f1_micro}


# ══════════════════════════════════════════════════════════════════════════════
# Round execution
# ══════════════════════════════════════════════════════════════════════════════

def get_round_config(round_num):
    """Get batch size and cumulative target for each round."""
    if round_num == 1:
        return {"batch_size": 10000, "cumulative_target": 10000, "label": "10k"}
    elif round_num == 2:
        return {"batch_size": 10000, "cumulative_target": 20000, "label": "20k"}
    elif round_num == 3:
        return {"batch_size": 10000, "cumulative_target": 30000, "label": "30k"}
    else:
        # Round 4+: add 30K each
        n = 30000 + (round_num - 3) * 30000
        return {"batch_size": 30000, "cumulative_target": n, "label": f"{n // 1000}k"}


def load_existing_labels(round_num):
    """Load cumulative labels from previous round."""
    if round_num <= 1:
        # Load original training data (synthetic + MAALEJ)
        base_path = Path("data/raw/sample_reviews.json")
        if base_path.exists():
            with open(base_path) as f:
                synthetic = json.load(f)
            # Convert to standard format
            labeled = []
            for r in synthetic:
                labeled.append({
                    "text": r["text"],
                    "rating": r["rating"],
                    "app_id": r.get("app_id", ""),
                    "timestamp": r.get("timestamp", ""),
                    "labels": r["labels"],
                    "confidence": 1.0,
                    "source": r.get("source", "synthetic"),
                    "needs_verification": False,
                })
            return labeled
        return []
    else:
        prev_config = get_round_config(round_num - 1)
        prev_path = Path(f"data/processed/round_{round_num - 1}/labeled_{prev_config['label']}.json")
        if prev_path.exists():
            with open(prev_path) as f:
                return json.load(f)
        else:
            print(f"  Warning: previous round data not found at {prev_path}")
            print(f"  Run round {round_num - 1} first.")
            sys.exit(1)


def run_round(round_num, conf_threshold, retrain_epochs):
    """Execute one round of progressive labeling."""
    config = get_round_config(round_num)
    print("=" * 70)
    print(f"ROUND {round_num}: Target {config['label']} cumulative labeled reviews")
    print(f"  Batch size: {config['batch_size']:,}")
    print(f"  Confidence threshold: {conf_threshold}")
    print("=" * 70)

    # ── Setup output dir ─────────────────────────────────────────────────
    output_dir = Path(f"data/processed/round_{round_num}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load existing labels ─────────────────────────────────────────────
    existing = load_existing_labels(round_num)
    existing_texts = {r["text"] for r in existing}
    print(f"\n  Existing labeled data: {len(existing):,}")

    # ── Load RRGen pool ──────────────────────────────────────────────────
    print("  Loading RRGen pool...")
    with open("data/raw/rrgen/rrgen_reviews.json") as f:
        rrgen_all = json.load(f)

    # Filter: reasonable length + not already labeled
    pool = [r for r in rrgen_all
            if 10 <= len(r["text"]) <= 500 and r["text"] not in existing_texts]
    print(f"  Available unlabeled pool: {len(pool):,}")

    # ── Round 1: Seed performance & compatibility ────────────────────────
    seeded = []
    if round_num == 1:
        seeded = seed_underrepresented(rrgen_all, existing_texts)
        # Remove seeded texts from pool
        seeded_texts = {r["text"] for r in seeded}
        pool = [r for r in pool if r["text"] not in seeded_texts]

    # ── Sample batch from pool ───────────────────────────────────────────
    batch_size = config["batch_size"]
    if len(pool) < batch_size:
        print(f"  Warning: only {len(pool)} available, using all")
        batch = pool
    else:
        batch = random.sample(pool, batch_size)
    print(f"  Sampled batch: {len(batch):,}")

    # ── Load or retrain classifier ───────────────────────────────────────
    if round_num == 1:
        # Use existing trained model
        model_path = "models/stage1_classifier"
        print(f"\n  Loading pre-trained classifier: {model_path}")
        classifier = ReviewClassifier.load(model_path)
    else:
        # Retrain on cumulative labeled data
        classifier, model_path, train_metrics = retrain_classifier(
            existing, round_num, output_dir, epochs=retrain_epochs
        )

    # ── Predict labels ───────────────────────────────────────────────────
    print(f"\n  Predicting labels on {len(batch):,} reviews...")
    high_conf, low_conf = label_batch(classifier, batch, conf_threshold)

    print(f"\n  Results:")
    print(f"    High confidence (>={conf_threshold}): {len(high_conf):,}")
    print(f"    Low confidence  (<{conf_threshold}):  {len(low_conf):,}")
    print(f"    Acceptance rate: {len(high_conf)/len(batch)*100:.1f}%")

    # ── Combine: existing + seeded + high-confidence new ─────────────────
    new_labels = seeded + high_conf
    cumulative = existing + new_labels

    print(f"\n  Cumulative labeled data: {len(cumulative):,}")

    # ── Distribution ─────────────────────────────────────────────────────
    label_counts = Counter()
    source_counts = Counter()
    for r in cumulative:
        for lbl in r["labels"]:
            label_counts[lbl] += 1
        source_counts[r.get("source", "unknown")] += 1

    print(f"\n  Label distribution (cumulative):")
    for label in LABELS:
        count = label_counts[label]
        pct = count / len(cumulative) * 100
        print(f"    {label:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n  Source distribution:")
    for src, count in source_counts.most_common():
        print(f"    {src:20s}: {count:6d}")

    # ── Save outputs ─────────────────────────────────────────────────────
    # Cumulative labeled data
    cum_path = output_dir / f"labeled_{config['label']}.json"
    with open(cum_path, "w") as f:
        json.dump(cumulative, f, indent=2)
    print(f"\n  Saved cumulative: {cum_path}")

    # New labels only (for review/verification)
    new_path = output_dir / f"new_labels_{config['label']}.json"
    with open(new_path, "w") as f:
        json.dump(new_labels, f, indent=2)
    print(f"  Saved new labels: {new_path}")

    # Low-confidence (for optional manual review)
    low_path = output_dir / f"low_confidence_{config['label']}.json"
    with open(low_path, "w") as f:
        json.dump(low_conf, f, indent=2)
    print(f"  Saved low-conf:   {low_path}")

    # CSV for easy viewing
    csv_path = output_dir / f"new_labels_{config['label']}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "rating", "predicted_label", "confidence",
                         "source", "needs_verification"])
        for i, r in enumerate(new_labels, 1):
            writer.writerow([
                i, r["text"], r["rating"],
                "|".join(r["labels"]), r.get("confidence", 0),
                r.get("source", ""), r.get("needs_verification", True),
            ])
    print(f"  Saved CSV:        {csv_path}")

    # Stats
    confs = [r.get("confidence", 0) for r in new_labels if r.get("confidence", 0) > 0]
    stats = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "conf_threshold": conf_threshold,
        "batch_sampled": len(batch),
        "seeded": len(seeded),
        "high_confidence_accepted": len(high_conf),
        "low_confidence_rejected": len(low_conf),
        "acceptance_rate": round(len(high_conf) / len(batch) * 100, 2) if batch else 0,
        "new_labels_total": len(new_labels),
        "cumulative_total": len(cumulative),
        "label_distribution": {l: label_counts[l] for l in LABELS},
        "source_distribution": dict(source_counts),
        "confidence_stats": {
            "mean": round(float(np.mean(confs)), 4) if confs else 0,
            "median": round(float(np.median(confs)), 4) if confs else 0,
            "min": round(float(np.min(confs)), 4) if confs else 0,
            "max": round(float(np.max(confs)), 4) if confs else 0,
        } if confs else {},
        "model_path": model_path if round_num > 1 else "models/stage1_classifier",
    }
    stats_path = output_dir / f"stats_{config['label']}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats:      {stats_path}")

    print(f"\n{'=' * 70}")
    print(f"ROUND {round_num} COMPLETE")
    print(f"  New labels added: {len(new_labels):,}")
    print(f"  Cumulative total: {len(cumulative):,}")
    if round_num < 3:
        next_config = get_round_config(round_num + 1)
        print(f"  Next: python3 scripts/progressive_labeling.py --round {round_num + 1}")
    print(f"{'=' * 70}\n")

    return cumulative


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Progressive semi-supervised labeling")
    parser.add_argument("--round", type=str, default="1",
                        help="Round number (1, 2, 3, ...) or 'all' for sequential execution")
    parser.add_argument("--conf-threshold", type=float, default=0.80,
                        help="Minimum confidence to auto-accept a label (default: 0.80)")
    parser.add_argument("--retrain-epochs", type=int, default=3,
                        help="Epochs for retraining (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.round == "all":
        # Run rounds 1 → 3 sequentially
        for r in range(1, 4):
            run_round(r, args.conf_threshold, args.retrain_epochs)
    else:
        round_num = int(args.round)
        run_round(round_num, args.conf_threshold, args.retrain_epochs)


if __name__ == "__main__":
    main()
