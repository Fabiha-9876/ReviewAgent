"""
RoBERTa-Based Labeling of Keyword-Filtered RRGen Reviews
=========================================================

Uses the trained RoBERTa classifier (F1: 0.7992) to label keyword-filtered
RRGen reviews, then exports for human verification (Y/N).

100% free, runs locally, no API needed.

Usage:
    # Test with 100 reviews
    python3 scripts/roberta_label_rrgen.py --max-reviews 100

    # Run on all keyword-filtered candidates
    python3 scripts/roberta_label_rrgen.py

    # Then generate verification sheet
    python3 scripts/generate_verification_sheet.py

Output:
    data/processed/rrgen_llm_labeled/
        llm_labeled_all.json
        llm_labeled_stats.json
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

# ======================================================================
# Keyword lists (same as llm_label_rrgen.py)
# ======================================================================

CATEGORY_KEYWORDS = {
    "performance": [
        "slow", "lag", "laggy", "lagging", "battery", "drain", "draining",
        "heat", "heats", "heating", "hot", "overheat", "overheating",
        "memory", "ram", "freeze", "freezing", "hung", "loading",
        "takes forever", "takes long", "takes too long", "so slow",
        "super slow", "very slow", "extremely slow", "painfully slow",
        "sluggish", "unresponsive", "cpu", "resource", "heavy",
        "power consumption", "battery life", "eats battery",
        "kills battery", "battery hog", "performance",
    ],
    "compatibility": [
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
    ],
    "usability": [
        "confusing", "hard to use", "hard to find", "not intuitive",
        "complicated", "unintuitive", "user friendly", "user-friendly",
        "ux", "ui", "interface", "layout", "navigation", "menu",
        "can't figure out", "difficult to", "too many steps",
        "cluttered", "messy", "overwhelming", "hidden", "buried",
    ],
    "feature_request": [
        "i wish", "would be nice", "please add", "should have",
        "need a", "needs a", "missing feature", "add option",
        "it would be great", "why can't", "why doesn't", "if only",
        "feature request", "suggestion", "want to be able",
        "dark mode", "widget", "offline", "export", "customize",
    ],
    "bug_report": [
        "crash", "crashes", "crashing", "bug", "error", "glitch",
        "broken", "doesn't work", "not working", "won't open",
        "force close", "black screen", "white screen", "stuck",
        "keeps closing", "shuts down", "can't login", "can't log in",
        "failed", "failure", "malfunction",
    ],
}


# ======================================================================
# Step 1: Keyword filtering
# ======================================================================

def keyword_filter_all(reviews, min_length=10, max_length=500):
    """Filter RRGen reviews by keywords across all categories."""
    candidates = []
    seen_texts = set()

    for r in reviews:
        text = r["text"]
        if len(text) < min_length or len(text) > max_length:
            continue
        if text in seen_texts:
            continue

        text_lower = text.lower()
        matched_categories = []

        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                matched_categories.append(category)

        if matched_categories:
            seen_texts.add(text)
            candidates.append({
                "text": text,
                "rating": r.get("rating", 0),
                "app_id": r.get("app_id", ""),
                "timestamp": r.get("timestamp", ""),
                "original_response": r.get("response", ""),
                "keyword_matched_categories": matched_categories,
            })

    return candidates


# ======================================================================
# Step 2: RoBERTa labeling
# ======================================================================

def label_batch_roberta(classifier, candidates, max_reviews=None, batch_size=64):
    """Label candidates using trained RoBERTa classifier."""
    if max_reviews:
        candidates = candidates[:max_reviews]

    total = len(candidates)
    texts = [c["text"] for c in candidates]
    all_results = []

    print(f"\n  Labeling {total:,} reviews with RoBERTa classifier...")

    start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        predictions = classifier.predict(batch)
        all_results.extend(predictions)

        done = min(i + batch_size, len(texts))
        if done % (batch_size * 10) == 0 or done == len(texts):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:,}/{total:,} done ({rate:.0f} reviews/sec)")

    # Attach predictions to candidates
    results = []
    for candidate, (pred_labels, confidences) in zip(candidates, all_results):
        primary_label = pred_labels[0] if pred_labels else "other"
        max_conf = max(confidences.values()) if confidences else 0

        # Use keyword match as reasoning
        keyword_cats = candidate.get("keyword_matched_categories", [])
        reasoning = f"RoBERTa prediction (keyword match: {', '.join(keyword_cats)})"

        candidate["llm_label"] = primary_label
        candidate["llm_confidence"] = round(max_conf, 4)
        candidate["llm_reasoning"] = reasoning
        candidate["llm_error"] = False
        candidate["all_confidences"] = {k: round(v, 4) for k, v in confidences.items()}
        candidate["needs_hitl"] = classifier.needs_hitl(confidences)
        results.append(candidate)

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s ({len(results)/elapsed:.0f} reviews/sec)")

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="RoBERTa-based labeling of keyword-filtered RRGen reviews")
    parser.add_argument("--model-path", default="models/stage1_classifier",
                        help="Path to trained RoBERTa classifier")
    parser.add_argument("--max-reviews", type=int, default=None,
                        help="Max reviews to label (default: all candidates)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path("data/processed/rrgen_llm_labeled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RRGen
    print("Loading RRGen dataset...")
    with open("data/raw/rrgen/rrgen_reviews.json") as f:
        rrgen_all = json.load(f)
    print(f"  Total RRGen reviews: {len(rrgen_all):,}")

    # Step 1: Keyword filter
    print("\nStep 1: Keyword filtering...")
    candidates = keyword_filter_all(rrgen_all)
    print(f"  Keyword-matched candidates: {len(candidates):,}")

    cat_counts = Counter()
    for c in candidates:
        for cat in c["keyword_matched_categories"]:
            cat_counts[cat] += 1
    for cat, count in cat_counts.most_common():
        print(f"    {cat:20s}: {count:,}")

    # Load classifier
    print(f"\nLoading RoBERTa classifier from {args.model_path}...")
    classifier = ReviewClassifier.load(args.model_path)
    print(f"  Device: {classifier.device}")

    # Step 2: RoBERTa labeling
    print("\nStep 2: RoBERTa labeling...")
    results = label_batch_roberta(
        classifier, candidates,
        max_reviews=args.max_reviews,
        batch_size=args.batch_size,
    )

    # Statistics
    label_counts = Counter(r["llm_label"] for r in results)
    hitl_count = sum(1 for r in results if r.get("needs_hitl"))
    confidences = [r["llm_confidence"] for r in results]

    print("\n" + "=" * 60)
    print("ROBERTA LABELING RESULTS")
    print("=" * 60)
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = count / len(results) * 100 if results else 0
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  Total labeled:     {len(results):,}")
    print(f"  Needs HITL:        {hitl_count:,} ({hitl_count/len(results)*100:.1f}%)")
    print(f"  High confidence:   {len(results)-hitl_count:,} ({(len(results)-hitl_count)/len(results)*100:.1f}%)")

    print(f"\n  Confidence stats:")
    print(f"    Mean:   {np.mean(confidences):.4f}")
    print(f"    Median: {np.median(confidences):.4f}")
    print(f"    >=0.8:  {sum(1 for c in confidences if c >= 0.8):,}")
    print(f"    >=0.9:  {sum(1 for c in confidences if c >= 0.9):,}")
    print(f"    <0.5:   {sum(1 for c in confidences if c < 0.5):,}")

    # Save output (same format as llm_label_rrgen.py for compatibility)
    out_path = output_dir / "llm_labeled_all.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    stats = {
        "timestamp": datetime.now().isoformat(),
        "provider": "roberta_local",
        "model": args.model_path,
        "total_rrgen": len(rrgen_all),
        "keyword_filtered": len(candidates),
        "labeled": len(results),
        "needs_hitl": hitl_count,
        "label_distribution": {l: label_counts.get(l, 0) for l in LABELS},
        "confidence_mean": round(float(np.mean(confidences)), 4),
        "confidence_median": round(float(np.median(confidences)), 4),
    }
    stats_path = output_dir / "llm_labeled_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    print(f"\nDone! Next step:")
    print(f"  python3 scripts/generate_verification_sheet.py")


if __name__ == "__main__":
    main()
