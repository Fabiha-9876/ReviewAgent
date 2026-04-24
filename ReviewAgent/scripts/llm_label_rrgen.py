"""
LLM-Based Labeling of Keyword-Filtered RRGen Reviews (Gemini Free Tier)
========================================================================

Step 1: Keyword-filter RRGen 310K -> ~5K-10K candidates
Step 2: Google Gemini (free) auto-labels each candidate

Usage:
    # Test with 20 reviews first
    python3 scripts/llm_label_rrgen.py --api-key YOUR_KEY --max-reviews 20

    # Run on all keyword-matched candidates
    python3 scripts/llm_label_rrgen.py --api-key YOUR_KEY

    # Resume interrupted run
    python3 scripts/llm_label_rrgen.py --api-key YOUR_KEY --resume

Output:
    data/processed/rrgen_llm_labeled/
        llm_labeled_all.json
        llm_labeled_stats.json
        llm_labeled_checkpoint.json
"""

import json
import os
import sys
import random
import argparse
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1.classifier import LABELS

# ======================================================================
# Keyword lists
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
# Step 2: Gemini labeling
# ======================================================================

CLASSIFICATION_PROMPT = """You are an expert app review classifier. Classify the given app store review into exactly ONE primary category.

Categories:
- bug_report: Reports of crashes, errors, broken features, malfunctions
- feature_request: Requests for new features or improvements to existing ones
- performance: Complaints about speed, battery drain, memory usage, lag, loading times
- usability: Complaints about confusing UI, hard-to-use features, poor navigation, bad design
- compatibility: Issues specific to a device, OS version, or screen size
- praise: Positive feedback, compliments, high satisfaction
- other: Doesn't fit any above category (general comments, questions, off-topic)

Rules:
1. Choose the SINGLE most fitting category
2. If a review mentions both a bug AND a device, choose "compatibility" only if the issue is device-specific; otherwise choose "bug_report"
3. "slow" or "lag" = performance (not bug_report)
4. "crash on my Samsung" = compatibility (device-specific)
5. "crash" (no device mentioned) = bug_report
6. Rate your confidence 0.0-1.0

Respond ONLY with this JSON format (no other text):
{"label": "category_name", "confidence": 0.85, "reasoning": "one short sentence"}"""


def label_single_review(client, model_name, text, rating):
    """Label a single review using Gemini."""
    user_prompt = CLASSIFICATION_PROMPT + '\n\nReview (rating: ' + str(rating) + '/5): "' + text + '"'
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config={"temperature": 0.1, "max_output_tokens": 150},
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        result = json.loads(raw)
        label = result.get("label", "other")
        if label not in LABELS:
            label = "other"

        return {
            "label": label,
            "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        return {
            "label": "other",
            "confidence": 0.0,
            "reasoning": "LLM error: " + str(e)[:100],
            "error": True,
        }


def label_batch(client, model_name, candidates, max_reviews=None, save_every=50, output_dir=None):
    """Label candidates with Gemini, respecting free tier rate limits."""
    if max_reviews:
        candidates = candidates[:max_reviews]

    total = len(candidates)
    results = []
    errors = 0

    print(f"\n  Labeling {total:,} reviews with Gemini...")
    print(f"  Rate limit: ~15 requests/min (free tier)")

    start_time = time.time()

    for i, review in enumerate(candidates):
        result = label_single_review(client, model_name, review["text"], review["rating"])

        review["llm_label"] = result["label"]
        review["llm_confidence"] = result["confidence"]
        review["llm_reasoning"] = result["reasoning"]
        review["llm_error"] = result.get("error", False)
        results.append(review)

        if result.get("error"):
            errors += 1
            if "429" in result["reasoning"] or "quota" in result["reasoning"].lower():
                print(f"    Rate limited at review {i+1}, waiting 60s...")
                time.sleep(60)
            else:
                time.sleep(2)
        else:
            # Gemini free tier: 15 RPM -> sleep ~4.5s between requests
            time.sleep(4.5)

        # Progress update every 25 reviews
        if (i + 1) % 25 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta_min = (total - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1:,}/{total:,} done ({rate:.0f}/min, {errors} errors, ETA: {eta_min:.0f} min)")

        # Save checkpoint periodically
        if output_dir and (i + 1) % save_every == 0:
            save_checkpoint(results, output_dir)

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed/60:.1f} min")
    print(f"  Errors: {errors}")

    return results


# ======================================================================
# Checkpoint support
# ======================================================================

def save_checkpoint(results, output_dir):
    cp_path = output_dir / "llm_labeled_checkpoint.json"
    with open(cp_path, "w") as f:
        json.dump(results, f, indent=2)


def load_checkpoint(output_dir):
    cp_path = output_dir / "llm_labeled_checkpoint.json"
    if cp_path.exists():
        with open(cp_path) as f:
            return json.load(f)
    return []


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Gemini-based labeling of keyword-filtered RRGen reviews")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Gemini model (default: gemini-2.0-flash)")
    parser.add_argument("--max-reviews", type=int, default=None,
                        help="Max reviews to label (default: all candidates)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Configure Gemini (new SDK)
    from google import genai

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: provide --api-key or set GEMINI_API_KEY env variable")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print(f"Using model: {args.model}")

    output_dir = Path("data/processed/rrgen_llm_labeled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RRGen
    print("\nLoading RRGen dataset...")
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

    # Resume support
    already_labeled = []
    if args.resume:
        already_labeled = load_checkpoint(output_dir)
        if already_labeled:
            labeled_texts = {r["text"] for r in already_labeled}
            candidates = [c for c in candidates if c["text"] not in labeled_texts]
            print(f"\n  Resuming: {len(already_labeled):,} already done, {len(candidates):,} remaining")

    # Step 2: Gemini labeling
    print(f"\nStep 2: Gemini labeling...")
    results = label_batch(
        client, args.model, candidates,
        max_reviews=args.max_reviews,
        save_every=50,
        output_dir=output_dir,
    )

    all_results = already_labeled + results
    save_checkpoint(all_results, output_dir)

    # Statistics
    label_counts = Counter(r["llm_label"] for r in all_results)
    error_count = sum(1 for r in all_results if r.get("llm_error"))
    confidences = [r["llm_confidence"] for r in all_results if not r.get("llm_error")]

    print("\n" + "=" * 60)
    print("LLM LABELING RESULTS")
    print("=" * 60)
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = count / len(all_results) * 100 if all_results else 0
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  Total labeled:     {len(all_results):,}")
    print(f"  Errors:            {error_count:,}")

    if confidences:
        import numpy as np
        print(f"\n  Confidence stats:")
        print(f"    Mean:   {np.mean(confidences):.4f}")
        print(f"    Median: {np.median(confidences):.4f}")
        print(f"    >=0.8:  {sum(1 for c in confidences if c >= 0.8):,}")
        print(f"    >=0.9:  {sum(1 for c in confidences if c >= 0.9):,}")
        print(f"    <0.5:   {sum(1 for c in confidences if c < 0.5):,}")

    # Save final output
    out_path = output_dir / "llm_labeled_all.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    stats = {
        "timestamp": datetime.now().isoformat(),
        "provider": "gemini",
        "model": args.model,
        "total_rrgen": len(rrgen_all),
        "keyword_filtered": len(candidates) + len(already_labeled),
        "llm_labeled": len(all_results),
        "errors": error_count,
        "label_distribution": {l: label_counts.get(l, 0) for l in LABELS},
        "confidence_mean": round(float(sum(confidences) / len(confidences)), 4) if confidences else 0,
    }
    stats_path = output_dir / "llm_labeled_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    print(f"\nDone! Next step: python3 scripts/generate_verification_sheet.py")


if __name__ == "__main__":
    main()
