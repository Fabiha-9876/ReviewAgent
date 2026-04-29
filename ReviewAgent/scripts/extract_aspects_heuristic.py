"""
Phase 2 of free-tier Stage 2: heuristic aspect extraction.

For each review, extract candidate aspects using:
  1. spaCy noun-phrase chunking (subject/object NPs).
  2. KeyBERT keyphrases ranked by similarity to the review.
  3. Pattern-based extraction ("the X is broken", "X doesn't work").

Output:
    data/processed/aspects_heuristic/aspects_per_review.json
        {idx: ["login button", "battery life", ...], ...}
    data/processed/aspects_heuristic/aspect_frequency.json
        Counter of all aspects across the dataset.

This is meant to be a fast, free baseline. Quality is medium — it catches
obvious aspects (entities/features mentioned by name) but misses semantic
nuance. For paper purposes, this becomes the "heuristic baseline" condition;
Phase 3 (local LLM) provides the higher-quality version on a sample for
gold-standard validation.

Usage:
    python3 scripts/extract_aspects_heuristic.py \
        --input data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json \
        --label-field v5_label \
        --out-dir data/processed/aspects_heuristic
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS

# Patterns that strongly indicate aspect-bearing phrases
ASPECT_PATTERNS = [
    # "the X is broken/slow/etc"
    r"the\s+([a-z][a-z\s]{1,30}?)\s+(?:is|are|was|were|gets|got|keeps|won't|can't|cannot|doesn'?t)",
    # "X doesn't work"
    r"([a-z][a-z\s]{1,30}?)\s+(?:doesn'?t|don'?t|won'?t|can'?t|cannot|fails to|stopped)\s+(?:work|load|open|sync|save|load|update)",
    # "love/hate/like the X"
    r"(?:love|hate|like|enjoy|dislike|miss|need)\s+the\s+([a-z][a-z\s]{1,30}?)\s+(?:feature|button|option|setting|screen|page|tab)",
    # "after [verb]ing the X"
    r"(?:after|when)\s+(?:opening|using|tapping|clicking|pressing)\s+(?:the\s+)?([a-z][a-z\s]{1,30}?)(?:\s|,|\.)",
]

# Common app review aspects we want to catch (used as a vocabulary filter)
COMMON_ASPECTS = {
    "login", "signup", "sign in", "sign up", "registration", "password",
    "battery", "battery life", "power consumption",
    "loading", "loading time", "load time", "speed", "performance", "lag",
    "crash", "crashes", "freezing", "freeze", "force close", "stuck",
    "ad", "ads", "advertisement", "advertisements",
    "notification", "notifications", "alerts", "push notification",
    "search", "search bar", "filter", "filters",
    "ui", "interface", "design", "layout", "theme", "dark mode", "light mode",
    "menu", "navigation", "button", "buttons", "tab", "tabs", "screen",
    "feature", "feature request", "functionality",
    "update", "updates", "upgrade", "version", "new version",
    "subscription", "premium", "payment", "purchase", "in-app purchase",
    "video", "videos", "audio", "music", "playback", "player",
    "image", "images", "photo", "photos", "picture", "pictures",
    "chat", "message", "messages", "messaging", "conversation",
    "feed", "timeline", "stories", "post", "posts",
    "camera", "gallery", "filter", "edit", "editing",
    "sync", "syncing", "synchronization", "cloud", "backup",
    "offline", "online", "wifi", "data", "internet connection",
    "language", "translation", "support", "help", "customer service",
    "tablet", "phone", "device", "compatibility",
    "ringtone", "sound", "volume", "vibration", "alert sound",
    "widget", "widgets", "shortcut", "icon",
    "account", "profile", "settings", "preferences", "privacy",
    "map", "maps", "location", "gps", "navigation",
}


def normalize_aspect(text: str) -> str:
    """Lowercase, strip, remove leading articles/possessives."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^(the|a|an|my|your|this|that)\s+", "", t)
    return t.strip()


def extract_aspects(text: str, nlp, kw_model=None, top_k_keybert: int = 3) -> list[str]:
    """Combine spaCy noun-phrase + pattern + KeyBERT extraction."""
    if not text or len(text) < 10:
        return []

    aspects: set[str] = set()
    text_lower = text.lower()

    # 1. Pattern-based
    for pattern in ASPECT_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            asp = normalize_aspect(m.group(1))
            if asp and 2 <= len(asp) <= 35:
                aspects.add(asp)

    # 2. Common-aspect vocabulary check
    for asp in COMMON_ASPECTS:
        if asp in text_lower:
            aspects.add(asp)

    # 3. spaCy noun-phrase chunking — pick NPs containing a noun
    doc = nlp(text)
    for np in doc.noun_chunks:
        # Skip pronouns/determiner-only chunks
        if not any(tok.pos_ == "NOUN" for tok in np):
            continue
        asp = normalize_aspect(np.text)
        # Filter to plausible aspects
        if asp and 2 <= len(asp) <= 35 and not asp.isdigit():
            aspects.add(asp)

    # 4. KeyBERT (optional, slower)
    if kw_model:
        try:
            kbs = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3),
                                             stop_words="english", top_n=top_k_keybert)
            for kw, _ in kbs:
                asp = normalize_aspect(kw)
                if asp and 2 <= len(asp) <= 35:
                    aspects.add(asp)
        except Exception:
            pass

    # Filter pronouns/empty/junk
    junk = {"i", "you", "we", "they", "it", "this", "that", "these", "those",
            "app", "app .", "the app", "everything", "nothing", "something",
            "anything", "thing", "things"}
    aspects = {a for a in aspects if a not in junk and not a.startswith("<")}

    return sorted(aspects)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"))
    ap.add_argument("--label-field", default="v5_label")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/aspects_heuristic"))
    ap.add_argument("--use-keybert", action="store_true",
                    help="Enable KeyBERT extraction (slower but richer aspects).")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--actionable-only", action="store_true", default=True,
                    help="Only extract aspects for bug/feature/perf/usability/compat reviews.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Allow fallback if V5 relabel hasn't finished
    if not args.input.exists():
        fb = Path("data/processed/rrgen_corrected_v2/rrgen_corrected_v2.json")
        print(f"Input not found, falling back to {fb}")
        args.input = fb
        if args.label_field == "v5_label":
            args.label_field = "final_label"

    print(f"Loading: {args.input}")
    with open(args.input) as f:
        rows = json.load(f)
    print(f"  {len(rows):,} rows")

    print("Loading spaCy en_core_web_sm")
    import spacy
    try:
        # Keep parser (required for noun_chunks); disable only NER for speed.
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        print("  model not installed — running: python3 -m spacy download en_core_web_sm")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    kw_model = None
    if args.use_keybert:
        print("Loading KeyBERT")
        from keybert import KeyBERT
        kw_model = KeyBERT("all-MiniLM-L6-v2")

    actionable = {"bug_report", "feature_request", "performance", "usability", "compatibility"}

    aspects_per_review = {}
    aspect_freq = Counter()
    aspects_per_class = defaultdict(Counter)
    n_processed = 0
    n_skipped = 0

    print(f"\nExtracting aspects (use_keybert={args.use_keybert})...")
    import time
    t0 = time.time()
    for i, r in enumerate(rows):
        if args.max_rows and n_processed >= args.max_rows:
            break
        if args.actionable_only and r.get(args.label_field) not in actionable:
            n_skipped += 1
            continue
        text = r["text"]
        aspects = extract_aspects(text, nlp, kw_model=kw_model)
        if aspects:
            aspects_per_review[i] = aspects
            for a in aspects:
                aspect_freq[a] += 1
                aspects_per_class[r.get(args.label_field, "unknown")][a] += 1
        n_processed += 1
        if n_processed % 5000 == 0:
            dt = time.time() - t0
            eta = dt / n_processed * (len(rows) - n_processed - n_skipped)
            print(f"  {n_processed:>7,} processed  |  {len(aspects_per_review):,} with aspects  |  "
                  f"elapsed {dt/60:.1f}min  ETA {eta/60:.1f}min", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  Processed: {n_processed:,}  Skipped: {n_skipped:,}")
    print(f"  Reviews with at least one aspect: {len(aspects_per_review):,}")

    # Save outputs
    with open(args.out_dir / "aspects_per_review.json", "w") as f:
        json.dump(aspects_per_review, f)
    with open(args.out_dir / "aspect_frequency.json", "w") as f:
        json.dump(aspect_freq.most_common(200), f, indent=2)
    with open(args.out_dir / "aspects_by_class.json", "w") as f:
        json.dump({k: dict(v.most_common(50)) for k, v in aspects_per_class.items()},
                  f, indent=2)

    # Summary print
    print(f"\nTop 30 aspects across dataset:")
    for asp, n in aspect_freq.most_common(30):
        print(f"  {n:>5,}  {asp}")
    print(f"\nTop 5 aspects per class:")
    for cls in actionable:
        if cls in aspects_per_class:
            top5 = aspects_per_class[cls].most_common(5)
            print(f"  {cls:20s}  {[a for a,_ in top5]}")

    print(f"\nOutputs: {args.out_dir}/")


if __name__ == "__main__":
    main()
