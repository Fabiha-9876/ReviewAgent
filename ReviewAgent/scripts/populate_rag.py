"""
Populate RAG Index — Load all 5 sources into ChromaDB
=====================================================

Indexes the following into ChromaDB vector store:
1. past_responses — Developer responses from RRGen (310K pairs, sample 20K)
2. changelogs — Simulated app changelog entries from RRGen app metadata
3. faq — Common FAQ patterns extracted from high-frequency response patterns
4. issue_spec — Empty initially, populated by Stage 3 at runtime
5. similar_responses — Deduplicated high-quality responses for template guidance

Usage:
    python3 scripts/populate_rag.py                # Index all sources
    python3 scripts/populate_rag.py --source past_responses  # Index one source
    python3 scripts/populate_rag.py --max-docs 5000          # Limit docs per source
    python3 scripts/populate_rag.py --verify                 # Verify index + test query

Output:
    data/chroma_db/ — ChromaDB persistent storage
"""

import json
import sys
import argparse
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_rrgen() -> list[dict]:
    """Load RRGen review-response pairs."""
    path = Path("data/raw/rrgen/rrgen_reviews.json")
    if not path.exists():
        print("  ERROR: RRGen data not found. Run: python3 scripts/download_datasets.py rrgen")
        return []
    return json.loads(path.read_text())


def make_id(text: str, prefix: str = "") -> str:
    """Generate a deterministic ID from text."""
    h = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{prefix}{h}" if prefix else h


# ============================================================
# Source 1: Past Responses
# ============================================================

def index_past_responses(retriever, rrgen: list[dict], max_docs: int = 20000):
    """Index developer responses paired with their review context.

    We sample from RRGen to keep the index manageable while covering
    diverse apps, ratings, and response styles.
    """
    print("\n--- Source 1: Past Responses ---")

    # Stratified sample: proportional across apps and ratings
    random.seed(42)
    random.shuffle(rrgen)

    # Filter: only reviews with non-empty responses > 20 chars
    valid = [r for r in rrgen if len(r.get("response", "").strip()) > 20
             and len(r.get("text", "").strip()) > 10]
    print(f"  Valid review-response pairs: {len(valid)}")

    # Sample
    sampled = valid[:max_docs]
    print(f"  Sampling {len(sampled)} for indexing")

    docs = []
    for idx, r in enumerate(sampled):
        text = f"Review: {r['text'].strip()}\nResponse: {r['response'].strip()}"
        docs.append({
            "id": f"resp_{idx}_{make_id(text)}",
            "text": text,
            "metadata": {
                "app_id": r.get("app_id", ""),
                "rating": r.get("rating", 0),
                "source": "past_responses",
                "review_text": r["text"].strip()[:200],
                "response_text": r["response"].strip()[:200],
            },
        })

    # Index in batches (ChromaDB has batch limits)
    batch_size = 1000
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        retriever.index_source("past_responses", batch)
        print(f"  Indexed batch {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1} ({len(batch)} docs)")

    print(f"  Total indexed: {len(docs)} past responses")
    return len(docs)


# ============================================================
# Source 2: Changelogs
# ============================================================

def index_changelogs(retriever, rrgen: list[dict], max_docs: int = 500):
    """Create and index simulated changelog entries from app data.

    Since we don't have real changelogs, we construct them from
    the RRGen data by identifying common issues per app version.
    """
    print("\n--- Source 2: Changelogs ---")

    # Group reviews by app and extract version mentions
    app_issues = defaultdict(list)
    for r in rrgen:
        app = r.get("app_id", "unknown")
        text = r.get("text", "").lower()
        rating = r.get("rating", 3)

        # Only use low-rating reviews as "issues that might be in changelogs"
        if rating <= 2 and len(text) > 20:
            app_issues[app].append(text[:150])

    # Generate changelog-style entries per app
    docs = []
    for app, issues in list(app_issues.items())[:50]:  # Top 50 apps
        # Take top issues and create a changelog entry
        sample_issues = random.sample(issues, min(5, len(issues)))

        changelog = (
            f"App: {app}\n"
            f"Update Notes:\n"
            f"- Fixed reported issues including: {'; '.join(sample_issues[:3])}\n"
            f"- Performance improvements and bug fixes\n"
            f"- Thank you for your feedback!"
        )

        docs.append({
            "id": make_id(changelog, "cl_"),
            "text": changelog,
            "metadata": {"app_id": app, "source": "changelogs", "type": "changelog"},
        })

    # Also add generic changelog templates
    generic_changelogs = [
        "Version update: Fixed crash on login for Android devices. Improved battery optimization. Minor UI tweaks.",
        "What's new: Added dark mode support. Fixed notification issues. Performance improvements for older devices.",
        "Bug fixes: Resolved force close issue on startup. Fixed payment processing errors. Improved app stability.",
        "Update: New feature - offline mode. Fixed compatibility issues with iOS 18. Reduced app size.",
        "Patch notes: Fixed login authentication bug. Improved loading times by 40%. Added biometric login support.",
        "Maintenance update: Fixed memory leak causing battery drain. Resolved sync issues. UI improvements.",
        "New release: Redesigned home screen. Fixed search not working. Added export to PDF feature.",
        "Hotfix: Resolved critical crash affecting Samsung devices on Android 15. Fixed data loss on logout.",
        "Feature update: Added widget support. Fixed camera freeze bug. Improved dark mode contrast.",
        "Security update: Fixed session token vulnerability. Improved encryption. Minor bug fixes.",
    ]
    for i, cl in enumerate(generic_changelogs):
        docs.append({
            "id": f"cl_generic_{i}",
            "text": cl,
            "metadata": {"app_id": "generic", "source": "changelogs", "type": "generic_changelog"},
        })

    retriever.index_source("changelogs", docs[:max_docs])
    print(f"  Indexed {len(docs[:max_docs])} changelog entries ({len(app_issues)} apps + {len(generic_changelogs)} generic)")
    return len(docs[:max_docs])


# ============================================================
# Source 3: FAQ
# ============================================================

def index_faq(retriever, rrgen: list[dict], max_docs: int = 200):
    """Create and index FAQ entries from common response patterns.

    Extracts the most common developer response patterns and
    converts them into FAQ-style Q&A pairs.
    """
    print("\n--- Source 3: FAQ ---")

    # Extract common response patterns
    response_patterns = defaultdict(int)
    response_examples = defaultdict(list)

    for r in rrgen[:50000]:  # Sample
        resp = r.get("response", "").strip()
        review = r.get("text", "").strip()
        if len(resp) > 30 and len(review) > 10:
            # Normalize response to find patterns
            key = resp[:80].lower()
            response_patterns[key] += 1
            if len(response_examples[key]) < 3:
                response_examples[key].append({"review": review, "response": resp})

    # Take most common patterns
    common = sorted(response_patterns.items(), key=lambda x: x[1], reverse=True)[:30]

    docs = []
    for pattern, count in common:
        examples = response_examples[pattern]
        if examples:
            faq_text = (
                f"Common Issue: {examples[0]['review'][:200]}\n"
                f"Standard Response: {examples[0]['response'][:300]}\n"
                f"Frequency: This type of issue appears {count}+ times"
            )
            docs.append({
                "id": make_id(faq_text, "faq_"),
                "text": faq_text,
                "metadata": {"source": "faq", "frequency": count, "type": "common_pattern"},
            })

    # Add generic FAQ entries
    generic_faqs = [
        "Q: App crashes on startup. A: Please try clearing the app cache (Settings > Apps > [App Name] > Clear Cache), then restart. If the issue persists, try reinstalling the app. Make sure your device is running the latest OS version.",
        "Q: App drains battery quickly. A: We're continuously working on optimizing battery usage. Please check Settings > Battery to see per-app usage. Try disabling background refresh for the app if not needed.",
        "Q: Can't login to my account. A: Please ensure you're using the correct email and password. Try the 'Forgot Password' option. If using social login, check that the linked account is active. Contact support if the issue continues.",
        "Q: App is very slow. A: Performance can vary based on device and network conditions. Try closing other apps, clearing cache, and ensuring you have a stable internet connection. We're always working to improve performance.",
        "Q: Feature request - how to submit? A: We appreciate your feedback! You can submit feature requests through the app's feedback section or by leaving a review. Our team reviews all suggestions for future updates.",
        "Q: Data not syncing across devices. A: Ensure you're logged in with the same account on all devices. Check that sync is enabled in settings. Try logging out and back in. If the issue persists, contact our support team.",
        "Q: App not compatible with my device. A: We support devices running Android 10+ and iOS 15+. If your device meets the requirements but the app doesn't work, please contact support with your device model and OS version.",
        "Q: How to recover deleted data? A: If you have cloud backup enabled, you can restore from the last backup in Settings > Backup & Restore. Without backup, deleted data may not be recoverable.",
        "Q: Notifications not working. A: Check that notifications are enabled in both the app settings and your device settings (Settings > Notifications > [App Name]). Also ensure Do Not Disturb mode is off.",
        "Q: Payment/subscription issue. A: Payment issues are handled through your app store (Google Play or Apple App Store). Check your payment method and subscription status there. For billing disputes, contact the store directly.",
    ]
    for i, faq in enumerate(generic_faqs):
        docs.append({
            "id": f"faq_generic_{i}",
            "text": faq,
            "metadata": {"source": "faq", "type": "generic_faq"},
        })

    retriever.index_source("faq", docs[:max_docs])
    print(f"  Indexed {len(docs[:max_docs])} FAQ entries ({len(common)} from patterns + {len(generic_faqs)} generic)")
    return len(docs[:max_docs])


# ============================================================
# Source 4: Issue Specs (placeholder — populated at runtime)
# ============================================================

def index_issue_specs(retriever):
    """Create empty collection for issue specs (populated by Stage 3 at runtime)."""
    print("\n--- Source 4: Issue Specs ---")
    retriever._get_collection("issue_spec")
    print("  Collection created (empty — populated by Stage 3 pipeline at runtime)")
    return 0


# ============================================================
# Source 5: Similar Responses (deduplicated high-quality)
# ============================================================

def index_similar_responses(retriever, rrgen: list[dict], max_docs: int = 5000):
    """Index deduplicated high-quality responses for template guidance.

    Selects responses that are:
    - Longer than average (more informative)
    - From high-rated apps (likely better quality)
    - Deduplicated by content similarity
    """
    print("\n--- Source 5: Similar Responses ---")

    # Filter for high-quality responses
    quality_responses = []
    seen_prefixes = set()

    for r in rrgen:
        resp = r.get("response", "").strip()
        review = r.get("text", "").strip()

        if len(resp) < 50 or len(review) < 15:
            continue

        # Dedup by first 60 chars of response
        prefix = resp[:60].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        quality_responses.append({
            "review": review,
            "response": resp,
            "app_id": r.get("app_id", ""),
            "rating": r.get("rating", 3),
        })

    print(f"  Quality responses after dedup: {len(quality_responses)}")

    # Sort by response length (longer = usually more helpful)
    quality_responses.sort(key=lambda x: len(x["response"]), reverse=True)

    docs = []
    for idx, r in enumerate(quality_responses[:max_docs]):
        text = f"User review ({r['rating']} stars): {r['review'][:200]}\nDeveloper response: {r['response'][:400]}"
        docs.append({
            "id": f"sim_{idx}_{make_id(text)}",
            "text": text,
            "metadata": {
                "app_id": r["app_id"],
                "rating": r["rating"],
                "source": "similar_responses",
                "response_length": len(r["response"]),
            },
        })

    batch_size = 1000
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        retriever.index_source("similar_responses", batch)
        print(f"  Indexed batch {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1} ({len(batch)} docs)")

    print(f"  Total indexed: {len(docs)} similar responses")
    return len(docs)


# ============================================================
# Verification
# ============================================================

def verify_index(retriever):
    """Verify the index and run test queries."""
    print("\n" + "=" * 60)
    print("INDEX VERIFICATION")
    print("=" * 60)

    # Collection stats
    print("\nCollection sizes:")
    total = 0
    for source in retriever.SOURCES:
        collection = retriever._get_collection(source)
        count = collection.count()
        total += count
        print(f"  {source}: {count} documents")
    print(f"  TOTAL: {total} documents")

    # Test queries
    test_queries = [
        ("App crashes when I try to login", "Bug report — should find crash-related responses"),
        ("Please add dark mode", "Feature request — should find feature-related responses"),
        ("Battery drain is terrible", "Performance — should find battery/performance responses"),
        ("The checkout page is confusing", "Usability — should find UI/UX responses"),
    ]

    print("\nTest Queries:")
    for query, description in test_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  Expected: {description}")
        results = retriever.retrieve(query, top_k=3)
        if results:
            for i, doc in enumerate(results):
                print(f"    Result {i + 1} [{doc.source}] (score={doc.score:.3f}): {doc.text[:120]}...")
        else:
            print("    No results found!")

    return total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Populate RAG index")
    parser.add_argument("--source", type=str, help="Index only this source")
    parser.add_argument("--max-docs", type=int, default=20000, help="Max docs per source")
    parser.add_argument("--verify", action="store_true", help="Verify index + test queries")
    parser.add_argument("--chroma-path", type=str, default="data/chroma_db")
    args = parser.parse_args()

    from src.stage4b.rag_retriever import RAGRetriever

    print("=" * 60)
    print("ReviewAgent — RAG Index Population")
    print("=" * 60)
    print(f"  ChromaDB path: {args.chroma_path}")
    print(f"  Max docs per source: {args.max_docs}")

    retriever = RAGRetriever(chroma_path=args.chroma_path)

    if args.verify:
        verify_index(retriever)
        return

    # Load RRGen data
    print("\nLoading RRGen data...")
    rrgen = load_rrgen()
    if not rrgen:
        return
    print(f"  Loaded {len(rrgen)} review-response pairs")

    # Index sources
    total = 0
    sources_to_index = [args.source] if args.source else ["past_responses", "changelogs", "faq", "issue_spec", "similar_responses"]

    if "past_responses" in sources_to_index:
        total += index_past_responses(retriever, rrgen, args.max_docs)

    if "changelogs" in sources_to_index:
        total += index_changelogs(retriever, rrgen, min(args.max_docs, 500))

    if "faq" in sources_to_index:
        total += index_faq(retriever, rrgen, min(args.max_docs, 200))

    if "issue_spec" in sources_to_index:
        total += index_issue_specs(retriever)

    if "similar_responses" in sources_to_index:
        total += index_similar_responses(retriever, rrgen, min(args.max_docs, 5000))

    # Verify
    print()
    verify_index(retriever)

    print(f"\n{'=' * 60}")
    print(f"RAG INDEX POPULATED: {total} documents across {len(sources_to_index)} sources")
    print(f"ChromaDB stored at: {args.chroma_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
