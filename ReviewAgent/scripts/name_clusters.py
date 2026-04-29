"""
Generate human-readable names for the 194 UMAP+HDBSCAN clusters using
Phase 2 heuristic aspects + TF-IDF-style distinctiveness scoring.

For each cluster:
  1. Look up Phase 2 aspects for every review in the cluster.
  2. Compute aspect frequency within the cluster (TF).
  3. Compute aspect frequency across clusters (DF) — generic aspects like "ad",
     "update", "phone" appear in many clusters and aren't distinctive.
  4. Score each aspect by TF * log(N_clusters / DF) — classic TF-IDF.
  5. Top-3 distinctive aspects become the cluster name.

Output:
    data/processed/clusters_umap/clusters_named.json     full clusters with auto_name
    data/processed/clusters_umap/clusters_named_summary.json  light version
    Console summary printing each cluster's name + sample review.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def main():
    clusters_path = Path("data/processed/clusters_umap/clusters_full.json")
    aspects_path  = Path("data/processed/aspects_heuristic/aspects_per_review.json")
    out_dir       = Path("data/processed/clusters_umap")

    print(f"Loading clusters: {clusters_path}")
    with open(clusters_path) as f:
        clusters = json.load(f)
    print(f"  {len(clusters):,} clusters")

    print(f"Loading heuristic aspects: {aspects_path}")
    with open(aspects_path) as f:
        aspects_by_idx = json.load(f)  # keys are str(idx)
    print(f"  {len(aspects_by_idx):,} reviews with aspects")

    # 1. Per-cluster aspect counts
    cluster_aspect_counts = []  # list of Counter, parallel to clusters
    for c in clusters:
        cnt = Counter()
        for gi in c.get("review_global_idxs", []):
            asps = aspects_by_idx.get(str(gi), [])
            for a in asps:
                cnt[a] += 1
        cluster_aspect_counts.append(cnt)

    # 2. Document frequency (in how many clusters does each aspect appear?)
    df = Counter()
    for cnt in cluster_aspect_counts:
        for a in cnt.keys():
            df[a] += 1

    # 3. Aspects to ignore (too generic / low signal)
    BLOCKLIST = {
        "ad", "ads", "phone", "app", "thing", "things", "update", "version",
        "lot", "way", "time", "today", "yesterday", "user", "lots", "everyone",
        "people", "anyone", "year", "years", "month", "months", "week", "day",
        "digit", "stuff", "everything", "nothing", "something", "anything",
        "developer", "developers", "support", "service", "team", "guy", "guys",
        "review", "reviews", "rating", "star", "stars",
    }

    N = len(clusters)

    # 4. Score aspects per cluster, pick top distinctive
    named_clusters = []
    for i, (c, cnt) in enumerate(zip(clusters, cluster_aspect_counts)):
        if c["review_count"] == 0:
            c["auto_name"] = f"{c['issue_type']}: (empty cluster)"
            c["top_aspects"] = []
            named_clusters.append(c)
            continue

        n_reviews = c["review_count"]
        scored = []
        for asp, tf in cnt.items():
            if asp in BLOCKLIST:
                continue
            if df[asp] == 0:
                continue
            # TF-IDF style: aspect must appear in >=5% of cluster's reviews to be considered
            coverage = tf / n_reviews
            if coverage < 0.05:
                continue
            idf = math.log(N / df[asp])
            score = coverage * idf
            scored.append((score, asp, tf, coverage, df[asp]))

        scored.sort(key=lambda x: -x[0])
        top = scored[:3]

        if top:
            name_parts = [t[1] for t in top]
            auto_name = f"{c['issue_type']}: {' / '.join(name_parts)}"
        else:
            # No distinctive aspects — fall back to first representative review (truncated)
            rep = c["representative_reviews"][0] if c.get("representative_reviews") else ""
            auto_name = f"{c['issue_type']}: {rep[:60]}..."

        c["auto_name"] = auto_name
        c["top_aspects"] = [
            {
                "aspect": t[1],
                "in_cluster_count": t[2],
                "in_cluster_pct": round(t[3] * 100, 1),
                "in_n_clusters": t[4],
                "score": round(t[0], 4),
            }
            for t in top
        ]
        named_clusters.append(c)

    # Save outputs
    with open(out_dir / "clusters_named.json", "w") as f:
        json.dump(named_clusters, f)
    light = []
    for c in named_clusters:
        light.append({k: v for k, v in c.items() if k != "review_texts"})
    with open(out_dir / "clusters_named_summary.json", "w") as f:
        json.dump(light, f, indent=2)

    # Console summary
    print("\n" + "=" * 80)
    print("AUTO-NAMED CLUSTERS — Top 30 by size")
    print("=" * 80)
    for c in sorted(named_clusters, key=lambda x: -x["review_count"])[:30]:
        print(f"\n  [{c['cluster_id']}] n={c['review_count']:>5}  {c['auto_name']}")
        if c.get("top_aspects"):
            asp_str = ", ".join(f"{a['aspect']} ({a['in_cluster_pct']}%)" for a in c["top_aspects"])
            print(f"    aspects: {asp_str}")
        rep = c["representative_reviews"][0] if c.get("representative_reviews") else ""
        print(f"    sample:  {rep[:100]}")

    # Per-class breakdown
    print("\n" + "=" * 80)
    print("Cluster name examples per class (top 5 by size)")
    print("=" * 80)
    for cls in ["bug_report", "feature_request", "performance", "usability", "compatibility"]:
        cls_clusters = [c for c in named_clusters if c["issue_type"] == cls]
        cls_clusters.sort(key=lambda x: -x["review_count"])
        print(f"\n[{cls}] {len(cls_clusters)} clusters")
        for c in cls_clusters[:5]:
            print(f"  n={c['review_count']:>5}  {c['auto_name']}")

    # Stats
    n_named  = sum(1 for c in named_clusters if c.get("top_aspects"))
    n_unnamed = len(named_clusters) - n_named
    print(f"\n{n_named}/{len(named_clusters)} clusters got distinctive aspect names")
    print(f"{n_unnamed} fell back to representative-review naming (no distinctive aspects)")
    print(f"\nOutputs: {out_dir}/clusters_named.json")
    print(f"         {out_dir}/clusters_named_summary.json")


if __name__ == "__main__":
    main()
