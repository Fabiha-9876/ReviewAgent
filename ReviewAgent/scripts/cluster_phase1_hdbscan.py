"""
Phase 1 of free-tier Stage 2: pure HDBSCAN clustering on V5-labeled reviews.

No aspect extraction (that's Phase 2/3). This produces baseline clusters by:
  1. Filter to actionable issue types (bug_report, feature_request, performance,
     usability, compatibility). Skip praise + other.
  2. For each issue type, embed all reviews with sentence-transformers.
  3. HDBSCAN cluster within each issue type.
  4. Build IssueCluster objects (with aspect="" — filled later in Phase 2/3).
  5. Output one master cluster file + per-class summaries.

Usage:
    python3 scripts/cluster_phase1_hdbscan.py \
        --input data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json \
        --label-field v5_label \
        --out-dir data/processed/clusters_no_aspect \
        --min-cluster-size 10
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS

ACTIONABLE = ["bug_report", "feature_request", "performance", "usability", "compatibility"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"))
    ap.add_argument("--label-field", default="v5_label",
                    help="Which label field to cluster on (v5_label, corrected_v2_label, etc.)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/clusters_no_aspect"))
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Cap reviews per class (for testing/sanity-check).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Allow fallback if V5 relabel hasn't finished yet
    if not args.input.exists():
        fallback = Path("data/processed/rrgen_corrected_v2/rrgen_corrected_v2.json")
        print(f"Input {args.input} not found, falling back to {fallback}")
        args.input = fallback
        if args.label_field == "v5_label":
            args.label_field = "final_label"
            print(f"  switching --label-field to {args.label_field}")

    print(f"Loading: {args.input}")
    with open(args.input) as f:
        rows = json.load(f)
    print(f"  {len(rows):,} rows")

    # Filter to actionable types
    by_type = defaultdict(list)
    for r in rows:
        lbl = r.get(args.label_field)
        if lbl in ACTIONABLE:
            by_type[lbl].append(r)
    print(f"\nReviews per actionable class (label_field={args.label_field}):")
    for lbl in ACTIONABLE:
        print(f"  {lbl:20s} {len(by_type[lbl]):>6,}")
    skipped = sum(1 for r in rows if r.get(args.label_field) in ("praise", "other"))
    print(f"  (skipped praise + other: {skipped:,})")

    print(f"\nLoading embedding model: {args.embedding_model}")
    encoder = SentenceTransformer(args.embedding_model)

    all_clusters = []
    cluster_counter = 0
    per_class_summary = {}

    for issue_type in ACTIONABLE:
        reviews = by_type[issue_type]
        if not reviews:
            print(f"\n[{issue_type}] empty, skipping")
            continue
        if args.max_per_class and len(reviews) > args.max_per_class:
            reviews = reviews[:args.max_per_class]

        n = len(reviews)
        print(f"\n[{issue_type}] {n:,} reviews")
        if n < args.min_cluster_size * 2:
            print(f"  too few reviews to cluster (need >={args.min_cluster_size*2}), keeping as 1 cluster")
            cluster_counter += 1
            cluster = {
                "cluster_id": f"c_{cluster_counter:05d}",
                "issue_type": issue_type,
                "aspect": "",
                "sub_category": "",
                "review_ids": [f"r_{i}" for i in range(n)],
                "review_count": n,
                "representative_reviews": [r["text"] for r in reviews[:3]],
                "review_texts": [r["text"] for r in reviews],
                "rating_distribution": dict(Counter(r.get("rating") for r in reviews)),
                "embedding_method": args.embedding_model,
                "clustering_method": "single_class_no_split",
            }
            all_clusters.append(cluster)
            per_class_summary[issue_type] = {
                "n_reviews": n, "n_clusters": 1, "n_noise": 0,
                "avg_cluster_size": n,
            }
            continue

        # Embed
        t0 = time.time()
        texts = [r["text"] for r in reviews]
        embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False,
                                     convert_to_numpy=True, normalize_embeddings=True)
        embed_time = time.time() - t0
        print(f"  embedded in {embed_time:.1f}s — shape {embeddings.shape}")

        # Cluster
        t0 = time.time()
        hdb = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",  # safe with normalized embeddings ≈ cosine
            cluster_selection_method="eom",
        )
        labels = hdb.fit_predict(embeddings)
        cluster_time = time.time() - t0

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"  clustered in {cluster_time:.1f}s — {n_clusters} clusters, {n_noise:,} noise points")

        per_class_summary[issue_type] = {
            "n_reviews": n,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": 100 * n_noise / n,
            "avg_cluster_size": (n - n_noise) / n_clusters if n_clusters > 0 else 0,
        }

        # Build IssueCluster objects
        for cluster_id_local in range(n_clusters):
            mask = labels == cluster_id_local
            members = [reviews[i] for i in range(n) if mask[i]]
            member_idxs = np.where(mask)[0]

            # Pick representatives: 3 closest to centroid
            cluster_embeds = embeddings[member_idxs]
            centroid = cluster_embeds.mean(axis=0)
            dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
            top3 = member_idxs[np.argsort(dists)[:3]]
            reps = [reviews[i]["text"] for i in top3]

            cluster_counter += 1
            cluster = {
                "cluster_id": f"c_{cluster_counter:05d}",
                "issue_type": issue_type,
                "aspect": "",  # filled in Phase 2/3
                "sub_category": "",
                "review_ids": [f"r_{i}" for i in member_idxs.tolist()],
                "review_count": len(members),
                "representative_reviews": reps,
                "review_texts": [m["text"] for m in members],
                "rating_distribution": {
                    str(k): v for k, v in Counter(m.get("rating") for m in members).items()
                },
                "embedding_method": args.embedding_model,
                "clustering_method": f"hdbscan(min={args.min_cluster_size},samples={args.min_samples})",
            }
            all_clusters.append(cluster)

    # Sort by issue_type then size
    all_clusters.sort(key=lambda c: (c["issue_type"], -c["review_count"]))

    # Write outputs
    print("\n" + "=" * 70)
    print("CLUSTERING SUMMARY (Phase 1 — no aspect grouping)")
    print("=" * 70)
    print(f"Total clusters: {len(all_clusters):,}")
    print(f"\n  {'class':20s} {'reviews':>8s} {'clusters':>9s} {'noise%':>8s} {'avg size':>9s}")
    for lbl in ACTIONABLE:
        s = per_class_summary.get(lbl, {})
        print(f"  {lbl:20s} {s.get('n_reviews',0):>8,} {s.get('n_clusters',0):>9,} "
              f"{s.get('noise_pct',0):>7.1f}% {s.get('avg_cluster_size',0):>9.1f}")

    # Show top 5 largest clusters
    print(f"\nTop 10 largest clusters (with sample texts):")
    for c in sorted(all_clusters, key=lambda x: -x["review_count"])[:10]:
        print(f"\n  [{c['cluster_id']}] {c['issue_type']:15s} count={c['review_count']:>4}")
        for rep in c["representative_reviews"][:2]:
            print(f"     - {rep[:110]}")

    # Save full clusters (with member texts, can be large)
    out_full = args.out_dir / "clusters_full.json"
    with open(out_full, "w") as f:
        json.dump(all_clusters, f)

    # Save lighter version (no member texts — just reps) for quick browsing
    light = []
    for c in all_clusters:
        c2 = {k: v for k, v in c.items() if k != "review_texts"}
        light.append(c2)
    with open(args.out_dir / "clusters_summary.json", "w") as f:
        json.dump(light, f, indent=2)

    with open(args.out_dir / "cluster_stats.json", "w") as f:
        json.dump({
            "input": str(args.input),
            "label_field": args.label_field,
            "total_clusters": len(all_clusters),
            "per_class": per_class_summary,
            "embedding_model": args.embedding_model,
            "hdbscan_params": {
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
            },
        }, f, indent=2)

    print(f"\nOutputs in {args.out_dir}/")
    print(f"  clusters_full.json     — all {len(all_clusters):,} clusters with member texts")
    print(f"  clusters_summary.json  — same but without member texts (faster to browse)")
    print(f"  cluster_stats.json     — summary stats")


if __name__ == "__main__":
    main()
