"""
Improved Phase 1 clustering with UMAP dim-reduction + class-specific HDBSCAN params.

Why this is better than the first Phase 1:
  - First run produced 3 clusters of 17K reviews each on bug_report and 92% noise on
    feature_request. Reason: 384-dim sentence embeddings are too sparse for
    HDBSCAN, so density-based clustering collapses or fragments badly.
  - Solution: UMAP-reduce 384 → 50 dims (preserves cluster structure, dramatically
    increases density), then HDBSCAN with class-tuned min_cluster_size.

Output:
    data/processed/clusters_umap/clusters_full.json
    data/processed/clusters_umap/clusters_summary.json
    data/processed/clusters_umap/cluster_stats.json
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
from umap import UMAP

sys.path.insert(0, str(Path(__file__).parent.parent))

ACTIONABLE = ["bug_report", "feature_request", "performance", "usability", "compatibility"]

# Per-class HDBSCAN params tuned to the relabeled distribution
# (after UMAP, we want ~50-300 clusters per large class)
CLASS_PARAMS = {
    "bug_report":      {"min_cluster_size": 200, "min_samples": 20, "umap_n_neighbors": 30},
    "feature_request": {"min_cluster_size": 100, "min_samples": 10, "umap_n_neighbors": 25},
    "performance":     {"min_cluster_size": 60,  "min_samples": 8,  "umap_n_neighbors": 20},
    "usability":       {"min_cluster_size": 60,  "min_samples": 8,  "umap_n_neighbors": 20},
    "compatibility":   {"min_cluster_size": 10,  "min_samples": 3,  "umap_n_neighbors": 10},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"))
    ap.add_argument("--label-field", default="v5_label")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/clusters_umap"))
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--umap-dims", type=int, default=50)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    by_type = defaultdict(list)
    for i, r in enumerate(rows):
        lbl = r.get(args.label_field)
        if lbl in ACTIONABLE:
            by_type[lbl].append((i, r))
    print(f"\nReviews per actionable class:")
    for lbl in ACTIONABLE:
        print(f"  {lbl:20s} {len(by_type[lbl]):>6,}")

    print(f"\nLoading embedding model: {args.embedding_model}")
    encoder = SentenceTransformer(args.embedding_model)

    all_clusters = []
    cluster_counter = 0
    per_class_summary = {}

    for issue_type in ACTIONABLE:
        items = by_type[issue_type]
        if not items:
            continue
        params = CLASS_PARAMS[issue_type]
        n = len(items)
        print(f"\n[{issue_type}] {n:,} reviews  params={params}")

        if n < params["min_cluster_size"] * 2:
            print(f"  too few to cluster, keeping as one bucket")
            cluster_counter += 1
            cluster = {
                "cluster_id": f"c_{cluster_counter:05d}",
                "issue_type": issue_type,
                "aspect": "",
                "sub_category": "",
                "review_global_idxs": [i for i, _ in items],
                "review_count": n,
                "representative_reviews": [r["text"] for _, r in items[:3]],
                "review_texts": [r["text"] for _, r in items],
                "rating_distribution": dict(Counter(r.get("rating") for _, r in items)),
                "embedding_method": args.embedding_model,
                "clustering_method": "single_bucket_too_small",
            }
            all_clusters.append(cluster)
            per_class_summary[issue_type] = {"n_reviews": n, "n_clusters": 1, "n_noise": 0,
                                              "noise_pct": 0, "avg_cluster_size": n}
            continue

        # Embed
        t0 = time.time()
        texts = [r["text"] for _, r in items]
        emb = encoder.encode(texts, batch_size=64, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)
        print(f"  embedded in {time.time()-t0:.1f}s — shape {emb.shape}")

        # UMAP
        t0 = time.time()
        n_neighbors = min(params["umap_n_neighbors"], n - 1)
        reducer = UMAP(
            n_components=args.umap_dims,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            verbose=False,
        )
        emb_low = reducer.fit_transform(emb)
        print(f"  UMAP {emb.shape[1]} → {args.umap_dims} dims in {time.time()-t0:.1f}s")

        # HDBSCAN
        t0 = time.time()
        hdb = HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = hdb.fit_predict(emb_low)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"  HDBSCAN in {time.time()-t0:.1f}s — {n_clusters} clusters, "
              f"{n_noise:,} noise ({100*n_noise/n:.1f}%)")

        per_class_summary[issue_type] = {
            "n_reviews": n,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": round(100 * n_noise / n, 2),
            "avg_cluster_size": round((n - n_noise) / n_clusters, 1) if n_clusters > 0 else 0,
        }

        for cid_local in range(n_clusters):
            mask = labels == cid_local
            members = [items[i] for i in range(n) if mask[i]]
            member_idxs = np.where(mask)[0]
            cluster_emb = emb_low[member_idxs]
            centroid = cluster_emb.mean(axis=0)
            dists = np.linalg.norm(cluster_emb - centroid, axis=1)
            top3 = member_idxs[np.argsort(dists)[:3]]
            reps = [items[i][1]["text"] for i in top3]

            cluster_counter += 1
            cluster = {
                "cluster_id": f"c_{cluster_counter:05d}",
                "issue_type": issue_type,
                "aspect": "",
                "sub_category": "",
                "review_global_idxs": [items[i][0] for i in member_idxs.tolist()],
                "review_count": int(mask.sum()),
                "representative_reviews": reps,
                "review_texts": [m[1]["text"] for m in members],
                "rating_distribution": {
                    str(k): v for k, v in Counter(m[1].get("rating") for m in members).items()
                },
                "embedding_method": args.embedding_model,
                "clustering_method": f"umap{args.umap_dims}+hdbscan(min={params['min_cluster_size']})",
            }
            all_clusters.append(cluster)

    all_clusters.sort(key=lambda c: (c["issue_type"], -c["review_count"]))

    print("\n" + "=" * 70)
    print("UMAP+HDBSCAN CLUSTERING SUMMARY")
    print("=" * 70)
    print(f"Total clusters: {len(all_clusters):,}")
    print(f"\n  {'class':18s} {'reviews':>8s} {'clusters':>9s} {'noise%':>8s} {'avg':>9s}")
    for lbl in ACTIONABLE:
        s = per_class_summary.get(lbl, {})
        print(f"  {lbl:18s} {s.get('n_reviews',0):>8,} {s.get('n_clusters',0):>9,} "
              f"{s.get('noise_pct',0):>7.1f}% {s.get('avg_cluster_size',0):>9.1f}")

    print(f"\nTop 10 largest clusters:")
    for c in sorted(all_clusters, key=lambda x: -x["review_count"])[:10]:
        print(f"\n  [{c['cluster_id']}] {c['issue_type']:15s} count={c['review_count']:>4}")
        for rep in c["representative_reviews"][:2]:
            print(f"     - {rep[:110]}")

    with open(args.out_dir / "clusters_full.json", "w") as f:
        json.dump(all_clusters, f)
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
            "umap_dims": args.umap_dims,
            "class_params": CLASS_PARAMS,
        }, f, indent=2)

    print(f"\nOutputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
