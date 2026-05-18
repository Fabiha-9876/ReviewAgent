"""
Ablation A1b: count-controlled flat-vs-KG comparison.

Reviewers ask: does the KG-grounded hierarchical clustering produce its observed
intrinsic-metric gains because of the *aspect structure* or just because it
produces more (smaller) clusters?

This script answers that by tuning the flat HDBSCAN min_cluster_size downward
until it produces approximately 605 clusters (matching the KG hierarchical
count), then comparing intrinsic metrics + a 50-cluster Y/P/N purity audit
between the two count-matched designs.

Inputs
------
  data/processed/clusters_no_aspect/clusters_full.json  (or re-run with finer thresholds)
  data/processed/kg_hierarchical/hierarchical_clusters.json
  embeddings used by the original clustering (re-derived from the embedding cache)

Outputs
-------
  data/processed/ablations/a1b_fine_flat_vs_kg.json
    flat_fine: {n_clusters, mean_size, silhouette, davies_bouldin, calinski_harabasz}
    hierarchical_kg: {n_clusters, mean_size, silhouette, davies_bouldin, calinski_harabasz}
    delta: per-metric (kg - flat_fine), signed
    interpretation: text summary

  (optional, if ANTHROPIC_API_KEY set) llm_audit_a1b.json
    50-cluster Y/P/N audit on each, weighted purity, +/-

Requires
--------
  pip install hdbscan umap-learn sentence-transformers scikit-learn anthropic

Usage
-----
  python scripts/ablation_a1b_fine_flat_vs_kg.py
  # ~ 15-30 min on CPU for the clustering + metrics; LLM audit is optional and adds ~$0.20.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HIER_PATH = Path("data/processed/kg_hierarchical/hierarchical_clusters.json")
FLAT_RAW_PATH = Path("data/processed/clusters_umap/clusters_full.json")
EMB_CACHE = Path("data/processed/embeddings_cache.npy")
OUT_PATH = Path("data/processed/ablations/a1b_fine_flat_vs_kg.json")


def find_min_cluster_size_for_target_count(embeddings, target=605):
    """Binary search HDBSCAN min_cluster_size to get approximately `target` clusters."""
    try:
        import hdbscan
    except ImportError:
        print("pip install hdbscan", file=sys.stderr); sys.exit(1)
    lo, hi = 5, 200
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        labels = hdbscan.HDBSCAN(min_cluster_size=mid, metric="euclidean").fit_predict(embeddings)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        if best is None or abs(n - target) < abs(best[0] - target):
            best = (n, mid, labels)
        if n > target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def compute_intrinsic(embeddings, labels):
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    mask = labels != -1
    if mask.sum() < 100 or len(set(labels[mask])) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    return {
        "silhouette": float(silhouette_score(embeddings[mask], labels[mask], sample_size=5000, random_state=42)),
        "davies_bouldin": float(davies_bouldin_score(embeddings[mask], labels[mask])),
        "calinski_harabasz": float(calinski_harabasz_score(embeddings[mask], labels[mask])),
    }


def cluster_sizes(labels):
    sizes = np.bincount(labels[labels >= 0])
    return {"n_clusters": int(len(sizes)), "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
             "median_size": float(np.median(sizes)) if len(sizes) else 0.0,
             "min": int(sizes.min()) if len(sizes) else 0,
             "max": int(sizes.max()) if len(sizes) else 0}


def main() -> int:
    if not EMB_CACHE.exists():
        print(f"Missing embeddings cache: {EMB_CACHE}", file=sys.stderr)
        print("Recompute embeddings with: python scripts/cluster_phase1b_umap_hdbscan.py --emit-cache", file=sys.stderr)
        return 1

    embeddings = np.load(EMB_CACHE)
    print(f"Loaded {embeddings.shape} embeddings", file=sys.stderr)

    # Fine-thresholded flat
    print("Tuning flat HDBSCAN to ~605 clusters...", file=sys.stderr)
    n_flat, min_cs, labels_flat = find_min_cluster_size_for_target_count(embeddings, target=605)
    print(f"Best: {n_flat} clusters at min_cluster_size={min_cs}", file=sys.stderr)
    flat_sizes = cluster_sizes(labels_flat)
    flat_intrinsic = compute_intrinsic(embeddings, labels_flat)

    # KG hierarchical (read labels from existing artifact)
    hier = json.load(open(HIER_PATH))
    # Build label vector: each cluster has a list of review indices in cluster_members or representative_reviews ids
    # The hierarchical clusters file may not store per-review labels; if not, this section requires the
    # original clustering pipeline to be re-run. Document the dependency.
    print(f"Hierarchical clusters: {len(hier)}", file=sys.stderr)
    # Try to extract per-review cluster assignments from member_indices if present
    labels_hier = np.full(len(embeddings), -1, dtype=np.int64)
    has_members = "member_indices" in hier[0]
    if has_members:
        for cid, c in enumerate(hier):
            for idx in c["member_indices"]:
                if idx < len(embeddings):
                    labels_hier[idx] = cid
        hier_sizes = cluster_sizes(labels_hier)
        hier_intrinsic = compute_intrinsic(embeddings, labels_hier)
    else:
        print("WARN: hierarchical_clusters.json lacks per-review member_indices; cannot compute intrinsic metrics for KG.", file=sys.stderr)
        print("      Re-run scripts/run_kg_hierarchical_clustering.py with --emit-member-indices, then re-run this.", file=sys.stderr)
        hier_sizes = {"n_clusters": len(hier)}
        hier_intrinsic = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
                           "note": "missing per-review labels in hierarchical_clusters.json"}

    delta = {}
    for k in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        f = flat_intrinsic.get(k); h = hier_intrinsic.get(k)
        delta[k] = (h - f) if (f is not None and h is not None) else None

    interpretation = []
    if delta.get("davies_bouldin") is not None:
        if delta["davies_bouldin"] < 0:
            interpretation.append("KG-grounded clusters are MORE compact/separated than count-matched flat (lower DB).")
        else:
            interpretation.append("KG-grounded clusters are LESS compact/separated than count-matched flat (KG structure is decorative for compactness).")
    if delta.get("silhouette") is not None:
        if delta["silhouette"] > 0:
            interpretation.append("KG-grounded clusters have higher silhouette (better-defined boundaries).")
        else:
            interpretation.append("KG-grounded clusters have lower silhouette.")

    out = {
        "flat_fine_threshold": {"min_cluster_size": min_cs, **flat_sizes, **flat_intrinsic},
        "hierarchical_kg": {**hier_sizes, **hier_intrinsic},
        "delta_kg_minus_flat": delta,
        "interpretation": interpretation or ["Insufficient data to compare; see WARN above."],
        "method": "count-matched HDBSCAN: tune min_cluster_size to match KG hierarchical cluster count (~605)",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved -> {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
