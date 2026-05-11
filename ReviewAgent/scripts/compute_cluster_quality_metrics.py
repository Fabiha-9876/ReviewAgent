"""
Compute standard cluster-quality metrics on flat vs hierarchical-KG clusterings
to address Reviewer Gaps #13–#16.

Metrics computed:
  - Cluster count, mean/median/std size  (already known; re-confirmed)
  - Silhouette coefficient (intrinsic, embedding-based)
  - Davies-Bouldin index (intrinsic, lower = better)
  - Calinski-Harabasz score (intrinsic, higher = better)
  - Aspect purity (per-cluster fraction of reviews sharing the dominant aspect)
  - Y/P/N weighted purity (the lead-author audit)
  - Per-design comparison delta

Outputs:
  data/processed/clusters_umap/quality_metrics_flat_vs_hierarchical.json
  data/processed/clusters_umap/quality_metrics_summary.txt
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

OUT_DIR = Path("data/processed/clusters_umap")
FLAT_FILE = OUT_DIR / "clusters_full.json"
HIER_FILE = Path("data/processed/kg_hierarchical/hierarchical_clusters_full.json")
VAL_FILE = OUT_DIR / "cluster_validation_score.json"


def load_cluster_labels(cluster_file, max_reviews=None):
    """Return (review_idxs, labels) where labels[i] is the cluster of review i."""
    with open(cluster_file) as f:
        clusters = json.load(f)

    review_idx_to_cluster = {}
    for c in clusters:
        cid = c["cluster_id"]
        idxs = c.get("review_global_idxs",
                       c.get("review_idxs",
                              c.get("review_ids", [])))
        for idx in idxs:
            # Hierarchical uses string ids like "r_12345"; flat uses ints
            review_idx_to_cluster[idx] = cid

    review_idxs = sorted(review_idx_to_cluster.keys(), key=str)
    labels = [review_idx_to_cluster[i] for i in review_idxs]
    cid_to_int = {cid: i for i, cid in enumerate(sorted(set(labels)))}
    int_labels = np.array([cid_to_int[l] for l in labels])

    if max_reviews and len(review_idxs) > max_reviews:
        rng = np.random.default_rng(42)
        sub = rng.choice(len(review_idxs), max_reviews, replace=False)
        review_idxs = [review_idxs[i] for i in sub]
        int_labels = int_labels[sub]

    return review_idxs, int_labels, len(set(labels)), clusters


def cluster_size_stats(labels):
    counts = Counter(labels.tolist())
    sizes = list(counts.values())
    return {
        "n_clusters": len(counts),
        "mean_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)),
        "size_std": float(np.std(sizes)),
        "max_size": int(max(sizes)),
        "min_size": int(min(sizes)),
    }


def compute_intrinsic_metrics(embeddings, labels):
    """Silhouette + Davies-Bouldin + Calinski-Harabasz."""
    if len(set(labels)) < 2:
        return None
    sil = silhouette_score(embeddings, labels, metric="cosine", sample_size=min(5000, len(labels)), random_state=42)
    db = davies_bouldin_score(embeddings, labels)
    ch = calinski_harabasz_score(embeddings, labels)
    return {
        "silhouette_cosine": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
    }


def compute_aspect_purity(clusters):
    """Per-cluster fraction sharing dominant aspect. Returns mean across clusters."""
    purities = []
    for c in clusters:
        aspect = c.get("aspect", "")
        sub = c.get("sub_category", "")
        # If cluster has explicit aspect tag (hierarchical), purity = 1 by construction
        if aspect:
            purities.append(1.0)
        else:
            # Flat: no aspect tag; aspect purity is undefined → skip
            pass
    if not purities:
        return None
    return float(np.mean(purities))


def main():
    # ----- Load embeddings (need to re-encode reviews) -----
    print("Loading sentence-transformer for re-embedding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------- FLAT clustering ----------
    print("\n=== FLAT (UMAP+HDBSCAN) ===")
    flat_idxs, flat_labels, n_flat, flat_clusters = load_cluster_labels(FLAT_FILE, max_reviews=10000)
    print(f"n reviews scored: {len(flat_idxs)}, n clusters: {n_flat}")

    # Re-encode the review texts
    flat_texts = []
    for c in flat_clusters:
        for t in c.get("review_texts", []):
            flat_texts.append(t)
            if len(flat_texts) >= len(flat_idxs):
                break
        if len(flat_texts) >= len(flat_idxs):
            break
    flat_texts = flat_texts[:len(flat_idxs)]
    if len(flat_texts) < len(flat_idxs):
        # Use available
        flat_labels = flat_labels[:len(flat_texts)]
    print(f"Encoding {len(flat_texts)} review texts...")
    flat_embs = model.encode(flat_texts, batch_size=64, show_progress_bar=False)
    print(f"Embeddings: {flat_embs.shape}")

    flat_size_stats = cluster_size_stats(flat_labels)
    flat_intrinsic = compute_intrinsic_metrics(flat_embs, flat_labels)
    flat_aspect_purity = compute_aspect_purity(flat_clusters)
    print(f"  size stats: {flat_size_stats}")
    print(f"  intrinsic:  {flat_intrinsic}")
    print(f"  aspect purity: {flat_aspect_purity}")

    # ---------- HIERARCHICAL KG clustering ----------
    print("\n=== HIERARCHICAL (Aspect-Grounded KG) ===")
    if HIER_FILE.exists():
        hier_idxs, hier_labels, n_hier, hier_clusters = load_cluster_labels(HIER_FILE, max_reviews=10000)
        print(f"n reviews scored: {len(hier_idxs)}, n clusters: {n_hier}")

        # Re-encode using representative reviews (hierarchical only stores reps, not all texts)
        # Build list of (text, cluster_label) pairs from representative_reviews per cluster
        hier_pairs = []
        cid_to_int = {c["cluster_id"]: i for i, c in enumerate(sorted(hier_clusters, key=lambda x: x["cluster_id"]))}
        for c in hier_clusters:
            cid_int = cid_to_int[c["cluster_id"]]
            reps = c.get("representative_reviews", [])
            for t in reps:
                if isinstance(t, str) and t.strip():
                    hier_pairs.append((t, cid_int))
        # Subsample
        if len(hier_pairs) > 10000:
            rng = np.random.default_rng(42)
            sub = rng.choice(len(hier_pairs), 10000, replace=False)
            hier_pairs = [hier_pairs[i] for i in sub]
        hier_texts = [p[0] for p in hier_pairs]
        hier_labels = np.array([p[1] for p in hier_pairs])
        print(f"Encoding {len(hier_texts)} review texts...")
        hier_embs = model.encode(hier_texts, batch_size=64, show_progress_bar=False)

        hier_size_stats = cluster_size_stats(hier_labels)
        hier_intrinsic = compute_intrinsic_metrics(hier_embs, hier_labels)
        hier_aspect_purity = compute_aspect_purity(hier_clusters)
        print(f"  size stats: {hier_size_stats}")
        print(f"  intrinsic:  {hier_intrinsic}")
        print(f"  aspect purity: {hier_aspect_purity}")
    else:
        hier_size_stats = hier_intrinsic = hier_aspect_purity = None
        print("Hierarchical file not found — skipping")

    # ---------- Y/P/N weighted purity (already audited, just load) ----------
    with open(VAL_FILE) as f:
        val = json.load(f)
    flat_yp_purity = val["overall_purity"]
    flat_yp_breakdown = val["overall_counts"]

    # ---------- Combined output ----------
    out = {
        "metric_definitions": {
            "silhouette_cosine":   "Mean over all reviews of (b-a)/max(a,b) where a = mean cosine distance to other reviews in same cluster, b = mean cosine distance to nearest other cluster. Range [-1, 1]; higher = better separation. Rousseeuw 1987.",
            "davies_bouldin":      "Mean over clusters of (max over other clusters) of (sigma_i + sigma_j) / d(c_i, c_j). Lower = better. Davies-Bouldin 1979.",
            "calinski_harabasz":   "Ratio of between-cluster to within-cluster dispersion. Higher = better. Calinski-Harabasz 1974.",
            "aspect_purity":       "Per-cluster fraction of reviews sharing the dominant aspect tag. Hierarchical pipeline assigns aspects by construction so purity = 1.0; flat pipeline does not assign aspects.",
            "yp_weighted_purity":  "Lead-author audit on a 50-cluster sample: per-cluster verdict in {Y (5/5 share theme), P (3-4/5 share theme), N (incoherent)}; weighted purity = (1*Y + 0.5*P + 0*N) / (Y+P+N). Manning et al. 2008; Steinbach et al. 2000.",
        },
        "flat_umap_hdbscan": {
            "size_stats": flat_size_stats,
            "intrinsic_metrics": flat_intrinsic,
            "aspect_purity": flat_aspect_purity,
            "yp_weighted_purity_50_audit": flat_yp_purity,
            "yp_breakdown": flat_yp_breakdown,
        },
        "hierarchical_kg": {
            "size_stats": hier_size_stats,
            "intrinsic_metrics": hier_intrinsic,
            "aspect_purity": hier_aspect_purity,
            "yp_weighted_purity_50_audit": "not yet audited (queued in §5.6 future work)",
        },
    }

    if hier_intrinsic and flat_intrinsic:
        out["delta_hierarchical_minus_flat"] = {
            "silhouette":         hier_intrinsic["silhouette_cosine"] - flat_intrinsic["silhouette_cosine"],
            "davies_bouldin":     hier_intrinsic["davies_bouldin"]    - flat_intrinsic["davies_bouldin"],
            "calinski_harabasz":  hier_intrinsic["calinski_harabasz"] - flat_intrinsic["calinski_harabasz"],
            "n_clusters_ratio":   hier_size_stats["n_clusters"] / flat_size_stats["n_clusters"],
            "mean_size_ratio":    flat_size_stats["mean_size"] / max(hier_size_stats["mean_size"], 1),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "quality_metrics_flat_vs_hierarchical.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'quality_metrics_flat_vs_hierarchical.json'}")

    # Summary
    summary = [
        "=" * 72,
        "Cluster quality metrics — flat (UMAP+HDBSCAN) vs hierarchical (KG)",
        "=" * 72,
        "",
        f"{'metric':<32} {'flat':>14} {'hierarchical':>16} {'Δ (h - f)':>14}",
        "-" * 72,
    ]

    def row(name, flat_val, hier_val, fmt="{:.3f}"):
        if flat_val is None and hier_val is None:
            return f"{name:<32} {'n/a':>14} {'n/a':>16} {'n/a':>14}"
        f_str = fmt.format(flat_val) if flat_val is not None else "n/a"
        h_str = fmt.format(hier_val) if hier_val is not None else "n/a"
        if flat_val is not None and hier_val is not None:
            delta = hier_val - flat_val
            return f"{name:<32} {f_str:>14} {h_str:>16} {delta:>+14.3f}"
        return f"{name:<32} {f_str:>14} {h_str:>16} {'n/a':>14}"

    summary.append(row("n_clusters",
                       flat_size_stats["n_clusters"],
                       hier_size_stats["n_clusters"] if hier_size_stats else None,
                       "{:.0f}"))
    summary.append(row("mean cluster size",
                       flat_size_stats["mean_size"],
                       hier_size_stats["mean_size"] if hier_size_stats else None,
                       "{:.1f}"))
    summary.append(row("median cluster size",
                       flat_size_stats["median_size"],
                       hier_size_stats["median_size"] if hier_size_stats else None,
                       "{:.1f}"))
    if flat_intrinsic:
        summary.append(row("silhouette (cosine, ↑)",
                           flat_intrinsic["silhouette_cosine"],
                           hier_intrinsic["silhouette_cosine"] if hier_intrinsic else None))
        summary.append(row("Davies-Bouldin (↓)",
                           flat_intrinsic["davies_bouldin"],
                           hier_intrinsic["davies_bouldin"] if hier_intrinsic else None))
        summary.append(row("Calinski-Harabasz (↑)",
                           flat_intrinsic["calinski_harabasz"],
                           hier_intrinsic["calinski_harabasz"] if hier_intrinsic else None,
                           "{:.1f}"))
    summary.append(row("aspect purity",
                       flat_aspect_purity,
                       hier_aspect_purity))
    summary.append(row("Y/P/N weighted purity (50-audit)",
                       flat_yp_purity,
                       None))

    summary.extend([
        "",
        "Notes:",
        "  - Y/P/N audit on hierarchical clusters is queued in §5.6 future work.",
        "  - Silhouette / DB / CH are *intrinsic* metrics: do not require true labels.",
        "  - Aspect purity is 1.0 for hierarchical by construction (aspect-grouped).",
        "  - Lower DB = better separation; higher silhouette and CH = better separation.",
    ])
    text = "\n".join(summary)
    print("\n" + text)
    with open(OUT_DIR / "quality_metrics_summary.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
