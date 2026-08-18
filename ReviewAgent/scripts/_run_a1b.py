"""Self-contained A1b runner. Uses the embeddings cache to:
  1. Build KG-605 label vector from hierarchical_clusters_full.json (review_ids).
  2. Re-cluster the same embeddings with global HDBSCAN, binary-searching
     min_cluster_size to hit ~605 clusters.
  3. Compute silhouette / Davies-Bouldin / Calinski-Harabasz on both.
  4. Report deltas + interpretation.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
EMB_PATH = BASE / "data/processed/embeddings_cache.npy"
IDX_PATH = BASE / "data/processed/embeddings_cache_index.json"
KG_PATH = BASE / "data/processed/kg_hierarchical/hierarchical_clusters_full.json"
OUT_PATH = BASE / "data/processed/ablations/a1b_fine_flat_vs_kg.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading embeddings", file=sys.stderr)
emb_full = np.load(EMB_PATH)
idx_meta = json.load(open(IDX_PATH))
actionable = idx_meta["actionable_indices"]
glob_to_row = {g: r for r, g in enumerate(actionable)}
print(f"  emb shape: {emb_full.shape}; actionable n={len(actionable):,}", file=sys.stderr)

print("UMAP-reducing 384 -> 50 dims (matches original pipeline)", file=sys.stderr)
t0 = time.time()
from umap import UMAP
um = UMAP(n_components=50, n_neighbors=30, metric="cosine", random_state=42,
          low_memory=True, n_jobs=-1)
emb = um.fit_transform(emb_full).astype(np.float32)
print(f"  UMAP done in {time.time()-t0:.1f}s, shape={emb.shape}", file=sys.stderr)

print("Loading KG hierarchical clusters", file=sys.stderr)
hier = json.load(open(KG_PATH))
print(f"  {len(hier)} clusters", file=sys.stderr)

labels_kg = np.full(emb.shape[0], -1, dtype=np.int64)
n_assigned = 0
n_missing = 0
for cid, cluster in enumerate(hier):
    for rid in cluster.get("review_ids", []):
        row = glob_to_row.get(rid)
        if row is None:
            n_missing += 1
            continue
        labels_kg[row] = cid
        n_assigned += 1
print(f"  KG assigned: {n_assigned:,}, missing-from-embeddings: {n_missing:,}", file=sys.stderr)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def intrinsic(emb, labels):
    mask = labels != -1
    n_clust = len(set(labels[mask]))
    if mask.sum() < 200 or n_clust < 2:
        return {"silhouette_cosine": None, "davies_bouldin": None, "calinski_harabasz": None,
                "n_used": int(mask.sum()), "n_clusters": int(n_clust)}
    # silhouette via subsampling to keep it fast
    sil = silhouette_score(emb[mask], labels[mask], metric="cosine",
                           sample_size=min(10000, int(mask.sum())), random_state=42)
    db = davies_bouldin_score(emb[mask], labels[mask])
    ch = calinski_harabasz_score(emb[mask], labels[mask])
    return {"silhouette_cosine": float(sil), "davies_bouldin": float(db),
            "calinski_harabasz": float(ch),
            "n_used": int(mask.sum()), "n_clusters": int(n_clust)}


print("Computing KG-605 intrinsic metrics", file=sys.stderr)
t0 = time.time()
kg_intr = intrinsic(emb, labels_kg)
print(f"  KG: {kg_intr}", file=sys.stderr)
print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr)


# Binary-search HDBSCAN min_cluster_size to hit ~605
from hdbscan import HDBSCAN
TARGET = 605

print(f"Tuning flat HDBSCAN min_cluster_size to ~{TARGET}", file=sys.stderr)
lo, hi = 30, 500
best = None
attempts = []
while lo <= hi:
    mid = (lo + hi) // 2
    t0 = time.time()
    h = HDBSCAN(min_cluster_size=mid, metric="euclidean", core_dist_n_jobs=-1)
    labs = h.fit_predict(emb)
    n = len(set(labs)) - (1 if -1 in labs else 0)
    dt = time.time() - t0
    attempts.append({"min_cluster_size": mid, "n_clusters": n, "seconds": round(dt,1)})
    print(f"  mcs={mid}: {n} clusters ({dt:.1f}s)", file=sys.stderr)
    if best is None or abs(n - TARGET) < abs(best["n"] - TARGET):
        best = {"mcs": mid, "n": n, "labels": labs}
    if n > TARGET:
        lo = mid + 1
    else:
        hi = mid - 1
print(f"  best: mcs={best['mcs']}, n_clusters={best['n']}", file=sys.stderr)

print("Computing flat-fine intrinsic metrics", file=sys.stderr)
t0 = time.time()
flat_intr = intrinsic(emb, best["labels"])
print(f"  flat-fine: {flat_intr}", file=sys.stderr)
print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr)

delta = {}
for k in ("silhouette_cosine", "davies_bouldin", "calinski_harabasz"):
    f = flat_intr.get(k); h = kg_intr.get(k)
    delta[k] = float(h - f) if (f is not None and h is not None) else None

interpretation = []
if delta.get("davies_bouldin") is not None:
    sign = "lower" if delta["davies_bouldin"] < 0 else "higher"
    interpretation.append(f"KG-605 DB is {sign} than count-matched flat-{best['n']} "
                          f"(KG DB={kg_intr['davies_bouldin']:.2f} vs flat={flat_intr['davies_bouldin']:.2f}).")
if delta.get("calinski_harabasz") is not None:
    sign = "higher" if delta["calinski_harabasz"] > 0 else "lower"
    interpretation.append(f"KG-605 CH is {sign} than count-matched flat "
                          f"(KG CH={kg_intr['calinski_harabasz']:.2f} vs flat={flat_intr['calinski_harabasz']:.2f}).")
if delta.get("silhouette_cosine") is not None:
    sign = "higher" if delta["silhouette_cosine"] > 0 else "lower"
    interpretation.append(f"KG-605 silhouette is {sign} ({kg_intr['silhouette_cosine']:.3f} vs {flat_intr['silhouette_cosine']:.3f}).")

out = {
    "method": "Global HDBSCAN with binary-searched min_cluster_size to hit ~605 clusters, "
              "compared against KG-hierarchical 605 on identical 384-dim sentence embeddings "
              "of the actionable-class subset (n=117,958).",
    "kg_605": kg_intr,
    "flat_fine": {"min_cluster_size": best["mcs"], **flat_intr},
    "delta_kg_minus_flat": delta,
    "attempts": attempts,
    "interpretation": interpretation,
}
json.dump(out, open(OUT_PATH, "w"), indent=2)
print(json.dumps(out, indent=2))
print(f"Saved -> {OUT_PATH}", file=sys.stderr)
