"""A1b rep-based: compare KG-605 representative clustering vs flat HDBSCAN on the
same 1815-text representative subset.

Methodology
-----------
1. Take the 3 representative texts per KG sub-cluster (605 * 3 = 1815 texts).
2. Embed them fresh with all-MiniLM-L6-v2 (same model used in the pipeline).
3. KG side: each rep inherits its parent sub-cluster_id as label (~605 clusters,
   exactly 3 reviews per cluster).
4. Flat side: run flat HDBSCAN on the same 1815 embeddings; tune min_cluster_size
   downward until it produces a comparable cluster count (target ~605).
5. Compute silhouette / Davies-Bouldin / Calinski-Harabasz for both label vectors
   on the SAME embeddings. Report deltas.

This matches the implicit subset that the paper's existing KG-605 intrinsic
metrics were computed on (mean_size = 3.0 = the rep count).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
KG_PATH = BASE / "data/processed/kg_hierarchical/hierarchical_clusters_full.json"
OUT_PATH = BASE / "data/processed/ablations/a1b_repbased.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading KG hierarchical clusters", file=sys.stderr)
hier = json.load(open(KG_PATH))
print(f"  {len(hier)} sub-clusters", file=sys.stderr)

texts, labels_kg = [], []
for cid, c in enumerate(hier):
    for rep in c.get("representative_reviews", []):
        texts.append(rep)
        labels_kg.append(cid)
labels_kg = np.array(labels_kg, dtype=np.int64)
print(f"  total reps: {len(texts)}, n_clusters: {len(set(labels_kg))}", file=sys.stderr)

# Embed
import torch
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}", file=sys.stderr)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
t0 = time.time()
emb = model.encode(texts, batch_size=64, show_progress_bar=False,
                   convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
print(f"  embedded in {time.time()-t0:.1f}s, shape={emb.shape}", file=sys.stderr)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def intrinsic(emb, labels, name=""):
    mask = labels != -1
    n_clust = len(set(labels[mask]))
    if mask.sum() < 50 or n_clust < 2:
        return {"silhouette_cosine": None, "davies_bouldin": None, "calinski_harabasz": None,
                "n_used": int(mask.sum()), "n_clusters": int(n_clust)}
    sil = silhouette_score(emb[mask], labels[mask], metric="cosine", random_state=42)
    db = davies_bouldin_score(emb[mask], labels[mask])
    ch = calinski_harabasz_score(emb[mask], labels[mask])
    return {"silhouette_cosine": float(sil), "davies_bouldin": float(db),
            "calinski_harabasz": float(ch),
            "n_used": int(mask.sum()), "n_clusters": int(n_clust)}


print("KG-605 intrinsic on rep subset", file=sys.stderr)
kg_intr = intrinsic(emb, labels_kg, "KG")
print(f"  KG: {kg_intr}", file=sys.stderr)


# Flat HDBSCAN on the same 1815 reps. Tune to give a comparable n_clusters.
from hdbscan import HDBSCAN
TARGET = 605
print(f"Tuning flat HDBSCAN on rep subset to ~{TARGET}", file=sys.stderr)

# With 1815 points and target 605, mean_size~3 -> min_cluster_size should be small
attempts = []
best = None
for mcs in [2, 3, 4, 5, 7, 10, 15]:
    h = HDBSCAN(min_cluster_size=mcs, metric="euclidean")
    labs = h.fit_predict(emb)
    n = len(set(labs)) - (1 if -1 in labs else 0)
    nz = int((labs != -1).sum())
    attempts.append({"min_cluster_size": mcs, "n_clusters": n, "non_noise": nz})
    print(f"  mcs={mcs}: {n} clusters, {nz}/{len(emb)} non-noise", file=sys.stderr)
    if best is None or abs(n - TARGET) < abs(best["n"] - TARGET):
        best = {"mcs": mcs, "n": n, "labels": labs}

print(f"  best: mcs={best['mcs']}, n_clusters={best['n']}", file=sys.stderr)
flat_intr = intrinsic(emb, best["labels"], "flat")
print(f"  flat: {flat_intr}", file=sys.stderr)


# Also a "natural" flat clustering at the same noise tolerance the KG implied (0 noise)
# by using AgglomerativeClustering with n_clusters=605, since HDBSCAN may keep most as noise.
from sklearn.cluster import AgglomerativeClustering
print("Agglomerative-605 on rep subset (forced exact count)", file=sys.stderr)
ag = AgglomerativeClustering(n_clusters=605, metric="cosine", linkage="average")
labs_ag = ag.fit_predict(emb)
ag_intr = intrinsic(emb, labs_ag, "agg")
print(f"  agg: {ag_intr}", file=sys.stderr)


delta_kg_vs_flat = {}
for k in ("silhouette_cosine", "davies_bouldin", "calinski_harabasz"):
    f = flat_intr.get(k); h = kg_intr.get(k)
    delta_kg_vs_flat[k] = float(h - f) if (f is not None and h is not None) else None

delta_kg_vs_agg = {}
for k in ("silhouette_cosine", "davies_bouldin", "calinski_harabasz"):
    a = ag_intr.get(k); h = kg_intr.get(k)
    delta_kg_vs_agg[k] = float(h - a) if (a is not None and h is not None) else None

interp = []
if delta_kg_vs_agg.get("davies_bouldin") is not None:
    sign = "lower (better)" if delta_kg_vs_agg["davies_bouldin"] < 0 else "higher (worse)"
    interp.append(f"KG-605 DB is {sign} vs count-matched Agglomerative-605 on the same reps "
                  f"(KG DB={kg_intr['davies_bouldin']:.2f} vs agg={ag_intr['davies_bouldin']:.2f}).")
if delta_kg_vs_agg.get("calinski_harabasz") is not None:
    sign = "higher (better)" if delta_kg_vs_agg["calinski_harabasz"] > 0 else "lower (worse)"
    interp.append(f"KG-605 CH is {sign} vs Agglomerative-605 "
                  f"(KG CH={kg_intr['calinski_harabasz']:.2f} vs agg={ag_intr['calinski_harabasz']:.2f}).")
if delta_kg_vs_agg.get("silhouette_cosine") is not None:
    interp.append(f"KG-605 silhouette={kg_intr['silhouette_cosine']:.3f} vs agg={ag_intr['silhouette_cosine']:.3f}.")

out = {
    "method": ("Count-controlled comparison on the 1,815 representative reviews "
               "(3 reps per KG sub-cluster, 605 sub-clusters). KG labels inherited "
               "from aspect/sub_cluster assignment; flat = HDBSCAN tuned to give "
               "comparable n_clusters; agglomerative = forced 605 clusters."),
    "n_reps": int(len(texts)),
    "kg_605": kg_intr,
    "flat_hdbscan_best": {"min_cluster_size": best["mcs"], **flat_intr},
    "agglomerative_605": ag_intr,
    "delta_kg_minus_flat": delta_kg_vs_flat,
    "delta_kg_minus_agglomerative": delta_kg_vs_agg,
    "hdbscan_attempts": attempts,
    "interpretation": interp,
}
json.dump(out, open(OUT_PATH, "w"), indent=2)
print(json.dumps(out, indent=2))
print(f"\nSaved -> {OUT_PATH}", file=sys.stderr)
