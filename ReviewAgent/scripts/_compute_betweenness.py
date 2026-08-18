"""Compute betweenness centrality on the KG (closing C3 spec).

Reuses the KG construction logic from run_kg_hierarchical_clustering.py
but only runs the KG layer (skips hierarchical clustering), so it's fast.
Updates kg_stats.json in place with top_aspects_by_betweenness alongside
the existing top_aspects_by_pagerank.
"""
from __future__ import annotations
import json, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.common.schemas import ReviewObject, AspectSentiment, ExtractedEntities
from src.stage2.kg_builder import ReviewKnowledgeGraph

N_SAMPLE = 10_000
SEED = 42
ACTIONABLE = ["bug_report", "feature_request", "performance", "usability", "compatibility"]
OUT = BASE / "data/processed/kg_hierarchical/kg_stats.json"

print("[1/3] Loading data", file=sys.stderr)
with open(BASE / "data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json") as f:
    all_reviews = json.load(f)
with open(BASE / "data/processed/aspects_heuristic/aspects_per_review.json") as f:
    aspects_by_idx = json.load(f)

eligible = [(i, r) for i, r in enumerate(all_reviews)
            if r.get("v5_label") in ACTIONABLE and str(i) in aspects_by_idx]
by_class = defaultdict(list)
for i, r in eligible: by_class[r["v5_label"]].append((i, r))
rng = random.Random(SEED)
target_per = N_SAMPLE // len(ACTIONABLE)
sampled = []
for cls in ACTIONABLE:
    pool = by_class[cls]
    rng.shuffle(pool)
    sampled.extend(pool[:target_per])
rng.shuffle(sampled)
print(f"  sampled {len(sampled):,}", file=sys.stderr)

print("[2/3] Rebuilding KG", file=sys.stderr)
t0 = time.time()
reviews = []
for i, r in sampled:
    asp_strings = aspects_by_idx.get(str(i), [])[:5]
    aspects = [AspectSentiment(aspect=a, sentiment="neutral", intensity=0.5) for a in asp_strings if a]
    entities = ExtractedEntities(devices=[], os_versions=[], app_versions=[], screens=[])
    try: ts = datetime.fromisoformat(r.get("timestamp", "2018-01-01"))
    except Exception: ts = datetime(2018, 1, 1)
    reviews.append(ReviewObject(
        review_id=f"r_{i}", text=r["text"][:500],
        rating=int(r.get("rating") or 3),
        app_id=r.get("app_id", "unknown"), timestamp=ts,
        labels=[r["v5_label"]], aspects=aspects, entities=entities,
    ))

kg = ReviewKnowledgeGraph()
kg.add_reviews(reviews)
print(f"  KG: {kg.graph.number_of_nodes():,} nodes, {kg.graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)", file=sys.stderr)

aspect_nodes = kg.get_aspect_nodes()
print(f"  aspect nodes: {len(aspect_nodes):,}", file=sys.stderr)

print("[3/3] Computing PageRank + betweenness", file=sys.stderr)
t0 = time.time()
pr = kg.compute_pagerank()
print(f"  PageRank done in {time.time()-t0:.1f}s", file=sys.stderr)

t0 = time.time()
# Betweenness on undirected projection (matches PageRank treatment in kg_builder)
undirected = kg.graph.to_undirected()
# Use approximation for tractability on 19K-node graph
bc = nx.betweenness_centrality(undirected, k=min(500, undirected.number_of_nodes()),
                                seed=SEED, normalized=True)
print(f"  Betweenness done in {time.time()-t0:.1f}s (k=500 sample approx)", file=sys.stderr)

# Top aspects by each centrality
aspect_pr = sorted([(n, pr[n]) for n in aspect_nodes], key=lambda x: -x[1])
aspect_bc = sorted([(n, bc[n]) for n in aspect_nodes], key=lambda x: -x[1])

print("\nTop-10 aspects by PageRank:", file=sys.stderr)
for n, s in aspect_pr[:10]:
    print(f"  {s:.5f}  {n}", file=sys.stderr)
print("\nTop-10 aspects by Betweenness:", file=sys.stderr)
for n, s in aspect_bc[:10]:
    print(f"  {s:.5f}  {n}", file=sys.stderr)

# Update kg_stats.json: keep existing fields, add betweenness ranking
stats = json.load(open(OUT))
stats["top_aspects_by_betweenness"] = [
    {"aspect": n, "betweenness": round(s, 6),
     "n_reviews": len(kg.get_reviews_for_aspect(n))}
    for n, s in aspect_bc[:50]
]
stats["betweenness_method"] = f"networkx betweenness_centrality on undirected projection, k=500 sampling, seed={SEED}"
json.dump(stats, open(OUT, "w"), indent=2)
print(f"\nSaved updated -> {OUT}", file=sys.stderr)
