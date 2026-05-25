"""
Reproducible PageRank-coverage computation for Stage 2 (Section 5.4).

Rebuilds the exact KG review sample used in run_kg_hierarchical_clustering.py
(SEED=42, N_SAMPLE=10000, 5 actionable classes, <=5 aspects/review), then reports:

  1. top-K aspects by PageRank,
  2. *deduplicated* coverage = |unique reviews touching any top-K aspect| / |reviews|,

both WITHOUT and WITH singular/plural aspect merging (e.g. ad/ads -> ad).

The dedup number is the honest "top-K aspects appear in X% of reviews"; the naive
sum-of-per-aspect-counts double-counts reviews that mention several top aspects.
"""
import json
import random
from collections import defaultdict

import networkx as nx

N_SAMPLE = 10_000
SEED = 42
ACTIONABLE = ["bug_report", "feature_request", "performance", "usability", "compatibility"]

BASE = "data/processed"
all_reviews = json.load(open(f"{BASE}/rrgen_v5_relabeled/rrgen_v5_relabeled.json"))
aspects_by_idx = json.load(open(f"{BASE}/aspects_heuristic/aspects_per_review.json"))


def sample_reviews():
    rng = random.Random(SEED)
    eligible = [(i, r) for i, r in enumerate(all_reviews)
                if r.get("v5_label") in ACTIONABLE and str(i) in aspects_by_idx]
    by_class = defaultdict(list)
    for i, r in eligible:
        by_class[r["v5_label"]].append((i, r))
    target_per = N_SAMPLE // len(ACTIONABLE)
    sampled = []
    for cls in ACTIONABLE:
        pool = by_class[cls]
        rng.shuffle(pool)
        sampled.extend(pool[:target_per])
    rng.shuffle(sampled)
    return sampled


def build_and_report(sampled, normalize):
    # plural->singular merge map (only when both forms appear), built on the fly
    raw_names = set()
    for i, _ in sampled:
        for a in aspects_by_idx.get(str(i), [])[:5]:
            if a:
                raw_names.add(a.strip().lower())

    def norm(a):
        a = a.strip().lower()
        if not normalize:
            return a
        if a == "ads":
            return "ad"
        if a.endswith("s") and a[:-1] in raw_names:
            return a[:-1]
        return a

    g = nx.DiGraph()
    for i, _ in sampled:
        rid = f"r_{i}"
        g.add_node(rid, node_type="review")
        for a in aspects_by_idx.get(str(i), [])[:5]:
            if not a:
                continue
            aid = f"aspect:{norm(a)}"
            if not g.has_node(aid):
                g.add_node(aid, node_type="aspect")
            g.add_edge(rid, aid)

    review_nodes = [n for n, d in g.nodes(data=True) if d["node_type"] == "review"]
    aspect_nodes = [n for n, d in g.nodes(data=True) if d["node_type"] == "aspect"]
    N = len(review_nodes)
    pr = nx.pagerank(g)
    ranked = sorted(aspect_nodes, key=lambda n: -pr[n])

    def reviews_for(aid):
        return {p for p in g.predecessors(aid)}

    def dedup_cov(k):
        u = set()
        for aid in ranked[:k]:
            u |= reviews_for(aid)
        return len(u), 100.0 * len(u) / N

    naive5 = sum(len(reviews_for(a)) for a in ranked[:5])
    naive10 = sum(len(reviews_for(a)) for a in ranked[:10])
    d5n, d5p = dedup_cov(5)
    d10n, d10p = dedup_cov(10)

    tag = "WITH ad/ads + plural merge" if normalize else "RAW (no merge, reproduces kg_stats)"
    print(f"\n===== {tag} =====")
    print(f"reviews={N}  aspects={len(aspect_nodes)}")
    print("top-10 aspects by PageRank:")
    for n in ranked[:10]:
        print(f"   {pr[n]:.5f}  {len(reviews_for(n)):>4} rev  {n}")
    print(f"NAIVE   top5 {naive5} ({100*naive5/N:.1f}%)   top10 {naive10} ({100*naive10/N:.1f}%)")
    print(f"DEDUP   top5 {d5n} ({d5p:.1f}%)   top10 {d10n} ({d10p:.1f}%)")
    return {"reviews": N, "aspects": len(aspect_nodes),
            "dedup_top5_pct": round(d5p, 1), "dedup_top10_pct": round(d10p, 1),
            "naive_top5_pct": round(100 * naive5 / N, 1),
            "top10": [n.replace("aspect:", "") for n in ranked[:10]]}


sampled = sample_reviews()
print(f"sampled {len(sampled)} reviews")
raw = build_and_report(sampled, normalize=False)
merged = build_and_report(sampled, normalize=True)
json.dump({"raw": raw, "merged": merged},
          open(f"{BASE}/kg_hierarchical/aspect_coverage.json", "w"), indent=2)
print("\nwrote data/processed/kg_hierarchical/aspect_coverage.json")
