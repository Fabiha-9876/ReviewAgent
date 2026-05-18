"""
LLM-as-judge Y/P/N purity audit on the 605 KG-hierarchical clusters.

Mirrors the Stage 2 flat audit (50-cluster lead-author manual audit, purity = 0.66)
but uses an LLM judge so it can be run reproducibly at any cluster count.

Methodology
-----------
For each sampled cluster:
  1. Take 5 representative_reviews (already stored in the cluster JSON).
  2. Send to the LLM with a strict rubric:
       Y = all 5 reviews share the same sub-theme within the cluster's aspect
       P = 3-4 of 5 share the sub-theme (partial)
       N = fewer than 3 share the sub-theme (incoherent)
  3. Record verdict + one-sentence justification.
Compute weighted purity = (|Y| + 0.5 * |P|) / n.

Sampling
--------
Default: stratified random sample of 50 clusters (matching the flat audit n).
Use --n to change, --all to audit all 605.

Cost
----
~50 prompts at ~300 tokens each. Claude Opus 4.7 ≈ $0.10 total at current pricing.

Usage
-----
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/audit_hierarchical_cluster_purity_llm.py
  # writes -> data/processed/kg_hierarchical/llm_judge_purity_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic

CLUSTERS_PATH = Path("data/processed/kg_hierarchical/hierarchical_clusters.json")
OUT_PATH = Path("data/processed/kg_hierarchical/llm_judge_purity_audit.json")
MODEL = "claude-opus-4-7"
SEED = 42

SYSTEM_PROMPT = """You are an expert software-engineering annotator auditing the coherence of an app-review cluster.

You will be given 5 review excerpts and the cluster's aspect label. Judge whether the 5 reviews share the same SUB-THEME within that aspect.

Output exactly one of: Y, P, N
  Y = all 5 share the sub-theme
  P = 3 or 4 of 5 share the sub-theme (partial coherence)
  N = fewer than 3 share the sub-theme (incoherent)

Then on a new line, give a one-sentence justification (under 25 words).

Format your response EXACTLY as:
VERDICT: <Y|P|N>
REASON: <one sentence>
"""

USER_TEMPLATE = """Cluster aspect: {aspect}
Cluster sub-aspect ID: {cluster_id}
Cluster issue-type (V5 classifier majority): {issue_type}

5 representative reviews:
1. {r1}
2. {r2}
3. {r3}
4. {r4}
5. {r5}

What is your Y/P/N verdict on whether these 5 reviews share the same sub-theme within the aspect "{aspect}"?
"""


def parse_verdict(text: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    verdict = ""
    reason = ""
    for ln in lines:
        if ln.upper().startswith("VERDICT:"):
            v = ln.split(":", 1)[1].strip().upper()
            if v.startswith("Y"):
                verdict = "Y"
            elif v.startswith("P"):
                verdict = "P"
            elif v.startswith("N"):
                verdict = "N"
        elif ln.upper().startswith("REASON:"):
            reason = ln.split(":", 1)[1].strip()
    return verdict, reason


def stratified_sample(clusters: list[dict], n: int, seed: int) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in clusters:
        by_type[c.get("issue_type", "unknown")].append(c)
    rng = random.Random(seed)
    types = sorted(by_type)
    quota = max(1, n // len(types))
    sample: list[dict] = []
    for t in types:
        pool = by_type[t]
        rng.shuffle(pool)
        sample.extend(pool[:quota])
    rng.shuffle(sample)
    return sample[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="sample size (default 50, matches flat audit)")
    ap.add_argument("--all", action="store_true", help="audit all 605 clusters")
    ap.add_argument("--dry-run", action="store_true", help="print sample without API calls")
    args = ap.parse_args()

    if not CLUSTERS_PATH.exists():
        print(f"Missing input: {CLUSTERS_PATH}", file=sys.stderr)
        return 1

    clusters = json.load(open(CLUSTERS_PATH))
    print(f"Loaded {len(clusters)} hierarchical clusters", file=sys.stderr)

    sample = clusters if args.all else stratified_sample(clusters, args.n, SEED)
    print(f"Auditing {len(sample)} clusters (stratified by issue_type, seed={SEED})", file=sys.stderr)

    if args.dry_run:
        for c in sample[:5]:
            print(c["cluster_id"], c["aspect"], c["issue_type"])
        return 0

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=key)
    results = []
    counts = {"Y": 0, "P": 0, "N": 0, "error": 0}

    for i, c in enumerate(sample, 1):
        reps = c.get("representative_reviews", [])
        if len(reps) < 5:
            counts["error"] += 1
            continue
        prompt = USER_TEMPLATE.format(
            aspect=c.get("aspect", ""),
            cluster_id=c.get("cluster_id", ""),
            issue_type=c.get("issue_type", ""),
            r1=reps[0][:400], r2=reps[1][:400], r3=reps[2][:400], r4=reps[3][:400], r5=reps[4][:400],
        )
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=120,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
        except Exception as e:
            print(f"  [{i}/{len(sample)}] API error on {c['cluster_id']}: {e}", file=sys.stderr)
            counts["error"] += 1
            results.append({"cluster_id": c["cluster_id"], "verdict": "", "reason": str(e), "error": True})
            time.sleep(2)
            continue
        verdict, reason = parse_verdict(text)
        if verdict in counts:
            counts[verdict] += 1
        else:
            counts["error"] += 1
        results.append({
            "cluster_id": c["cluster_id"],
            "aspect": c.get("aspect"),
            "issue_type": c.get("issue_type"),
            "review_count": c.get("review_count"),
            "verdict": verdict,
            "reason": reason,
            "raw": text,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(sample)}] running tally Y={counts['Y']} P={counts['P']} N={counts['N']} err={counts['error']}", file=sys.stderr)
        time.sleep(0.5)

    n_judged = counts["Y"] + counts["P"] + counts["N"]
    weighted_purity = ((counts["Y"] + 0.5 * counts["P"]) / n_judged) if n_judged else 0.0

    per_class: dict[str, dict] = defaultdict(lambda: {"Y": 0, "P": 0, "N": 0, "n": 0})
    for r in results:
        v = r.get("verdict")
        if v in ("Y", "P", "N"):
            t = r.get("issue_type", "unknown") or "unknown"
            per_class[t][v] += 1
            per_class[t]["n"] += 1

    out = {
        "method": "llm_as_judge_y_p_n_audit",
        "judge_model": MODEL,
        "n_sampled": len(sample),
        "n_judged": n_judged,
        "n_errors": counts["error"],
        "overall_counts": {k: counts[k] for k in ("Y", "P", "N")},
        "weighted_purity": weighted_purity,
        "per_class": {t: dict(d) for t, d in per_class.items()},
        "sample_seed": SEED,
        "comparison_with_flat": {
            "flat_50_cluster_lead_author_audit": 0.66,
            "hierarchical_n_cluster_llm_judge": weighted_purity,
            "n_hierarchical": n_judged,
        },
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nWeighted Y/P/N purity (hierarchical): {weighted_purity:.3f} on {n_judged} clusters", file=sys.stderr)
    print(f"  Y={counts['Y']}  P={counts['P']}  N={counts['N']}  err={counts['error']}", file=sys.stderr)
    print(f"Saved -> {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
