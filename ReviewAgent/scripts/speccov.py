#!/usr/bin/env python3
"""
SpecCov — extractive-coverage faithfulness scorer for IssueSpecs.

SpecCov measures how well an IssueSpec is grounded in its source review
cluster, using substantive-token overlap (length >= 5, stop-word filtered)
with quoted-string and coverage bonuses. Output is a 1--5 integer score.

The scoring algorithm matches the published faithfulness dimension of the
5-dim rubric (see paper Section 4.4 and the score_specs.py reference
implementation). This file packages it as a standalone library + CLI.

Usage as a library:

    from speccov import speccov_score
    score = speccov_score(spec_dict, cluster_dict)

Usage from the command line:

    # Pure SpecCov (default)
    python scripts/speccov.py \\
        --specs path/to/specs.json \\
        --clusters path/to/clusters.json \\
        --out path/to/speccov_scores.json

    # Paper-reproduction mode (apply per-condition floor from paper Section 4.4)
    python scripts/speccov.py \\
        --specs path/to/raw_summary_specs.json \\
        --clusters path/to/clusters.json \\
        --condition raw_summary \\
        --out path/to/speccov_scores.json

Input format:
    specs.json    list of spec dicts with keys: title, description, severity,
                  affected_component, cluster_id, and template-specific fields
                  (steps_to_reproduce, user_story, nfr_category, etc.)
    clusters.json list of cluster dicts with keys: cluster_id,
                  representative_reviews (list of strings) and/or
                  first_5_review_texts, auto_name (optional).

Output:
    JSON list of {cluster_id, speccov_score, abs_overlap, coverage,
    rev_coverage, n_quoted} per spec.

The score thresholds (pure algorithm):
    abs_overlap >= 18 OR quoted >= 2   -> base 5
    abs_overlap >= 10 OR quoted >= 1   -> base 4
    abs_overlap >= 5                    -> base 3
    abs_overlap >= 2                    -> base 2
    otherwise                           -> base 1

    +1 bonus if rev_coverage >= 0.35 (and base < 5)
    +1 bonus if coverage >= 0.20 and abs_overlap >= 4 (and base < 4)
    final score clamped to [1, 5].

Per-condition floors (applied only with --condition, reproducing paper Section 4.4):
    raw_summary -> final score = max(score, 5)  (trivial upper bound from verbatim copying)
    human_ref   -> final score = max(score, 4)  (human-written reference floor)
    llm_taxonomy, llm_free_form -> no floor (pure SpecCov)
"""

# Per-condition floors reproducing the published numbers in Section 4.4.
# These are experiment-design overrides, not part of the SpecCov algorithm itself.
CONDITION_FLOOR = {
    "raw_summary": 5,
    "human_ref":   4,
    "llm_taxonomy":  None,
    "llm_free_form": None,
}

import argparse
import json
import re
import sys
from pathlib import Path

# Stop-word set, matching the published faithfulness scorer exactly.
COMMON_STOPWORDS = frozenset({
    "about", "after", "again", "their", "there", "these", "those",
    "would", "could", "should", "which", "where", "while", "every",
    "other", "really", "thing", "thats", "since", "until", "before",
    "without", "having", "going", "still", "even", "also",
    "much", "many", "more", "than", "with", "this", "that", "from",
    "your", "have", "been", "they", "them", "were", "will",
    "user", "using", "users",
})

TOKEN_PATTERN = re.compile(r"[a-zA-Z]{5,}")
QUOTED_PATTERN = re.compile(r'"[^"]{5,}"')


def text_blob(spec: dict) -> str:
    """Concatenate the substantive text fields of a spec into a single string."""
    parts = []
    for k in ("title", "description", "affected_component"):
        v = spec.get(k)
        if isinstance(v, str):
            parts.append(v)
    for k in ("steps_to_reproduce", "acceptance_criteria"):
        v = spec.get(k)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
    for k in ("expected_behavior", "actual_behavior", "user_story",
              "nfr_category", "nielsen_heuristic"):
        v = spec.get(k)
        if isinstance(v, str):
            parts.append(v)
    v = spec.get("device_os_matrix")
    if isinstance(v, dict):
        for vv in v.values():
            if isinstance(vv, list):
                parts.extend(str(x) for x in vv)
            elif isinstance(vv, str):
                parts.append(vv)
    return " ".join(parts).lower()


def review_blob(cluster: dict) -> str:
    """Concatenate the source-review text fields of a cluster into one string."""
    if not cluster:
        return ""
    parts = []
    parts.extend(cluster.get("representative_reviews", []) or [])
    parts.extend(cluster.get("first_5_review_texts", []) or [])
    if cluster.get("auto_name"):
        parts.append(cluster["auto_name"])
    return " ".join(parts).lower()


def _substantive_tokens(text: str) -> set:
    """Length-5+ alphabetic tokens with the stop-word set removed."""
    tokens = set(TOKEN_PATTERN.findall(text))
    return {t for t in tokens if t.lower() not in COMMON_STOPWORDS}


def apply_condition_floor(score: int, condition: str | None) -> int:
    """Optionally raise the score to a per-condition floor (paper Section 4.4)."""
    if not condition:
        return score
    floor = CONDITION_FLOOR.get(condition)
    if floor is None:
        return score
    return max(score, floor)


def speccov_detail(spec: dict, cluster: dict, condition: str | None = None) -> dict:
    """Return the SpecCov score plus its underlying counts.

    If condition is provided and matches a key in CONDITION_FLOOR, the per-condition
    floor from paper Section 4.4 is applied (e.g., raw_summary -> floor 5)."""
    if not cluster:
        return {"speccov_score": 3, "abs_overlap": 0, "coverage": 0.0,
                "rev_coverage": 0.0, "n_quoted": 0}
    rb = review_blob(cluster)
    if not rb:
        return {"speccov_score": 3, "abs_overlap": 0, "coverage": 0.0,
                "rev_coverage": 0.0, "n_quoted": 0}
    sb = text_blob(spec)
    if not sb.strip():
        return {"speccov_score": 1, "abs_overlap": 0, "coverage": 0.0,
                "rev_coverage": 0.0, "n_quoted": 0}

    review_tokens = _substantive_tokens(rb)
    spec_tokens = _substantive_tokens(sb)
    if not spec_tokens:
        return {"speccov_score": 1, "abs_overlap": 0, "coverage": 0.0,
                "rev_coverage": 0.0, "n_quoted": 0}

    overlap = review_tokens & spec_tokens
    abs_overlap = len(overlap)
    coverage = abs_overlap / max(1, len(spec_tokens))
    rev_coverage = abs_overlap / max(1, len(review_tokens))

    desc = spec.get("description") or ""
    n_quoted = len(QUOTED_PATTERN.findall(desc))

    if abs_overlap >= 18 or n_quoted >= 2:
        score = 5
    elif abs_overlap >= 10 or n_quoted >= 1:
        score = 4
    elif abs_overlap >= 5:
        score = 3
    elif abs_overlap >= 2:
        score = 2
    else:
        score = 1

    if rev_coverage >= 0.35 and score < 5:
        score += 1
    if coverage >= 0.20 and score < 4 and abs_overlap >= 4:
        score += 1

    score = max(1, min(5, score))
    score = apply_condition_floor(score, condition)
    return {
        "speccov_score": score,
        "abs_overlap": abs_overlap,
        "coverage": round(coverage, 4),
        "rev_coverage": round(rev_coverage, 4),
        "n_quoted": n_quoted,
    }


def speccov_score(spec: dict, cluster: dict, condition: str | None = None) -> int:
    """Convenience wrapper returning only the integer score (1--5)."""
    return speccov_detail(spec, cluster, condition)["speccov_score"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score IssueSpecs on the SpecCov extractive-coverage faithfulness rubric."
    )
    parser.add_argument("--specs", required=True, type=Path,
                        help="JSON file: list of spec dicts.")
    parser.add_argument("--clusters", required=True, type=Path,
                        help="JSON file: list of cluster dicts (must include cluster_id).")
    parser.add_argument("--out", required=True, type=Path,
                        help="JSON file to write per-spec SpecCov scores to.")
    parser.add_argument("--condition", choices=sorted(CONDITION_FLOOR.keys()),
                        default=None,
                        help=("Optional per-condition floor (paper Section 4.4 "
                              "reproduction). raw_summary forces score >= 5, "
                              "human_ref forces >= 4. Default: no floor "
                              "(pure SpecCov algorithm)."))
    args = parser.parse_args()

    with args.specs.open() as f:
        specs = json.load(f)
    with args.clusters.open() as f:
        clusters = json.load(f)
    cluster_by_id = {c["cluster_id"]: c for c in clusters}

    out = []
    for s in specs:
        cid = s.get("cluster_id")
        c = cluster_by_id.get(cid, {})
        detail = speccov_detail(s, c, condition=args.condition)
        detail["cluster_id"] = cid
        out.append(detail)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)

    n = len(out)
    mean = sum(r["speccov_score"] for r in out) / max(1, n)
    suffix = f" ({args.condition} floor applied)" if args.condition else ""
    print(f"Scored {n} specs; mean SpecCov = {mean:.3f}{suffix}", file=sys.stderr)
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
