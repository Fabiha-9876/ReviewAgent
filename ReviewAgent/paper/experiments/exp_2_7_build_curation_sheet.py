"""
Tier 2.7: Build the 20-cluster curation sheet for the inter-curator check.

This produces a CSV the non-author rater fills in. Pair it with
`exp_2_7_curation_rubric.md`. After the rater returns the CSV,
`exp_2_7_score_curation.py` (next step, not yet written — wire it in
once you have the returned CSV) computes Cohen's κ between the two
raters' KEEP/RENAME/MERGE/SPLIT labels.

Run:
    cd /Users/fabihajalal/Desktop/Review\\ Agent/ReviewAgent
    python paper/experiments/exp_2_7_build_curation_sheet.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent")
INPUT = REPO_ROOT / "data/processed/issue_specs/sample_100_clusters.json"
OUTPUT = REPO_ROOT / "paper/experiments/curation_sheet_20clusters.csv"

SEED = 42
N_SAMPLE = 20


def main() -> None:
    with INPUT.open() as f:
        clusters = json.load(f)
    if len(clusters) < N_SAMPLE:
        raise SystemExit(
            f"need at least {N_SAMPLE} clusters in {INPUT}, got {len(clusters)}"
        )

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(clusters), size=N_SAMPLE, replace=False)

    rows = []
    for idx in pick:
        c = clusters[int(idx)]
        reviews = c.get("reviews") or c.get("sample_reviews") or []
        review_texts = []
        for r in reviews[:5]:
            t = r.get("text") if isinstance(r, dict) else str(r)
            review_texts.append((t or "").strip()[:300])
        rows.append({
            "cluster_id": c.get("cluster_id"),
            "auto_name": c.get("auto_name") or c.get("name") or "",
            "review_1": review_texts[0] if len(review_texts) > 0 else "",
            "review_2": review_texts[1] if len(review_texts) > 1 else "",
            "review_3": review_texts[2] if len(review_texts) > 2 else "",
            "review_4": review_texts[3] if len(review_texts) > 3 else "",
            "review_5": review_texts[4] if len(review_texts) > 4 else "",
            "operation_KEEP_RENAME_MERGE_SPLIT": "",
            "new_name_if_RENAME": "",
            "merge_target_cluster_id_if_MERGE": "",
            "split_note_if_SPLIT": "",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(
        f"Hand this CSV plus exp_2_7_curation_rubric.md to a non-author rater. "
        "After they return it, write exp_2_7_score_curation.py to compute "
        "Cohen's kappa between the two raters' operation labels."
    )


if __name__ == "__main__":
    main()
