"""
Compute 3-rater inter-rater statistics on the full 400-row Stage 4b evaluation.

Raters:
  1. Lead author     -> data/processed/responses/pairwise_ratings_human.json (n=400)
  2. Labmate (R2)    -> data/processed/responses/labmate_inter_rater_aggregate.json (n=30; aggregate only)
  3. LLM judge       -> data/processed/responses/llm_as_judge_full_400.json (run by scripts/llm_as_judge_full_400.py)

Outputs (data/processed/responses/three_rater_agreement.json):
  - Pairwise Cohen's kappa for each pair of raters (quality, helpful, preference)
  - 3-rater Krippendorff's alpha (ordinal for quality, nominal for helpful & preference)
  - Within-1 agreement % for quality
  - Directional check: does each rater independently prefer `reviewagent_full`?

Requires: pip install krippendorff scikit-learn
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import krippendorff
except ImportError:
    print("pip install krippendorff", file=sys.stderr); sys.exit(1)
from sklearn.metrics import cohen_kappa_score

LEAD = Path("data/processed/responses/pairwise_ratings_human.json")
LABMATE_AGG = Path("data/processed/responses/labmate_inter_rater_aggregate.json")
LLM = Path("data/processed/responses/llm_as_judge_full_400.json")
OUT = Path("data/processed/responses/three_rater_agreement.json")


def main() -> int:
    if not LEAD.exists() or not LLM.exists():
        print(f"Missing: lead={LEAD.exists()} llm={LLM.exists()}", file=sys.stderr); return 1

    lead = json.load(open(LEAD))
    llm = json.load(open(LLM)).get("results", [])
    print(f"Lead n={len(lead)}, LLM n={len(llm)}", file=sys.stderr)

    # Align by review_index
    lead_by_idx = {r["review_index"]: r for r in lead}
    llm_by_idx = {r["review_index"]: r for r in llm}
    common = sorted(set(lead_by_idx) & set(llm_by_idx))
    print(f"Common review_index count: {len(common)}", file=sys.stderr)

    lead_q_full, llm_q_full = [], []
    lead_q_no, llm_q_no = [], []
    lead_help_full, llm_help_full = [], []
    lead_help_no, llm_help_no = [], []
    lead_pref, llm_pref = [], []

    for idx in common:
        L = lead_by_idx[idx]; M = llm_by_idx[idx]
        # Lead: A_condition / B_condition tell us which side is which
        lead_full_q = L["A_quality"] if L["A_condition"] == "with_spec" or L["A_condition"] == "full" else L["B_quality"]
        lead_no_q = L["B_quality"] if L["A_condition"] == "with_spec" or L["A_condition"] == "full" else L["A_quality"]
        lead_full_h = L["A_helpful"] if L["A_condition"] == "with_spec" or L["A_condition"] == "full" else L["B_helpful"]
        lead_no_h = L["B_helpful"] if L["A_condition"] == "with_spec" or L["A_condition"] == "full" else L["A_helpful"]
        lead_p = "full" if (L["preferred"] == "A" and L["A_condition"] in ("with_spec", "full")) or \
                            (L["preferred"] == "B" and L["B_condition"] in ("with_spec", "full")) else "no_spec"
        if M.get("full_quality") is None or M.get("no_spec_quality") is None:
            continue
        lead_q_full.append(lead_full_q); llm_q_full.append(M["full_quality"])
        lead_q_no.append(lead_no_q);     llm_q_no.append(M["no_spec_quality"])
        lead_help_full.append(lead_full_h); llm_help_full.append(M["full_helpful"])
        lead_help_no.append(lead_no_h);     llm_help_no.append(M["no_spec_helpful"])
        lead_pref.append(lead_p);         llm_pref.append(M.get("preference_condition", "tie"))

    def kappa(a, b, weights=None):
        if weights:
            return cohen_kappa_score(a, b, weights=weights)
        return cohen_kappa_score(a, b)

    def within_1(a, b):
        return sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / max(1, len(a))

    res = {
        "n_common": len(lead_q_full),
        "lead_vs_llm": {
            "quality_full_kappa_weighted": kappa(lead_q_full, llm_q_full, weights="quadratic"),
            "quality_full_within1": within_1(lead_q_full, llm_q_full),
            "quality_no_spec_kappa_weighted": kappa(lead_q_no, llm_q_no, weights="quadratic"),
            "quality_no_spec_within1": within_1(lead_q_no, llm_q_no),
            "helpful_full_kappa": kappa(lead_help_full, llm_help_full),
            "helpful_no_spec_kappa": kappa(lead_help_no, llm_help_no),
            "preference_kappa": kappa(lead_pref, llm_pref),
            "preference_exact_pct": sum(1 for a, b in zip(lead_pref, llm_pref) if a == b) / max(1, len(lead_pref)),
        },
        "lead_prefers_full_pct": sum(1 for p in lead_pref if p == "full") / max(1, len(lead_pref)),
        "llm_prefers_full_pct": sum(1 for p in llm_pref if p == "full") / max(1, len(llm_pref)),
    }

    # Krippendorff's alpha for full-system quality (2 raters: lead + LLM)
    # Reliability data: rows = raters, cols = items
    rd_quality_full = [lead_q_full, llm_q_full]
    rd_quality_no = [lead_q_no, llm_q_no]
    res["lead_vs_llm"]["quality_full_krippendorff_ordinal"] = krippendorff.alpha(
        reliability_data=rd_quality_full, level_of_measurement="ordinal")
    res["lead_vs_llm"]["quality_no_spec_krippendorff_ordinal"] = krippendorff.alpha(
        reliability_data=rd_quality_no, level_of_measurement="ordinal")

    # 3-rater Krippendorff including 30-pair labmate data would require raw labmate ratings.
    # We have only aggregate labmate stats; flag this in output.
    res["three_rater_note"] = (
        "Full 3-rater Krippendorff requires raw labmate per-pair ratings, not just the aggregate "
        "summary at data/processed/responses/labmate_inter_rater_aggregate.json. The headline 3-rater "
        "Krippendorff in the paper is computed on the 99-review classification subsample (§4.6), "
        "where 3 raters' raw ratings are available; here we report 2-rater lead-vs-LLM agreement "
        "on the full 400-row Stage 4b set."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nSaved -> {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
