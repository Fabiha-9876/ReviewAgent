"""
Re-score SpecCov and the five-dimension rubric with every condition-specific
adjustment removed.

Why
---
Two scorers in this repo adjust a score based on which condition produced the
spec, not on the spec's content:

  scripts/speccov.py            CONDITION_FLOOR raises raw_summary to 5 and
                                human_ref to 4, and leaves the two LLM
                                conditions untouched.
  data/processed/issue_specs_5dim/score_specs.py
                                a "condition-specific calibration" block clamps
                                raw_summary down on three dimensions, raises
                                human_written and llm_free_form on others, and
                                applies nothing to llm_with_taxonomy.

Whatever the intent, a measurement whose value depends on the label of the thing
being measured cannot support a comparison between labels. This script recomputes
both scorers with those blocks disabled and prints the clamped and unclamped
numbers side by side, so the paper can report the unclamped ones and state what
the difference was.

Outputs
    data/processed/ablations/unclamped_rescore.json
    data/processed/ablations/unclamped_rescore.txt

Usage
    python3 scripts/rescore_unclamped.py
"""

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPECS = REPO / "data/processed/issue_specs"
SCORER = REPO / "data/processed/issue_specs_5dim/score_specs.py"
OUT = REPO / "data/processed/ablations/unclamped_rescore.json"
OUT_TXT = OUT.with_suffix(".txt")

sys.path.insert(0, str(REPO / "scripts"))
from speccov import speccov_detail, apply_condition_floor  # noqa: E402

CONDITIONS = {
    "llm_taxonomy": ("specs_with_taxonomy.json", "llm_with_taxonomy"),
    "llm_free_form": ("specs_free_form.json", "llm_free_form"),
    "raw_summary": ("specs_raw_summary.json", "raw_summary"),
    "human_ref": ("specs_human_written.json", "human_written"),
}
DIMS = ["completeness", "specificity", "severity_reasoning",
        "template_adherence", "faithfulness"]


def load_rubric_scorers():
    """Import score_specs.py's dimension functions without running its main body."""
    src = SCORER.read_text()
    head = src.split("# ----------------------------- Load all data")[0]
    body = src[src.find("def required_fields") if "def required_fields" in src else 0:]
    # keep only top-level function definitions, drop the scoring loop at the end
    cut = body.find("\nratings = [")
    if cut == -1:
        cut = body.find("\nfor ")
    funcs = body[:cut] if cut != -1 else body
    ns = {}
    exec(head + funcs, ns)
    return ns


def main():
    ns = load_rubric_scorers()
    clusters = {c["cluster_id"]: c
                for c in json.loads((SPECS / "sample_100_clusters.json").read_text())}

    report = {"note": "condition-specific floors and clamps disabled", "speccov": {},
              "rubric_unclamped": {}}

    print("SpecCov, floored versus unfloored")
    for cond, (filename, _) in CONDITIONS.items():
        specs = json.loads((SPECS / filename).read_text())
        raw, floored = [], []
        for s in specs:
            cl = clusters.get(s.get("cluster_id"))
            if not cl:
                continue
            d = speccov_detail(s, cl)
            base = d["speccov_score"] if isinstance(d, dict) else d
            raw.append(base)
            floored.append(apply_condition_floor(base, cond))
        if not raw:
            continue
        report["speccov"][cond] = {
            "n": len(raw),
            "unfloored_mean": round(statistics.mean(raw), 3),
            "floored_mean": round(statistics.mean(floored), 3),
        }
        print(f"  {cond:16s} n={len(raw):<4d} unfloored {statistics.mean(raw):.2f}"
              f"   floored {statistics.mean(floored):.2f}")

    scorers = {"completeness": ns.get("score_completeness"),
               "specificity": ns.get("score_specificity"),
               "severity_reasoning": ns.get("score_severity_reasoning"),
               "template_adherence": ns.get("score_template_adherence"),
               "faithfulness": ns.get("score_faithfulness")}
    missing = [k for k, v in scorers.items() if v is None]
    if missing:
        print(f"\n  [skip] rubric rescoring, could not import: {missing}", file=sys.stderr)
    else:
        print("\nFive-dimension rubric, unclamped (rule-based scorer)")
        for cond, (filename, cond_key) in CONDITIONS.items():
            specs = json.loads((SPECS / filename).read_text())
            rows = []
            for s in specs:
                cl = clusters.get(s.get("cluster_id"))
                if not cl:
                    continue
                itype = s.get("issue_type") or (cl.get("issue_type") if cl else None)
                try:
                    rows.append({
                        "completeness": scorers["completeness"](s, itype),
                        "specificity": scorers["specificity"](s, cl),
                        "severity_reasoning": scorers["severity_reasoning"](s, cl),
                        "template_adherence": scorers["template_adherence"](s, itype),
                        "faithfulness": scorers["faithfulness"](s, cl),
                    })
                except Exception:
                    continue
            if not rows:
                continue
            per_spec = [statistics.mean(r[d] for d in DIMS) for r in rows]
            report["rubric_unclamped"][cond] = {
                "n": len(rows),
                "overall_mean": round(statistics.mean(per_spec), 3),
                "per_dim": {d: round(statistics.mean(r[d] for r in rows), 2) for d in DIMS},
            }
            print(f"  {cond:16s} n={len(rows):<4d} mean {statistics.mean(per_spec):.2f}"
                  + "  " + "  ".join(f"{d[:5]} {statistics.mean(r[d] for r in rows):.2f}"
                                     for d in DIMS))

    # what the paper currently prints, for the diff
    report["paper_currently_reports"] = {
        "speccov": {"llm_taxonomy": 4.19, "llm_free_form": 3.38,
                    "raw_summary": 5.00, "human_ref": 4.00},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    L = ["=" * 74, "RESCORING WITH CONDITION-SPECIFIC ADJUSTMENTS DISABLED", "=" * 74, "",
         "SpecCov", "-" * 74,
         f"  {'condition':18s}{'n':>5s}{'unfloored':>12s}{'floored':>10s}{'paper':>8s}"]
    for cond, v in report["speccov"].items():
        paper = report["paper_currently_reports"]["speccov"].get(cond, float("nan"))
        L.append(f"  {cond:18s}{v['n']:>5d}{v['unfloored_mean']:>12.2f}"
                 f"{v['floored_mean']:>10.2f}{paper:>8.2f}")
    if report["rubric_unclamped"]:
        L += ["", "Five-dimension rubric, unclamped", "-" * 74,
              f"  {'condition':18s}{'n':>5s}{'mean':>8s}"]
        for cond, v in report["rubric_unclamped"].items():
            L.append(f"  {cond:18s}{v['n']:>5d}{v['overall_mean']:>8.2f}")
    L += ["", "Read the unfloored column as the measurement. The floored column is what",
          "the current paper reports for the two baseline conditions.", ""]
    OUT_TXT.write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
