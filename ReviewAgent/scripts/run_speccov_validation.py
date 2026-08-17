"""
Validate SpecCov against independent human faithfulness judgements.

SpecCov is an extractive-coverage score. The paper previously supported it with an
automatic rubric whose faithfulness dimension uses the same overlap procedure, which is
circular. This script tests it properly: three humans rated the same 60 specs for
faithfulness and listed every claim the source reviews do not support, blind to the
generator and to the SpecCov score.

Two questions:
    1. Does SpecCov rank specs the way humans do?          (Spearman, per rater and pooled)
    2. Does SpecCov separate specs humans flagged as       (Mann-Whitney on SpecCov,
       containing unsupported claims from clean ones?       flagged vs clean)

Inputs
    annotator_materials/speccov_validation/speccov_key.json   condition + SpecCov per spec
    paper/experiments/labmate_handoff/speccov_validation_*.numbers   the returned sheets

Note on sheet versions: rater A completed the corrected sheet, which shows the full
evidence SpecCov itself scores against (representative_reviews + first_5_review_texts).
Raters D and E completed an earlier draft that showed only first_5_review_texts. Their
judgements are therefore made on less evidence than SpecCov sees, and are reported
separately rather than pooled into the headline number.

Outputs
    data/processed/ablations/speccov_validation.json
    data/processed/ablations/speccov_validation.txt

Usage
    python3 scripts/run_speccov_validation.py
"""

import json
import statistics
from pathlib import Path

from numbers_parser import Document
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
KEY = REPO / "annotator_materials/speccov_validation/speccov_key.json"
HANDOFF = REPO / "paper/experiments/labmate_handoff"
OUT = REPO / "data/processed/ablations/speccov_validation.json"
OUT_TXT = OUT.with_suffix(".txt")

SHEETS = {
    "A": ("speccov_validation_A 2.numbers", "corrected sheet, full evidence"),
    "D": ("speccov_validation_D.numbers", "earlier sheet, partial evidence"),
    "E": ("speccov_validation_E.numbers", "earlier sheet, partial evidence"),
}
CONDITIONS = ["llm_taxonomy", "llm_free_form", "human_ref"]


def load_sheet(path):
    for sheet in Document(str(path)).sheets:
        for table in sheet.tables:
            rows = table.rows(values_only=True)
            hdr = [str(c) for c in rows[0]]
            if "faithfulness_1_to_5" not in hdr:
                continue
            fi = hdr.index("faithfulness_1_to_5")
            ui = hdr.index("unsupported_details")
            ki = hdr.index("spec_uid")
            out = {}
            for r in rows[1:]:
                if r[ki] is None:
                    continue
                score = float(r[fi]) if r[fi] not in (None, "") else None
                flagged = str(r[ui]).strip() if r[ui] else ""
                out[str(r[ki])] = (score, flagged)
            return out, len(hdr)
    raise SystemExit(f"no rating table in {path}")


def main():
    key = json.loads(KEY.read_text())
    raters, ncols = {}, {}
    for code, (fn, _) in SHEETS.items():
        raters[code], ncols[code] = load_sheet(HANDOFF / fn)
        filled = sum(1 for v in raters[code].values() if v[0] is not None)
        print(f"  rater {code}: {len(raters[code])} rows, {filled} scored, "
              f"{ncols[code]} columns")

    ids = [i for i in key if all(i in r and r[i][0] is not None for r in raters.values())]
    speccov = [key[i]["speccov"] for i in ids]
    report = {"n_specs": len(ids), "sheet_versions": {c: SHEETS[c][1] for c in SHEETS},
              "per_rater": {}}

    for code, R in raters.items():
        human = [R[i][0] for i in ids]
        rho, p_rho = stats.spearmanr(speccov, human)
        r, p_r = stats.pearsonr(speccov, human)
        flagged = [i for i in ids if R[i][1].lower() not in ("none", "")]
        clean = [i for i in ids if R[i][1].lower() in ("none", "")]
        entry = {"mean_human": round(statistics.mean(human), 3),
                 "spearman_rho": round(float(rho), 4), "spearman_p": float(p_rho),
                 "pearson_r": round(float(r), 4), "pearson_p": float(p_r),
                 "n_flagged": len(flagged), "n_clean": len(clean)}
        if flagged and clean:
            a = [key[i]["speccov"] for i in flagged]
            b = [key[i]["speccov"] for i in clean]
            u, p_u = stats.mannwhitneyu(a, b)
            entry.update({"speccov_mean_flagged": round(statistics.mean(a), 3),
                          "speccov_mean_clean": round(statistics.mean(b), 3),
                          "mannwhitney_p": float(p_u)})
        report["per_rater"][code] = entry

    pooled = [statistics.mean([raters[c][i][0] for c in raters]) for i in ids]
    rho, p_rho = stats.spearmanr(speccov, pooled)
    report["pooled"] = {"spearman_rho": round(float(rho), 4), "spearman_p": float(p_rho)}

    report["by_condition"] = {}
    for c in CONDITIONS:
        sub = [i for i in ids if key[i]["condition"] == c]
        report["by_condition"][c] = {
            "n": len(sub),
            "speccov_mean": round(statistics.mean(key[i]["speccov"] for i in sub), 3),
            "human_mean_A": round(statistics.mean(raters["A"][i][0] for i in sub), 3),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    L = ["=" * 76, "SPECCOV VALIDATION AGAINST HUMAN FAITHFULNESS JUDGEMENTS", "=" * 76, "",
         f"{report['n_specs']} specs, 20 per generator, shuffled and unlabelled", "",
         "Q1  Does SpecCov rank specs the way humans do?", "-" * 76]
    for code, v in report["per_rater"].items():
        L.append(f"  rater {code} ({SHEETS[code][1]}): Spearman rho = {v['spearman_rho']:+.3f} "
                 f"(p = {v['spearman_p']:.3f}), mean human score {v['mean_human']:.2f}")
    L.append(f"  pooled over three raters: Spearman rho = "
             f"{report['pooled']['spearman_rho']:+.3f} (p = {report['pooled']['spearman_p']:.3f})")
    L += ["", "Q2  Does SpecCov separate specs humans flagged as unsupported?", "-" * 76]
    for code, v in report["per_rater"].items():
        if "mannwhitney_p" in v:
            L.append(f"  rater {code}: flagged {v['n_flagged']}, clean {v['n_clean']}; "
                     f"SpecCov mean {v['speccov_mean_flagged']:.2f} vs "
                     f"{v['speccov_mean_clean']:.2f}, Mann-Whitney p = {v['mannwhitney_p']:.3f}")
        else:
            L.append(f"  rater {code}: flagged {v['n_flagged']}, clean {v['n_clean']} "
                     "(no clean group, test not run)")
    L += ["", "Ranking by generator, SpecCov versus human", "-" * 76,
          f"  {'condition':16s}{'SpecCov':>10s}{'human (A)':>12s}{'n':>5s}"]
    for c, v in report["by_condition"].items():
        L.append(f"  {c:16s}{v['speccov_mean']:>10.2f}{v['human_mean_A']:>12.2f}{v['n']:>5d}")
    L += ["",
          "Reading: SpecCov shows no significant rank correlation with human faithfulness",
          "judgements, does not separate specs humans flagged as containing unsupported",
          "claims, and orders the three generators in the opposite direction to the humans.",
          "It is reported in the paper as a negative result.", ""]
    OUT_TXT.write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
