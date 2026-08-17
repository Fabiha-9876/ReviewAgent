"""
Round-2 inter-annotator agreement on the 490-review gold-standard verification.

Raters
    A  lead author            annotator_materials/annotator_A.numbers
    D  second human rater     paper/experiments/labmate_handoff/annotator_D.numbers
    E  third human rater      paper/experiments/labmate_handoff/annotator_E.numbers

All three verified the identical 490-review stratified sample, aligned by row_id.

Two units of analysis
    1. Verification decision (Y/N): is the classifier's predicted label correct?
    2. Effective 7-class label: predicted_label when Y, correct_label_if_no when N.
       This is the unit that matters for the gold standard, since it is the label
       a majority vote produces.

Outputs
    data/processed/inter_annotator/round2_agreement.json
    data/processed/inter_annotator/round2_agreement.txt
    data/processed/inter_annotator/round2_majority_gold.json

Usage
    python3 scripts/run_round2_agreement.py
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from numbers_parser import Document

REPO = Path(__file__).resolve().parent.parent
SHEETS = {
    "A": REPO / "annotator_materials/annotator_A.numbers",
    "D": REPO / "paper/experiments/labmate_handoff/annotator_D.numbers",
    "E": REPO / "paper/experiments/labmate_handoff/annotator_E.numbers",
}
OUT_DIR = REPO / "data/processed/inter_annotator"

LABELS = ["bug_report", "feature_request", "performance", "usability",
          "compatibility", "praise", "other"]


def norm_label(v):
    if v is None:
        return ""
    return str(v).strip().lower().replace(" ", "_")


def load_sheet(path):
    """Return {row_id: {predicted, verdict, correction, effective}}."""
    doc = Document(str(path))
    for sheet in doc.sheets:
        for table in sheet.tables:
            rows = table.rows(values_only=True)
            hdr = [str(c) for c in rows[0]]
            if "correct_yn" not in hdr:
                continue
            ri, pi = hdr.index("row_id"), hdr.index("predicted_label")
            vi, ci = hdr.index("correct_yn"), hdr.index("correct_label_if_no")
            out = {}
            for r in rows[1:]:
                if r[ri] is None:
                    continue
                verdict = str(r[vi]).strip().upper() if r[vi] else ""
                predicted = norm_label(r[pi])
                correction = norm_label(r[ci])
                if verdict == "Y":
                    effective = predicted
                elif verdict == "N":
                    effective = correction or "UNSPECIFIED"
                else:
                    effective = ""
                out[int(float(r[ri]))] = {"predicted": predicted, "verdict": verdict,
                                          "correction": correction, "effective": effective}
            return out
    raise SystemExit(f"no verification table found in {path}")


def cohen_kappa(x, y):
    n = len(x)
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] / n * cy[c] / n for c in set(cx) | set(cy))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss_kappa(ratings, categories):
    """ratings: list of per-item lists of category labels (same rater count each)."""
    n_items = len(ratings)
    n_raters = len(ratings[0])
    counts = [[row.count(c) for c in categories] for row in ratings]

    p_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
           for row in counts]
    p_bar = sum(p_i) / n_items
    p_j = [sum(row[j] for row in counts) / (n_items * n_raters)
           for j in range(len(categories))]
    p_e = sum(p * p for p in p_j)
    return p_bar, p_e, (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def krippendorff_alpha_nominal(ratings):
    """Nominal alpha for complete data, computed from the coincidence matrix."""
    cats = sorted({v for row in ratings for v in row})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    coincidence = [[0.0] * k for _ in range(k)]
    for row in ratings:
        m = len(row)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                coincidence[idx[row[a]]][idx[row[b]]] += 1.0 / (m - 1)

    n_total = sum(sum(r) for r in coincidence)
    n_c = [sum(coincidence[c]) for c in range(k)]
    do = sum(coincidence[c][d] for c in range(k) for d in range(k) if c != d) / n_total
    de = sum(n_c[c] * n_c[d] for c in range(k) for d in range(k) if c != d)
    de /= n_total * (n_total - 1)
    return 1 - do / de if de else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheets = {k: load_sheet(p) for k, p in SHEETS.items()}
    for k, v in sheets.items():
        filled = sum(1 for r in v.values() if r["verdict"])
        print(f"  rater {k}: {len(v)} rows, {filled} verdicts filled")

    ids = sorted(set.intersection(*(set(v) for v in sheets.values())))
    print(f"  common items: {len(ids)}")

    unspecified = {k: [i for i in ids if sheets[k][i]["effective"] == "UNSPECIFIED"]
                   for k in sheets}
    for k, rows in unspecified.items():
        if rows:
            print(f"  WARNING rater {k}: {len(rows)} rows marked N with no correction label")

    usable = [i for i in ids
              if all(sheets[k][i]["verdict"] in ("Y", "N") for k in sheets)
              and all(sheets[k][i]["effective"] not in ("", "UNSPECIFIED") for k in sheets)]
    print(f"  usable for label-level stats: {len(usable)}")

    report = {"n_items": len(ids), "n_usable_label_level": len(usable), "raters": list(sheets)}

    # --- pairwise, verification decision -------------------------------------
    report["pairwise_verdict"] = {}
    for a, b in combinations(sheets, 2):
        po, k = cohen_kappa([sheets[a][i]["verdict"] for i in ids],
                            [sheets[b][i]["verdict"] for i in ids])
        report["pairwise_verdict"][f"{a}_vs_{b}"] = {"raw_agreement": po, "cohen_kappa": k}

    # --- pairwise, effective 7-class label -----------------------------------
    report["pairwise_label"] = {}
    for a, b in combinations(sheets, 2):
        po, k = cohen_kappa([sheets[a][i]["effective"] for i in usable],
                            [sheets[b][i]["effective"] for i in usable])
        report["pairwise_label"][f"{a}_vs_{b}"] = {"raw_agreement": po, "cohen_kappa": k}

    # --- three-rater agreement ------------------------------------------------
    verdict_rows = [[sheets[k][i]["verdict"] for k in sheets] for i in ids]
    p_bar, p_e, fk = fleiss_kappa(verdict_rows, ["Y", "N"])
    report["fleiss_verdict"] = {"kappa": fk, "p_bar": p_bar, "p_e": p_e, "n": len(ids)}

    label_rows = [[sheets[k][i]["effective"] for k in sheets] for i in usable]
    cats = sorted({v for row in label_rows for v in row})
    p_bar_l, p_e_l, fk_l = fleiss_kappa(label_rows, cats)
    report["fleiss_label"] = {"kappa": fk_l, "p_bar": p_bar_l, "p_e": p_e_l,
                              "n": len(usable), "categories": cats}
    report["krippendorff_alpha_label"] = krippendorff_alpha_nominal(label_rows)
    report["krippendorff_alpha_verdict"] = krippendorff_alpha_nominal(verdict_rows)

    report["unanimous_verdict"] = sum(1 for row in verdict_rows if len(set(row)) == 1)
    report["unanimous_label"] = sum(1 for row in label_rows if len(set(row)) == 1)

    # --- majority-vote gold ---------------------------------------------------
    gold, ties = {}, []
    for i in usable:
        votes = Counter(sheets[k][i]["effective"] for k in sheets)
        top, n_top = votes.most_common(1)[0]
        if n_top == 1:
            ties.append(i)
            continue
        gold[i] = {"label": top, "votes": n_top, "unanimous": n_top == len(sheets)}
    report["majority_gold"] = {"n": len(gold), "n_three_way_ties": len(ties),
                               "distribution": dict(Counter(v["label"] for v in gold.values()))}

    (OUT_DIR / "round2_majority_gold.json").write_text(json.dumps(
        {"gold": gold, "ties": ties}, indent=2))

    lines = [
        "=" * 78,
        "ROUND-2 INTER-ANNOTATOR AGREEMENT — 3 HUMAN RATERS",
        "=" * 78,
        "",
        f"Sample: {len(ids)} reviews (stratified, 70 per class x 7 classes)",
        "Raters: A (lead author), D, E — independent, identical item set",
        f"Usable for label-level statistics: {len(usable)}",
        "",
        "-" * 78,
        "UNIT 1 — VERIFICATION DECISION (Y/N)",
        "-" * 78,
    ]
    for pair, v in report["pairwise_verdict"].items():
        lines.append(f"  Cohen kappa {pair:10s} {v['cohen_kappa']:.4f}   "
                     f"(raw {v['raw_agreement']*100:.1f}%)")
    lines += [
        f"  Fleiss kappa (3 raters)   {fk:.4f}",
        f"  Krippendorff alpha        {report['krippendorff_alpha_verdict']:.4f}",
        f"  Unanimous on {report['unanimous_verdict']}/{len(ids)} items",
        "",
        "-" * 78,
        "UNIT 2 — EFFECTIVE 7-CLASS LABEL",
        "-" * 78,
    ]
    for pair, v in report["pairwise_label"].items():
        lines.append(f"  Cohen kappa {pair:10s} {v['cohen_kappa']:.4f}   "
                     f"(raw {v['raw_agreement']*100:.1f}%)")
    lines += [
        f"  Fleiss kappa (3 raters)   {fk_l:.4f}",
        f"  Krippendorff alpha        {report['krippendorff_alpha_label']:.4f}",
        f"  Unanimous on {report['unanimous_label']}/{len(usable)} items",
        "",
        "Interpretation, Landis-Koch (kappa): >0.81 almost perfect, 0.61-0.80 substantial,",
        "0.41-0.60 moderate, 0.21-0.40 fair, <0.20 slight.",
        "Krippendorff: >=0.800 strong, >=0.667 acceptable.",
        "",
        "-" * 78,
        "MAJORITY-VOTE GOLD STANDARD",
        "-" * 78,
        f"  Items with a majority label: {report['majority_gold']['n']}",
        f"  Three-way ties (excluded):   {report['majority_gold']['n_three_way_ties']}",
        "  Distribution:",
    ]
    for lbl, n in sorted(report["majority_gold"]["distribution"].items(),
                         key=lambda kv: -kv[1]):
        lines.append(f"    {lbl:18s} {n:>4}")
    text = "\n".join(lines)

    (OUT_DIR / "round2_agreement.txt").write_text(text + "\n")
    (OUT_DIR / "round2_agreement.json").write_text(json.dumps(report, indent=2))
    print("\n" + text)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
