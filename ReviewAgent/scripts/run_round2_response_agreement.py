"""
Three-rater analysis of the Stage-4 response evaluation (400 blinded ratings).

Raters
    A  lead author   human_work/response_ratings.xlsx
    D  second rater  paper/experiments/labmate_handoff/response_ratings_D.numbers
    E  third rater   paper/experiments/labmate_handoff/response_ratings_E.numbers

All three rate the identical 400 rows (100 reviews x 4 blinded conditions), aligned by
(review_index, blind_id). Condition labels are unsealed only here, from
human_work/response_ratings_blinding.json.

Reports, per rater and pooled:
    mean quality / specificity / helpful-rate per condition
    the full-vs-no_spec paired gain with a bootstrap CI, Wilcoxon, and Cliff's delta
    inter-rater reliability: exact and within-1 agreement, quadratic-weighted kappa on the
    Likert columns, Cohen's kappa and Krippendorff's alpha on the binary helpful column

Outputs
    data/processed/inter_annotator/round2_response_agreement.json
    data/processed/inter_annotator/round2_response_agreement.txt

Usage
    python3 scripts/run_round2_response_agreement.py
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numbers_parser import Document
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
LEAD = REPO / "human_work/response_ratings.xlsx"
HANDOFF = REPO / "paper/experiments/labmate_handoff"
BLINDING = REPO / "human_work/response_ratings_blinding.json"
OUT_DIR = REPO / "data/processed/inter_annotator"

LIKERT = ["quality_1_to_5", "specificity_1_to_5"]
CONDITIONS = ["rrgen_baseline", "prompt_baseline", "reviewagent_no_spec", "reviewagent_full"]
PRETTY = {"rrgen_baseline": "rrgen baseline", "prompt_baseline": "prompt baseline",
          "reviewagent_no_spec": "RAG, no IssueSpec", "reviewagent_full": "RAG + IssueSpec"}
BOOT = 10_000
SEED = 42


def load_numbers(path):
    for sheet in Document(str(path)).sheets:
        for table in sheet.tables:
            rows = table.rows(values_only=True)
            hdr = [str(c) for c in rows[0]]
            if "quality_1_to_5" in hdr:
                return pd.DataFrame(rows[1:], columns=hdr)
    raise SystemExit(f"no ratings table in {path}")


def normalise(df):
    df = df.copy()
    df["review_index"] = df["review_index"].astype(float).astype(int)
    df["blind_id"] = df["blind_id"].astype(str).str.strip()
    for c in LIKERT:
        df[c] = df[c].astype(float)
    df["helpful_y_n"] = df["helpful_y_n"].astype(str).str.strip().str.upper()
    return df.set_index(["review_index", "blind_id"]).sort_index()


def weighted_kappa(x, y, categories):
    """Quadratic-weighted kappa."""
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    n = len(x)
    obs = np.zeros((k, k))
    for a, b in zip(x, y):
        obs[idx[a]][idx[b]] += 1
    obs /= n
    rx = obs.sum(axis=1)
    ry = obs.sum(axis=0)
    exp = np.outer(rx, ry)
    w = np.array([[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)])
    denom = (w * exp).sum()
    return 1 - (w * obs).sum() / denom if denom else float("nan")


def cohen_kappa(x, y):
    n = len(x)
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] / n * cy[c] / n for c in set(cx) | set(cy))
    return po, ((po - pe) / (1 - pe) if pe < 1 else float("nan"))


def krippendorff_alpha_nominal(rows):
    cats = sorted({v for row in rows for v in row})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    co = np.zeros((k, k))
    for row in rows:
        m = len(row)
        for a in range(m):
            for b in range(m):
                if a != b:
                    co[idx[row[a]]][idx[row[b]]] += 1.0 / (m - 1)
    n_total = co.sum()
    n_c = co.sum(axis=1)
    do = (co.sum() - np.trace(co)) / n_total
    de = (n_c.sum() ** 2 - (n_c ** 2).sum()) / (n_total * (n_total - 1))
    return 1 - do / de if de else float("nan")


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def paired_gain(full, nospec, rng):
    diff = np.asarray(full) - np.asarray(nospec)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        w_stat, w_p = stats.wilcoxon(full, nospec)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    sd = diff.std(ddof=1)
    return {"mean_gain": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "wilcoxon_p": float(w_p), "cohens_dz": float(diff.mean() / sd) if sd else float("nan"),
            "cliffs_delta": float(cliffs_delta(full, nospec)),
            "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
            "ties": int((diff == 0).sum())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raters = {
        "A": normalise(pd.read_excel(LEAD, sheet_name="Ratings")),
        "D": normalise(load_numbers(HANDOFF / "response_ratings_D.numbers")),
        "E": normalise(load_numbers(HANDOFF / "response_ratings_E.numbers")),
    }
    keys = raters["A"].index
    for name, df in raters.items():
        if not df.index.equals(keys):
            raise SystemExit(f"rater {name} does not share the row set with A")
        print(f"  rater {name}: {len(df)} rows aligned")

    blinding = {b["review_index"]: b["blinding"] for b in json.loads(BLINDING.read_text())}
    cond = pd.Series({(ri, bid): blinding[ri][bid] for ri, bid in keys}, name="condition")
    cond.index = pd.MultiIndex.from_tuples(cond.index, names=["review_index", "blind_id"])

    rng = np.random.default_rng(SEED)
    report = {"n_rows": len(keys), "n_reviews": len(blinding), "raters": list(raters)}

    # --- per-rater condition means and the headline gain -------------------------
    report["per_rater"] = {}
    for name, df in raters.items():
        d = df.join(cond)
        means = {c: {"quality": float(d.loc[d.condition == c, "quality_1_to_5"].mean()),
                     "specificity": float(d.loc[d.condition == c, "specificity_1_to_5"].mean()),
                     "helpful_pct": float((d.loc[d.condition == c, "helpful_y_n"] == "Y").mean() * 100)}
                 for c in CONDITIONS}
        full = d.loc[d.condition == "reviewagent_full"].sort_index().reset_index()
        nos = d.loc[d.condition == "reviewagent_no_spec"].sort_index().reset_index()
        full = full.sort_values("review_index")["quality_1_to_5"].to_numpy()
        nos = nos.sort_values("review_index")["quality_1_to_5"].to_numpy()
        report["per_rater"][name] = {"condition_means": means,
                                     "full_vs_nospec_quality": paired_gain(full, nos, rng)}

    # --- pooled over the three raters --------------------------------------------
    pooled = pd.concat([df.join(cond).assign(rater=n) for n, df in raters.items()])
    report["pooled_condition_means"] = {
        c: {"quality": float(pooled.loc[pooled.condition == c, "quality_1_to_5"].mean()),
            "specificity": float(pooled.loc[pooled.condition == c, "specificity_1_to_5"].mean()),
            "helpful_pct": float((pooled.loc[pooled.condition == c, "helpful_y_n"] == "Y").mean() * 100)}
        for c in CONDITIONS}

    per_review = (pooled.reset_index()
                  .groupby(["review_index", "condition"])["quality_1_to_5"].mean().unstack())
    report["pooled_full_vs_nospec_quality"] = paired_gain(
        per_review["reviewagent_full"].to_numpy(),
        per_review["reviewagent_no_spec"].to_numpy(), rng)

    # --- inter-rater reliability ---------------------------------------------------
    rel = {}
    for a, b in combinations(raters, 2):
        pair = {}
        for col in LIKERT:
            x = raters[a][col].to_numpy()
            y = raters[b][col].to_numpy()
            pair[col] = {
                "exact_pct": float((x == y).mean() * 100),
                "within1_pct": float((np.abs(x - y) <= 1).mean() * 100),
                "quadratic_weighted_kappa": float(weighted_kappa(
                    list(x), list(y), [1.0, 2.0, 3.0, 4.0, 5.0])),
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            }
        po, k = cohen_kappa(list(raters[a]["helpful_y_n"]), list(raters[b]["helpful_y_n"]))
        pair["helpful_y_n"] = {"exact_pct": po * 100, "cohen_kappa": k}
        rel[f"{a}_vs_{b}"] = pair
    report["inter_rater"] = rel
    report["helpful_krippendorff_alpha"] = krippendorff_alpha_nominal(
        [[raters[n]["helpful_y_n"].iloc[i] for n in raters] for i in range(len(keys))])
    report["unanimous_helpful"] = int(sum(
        len({raters[n]["helpful_y_n"].iloc[i] for n in raters}) == 1 for i in range(len(keys))))

    # --- text report ----------------------------------------------------------------
    L = ["=" * 78,
         "STAGE-4 RESPONSE EVALUATION — 3 HUMAN RATERS",
         "=" * 78, "",
         f"{report['n_rows']} blinded ratings ({report['n_reviews']} reviews x 4 conditions), "
         "identical rows for all three raters", "",
         "-" * 78, "MEAN QUALITY BY CONDITION", "-" * 78,
         f"  {'condition':22s}" + "".join(f"{n:>9s}" for n in raters) + f"{'pooled':>9s}"]
    for c in CONDITIONS:
        row = f"  {PRETTY[c]:22s}"
        for n in raters:
            row += f"{report['per_rater'][n]['condition_means'][c]['quality']:>9.2f}"
        row += f"{report['pooled_condition_means'][c]['quality']:>9.2f}"
        L.append(row)
    L += ["", "-" * 78, "HEADLINE GAIN, RAG + IssueSpec vs RAG without it (quality)", "-" * 78]
    for n in raters:
        g = report["per_rater"][n]["full_vs_nospec_quality"]
        L.append(f"  rater {n}: {g['mean_gain']:+.2f}  CI95 [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}]"
                 f"  Wilcoxon p={g['wilcoxon_p']:.2e}  d_z={g['cohens_dz']:.2f}"
                 f"  delta={g['cliffs_delta']:.3f}  ({g['wins']}W/{g['losses']}L/{g['ties']}T)")
    g = report["pooled_full_vs_nospec_quality"]
    L.append(f"  pooled : {g['mean_gain']:+.2f}  CI95 [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}]"
             f"  Wilcoxon p={g['wilcoxon_p']:.2e}  d_z={g['cohens_dz']:.2f}"
             f"  delta={g['cliffs_delta']:.3f}  ({g['wins']}W/{g['losses']}L/{g['ties']}T)")
    L += ["", "-" * 78, "INTER-RATER RELIABILITY (400 rows)", "-" * 78]
    for pair, v in rel.items():
        L.append(f"  {pair}")
        for col in LIKERT:
            s = v[col]
            L.append(f"    {col:20s} exact {s['exact_pct']:5.1f}%  within-1 {s['within1_pct']:5.1f}%"
                     f"  weighted kappa {s['quadratic_weighted_kappa']:.3f}  r {s['pearson_r']:.3f}")
        s = v["helpful_y_n"]
        L.append(f"    {'helpful_y_n':20s} exact {s['exact_pct']:5.1f}%  Cohen kappa {s['cohen_kappa']:.3f}")
    L += ["",
          f"  helpful, 3-rater Krippendorff alpha: {report['helpful_krippendorff_alpha']:.3f}",
          f"  helpful, unanimous on {report['unanimous_helpful']}/{report['n_rows']} rows", ""]
    text = "\n".join(L)

    (OUT_DIR / "round2_response_agreement.txt").write_text(text + "\n")
    (OUT_DIR / "round2_response_agreement.json").write_text(json.dumps(report, indent=2))
    print("\n" + text)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
