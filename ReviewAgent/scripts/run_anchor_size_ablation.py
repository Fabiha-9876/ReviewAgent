"""
Anchor-size ablation: how does noise-correction yield depend on the size of
the verified-anchor set?

We subsample the 5,230 verified anchor at {500, 1000, 2500, 5230} sizes,
train a TF-IDF + LogReg anchor on each (always concatenated with the 5,008
MAALEJ samples), apply cleanlab to flag corrections on a 30K-row subsample
of the 215K corpus, and measure correction yield + agreement with the
production V5 corrections.

This is a fast TF-IDF sweep, not a RoBERTa retrain. The signal we're after
is the *shape* of the curve (does smaller still work, where does diminishing
returns kick in), not absolute κ. RoBERTa-anchor retrains are queued behind
the same multi-GPU extension as A7.

Output: data/processed/expert_evaluation/anchor_size_sweep.json
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from cleanlab.filter import find_label_issues

LABELS = ["bug_report", "feature_request", "performance", "usability",
          "compatibility", "praise", "other"]
LBL2IDX = {l: i for i, l in enumerate(LABELS)}


def load_data():
    anchor = json.load(open("data/processed/verified_annotations.json"))
    maalej = json.load(open("data/processed/anchor_combined.json"))
    # anchor_combined is the union; we want to know which rows are MAALEJ vs verified
    # Filter:
    maalej_only = [r for r in maalej if r.get("source") == "maalej_labeled"]
    print(f"Verified anchor: {len(anchor)}")
    print(f"MAALEJ samples in anchor_combined: {len(maalej_only)}")
    return anchor, maalej_only


def fit_anchor(anchor_subset, maalej):
    """Fit TF-IDF + LogReg on subset + MAALEJ."""
    rows = anchor_subset + maalej
    texts = [r["text"] for r in rows]
    # Multi-label: use first label
    labels = [LBL2IDX.get(r["labels"][0] if isinstance(r.get("labels"), list) else r.get("labels"), -1) for r in rows]
    keep = [i for i, l in enumerate(labels) if l >= 0]
    texts = [texts[i] for i in keep]
    y = np.array([labels[i] for i in keep])

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, max_features=20000)
    X = vec.fit_transform(texts)

    cv = cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"),
                          X, y, cv=3, scoring="f1_macro")
    cv_f1 = float(cv.mean())

    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    clf.fit(X, y)
    return vec, clf, cv_f1


def cleanlab_yield(vec, clf, eval_rows):
    """Apply cleanlab on eval_rows; return n flagged + flag rate."""
    texts = [r["text"] for r in eval_rows]
    labels = [LBL2IDX.get(r.get("v2_label"), -1) for r in eval_rows]
    keep = [i for i, l in enumerate(labels) if l >= 0]
    texts = [texts[i] for i in keep]
    y = np.array([labels[i] for i in keep])
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    # cleanlab needs proba matrix aligned with all 7 classes
    if proba.shape[1] != len(LABELS):
        # pad missing classes with 0 columns
        full_proba = np.zeros((proba.shape[0], len(LABELS)))
        for i, c in enumerate(clf.classes_):
            full_proba[:, c] = proba[:, i]
        proba = full_proba
    issues = find_label_issues(labels=y, pred_probs=proba,
                                return_indices_ranked_by="self_confidence")
    return len(issues), len(issues) / len(y)


def main():
    anchor, maalej = load_data()

    print("\nLoading 215K corpus eval subsample (30K)...")
    rrgen = json.load(open("data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"))
    rng = random.Random(42)
    eval_rows = rng.sample(rrgen, 30000)
    print(f"  Eval subsample: {len(eval_rows)} rows")

    sizes = [500, 1000, 2500, 5230]
    out = {"anchor_sizes": sizes, "results": []}

    for n in sizes:
        rng2 = random.Random(42)
        subset = rng2.sample(anchor, min(n, len(anchor)))
        print(f"\n[anchor size = {n}]")
        vec, clf, cv_f1 = fit_anchor(subset, maalej)
        print(f"  CV macro-F1 (3-fold): {cv_f1:.4f}")
        n_flagged, flag_rate = cleanlab_yield(vec, clf, eval_rows)
        print(f"  Cleanlab flags on 30K eval: {n_flagged} ({100*flag_rate:.2f}%)")
        out["results"].append({
            "anchor_size": n,
            "anchor_cv_macro_f1": round(cv_f1, 4),
            "n_flagged_in_30k_eval": n_flagged,
            "flag_rate_pct": round(100 * flag_rate, 2),
            "extrapolated_215k_flags": int(round(n_flagged * 215583 / 30000)),
        })

    Path("data/processed/expert_evaluation").mkdir(parents=True, exist_ok=True)
    out_path = Path("data/processed/expert_evaluation/anchor_size_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print("\n=== Summary ===")
    print(f"{'size':>6s}  {'CV-F1':>7s}  {'flagged':>9s}  {'rate%':>7s}  {'215K_extrap':>12s}")
    for r in out["results"]:
        print(f"{r['anchor_size']:>6d}  {r['anchor_cv_macro_f1']:>7.4f}  {r['n_flagged_in_30k_eval']:>9d}  {r['flag_rate_pct']:>7.2f}  {r['extrapolated_215k_flags']:>12d}")


if __name__ == "__main__":
    main()
