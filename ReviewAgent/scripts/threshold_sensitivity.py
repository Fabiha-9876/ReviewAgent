"""
Cleanlab threshold sensitivity for the verified-anchor correction pipeline.

Reviewers ask: are the headline (min_anchor_conf=0.70, max_llm_prob=0.20) thresholds
cherry-picked? This script re-applies the cleanlab + threshold logic at a sweep of
(min_anchor_conf, max_llm_prob) pairs using the cached anchor RoBERTa probabilities,
then computes V5 endorsement rate on each resulting correction set.

Uses cached artifacts only (no model reruns):
  - data/processed/rrgen_corrected_v2/anchor_probs.npy     (215583 x 7 RoBERTa probs)
  - data/processed/rrgen_full_labeled/rrgen_full_labeled.json   (V2 LLM labels)
  - data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json   (V5 labels)
  - data/processed/verified_annotations.json               (human-verified anchor set)

Output:
  data/processed/ablations/threshold_sensitivity.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

LABELS = ["bug_report", "feature_request", "performance", "usability",
          "compatibility", "praise", "other"]
LBL2I = {l: i for i, l in enumerate(LABELS)}


def load_v5_labels(path: Path) -> list[str]:
    with open(path) as f:
        rows = json.load(f)
    # row schema includes v5_label, v2_label, corrected_v2_label
    return [r["v5_label"] for r in rows], [r["v2_label"] for r in rows]


def load_llm_labels(path: Path) -> list[str]:
    with open(path) as f:
        rows = json.load(f)
    return [r["predicted_label"] for r in rows]


def load_verified(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    return {v["text"]: v["labels"][0] for v in rows}


def apply_thresholds(pred_probs: np.ndarray, llm_labels_idx: np.ndarray,
                     issue_idx_set: set[int],
                     verified_idx_set: set[int], verified_labels: dict[int, str],
                     min_anchor_conf: float, max_llm_prob: float
                     ) -> tuple[np.ndarray, int, int, int]:
    """Mirrors correct_rrgen_v2.py exactly: cleanlab flag AND thresholds AND label diff.
    issue_idx_set comes from a single cleanlab.find_label_issues call (same for all sweeps)."""
    anchor_idx = pred_probs.argmax(axis=1)
    anchor_conf = pred_probs[np.arange(len(pred_probs)), anchor_idx]
    prob_of_llm = pred_probs[np.arange(len(pred_probs)), llm_labels_idx]

    n = len(pred_probs)
    issue_mask = np.zeros(n, dtype=bool)
    issue_mask[list(issue_idx_set)] = True

    cleanlab_corrected = (
        issue_mask
        & (anchor_idx != llm_labels_idx)
        & (anchor_conf >= min_anchor_conf)
        & (prob_of_llm <= max_llm_prob)
    )

    final = llm_labels_idx.copy()
    final[cleanlab_corrected] = anchor_idx[cleanlab_corrected]
    for i, lbl in verified_labels.items():
        final[i] = LBL2I[lbl]

    n_anchor_changed = int((cleanlab_corrected & ~np.isin(np.arange(n), list(verified_idx_set))).sum())
    n_verified_total = len(verified_idx_set)
    n_total_changed = int((final != llm_labels_idx).sum())
    return final, n_total_changed, n_anchor_changed, n_verified_total


def main() -> int:
    base = Path(__file__).resolve().parent.parent
    pp_path = base / "data/processed/rrgen_corrected_v2/anchor_probs.npy"
    llm_path = base / "data/processed/rrgen_full_labeled/rrgen_full_labeled.json"
    v5_path = base / "data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"
    ver_path = base / "data/processed/verified_annotations.json"
    out_path = base / "data/processed/ablations/threshold_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading anchor_probs.npy", file=sys.stderr)
    pred_probs = np.load(pp_path)
    print(f"      shape: {pred_probs.shape}", file=sys.stderr)

    print(f"[2/4] Loading V2 LLM labels", file=sys.stderr)
    llm_str = load_llm_labels(llm_path)
    llm_idx = np.array([LBL2I[l] for l in llm_str], dtype=np.int64)
    print(f"      {len(llm_idx):,} rows", file=sys.stderr)

    print(f"[3/4] Loading V5 labels (for endorsement check)", file=sys.stderr)
    with open(v5_path) as f:
        v5_rows = json.load(f)
    v5_str = [r["v5_label"] for r in v5_rows]
    v5_idx = np.array([LBL2I[l] for l in v5_str], dtype=np.int64)
    print(f"      {len(v5_idx):,} rows", file=sys.stderr)
    assert len(v5_idx) == len(llm_idx)

    print(f"[4/4] Building verified-anchor mask", file=sys.stderr)
    verified_by_text = load_verified(ver_path)
    print(f"      {len(verified_by_text):,} verified texts", file=sys.stderr)
    # Need texts to map to indices
    with open(llm_path) as f:
        noisy = json.load(f)
    texts = [r["text"] for r in noisy]
    verified_idx_to_label = {}
    for i, t in enumerate(texts):
        if t in verified_by_text:
            verified_idx_to_label[i] = verified_by_text[t]
    verified_idx_set = set(verified_idx_to_label.keys())
    print(f"      mapped to {len(verified_idx_set):,} review indices", file=sys.stderr)

    print(f"[5/5] Running cleanlab.find_label_issues once (shared across sweeps)", file=sys.stderr)
    from cleanlab.filter import find_label_issues
    issue_arr = find_label_issues(
        labels=llm_idx, pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    issue_idx_set = set(int(i) for i in issue_arr)
    print(f"      cleanlab flagged {len(issue_idx_set):,} suspect rows", file=sys.stderr)

    # Threshold sweep: full 3x3 = 9-point grid around the headline (0.70, 0.20).
    # anchor_conf in {0.65, 0.70, 0.75}, max_llm_prob in {0.15, 0.20, 0.25}.
    sweeps = [(mac, mlp) for mac in (0.65, 0.70, 0.75)
              for mlp in (0.15, 0.20, 0.25)]
    results = []
    for mac, mlp in sweeps:
        final, n_total, n_anchor, n_ver = apply_thresholds(
            pred_probs, llm_idx, issue_idx_set, verified_idx_set, verified_idx_to_label, mac, mlp
        )
        # V5 endorsement: rows where final != llm AND v5_label == final_label
        changed_mask = (final != llm_idx)
        n_changed = int(changed_mask.sum())
        v5_supports_correction = int(((v5_idx == final) & changed_mask).sum())
        v5_supports_v2 = int(((v5_idx == llm_idx) & changed_mask).sum())
        v5_third = n_changed - v5_supports_correction - v5_supports_v2
        endorsement_pct = 100.0 * v5_supports_correction / n_changed if n_changed else 0.0
        result = {
            "min_anchor_conf": mac,
            "max_llm_prob": mlp,
            "n_total_changed_strict": n_total,   # rows where final label != V2 LLM label
            "n_changed_anchor_only": n_anchor,
            "n_verified": n_ver,
            "v5_supports_correction": v5_supports_correction,
            "v5_supports_v2_original": v5_supports_v2,
            "v5_third_opinion": v5_third,
            "v5_endorsement_pct": round(endorsement_pct, 2),
        }
        results.append(result)
        print(f"  ({mac:.2f}, {mlp:.2f}) -> {n_changed:,} changes, "
              f"V5 endorsement {endorsement_pct:.2f}%", file=sys.stderr)

    out = {
        "headline_thresholds": {"min_anchor_conf": 0.70, "max_llm_prob": 0.20},
        "method": "Re-apply cleanlab anchor-confidence + LLM-confidence thresholds "
                  "on cached RoBERTa probs; count strict label changes; V5 endorsement "
                  "= V5_label matches corrected label on those changed rows.",
        "sweeps": results,
    }
    json.dump(out, open(out_path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
