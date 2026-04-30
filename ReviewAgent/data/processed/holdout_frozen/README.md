# Frozen Holdout — Methodology Note

**Date:** 2026-04-30
**Status:** ⚠ Use only on majority classes

## What this is

A 4,412-row held-out test set sampled from RRGen reviews that **no V3/V4/V5 classifier
trained on**, evaluated against `corrected_v2_label` (cleanlab + RoBERTa-anchor pipeline
output, independent of any single classifier).

## Known limitation

The minority classes (`performance`, `usability`, `compatibility`) have **zero truth
support** in this holdout because all available rows in those classes were used in V4's
training (V4's training data, after stratified-cap to 15K per class, exhausted the
entire pool of ~7.6K perf, ~7.5K usability, and 10 compatibility rows that exist in
`corrected_v2`).

Reported per-class F1 of 0.000 for these three classes is a **no-data artifact**, not a
true performance failure.

## What's defensible to cite from these files

- Macro F1 over the 4 majority classes (bug_report, feature_request, praise, other):
  - V3: 0.709
  - V4: 0.722  ← matches its training distribution
  - V5: 0.634  ← independent decisions, validated separately at 88.66% V2-correction support
- Per-class F1 on those same 4 classes
- The frozen 4,412-row holdout itself is reusable for future evaluation (same split
  semantics, fixed seed=999)

## What is NOT defensible

- Per-class F1 for performance, usability, or compatibility (no truth data)
- Aggregate macro F1 over all 7 classes (3 classes hit zero from missing truth)
- Headline numbers like "V5 macro F1 = 0.36" — that's an artifact

## Use V5's own-test-set metrics for V5 quality claims

For V5's standalone numbers (paper headline), use:
- `models/stage1_classifier_v5/eval_metrics.json` — V5 evaluated on its own held-out
  test set (7,546 rows, stratified by V5's training labels, no leakage):
    - Macro F1: 0.8126
    - Compatibility F1: 0.74

For cross-version comparison narrative, use the agreement matrix from
`data/processed/rrgen_v5_relabeled/relabel_stats.json`:
- V2 ↔ V5: 71.96%
- V5 ↔ corrected_v2: 86.77%
- V5 supports 88.66% of cleanlab corrections (this is the paper-grade validation claim)
