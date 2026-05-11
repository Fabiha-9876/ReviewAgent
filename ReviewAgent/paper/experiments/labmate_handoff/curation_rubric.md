# Pre-Registered Cluster Curation Rubric

This document specifies the operations a non-author rater applies to a
sample of clusters to produce a "curation-aware purity" score that can be
reproduced independently. Pre-registering the rubric (specifying it before
seeing the data) addresses the reviewer concern that the 0.66 → 0.81
purity boost in §4.4 is post-hoc cherry-picking by the lead author.

## What this is for

Right now §4.4 reports:
- Weighted purity from automatic clustering: 0.66 (50 clusters, 5 reviews each)
- Curation-aware purity after lead-author curation of 100 clusters: 0.81
- Curation operation counts: 61 Keep, 6 Rename, 12 Merge, 21 Split

A reviewer will ask: "If you re-curated, would you get 0.81 again? Or did
you tune the curation to maximize purity?" The way to answer that is:

1. Lock the rubric (this document) BEFORE the rater sees the clusters.
2. Have a non-author apply it to a 20-cluster subset.
3. Compute inter-curator agreement (Cohen's κ on the operation each
   curator picked for each cluster).
4. Report the inter-curator κ alongside the purity figure.

## The four operations

For each cluster, the rater chooses exactly one of:

### KEEP
The cluster's auto-assigned name accurately describes the dominant theme
of all 5 reviews. At most 1 of 5 reviews is off-theme; that 1 is allowed
to be a near-neighbor (e.g., a UI complaint in a feature-request cluster
that mentions UI). Pick KEEP if a reasonable engineer would route all 5
reviews to the same triage queue.

### RENAME
All 5 reviews share a coherent theme, but the auto-assigned name is
wrong, vague, or misleading. The rater writes a corrected name (≤ 8
words). Use RENAME when the cluster is internally coherent but the LABEL
is bad. Common reason: the auto-namer picked a surface keyword
("battery") when the deeper theme is different ("charging-while-using
overheats and discharges").

### MERGE
Two or more clusters describe substantially the same issue and should be
combined. The rater writes the IDs of the clusters to merge with. Use
MERGE when reviewing this cluster causes you to recall another cluster
that's the same thing. If you flag MERGE, you must specify the merge
target by cluster_id.

### SPLIT
The cluster contains 2 or more genuinely different issues that happen to
share keywords. The rater writes a brief note describing the split
(e.g., "split into 'login loop' and 'payment failures'"). Use SPLIT
when 5 reviews break naturally into 2+ engineering-distinct buckets.

## Decision order

When in doubt, the rater applies these tests in order:

1. **Are all 5 reviews about the same thing a developer would fix?** If
   yes, go to step 2. If no, choose SPLIT.
2. **Does the auto-name describe that thing?** If yes and accurate, KEEP.
   If close but wrong words, RENAME.
3. **Do you remember another cluster that's the same?** If yes, MERGE
   (record the target cluster_id).

## What "purity" means here

For the inter-curator agreement check, treat each cluster's KEEP / RENAME
/ MERGE / SPLIT label as a 4-class outcome and compute Cohen's κ across
two raters. We expect:
- κ < 0.4: rubric is too vague, rewrite before applying to full 100
- 0.4 ≤ κ < 0.6: usable, report as a limitation
- κ ≥ 0.6: rubric is reproducible

## Sampling

The 20-cluster subset for the inter-curator check should be drawn
RANDOMLY (seeded) from the 100 clusters, NOT cherry-picked. Use
`numpy.random.default_rng(42).choice(100, size=20, replace=False)` so
this script is reproducible.

## What goes in the paper

After running the inter-curator check, add to §4.4:

> "To validate the curation step is reproducible, a second rater
> independent of the lead author applied the pre-registered rubric
> (Appendix~B) to a random 20-cluster subset. Inter-curator agreement
> on the four operations was κ = X.XX, with disagreement concentrated
> in the M_keep ↔ M_rename boundary (Y of Z disagreements)."

If κ ≥ 0.6, this neutralizes the cherry-picking concern. If 0.4 ≤ κ <
0.6, report it honestly and explain in Limitations.

## Practical instructions for the rater

You will be given:
1. This rubric document.
2. A spreadsheet with 20 rows, one per cluster. Each row has:
   - cluster_id
   - auto_name (current label)
   - 5 sample reviews from that cluster
3. Empty columns to fill: operation (KEEP/RENAME/MERGE/SPLIT), new_name
   (if RENAME), merge_target (if MERGE), split_note (if SPLIT).

Spend roughly 2 minutes per cluster. Total time: 40-60 minutes.

DO NOT discuss with the lead author until you have completed all 20.
DO NOT look at the existing curation labels.

## Reference materials the rater should NOT see

- `data/processed/expert_evaluation/cluster_curation_labels.json` (lead-author labels)
- §4.4 of the paper (gives away the expected operation counts)

The rater can see the auto_name and the 5 sample reviews per cluster.
That is the entire input.
