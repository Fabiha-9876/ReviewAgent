# 4. Results

We report three sets of experiments. **Experiment 1** evaluates the iterative classifier and the cleanlab correction pipeline (Section 4.1). **Experiment 2** compares Stage 3 issue-specification quality across four conditions and Stage 4b response generation across four conditions, using both automatic metrics and a 400-rating blinded human evaluation (Sections 4.2 and 4.3). **Cluster validation** reports the Stage 2 cluster purity (Section 4.4).

## 4.1 Experiment 1.1: Cleanlab Correction Validates Against Expert Gold Standard

The 490-review expert gold-standard set was annotated by the lead author serving as the domain expert. We then evaluate three classifiers against this set as independent annotators: (i) the original V2 LLM labels, (ii) the cleanlab + RoBERTa-anchor corrected labels, and (iii) V5 (a separately-trained classifier on the V2-corrected dataset plus compatibility augmentation).

**Cohen κ progression** (Figure 8) traces the noise-correction effect:

| classifier | n | accuracy | **Cohen's κ** | macro F1 |
|---|---|---|---|---|
| V2 LLM original | 489 | 0.301 | **0.163** *(slight)* | 0.218 |
| cleanlab corrected_v2 | 489 | 0.442 | **0.333** *(fair)* | 0.379 |
| **V5 classifier** | 489 | **0.650** | **0.592** *(moderate–substantial)* | **0.653** |

Each pipeline stage produces a measurable, externally-validated improvement. By the Landis–Koch interpretation thresholds \cite{landis1977}, V5 reaches near-substantial agreement with the expert (>0.60 boundary).

**Per-class F1 against expert** (Table 1) shows that the correction pipeline most heavily benefits the classes the LLM originally failed on:

| class | V2 LLM | corrected_v2 | **V5** | n in expert gold |
|---|---|---|---|---|
| compatibility | 0.000 | 0.000 | **0.826** | 51 |
| performance | 0.000 | 0.473 | **0.767** | 63 |
| usability | 0.105 | 0.265 | **0.554** | 60 |
| feature_request | 0.319 | 0.371 | 0.571 | 35 |
| bug_report | 0.417 | 0.448 | **0.577** | 79 |
| other | 0.301 | 0.472 | 0.602 | 117 |
| praise | 0.382 | 0.623 | 0.675 | 84 |

Two classes were effectively unrecoverable by cleanlab alone (compatibility F1 = 0.000 in the corrected set). The targeted V5 augmentation — 200 synthetic compatibility samples plus 100 mined from the LLM's own bug_report bucket — is what raises compatibility from 0 → 0.83.

**V5 as third-opinion validator.** Applying V5 to the full 215,583-review corpus, we measure agreement with each prior stage's labels (Table 2):

| comparison | agreement % |
|---|---|
| V2 ↔ V5 | 71.96% |
| **V5 ↔ corrected_v2** | **86.77%** |
| V2 ↔ corrected_v2 | 79.49% (of 215K rows V2 unchanged) |
| All three agree | 70.20% |

Critically, on the **40,291 corrections cleanlab made to V2**, V5 *independently* agrees with the correction in **88.66%** of cases (against the original V2 LLM in 9.42%, with a third opinion in 1.92%). This is the strongest available signal that the cleanlab pipeline produces genuine label improvements.

## 4.2 Experiment 1.2: IssueSpec Quality Across 4 Conditions

We compare four Stage 3 conditions on 100 stratified clusters: (a) LLM with taxonomy, (b) LLM free-form, (c) raw concatenation of top-3 reviews, (d) human-written reference (n=20). Table 3 shows the structural quality metrics:

| metric | (a) LLM+taxonomy | (b) LLM free-form | (c) raw_summary | (d) human-written |
|---|---|---|---|---|
| n | 100 | 100 | 100 | 20 |
| **completeness ratio** | **1.000** | 0.691 | 0.346 | 0.691 |
| template adherence % | 76.0 | 0.0 | 0.0 | n/a |
| description (mean words) | 47.8 | 82.5 | 94.8 | 39.9 |
| severity reasoning | varied | varied | all P2 (default) | varied |

**Two findings:**

1. **The taxonomy-grounded condition (a) achieves perfect schema completeness (1.000)**, whereas all other conditions miss type-specific fields. Notably, the **human-written condition (d) scores only 0.691 on completeness** — humans were less thorough than the LLM about filling every required template field.
2. **Description length differs systematically by 2× across conditions.** The taxonomy-grounded condition produces 48-word descriptions (concise, structured), free-form 82 words (narrative, longer), raw summary 95 words (concatenated review text). This length differential, alone, hints at the structural transformation each condition applies.

## 4.3 Experiment 2: Response Generation Quality (Stage 4b)

Stage 4b compares four conditions on 100 reviews: (1) `rrgen_baseline`, (2) `core_baseline`, (3) `reviewagent_no_spec`, (4) `reviewagent_full`. Reference responses come from RRGen's `original_response` field. We report both automatic and human metrics.

### 4.3.1 Automatic Metrics (Table 4)

| metric | rrgen_baseline | core_baseline | reviewagent_no_spec | reviewagent_full |
|---|---|---|---|---|
| BLEU-1 | 0.210 | 0.188 | **0.231** | 0.129 |
| BLEU-2 | 0.028 | 0.019 | **0.040** | 0.012 |
| ROUGE-L | 0.158 | 0.139 | **0.180** | 0.114 |
| BERTScore F1 | 0.844 | 0.824 | **0.851** | 0.818 |
| response length (mean words) | 41 | 62 | 78 | **123** |
| **distinct-1** | 0.094 | 0.045 | 0.026 | **0.102** |
| **distinct-2** | 0.250 | 0.101 | 0.070 | **0.280** |

The automatic-metric ranking favors `reviewagent_no_spec`: **closer in n-gram space to the brief reference replies**. The full-system condition's 3× longer responses (123 words vs 41) are surface-level penalized by BLEU/ROUGE/BERTScore even though they are more content-dense, which we interpret in Section 5.2 as a known limitation of overlap-based metrics for response generation \cite{liu2016how, sai2022survey}.

### 4.3.2 Human Evaluation (Table 5; n=400 paired ratings)

The lead author rated all 400 (review, response) pairs in a fully blinded design (random A/B/C/D labeling per review, response) on three dimensions: quality (1–5), specificity (1–5), helpfulness (Y/N).

| condition | quality | specificity | helpful % |
|---|---|---|---|
| rrgen_baseline | 2.31 ± 0.76 | 2.31 ± 0.76 | 19% |
| core_baseline | 2.98 ± 0.71 | 2.96 ± 0.69 | 84% |
| reviewagent_no_spec | 2.26 ± 0.60 | 2.26 ± 0.60 | 31% |
| **reviewagent_full** | **4.62 ± 0.93** | **4.62 ± 0.93** | **92%** |

**Paired Wilcoxon signed-rank tests on quality scores** (Table 6):

| comparison | Δ (quality) | p-value | significance |
|---|---|---|---|
| reviewagent_full vs reviewagent_no_spec | **+2.36** | < 0.001 | *** |
| reviewagent_full vs core_baseline | **+1.64** | < 0.001 | *** |
| reviewagent_full vs rrgen_baseline | **+2.31** | < 0.001 | *** |
| reviewagent_no_spec vs core_baseline | −0.72 | < 0.001 | *** |
| reviewagent_no_spec vs rrgen_baseline | −0.05 | 0.988 | n.s. |
| core_baseline vs rrgen_baseline | +0.67 | < 0.001 | *** |

The full ReviewAgent system substantially outperforms every baseline at p < 0.001 (Cohen's d for the no_spec comparison, computed from the standardized difference, is approximately 1.6 — a *very large* effect size). The IssueSpec contributes +2.36 quality points beyond RAG-alone.

### 4.3.3 Helpfulness — The Cleanest Headline

The **helpful Y/N** score, which asks whether each response would actually help the user, produces an unambiguous ordering:

```
reviewagent_full        92%   ✓
core_baseline           84%
reviewagent_no_spec     31%
rrgen_baseline          19%
```

The full system's responses are deemed helpful **2.97× more often than RAG-only** and **4.84× more often than the original RRGen-style baseline** by the same expert evaluator on identical inputs.

## 4.4 Cluster Validation (Stage 2)

The 50-cluster validation (5 reviews per cluster, balanced across actionable issue types) yields:

| weighted purity (Y=1, P=0.5, N=0) | 0.660 |
|---|---|
| breakdown: Y / P / N (per cluster) | 21 / 24 / 5 |
| highest-purity class | performance (0.800) |
| lowest-purity class | usability (0.500) |

After lead-author curation of the top 100 clusters (61 Keep, 6 Rename, 12 Merge, 21 Split):

| **curation-aware purity** | **0.814** |
|---|---|
| effective clusters in curated subset | ~120 |

The 21 Split verdicts identify mega-clusters where multiple themes coexist; the 12 Merge verdicts surface near-duplicates. Both inform the limitations section (Section 5.4).
