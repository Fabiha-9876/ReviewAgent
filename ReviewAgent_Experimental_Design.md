# ReviewAgent — Detailed Experimental Design

---

## Overview: How Experiments Map to the Pipeline

```
RQ1 (Experiment 1) ──→ Tests Stage 3 (Review-to-Issue Translation)
RQ2 (Experiment 2) ──→ Tests Stage 4b (Response Generation)
RQ3 (Experiment 3) ──→ Tests Stage 5 (RLHF Feedback Loop)
Ablations A1-A7    ──→ Test individual component contributions across all stages
```

The logic: each experiment isolates one research question and validates one core contribution. Together, the 3 experiments + 7 ablations provide complete coverage of the pipeline.

---

## Experiment 1: Review-to-Issue Translation Quality (RQ1)

**Research Question:** *How accurately can an LLM-based agent translate noisy, unstructured app review clusters into structured, taxonomy-grounded issue specifications, and how do these compare to human-written GitHub issues?*

### What You're Testing

This is the core novel contribution — no prior work does this. You need to prove that the LLM can produce issue specs that are *good enough* for a developer to act on, and that taxonomy grounding (Zimmermann template, Nielsen's heuristics, etc.) improves quality over free-form generation.

### Data Preparation

1. **Start with the KG output from Stage 2** — you'll have hundreds of prioritized, schema-mapped review clusters.
2. **Randomly sample 100 clusters** stratified by issue type:
   - ~40 bug reports
   - ~20 feature requests
   - ~15 performance complaints
   - ~15 usability issues
   - ~10 compatibility issues

   (Stratification ensures you test all taxonomy types, not just the most common one.)

3. **Recruit 3 domain experts** (experienced mobile developers or QA engineers) who will:
   - Write gold-standard issue specs for all 100 clusters (this becomes your benchmark dataset — Contribution 5)
   - Later score all conditions using the rubric

### Conditions (4 levels of the independent variable)

| Condition | What It Does | Why It's Included |
|---|---|---|
| **(a) LLM + taxonomy grounding** (full system) | LLM receives the cluster schema + taxonomy templates (Zimmermann for bugs, user stories for features, etc.) and generates structured issue specs | This is your proposed system — what you're trying to prove works |
| **(b) LLM without taxonomy** (free-form) | Same LLM, same cluster schema input, but NO taxonomy templates — just "generate an issue specification from this cluster" | This isolates the value of taxonomy grounding. If (a) > (b), taxonomy templates matter |
| **(c) Raw review summary** (lower bound) | No LLM structuring at all — just concatenate the top-5 representative reviews from the cluster as a "summary" | This is the naive baseline. Shows what you get without any intelligence |
| **(d) Human-written specs** (upper bound) | The 3 experts' gold-standard specs | This is the ceiling. Your system should approach but likely not exceed this |

### Procedure

1. For each of the 100 clusters, generate outputs for conditions (a), (b), (c). Condition (d) already exists from expert writing.
2. **Blind and randomize** all 400 outputs (100 clusters x 4 conditions). Raters should NOT know which condition produced which spec.
3. Each of the 3 raters independently scores every output on the **5-dimension rubric**:

| Dimension | What the Rater Asks | Score |
|---|---|---|
| **Completeness** | Are all required fields present? (title, type, steps to reproduce, expected/actual behavior, environment, severity, affected component) | 1-5 |
| **Accuracy** | Are the inferred reproduction steps plausible? Is the component mapping correct? Is severity justified by the evidence? | 1-5 |
| **Actionability** | Could a developer pick this up and start debugging *without* reading the original 200 reviews? | 1-5 |
| **Specificity** | Is this issue clearly distinct from similar but different issues? Would someone confuse it with another cluster? | 1-5 |
| **Clarity** | Is the language precise, unambiguous, well-structured? No jargon confusion? | 1-5 |

4. Compute **Krippendorff's alpha** across the 3 raters to verify they agree (target: alpha > 0.67 for acceptable agreement, > 0.80 for strong).

### Statistical Analysis

- **Primary test:** Paired Wilcoxon signed-rank test comparing rubric scores between conditions (a) vs (b), (a) vs (c), (a) vs (d). Paired because each condition is evaluated on the *same* 100 clusters.
- **Why Wilcoxon, not t-test?** Rubric scores (1-5) are ordinal, not necessarily normally distributed. Wilcoxon is the non-parametric alternative.
- **Multiple comparisons correction:** Apply Bonferroni correction (6 pairwise comparisons → significance threshold p < 0.05/6 = 0.0083).
- **Effect size:** Report Cliff's delta (non-parametric effect size) for each comparison.

### What Success Looks Like

- Condition (a) scores significantly higher than (b) on completeness and actionability → **taxonomy grounding helps**
- Condition (a) scores significantly higher than (c) on all dimensions → **LLM structuring adds value**
- Condition (a) approaches condition (d) → **system produces near-human-quality specs**
- Krippendorff's alpha > 0.67 → **rubric is reliable**

### Secondary Metrics

- **BERTScore** between LLM-generated specs and human-written specs (semantic similarity)
- **Completeness ratio** — automated check: what % of required schema fields are filled?

---

## Experiment 2: Coupled vs. Uncoupled Response Generation (RQ2)

**Research Question:** *Does coupling knowledge-graph-based issue prioritization with issue-spec-aware response generation produce more specific, actionable, and helpful user responses compared to context-unaware baselines?*

### What You're Testing

You're proving that giving the response generator access to the structured issue spec from Stage 3 produces *better* responses than systems that don't have this context. This validates Contribution 4 (the coupling between issue translation and response generation).

### Data Preparation

- Use the **same 100 clusters** from Experiment 1.
- For each cluster, you need representative reviews that a response would address.
- Collect **ground-truth developer responses** where available (from RRGen dataset) for reference.

### Conditions (4 levels)

| Condition | Input to Response Generator | Why It's Included |
|---|---|---|
| **(a) RRGen** | Raw review text only, no issue context, no RAG | Established baseline (Gao et al., 2019). Pure sequence-to-sequence response generation |
| **(b) CoRe** | Review text + app context (changelogs, FAQs) but no structured issue spec | Stronger baseline (contextual but unstructured). Shows what context alone gives you |
| **(c) ReviewAgent 4b WITHOUT issue spec** | RAG with 4 of 5 sources (past responses, changelog, FAQ, similar responses) but NO issue spec from Stage 3 | **Critical ablation** — isolates the value of the issue spec. Same system, minus the coupling |
| **(d) ReviewAgent 4b WITH issue spec** (full system) | RAG with all 5 sources INCLUDING the structured issue spec from Stage 3 | Your full proposed system |

### Why Condition (c) Is Critical

The comparison between (c) and (d) is the most important one. Both use the same RAG architecture, same LLM, same 4 out of 5 input sources. The ONLY difference is whether the structured issue spec is included. If (d) > (c), you've proven that the coupling between Stage 3 and Stage 4b adds measurable value.

### Procedure

1. Generate responses for all 100 clusters under all 4 conditions.
2. **Blind and randomize** all 400 responses.
3. **Automatic evaluation:**
   - **BLEU** — n-gram overlap with ground-truth responses (if available)
   - **ROUGE-L** — longest common subsequence overlap
   - **BERTScore** — semantic similarity using contextual embeddings

4. **Human evaluation** (same 3 raters from Experiment 1):

| Dimension | What the Rater Asks | Score |
|---|---|---|
| **Helpfulness** | Does this response address the user's actual complaint? | 1-5 |
| **Specificity** | Does it reference the specific issue (e.g., "login crash on Android v3.2") or is it a generic template ("we're sorry for the inconvenience")? | 1-5 |
| **Empathy** | Does it acknowledge the user's frustration appropriately? | 1-5 |
| **Accuracy** | Does it correctly describe the issue status? Does it make false promises? | 1-5 |

### Statistical Analysis

- **Primary test:** Friedman test — the non-parametric equivalent of repeated-measures ANOVA. Tests whether there's a significant difference across the 4 conditions.
- **Why Friedman?** You have 4 related conditions (same clusters, different treatments), ordinal scores, and can't assume normality.
- **Post-hoc:** If Friedman is significant (p < 0.05), run **Nemenyi pairwise comparisons** to identify which specific pairs differ.
- **For automatic metrics:** Since BLEU/ROUGE are continuous, you can use repeated-measures ANOVA with Tukey HSD post-hoc if normality holds, otherwise Friedman + Nemenyi.

### What Success Looks Like

- (d) > (c) on specificity and accuracy → **issue spec coupling adds value**
- (d) > (b) > (a) on helpfulness → **progressive context enrichment helps**
- (d) significantly outperforms (a) and (b) on all human dimensions → **full system beats baselines**
- Automatic metrics (BLEU, BERTScore) show consistent trends with human evaluation → **metrics are reliable**

---

## Experiment 3: Dual-Objective vs. Single-Objective RLHF (RQ3)

**Research Question:** *Does dual-objective RLHF (quality + compliance) with dimension-level expert feedback outperform single-objective preference tuning in generating safe, helpful, and accurate app review responses?*

### What You're Testing

You're proving that decomposing feedback into two streams (quality and compliance), grounded in CMDP theory (Altman, 1999), produces better and safer responses than treating everything as one optimization objective.

### Data Preparation

This experiment requires **iterative training data** collection:
- **Iteration 1:** 500 generated responses, each rated by experts on both streams
- **Iteration 2:** 1000 additional responses (total 1500)
- **Iteration 3:** 500 more (total 2000)

Each response is rated on:

**Stream 1 — Quality (5 dimensions, 1-5 each):**
- Helpfulness, Specificity, Empathy, Accuracy, Actionability

**Stream 2 — Compliance (binary per criterion):**
- Does it make unauthorized promises? (yes/no)
- Does it leak internal information? (yes/no)
- Does it follow tone guidelines? (yes/no)
- Is it legally safe? (yes/no)

### Conditions (3 RLHF training strategies)

| Condition | Training Signal | Method | Theoretical Basis |
|---|---|---|---|
| **(a) Single-objective KTO** | Binary good/bad label per response (collapses quality + compliance into one signal) | KTO (Ethayarajh et al., 2024) | Prospect theory — simple but coarse |
| **(b) Single-objective DPO** | "Response A is better than B" pairs (overall quality, no compliance separation) | DPO (Rafailov et al., 2023) | Implicit reward modeling — better signal but still one objective |
| **(c) Dual-objective Constrained PPO** | Quality score (continuous, from Stream 1) + compliance constraint (binary, from Stream 2) as SEPARATE signals | Constrained PPO (Dai et al., 2023) | CMDP theory — maximize quality SUBJECT TO compliance threshold |

### Why This Comparison Matters (The CMDP Argument)

Condition (a) gives the model a single bit of information: "this response is good" or "this response is bad." A response that is helpful but makes an unauthorized promise gets the same "bad" label as one that is unhelpful but safe — the model can't distinguish *why* it's bad.

Condition (b) is better — it learns that "Response A is better than B" — but still conflates the two objectives. A very helpful but slightly unsafe response might be preferred over a safe but unhelpful one, teaching the model to trade off safety for helpfulness.

Condition (c) explicitly separates the objectives: "maximize helpfulness" is the reward, "don't violate compliance" is the constraint. The model is mathematically prevented from trading safety for helpfulness — it navigates the **Pareto frontier** instead.

### Procedure

1. **Train all 3 models** on the same base LLM (e.g., Llama 3 or GPT-4 fine-tuned) using the same initial 500-response dataset.
2. After each iteration, generate 200 test responses per model (600 total).
3. **Pairwise human preference evaluation:** Present raters with pairs of responses (from different models) for the same review cluster. Ask: "Which response is better overall?" Record the choice.
4. **Safety audit:** For each of the 600 test responses, an expert checks whether it violates any compliance criterion.
5. **Repeat for 3 iterations** — tracking how each model improves over time.

### Statistical Analysis

**For preference data:**
- **Bradley-Terry model** — the standard method for analyzing pairwise preference data. It estimates the "strength" of each model on a continuous scale from binary comparison data.
- Reports: win rate of (c) vs (a), (c) vs (b), and (b) vs (a), with 95% confidence intervals.

**For safety violations:**
- **McNemar's test** — compares paired binary outcomes. For each of the 200 test responses, you have: did model (a) produce a violation? Did model (c) produce a violation? McNemar's tests whether the violation rates are significantly different.

**For improvement over iterations:**
- **Mixed-effects regression** — models rubric scores as a function of (model condition) x (iteration number), with random effects for cluster. Tests whether the rate of improvement differs across conditions.

### What Success Looks Like

- (c) has higher preference win rate than (a) and (b) → **dual-objective produces better responses**
- (c) has significantly lower safety violation rate than (a) and (b) → **explicit constraint enforcement works**
- (c) improves faster across iterations → **richer feedback signal accelerates learning**
- (b) > (a) on quality but (b) ≈ (a) on safety → **DPO improves quality but doesn't help safety** (confirming the need for explicit constraints)

---

## Ablation Studies — How They Complement the Experiments

The 3 experiments test the **main claims**. The 7 ablations test **individual component contributions**:

```
Experiment 1 (RQ1) answers: "Does the translation work?"
  └── Ablation A1: "Does the KG help the translation?"
  └── Ablation A2: "Does hierarchical clustering help?"
  └── Ablation A3: "Does taxonomy grounding help?"
  └── Ablation A4: "Does HITL validation at Stage 3 help?"

Experiment 2 (RQ2) answers: "Does coupled response generation work?"
  └── Ablation A5: "Does RAG help?"
  └── Ablation A6: "Does the issue spec specifically help?"

Experiment 3 (RQ3) answers: "Does dual-objective RLHF work?"
  └── Ablation A7: "Does separating quality from compliance help?"
```

### How Each Ablation Works

| Ablation | Full System | Ablated Version | Comparison |
|---|---|---|---|
| **A1: No KG** | Reviews → KG → Clustering → Schema → LLM Translation | Reviews → Direct LLM Translation (skip Stage 2 entirely) | If rubric scores drop, KG adds value |
| **A2: No hierarchy** | Aspect-level grouping → Sub-clustering within aspect | Flat K-means clustering (no aspect-level first pass) | If cluster purity/NMI drops, hierarchy matters |
| **A3: No taxonomy** | LLM uses Zimmermann template, Nielsen's, ISO 25010 | LLM gets "generate an issue spec" with no template guidance | Same as Exp 1 condition (a) vs (b) — measures taxonomy value |
| **A4: No HITL at Stage 3** | Expert rubric validation before Stage 4b | All LLM specs go directly to Stage 4b, no human check | If end-to-end response quality drops, HITL checkpoint matters |
| **A5: No RAG** | RAG with 5 sources + issue spec → response | Issue spec + LLM only (no retrieval) → response | If BLEU/human scores drop, RAG adds value |
| **A6: No issue spec** | RAG with 5 sources (including issue spec) → response | RAG with 4 sources (excluding issue spec) → response | Same as Exp 2 condition (c) vs (d) — measures coupling value |
| **A7: Single-stream** | Quality (Stream 1) + Compliance (Stream 2) as separate objectives | Single merged score (average quality and compliance into one number) | If preference win rate drops or violations increase, dual-objective matters |

---

## Timeline and Sample Size Justification

### Why 100 Clusters?

- With 100 clusters, 4 conditions, and 3 raters, Experiment 1 produces **1,200 rubric ratings** (100 x 4 x 3) per dimension, or **6,000 total data points** across 5 dimensions.
- Power analysis: For detecting a medium effect size (Cohen's d = 0.5) with paired Wilcoxon at alpha = 0.0083 (Bonferroni-corrected) and 80% power, you need ~85 pairs. 100 clusters provides sufficient power.

### Why 3 Raters?

- Minimum for Krippendorff's alpha (which requires 3+ raters for reliable computation).
- 3 independent ratings per item allows majority-vote resolution of disagreements.
- Practical: more raters increase cost without proportional reliability gains.

### Why 3 Iterations for RLHF?

- Iteration 1 (500 responses): Enough for KTO (binary signals need less data).
- Iteration 2 (1500 cumulative): Enough for DPO (paired preferences need more).
- Iteration 3 (2000 cumulative): Enough for Constrained PPO (needs dimension-level scores at scale).
- 3 iterations show a learning curve — whether improvement plateaus or continues.

---

## Summary: The Complete Experimental Plan

| | Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|---|
| **Tests** | Stage 3 (Translation) | Stage 4b (Response) | Stage 5 (RLHF) |
| **RQ** | RQ1 | RQ2 | RQ3 |
| **Conditions** | 4 (LLM+taxonomy, LLM free-form, raw summary, human) | 4 (RRGen, CoRe, ReviewAgent-no-spec, ReviewAgent-full) | 3 (KTO, DPO, Constrained PPO) |
| **Sample** | 100 clusters | 100 clusters | 200 test responses x 3 iterations |
| **Raters** | 3 experts | 3 experts | 3 experts |
| **Human metrics** | 5-dim rubric (1-5) | 4-dim rubric (1-5) | Pairwise preference + safety audit |
| **Auto metrics** | BERTScore, completeness ratio | BLEU, ROUGE-L, BERTScore | Violation rate |
| **Statistical test** | Wilcoxon + Bonferroni | Friedman + Nemenyi | Bradley-Terry + McNemar |
| **Key comparison** | (a) vs (b): taxonomy value | (c) vs (d): coupling value | (a/b) vs (c): dual-objective value |
| **Ablations** | A1, A2, A3, A4 | A5, A6 | A7 |
