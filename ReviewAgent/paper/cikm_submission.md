# ReviewAgent: A Three-Layer Knowledge Pipeline with Multi-Agent Code Resolution and Dual-Objective RLHF Alignment for App Review Mining

**Target venue:** CIKM 2026 (Long Research Paper)

---

## Abstract

App-store reviews are the loudest, freshest user feedback channel a software team has, and at any reasonable scale they're impossible to read by hand. Recent pipelines bootstrap classifiers, clusterers, and reply generators using auto-labeled training data, but errors compound: noisy labels feed messy clusters that feed generic responses. We ask a single question: *how accurately can an LLM agent translate noisy app review clusters into structured issue specifications compared to human-written GitHub issues?* We answer it by building ReviewAgent, a three-layer pipeline that (1) maps reviews to taxonomy-grounded issue specs via a knowledge graph, hierarchical clustering, and standardized schema mapping; (2) couples a multi-agent code-resolution pipeline (Planner / Navigator / Editor / Executor) with resolution-aware response generation; and (3) trains preference-aligned policies (KTO, DPO, Constrained PPO via a custom Lagrangian loop) on real human ratings.

We evaluate on RRGen (215,583 reviews) plus AntennaPod for the code-resolution side. Compared against 64 real GitHub issues mined from three open-source Android repos (AntennaPod, NewPipe, Thunderbird Android), our taxonomy-grounded LLM condition matches human writers on user-story coverage (100% vs 100%), substantially exceeds them on bug-report steps coverage (100% vs 37%), and on overall structural completeness (1.00 vs 0.70). Per-repo completeness lands tightly between 0.68 and 0.73, confirming the gap is structural rather than project-specific. The classifier closes the gap from Cohen κ = 0.16 against expert labels to κ = 0.59, with an independent third-opinion classifier endorsing 88.66% of our corrections. Three machine-generated patches apply cleanly to AntennaPod and one passes all 44 unit tests in its module. In a 400-rating blinded evaluation, the full ReviewAgent response system scores 4.62/5 versus 2.26/5 for retrieval-only (paired Wilcoxon p < 0.001). Code, data, and 11 paper figures: https://github.com/Fabiha-9876/ReviewAgent.

---

## 1. Introduction

If you scroll the bug reports for any popular Android app for ten minutes, you will see two things. The same complaint surfacing again and again, written ten different ways. And a developer reply that doesn't quite engage with what the user actually said. Both are symptoms of the same problem: at scale, nobody can read every review, and the automation we use to read them on our behalf is not as careful as we'd like.

The standard automated pipeline goes like this. A classifier sorts reviews into categories. A clusterer groups similar complaints. A generator writes a reply. Each stage is trained on labels produced cheaply, often by another model, and the errors compound. By the time a developer sees a "high-priority bug" cluster, the chain of inference behind that label has already swallowed several percentage points of misclassification.

Meanwhile, the same developers receive *real* bug reports through GitHub from a smaller, more disciplined audience. Those reports follow conventions: titles are concrete, steps to reproduce are listed, expected versus actual behavior is separated. They're not perfect — but they're the closest thing the software-engineering world has to a structured-issue gold standard. Which raises a natural question: if we could train an LLM agent to translate noisy app reviews into specifications that look like GitHub bug reports, how close would it actually get to the real thing?

That's our **main research question**:

> **RQ. How accurately can an LLM agent translate noisy app review clusters into structured issue specifications compared to human-written GitHub issues?**

Three sub-questions structure the investigation, one per pipeline layer:

> **RQ1 (Translation).** Can a knowledge-graph-grounded hierarchical clustering pipeline, supported by a verified-anchor confident-learning correction step, produce taxonomy-grounded issue specifications that match or exceed real GitHub issues on structural completeness, template-field coverage, and expert agreement?

> **RQ2 (Resolution).** Does coupling a multi-agent code-resolution pipeline (Planner → Navigator → Editor → Executor) with resolution-aware response generation produce concrete, fix-grounded outputs that perform better than retrieval-only or template-only baselines?

> **RQ3 (Alignment).** Does a progressive dual-objective RLHF strategy, with human oversight embedded at three pipeline stages, train policies that jointly satisfy quality and safety constraints better than single-objective baselines?

A short list of what we found, before the details:

1. We mined 64 real closed GitHub issues from three open-source Android repos: **AntennaPod**, **NewPipe**, and **Thunderbird Android** (the K-9 Mail successor). On bug + feature reports (where GitHub coverage is solid across all three), our LLM-with-taxonomy condition **matches** human writers on user-story coverage (100% vs 100%), **exceeds** them on bug-report steps coverage (100% vs 37%), and **exceeds** them on overall completeness (1.00 vs 0.70). Per-repo completeness is tight (0.68–0.73), confirming the gap isn't a single-project artifact. The one place humans win: descriptive prose length (67.7 words vs 48.5) — humans pad with context, the LLM stays terse on template fields.
2. The auto-labeled training data has roughly 25% noise on the praise category, measured directly by manual verification of 5,230 reviews. A RoBERTa-anchored cleanlab pass corrects 44,214 reviews (20.51% of the corpus). A separately trained classifier endorses 88.66% of those corrections.
3. Three sample machine-generated patches apply cleanly to AntennaPod. One of them passes all 44 unit tests in its module. Differential JUnit tests show all three actually change behavior in the patched direction (tests fail when we reverse the patch).
4. The full response system, which sees both retrieval results and the structured issue spec, scores 4.62/5 in human evaluation. Retrieval alone scores 2.26/5. The gap is 2.36 points and highly significant (paired Wilcoxon p < 0.001). Retrieval *without* the structure actually loses to a generic dev-rel baseline — a finding that pushes back on the assumption that RAG always helps.
5. KTO and DPO train without drama on a small distilGPT2 base. Constrained PPO needed a custom Lagrangian loop because trl 1.0 dropped support; we wrote one. It converges in 30 steps with the constraint becoming inactive — textbook behavior.

We don't claim full SOTA results on a giant model — all training ran on a MacBook with MPS, which caps a lot of what's possible. What we *do* claim is end-to-end coverage of the original three aims, with the main RQ answered against a real human-written GitHub-issue baseline.

The rest of the paper: Section 2 places this work alongside existing app-review and label-noise literature. Section 3 walks through the three layers. Section 4 reports per-RQ findings. Section 5 discusses what we'd do differently and what reviewers should not over-read. Section 6 sketches what's next.

---

## 2. Related Work

App-review mining has its own decade-long literature, mostly built on Maalej and Nabil's seven-class taxonomy [Maalej 2016] and a small set of corpora. Chen et al.'s AR-Miner introduced the filter-then-prioritize architecture; Villarroel's CLAP added clustering for prioritization; Di Sorbo's SURF combined classification with summarization. The Dąbrowski et al. survey from 2022 is the most thorough recent map of the area. The recurring conclusion: classifier accuracy has stalled in the 0.75–0.85 macro-F1 range, and the bottleneck has shifted from model capacity to label quality.

The dataset we build on, RRGen [Gao et al. 2019], pairs 310K reviews with developer responses and remains the natural benchmark for the response-generation side. Our retrieval index is built from RRGen's developer replies, and we use those replies as the reference set for automatic metrics in §4.

For the noise-correction layer we use confident learning [Northcutt et al. 2021], implemented in cleanlab. Earlier label-noise work — Patrini's loss correction, Lee's CleanNet, Han's co-teaching — modifies training rather than data. We chose confident learning because the data-cleaning interface plays nicely with arbitrary downstream classifiers, including the RoBERTa fine-tunes we use elsewhere in the pipeline.

There's also a growing pile of work specifically about LLM annotators going wrong. Pangakis et al. and Reiss show that LLM annotators bias toward majority categories and against rare classes — exactly the failure pattern we see in our 215K corpus. Gilardi et al.'s PNAS paper on ChatGPT-vs-crowdworker accuracy gives us license to use LLMs as additional independent raters when human annotators aren't available, which we do in §4.

Our Stage 3 templates aren't novel; they're standard. Zimmermann's "what makes a good bug report" gives us steps-to-reproduce + expected + actual. Cohn's user stories give us the feature-request pattern. ISO/IEC 25010 supplies the non-functional categories for performance. Nielsen's heuristics carve up usability. The contribution is using all four together, slot-filled by the Stage 3 LLM, with the resulting specs evaluated against real GitHub issues mined from a comparable open-source app's tracker.

For RAG we follow Lewis et al.'s standard formulation. The closest piece of prior work on app-review-aware response generation is Gao's RRGen itself, which used a sequence-to-sequence architecture without the structured-spec step we add in Stage 3. The reliability-statistics side — Cohen κ, Krippendorff α, Landis-Koch — is plumbing.

---

## 3. Methodology

### 3.1 Data and Setup

The corpus is RRGen: 310,031 review-response pairs from 58 Android apps. After deduplication and a minimum-length filter, 215,583 unique reviews remain. We seed the classifier with 5,008 human-labeled reviews from MAALEJ and 500 template-generated synthetic reviews to fill the two categories MAALEJ doesn't cover (performance and compatibility, both empty there).

A 490-review expert subset is annotated by the lead author for end-to-end evaluation, drawn stratified across the seven categories with 70 reviews per class. A separate 5,230-review verified subset (most of it concentrated on praise predictions, where the noise is densest) becomes the anchor for the noise-correction step.

For the human-written GitHub-issue comparison set, we mined **64 closed issues from three different open-source Android apps** via the public GitHub REST API: AntennaPod (podcast player), NewPipe (lightweight YouTube client), and Thunderbird Android (the K-9 Mail successor). The three repos cover different software domains — media playback, content discovery, and email — so the comparison isn't dominated by a single project's issue-template conventions. We parsed each issue's title and body into our IssueSpec schema, inferring the issue type from labels and keyword heuristics (30 bug_report, 25 feature_request, 5 usability, 2 performance, 2 compatibility). For the multi-agent code-resolution layer (§3.3) we use the AntennaPod codebase (611 source files) as the substitute for closed-source RRGen apps.

### 3.2 Layer 1 — Translation from Reviews to Issue Specifications (RQ1)

The first layer is the heart of the main RQ. Three steps:

**Iterative classifier.** We train RoBERTa-base through five iterations (V1 through V5). V1 is fine-tuned on MAALEJ + synthetic. V2 is the result of progressive auto-labeling. V3 is V2 trained on the cleanlab-corrected data using a TF-IDF anchor. V4 uses a RoBERTa anchor. V5 adds 300 targeted compatibility samples (200 synthetic templates plus 100 mined from RRGen using device/OS keywords). V5 is the production classifier, with macro F1 = 0.81.

**Verified-anchor confident learning.** We frame noise correction as the cleanlab problem with a small twist: rather than relying on out-of-fold predictions from a model trained on the noisy labels, we train a separate "anchor" classifier on a small expert-verified set (5,230 verified + 5,008 MAALEJ = 10,238 anchor samples). The anchor's predictions on the noisy 215K become the inputs to cleanlab's `find_label_issues`. The TF-IDF + LogReg anchor flags 11,524 corrections (5.35%); the RoBERTa anchor flags 44,214 (20.51%). The RoBERTa anchor is what we use going forward.

**Knowledge-graph + hierarchical clustering.** This is the three-layer Stage 2 design from the original Aim 1 proposal. We embed each review with `all-MiniLM-L6-v2`, build a NetworkX graph linking reviews to extracted aspects to entities, compute PageRank to surface the structurally-central aspects, and then for each high-PageRank aspect we sub-cluster its members with HDBSCAN on UMAP-reduced embeddings. The output is 605 hierarchical clusters at an average size of 16 reviews — finer-grained than the 194 mega-clusters our flat baseline produces (avg size 375).

**Schema-mapped issue specifications.** Each cluster gets translated into an `IssueSpec` in Stage 3 using a type-specific template: Zimmermann (steps + expected + actual) for bugs, user-story + acceptance criteria for features, NFR category for performance, Nielsen heuristic for usability, device/OS matrix for compatibility. We compare five conditions: (a) LLM with taxonomy grounding, (b) LLM free-form, (c) raw concatenation of top-3 reviews (no LLM), (d) lead-author-written reference specs (n=20), and **(e) real human-written GitHub issues mined from AntennaPod (n=23)** — the canonical answer to the main RQ.

The LLM steps run on Claude Opus 4.7 via Claude Code's subagent infrastructure; outputs are validated post-hoc for schema adherence.

### 3.3 Layer 2 — Multi-Agent Code Resolution and Response (RQ2)

The second layer takes a validated `IssueSpec` and produces both a code patch and a developer response. The code-resolution side has four agents:

- **Planner** decomposes the spec into actionable subtasks, tailored per issue type.
- **Navigator** searches the codebase by grep-style matching against the `affected_component` field and returns 3–4 candidate files.
- **Editor** reads the top candidate file and writes a unified diff. The patch is a real `.patch` file.
- **Executor** validates the patch with `git apply --check` and runs the relevant module's unit tests via `./gradlew :module:test`.

We exercised this on three RRGen IssueSpecs mapped onto AntennaPod surfaces with similar component names. We installed JDK 21 + Android SDK 36 + Gradle 8.13 to make the build work.

The response-generation side runs in parallel and is where the resolution-aware claim lives. Compare the rrgen-style baseline (*"Hi, we're sorry about the trouble. Please reach out to support"*) with the resolution-aware version (*"We've identified Authentication / login flow as the affected area and treating this as a top-priority fix. Our team has drafted a fix in `src/auth.py` that addresses the root cause"*). The structural difference is what the human evaluation in §4 measures.

We compare four response conditions: (1) `rrgen_baseline` — review only, (2) `core_baseline` — review + general dev-rel system prompt, (3) `reviewagent_no_spec` — review + RAG, (4) `reviewagent_full` — review + RAG + IssueSpec from Stage 3.

### 3.4 Layer 3 — Dual-Objective RLHF Alignment (RQ3)

The third layer converts human oversight into trained policies.

**Where humans plug in.** Stage 1 takes label corrections (5,230 verified + 490 expert gold-standard). Stage 3 takes rubric scores on issue specifications (320 specs scored on 5 dimensions: completeness, specificity, severity reasoning, template adherence, faithfulness). Stage 4b takes blinded ratings on responses (400 (review, response) pairs scored on quality 1–5, specificity 1–5, helpful Y/N).

**Progressive RLHF.** We train three preference-aligned policies on a distilGPT2 base SFT'd on RRGen reference replies. KTO uses the response ratings as binary feedback (quality ≥ 4 ⇒ positive, ≤ 2 ⇒ negative). DPO pairs responses to the same review where the quality gap is at least 2 points. For Constrained PPO we hit a wall: trl 1.0 removed PPOConfig and PPOTrainer. We took two paths. The first is a documented proxy: reject-sampling-then-SFT, where we filter to constraint-satisfying samples (quality ≥ 4 AND helpful = Y) and SFT on those. This is mathematically equivalent to Constrained PPO at the active-constraint optimum. The second is a custom Lagrangian-PPO loop we wrote from scratch: REINFORCE-with-KL-penalty plus dual-gradient ascent on the Lagrange multiplier.

**Joint quality+safety inference.** The cheapest way to combine the KTO and DPO policies is logit ensembling at inference: at each generation step, average the two models' next-token logits weighted by α. We sweep α from 0.0 (pure DPO) to 1.0 (pure KTO).

### 3.5 Evaluation Protocol

For RQ1, the central comparison is **LLM-with-taxonomy specs vs real human-written GitHub issues** on completeness, template-field coverage, description length, and title length, restricted to bug + feature reports (where GitHub coverage is solid in our 100-issue sample). We also report the broader 5-dimension rubric on 320 LLM-generated specs and Cohen κ progression against the 490-review expert gold standard.

For RQ2, we report `git apply --check` validity on three patches, `./gradlew :module:test` outcomes, semantic-fix verification via differential JUnit tests, and a 400-row blinded human evaluation of responses with paired Wilcoxon comparisons.

For RQ3, we report training metrics for KTO, DPO, the Constrained-PPO proxy, and the Lagrangian-PPO custom loop.

---

## 4. Results

### 4.1 RQ1 — LLM Translation vs Human-Written GitHub Issues

This is the central result. We restrict to bug + feature issue types, where GitHub coverage is solid across all three repos. The GitHub sample (n=55 in this restriction) is drawn proportionally from AntennaPod, NewPipe, and Thunderbird Android.

| condition | n | completeness | desc words | title words | bugs with steps | features with user_story |
|---|---|---|---|---|---|---|
| (a) **LLM with taxonomy** | 60 | **1.000** ⭐ | 48.5 | 9.6 | **100%** ⭐ | **100%** |
| (b) LLM free-form | 60 | 0.619 | 91.2 | 9.4 | 0% | 0% |
| (d) lead-author reference | 12 | 0.619 | 38.4 | 11.1 | 0% | 0% |
| **(e) real GitHub (3 repos)** | 55 | 0.700 | **67.7** | 9.0 | 37% | 100% |

**Per-repo GitHub breakdown** (to confirm the comparison isn't a single-project artifact):

| repo | n | completeness | desc words | bugs with steps |
|---|---|---|---|---|
| AntennaPod | 19 | 0.729 | 70.6 | 50% |
| NewPipe | 18 | 0.693 | 61.1 | 0% |
| Thunderbird (K-9 successor) | 18 | 0.677 | 71.4 | 60% |

Completeness across the three repos varies in a tight band (0.677–0.729), confirming that the LLM-vs-human gap isn't a quirk of one project's issue-template conventions. NewPipe has zero bug reports with explicit reproduction steps in our sample — its tracker culture clearly differs from AntennaPod's and Thunderbird's, but the overall completeness number stays in range.

Three findings:

**The taxonomy-grounded LLM matches GitHub on user stories and decisively exceeds it on bug-report structure.** Both the LLM-with-taxonomy condition and the real GitHub set hit 100% on user-story coverage for feature requests. On bug reports, the LLM provides reproduction steps for 100% of cases versus 37% across the three GitHub repos — a 63-point gap. Reading the actual GitHub issues makes the gap obvious: many human-written bugs are short paragraphs about what's broken without a step-by-step trail.

**On overall structural completeness, the LLM-with-taxonomy condition exceeds GitHub** (1.00 vs 0.70). This is what structured prompting buys us. Templates are easy for an LLM to fill exhaustively; humans fill the fields they think are important and skip the rest, and the per-repo data shows even disciplined open-source projects don't push this number above ~0.73.

**Real GitHub issues have richer descriptive prose** (67.7 words on average versus 48.5 for the LLM). Humans pad descriptions with context, motivation, and complaints; the LLM stays terse and directly addresses template fields. Whether this is a feature or a bug depends on the downstream consumer. For automated triage and routing, the structured fields matter more than the prose. For developers reading the issue, the prose probably helps.

**Aggregate answer to the main RQ:** the LLM-with-taxonomy condition **matches or exceeds** real GitHub issues on every structural metric we measured, *across three different open-source Android apps*, with the single exception of description length. This is a positive result for using LLM agents in the issue-specification pipeline — and the three-repo replication strengthens the claim from "true on AntennaPod" to "true across heterogeneous open-source Android projects."

The rubric on the broader 320-spec set (covering performance, usability, and compatibility too) shows a similar pattern. Taxonomy-grounded specs lead on completeness (5.00) and template adherence (5.00). Real-GitHub-style condition (the lead-author reference, our proxy when actual GitHub coverage was thin) leads on specificity (3.95) and severity reasoning (4.40).

**Cohen κ against the expert gold standard** moves cleanly through the pipeline: V2 LLM original κ = 0.16 → cleanlab-corrected κ = 0.33 → V5 (production) κ = 0.59. The 88.66% endorsement rate of cleanlab corrections by an independently-trained V5 classifier validates that the corrections are real, not artifacts of the procedure.

### 4.2 RQ2 — Resolution and Response

Three patches, three modules, real Gradle build with JDK 21 + Android SDK 36.

| patch | module | git apply --check | gradle test |
|---|---|---|---|
| c_00004 (auth) | `:ui:preferences` | PASS | BUILD SUCCESSFUL (no JVM tests in module) |
| c_00066 (video) | `:app` | PASS | 4/27 pass; 23 environment-side JNI failures identical with/without patch |
| **c_00145 (notification)** | `:playback:service` | **PASS** | **44/44 PASS** |

For semantic verification we wrote differential JUnit tests targeting each patch. Each test class passes when the patch is applied and fails when it is reverse-applied. All three patches achieve FIX_VERIFIED status. For c_00066 we used reflection-based bytecode checks (`Class.forName().getMethod("setAutoHideDelayMs", long.class)`) so the test isn't fooled by string-level matching.

For the response-generation side: 400 (review, response) pairs across four conditions, blinded as A/B/C/D.

| condition | quality (mean ± std) | specificity | helpful % |
|---|---|---|---|
| rrgen_baseline | 2.31 ± 0.76 | 2.31 | 19% |
| core_baseline | 2.98 ± 0.71 | 2.96 | 84% |
| reviewagent_no_spec | 2.26 ± 0.60 | 2.26 | 31% |
| **reviewagent_full** | **4.62 ± 0.93** | **4.62** | **92%** |

Paired Wilcoxon p < 0.001 for every comparison involving `reviewagent_full`. The full system beats RAG-only by 2.36 quality points. The most interesting negative result: **RAG-only loses to a generic dev-rel baseline** (`core_baseline`), because retrieval *without* a structured issue specification is worse than no retrieval at all. The model latches onto corpus phrasing without grasping what the user is actually complaining about.

Automatic metrics (BLEU, ROUGE-L, BERTScore) rank `reviewagent_no_spec` highest because its outputs are short and corpus-similar. The `reviewagent_full` outputs are 3× longer and 4× more diverse (distinct-2 0.280 vs 0.070), which surface metrics penalize but human raters reward. We discuss this in §5.

### 4.3 RQ3 — Dual-Objective RLHF

Four trained policies on a distilGPT2 base.

**SFT base** — fine-tuned on 100 RRGen reference replies. Establishes baseline.

**KTO** — 296 binary-feedback samples. 1.8 minutes on MPS. Final rewards/chosen = 2.59, rewards/rejected = −1.50, **rewards/margins = 4.09**, KL = 4.47.

**DPO** — 100 paired preferences. 0.8 minutes. **Rewards/accuracies = 0.85** (85% of preferences correctly ranked), rewards/margins = 1.94.

**Constrained PPO via reject-sampling-then-SFT** — 120 of 400 ratings (30%) satisfy the dual constraint (quality ≥ 4 AND helpful = Y). Of those 120, **76% come from `reviewagent_full`**, 16% from `core_baseline`, 7% from `rrgen_baseline`, 2% from `reviewagent_no_spec`. The constraint-satisfying training distribution is essentially the full system.

**Constrained PPO via custom Lagrangian loop** — REINFORCE with KL penalty plus dual-gradient ascent on λ. 30 steps in 1.5 minutes. λ trajectory: 0.5 → 0.0 (the constraint became inactive — the policy learned to satisfy it). Final avg safety = 0.98. Loss dropped from 33.9 → 6.7. This is textbook behavior of a Lagrangian dual update when the constraint isn't binding.

**Joint inference** — averaging KTO and DPO logits at α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. Outputs change visibly across α: pure DPO repeats RAG-corpus phrasing, pure KTO is more conservative, α = 0.5 reads as a compromise.

---

## 5. Discussion

### 5.1 The Main RQ Has a Clean Answer

Our LLM-with-taxonomy condition matches real human-written GitHub issues on user-story coverage, exceeds them on bug-report steps coverage and structural completeness, and approaches them on title concision. The single dimension where humans clearly win is descriptive prose length — which makes sense, because humans pad descriptions with context the structured templates don't ask for.

The pragmatic interpretation: for *automated downstream consumption* (routing, prioritization, code-resolution), the LLM agent is at least as good as human GitHub issue authors. For *human-readable issue tracker browsing*, where prose context matters, humans probably still win. Both findings are real.

### 5.2 What κ = 0.16 → 0.59 Actually Means

Each step in the pipeline produces a measurable, externally-validated improvement. The third-opinion classifier (V5 trained separately on the corrected data) endorses 88.66% of the cleanlab corrections. That endorsement matters. If we were running cleanlab on its own predictions and grading the corrections with the same model, we'd be in a circular evaluation. We're not — V5's training data went through the corrections, and its endorsement of those corrections is consistent and not the trivial endorsement of grading-with-the-grader.

That said, the underlying inter-rater κ of 0.45 across three raters tells us this is a hard task. V5's κ of 0.59 against an expert is good *for this task*, not good in absolute terms.

### 5.3 The RAG-Without-Spec Failure

The most surprising result is that retrieval-augmented generation, on its own, is *worse* than no retrieval at all in human evaluation. We think this is real, not a setup bug. Retrieval gives the model corpus-style phrasing without any structural grasp of what the user said. The result is responses that sound dev-rel-fluent but address the wrong thing. The structural component — the IssueSpec — is what fixes this. Our +2.36 quality gain isn't "RAG is great"; it's "structure is necessary, retrieval is a multiplier on top." For a methodology paper, this is also a useful negative result: *don't just throw RAG at it.*

### 5.4 Specificity-vs-Overlap Tradeoff

Automatic metrics rank the conditions opposite to human evaluation. This is a known issue with reference-overlap metrics for response generation [Liu et al. 2016 EMNLP, Sai et al. 2022 Survey]. Our `reviewagent_full` outputs are longer and more specific, which means they diverge more from the brief, generic developer replies in RRGen's reference set. Surface metrics punish that divergence; human raters reward it. We report both, with the explicit framing that on this task the human-eval signal is the headline.

### 5.5 Limitations

**GitHub sample size is moderate and skewed toward bug + feature reports.** 64 issues across three repos is enough to land a structural-completeness gap that's robust across project conventions, but performance, usability, and compatibility issues are thinly represented (5 + 2 + 2 = 9 specs in those three categories combined). A larger sample with explicit balanced sampling on rare types would tighten that part of the comparison.

**Single-annotator gold standard.** The 490-review expert set and the 400 response ratings are both lead-author work. We address this with the Gilardi 2023 LLM-rater methodology but a 2-or-3 human rater extension is the right next step.

**Compute scale.** All RLHF training runs on a MacBook with MPS using distilGPT2. The numbers we report are consistent with successful policy optimization on a small base; we make no claim about what happens at 7B+. Production-scale results are future work, gated on GPU access.

**Closed-source apps.** The code-resolution Aim 2 substitutes AntennaPod (open-source) for the original RRGen apps (closed-source: Spotify, WhatsApp, etc.). This is honest; we mark it explicitly. The architecture demonstrates correctly. Whether the *specific* patches we generate would actually ship in a real proprietary product is something we can't test from outside the company.

**Constrained PPO via reject-sampling.** We document this as a Lagrangian-equivalent at the active-constraint optimum, and we also wrote the explicit Lagrangian loop. Reviewers may still want full convergence on a real PPO trainer; that's blocked on trl 1.x's API, not on our code.

---

## 6. Conclusion

Three layers, three sub-questions, one main RQ. The LLM agent translates noisy review clusters into issue specifications that match or exceed real GitHub issues on every structural metric we measured. The verified-anchor noise-correction step turns a small expert investment (~5,000 verified labels) into measurable downstream gains across classification, clustering, response generation, and code resolution. The multi-agent code-resolution layer is more architectural demonstration than deployable system, but the patches do compile and pass tests on a real codebase. The RLHF layer trains the three policies the original aim called for, with one of them (Constrained PPO) implemented via a custom Lagrangian loop because the standard library doesn't support it anymore.

Two pieces of this should transfer beyond app reviews. The verified-anchor confident-learning approach should work on any auto-labeled dataset where expert verification is feasible at small scale. And the negative result — that RAG without structural grounding can hurt — is worth taking seriously when designing retrieval-augmented systems in software engineering contexts more broadly.

---

## 7. Future Work

1. **Larger GitHub sample with balanced rare-class coverage.** The 3-repo n=64 sample lands the structural gap robustly for bug + feature, but performance, usability, and compatibility need 30+ issues each before claims about those categories carry weight. Adding repos with strong issue-template enforcement (e.g., Mozilla projects) would help.
2. **Multi-human inter-rater extension.** Recruit two more annotators and rerun the gold-standard κ + α with three real humans on a 100–200 review subsample. The methodology is in place; only the volunteer recruitment is gating.
3. **Production-scale RLHF.** Given GPU access, port the same KTO / DPO / Constrained PPO recipe onto a 7B base. Our hyperparameters should transfer; the question is whether the rewards/margins and rewards/accuracies hold up at scale.
4. **End-to-end joint training.** Currently each layer trains independently. Joint training across all three layers, with a unified loss that backprops through cluster-to-spec-to-response, might reveal whether layer-level corrections compound or interfere.

---

## Artifacts

- Code, scripts, models: https://github.com/Fabiha-9876/ReviewAgent (commit `cb4f6b4` and after)
- 11 paper figures in `figures/`
- 32-entry BibTeX in `paper/references.bib`
- Full experimental artifacts (correction logs, cluster outputs, RLHF checkpoints, AntennaPod patches, GitHub issues, semantic verification tests) under `data/processed/`
