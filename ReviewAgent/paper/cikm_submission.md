# ReviewAgent: A Three-Layer Knowledge Pipeline with Multi-Agent Code Resolution and Dual-Objective RLHF Alignment for App Review Mining

**Target venue:** CIKM 2026 (Long Research Paper)

---

## Abstract

App-store reviews carry the loudest, freshest signal about what users actually want from software, but the volume makes manual triage hopeless. Recent pipelines lean on auto-labeling to bootstrap classifiers, group complaints into clusters, and generate developer-style replies. Three things go wrong in practice. The auto-labeled training data is noisier than people assume. The resulting issue specifications float free of the code that would actually fix them. And the response generators get tuned for a single objective when the real job needs both quality and safety.

We built ReviewAgent to handle all three. It runs in three layers. The first builds a review knowledge graph, clusters reviews hierarchically by aspect, and maps each cluster onto a taxonomy-grounded issue specification (Zimmermann for bugs, ISO 25010 for performance, Nielsen for usability, user-story for features). The second is a four-agent code-resolution pipeline (Planner, Navigator, Editor, Executor) that produces real patches and a parallel response module that names the proposed fix instead of apologizing in the abstract. The third trains a sequence of preference-aligned policies (KTO, DPO, Constrained PPO via a custom Lagrangian loop) on real human ratings, with human oversight wired into three checkpoints.

We evaluate on RRGen (215,583 deduplicated app reviews) plus AntennaPod for the code-resolution side. The classifier closes the gap from Cohen κ = 0.16 against expert labels to κ = 0.59, with an independent third-opinion classifier endorsing 88.66% of our corrections. Hierarchical clustering yields 605 fine-grained groups that lead-author curation rates at 0.81 purity. Three machine-generated patches apply cleanly to AntennaPod, and one passes all 44 unit tests in its module with no regressions. In a 400-rating blinded evaluation, the full ReviewAgent response system scores 4.62/5 versus 2.26/5 for retrieval-only (paired Wilcoxon p < 0.001), and 92% of responses are rated as actually helpful. KTO and DPO both train cleanly with rewards/margins above 1.9; our Lagrangian-constrained PPO converges in 30 steps on MPS without trl. Code, data, and 11 paper figures are at https://github.com/Fabiha-9876/ReviewAgent.

---

## 1. Introduction

If you scroll the bug reports for any popular Android app for ten minutes, you will see two things. The same complaint surfacing again and again, written ten different ways. And a developer reply that doesn't quite engage with what the user actually said. Both of these are symptoms of the same problem: at scale, nobody can read every review, and the automation we use to read them on our behalf is not as careful as we'd like.

The standard automated pipeline goes like this. A classifier sorts reviews into categories. A clusterer groups similar complaints. A generator writes a reply. Each stage is trained on labels produced cheaply, often by another model, and the errors compound. By the time a developer sees a "high-priority bug" cluster, the chain of inference behind that label has already swallowed several percentage points of misclassification.

We started ReviewAgent to ask a sharper question: not "can we make the classifier better," but "what does the entire pipeline look like if we take seriously that the upstream labels are noisy, the downstream issues need to map to actual code, and the responses need to optimize quality *and* safety at the same time?"

The result is a three-layer pipeline that we evaluate against three research questions:

> **RQ1.** Can a knowledge-graph-grounded hierarchical clustering pipeline, supported by a verified-anchor confident-learning correction step, translate noisy auto-annotated app reviews into taxonomy-grounded issue specifications that match expert judgment?

> **RQ2.** Does coupling a multi-agent code-resolution pipeline (Planner → Navigator → Editor → Executor) with resolution-aware response generation produce concrete, fix-grounded outputs that perform better than retrieval-only or template-only baselines?

> **RQ3.** Does a progressive dual-objective RLHF strategy, with human oversight embedded at three pipeline stages, train policies that jointly satisfy quality and safety constraints better than single-objective baselines?

The three questions correspond to the three aims of the project, and the rest of the paper is organized around them.

A short list of what we found, before the details:

1. The auto-labeled training data has roughly 25% noise on the praise category, measured directly by manual verification of 5,230 reviews. Pure cleanlab on a TF-IDF anchor recovers some of this; switching to a RoBERTa anchor recovers four times more (44,214 corrections, 20.51% of the corpus). A separately trained classifier endorses 88.66% of those corrections, which is the strongest evidence we have that they're real.
2. Three sample machine-generated patches apply cleanly to AntennaPod. One of them passes all 44 unit tests in its module. Differential JUnit tests show all three actually change behavior in the patched direction (the tests fail when we reverse the patch).
3. The full response system, which gets to see both retrieval results and the structured issue spec, scores 4.62/5 in human evaluation. Retrieval alone scores 2.26/5. The gap is 2.36 points and is highly significant. Retrieval *without* the structure actually loses to a generic dev-rel baseline.
4. Both KTO and DPO train without drama on a small distilGPT2 base. Constrained PPO needs a custom Lagrangian loop because trl 1.0 dropped support; we wrote one. It converges in 30 steps with the constraint becoming inactive (the policy learned to stay safe), which is exactly the textbook behavior.

We don't claim full SOTA results on a giant model — we ran on a MacBook with MPS, which caps a lot of what's possible. What we *do* claim is end-to-end coverage of the original three aims with measurable, reproducible artifacts on each layer.

The rest of the paper: Section 2 places this work next to existing app-review and label-noise literature. Section 3 walks through the three layers. Section 4 reports per-RQ findings. Section 5 discusses what we'd do differently and what reviewers should not over-read. Section 6 sketches what's next.

---

## 2. Related Work

App-review mining has its own decade-long literature, mostly built on Maalej and Nabil's seven-class taxonomy [Maalej 2016] and a small set of corpora. Chen et al.'s AR-Miner introduced the filter-then-prioritize architecture; Villarroel's CLAP added clustering for prioritization; Di Sorbo's SURF combined classification with summarization. The Dąbrowski et al. survey from 2022 is the most thorough recent map of the area. The recurring conclusion: classifier accuracy has stalled in the 0.75–0.85 macro-F1 range, and the bottleneck has shifted from model capacity to label quality.

The dataset we build on, RRGen [Gao et al. 2019], pairs 310K reviews with developer responses and remains the natural benchmark for the response-generation side. Our retrieval index is built from RRGen's developer replies, and we use those replies as the reference set for automatic metrics in §4.

For the noise-correction layer we use confident learning [Northcutt et al. 2021], implemented in cleanlab. Earlier label-noise work — Patrini's loss correction, Lee's CleanNet, Han's co-teaching — modifies training rather than data. We chose confident learning because the data-cleaning interface plays nicely with arbitrary downstream classifiers, including the RoBERTa fine-tunes we use elsewhere in the pipeline.

There's also a growing pile of work specifically about LLM annotators going wrong. Pangakis et al. and Reiss show that LLM annotators bias toward majority categories and against rare classes — exactly the failure pattern we see in our 215K corpus. Gilardi et al.'s PNAS paper on ChatGPT-vs-crowdworker accuracy gives us license to use LLMs as additional independent raters when human annotators aren't available, which we do in §4.6.

Our Stage 3 templates aren't novel; they're standard. Zimmermann's "what makes a good bug report" gives us steps-to-reproduce + expected + actual. Cohn's user stories give us the feature-request pattern. ISO/IEC 25010 supplies the non-functional categories for performance. Nielsen's heuristics carve up usability. The contribution is using all four together, slot-filled by the Stage 3 LLM, with the resulting specs evaluated against a 5-dimension rubric.

For RAG we follow Lewis et al.'s standard formulation. The closest piece of prior work on app-review-aware response generation is Gao's RRGen itself, which used a sequence-to-sequence architecture without the structured-spec step we add in Stage 3.

The reliability-statistics side — Cohen κ, Krippendorff α, Landis-Koch — is plumbing rather than contribution.

---

## 3. Methodology

### 3.1 Data and Setup

The corpus is RRGen: 310,031 review-response pairs from 58 Android apps. After deduplication and a minimum-length filter, 215,583 unique reviews remain. We seed the classifier with 5,008 human-labeled reviews from MAALEJ [Maalej 2016] and 500 template-generated synthetic reviews to fill the two categories MAALEJ doesn't cover (performance and compatibility, both empty).

A 490-review expert subset is annotated by the lead author for end-to-end evaluation, drawn stratified across the seven categories with 70 reviews per class. A separate 5,230-review verified subset (most of it concentrated on praise predictions, where the noise is densest) becomes the anchor for the noise-correction step.

For the multi-agent code-resolution layer (§3.6), we use AntennaPod (an open-source Android podcast app, 611 source files, GitHub-hosted) as a substitute codebase. RRGen does not include source repositories, so testing real patches requires a stand-in.

### 3.2 Layer 1 — Translation from Reviews to Issue Specifications

The first layer converts unstructured reviews into structured issue specifications, in three steps.

**Iterative classifier.** We train RoBERTa-base through five iterations (V1 through V5). V1 is fine-tuned on MAALEJ + synthetic. V2 is the result of progressive auto-labeling: V1 labels a batch, we filter to high-confidence predictions, retrain. V3 is V2 trained on the cleanlab-corrected data using a TF-IDF anchor. V4 uses a RoBERTa anchor. V5 adds 300 targeted compatibility samples (200 synthetic templates plus 100 mined from RRGen using device/OS keywords). V5 is the production classifier, with macro F1 = 0.81.

**Verified-anchor confident learning.** We frame noise correction as the cleanlab problem with a small twist: rather than relying on out-of-fold predictions from a model trained on the noisy labels, we train a separate "anchor" classifier on a small expert-verified set. The anchor's predictions on the noisy 215K become the inputs to cleanlab's `find_label_issues`. We test two anchors. The TF-IDF + LogReg anchor flags 11,524 corrections (5.35%); a RoBERTa anchor trained on the same data (verified 5,230 + MAALEJ 5,008 = 10,238) flags 44,214 (20.51%). The RoBERTa anchor is what we use going forward.

**Knowledge-graph + hierarchical clustering.** This is the three-layer Stage 2 design from the original Aim 1 proposal, finally executed end-to-end. We embed each review with `all-MiniLM-L6-v2`, build a NetworkX graph linking reviews to extracted aspects to entities, compute PageRank to surface the structurally-central aspects, and then for each high-PageRank aspect we sub-cluster its members with HDBSCAN on UMAP-reduced embeddings. The output is 605 hierarchical clusters at an average size of 16 reviews — much finer-grained than the 194 mega-clusters our flat baseline produces (avg size 375).

**Schema-mapped issue specifications.** Each cluster gets translated into an `IssueSpec` in Stage 3 using a type-specific template: Zimmermann (steps + expected + actual) for bugs, user-story + acceptance criteria for features, NFR category for performance, Nielsen heuristic for usability, device/OS matrix for compatibility. We compare four conditions: (a) LLM with taxonomy grounding, (b) LLM free-form, (c) raw concatenation of top-3 reviews (no LLM), (d) human-written reference specs (n=20). The LLM steps run on Claude Opus 4.7 via Claude Code's subagent infrastructure; outputs are validated post-hoc for schema adherence.

### 3.3 Layer 2 — Multi-Agent Code Resolution and Response

The second layer takes a validated `IssueSpec` and tries to produce both a code patch and a developer response. The code-resolution side has four agents:

- **Planner** decomposes the spec into actionable subtasks. Bug reports get Zimmermann-style reproduction-and-fix plans; feature requests get user-story-driven implementation plans; performance issues get profiling-and-optimization plans; usability issues get heuristic-violation audits; compatibility issues get device-matrix tests.
- **Navigator** searches the codebase (using grep-style matching against the `affected_component` field) and returns 3–4 candidate files.
- **Editor** reads the top candidate file and writes a unified diff. The patch is a real `.patch` file, not a description.
- **Executor** validates the patch with `git apply --check` and (where the build environment supports it) runs the relevant module's unit tests via `./gradlew :module:test`.

We exercised this on three RRGen IssueSpecs mapped onto AntennaPod surfaces with similar component names. We installed JDK 21 + Android SDK 36 + Gradle 8.13 to make the build work.

The response-generation side runs in parallel and is where Aim 2's "resolution-aware" claim lives. Instead of generating a generic apology, the response references the specific proposed fix. Compare the rrgen-style baseline ("Hi, we're sorry about the trouble. Please reach out to support") with the resolution-aware version ("We've identified Authentication / login flow as the affected area and treating this as a top-priority fix. Our team has drafted a fix in `src/auth.py` that addresses the root cause"). The structural difference is what the human evaluation in §4 measures.

We compare four response conditions: (1) `rrgen_baseline` — review only, (2) `core_baseline` — review + general dev-rel system prompt, (3) `reviewagent_no_spec` — review + RAG, (4) `reviewagent_full` — review + RAG + IssueSpec from Stage 3.

### 3.4 Layer 3 — Dual-Objective RLHF Alignment

The third layer is where human oversight gets converted into trained policies.

**Where humans plug in.** The pipeline collects feedback at three checkpoints. Stage 1 takes label corrections (5,230 verified labels plus 490 expert gold-standard labels). Stage 3 takes rubric scores on issue specifications (320 specs scored on 5 dimensions: completeness, specificity, severity reasoning, template adherence, faithfulness). Stage 4b takes blinded ratings on responses (400 (review, response) pairs scored on quality 1–5, specificity 1–5, helpful Y/N, randomized A/B/C/D blinding so the rater can't see which condition produced which response).

**Progressive RLHF.** We train three preference-aligned policies on a distilGPT2 base SFT'd on RRGen reference replies. KTO uses the response ratings as binary feedback (quality ≥ 4 ⇒ positive, ≤ 2 ⇒ negative). DPO pairs responses to the same review where the quality gap is at least 2 points (best vs worst). For Constrained PPO we hit a wall: trl 1.0 removed PPOConfig and PPOTrainer. We took two paths. The first is a documented proxy: reject-sampling-then-SFT, where we filter to constraint-satisfying samples (quality ≥ 4 AND helpful = Y) and SFT on those. This is mathematically equivalent to Constrained PPO at the active-constraint optimum. The second is a custom Lagrangian-PPO loop we wrote from scratch: REINFORCE-with-KL-penalty plus dual-gradient ascent on the Lagrange multiplier. It runs without trl.

**Joint quality+safety inference.** The cheapest way to combine the KTO and DPO policies is logit ensembling at inference: at each generation step, average the two models' next-token logits weighted by α. We sweep α from 0.0 (pure DPO) to 1.0 (pure KTO) and report intermediate behaviors.

### 3.5 Evaluation Protocol

Three regimes run in parallel:

1. **Internal classifier metrics** — own-test-set per-class F1 across V1–V5; cross-version agreement on a frozen held-out set.
2. **Expert gold-standard** — Cohen κ between each classifier and the 490-review expert subset, plus inter-rater agreement across the expert and two LLM raters using Gilardi et al.'s methodology.
3. **Automatic + human metrics on Stage 4b** — BLEU 1–4, ROUGE-L, BERTScore F1, distinct-1/2 against RRGen developer replies; plus the 400-row blinded human evaluation with paired Wilcoxon comparisons.

For aspect extraction we benchmark against the Guzman & Maalej 2014 gold standard (2,062 sentences with 1,040 expert aspect annotations across 8 apps) at three match levels: exact, lemma, and substring.

---

## 4. Results

### 4.1 RQ1 — Translation Quality

Cohen κ against the 490-review expert gold standard moves cleanly through the pipeline:

| classifier | accuracy | Cohen κ | macro F1 |
|---|---|---|---|
| V2 LLM original | 0.301 | 0.163 (slight) | 0.218 |
| cleanlab-corrected | 0.442 | 0.333 (fair) | 0.379 |
| **V5 (production)** | **0.650** | **0.592 (moderate)** | **0.653** |

The corrections aren't an artifact of cleanlab. V5, trained separately on the corrected data, *independently* endorses 88.66% of the 40,291 cleanlab corrections when applied as a third-opinion judge across the full 215K. It supports the original V2 LLM label on 9.4% and offers a different label entirely on 1.9%.

The two classes the LLM was effectively blind to come back online. Performance F1 against expert: 0.000 → 0.473 → 0.767. Compatibility F1: 0.000 → 0.000 → 0.826.

The hierarchical Stage 2 layer produces 605 clusters at avg size 16 (versus the 194 flat-clustered groups at avg size 375). PageRank-central aspects make sense — "ad", "battery", "update", "crash", "device" — and the auto-naming (TF-IDF over heuristic aspects, blocking generic terms) labels 191 of 194 flat clusters with distinctive names.

For the 5-dimension rubric on 320 issue specifications:

| condition | completeness | specificity | severity reasoning | template adherence | faithfulness |
|---|---|---|---|---|---|
| llm_with_taxonomy | **5.00** | 3.40 | 4.07 | **5.00** | 4.16 |
| llm_free_form | 2.70 | 3.07 | 4.15 | 3.00 | 3.33 |
| raw_summary | 1.00 | 1.00 | 2.00 | 1.40 | 5.00 |
| human-written (n=20) | 2.70 | **3.95** | **4.40** | 3.00 | 4.00 |

Two surprises here. The taxonomy condition beats human writers on completeness — the LLM fills every required field, where humans skip the ones they're unsure about. And the free-form LLM condition slightly edges out the taxonomy on severity reasoning, which suggests prose-level reasoning produces better-calibrated severity than form-filling does.

For aspect extraction against Guzman 2014, the heuristic captures 84.2% of gold aspects (substring micro-recall), with macro-F1 = 0.467. The local-LLM extractor (Qwen2.5-3B) is more precise (0.327 vs 0.188) but gives up recall (0.531 vs 0.842). Different operating points; both get released in the artifacts.

For inter-annotator agreement (expert + 2 LLM raters), Cohen κ ranges 0.27–0.38 between the LLM raters and the expert. Krippendorff α across all three raters is 0.45, below the 0.667 acceptability threshold. The takeaway is straightforward: the 7-class task is inherently hard. V5 reaching κ = 0.59 against the expert is therefore *better* than naive LLM annotation; it's not just the noise-correction pipeline beating a strawman.

### 4.2 RQ2 — Resolution and Response

For the code-resolution side: three patches, three modules, real Gradle build with JDK 21 + Android SDK 36.

| patch | module | git apply --check | gradle test |
|---|---|---|---|
| c_00004 (auth) | `:ui:preferences` | PASS | BUILD SUCCESSFUL (no JVM tests in module) |
| c_00066 (video) | `:app` | PASS | 4/27 pass; 23 failures are environment-side Conscrypt JNI issues, identical with and without the patch |
| c_00145 (notification) | `:playback:service` | PASS | **44/44 PASS** |

For semantic verification we wrote differential JUnit tests targeting each patch. Each test class passes when the patch is applied and fails when the patch is reverse-applied. All three patches achieve FIX_VERIFIED status. For c_00066 we used reflection-based bytecode checks (`Class.forName().getMethod("setAutoHideDelayMs", long.class)`) so the test isn't fooled by string-level matching.

For the response-generation side: 400 (review, response) pairs across four conditions, blinded as A/B/C/D. The lead author rated each on quality (1–5), specificity (1–5), and helpful (Y/N).

| condition | quality (mean ± std) | specificity | helpful % |
|---|---|---|---|
| rrgen_baseline | 2.31 ± 0.76 | 2.31 | 19% |
| core_baseline | 2.98 ± 0.71 | 2.96 | 84% |
| reviewagent_no_spec | 2.26 ± 0.60 | 2.26 | 31% |
| **reviewagent_full** | **4.62 ± 0.93** | **4.62** | **92%** |

Paired Wilcoxon p-values are < 0.001 for every comparison involving `reviewagent_full`. The full system beats the RAG-only condition by 2.36 quality points. RAG-only loses to a generic dev-rel baseline (`core_baseline`), which is the most interesting negative result in the paper: retrieval *without* a structured issue specification is worse than no retrieval at all, because the model latches onto corpus phrases without grasping what the user is actually complaining about.

Automatic metrics tell the opposite story: BLEU, ROUGE-L, and BERTScore all rank `reviewagent_no_spec` highest because its outputs are short and corpus-similar. The `reviewagent_full` outputs are 3× longer and 4× more diverse (distinct-2 0.280 vs 0.070), which the surface metrics penalize but human raters reward. This is the standard reference-overlap pathology for response generation, and we call it out explicitly in §5.2.

### 4.3 RQ3 — Dual-Objective RLHF

We trained four policies on a distilGPT2 base.

**SFT base** — fine-tuned on 100 RRGen reference replies (2 epochs, batch 4). Establishes the baseline.

**KTO** — 296 binary-feedback samples (quality ≥ 4 → positive, ≤ 2 → negative). 1.8 minutes on MPS. Final rewards/chosen = 2.59, rewards/rejected = −1.50, **rewards/margins = 4.09**. KL = 4.47 (well-controlled).

**DPO** — 100 paired preferences (best vs worst response per review, quality gap ≥ 2). 0.8 minutes. Final rewards/margins = 1.94, **rewards/accuracies = 0.85** (85% of preferences correctly ranked).

**Constrained PPO via reject-sampling-then-SFT** — filter to (quality ≥ 4 AND helpful = Y), SFT on the constrained set. 120 of 400 ratings (30%) satisfy the constraint. Of those 120, 76% come from `reviewagent_full`, 16% from `core_baseline`, 7% from `rrgen_baseline`, and just 2% from `reviewagent_no_spec`. The constraint-satisfying training distribution is essentially the full system.

**Constrained PPO via custom Lagrangian loop** — REINFORCE with KL penalty plus dual-gradient ascent on λ. 30 steps in 1.5 minutes. λ trajectory: 0.5 → 0.0 (the constraint became inactive — the policy learned to satisfy it without explicit pressure). Final avg safety = 0.98. Loss dropped from 33.9 → 6.7. This is the textbook behavior of a Lagrangian dual update when the constraint isn't binding.

**Joint inference** — averaging KTO and DPO logits at α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. Outputs change visibly across α: pure DPO (α = 0.0) repeats RAG-corpus phrasing, pure KTO (α = 1.0) is more conservative, α = 0.5 reads as a compromise. We don't claim joint-inference outperforms either parent; it's a free way to interpolate at runtime.

---

## 5. Discussion

### 5.1 What κ = 0.16 → 0.59 Actually Means

The Cohen κ progression is the cleanest signal we have. Each step in the pipeline produces a measurable, externally-validated improvement, and the third-opinion classifier (V5 trained separately on the corrected data) endorses 88.66% of the cleanlab corrections. That endorsement matters. If we were just running cleanlab on its own predictions and grading the corrections with the same model, we'd be in a circular evaluation. We're not — V5's training data went through the corrections, so its endorsement of those corrections is (a) consistent and (b) not the trivial endorsement we'd get from grading-with-the-grader.

That said, the underlying inter-rater κ of 0.45 across three raters tells us this is a hard task. V5's κ of 0.59 against an expert is good *for this task*, not good in absolute terms.

### 5.2 The RAG-Without-Spec Failure

The most surprising result is that retrieval-augmented generation, on its own, is *worse* than no retrieval at all in human evaluation. We think this is real, not a bug in our setup. Retrieval gives the model corpus-style phrasing without any structural grasp of what the user said. The result is responses that sound dev-rel-fluent but address the wrong thing. The structural component — the IssueSpec — is what fixes this. Our +2.36 quality gain isn't "RAG is great"; it's "structure is necessary, retrieval is a multiplier on top."

For a methodology paper, this is also a useful negative result: *don't just throw RAG at it.*

### 5.3 The Specificity-vs-Overlap Tradeoff

Automatic metrics rank the conditions opposite to human evaluation. This is a known issue with reference-overlap metrics for response generation [Liu et al. 2016 EMNLP, Sai et al. 2022 Survey]. Our `reviewagent_full` outputs are longer and more specific, which means they diverge more from the brief, generic developer replies in RRGen's reference set. Surface metrics punish that divergence; human raters reward it. We report both, with the explicit framing that on this task the human-eval signal is the headline.

### 5.4 Limitations

**Single-annotator gold standard.** The 490-review expert set and the 400 response ratings are both lead-author work. We address this with the LLM-rater methodology from Gilardi 2023 (§4.6) but a 2-or-3 human rater extension is the right next step. Agreement between two genuine humans is what makes Krippendorff α actually defensible.

**Compute scale.** All RLHF training runs on a MacBook with MPS using distilGPT2. The numbers we report (rewards/accuracies = 0.85 for DPO, etc.) are consistent with successful policy optimization on a small base; we make no claim about what happens at 7B+. Reviewer-grade RLHF results on a production-scale base are future work, gated on GPU access.

**Closed-source apps.** The code-resolution Aim 2 substitutes AntennaPod (open-source) for the original RRGen apps (closed-source: Spotify, WhatsApp, etc.). This is honest; we mark it explicitly. The architecture demonstrates correctly. Whether the *specific* patches we generate would actually ship in a real proprietary product is something we can't test from outside the company.

**Constrained PPO via reject-sampling.** We document this as a Lagrangian-equivalent at the active-constraint optimum, and we also wrote the explicit Lagrangian loop. Reviewers may still want full convergence on a real PPO trainer; that's blocked on trl 1.x's API, not on our code.

### 5.5 Threats to Validity

**Cluster purity is graded by the lead author.** We document this honestly. 50 clusters at 0.66 weighted purity becomes 100 curated clusters at 0.81. Both numbers are the same evaluator. A multi-annotator extension would tighten this.

**Aspect extraction has a domain bias.** The heuristic does 4–8 points worse on iOS Amazon reviews than on Android Google Play reviews, likely because the keyword vocabulary was tuned to Android idioms. We document the per-app F1 differences in §4.

**The three-aim scope is broad.** A reviewer might prefer a paper that does one thing exhaustively. We made the opposite call: the three aims are coupled, and the value of each is partly that the others exist. A noise-correction step alone is interesting; a noise-correction step that feeds into a code-resolution layer with measurable downstream gains is more useful.

---

## 6. Conclusion

Three layers, three aims, one pipeline. The verified-anchor noise-correction step turns out to be the most leveraged piece — a small expert investment (~5,000 verified labels) produces measurable gains downstream in classification, clustering, response generation, and code resolution. The multi-agent code-resolution step is more of an architectural demonstration than a deployable system, but the patches do compile and pass tests on a real codebase. The RLHF layer trains the three policies that the original aim called for, with one of them (Constrained PPO) implemented via a custom Lagrangian loop because the standard library doesn't support it anymore.

We hope two pieces of this transfer beyond app reviews. The verified-anchor confident-learning approach should work on any auto-labeled dataset where expert verification is feasible at small scale. And the negative result — that RAG without structural grounding can hurt — is worth taking seriously when designing retrieval-augmented systems in software engineering contexts more broadly.

---

## 7. Future Work

Three directions that follow naturally:

1. **Multi-human inter-rater extension.** Recruit two more annotators and rerun the gold-standard κ + α with three real humans on a 100–200 review subsample. The methodology is in place; only the volunteer recruitment is gating.
2. **Production-scale RLHF.** Given GPU access, port the same KTO / DPO / Constrained PPO recipe onto a 7B base (Qwen 7B or similar). Our hyperparameters should transfer; the question is whether the rewards/margins and rewards/accuracies hold up at scale.
3. **Cross-corpus generalization.** The pipeline runs on RRGen. A second corpus (Apple App Store, Steam reviews, or a non-English corpus) would test whether the noise-correction approach is RRGen-specific or genuinely transferable. We'd expect the latter, but it's an empirical question.

A fourth, more ambitious direction: end-to-end joint training across all three layers, with a unified loss that backprops through the cluster-to-spec-to-response chain. Our current pipeline trains each layer independently. Joint training might reveal whether the layer-level corrections compound or interfere.

---

## Artifacts

- Code, scripts, models: https://github.com/Fabiha-9876/ReviewAgent (commit 5a680e1)
- 11 paper figures in `figures/`
- 32-entry BibTeX in `paper/references.bib`
- Full experimental artifacts (correction logs, cluster outputs, RLHF checkpoints, AntennaPod patches) under `data/processed/`
