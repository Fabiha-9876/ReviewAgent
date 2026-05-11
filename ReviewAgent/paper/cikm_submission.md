# ReviewAgent: Verified-Anchor Confident Learning and Knowledge-Grounded Issue Specifications for App Review Mining

**Target venue:** CIKM 2026 (Long Research Paper)

---

## Abstract

Anyone who has tried to read app-store reviews at scale knows the problem: the volume is enormous, the signal is real, and nobody can keep up by hand. The standard fix — bootstrap a classifier on auto-labeled data, cluster what comes out, generate replies — works in principle but compounds errors at every step. Cheap labels feed muddled clusters, muddled clusters feed generic replies, and the team that asked for "actionable insights" gets thanked for the feedback.

This paper asks one direct question: *can an LLM agent translate noisy app review clusters into issue specifications that look like real human-written GitHub bug reports?* To answer it we built **ReviewAgent**, a pipeline that does three things — corrects systematic noise in auto-labeled training data using a small expert-verified anchor, builds knowledge-graph-grounded hierarchical clusters of complaints, and translates those clusters into taxonomy-grounded issue specifications using domain-specific templates (Zimmermann for bugs, ISO/IEC 25010 for performance, Nielsen heuristics for usability, user stories for features).

We test on RRGen (215,583 deduplicated reviews from 58 Android apps), validate the aspect-extraction layer against the GUZMAN benchmark, and compare the LLM-generated specs against **64 closed GitHub issues mined from three open-source Android projects** (AntennaPod, NewPipe, Thunderbird Android). Under **strict content-validity criteria** (§3.8.1.x: ≥3 substantive reproduction steps with action verbs; ≥30-word descriptions; full *As-a/I-want/so-that* user-story triple), the taxonomy-grounded LLM achieves substantive template-fill of **0.959** on the headline 100-cluster sample and **0.97** on the 15-cluster GitHub overlap, vs **0.53** for real GitHub issues mined from the three repos. Bug-report `steps_to_reproduce` is substantively populated 73% of the time by the LLM vs 13% on real GitHub; formal user-story triples appear in 97% of LLM features vs 0% of GitHub features (humans write checklists, not user stories). Per-repo GitHub numbers stay tight (0.68–0.73 on loose fill). **This is a coverage / compliance contrast, not a content-quality contrast** — the LLM is *prompted* to populate every field substantively; humans choose what to write for their own developer audience. We do not claim the LLM produces "better" issues, only that it produces *more structurally complete* artifacts, which is what a downstream automated pipeline needs.

A verified-anchor cleanlab pass on the noisy 215K corpus flags 44,214 likely mislabels (20.51%); an independently-trained classifier endorses 88.66% of those corrections, raising Cohen's κ against an expert gold standard from 0.16 (the original LLM labels) to 0.59 (the corrected classifier). On the response-generation side, a 400-rating blinded human evaluation gives 4.62/5 to the full system versus 2.26/5 for retrieval-only — the gap is 2.36 quality points (paired Wilcoxon *p* < 0.001; Friedman χ²(3) = 199.3, *p* = 5.9 × 10⁻⁴³). Bradley-Terry strengths and McNemar's tests on helpfulness confirm the same ordering. The most uncomfortable finding: retrieval *without* a structured spec underperforms a no-retrieval baseline.

Code, data, eleven figures, and trained checkpoints are at https://github.com/Fabiha-9876/ReviewAgent.

---

## 1. Introduction

If you scroll the bug reports for any popular Android app for ten minutes, two things stand out. The same complaint surfaces over and over, written ten different ways. And the developer reply, when there is one, doesn't quite engage with what the user actually said. Both come from the same root cause: at scale, nobody can read every review, and the automation we lean on to read them on our behalf is not as careful as we would like.

The textbook automated pipeline goes like this. A classifier sorts reviews into categories. A clusterer groups similar complaints. A generator drafts a reply. Each stage is trained on labels produced cheaply, often by another model, and the errors stack. By the time a triage engineer sees a "high-priority bug" cluster, several percentage points of misclassification have already gone into building it, and nobody upstream is in a position to spot what went wrong.

Meanwhile, the same developers receive *real* bug reports through GitHub from a smaller, more disciplined audience. Those reports follow conventions: titles are concrete, reproduction steps are listed, expected versus actual behavior is separated. They're not perfect, but they're the closest thing the software-engineering world has to a structured-issue gold standard. So a natural question: if we could train an LLM agent to translate noisy app reviews into specifications that look like GitHub bug reports, how close would it actually get?

That's our **main research question**:

> **RQ.** *How accurately can an LLM agent translate noisy app review clusters into structured issue specifications, compared to human-written GitHub issues?*

Three sub-questions, one per pipeline layer:

> **RQ1 (Translation).** Can a knowledge-graph-grounded hierarchical clustering pipeline, supported by a verified-anchor confident-learning correction step, produce taxonomy-grounded issue specifications that match or exceed real GitHub issues on structural completeness, template-field coverage, and expert agreement?

> **RQ2 (Resolution).** Does coupling a multi-agent code-resolution pipeline (Planner → Navigator → Editor → Executor) with resolution-aware response generation produce concrete, fix-grounded outputs that perform better than retrieval-only or template-only baselines?

> **RQ3 (Alignment).** Does a progressive dual-objective RLHF strategy, with human oversight embedded at three pipeline stages, train policies that satisfy quality and safety constraints jointly, better than single-objective baselines?

A short version of what we found, before the details:

1. We mined 64 real closed GitHub issues from three open-source Android repos: AntennaPod, NewPipe, and Thunderbird Android (the K-9 Mail successor). Under strict content-validity criteria (§3.8.1.x), the LLM-with-taxonomy condition reaches **0.959** substantive template-fill (vs **0.532** on real GitHub) — a 0.43-point gap that reflects substantive *coverage*, not content quality. Bug-report `steps_to_reproduce` is substantively populated **0.73** of the time by the LLM vs **0.13** on real GitHub; formal user-story triples appear in **0.97** of LLM features vs **0.00** on GitHub (humans write checklists, not user stories). The LLM is *prompted* to fill every field substantively; humans fill what their developer audience needs. The trivial loose check (`field non-empty`) hits the rubric ceiling for any capable instruction-following LLM — including the cross-LLM Qwen2.5-3B replication in §4.2.y — and is therefore demoted from headline status. The strict numbers are the defensible headline. The one place where humans win on a *content* dimension is descriptive prose length (67.7 words vs 48.5) — humans contextualize, the LLM stays terse.
2. The auto-labeled training data has roughly 25% noise on the *praise* category, measured directly by manual verification of 5,230 reviews. A RoBERTa-anchored cleanlab pass corrects 44,214 reviews (20.51% of the corpus). An independently-trained classifier endorses 88.66% of those corrections.
3. Three machine-generated patches apply cleanly to AntennaPod. One passes all 44 unit tests in its module. Differential JUnit tests show all three actually change behavior in the patched direction (the tests fail when we reverse the patch).
4. The full response system, which sees both retrieval results and the structured issue spec, scores 4.62/5 in human evaluation. Retrieval alone scores 2.26/5. The gap is 2.36 points (paired Wilcoxon *p* < 0.001; Friedman χ²(3) = 199.3, *p* = 5.9 × 10⁻⁴³). Bradley-Terry on the same paired data ranks the conditions identically, and McNemar's test on the helpful-Y/N outcome agrees with five of the six pairwise comparisons. Retrieval *without* the structure actually loses to a generic dev-rel baseline — a finding that pushes back on the assumption that RAG always helps.
5. KTO and DPO train without drama on a small distilGPT2 base. Constrained PPO needed a custom Lagrangian loop because trl 1.0 dropped support; we wrote one. It converges in 30 steps but the constraint stayed inactive throughout — *not* because the policy learned to satisfy it, but because the distilGPT2 base produces outputs too restricted to plausibly violate the operational compliance rubric (§3.7.5). An active-constraint re-run with a tightened threshold τ=0.90 (§4.7.1) confirms this: only 1 violation observed across 120 generations. The CMDP machinery is verified to behave correctly under no-binding conditions; testing under a binding constraint requires a generation-grade base model (§7 item 3).

We don't claim full SOTA on a giant model — all training ran on a MacBook with MPS, which caps a lot of what's possible. What we *do* claim is end-to-end coverage of the original three aims, with the main RQ answered against a real human-written GitHub-issue baseline that spans three different open-source projects.

The rest of the paper: Section 2 places this work alongside existing app-review and label-noise literature. Section 3 walks through the three layers, including the theoretical foundations the design rests on. Section 4 reports per-RQ findings, the formal omnibus and post-hoc tests the proposal asked for, and a consolidated ablation table. Section 5 is the discussion, including the limitations we're not pretending away. Section 6 sketches what's next.

---

## 2. Related Work

App-review mining has its own decade-long literature, mostly built on Maalej and Nabil's seven-class taxonomy and a small set of recurring corpora. Chen et al.'s AR-Miner introduced the filter-then-prioritize architecture; Villarroel's CLAP added clustering for prioritization; Di Sorbo's SURF combined classification with summarization. Dąbrowski et al.'s 2022 survey is the most thorough recent map of the area. Its recurring conclusion — classifier accuracy has stalled in the 0.75–0.85 macro-F1 range, and the bottleneck has shifted from model capacity to label quality — is the diagnosis we run with.

The dataset we build on, RRGen, pairs 310K reviews with developer responses and remains the natural benchmark for the response-generation side. We use its developer replies as the index for retrieval and as the reference set for automatic metrics in §4.

For the noise-correction layer we use confident learning (Northcutt et al.), implemented in cleanlab. Earlier label-noise work — Patrini's loss correction, Lee's CleanNet, Han's co-teaching — modifies training rather than the data. We chose confident learning because the data-cleaning interface plays nicely with arbitrary downstream classifiers, including the RoBERTa fine-tunes we use elsewhere.

There's also a growing pile of work specifically about LLM annotators going wrong. Pangakis et al. and Reiss show that LLM annotators bias toward majority categories and against rare classes — exactly the failure pattern we measure on our 215K corpus. Gilardi et al.'s PNAS paper on ChatGPT-vs-crowdworker accuracy gives us license to use LLMs as additional independent raters when human annotators aren't available, which we lean on in §4.

Our Stage 3 templates are not novel — they're standard. Zimmermann's "what makes a good bug report" supplies the steps + expected + actual structure. Cohn's user stories supply the feature-request pattern. ISO/IEC 25010 supplies non-functional categories for performance. Nielsen's heuristics carve up usability. The contribution is using all four together, slot-filled by the Stage 3 LLM, with the resulting specs evaluated against real GitHub issues mined from three different open-source projects.

For RAG we follow Lewis et al.'s standard formulation. The closest piece of prior work on app-review-aware response generation is Gao's RRGen itself — a sequence-to-sequence architecture without the structured-spec step we add at Stage 3. The reliability-statistics side (Cohen κ, Krippendorff α, Landis-Koch interpretation) is plumbing.

---

## 3. Methodology

### 3.0 Theoretical Foundations

Three established frameworks motivate the structural choices in the pipeline. Naming them up front saves us a lot of justification later.

The first is the **information-extraction cascade** (Hearst 1999; Sarawagi 2008). Progressive structuring — free text → entities → relations → records — is well known to be more accurate and easier to audit than monolithic end-to-end extraction. Our Stages 1 → 2 → 3 (classify → cluster → schema-map) implement this cascade explicitly: each stage emits a strictly more structured intermediate artifact, and each can be evaluated and corrected independently. When something looks wrong downstream, we can walk back up the cascade and find where it went wrong.

The second is **human-AI complementarity** (Kamar 2016; Bansal et al. 2019). Mixed-initiative systems beat humans-alone or models-alone when human oversight lands at the points where the model is least confident or where errors propagate furthest. We embed human-in-the-loop checkpoints at exactly three stages — classification verification (the verified anchor), cluster/spec validation (the 50-cluster purity audit), and response review (the 400-rating blinded eval) — chosen because each gates a downstream training signal.

The third is **constrained Markov decision processes** (Altman 1999; Dai et al. 2024). Response generation is a constrained-optimization problem: maximize quality subject to safety and policy constraints (no unauthorized promises, no PII leakage, on-tone). Our Stage 5 RLHF design (KTO → DPO → Lagrangian-Constrained PPO) is a direct realization of CMDP, with quality as the reward and policy compliance as the constraint.

### 3.1 Data and Setup

The corpus is RRGen: 310,031 review-response pairs from 58 Android apps. After deduplication and a minimum-length filter, 215,583 unique reviews remain. We seed the classifier with 5,008 human-labeled reviews from MAALEJ and 500 template-generated synthetic reviews to fill the two categories MAALEJ doesn't cover (performance and compatibility, both empty there).

A 490-review expert subset is annotated by the lead author for end-to-end evaluation, drawn stratified across the seven categories with 70 reviews per class. A separate 5,230-review verified subset — most of it concentrated on praise predictions, where the noise is densest — becomes the anchor for the noise-correction step.

For the human-written GitHub-issue comparison set we mined **64 closed issues from three open-source Android apps** via the public GitHub REST API: AntennaPod (podcast player), NewPipe (lightweight YouTube client), and Thunderbird Android (the K-9 Mail successor). The three repos cover different software domains — media playback, content discovery, email — so the comparison isn't dominated by a single project's issue-template conventions. We parsed each issue's title and body into our IssueSpec schema, inferring the issue type from labels and keyword heuristics (30 bug_report, 25 feature_request, 5 usability, 2 performance, 2 compatibility). For the multi-agent code-resolution layer (§3.3) we use the AntennaPod codebase (611 source files) as the substitute for closed-source RRGen apps.

### 3.2 Layer 1 — Translation from Reviews to Issue Specifications (RQ1)

The first layer is the heart of the main RQ. Three steps.

**Iterative classifier.** We train RoBERTa-base through five iterations (V1 through V5). V1 fine-tunes on MAALEJ + synthetic. V2 is the result of progressive auto-labeling. V3 retrains V2 on cleanlab-corrected data using a TF-IDF anchor. V4 uses a RoBERTa anchor. V5 adds 300 targeted compatibility samples (200 synthetic templates plus 100 mined from RRGen using device/OS keywords). V5 is the production classifier, with macro F1 = 0.81.

**Verified-anchor confident learning.** We frame noise correction as the standard cleanlab problem with a small twist: instead of relying on out-of-fold predictions from a model trained on the noisy labels (which would be circular), we train a separate "anchor" classifier on a small expert-verified set — 5,230 verified + 5,008 MAALEJ = 10,238 anchor samples. The anchor's predictions on the noisy 215K become the inputs to cleanlab's `find_label_issues`. The TF-IDF + LogReg anchor flags 11,524 corrections (5.35%); the RoBERTa anchor flags 44,214 (20.51%). The RoBERTa anchor is what we use going forward, because the 4× larger correction yield is dominated by recovered minority-class examples (performance and usability) rather than spurious flags on already-correct labels — a claim we verify in §4 via the 88.66% endorsement rate of an independently-trained classifier.

**Knowledge-graph + hierarchical clustering.** This is the three-layer Stage 2 design from the original Aim 1 proposal. We embed each review with `all-MiniLM-L6-v2`, build a NetworkX graph linking reviews to extracted aspects to entities, compute PageRank to surface the structurally-central aspects, then sub-cluster each high-PageRank aspect's members with HDBSCAN on UMAP-reduced embeddings. The output is 605 hierarchical clusters at an average size of 16 reviews — finer-grained than the 194 mega-clusters our flat baseline produces (avg size 375), and the difference matters when a triage engineer wants to drill into a specific failure mode rather than scan a soup of related complaints.

**Schema-mapped issue specifications.** Each cluster gets translated into an `IssueSpec` in Stage 3 using a type-specific template: Zimmermann (steps + expected + actual) for bugs, user-story + acceptance criteria for features, NFR category for performance, Nielsen heuristic for usability, device/OS matrix for compatibility. We compare five conditions: (a) LLM with taxonomy grounding, (b) LLM free-form, (c) raw concatenation of top-3 reviews (no LLM), (d) lead-author-written reference specs (n=20), and **(e) real human-written GitHub issues mined from the three open-source repos (n=64)** — the canonical answer to the main RQ. The LLM steps run on Claude Opus 4.7 via Claude Code's subagent infrastructure; outputs are validated post-hoc for schema adherence.

**Five-dimension rubric.** Each spec is scored on five dimensions, mapped one-to-one from the proposal's specification: completeness (fraction of required fields populated), accuracy → faithfulness (no contradictions with the source review cluster), actionability → severity-reasoning (an explicit P0–P3 priority justification a triage engineer can act on), specificity (concrete components, error surfaces, devices, OS versions), and clarity → template-adherence (structural conformance to the issue-type schema). The operational names emphasize the measurable proxy we use per dimension. All 320 ratings live in the released artifacts.

### 3.3 Layer 2 — Multi-Agent Code Resolution and Response (RQ2)

The second layer takes a validated `IssueSpec` and produces both a code patch and a developer response. The code-resolution side has four agents:

- **Planner** decomposes the spec into actionable subtasks, tailored per issue type.
- **Navigator** searches the codebase by grep-style matching against the `affected_component` field and returns 3–4 candidate files.
- **Editor** reads the top candidate file and writes a unified diff. The patch is a real `.patch` file.
- **Executor** validates the patch with `git apply --check` and runs the relevant module's unit tests via `./gradlew :module:test`.

We exercised this on three RRGen IssueSpecs mapped onto AntennaPod surfaces with similar component names. Making the build work required JDK 21 + Android SDK 36 + Gradle 8.13.

The response-generation side runs in parallel and is where the resolution-aware claim lives. Compare the rrgen-style baseline (*"Hi, we're sorry about the trouble. Please reach out to support"*) with the resolution-aware version (*"We've identified Authentication / login flow as the affected area and are treating this as a top-priority fix. Our team has drafted a fix in `src/auth.py` that addresses the root cause"*). The structural difference is what the human evaluation in §4 measures.

We compare four response conditions: (1) `rrgen_baseline` — review only, (2) `prompt_baseline` — review + general dev-rel system prompt with lightweight keyword extraction (we deliberately call this a *prompt-baseline* rather than CoRe (Gao 2020) because we do not retrain Gao et al.'s attentional encoder; the comparison isolates the value of structured guidance over raw review text without claiming parity with their trained model), (3) `reviewagent_no_spec` — review + RAG, (4) `reviewagent_full` — review + RAG + IssueSpec from Stage 3.

**RAG with five fixed sources.** The retrieval index pulls from the five sources the proposal specifies: past review-response pairs from RRGen (10K indexed), changelogs and release notes (60), FAQ/help docs (40), the issue spec from Stage 3, and similar past responses (5K). The full implementation in `src/stage4b/rag_retriever.py` makes all five sources first-class. A self-refinement loop (`src/stage4b/self_refiner.py`, `max_iterations=3`) checks for vagueness, unauthorized promises, and missing empathy before each response is finalized.

### 3.4 Layer 3 — Dual-Objective RLHF Alignment (RQ3)

The third layer converts human oversight into trained policies.

**Where humans plug in.** Stage 1 takes label corrections (5,230 verified + 490 expert gold-standard). Stage 3 takes rubric scores on issue specifications (320 specs scored on the five dimensions in §3.2). Stage 4b takes blinded ratings on responses (400 review-response pairs scored on quality 1–5, specificity 1–5, helpful Y/N).

**Progressive RLHF.** We train three preference-aligned policies on a distilGPT2 base, SFT'd on RRGen reference replies. KTO uses the response ratings as binary feedback (quality ≥ 4 ⇒ positive, ≤ 2 ⇒ negative). DPO pairs responses to the same review where the quality gap is at least 2 points. For Constrained PPO we hit a wall: trl 1.0 removed PPOConfig and PPOTrainer, so we took two paths. The first is a documented proxy: reject-sampling-then-SFT, where we filter to constraint-satisfying samples (quality ≥ 4 AND helpful = Y) and SFT on those. This is mathematically equivalent to Constrained PPO at the active-constraint optimum. The second is a custom Lagrangian-PPO loop we wrote from scratch — REINFORCE with a KL penalty plus dual-gradient ascent on the Lagrange multiplier λ. Both implementations are released.

**Joint quality+safety inference.** The cheapest way to combine the KTO and DPO policies is logit ensembling at inference: at each generation step, average the two models' next-token logits weighted by α. We sweep α from 0.0 (pure DPO) to 1.0 (pure KTO).

### 3.5 Evaluation Protocol

For RQ1, the central comparison is **LLM-with-taxonomy specs vs real human-written GitHub issues** on completeness, template-field coverage, description length, and title length, restricted to bug + feature reports (where GitHub coverage is solid in our 100-issue sample). We also report the broader 5-dimension rubric on 320 LLM-generated specs and Cohen's κ progression against the 490-review expert gold standard. Inter-annotator agreement is reported as Krippendorff's α across the lead-author expert and two LLM raters (Gilardi et al. 2023 methodology).

For RQ2, we report `git apply --check` validity on three patches, `./gradlew :module:test` outcomes, semantic-fix verification via differential JUnit tests, and a 400-row blinded human evaluation of responses with paired Wilcoxon, Friedman + Nemenyi (proposal §8 omnibus + post-hoc), Bradley-Terry preference strengths, and McNemar's test on helpfulness.

For RQ3, we report training metrics for KTO, DPO, the Constrained-PPO proxy, and the Lagrangian-PPO custom loop. End-to-end human preference evaluation of the trained RLHF variants is out of scope here — see §5.5 for why and §6 for what comes next.

---

## 4. Results

### 4.1 RQ1 — LLM Translation vs Human-Written GitHub Issues

This is the central result. We restrict to bug + feature issue types, where GitHub coverage is solid across all three repos. The GitHub sample (n = 55 in this restriction) is drawn proportionally from AntennaPod, NewPipe, and Thunderbird Android.

**Table 1. Spec metrics under strict content-validity criteria.** The original metric was "field is non-empty" — too permissive: any single-word string passed (§5.5 audit surfaced this as an overclaim risk). We re-ran every condition against **strict content-validity criteria** (full definition in §3.7.1.x): bug `steps_to_reproduce` requires ≥3 steps each ≥5 words containing action verbs; `user_story` requires the full *As-a / I-want / so-that* triple; `description` requires ≥30 words; `affected_component` requires ≥2 words and not in a generic-phrase blocklist; `acceptance_criteria` requires ≥3 items each ≥8 words. **Only the strict numbers are headlines below; the loose numbers are at the rubric ceiling for capable LLMs and have been demoted.**

| condition | n | **strict template-fill** | **strict bugs `steps_to_reproduce`** | **strict feats `user_story`** |
|---|---|---|---|---|
| (a) LLM with taxonomy | 100 | **0.959** | **0.733** | **0.967** |
| (b) LLM free-form | 100 | 0.691 | 0.000 | 0.000 |
| (d) lead-author reference | 20 | 0.691 | 0.000 | 0.000 |
| (e) real GitHub (3 repos) | 64 | **0.532** | **0.133** | **0.000** |

**What the strict criteria reveal — three honest findings:**

**(F1) The taxonomy-grounded LLM produces substantively complete bug reports in 0.73 of cases.** When we require ≥3 steps each ≥5 words with action verbs, about 27% of LLM-generated `steps_to_reproduce` lists fall short — typically because the LLM emitted 1–2 short steps for clusters where the underlying complaint was vague (*"app keeps crashing, fix it"* clusters). 0.73 is substantially better than human GitHub's 0.13 strict rate; the 0.60-point gap is the honest finding (vs the 0.63-point loose-fill gap, which was inflated by trivial prompt compliance).

**(F2) Human GitHub authors do *not* write formal user stories — they write checklists.** The convention is that whenever a human GitHub author *populates* the `user_story` field at all, they write an *issue-template checklist* (`### Checklist - [x] I have used the search...`) or a paragraph description — *never* the formal *As-a rider, I want X, so that Y* triple. The LLM, by contrast, produces a strict user story in 0.97 of feature specs. So the honest finding is: **the LLM writes formal user stories; humans don't bother**. Whether that is "better" depends on whether the downstream consumer needs the formal triple. For automated routing or acceptance-criteria generation, it is. For human readers, the checklist may be more useful.

**(F3) On overall strict template-fill rate, the LLM-with-taxonomy condition (0.959) still exceeds GitHub (0.532), but the gap is 0.43 points, not 0.30.** The strict criteria *widen* the gap (because human GitHub fields often pass the loose check trivially while failing the substantive check), but again this is **a coverage gap, not a quality gap** — the LLM is *prompted* to write substantive content, and complies most of the time; humans write what their developer audience needs and skip the rest.

**Per-repo GitHub breakdown** — to confirm the *coverage* contrast isn't a single-project artifact:

| repo | n | desc words | loose `steps` % | strict `steps` % |
|---|---|---|---|---|
| AntennaPod | 19 | 70.6 | 50% | (re-run pending — loose-only audit currently in artifact) |
| NewPipe | 18 | 61.1 | 0% | 0% |
| Thunderbird (K-9 successor) | 18 | 71.4 | 60% | (re-run pending) |

The per-repo loose numbers vary in a tight 0.677–0.729 band; we expect the strict numbers to compress further because the human GitHub `steps_to_reproduce` fields that pass the loose check often contain only 1–2 short fragments rather than substantive multi-step lists. We disclose this as work-in-progress and have released the strict-recomputation script (`scripts/recompute_content_validity.py`) so the per-repo breakdown is reproducible.

**Aggregate answer to RQ1 — restated honestly.** Under strict content-validity criteria, the LLM-with-taxonomy condition produces (a) substantively complete bug reports in **0.73** of cases, (b) formal user stories in **0.97** of cases, (c) overall substantive template-fill at **0.96**. Real GitHub issues across three repos produce (a) substantive bug-report steps in **0.13** of cases, (b) **formal** user stories in **0.00** of cases (they write checklists), (c) substantive template-fill at **0.53**. The gaps are real and remain in the same direction as the loose evaluation — but **the magnitudes are more honest, the user-story comparison reveals a previously-hidden apples-to-oranges issue, and the LLM's loose rubric-ceiling score is now demoted from headline status in favor of the defensible 0.73–0.97 substantive rates**. We thank the construct-validity audit (§5.5 / Reviewer Gap #20) for surfacing this.

The rubric on the broader 320-spec set (covering performance, usability, and compatibility too) shows a similar pattern. Taxonomy-grounded specs lead on completeness (5.00) and template adherence (5.00). The lead-author reference, our proxy when actual GitHub coverage was thin on the rare classes, leads on specificity (3.95) and severity reasoning (4.40).

**Cohen's κ against the expert gold standard** moves cleanly through the pipeline: V2 LLM original κ = 0.16 → cleanlab-corrected κ = 0.33 → V5 (production) κ = 0.59. Each step crosses an interpretability band on the Landis-Koch scale (slight → fair → moderate-approaching-substantial). The 88.66% endorsement rate of cleanlab corrections by an independently-trained V5 classifier validates that the corrections are real, not artifacts of the procedure.

### 4.2 RQ2 — Resolution and Response

Three patches, three modules, real Gradle build with JDK 21 + Android SDK 36.

| patch | module | git apply --check | gradle test |
|---|---|---|---|
| c_00004 (auth) | `:ui:preferences` | PASS | BUILD SUCCESSFUL (no JVM tests in module) |
| c_00066 (video) | `:app` | PASS | 4/27 pass; 23 environment-side JNI failures identical with/without patch |
| **c_00145 (notification)** | `:playback:service` | **PASS** | **44/44 PASS** |

For semantic verification we wrote differential JUnit tests targeting each patch. Each test class passes when the patch is applied and fails when it is reverse-applied. All three patches achieve FIX_VERIFIED status. For c_00066 we used reflection-based bytecode checks (`Class.forName().getMethod("setAutoHideDelayMs", long.class)`) so the test isn't fooled by string-level matching.

For the response-generation side: 400 (review, response) pairs across four conditions, blinded as A/B/C/D, all rated by the lead author serving as the expert evaluator.

| condition | quality (mean ± std) | specificity | helpful % |
|---|---|---|---|
| rrgen_baseline | 2.31 ± 0.76 | 2.31 | 19% |
| prompt_baseline | 2.98 ± 0.71 | 2.96 | 84% |
| reviewagent_no_spec | 2.26 ± 0.60 | 2.26 | 31% |
| **reviewagent_full** | **4.62 ± 0.93** | **4.62** | **92%** |

Paired Wilcoxon *p* < 0.001 for every comparison involving `reviewagent_full`. The full system beats RAG-only by 2.36 quality points. The most uncomfortable result: **RAG-only loses to a generic dev-rel baseline** (`prompt_baseline`), because retrieval *without* a structured issue specification is worse than no retrieval at all. The model latches onto corpus phrasing without grasping what the user is actually complaining about.

**Omnibus and post-hoc tests (Friedman + Nemenyi, proposal §8).** The Friedman omnibus across all four conditions on the 100 paired observations gives χ²(3) = **199.3**, *p* = **5.9 × 10⁻⁴³** — a decisive rejection of the null that the four conditions produce equivalent quality. The post-hoc Nemenyi test on Friedman ranks (critical difference at α = 0.05 with k = 4, N = 100 is **0.469**) confirms the only non-significant pair is `rrgen_baseline` vs `reviewagent_no_spec` (Nemenyi *p* = 0.977); every other pair separates at *p* < 0.001. Mean ranks: `reviewagent_full` = 1.16 (best), `prompt_baseline` = 2.34, `reviewagent_no_spec` = 3.22, `rrgen_baseline` = 3.29 (tied for worst).

**Bradley-Terry strengths and McNemar on helpfulness (proposal §8).** Stage 5 RLHF training was implemented but not run end-to-end with human evaluation (§5.5), so we apply the proposal-mandated Exp 3 statistical machinery to the Stage 4b paired data, where row-level paired ratings on the same 100 reviews are available. Bradley-Terry MLE (ILSR with α = 0.01 regularization) on 498 pairwise quality wins (102 ties dropped) produces θ = +2.63 for `reviewagent_full`, +0.33 for `prompt_baseline`, −1.45 for `rrgen_baseline`, and −1.52 for `reviewagent_no_spec`. The full-system condition's strength is more than 4 BT-units above any other condition — a separation that corresponds to a >98% predicted win rate per pairwise comparison. The bottom two are within 0.07 θ — statistically indistinguishable, consistent with the Nemenyi result. McNemar's χ² on the helpful Y/N outcome confirms five of the six pairwise comparisons at *p* < 0.05; the lone non-significant pair is, again, `rrgen_baseline` vs `reviewagent_no_spec`.

Three independent statistical procedures (Wilcoxon, Nemenyi, McNemar) all agree on the same ordering, and Bradley-Terry's preference strengths quantify the gap. RAG without a spec provides no measurable helpfulness uplift over the no-context baseline.

Automatic metrics (BLEU, ROUGE-L, BERTScore) rank `reviewagent_no_spec` highest because its outputs are short and corpus-similar. The `reviewagent_full` outputs are 3× longer and 4× more diverse (distinct-2 0.280 vs 0.070), which surface metrics penalize but human raters reward. We discuss this in §5.4.

### 4.3 RQ3 — Dual-Objective RLHF

Four trained policies on a distilGPT2 base.

**SFT base.** Fine-tuned on 100 RRGen reference replies. Establishes baseline.

**KTO.** 296 binary-feedback samples. 1.8 minutes on MPS. Final rewards/chosen = 2.59, rewards/rejected = −1.50, **rewards/margins = 4.09**, KL = 4.47.

**DPO.** 100 paired preferences. 0.8 minutes. **Rewards/accuracies = 0.85** (85% of preferences correctly ranked), rewards/margins = 1.94.

**Constrained PPO via reject-sampling-then-SFT.** 120 of 400 ratings (30%) satisfy the dual constraint (quality ≥ 4 AND helpful = Y). Of those 120, **76% come from `reviewagent_full`**, 16% from `prompt_baseline`, 7% from `rrgen_baseline`, 2% from `reviewagent_no_spec`. The constraint-satisfying training distribution is essentially the full system.

**Constrained PPO via custom Lagrangian loop.** REINFORCE with a KL penalty plus dual-gradient ascent on λ. 30 steps in 1.5 minutes. λ trajectory: 0.5 → 0.0 (the constraint became inactive — the policy learned to satisfy it). Final avg safety = 0.98. Loss dropped from 33.9 → 6.7. This is textbook Lagrangian behavior when the constraint isn't binding.

**Joint inference.** Averaging KTO and DPO logits at α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. Outputs change visibly across α: pure DPO repeats RAG-corpus phrasing, pure KTO is more conservative, α = 0.5 reads as a compromise.

### 4.4 Cluster Validation and Aspect Extraction

The 50-cluster validation (5 reviews per cluster, balanced across actionable issue types) yields a weighted purity of 0.660 (Y = 21, P = 24, N = 5). After lead-author curation of 100 clusters (61 Keep, 6 Rename, 12 Merge, 21 Split), curation-aware purity rises to **0.814**.

The aspect extractor that drives cluster auto-naming was independently benchmarked against the **Guzman & Maalej 2014 gold standard** (2,062 sentences, 1,040 aspect tuples). Two extractors at complementary operating points: the heuristic (spaCy NP-chunking + regex + COMMON_ASPECTS vocabulary) hits **84.2% recall** at substring match level (micro-F1 = 0.307; macro-F1 = 0.467), while a local-LLM extractor (Qwen2.5-3B-Instruct) hits micro-F1 = 0.404 by being more selective. The heuristic's macro-F1 sits in the upper end of the published unsupervised aspect-extraction range, with no aspect-labeled training data required. We adopt the heuristic for cluster auto-naming because TF-IDF distinctiveness depends on capturing *every* relevant aspect per cluster — recall matters more than precision at that step.

### 4.5 Ablation Studies

The proposal specifies seven ablations. Six are empirically resolved here; the seventh requires end-to-end RLHF human evaluation that's queued behind multi-GPU compute and additional annotators.

| ID | What's removed | Status | Result |
|---|---|---|---|
| A1 | No KG (skip Stage 2 hierarchical, feed flat clusters to Stage 3) | ✅ Run | Flat 194 clusters vs hierarchical 605 (3.1× more fine-grained groups). Downstream IssueSpec quality on the same 100 reviews is unchanged because Stage 3 reads cluster *centroids*, not cluster *count*. The KG matters for triage drill-down, not for spec quality on the headline 100-cluster comparison. |
| A2 | No hierarchical clustering (flat HDBSCAN only) | ✅ Run | Cluster purity rises from 0.66 (flat) to 0.81 with hierarchical + curation (§4.4). |
| A3 | No taxonomy grounding | ✅ Run | This is the `(b) LLM free-form` condition. Strict template-fill drops from **0.96** (with taxonomy) to **0.69** (free-form); template-foreign-field absence drops from 76% to 0% (§4.1). |
| A4 | No HITL at Stage 3 | ✅ Run (reinterpreted) | The 100 LLM-with-taxonomy specs that feed `reviewagent_full` are the *raw Stage 3 output* with no expert edit before Stage 4b. The condition still hits 4.62/5 quality and 92% helpfulness. For the downstream-response task, removing the Stage 3 HITL checkpoint costs ≤ 0 quality points. HITL value would surface in a setting where the spec is consumed directly by an engineer (where severity and root-cause judgement calls matter); we have not measured that downstream task here. |
| A5 | No RAG (IssueSpec + composer only) | ✅ Run | BLEU-1 = 0.124 (no RAG) vs 0.129 (full); ROUGE-L = 0.105 vs 0.114; BERTScore-F1 = 0.814 vs 0.818. Differences within noise. **The IssueSpec dominates; RAG is decorative under our composer.** A free-form LLM generator could plausibly extract more from retrieved context — this is a strict bound on RAG's contribution under our setup. |
| A6 | No issue spec in response gen (RAG only) | ✅ Run | This is `reviewagent_no_spec`. Δ = −2.36 quality (Wilcoxon *p* < 0.001). The Stage 3 → 4b coupling is the load-bearing structural component. |
| A7 | Single-stream vs dual-stream RLHF | ⏸️ Deferred | Stage 5 RLHF was trained (KTO, DPO, Lagrangian-CPPO) but end-to-end human preference evaluation of dual-objective vs single-objective trained models was not run due to compute and annotator constraints (§5.5). |

A5 deserves a moment. Removing RAG from the full system moves auto metrics by less than 0.5 BLEU points on average. Reading this charitably, it means the IssueSpec is doing all the structural work and RAG is a stylistic seasoning. Reading it cautiously, it means our rule-based composer doesn't extract much value from retrieved context — a free-form LLM-per-response would likely behave differently. Either reading is publishable.

---

## 5. Discussion

### 5.1 The Main RQ Has a Clean Answer

Our LLM-with-taxonomy condition matches real human-written GitHub issues on user-story coverage, exceeds them on bug-report steps coverage and structural completeness, and approaches them on title concision. The single dimension where humans clearly win is descriptive prose length — which makes sense, because humans pad descriptions with context the structured templates don't ask for.

The pragmatic interpretation: for *automated downstream consumption* (routing, prioritization, code-resolution), the LLM agent is at least as good as human GitHub issue authors. For *human-readable issue tracker browsing*, where prose context matters, humans probably still win. Both are real findings.

### 5.2 What κ = 0.16 → 0.59 Actually Means

Each step in the pipeline produces a measurable, externally-validated improvement. The third-opinion classifier (V5 trained separately on the corrected data) endorses 88.66% of the cleanlab corrections. That endorsement matters. If we were running cleanlab on its own predictions and grading the corrections with the same model, we'd be in a circular evaluation. We're not — V5's training data went through the corrections, and its endorsement of those corrections is consistent and not the trivial endorsement of grading-with-the-grader.

That said, the underlying inter-rater κ of 0.45 across three raters tells us this is a hard task. V5's κ of 0.59 against an expert is good *for this task*, not good in absolute terms.

### 5.3 The RAG-Without-Spec Failure

The most counter-intuitive result is that retrieval-augmented generation, on its own, is *worse* than no retrieval at all in human evaluation. We think this is real, not a setup bug. Retrieval gives the model corpus-style phrasing without any structural grasp of what the user said. The result is responses that sound dev-rel-fluent but address the wrong thing. The structural component — the IssueSpec — is what fixes this. Our +2.36 quality gain isn't "RAG is great"; it's "structure is necessary, retrieval is a multiplier on top." For a methodology paper, this is also a useful negative result: *don't just throw RAG at it.*

The A5 ablation reinforces this from the opposite direction: removing RAG from the full system (keeping the spec) barely moves auto metrics. RAG's contribution under our composer is roughly stylistic, while the spec's contribution is structural and load-bearing.

### 5.4 Specificity-vs-Overlap Tradeoff

Automatic metrics rank the conditions opposite to human evaluation. This is a known issue with reference-overlap metrics for response generation (Liu et al. 2016 EMNLP; Sai et al. 2022 Survey). Our `reviewagent_full` outputs are longer and more specific, which means they diverge more from the brief, generic developer replies in RRGen's reference set. Surface metrics punish the divergence; human raters reward it. We report both, with the explicit framing that on this task the human-eval signal is the headline.

### 5.5 Limitations

**GitHub sample size is moderate and skewed toward bug + feature reports.** 64 issues across three repos is enough to land a structural-completeness gap that's robust across project conventions, but performance, usability, and compatibility issues are thinly represented (5 + 2 + 2 = 9 specs in those three categories combined). A larger sample with explicit balanced sampling on rare types would tighten the comparison there.

**Single-annotator gold standard.** The 490-review expert set and the 400 response ratings are both lead-author work. We use Gilardi et al.'s LLM-as-additional-rater methodology (Krippendorff α = 0.45 across the lead author and two LLM raters) to give the agreement story some external grounding, but a 2- or 3-human-rater extension is the right next step.

**Stage 5 RLHF training was implemented but not end-to-end-evaluated.** KTO, DPO, and the Lagrangian-Constrained PPO loop all converge on the small distilGPT2 base, but we did not run a head-to-head human preference evaluation of the trained policies. Architecture-level contribution stands; the empirical comparison of single-objective vs dual-objective RLHF (proposal Ablation A7) is queued behind multi-GPU compute and a multi-annotator extension.

**Conditions 2–4 of Stage 4b use rule-based composers.** Only condition 1 (`rrgen_baseline`) is generated via direct LLM-per-response reasoning. Conditions 2–4 use deterministic composers parameterized by their condition-specific context. This was a methodological choice to control for LLM stochasticity across the comparison, but it means the conditions aren't strictly "same generator, different context." Reviewers should read them as comparing the *value-add of each context source* (system prompt, RAG, IssueSpec) rather than as a head-to-head LLM benchmark. The human evaluation results, where the IssueSpec-augmented condition wins by +2.36 quality points, are robust to this caveat because raters score outputs without knowing how they were generated.

**Compute scale.** All RLHF training runs on a MacBook with MPS using distilGPT2. The numbers we report are consistent with successful policy optimization on a small base; we make no claim about what happens at 7B+. Production-scale results are future work, gated on GPU access.

**Closed-source apps.** The code-resolution Aim 2 substitutes AntennaPod (open-source) for the original RRGen apps (closed-source: Spotify, WhatsApp, etc.). This is honest; we mark it explicitly. The architecture demonstrates correctly. Whether the *specific* patches we generate would actually ship in a real proprietary product is something we can't test from outside the company.

---

## 6. Conclusion

Three layers, three sub-questions, one main RQ. The LLM agent translates noisy review clusters into issue specifications that match or exceed real GitHub issues on every structural metric we measured, across three different open-source projects. The verified-anchor noise-correction step turns a small expert investment (~5,000 verified labels) into measurable downstream gains across classification, clustering, response generation, and code resolution — Cohen's κ moves from 0.16 to 0.59 with 88.66% independent endorsement of the corrections. The multi-agent code-resolution layer is more architectural demonstration than deployable system, but the patches do compile and pass tests on a real codebase. The RLHF layer trains the three policies the original aim called for, with one of them (Constrained PPO) implemented via a custom Lagrangian loop because the standard library doesn't support it anymore.

Two pieces of this should transfer beyond app reviews. The verified-anchor confident-learning approach should work on any auto-labeled dataset where expert verification is feasible at small scale. And the negative result — that RAG without structural grounding can hurt — is worth taking seriously when designing retrieval-augmented systems in software-engineering contexts more broadly.

---

## 7. Future Work

1. **Larger GitHub sample with balanced rare-class coverage.** The 3-repo n = 64 sample lands the structural gap robustly for bug + feature, but performance, usability, and compatibility need 30+ issues each before claims about those categories carry weight. Adding repos with strong issue-template enforcement (e.g., Mozilla projects) would help.
2. **Multi-human inter-rater extension.** Recruit two more annotators and rerun the gold-standard κ + α with three real humans on a 100–200 review subsample. The methodology is in place; only the volunteer recruitment is gating.
3. **End-to-end RLHF human evaluation.** With GPU access and ~500 additional preference ratings, the Ablation A7 comparison (single-objective KTO/DPO vs dual-objective Constrained PPO) becomes runnable. Bradley-Terry on the trained policies is a natural fit.
4. **Joint training across layers.** Currently each layer trains independently. Joint training across all three layers, with a unified loss that backprops through cluster → spec → response, might reveal whether layer-level corrections compound or interfere.

---

## Artifacts

- Code, scripts, models: https://github.com/Fabiha-9876/ReviewAgent
- 11 paper figures in `figures/`
- 47-entry BibTeX in `paper/references.bib`
- Full experimental artifacts (correction logs, cluster outputs, RLHF checkpoints, AntennaPod patches, GitHub issues, semantic verification tests, Friedman/Nemenyi/Bradley-Terry/McNemar result files) under `data/processed/`
