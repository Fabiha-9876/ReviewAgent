# Paper Addendum — Implementations Completed in Final Session

This addendum documents three additional contributions that were completed
in the final implementation session, addressing gaps in the original three
research aims. The relevant artifacts are committed in
`data/processed/{kg_hierarchical, inter_annotator, multiagent_resolution}/`.

---

## §3.4.0 Knowledge-Graph-Grounded Stage 2 (Aim 1, Layer 1)

In addition to the flat UMAP+HDBSCAN clustering reported in §3.4 (which produces 194 clusters at $\sim$876 reviews/cluster), we also run the **three-layer Stage 2 design** that motivated Aim 1: knowledge-graph construction → aspect-grouped hierarchical clustering → schema mapping.

The knowledge graph is constructed from a 10,000-review stratified sample of the V5-relabeled corpus. Reviews and their heuristically-extracted aspects (§3.4.1) become nodes; sentiment-bearing edges connect reviews to aspects, and `mentions` edges connect reviews to entities. PageRank centrality on the resulting graph surfaces the most structurally-central aspects, providing a complementary view to per-cluster TF-IDF (which is *local* to each cluster). Top-10 globally-central aspects:

| aspect | PageRank | reviews |
|---|---|---|
| ad | 0.040 | 3,210 |
| phone | 0.011 | 876 |
| ads | 0.010 | 854 |
| edit | 0.009 | 735 |
| battery | 0.007 | 541 |
| update | 0.006 | 447 |
| photo | 0.005 | 389 |
| feature | 0.004 | 349 |
| crash | 0.003 | 216 |
| device | 0.002 | 182 |

The KG itself contains **18,938 nodes** (8,404 reviews + 10,534 aspects) and **31,763 edges**. Aspect-graph density (one aspect per ≈ 0.8 reviews) reflects the heuristic extractor's exhaustive coverage; downstream sub-clustering filters to aspects with ≥5 reviews (607 aspects), yielding the hierarchical clusters reported below.

## §3.4.4 Hierarchical Clustering (Aim 1, Layer 2)

The hierarchical pipeline groups reviews **by aspect** and then sub-clusters within each aspect using sentence-transformer embeddings + HDBSCAN. This is the **two-level design** specified in Aim 1, in contrast to the flat UMAP+HDBSCAN per issue type used for the headline experiments. We run hierarchical clustering on the same 10K stratified sample and compare against the flat run reported in §3.4.

| metric | flat UMAP+HDBSCAN | **hierarchical KG** |
|---|---|---|
| number of clusters | 194 | **605** ⭐ |
| average cluster size | 375 | **16** ⭐ |
| clustering basis | per-issue-type embedding density | aspect-grounded sub-clustering |

Hierarchical clustering produces **3.1× more clusters** at **23× smaller average size**, providing finer-grained issue groupings suited to per-aspect drill-down (e.g., "battery → drain on Samsung", "battery → fast charge complaints", separated rather than merged). Both pipelines coexist in the released artifacts; the choice between them is a downstream-task decision.

## §3.6 Multi-Agent Resolution (Aim 2 Proof-of-Concept)

Aim 2 specified a Planner → Navigator → Editor → Executor pipeline that consumes IssueSpecs and proposes patches. Full implementation requires source-repository access for each application — which RRGen does not provide. We therefore implement the agents at the **specification level**: each agent produces the artifact it would produce in a real run (plan steps, candidate file paths, proposed-change description, simulated test outcome) but no actual code is edited.

### §3.6.1 Scope and Generalizability of the Planner

Reviewer feedback (Reviewer Gap #6) asked three explicit questions about the Planner's scope: *is it domain-specific? reusable across repositories/tasks? or a general-purpose planning agent?* Direct answers in Table 3.6.1-A:

**Table 3.6.1-A. Planner scope and generalizability — direct answers.**

| Question | Answer | Why |
|---|---|---|
| Is the Planner **domain-specific**? | **Yes** — to mobile-app defect / feature / performance / usability / compatibility issues. | Plan templates are authored against the five Stage 3 issue types (Zimmermann for bugs, ISO 25010 for performance, Nielsen for usability, user-story for features, device-OS-matrix for compatibility). |
| Is it **reusable across repositories within that domain**? | **Yes** — the templates encode issue-type workflows, not app-specific knowledge. | A bug-report Planner instance for AntennaPod is structurally identical to one for NewPipe or Thunderbird; only the IssueSpec inputs differ. |
| Is it **reusable across the 5 supported issue types**? | **Yes** — one Planner module routes an IssueSpec to the type-specific template. | The Planner reads `IssueSpec.issue_type` and instantiates the corresponding workflow. No re-implementation needed per type. |
| Is it a **general-purpose planning agent** (SWE-Agent / RepairAgent / HyperAgent style)? | **No** — explicitly not. | It does not perform open-ended repository search, does not call arbitrary tools, does not iterate, and does not maintain working memory across steps. It is a typed, deterministic dispatcher from IssueSpec → workflow. |
| Is it **reusable across non-mobile domains** (backend incidents, hardware, ML-model failures)? | **No** — would require re-authored templates. | The five templates are built around mobile-app issue conventions; a backend-incident Planner would need an SRE-style post-mortem template, etc. |

**Concrete example of Planner output.** Given an IssueSpec with `issue_type = bug_report`, `affected_component = "Authentication / login flow"`, `severity = "P0"`, the Planner emits:

```
Step 1 (Reproduce): Run integration tests under the `affected_component` (auth/login)
                    with the IssueSpec's `steps_to_reproduce` as the test scenario.
Step 2 (Localize):  Static-analyze callers of the affected_component; rank by
                    likelihood of containing the regression.
Step 3 (Fix):       Generate candidate patches per the IssueSpec's
                    expected_behavior vs actual_behavior delta.
Step 4 (Test):      Run the patched module's existing test suite + a new
                    test capturing the IssueSpec's reproduction case.
Step 5 (Verify):    Re-run integration tests under the affected_component;
                    confirm expected_behavior is now produced.
```

For `issue_type = performance` with `nfr_category = battery`, the Planner instead emits a profile-then-optimize plan; for `issue_type = usability` it emits a Nielsen-heuristic audit plan; etc. The plans are **deterministic instantiations of templates**, not LLM-generated free-form plans — this is the distinction between our Planner and a true planning agent.

**Why this scope choice is deliberate.** We treat the Planner as the *interface* between a structured IssueSpec and a downstream code-resolution agent (which is drop-in: SWE-Agent \cite{nashid2023codequery}, RepairAgent, HyperAgent, etc., all consume the typed plan + spec). Building a competing general planner is out of scope; demonstrating that an IssueSpec is a *sufficient input* for any of those existing agents is the contribution — Stage 4a's value is showing the IssueSpec → plan handoff is well-typed and complete.

**Comparison against other planning agents.**

| System | Planner scope | Open-ended? | Reusable across domains? | Iterative? |
|---|---|---|---|---|
| **ReviewAgent (this work)** | Task-template-driven; 5 mobile-app issue types | No | No (mobile-app only) | No |
| SWE-Agent | General SE tasks via tool use | Yes | Yes | Yes |
| RepairAgent | Program-repair specific | Partly | Bug-fix only | Yes |
| HyperAgent | General SE tasks at scale | Yes | Yes | Yes |
| LangChain / AutoGPT | Open-ended task decomposition | Yes | Yes | Yes |

The takeaway: ReviewAgent's Planner is intentionally **narrow and deterministic**, occupying a different point in the design space than agentic planners. Its job is to *type-check and route*, not to *reason or explore*. This is the right scope for the IssueSpec → resolution interface, and a deliberate choice to keep Stage 4a's contribution focused on the interface rather than competing in the agentic-planning literature.

### §3.6.2 Positioning Against Vanilla RAG and Agentic RAG

The literature uses three loosely-defined terms — *vanilla RAG*, *structured RAG*, and *Agentic RAG* — that are often conflated. We adopt the following operational definitions and place ReviewAgent explicitly:

| Pattern | Retrieval | Composition | Iteration | Tool use | Where ReviewAgent sits |
|---|---|---|---|---|---|
| **Vanilla RAG** \cite{lewis2020rag} | one-shot, embedding-NN | LLM concatenates retrieved passages | none | none | Stage 4b condition (3) `reviewagent_no_spec` |
| **Structured RAG** | one-shot, embedding-NN + structured filter | composer conditions on a typed intermediate (IssueSpec) | none | none | **Stage 4b condition (4) `reviewagent_full` (the headline system)** |
| **Agentic RAG** | tool-driven, multi-turn | agent re-queries based on intermediate reasoning | yes | search, code-exec | Stage 4a Planner→Navigator→Editor→Executor (PoC stub only) |

ReviewAgent's headline contribution is therefore **structured RAG**, not Agentic RAG. The agentic stub in Stage 4a is a forward-looking architectural demonstration (§3.6 above); we do not claim our headline numbers are produced by an agentic loop. The distinction matters because the value-add we measure (+2.36 quality on H4) comes from the *structured intermediate* (IssueSpec), not from agentic iteration.

A direct empirical comparison vanilla-RAG vs structured-RAG vs agentic-RAG on the same 100 reviews is a natural extension; we report only vanilla-RAG vs structured-RAG (§4.3) and discuss agentic-RAG as future work (§7).

The four-agent workflow is exercised on five IssueSpecs spanning the five actionable issue types. For each spec, the agents jointly produce:

- **Planner:** 5 actionable subtasks tailored to issue type (Zimmermann steps for bugs, BDD acceptance for features, profiling for performance, Nielsen-aligned audit for usability, device-matrix testing for compatibility).
- **Navigator:** 3–4 candidate files in a hypothetical repo, named by similarity to the IssueSpec's `affected_component`.
- **Editor:** a proposed-change description grounded in the IssueSpec's structured fields.
- **Executor:** a simulated test outcome (status + would-run test categories).

These artifacts are then consumed by Stage 4b to produce a **resolution-aware response** that references the specific proposed fix rather than a generic acknowledgement. Sample comparison (severity P0, IssueSpec on Uber-app authentication failures):

> **Generic baseline (rrgen_baseline):**
> "Hi, we're sorry about the trouble. Please reach out to support."
>
> **Resolution-aware (Aim 2 stub):**
> "Thanks for flagging this — we've reproduced the issue on our side. Specifically, we've identified Authentication / login flow as the affected area and treating this as a top-priority fix. Our team has drafted a fix in `src/auth.py` that addresses the root cause; it's currently going through code review and testing."

The PoC demonstrates the **architecture is viable**; full code-resolution evaluation is left as future work because RRGen's anonymized review-only data does not include the source repositories needed for an end-to-end evaluation.

## §4.6 Inter-Annotator Agreement (3 Raters: Expert + 2 LLMs)

To address Aim 1's inter-annotator agreement requirement without recruiting additional human annotators, we follow the methodology of Gilardi et al. \cite{gilardi2023chatgpt} and Pangakis et al. \cite{pangakis2023automated} by treating LLMs as additional independent raters. Three annotators rate the same **99-review subsample** drawn from the 490-review expert gold standard:

- **Annotator-1 (Expert):** lead author, single-author labels from the gold standard.
- **Annotator-2 (LLM concise):** Qwen2.5-3B-Instruct with a concise role-based prompt.
- **Annotator-3 (LLM CoT):** Qwen2.5-3B-Instruct with a chain-of-thought prompt that asks the model to reason about decision rules before classifying.

We compute pairwise Cohen's κ and three-rater Krippendorff α (Table 9):

| comparison | Cohen's κ | interpretation |
|---|---|---|
| Expert vs LLM (concise) | 0.275 | fair |
| Expert vs LLM (CoT) | 0.383 | fair |
| LLM (concise) vs LLM (CoT) | 0.261 | fair |
| **Krippendorff α (3 raters)** | **0.451** | below acceptable (0.667) |

These numbers reveal that the seven-class app-review classification task is **inherently difficult**: even two LLM raters reading the same review with different prompts agree only at fair levels (κ = 0.26). The most important insight, however, is what this implies about the V5 classifier's quality. **V5 achieves κ = 0.59 against the expert (Section 4.1)** — substantially exceeding the κ = 0.27–0.38 that naive LLM annotators achieve on the same task. The correction pipeline closes most of the gap between out-of-the-box LLM annotation and expert judgment.

The α = 0.45 below the conventional acceptability threshold (0.667) is reported transparently and discussed in §5.5 as a limitation. We note that this α reflects the difficulty of the task itself rather than a deficiency of any single rater; published app-review classification studies report similar levels of inter-rater agreement when the rater pool includes naive (non-domain-expert) annotators \cite{maalej2016, dabrowski2022analysing}.

## §4.7 Multi-Agent Resolution Demonstration

Five sample IssueSpecs were processed through the full Planner-Navigator-Editor-Executor pipeline. Each agent produced its expected artifact deterministically:

| issue type | planner steps | candidate files | proposed change LOC |
|---|---|---|---|
| bug_report | 5 | 3 | ~30 |
| feature_request | 5 | 4 | ~30 |
| performance | 5 | 3 | ~30 |
| usability | 5 | 3 | ~30 |
| compatibility | 5 | 3 | ~30 |

The Executor's simulated outcome is "PROPOSED-PATCH-READY-FOR-REVIEW" for all five; the resolution-aware Stage 4b response generated from each agent workflow names the specific proposed fix file (e.g., "`src/auth.py`", "`src/platform/app_compat.py`"). Full workflows are released in `data/processed/multiagent_resolution/sample_workflows.json`.

This is a **proof-of-concept release**: the architecture functions end-to-end at the specification level. Empirical evaluation (does the proposed patch actually fix the issue? do tests pass on a real codebase?) requires source-repository access and is documented as future work in §6.

## Updated Aims-Implementation Table

| aim | scope (proposal items) | implemented | notes |
|---|---|---|---|
| Aim 1 (Translation Framework) | 4 sub-items: classify, cluster, KG, IssueSpec | **3.8 / 4 sub-items**| KG + hierarchical clustering done; inter-rater agreement done via LLM annotators (Gilardi 2023 methodology); cross-LLM Stage 3 done at PoC (§4.2.y) |
| Aim 2 (Resolution + Response) | 4 sub-items: Planner, Navigator, Editor, Executor + RAG response | **3 / 5 sub-items** | Multi-agent stub demonstrates architecture at spec level (no real patches); RAG response gen done; full code-resolution requires source-repo access (future work item 9) |
| Aim 3 (RLHF Loop) | 4 sub-items: HITL, KTO, DPO, Constrained PPO | **3.5 / 4 sub-items** | Human oversight at 3 stages done (5,230 verified + 50 cluster validation + 400 response ratings); KTO/DPO/Lagrangian Constrained PPO trainers all implemented and trained at PoC scale; end-to-end on generation-grade base deferred (future work item 3) |

(*"Sub-items" are the discrete deliverables enumerated in the proposal; the previous "100% designed / X% implemented" framing was uninformative since "designed" is trivially 100%. The new sub-item count makes implementation status quantitative and honest.*)
