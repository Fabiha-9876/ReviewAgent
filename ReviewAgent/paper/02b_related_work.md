# 2. Related Work

We organize related work along the five gaps identified in §1.3, so that each subsection traces a single argument: **what existing work does → where it falls short → what unresolved challenge remains → how this paper addresses it.** §2.7 collects the comparison into a single positioning table.

## 2.1 App-Review Mining and Classification (Gap G2)

**What exists.** **Maalej and Nabil** \cite{maalej2016} established the canonical seven-class taxonomy and released a 4,400-review human-annotated corpus. **Chen et al.'s AR-Miner** \cite{chen2014arminer} introduced filter-then-prioritize architectures for review triage at scale. **Villarroel et al.'s CLAP** \cite{villarroel2016} added clustering for release planning. **Di Sorbo et al.'s SURF** \cite{disorbo2016surf} combined intention classification with summarization. **Dąbrowski et al.** \cite{dabrowski2022analysing} surveyed the field and observed that classification F1 has plateaued at 0.75–0.85, with **label quality, not model capacity, as the bottleneck.**

**Where it falls short.** All of the above stop at *labeling*, a per-review category, rather than producing a structured artifact a developer can act on. The implicit assumption is that classification + retrieval + summarization is enough; in practice, a developer triage engineer cannot route a "bug_report" label into a defect tracker without reproduction steps, expected/actual behavior, severity, and affected component. None of the cited works produces these fields.

**Unresolved challenge.** A representation that is to a noisy review cluster what a GitHub issue is to a hand-written bug report.

**Our contribution.** Stage 3 produces a typed `IssueSpec` whose fields are determined by issue-type-specific templates (§3.5). The IssueSpec is the missing actionable artifact.

## 2.2 Confident Learning and Label-Noise Correction (Gap G1, methodology)

**What exists.** **Northcutt et al.** \cite{northcutt2021cleanlab} formalized confident learning, which estimates the joint distribution of given vs latent true labels from out-of-sample model predictions and flags improbable assignments. Earlier work on label-noise mitigation includes **Patrini et al.'s loss correction** \cite{patrini2017making}, **Lee et al.'s self-paced cleansing** \cite{lee2018cleannet}, and **Han et al.'s co-teaching** \cite{han2018coteaching}.

**Where it falls short.** Confident learning has been validated extensively on *crowd-labeled* image and text benchmarks, but not on **LLM-labeled software-engineering corpora** at the scale where LLMs are now the default annotator. The systematic class-collapse failure mode of LLM annotation is qualitatively different from random crowd noise, it is correlated with the LLM's own confidence and training distribution.

**Unresolved challenge.** A correction pipeline that handles *LLM-flavored* noise (concentrated on minority classes, structurally biased) using a small, affordable expert anchor.

**Our contribution.** §3.3 introduces *verified-anchor confident learning*: a two-tier procedure where a small expert-verified set + a public human-annotated set jointly train a RoBERTa anchor whose probabilities, not the LLM's, drive the cleanlab corrections. We contribute the empirical finding (§4.1) that an independently-trained classifier (V5) endorses 88.66% of those corrections, the strongest available third-opinion check.

## 2.3 LLM Annotation and Its Limitations (Gap G1, characterization)

**What exists.** **Wang et al.** \cite{wang2021want} showed that GPT-3 can replace crowd-workers on certain text-classification tasks at lower cost. **Gilardi et al.** \cite{gilardi2023chatgpt} reported that ChatGPT outperforms crowdworkers on annotation accuracy in political-text classification. **Pangakis et al.** \cite{pangakis2023automated}, **Reiss** \cite{reiss2023testing}, and **Laban et al.** \cite{laban2023llm} document that LLM annotators introduce systematic biases, particularly toward majority categories.

**Where it falls short.** These prior findings are diagnostic, they characterize the *existence* of LLM annotation bias, but they do not provide an actionable correction recipe at scale. The reported scales are also typically below 10K examples; the failure modes at 100K+ scale on software-engineering data are uncharacterized.

**Unresolved challenge.** A measurement + correction pipeline at the corpus scales (215K+) where SE LLM-annotation is actually deployed.

**Our contribution.** Direct measurement on 5,230 LLM-labeled reviews (§4.1) quantifies the praise mislabeling rate at 25% and identifies *class collapse* and *boundary confusion* as the two dominant failure modes. The verified-anchor pipeline (§3.3) is the corresponding actionable correction.

## 2.4 Issue-Specification Templates and Taxonomies (Gap G2, grounding)

**What exists.** **Zimmermann et al.** \cite{zimmermann2010} formalized the bug-report template (steps-to-reproduce / expected / actual) that has since become standard in defect-tracking systems. **Cohn** \cite{cohn2004user} popularized the user-story format for feature specification. **Wynne et al.** \cite{wynne2017cucumber} extend with BDD acceptance criteria. **ISO/IEC 25010** \cite{iso25010} standardizes non-functional-requirement categories. **Nielsen** \cite{nielsen1994} enumerates ten usability heuristics.

**Where it falls short.** These templates are well established in human practice but, to our knowledge, no app-review pipeline grounds *automatic* issue-specification generation in this combination of templates simultaneously. Existing review-summarization pipelines produce free-form text; existing IssueSpec generators (e.g., for GitHub-bot use cases) typically use a single template (the Zimmermann bug template) without type-routing.

**Unresolved challenge.** A type-routed grounding scheme that selects the *right* template per cluster (bug → Zimmermann, performance → ISO 25010, usability → Nielsen, feature → user-story, compatibility → device-OS matrix).

**Our contribution.** §3.5 defines the type-routed template selection and §4.2 quantifies the resulting completeness gain (1.00 with taxonomy vs 0.69 free-form).

## 2.5 Retrieval-Augmented Generation in SE Applications (Gaps G3, G4)

**What exists.** RAG \cite{lewis2020rag} has become a standard pattern for grounding LLM outputs in domain-specific text. **Robillard et al.** \cite{robillard2017demand} survey retrieval for code documentation; **Nashid et al.** \cite{nashid2023codequery} and **Ahmed et al.** \cite{ahmed2024automatic} apply RAG to code generation and summarization. In review-response specifically, **Gao et al.'s RRGen** \cite{gao2019rrgen} learns a sequence-to-sequence model over (review, response) pairs; **Gao et al.'s CoRe** \cite{gao2020core} adds contextual encoding of past responses.

**Where it falls short.** All of these condition generation on retrieval and/or raw text. None conditions on a *structured intermediate representation* (typed issue spec, severity, component). Empirically (§4.3), RAG *without* an issue spec is statistically indistinguishable from no-context baselines on our evaluation, retrieval anchors style but does not supply structural understanding of *what* the user is complaining about.

**Unresolved challenge.** Distinguishing the value-add of *structure* (an IssueSpec) from the value-add of *retrieval* (RAG), and demonstrating that they are complementary rather than substitutable.

**Our contribution.** Stage 4b is a spec-aware retrieval-augmented composer; the four-condition ablation in §4.3 isolates the contribution of structure (Δ = +2.36 quality points) vs retrieval (≈ 0).

We also clarify in §3.6 the relationship between our system and the *Agentic RAG* literature: ReviewAgent is best described as **structured-RAG with light agentic orchestration**, there is a Planner-Navigator-Editor-Executor stub (Stage 4a, future work), but the headline pipeline is non-iterative retrieval + structured composition rather than the multi-turn tool-use loop that defines fully-agentic RAG.

## 2.6 RLHF Alignment for Generation (Gap G5)

**What exists.** **Rafailov et al.'s DPO** \cite{rafailov2023} simplifies RLHF by treating the LM as its own reward model on paired preferences. **Ethayarajh et al.'s KTO** \cite{ethayarajh2024} reduces feedback to binary good/bad signals using prospect-theoretic loss. **Dai et al.'s Safe RLHF** \cite{dai2023} formulates safety as a constraint and uses Lagrangian-corrected PPO; **Altman** \cite{altman1999} provides the underlying CMDP formalism.

**Where it falls short.** Single-objective methods (DPO, KTO) collapse quality and policy compliance into one signal. They have not been applied to app-review response generation specifically. Safe-RLHF has not been demonstrated end-to-end on a domain where the safety constraint is operationally meaningful (no unauthorized promises, no PII leakage in dev-rel responses).

**Unresolved challenge.** A dual-objective RLHF pipeline whose constraint is grounded in a domain-specific compliance rubric, with empirical validation that dual outperforms single on the quality–compliance frontier.

**Our contribution.** §3.0 motivates the CMDP formulation; `src/stage5/` implements KTO, DPO, and Lagrangian Constrained PPO. We honestly report (§5.5) that end-to-end empirical validation was performed only at proof-of-concept scale (distilGPT2, 400 SFT samples), full Llama-3-8B training is the highest-priority future work.

## 2.7 Comparative Positioning, Five Per-Category Tables

Reviewer feedback (Gap #9) asked for a structured comparison across each of the five system categories ReviewAgent composes. We report per-category positioning tables below; the consolidated cross-category view is in Table 2.7-F.

### 2.7-A. App-Review Mining and Classification

| System | Year | Classification scheme | Output | Limitation |
|,|,|,|,|,|
| Maalej & Nabil \cite{maalej2016} | 2016 | 7-class taxonomy | Per-review label | Stops at label; no actionable artifact |
| AR-Miner \cite{chen2014arminer} | 2014 | informative-vs-not | Filtered review stream | Filter-only; no per-review structure |
| SURF \cite{disorbo2016surf} | 2016 | intention | Summary text | Summary, not actionable; no triage |
| Dąbrowski survey \cite{dabrowski2022analysing} | 2022 | meta-analysis | "F1 plateau" diagnosis | Identifies *label quality* as the bottleneck, no remedy |
| **ReviewAgent (ours)** | 2026 | **7-class V5 with verified anchor + cleanlab correction** | **Corrected labels, ready for downstream KG** | First to apply confident-learning to LLM-flavored noise at 215K scale |

### 2.7-B. Clustering and Triage

| System | Year | Clustering basis | Granularity | KG-aware? | Limitation |
|,|,|,|,|,|,|
| CLAP \cite{villarroel2016} | 2016 | k-means on text features | Flat | No | Flat clusters mix sub-themes |
| SURF \cite{disorbo2016surf} | 2016 | LDA-style topic model | Flat | No | Topic-model coherence; no aspect drill-down |
| Keertipati et al. \cite{keertipati2016} | 2016 | Frequency + manual | Flat | No | Manual prioritization step |
| ReviewGraph \cite{xu2025reviewgraph} | 2025 | KG embeddings | Document-level | Yes (rating prediction) | Rating prediction, not triage |
| **ReviewAgent (ours)** | 2026 | **UMAP+HDBSCAN flat (194) + aspect-grounded KG hierarchical (605)** | **Two-level: aspect → sub-aspect** | **Yes (sentiment-weighted)** | First to combine flat + KG-guided hierarchical at the same triage stage |

### 2.7-C. IssueSpec Generation and Templates

| System | Year | Output format | Type-routed? | Templates used | Limitation |
|,|,|,|,|,|,|
| RRGen \cite{gao2019rrgen} | 2019 | Free-form response | No | None | No structured intermediate |
| Lead-author hand-write | n/a | Free-form spec | Implicit | Implicit | Slow, single-author |
| Real GitHub issues (mined) | n/a | Free-form / partly templated | No | Project-specific | 0% formal user-stories under strict criteria; 13% substantive `steps_to_reproduce` |
| **ReviewAgent (ours)** | 2026 | **Typed `IssueSpec`** | **Yes (5 issue types)** | **Zimmermann \cite{zimmermann2010} + ISO 25010 \cite{iso25010} + Nielsen \cite{nielsen1994} + user-story \cite{cohn2004user, wynne2017cucumber} + device-OS-matrix** | First to type-route across 4 industry-standard templates |

### 2.7-D. RAG Systems for Software Engineering

| System | Year | Retrieval | Structured intermediate? | Iteration | Use case | Limitation |
|,|,|,|,|,|,|,|
| Vanilla RAG \cite{lewis2020rag} | 2020 | one-shot embedding-NN | No | None | General KILT | Surface-form grounding only |
| RRGen \cite{gao2019rrgen} | 2019 | seq2seq encoder | No | None | App-review responses | No issue context |
| CoRe \cite{gao2020core} | 2020 | contextual encoder + RAG | No | None | App-review responses | Style-anchored, not structure-anchored |
| Self-RAG \cite{asai2024selfrag} | 2024 | one-shot + self-reflection tokens | Reflection only | Implicit | General KILT | Self-reflection ≠ typed issue spec |
| DSP \cite{khattab2022demonstrate} | 2022 | program-composed | Typed program | Yes | Multi-hop QA | Code, not response generation |
| **ReviewAgent (ours)** | 2026 | **embedding-NN + structured IssueSpec filter** | **Yes (typed IssueSpec)** | **None** | **App-review response generation** | First *structured-RAG* explicitly distinguished from vanilla and agentic RAG (§3.7.0) |

### 2.7-E. RLHF Alignment Systems

| System | Year | Objective | Constraint formulation | Domain | Limitation |
|,|,|,|,|,|,|
| Vanilla PPO with RLHF \cite{schulman2017ppo, christiano2017deep, ouyang2022instructgpt} | 2017+ | Single reward | None | General LM | Conflates quality + compliance |
| DPO \cite{rafailov2023} | 2023 | Single (paired preferences) | None | General LM | Single-scalar; no constraint |
| KTO \cite{ethayarajh2024} | 2024 | Single (binary feedback) | None | General LM | Binary; no constraint |
| Safe RLHF \cite{dai2023} | 2024 | Dual (CMDP) | Lagrangian | General LM safety | Not applied to app-review responses |
| **ReviewAgent (ours)** | 2026 | **Dual (CMDP)** | **Lagrangian** | **App-review responses (PoC)** | First app-review-domain CMDP RLHF; PoC scale only (§4.7) |

### 2.7-F. Cross-Category Summary, Where ReviewAgent Fits

| System | Classify | Cluster | IssueSpec | Response gen | RLHF | Composes? |
|,|,|,|,|,|,|,|
| AR-Miner \cite{chen2014arminer} | filter | filter |, |, |, | No |
| CLAP \cite{villarroel2016} | rating | flat k-means |, |, |, | No |
| SURF \cite{disorbo2016surf} | intention | LDA |, | summary |, | No |
| RRGen \cite{gao2019rrgen} |, |, |, | seq2seq |, | No |
| CoRe \cite{gao2020core} |, |, |, | contextual+RAG |, | No |
| Maalej \cite{maalej2016} | 7-class |, |, |, |, | No |
| ReviewGraph \cite{xu2025reviewgraph} | rating | KG |, |, |, | No |
| Safe-RLHF \cite{dai2023} |, |, |, | general | dual-CMDP | No (different domain) |
| **ReviewAgent (ours)** | **V5 corrected** | **flat + aspect-KG** | **type-routed** | **structured RAG** | **dual-CMDP (PoC)** | **Yes, five stages composed** |

The cross-category point is that **no prior system composes more than two of these stages**; ReviewAgent's contribution is the *composition*, with each stage producing an independently-evaluable typed artifact.

## 2.8 Inter-Annotator Reliability

Standard reliability measures in classification studies include **Krippendorff's α** \cite{krippendorff2004content}, **Fleiss' κ** \cite{fleiss1971}, and **Cohen's κ** \cite{cohen1960} for pairwise comparisons; per-pair interpretation thresholds (>0.80 almost-perfect, 0.60–0.80 substantial, 0.40–0.60 moderate) follow **Landis and Koch** \cite{landis1977}. We discuss the single-annotator limitation of our gold-standard set in §5.5; the κ-progression result in §4.1 nonetheless yields a defensible signal because each successive classifier (V2 LLM, cleanlab-corrected, V5) is independently trained, so the comparison resembles a between-annotator analysis where only one annotator (the lead author) is human and the other three are model-derived.
