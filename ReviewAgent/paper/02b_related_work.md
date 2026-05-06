# 2. Related Work

## 2.1 App-Review Mining and Classification

The empirical software engineering community has produced a sustained line of work on mining and classifying user reviews from mobile-app stores. **Maalej and Nabil** \cite{maalej2016} established the canonical seven-class taxonomy (`bug_report`, `feature_request`, `user_experience`, `rating`, `information_giving`, `information_seeking`, plus catch-alls) and released a 4,400-review human-annotated corpus that remains one of the most cited starting points in the field. **Chen et al.'s AR-Miner** \cite{chen2014arminer} introduced filter-then-prioritize architectures, demonstrating that semi-automated review triage can surface actionable issues at scale. **Villarroel et al.'s CLAP** \cite{villarroel2016} extended this with explicit clustering for prioritization.

More recent work has shifted to neural and LLM-based approaches. **Di Sorbo et al.'s SURF** \cite{disorbo2016surf} combined intention classification with summarization. **Dąbrowski et al.** \cite{dabrowski2022analysing} provided the most comprehensive recent survey, finding that classification accuracy on app reviews has plateaued in the 0.75–0.85 macro-F1 range, with the bottleneck increasingly being **label quality, not model capacity**. Our work targets this bottleneck directly.

The **RRGen** dataset and corresponding response-generation work by **Gao et al.** \cite{gao2019rrgen} is our primary corpus and the closest baseline for Stage 4b. RRGen pairs 310K reviews with developer responses and proposes a sequence-to-sequence model for response generation. The original paper reported BLEU-1 around 0.22 against held-out responses. We use RRGen's data and developer-reply corpus as the foundation for our retrieval index and reference set.

## 2.2 Confident Learning and Label-Noise Correction

The methodology underlying our correction pipeline is **confident learning** as formalized by **Northcutt et al.** \cite{northcutt2021cleanlab}, implemented in the open-source `cleanlab` library. Confident learning estimates the joint distribution of given labels and (latent) true labels using out-of-sample model predictions, then flags examples whose given labels are unlikely under the estimated joint distribution.

Earlier work on label-noise mitigation includes **Patrini et al.'s loss correction** \cite{patrini2017making}, **Lee et al.'s self-paced cleansing** \cite{lee2018cleannet}, and **Han et al.'s co-teaching** \cite{han2018coteaching}. Confident learning differs in being a post-hoc data-cleaning step rather than a training-time loss modification, which makes it natural to combine with arbitrary downstream classifiers (in our case, a RoBERTa fine-tune). To the best of our knowledge, confident learning has not previously been applied to LLM-labeled software-engineering datasets at the scale we report (215,583 reviews).

## 2.3 LLM Annotation and Its Limitations

A growing body of work uses LLMs to produce training labels at scale. **Wang et al.** \cite{wang2021want} showed that GPT-3 can replace crowd-workers on certain text-classification tasks at lower cost. **Gilardi et al.** \cite{gilardi2023chatgpt} reported that ChatGPT outperforms crowdworkers on annotation accuracy in political-text classification. However, **Pangakis et al.** \cite{pangakis2023automated} and **Reiss** \cite{reiss2023testing} both find LLM annotators introduce systematic biases — particularly toward majority categories and against rare/minority classes — that mirror the failure modes we measure in the present work (LLM under-predicting `performance` and `compatibility`). **Laban et al.** \cite{laban2023llm} provide a survey-style treatment focused on LLM annotators' calibration problems.

Our contribution to this thread is **methodological rather than diagnostic**: we accept that LLM annotators err systematically and provide a concrete, reproducible pipeline for correcting their errors using a small expert-verified anchor.

## 2.4 Issue-Specification Templates and Taxonomies

Stage 3 of our pipeline uses domain-established templates rather than free-form generation. The templates are drawn from four sources:

- **Zimmermann et al.** \cite{zimmermann2010} formalized the bug-report template (steps-to-reproduce / expected / actual) that has since become standard in defect-tracking systems.
- **Cohn** \cite{cohn2004user} popularized the user-story format for feature specification (As-a / I-want / So-that), with extensions like acceptance criteria from **Wynne et al.** \cite{wynne2017cucumber}.
- **ISO/IEC 25010** \cite{iso25010} standardizes non-functional requirement categories (we use the speed/battery/memory/responsiveness/scalability subset relevant to mobile applications).
- **Nielsen** \cite{nielsen1994} enumerates ten usability heuristics (visibility, match-real-world, user-control, etc.) which serve as our usability classification scheme.

To our knowledge, no prior app-review pipeline grounds Stage-3-equivalent issue-specification generation in this combination of templates simultaneously.

## 2.5 Retrieval-Augmented Generation in SE Applications

RAG \cite{lewis2020rag} has become a standard pattern for grounding LLM outputs in domain-specific text. Within software engineering, **Robillard et al.** \cite{robillard2017demand} surveyed the use of retrieval for code documentation; more recent work \cite{nashid2023codequery, ahmed2024automatic} applies RAG to code generation and review tasks. Our use of RAG over a developer-response corpus is most similar to **Gao et al.'s** original RRGen approach, with the addition of a structured IssueSpec layer that conditions generation on the analyzed cluster context — an extension that, as we show, is independently necessary for paper-grade response quality (Section 5.3).

## 2.6 Inter-Annotator Reliability

Standard reliability measures in classification studies include **Krippendorff's α** \cite{krippendorff2004content} and **Fleiss' κ** \cite{fleiss1971}, with **Cohen's κ** \cite{cohen1960} for pairwise comparisons. We use Cohen's κ throughout to evaluate classifier-vs-expert agreement; the per-pair interpretation thresholds (>0.80 almost-perfect, 0.60–0.80 substantial, 0.40–0.60 moderate) follow **Landis and Koch** \cite{landis1977}. As discussed in Section 5.5, our gold-standard set is single-annotator, which precludes α/κ reporting on the gold itself. We instead use the gold-standard labels as the reference against which three independent classifiers (V2 LLM, cleanlab-corrected, V5) are compared, treating each classifier as an annotator — a design that yields a defensible κ progression even without multi-human verification.
