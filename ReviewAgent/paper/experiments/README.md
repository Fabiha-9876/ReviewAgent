# CIKM 2026 Pre-Submission Experiments

These scripts close the highest-leverage gaps a CIKM reviewer would flag.
None of them have been executed yet. After running each, follow the
"Next steps" printed by the script (or in the rubric) to wire results
back into `paper/build/main.tex`.

## What each one fixes

| ID | Reviewer concern | Script | Action |
|----|------------------|--------|--------|
| 1.2 | Three of four Stage 4b baselines are rule-based, not LLM, so the 4.62 vs 2.26 gap conflates IssueSpec grounding with LLM-vs-template generator. | `exp_1_2_llm_no_spec.py` | Generate an LLM counterpart for `reviewagent_no_spec` (RAG, no IssueSpec) on the same 100 reviews; human-rate; add as a 5th column to Table 3. |
| 1.4 | The "LLM free-form" baseline in Table 1 uses a one-sentence prompt; reviewer will say the win comes from prompt engineering, not taxonomy grounding. | `exp_1_4_llm_cot_no_taxonomy.py` | Generate IssueSpecs with a competitive 2024-era CoT prompt (no taxonomy); score with the existing rubric; add as a column to Table 1. |
| 2.6 | 64 GitHub issues across 3 repos is moderate; a 4th repo in a different domain hardens the structural-completeness claim. | `exp_2_6_mine_4th_repo.py` | Mine a 4th repo (suggestions in script); add a row to Table 2; update intro/abstract to "four projects". |
| 2.7 | The cluster curation 0.66 → 0.81 purity boost is from lead-author curation, which is potentially circular. | `exp_2_7_build_curation_sheet.py` + `exp_2_7_curation_rubric.md` | Hand the rubric and the auto-built 20-cluster CSV to a non-author rater; compute inter-curator κ; report alongside the curation purity figure. |

## What I cannot do

- **1.1 Second human rater on 50–100 reviews.** Single biggest fix. Needs an actual second human; pick someone unfamiliar with the system to rate the existing 490-review gold-standard subset and the 400 response ratings.

## Order of operations (suggested)

1. Run **2.6** first (cheapest, no LLM cost): mining a 4th repo takes ~5 minutes. Lets you update the abstract/intro to "four projects" early.
2. Run **1.4** next (~$3-5 in API): LLM-CoT-no-taxonomy IssueSpecs. Score against existing rubric.
3. Hand off **2.7** to your non-author rater (40-60 min of their time). Wait for return.
4. Run **1.2** (~$1-2 in API): LLM-no-spec responses. Note: this still needs human rating after generation, so plan for that handoff too.
5. **1.1** (second human rater) is the single highest-leverage move and requires the most external coordination. Start finding a rater now.

## Cost estimate (Claude Opus 4.7)

- 1.2: 100 calls × ~512 tokens out = roughly $1-2
- 1.4: 60 calls × ~1500 tokens out = roughly $3-5
- 2.6: free (GitHub API)
- 2.7: free

Total API cost for the runnable pieces: under $10.
