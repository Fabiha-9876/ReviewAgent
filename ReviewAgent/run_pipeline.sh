#!/usr/bin/env bash
#
# run_pipeline.sh — orchestrate the IssueSpec five-stage pipeline end-to-end.
#
# Usage:
#   ./run_pipeline.sh              # run all five stages in order
#   ./run_pipeline.sh 2            # run only Stage 2
#   ./run_pipeline.sh 2 3 4        # run Stages 2, 3, 4
#   ./run_pipeline.sh verify       # just re-verify paper numbers (needs data bundle)
#
# Run from the repository root. Each stage's scripts carry docstrings describing
# their exact inputs/outputs. Stages 1, 3, 4 need an LLM (Gemini/Qwen/Claude)
# and/or a GPU; Stage 2 is CPU-bound; Stage 5 needs a GPU (small, distilGPT2).
#
set -euo pipefail

cd "$(dirname "$0")"          # always run from the repo root
PY="${PYTHON:-python3}"

# ----------------------------------------------------------------------
log()   { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
stage() { printf '\n\033[1;35m========== STAGE %s ==========\033[0m\n' "$*"; }
run()   { printf '   \033[0;33m$ %s\033[0m\n' "$*"; "$@"; }
warn()  { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }

# ----------------------------------------------------------------------
preflight() {
  log "Pre-flight checks"
  command -v "$PY" >/dev/null || { echo "ERROR: $PY not found"; exit 1; }
  [ -d scripts ] || { echo "ERROR: run from the repository root (no scripts/ here)"; exit 1; }
  if [ ! -d data/processed ]; then
    warn "data/processed/ not found — download the Zenodo bundle (10.5281/zenodo.20320410)"
    warn "and extract it here, or Stage scripts will regenerate it from raw RRGen."
  fi
  [ -f .env ] || warn ".env not found — Stage 1/3/4 LLM steps may need ANTHROPIC_API_KEY / HF_TOKEN."
}

# ----------------------------------------------------------------------
stage1() {
  stage "1 — Intake, Classification, Aspect Extraction"
  run "$PY" scripts/llm_label_rrgen.py
  run "$PY" scripts/cleanlab_find_label_issues.py
  run "$PY" scripts/correct_rrgen_v2.py
  run "$PY" scripts/ingest_verified_labels.py
  run "$PY" scripts/build_compat_data.py
  run "$PY" scripts/kfold_classifier.py
  run "$PY" scripts/_eval_v5_on_maalej.py
  log "Stage 1 done — expect V5 Cohen's kappa ~0.59 on the 490-review gold."
}

stage2() {
  stage "2 — Three-Layer Knowledge Graph"
  run "$PY" scripts/extract_aspects_local_llm.py
  run "$PY" scripts/cluster_phase1b_umap_hdbscan.py
  run "$PY" scripts/compute_cluster_quality_metrics.py
  run "$PY" scripts/ablation_a1b_fine_flat_vs_kg.py
  run "$PY" scripts/audit_hierarchical_cluster_purity_llm.py
  log "Stage 2 done — expect 605 sub-clusters, 5.4x lower DB vs flat."
}

stage3() {
  stage "3 — Review-to-Issue Translation (IssueSpec)"
  run "$PY" scripts/multi_llm_stage3_comparison.py
  run "$PY" scripts/speccov.py \
      --specs   data/processed/issue_specs/specs_with_taxonomy.json \
      --clusters data/processed/issue_specs/sample_100_clusters.json \
      --out      data/processed/speccov_scores.json
  run "$PY" scripts/_qwen_judge_5dim_rubric.py
  run "$PY" scripts/mine_github_issues.py
  log "Stage 3 done — expect template-fill 0.96 vs 0.53 GitHub, rubric 3.89/5."
}

stage4() {
  stage "4 — Spec-aware Response Generation (RAG)"
  run "$PY" scripts/populate_rag.py
  run "$PY" scripts/generate_reviewagent_full.py
  run "$PY" scripts/generate_reviewagent_no_spec.py
  run "$PY" scripts/run_ablation_a5_no_rag.py
  run "$PY" scripts/_agentic_vs_vanilla_rag.py
  log "Stage 4 done — expect +2.36 Likert (full vs no_spec)."
}

stage5() {
  stage "5 — CMDP-grounded RLHF"
  run "$PY" scripts/run_rlhf_proof_of_concept.py
  run "$PY" scripts/run_constrained_ppo_proxy.py
  run "$PY" scripts/run_lagrangian_constrained_ppo.py
  run "$PY" scripts/run_rlhf_head_to_head.py
  run "$PY" scripts/score_rlhf_policies_with_rubric.py
  log "Stage 5 done — expect constrained-proxy +52% BLEU-1 over SFT-base."
}

verify() {
  stage "VERIFY — reproduce every paper number from saved data"
  run "$PY" verify_paper_results.py
}

# ----------------------------------------------------------------------
preflight

if [ "$#" -eq 0 ]; then
  TARGETS=(1 2 3 4 5)
else
  TARGETS=("$@")
fi

for t in "${TARGETS[@]}"; do
  case "$t" in
    1) stage1 ;;
    2) stage2 ;;
    3) stage3 ;;
    4) stage4 ;;
    5) stage5 ;;
    verify) verify ;;
    *) echo "Unknown target: $t (use 1-5 or 'verify')"; exit 1 ;;
  esac
done

log "Pipeline finished. Run './run_pipeline.sh verify' to check all numbers."
