"""
Ablation A5 (proposal §9): No RAG — generate responses from IssueSpec + composer
only, with RAG retrieval set to empty. Compare BLEU/ROUGE-L/BERTScore vs the full
system.

Mechanically:
  1. Load sample_100_reviews_with_rag.json
  2. Zero out rag_past_responses and rag_similar_responses
  3. Save as sample_100_reviews_no_rag.json
  4. Run the existing full-system composer on the no-RAG sample (the composer
     already reads from sample_100_reviews_with_rag — we monkey-patch the input)
  5. Score outputs against original_response references

Output: data/processed/responses/responses_ablation_a5_no_rag.json
        data/processed/experiments/ablation_a5_results.json
"""

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FULL = ROOT / "data/processed/responses/sample_100_reviews_with_rag.json"
SAMPLE_NORAG = ROOT / "data/processed/responses/sample_100_reviews_no_rag.json"
OUT_RESPONSES = ROOT / "data/processed/responses/responses_ablation_a5_no_rag.json"
OUT_RESULTS = ROOT / "data/processed/experiments/ablation_a5_results.json"

REFERENCE_PATH = ROOT / "data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"


def make_no_rag_sample():
    sample = json.load(open(SAMPLE_FULL))
    for r in sample:
        r["rag_past_responses"] = []
        r["rag_similar_responses"] = []
    with open(SAMPLE_NORAG, "w") as f:
        json.dump(sample, f, indent=2)
    return len(sample)


def regenerate_using_full_composer():
    """Import generate_reviewagent_full.build_response and run on no-RAG sample."""
    spec_path = ROOT / "scripts/generate_reviewagent_full.py"
    spec = importlib.util.spec_from_file_location("genfull", spec_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["genfull"] = mod
    spec.loader.exec_module(mod)

    sample = json.load(open(SAMPLE_NORAG))
    out = []
    for r in sample:
        # The build_response signature in the full generator takes review_dict
        # and emits a dict with response_text. We mimic the full script's
        # processing loop.
        if hasattr(mod, "build_response"):
            resp = mod.build_response(r)
        elif hasattr(mod, "compose_response"):
            resp = mod.compose_response(r)
        else:
            # Fall back: scan the module for the main loop
            raise RuntimeError("No callable response builder found in generate_reviewagent_full.py")
        rec = {
            "review_index": r.get("review_index"),
            "cluster_id": r.get("cluster_id"),
            "issue_type": r.get("issue_type"),
            "review_text": r.get("review_text"),
            "response_text": resp if isinstance(resp, str) else resp.get("response_text"),
            "rag_used": False,
            "condition": "ablation_a5_no_rag",
        }
        out.append(rec)
    with open(OUT_RESPONSES, "w") as f:
        json.dump(out, f, indent=2)
    return out


def score_against_references(responses):
    """Compute BLEU-1/2, ROUGE-L, BERTScore-F1 against RRGen original_response."""
    relabel = json.load(open(REFERENCE_PATH))
    # rrgen_v5_relabeled is a flat list — review_index is the row index.
    pairs = []
    for r in responses:
        idx = r["review_index"]
        if idx is None or idx >= len(relabel):
            continue
        ref = relabel[idx].get("original_response", "")
        if ref:
            pairs.append((r["response_text"], ref))
    print(f"  scored {len(pairs)} (response, ref) pairs")
    if not pairs:
        return {}

    # Reuse the project's own bleu/rouge_l helpers from run_experiments_1_and_2
    spec_e = importlib.util.spec_from_file_location(
        "exp12", ROOT / "scripts/run_experiments_1_and_2.py")
    exp12 = importlib.util.module_from_spec(spec_e)
    sys.modules["exp12"] = exp12
    spec_e.loader.exec_module(exp12)

    cands = [p[0] for p in pairs]
    refs = [p[1] for p in pairs]
    bleu_scores = [exp12.bleu_score(r, c) for c, r in pairs]
    bleu1 = np.mean([b["bleu_1"] for b in bleu_scores])
    bleu2 = np.mean([b["bleu_2"] for b in bleu_scores])
    bleu3 = np.mean([b["bleu_3"] for b in bleu_scores])
    bleu4 = np.mean([b["bleu_4"] for b in bleu_scores])
    rl = np.mean([exp12.rouge_l(r, c) for c, r in pairs])

    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn(cands, refs, lang="en", verbose=False)
        bert_f1 = float(F1.mean())
    except Exception as e:
        print(f"  BERTScore skipped: {e}")
        bert_f1 = None

    mean_len = np.mean([len(c.split()) for c in cands])
    return {
        "n": len(pairs),
        "bleu_1_mean": round(float(bleu1), 4),
        "bleu_2_mean": round(float(bleu2), 4),
        "bleu_3_mean": round(float(bleu3), 4),
        "bleu_4_mean": round(float(bleu4), 4),
        "rouge_l_mean": round(float(rl), 4),
        "bertscore_f1_mean": round(bert_f1, 4) if bert_f1 is not None else None,
        "response_length_mean_words": round(float(mean_len), 2),
    }


def main():
    n = make_no_rag_sample()
    print(f"Built no-RAG sample at {SAMPLE_NORAG} ({n} reviews; rag fields zeroed)")

    print("Regenerating responses with full composer on no-RAG sample...")
    responses = regenerate_using_full_composer()
    print(f"Saved {OUT_RESPONSES} ({len(responses)} responses)")

    print("\nScoring against RRGen references...")
    metrics = score_against_references(responses)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Compare to full-system metrics from exp2_results.json (per-condition top-level)
    exp2 = json.load(open(ROOT / "data/processed/experiments/exp2_results.json"))
    full_metrics = exp2.get("reviewagent_full", {})
    print(f"\nFull system (with RAG) for comparison: {json.dumps(full_metrics, indent=2) if full_metrics else 'N/A'}")

    out = {
        "ablation": "A5 — No RAG (IssueSpec + composer only)",
        "no_rag_metrics": metrics,
        "full_system_metrics_for_comparison": full_metrics,
    }
    Path(OUT_RESULTS).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_RESULTS}")


if __name__ == "__main__":
    main()
