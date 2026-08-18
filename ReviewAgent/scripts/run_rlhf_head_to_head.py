"""
Head-to-head comparison of the 5 trained RLHF policies on the same 100-review
test set used for Stage 4b human evaluation.

Generates one response per (policy, review) pair, scores against RRGen
original_response with BLEU-1/2 + ROUGE-L + BERTScore-F1, and reports per-policy
aggregates. Addresses the Tier-1 weakness "RLHF doesn't actually have a result."

Output:
  data/processed/rlhf/head_to_head/responses.json
  data/processed/rlhf/head_to_head/metrics.json
  data/processed/rlhf/head_to_head/comparison.txt
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data/processed/responses/sample_100_reviews_with_rag.json"
RR = ROOT / "data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"
OUT_DIR = ROOT / "data/processed/rlhf/head_to_head"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = [
    ("sft_base",          "data/processed/rlhf/sft_base"),
    ("kto_model",         "data/processed/rlhf/kto_model"),
    ("dpo_model",         "data/processed/rlhf/dpo_model"),
    ("constrained_proxy", "data/processed/rlhf/constrained_proxy"),
    ("lagrangian_ppo",    "data/processed/rlhf/lagrangian_ppo"),
]

MAX_NEW_TOKENS = 60
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    model.eval()
    return tok, model


def generate(tok, model, prompts):
    """Greedy decode (matches training-time deterministic eval)."""
    out = []
    for p in prompts:
        inputs = tok(p, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                num_beams=1,
            )
        text = tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append(text.strip())
    return out


def main():
    sample = json.load(open(SAMPLE))
    rr = json.load(open(RR))
    prompts = [f"Review: {s['review_text'][:100]}\nResponse:" for s in sample]
    refs = [rr[s["review_index"]]["original_response"] for s in sample]

    # Reuse project's BLEU/ROUGE helpers
    spec = importlib.util.spec_from_file_location(
        "exp12", ROOT / "scripts/run_experiments_1_and_2.py")
    exp12 = importlib.util.module_from_spec(spec)
    sys.modules["exp12"] = exp12
    spec.loader.exec_module(exp12)

    all_responses = {}
    metrics = {}

    for name, path in POLICIES:
        print(f"\n=== {name} ===")
        tok, model = load_model(path)
        gens = generate(tok, model, prompts)
        all_responses[name] = gens

        # Auto metrics
        bleu_scores = [exp12.bleu_score(r, c) for c, r in zip(gens, refs)]
        bleu_1 = float(np.mean([b["bleu_1"] for b in bleu_scores]))
        bleu_2 = float(np.mean([b["bleu_2"] for b in bleu_scores]))
        rl     = float(np.mean([exp12.rouge_l(r, c) for c, r in zip(gens, refs)]))
        try:
            from bert_score import score as bert_score_fn
            _, _, F1 = bert_score_fn(gens, refs, lang="en", verbose=False)
            bert_f1 = float(F1.mean())
        except Exception as e:
            bert_f1 = None
        mean_len = float(np.mean([len(g.split()) for g in gens]))

        metrics[name] = {
            "bleu_1": round(bleu_1, 4),
            "bleu_2": round(bleu_2, 4),
            "rouge_l": round(rl, 4),
            "bertscore_f1": round(bert_f1, 4) if bert_f1 else None,
            "response_length_words": round(mean_len, 1),
            "n_generated": len(gens),
        }
        bs_str = f"{bert_f1:.4f}" if bert_f1 is not None else "NA"
        print(f"  BLEU-1={bleu_1:.4f}  ROUGE-L={rl:.4f}  BERTScore={bs_str}  len={mean_len:.1f}")

        del model, tok
        if DEVICE == "mps":
            torch.mps.empty_cache()

    with open(OUT_DIR / "responses.json", "w") as f:
        json.dump({k: [{"prompt_idx": i, "response": r} for i, r in enumerate(v)] for k, v in all_responses.items()}, f, indent=2)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Comparison table
    lines = ["RLHF policies head-to-head on 100-review test set", "=" * 70]
    lines.append(f"{'policy':25s} {'BLEU-1':>8s} {'BLEU-2':>8s} {'ROUGE-L':>8s} {'BERTScore':>10s} {'len':>5s}")
    lines.append("-" * 70)
    for name, _ in POLICIES:
        m = metrics[name]
        bs = f"{m['bertscore_f1']:.4f}" if m["bertscore_f1"] is not None else "NA"
        lines.append(f"{name:25s} {m['bleu_1']:>8.4f} {m['bleu_2']:>8.4f} {m['rouge_l']:>8.4f} {bs:>10s} {m['response_length_words']:>5.1f}")
    out = "\n".join(lines)
    print("\n" + out)
    with open(OUT_DIR / "comparison.txt", "w") as f:
        f.write(out)
    print(f"\nSaved {OUT_DIR}/")


if __name__ == "__main__":
    main()
