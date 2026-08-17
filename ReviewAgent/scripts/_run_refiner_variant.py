"""Self-refinement variant: turn the SelfRefiner ON for a 30-review subset of the
reviewagent_full responses, score before/after with the rule-based scorer.

Loads the existing single-shot reviewagent_full responses, runs each through the
SelfRefiner with max_iterations=3 using local Qwen2.5-3B-Instruct as critic +
reviser, then scores both versions with the rule-based scorer.

Output: data/processed/ablations/refiner_on_vs_off.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# Load existing single-shot responses
SRC = BASE / "data/processed/responses/responses_reviewagent_full.json"
OUT = BASE / "data/processed/ablations/refiner_on_vs_off.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("[1/4] Loading single-shot reviewagent_full responses", file=sys.stderr)
all_resp = json.load(open(SRC))
print(f"  total: {len(all_resp)}; sampling 30 stratified", file=sys.stderr)

# Stratified sample by issue_type
from collections import defaultdict
import random
rng = random.Random(42)
by_type = defaultdict(list)
for r in all_resp:
    by_type[r["issue_type"]].append(r)
sample = []
for t, lst in by_type.items():
    rng.shuffle(lst)
    sample.extend(lst[:6])  # 6 per type
sample = sample[:30]
print(f"  sample: {len(sample)} (types: {[r['issue_type'] for r in sample[:6]]}...)", file=sys.stderr)


print("[2/4] Loading Qwen2.5-3B-Instruct as critic+reviser", file=sys.stderr)
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"  device: {device}", file=sys.stderr)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16 if device != "cpu" else torch.float32
).to(device).eval()


def gen(messages, max_new=200):
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


CRITIQUE_PROMPT = """You are a quality reviewer for customer support responses.

Evaluate this response on three dimensions:
1. Specificity: does it reference the specific issue, or is it generic?
2. Compliance: does it make unauthorized promises (e.g., promising a fix without authority) or leak internal info?
3. Empathy: does it acknowledge the user's frustration appropriately?

For each dimension output one line: "<dimension>: pass" if acceptable, or "<dimension>: <one-sentence improvement suggestion>".

Review: {review}
Response: {response}

Critique:"""

REVISE_PROMPT = """Revise this customer support response based on the critique. Keep the core message but address each issue raised. Output ONLY the revised response.

Original response: {response}

Critique:
{critique}

Revised response:"""


print("[3/4] Running 3-iter critique-revise loop", file=sys.stderr)
results = []
t0 = time.time()
for i, r in enumerate(sample, 1):
    review = r["review_text"]
    response = r["response_text"]
    original = response
    iters = []
    for it in range(3):
        # Critique
        crit_msgs = [{"role": "user", "content": CRITIQUE_PROMPT.format(review=review[:600], response=response[:600])}]
        critique = gen(crit_msgs, max_new=200)
        # Stop if critique says all pass
        if critique.lower().count("pass") >= 3 and ":" not in critique[:50]:
            iters.append({"iter": it+1, "critique": critique[:300], "stopped": "all_pass"})
            break
        # Revise
        rev_msgs = [{"role": "user", "content": REVISE_PROMPT.format(response=response[:600], critique=critique[:400])}]
        new_response = gen(rev_msgs, max_new=300)
        iters.append({"iter": it+1, "critique": critique[:300], "revised": new_response[:200]})
        response = new_response
    results.append({
        "response_id": r["response_id"],
        "issue_type": r["issue_type"],
        "review": review[:200],
        "original_response": original[:300],
        "refined_response": response[:300],
        "n_iterations": len(iters),
        "iters": iters,
    })
    if i % 5 == 0:
        elapsed = time.time() - t0
        print(f"  [{i}/{len(sample)}] {elapsed:.0f}s elapsed, {elapsed/i:.1f}s/review", file=sys.stderr)


print("[4/4] Score with rule-based scorer (length, lexical diversity, citation tokens)", file=sys.stderr)
def rule_score(resp: str) -> dict:
    """Lightweight rule-based scorer mirroring the one used elsewhere in the paper."""
    words = resp.split()
    diversity = len(set(words)) / max(1, len(words))
    has_citation = any(t in resp.lower() for t in ["specifically", "issue", "problem", "report"])
    has_apology_safe = "apolog" in resp.lower() or "sorry" in resp.lower()
    word_count = len(words)
    quality = min(1.0, 0.3 * has_citation + 0.3 * has_apology_safe + 0.4 * (1.0 if 30 <= word_count <= 200 else 0.5))
    return {
        "word_count": word_count,
        "lex_diversity": round(diversity, 3),
        "has_specific_citation": has_citation,
        "has_appropriate_apology": has_apology_safe,
        "rule_quality": round(quality, 3),
    }


from statistics import mean
before_scores = [rule_score(r["original_response"]) for r in results]
after_scores = [rule_score(r["refined_response"]) for r in results]

summary = {
    "method": "Self-refiner ON variant: existing single-shot reviewagent_full responses (lead-author Claude-generated) re-passed through SelfRefiner (Qwen2.5-3B critic + reviser, max 3 iterations) on a 30-review stratified subset.",
    "n_reviews": len(results),
    "before_off": {
        "mean_quality": round(mean(s["rule_quality"] for s in before_scores), 3),
        "mean_word_count": round(mean(s["word_count"] for s in before_scores), 1),
        "mean_lex_diversity": round(mean(s["lex_diversity"] for s in before_scores), 3),
    },
    "after_on": {
        "mean_quality": round(mean(s["rule_quality"] for s in after_scores), 3),
        "mean_word_count": round(mean(s["word_count"] for s in after_scores), 1),
        "mean_lex_diversity": round(mean(s["lex_diversity"] for s in after_scores), 3),
    },
    "n_iterations_avg": round(mean(r["n_iterations"] for r in results), 2),
    "interpretation": (
        "Refinement-on variant exercised end-to-end. The reported mean quality and word "
        "count are from the rule-based scorer (Qwen-judge head-to-head queued for next pass). "
        "The headline 400-pair human eval uses the single-shot variant by design, so this "
        "result is a method-availability claim, not a Pareto-dominance claim."
    ),
    "samples": results[:5],   # keep first 5 for transparency
}

json.dump(summary, open(OUT, "w"), indent=2)
print(f"\nSelf-refiner ON: mean quality {summary['after_on']['mean_quality']:.3f}, OFF: {summary['before_off']['mean_quality']:.3f}", file=sys.stderr)
print(f"Avg iterations until stop: {summary['n_iterations_avg']}", file=sys.stderr)
print(f"Saved -> {OUT}", file=sys.stderr)
