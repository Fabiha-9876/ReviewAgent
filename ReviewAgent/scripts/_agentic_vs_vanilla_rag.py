"""Agentic vs vanilla RAG empirical comparison (closes #7 gap).

Vanilla RAG = existing reviewagent_full responses (single-shot, refinement_iterations=0).
Agentic RAG = same generator + SelfRefiner with up to 2 critique-revise iterations,
              using local Qwen2.5-3B-Instruct as critic + reviser.

Sample: 10 stratified IssueSpec responses (2 per issue_type).
Scoring: rule-based scorer (length, lexical diversity, citation tokens, apology safety).

Output: data/processed/ablations/agentic_vs_vanilla_rag.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import defaultdict
from statistics import mean

import torch

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data/processed/responses/responses_reviewagent_full.json"
OUT = BASE / "data/processed/ablations/agentic_vs_vanilla_rag.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("[1/4] Loading vanilla-RAG (single-shot) responses + sampling 10", file=sys.stderr)
all_resp = json.load(open(SRC))
import random
rng = random.Random(42)
by_type = defaultdict(list)
for r in all_resp:
    by_type[r["issue_type"]].append(r)
sample = []
for t, lst in by_type.items():
    rng.shuffle(lst)
    sample.extend(lst[:2])  # 2 per issue_type
sample = sample[:10]
print(f"  sample: {len(sample)} (types: {[r['issue_type'] for r in sample]})", file=sys.stderr)


print("[2/4] Loading Qwen2.5-3B-Instruct (critic + reviser)", file=sys.stderr)
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16 if device != "cpu" else torch.float32
).to(device).eval()


def gen(messages, max_new=180):
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


CRITIQUE = """You are a quality reviewer for customer support responses. Score the response on three dimensions and suggest improvements.

1. Specificity: does it reference the specific issue?
2. Compliance: does it make unauthorized promises?
3. Empathy: does it acknowledge frustration?

For each: write "<dim>: pass" if acceptable, or "<dim>: <one-sentence suggestion>" if not.

Review: {review}
Response: {response}

Critique:"""

REVISE = """Revise the response based on the critique. Keep the core message but address each issue. Output ONLY the revised response, no preamble.

Original: {response}

Critique:
{critique}

Revised:"""


print("[3/4] Running 2-iter agentic loop on Qwen", file=sys.stderr)
results = []
t0 = time.time()
MAX_ITERS = 2
for i, r in enumerate(sample, 1):
    review = r["review_text"][:500]
    vanilla = r["response_text"]
    response = vanilla
    iters_log = []
    for it in range(MAX_ITERS):
        c_msgs = [{"role": "user", "content": CRITIQUE.format(review=review, response=response[:500])}]
        critique = gen(c_msgs, max_new=180)
        # Early stop if critique passes everything
        passes = sum(critique.lower().count(f"{d}: pass") for d in ["specificity", "compliance", "empathy"])
        if passes >= 3:
            iters_log.append({"iter": it+1, "critique": critique[:200], "stop": "all_pass"})
            break
        r_msgs = [{"role": "user", "content": REVISE.format(response=response[:500], critique=critique[:300])}]
        new_response = gen(r_msgs, max_new=250)
        iters_log.append({"iter": it+1, "critique": critique[:200], "revised_preview": new_response[:120]})
        response = new_response
    results.append({
        "response_id": r["response_id"],
        "issue_type": r["issue_type"],
        "review": review[:200],
        "vanilla_response": vanilla[:300],
        "agentic_response": response[:300],
        "n_iterations": len(iters_log),
        "iters": iters_log,
    })
    elapsed = time.time() - t0
    print(f"  [{i}/{len(sample)}] {elapsed:.0f}s elapsed, {elapsed/i:.0f}s/review", file=sys.stderr)


print("[4/4] Score with rule-based scorer", file=sys.stderr)
def score(text):
    words = text.split()
    div = len(set(words)) / max(1, len(words))
    has_cite = any(t in text.lower() for t in ["specifically", "issue", "problem", "report", "we've identified"])
    has_apol = "apolog" in text.lower() or "sorry" in text.lower()
    wc = len(words)
    q = min(1.0, 0.3 * has_cite + 0.3 * has_apol + 0.4 * (1.0 if 30 <= wc <= 250 else 0.5))
    return {"word_count": wc, "lex_div": round(div, 3),
            "has_citation": has_cite, "has_apology": has_apol,
            "rule_quality": round(q, 3)}


vanilla_scores = [score(r["vanilla_response"]) for r in results]
agentic_scores = [score(r["agentic_response"]) for r in results]

summary = {
    "method": ("Agentic vs vanilla RAG empirical comparison. Vanilla = existing single-shot "
               "reviewagent_full responses (Claude Opus 4.7). Agentic = same baseline + "
               "Qwen2.5-3B-Instruct SelfRefiner with up to 2 critique-revise iterations. "
               "Stratified 10-review sample (2 per issue type)."),
    "n_reviews": len(results),
    "max_iterations": MAX_ITERS,
    "vanilla_rag": {
        "mean_quality": round(mean(s["rule_quality"] for s in vanilla_scores), 3),
        "mean_word_count": round(mean(s["word_count"] for s in vanilla_scores), 1),
        "mean_lex_diversity": round(mean(s["lex_div"] for s in vanilla_scores), 3),
        "pct_with_citation": round(100 * mean(s["has_citation"] for s in vanilla_scores), 1),
    },
    "agentic_rag": {
        "mean_quality": round(mean(s["rule_quality"] for s in agentic_scores), 3),
        "mean_word_count": round(mean(s["word_count"] for s in agentic_scores), 1),
        "mean_lex_diversity": round(mean(s["lex_div"] for s in agentic_scores), 3),
        "pct_with_citation": round(100 * mean(s["has_citation"] for s in agentic_scores), 1),
    },
    "delta_agentic_minus_vanilla": {
        "mean_quality": round(mean(s["rule_quality"] for s in agentic_scores) -
                              mean(s["rule_quality"] for s in vanilla_scores), 3),
        "mean_word_count": round(mean(s["word_count"] for s in agentic_scores) -
                                 mean(s["word_count"] for s in vanilla_scores), 1),
    },
    "n_iterations_avg": round(mean(r["n_iterations"] for r in results), 2),
    "interpretation": (
        "Empirical agentic-RAG vs vanilla-RAG comparison (closes the empirical gap "
        "noted in Sec. 'What we compare against'). Direction matters: positive delta "
        "means refinement adds value at PoC scale; near-zero delta means single-shot "
        "is already near the local optimum. Headline 400-pair eval uses vanilla."
    ),
    "samples": results[:3],   # first 3 for transparency
}

json.dump(summary, open(OUT, "w"), indent=2)
print(f"\nVanilla RAG quality: {summary['vanilla_rag']['mean_quality']}", file=sys.stderr)
print(f"Agentic RAG quality: {summary['agentic_rag']['mean_quality']}", file=sys.stderr)
print(f"Delta:               {summary['delta_agentic_minus_vanilla']['mean_quality']:+.3f}", file=sys.stderr)
print(f"Avg iterations:      {summary['n_iterations_avg']}", file=sys.stderr)
print(f"Saved -> {OUT}", file=sys.stderr)
