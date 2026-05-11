"""
Run Qwen2.5-3B-Instruct on Stage 4b condition (1) rrgen_baseline.

Re-generate the 100 responses (currently Claude-only) using Qwen, then
score the new Qwen responses with the same rule-based quality + safety
scorers used in §4.7.2 to enable side-by-side cross-LLM comparison.

Outputs:
    data/processed/responses/responses_rrgen_baseline_qwen.json
    data/processed/responses/stage4b_cross_llm_summary.txt
"""

import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, "scripts")
from score_rlhf_policies_with_rubric import score_quality, score_safety_strict

CLAUDE_FILE = Path("data/processed/responses/responses_rrgen_baseline.json")
OUT_QWEN = Path("data/processed/responses/responses_rrgen_baseline_qwen.json")
OUT_SUM = Path("data/processed/responses/stage4b_cross_llm_summary.txt")

PROMPT = """You are a developer-relations responder writing a brief, helpful reply to a mobile-app user review.

Review:
"{review}"

Write a 2-3 sentence response that acknowledges the issue, expresses empathy, and offers a concrete next step (e.g., contact support with a specific bit of info, or note that the team is working on it). Keep it concise and dev-rel professional. Do not over-promise. Output only the response, no preamble."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    model.eval()

    with open(CLAUDE_FILE) as f:
        claude = json.load(f)

    n = min(args.n, len(claude))
    print(f"Re-generating {n} responses with Qwen2.5-3B...")

    qwen = []
    t0 = time.time()
    for i in range(n):
        rec = claude[i]
        prompt_text = PROMPT.format(review=rec["review_text"][:400])
        msgs = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        qwen.append({
            "response_id": rec["response_id"].replace("resp_b1_", "resp_qwen_"),
            "review_index": rec["review_index"],
            "cluster_id": rec["cluster_id"],
            "issue_type": rec["issue_type"],
            "review_text": rec["review_text"],
            "response_text": out,
            "condition": "rrgen_baseline_qwen2_5_3b",
        })

        elapsed = time.time() - t0
        if (i+1) % 10 == 0:
            eta = elapsed / (i+1) * (n - i - 1)
            print(f"  {i+1}/{n} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    print(f"Done in {(time.time()-t0)/60:.1f} min")

    OUT_QWEN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_QWEN, "w") as f:
        json.dump(qwen, f, indent=2)
    print(f"Saved Qwen responses: {OUT_QWEN}")

    # Score both with rule-based scorers
    import numpy as np
    def aggregate(responses):
        qs = [score_quality(r["response_text"]) for r in responses]
        ss = []
        viols = 0
        for r in responses:
            s, v = score_safety_strict(r["response_text"])
            ss.append(s)
            viols += len(v)
        return {
            "n": len(responses),
            "quality_mean": float(np.mean(qs)),
            "quality_std": float(np.std(qs)),
            "safety_mean": float(np.mean(ss)),
            "n_violations": viols,
            "mean_words": float(np.mean([len(r["response_text"].split()) for r in responses])),
        }

    claude_score = aggregate(claude[:n])
    qwen_score = aggregate(qwen)

    summary = [
        "=" * 72,
        "Stage 4b condition (1) rrgen_baseline -- cross-LLM comparison",
        "=" * 72,
        f"n responses each: {n}",
        "",
        f"{'metric':<25} {'Claude Opus 4.7':>18} {'Qwen2.5-3B':>15}",
        "-" * 60,
        f"{'quality (rule-based)':<25} {claude_score['quality_mean']:>18.3f} {qwen_score['quality_mean']:>15.3f}",
        f"{'safety (rule-based)':<25} {claude_score['safety_mean']:>18.3f} {qwen_score['safety_mean']:>15.3f}",
        f"{'rule violations (total)':<25} {claude_score['n_violations']:>18} {qwen_score['n_violations']:>15}",
        f"{'mean response length':<25} {claude_score['mean_words']:>18.1f} {qwen_score['mean_words']:>15.1f}",
        "",
        "Note: rule-based scorer is the §3.7.5 operational rubric (quality keywords",
        "+ §3.7.5 compliance violations: over-promising, internal-knowledge leak,",
        "tone violation, off-policy commitment).",
    ]
    text = "\n".join(summary)
    print("\n" + text)
    with open(OUT_SUM, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
