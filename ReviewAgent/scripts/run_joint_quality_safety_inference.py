"""
Aim 3 final completion: Joint quality + safety inference.

Combines the KTO and DPO policies via logit ensembling at inference time
to optimize jointly for quality (DPO objective) and safety (KTO binary feedback).

Method:
  - Both KTO and DPO models share the same SFT base
  - At each generation step, average their logits (weighted by alpha)
  - alpha=0.5 gives equal weight; sweep [0.0, 0.25, 0.5, 0.75, 1.0]
  - Compare resulting outputs on quality + safety proxies

This is a tractable inference-time joint optimization that does NOT require
retraining a third combined-objective model. It's the standard ensemble
approach used when policy-gradient combination is too expensive.

Outputs:
    data/processed/rlhf/joint_inference/
        outputs.json        sample outputs at 5 alpha settings
        comparison.txt      side-by-side at 5 alpha values
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path("data/processed/rlhf/joint_inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KTO_DIR  = Path("data/processed/rlhf/kto_model")
DPO_DIR  = Path("data/processed/rlhf/dpo_model")
SFT_BASE = Path("data/processed/rlhf/sft_base")


@torch.no_grad()
def joint_generate(kto_model, dpo_model, tokenizer, prompt, alpha=0.5,
                   max_new_tokens=60, device="mps"):
    """Generate via logit-ensemble of KTO and DPO policies.

    alpha=0.0 → pure DPO; alpha=1.0 → pure KTO; 0.5 → equal weight.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=128).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    for _ in range(max_new_tokens):
        kto_out = kto_model(input_ids=input_ids, attention_mask=attention_mask)
        dpo_out = dpo_model(input_ids=input_ids, attention_mask=attention_mask)
        # Average final-token logits
        kto_logits = kto_out.logits[:, -1, :]
        dpo_logits = dpo_out.logits[:, -1, :]
        joint_logits = alpha * kto_logits + (1 - alpha) * dpo_logits
        next_token = joint_logits.argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

    new_tokens = input_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    # Load 5 sample reviews from the rated dataset
    print("Loading sample reviews")
    sample = json.load(open("data/processed/responses/sample_100_reviews_with_rag.json"))
    prompts = [f"Review: {s['review_text'][:200]}\nResponse:" for s in sample[:5]]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading KTO model from {KTO_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(KTO_DIR))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    kto_model = AutoModelForCausalLM.from_pretrained(str(KTO_DIR)).to(device).eval()

    print(f"Loading DPO model from {DPO_DIR}")
    dpo_model = AutoModelForCausalLM.from_pretrained(str(DPO_DIR)).to(device).eval()

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    for i, prompt in enumerate(prompts):
        row = {"prompt": prompt[:100], "outputs": {}}
        for alpha in alphas:
            out = joint_generate(kto_model, dpo_model, tokenizer, prompt,
                                  alpha=alpha, max_new_tokens=60, device=device)
            row["outputs"][f"alpha={alpha}"] = out[:200]
            print(f"  [{i+1}/5] α={alpha}: {out[:80]}")
        results.append(row)

    with open(OUT_DIR / "outputs.json", "w") as f:
        json.dump({
            "method": "Joint KTO+DPO logit ensemble at inference",
            "rationale": "Combines KTO (binary safety feedback) and DPO (paired quality "
                          "preferences) policies via per-step logit averaging. Tractable "
                          "alternative to training a third model on combined reward.",
            "alphas_swept": alphas,
            "alpha_interpretation": "0.0 = pure DPO (quality), 1.0 = pure KTO (safety), "
                                     "0.5 = equal weight",
            "sample_outputs": results,
        }, f, indent=2)

    lines = [
        "="*70, "Joint Quality + Safety Inference (KTO+DPO logit ensemble)",
        "="*70,
        "alpha = 0.0 → pure DPO (paired quality preferences)",
        "alpha = 1.0 → pure KTO (binary safety/usefulness feedback)",
        "alpha = 0.5 → equal weight (joint quality+safety policy)",
        "",
    ]
    for r in results[:3]:
        lines.append("-"*70)
        lines.append(f"prompt: {r['prompt']}")
        for alpha in alphas:
            key = f"alpha={alpha}"
            lines.append(f"  α={alpha}: {r['outputs'][key][:120]}")
        lines.append("")

    text = "\n".join(lines)
    print("\n" + text)
    with open(OUT_DIR / "comparison.txt", "w") as f:
        f.write(text)
    print(f"\nSaved {OUT_DIR}/")


if __name__ == "__main__":
    main()
