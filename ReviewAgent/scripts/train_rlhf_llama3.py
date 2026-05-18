"""
LoRA fine-tune Llama-3-8B on the 400-pair preference data using KTO, DPO, and
Lagrangian-Constrained PPO. Replaces the distilGPT2 PoC of §3.7.5 with a
generation-grade base model and probes whether the dual-objective CMDP claim is
empirically supported when the constraint can actually bind.

Why distilGPT2 wasn't enough
----------------------------
distilGPT2 already satisfied the safety constraint at initialization
(safety = 0.94 >= tau = 0.50), so the Lagrange multiplier collapsed to zero and
the CMDP machinery never bound. A capable generation model can plausibly violate
the rubric (over-promise a fix date, leak that the team is aware of the bug),
which is the regime the dual-objective formulation is designed for.

Requires
--------
  - GPU with >= 24 GB VRAM (LoRA r=16). Full fine-tune needs >= 80 GB.
  - HF_TOKEN with gated-model access for meta-llama/Meta-Llama-3-8B-Instruct
  - pip install trl peft transformers accelerate bitsandbytes datasets

Usage
-----
  export HF_TOKEN=hf_...
  python scripts/train_rlhf_llama3.py --method kto
  python scripts/train_rlhf_llama3.py --method dpo
  python scripts/train_rlhf_llama3.py --method constrained_ppo

After all three are trained:
  python scripts/run_rlhf_head_to_head.py --base llama3-8b   # existing harness
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PREFS = Path("data/processed/responses/pairwise_ratings_human.json")
OUT_BASE = Path("data/processed/rlhf_llama3")
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_pref_dataset():
    """Convert the 400-pair human ratings into a preference dataset for trl."""
    pairs = json.load(open(PREFS))
    items = []
    for p in pairs:
        chosen_cond = "with_spec" if p["preferred"] == ("A" if p["A_condition"] in ("with_spec", "full") else "B") else "no_spec"
        rejected_cond = "no_spec" if chosen_cond == "with_spec" else "with_spec"
        items.append({"chosen_cond": chosen_cond, "rejected_cond": rejected_cond, "review_index": p["review_index"]})
    return items


def build_prompts(items):
    """Pull review text + response text from the existing response files and emit (prompt, chosen, rejected)."""
    no_spec = {r["review_index"]: r for r in json.load(open("data/processed/responses/responses_reviewagent_no_spec.json"))}
    full = {r["review_index"]: r for r in json.load(open("data/processed/responses/responses_reviewagent_full.json"))}
    out = []
    for it in items:
        ridx = it["review_index"]
        review = no_spec.get(ridx, {}).get("review", "") or full.get(ridx, {}).get("review", "")
        chosen = (full if it["chosen_cond"] == "with_spec" else no_spec).get(ridx, {}).get("response", "")
        rejected = (no_spec if it["chosen_cond"] == "with_spec" else full).get(ridx, {}).get("response", "")
        if not review or not chosen or not rejected:
            continue
        prompt = f"User review:\n{review}\n\nWrite a developer-style response to this user."
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out


def train_kto(ds, out_dir):
    from trl import KTOTrainer, KTOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    cfg = KTOConfig(output_dir=str(out_dir), num_train_epochs=2, per_device_train_batch_size=2,
                     gradient_accumulation_steps=4, learning_rate=5e-5, beta=0.1, logging_steps=10, save_steps=200)
    trainer = KTOTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))


def train_dpo(ds, out_dir):
    from trl import DPOTrainer, DPOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    cfg = DPOConfig(output_dir=str(out_dir), num_train_epochs=2, per_device_train_batch_size=2,
                     gradient_accumulation_steps=4, learning_rate=5e-5, beta=0.1, logging_steps=10, save_steps=200)
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))


def train_constrained_ppo(ds, out_dir):
    """
    Lagrangian Constrained PPO with quality reward + compliance constraint.
    See src/stage5/constrained_ppo.py for the constraint-violation scorer.
    Llama-3-8B is a capable enough generator that the constraint can plausibly bind,
    which is the regime the CMDP formulation is designed for.
    """
    from src.stage5.constrained_ppo import LagrangianConstrainedPPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    trainer = LagrangianConstrainedPPOTrainer(model=model, tokenizer=tok, dataset=ds, output_dir=str(out_dir),
                                                 num_steps=300, batch_size=4, lr=1e-5,
                                                 safety_threshold=0.90, lambda_init=0.0, lambda_lr=0.01)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["kto", "dpo", "constrained_ppo"], required=True)
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set (Llama-3 is a gated model)", file=sys.stderr); return 1

    out_dir = OUT_BASE / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading preference data from {PREFS}...", file=sys.stderr)
    items = load_pref_dataset()
    prompts = build_prompts(items)
    print(f"Built {len(prompts)} (prompt, chosen, rejected) triples", file=sys.stderr)

    from datasets import Dataset
    ds = Dataset.from_list(prompts)

    if args.method == "kto":
        train_kto(ds, out_dir)
    elif args.method == "dpo":
        train_dpo(ds, out_dir)
    elif args.method == "constrained_ppo":
        train_constrained_ppo(ds, out_dir)

    print(f"Saved -> {out_dir}/final", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
