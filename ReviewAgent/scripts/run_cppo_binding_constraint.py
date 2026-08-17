"""
Lagrangian Constrained PPO with a constraint that actually binds.

Why this run exists
-------------------
The distilGPT2 proof-of-concept could not test the CMDP: a 82M model never produces the
fluent developer-relations phrasing the compliance rubric penalises, so compliance was
satisfied at initialisation, lambda collapsed to zero, and the dual objective never engaged.
Reviewers were right that this leaves the quality-versus-compliance trade-off undemonstrated.

The fix is not a bigger model for its own sake, it is a policy capable of violating the
rubric. A probe over Qwen2.5-1.5B-Instruct replies puts the zero-shot violation rate at
5.0 percent, the same rate as the paper's own Stage-4 generations (5.2 percent), and every
violation is over-promising. With a zero-tolerance threshold the constraint therefore binds
from step one.

Setup
    policy       Qwen2.5-1.5B-Instruct with LoRA adapters (the frozen base doubles as the
                 KL reference, so no second copy of the weights is needed)
    objective    maximise E[quality] subject to E[compliance] >= tau
    update       REINFORCE with a KL penalty, plus a Lagrangian dual ascent step on lambda
    quality      the Section 3.7.5 quality rubric
    compliance   the Section 3.7.5 operational rubric: over-promising, internal-knowledge
                 leak, tone violation, off-policy commitment; each violation costs 0.20

What counts as success here is not a better BLEU score. It is evidence that the machinery
engages: lambda rises above zero, the number of binding steps is greater than zero, and
compliance improves while quality pays for it.

Outputs
    data/processed/rlhf/cppo_binding/training_log.json
    data/processed/rlhf/cppo_binding/summary.txt

Usage
    python3 scripts/run_cppo_binding_constraint.py [--steps 30] [--tau 1.0]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/processed/rlhf/cppo_binding"
RATINGS = REPO / "human_work/response_ratings.xlsx"
CPPO_SRC = REPO / "scripts/run_lagrangian_ppo_active_constraint.py"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYS = ("You are an app developer replying publicly to a user review. "
       "Write a short, warm, concrete reply (2-3 sentences) that acknowledges the problem "
       "and tells the user what happens next.")


def load_scorers():
    """Reuse the operational scorers from the distilGPT2 run, unchanged."""
    src = CPPO_SRC.read_text()
    body = ("# scorers"
            + src.split("# ---------------- Operational compliance scorers")[1].split("def main")[0])
    ns = {"re": re}
    exec(body, ns)
    return ns["score_quality"], ns["score_safety_strict"]


def as_score(value):
    return value[0] if isinstance(value, tuple) else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="compliance threshold; 1.0 is zero tolerance for violations")
    ap.add_argument("--kl-beta", type=float, default=0.05)
    ap.add_argument("--lr-policy", type=float, default=1e-5)
    ap.add_argument("--lr-lambda", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=70)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    score_quality, score_safety = load_scorers()

    reviews = (pd.read_excel(RATINGS, sheet_name="Ratings")
               .drop_duplicates("review_index")["review_text"].tolist())
    print(f"{len(reviews)} distinct reviews available as prompts", file=sys.stderr)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL} on {device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32 if device == "cpu" else torch.float16).to(device)
    policy = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    policy.print_trainable_parameters()

    opt = Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr_policy)
    lam = torch.tensor(0.0, device=device)
    log, rng = [], torch.Generator().manual_seed(args.seed)

    def prompt_ids(review):
        chat = tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": f"Review: {review}\n\nYour reply:"}],
            tokenize=False, add_generation_prompt=True)
        return tok(chat, return_tensors="pt").to(device)

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, len(reviews), (args.batch_size,), generator=rng).tolist()
        losses, q_batch, c_batch = [], [], []

        for j in idx:
            enc = prompt_ids(reviews[j])
            plen = enc["input_ids"].shape[1]

            policy.eval()
            with torch.no_grad():
                gen = policy.generate(**enc, max_new_tokens=args.max_new_tokens,
                                      do_sample=True, temperature=0.8, top_p=0.95,
                                      pad_token_id=tok.eos_token_id)
            seq = gen[0].detach().clone()
            completion = seq[plen:]
            if completion.numel() == 0:
                continue
            text = tok.decode(completion, skip_special_tokens=True)
            q = as_score(score_quality(text))
            c = as_score(score_safety(text))
            q_batch.append(q)
            c_batch.append(c)

            full = seq.unsqueeze(0).clone()
            policy.train()
            out = policy(full)
            logits = out.logits[:, plen - 1:-1, :]
            logp = F.log_softmax(logits.float(), dim=-1)
            chosen = logp.gather(-1, completion.view(1, -1, 1)).squeeze(-1)

            with torch.no_grad():
                with policy.disable_adapter():
                    ref_logits = policy(full).logits[:, plen - 1:-1, :]
            ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
            ref_chosen = ref_logp.gather(-1, completion.view(1, -1, 1)).squeeze(-1)
            kl = (chosen - ref_chosen).mean()

            advantage = q + lam.item() * (c - args.tau)
            losses.append(-advantage * chosen.mean() + args.kl_beta * kl)

        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()

        c_mean = statistics.mean(c_batch)
        q_mean = statistics.mean(q_batch)
        # dual ascent: lambda grows while the constraint is violated
        lam = torch.clamp(lam + args.lr_lambda * (args.tau - c_mean), min=0.0)
        binding = c_mean < args.tau
        log.append({"step": step, "quality": round(q_mean, 4), "compliance": round(c_mean, 4),
                    "lambda": round(float(lam), 4), "binding": bool(binding),
                    "loss": round(float(loss), 4),
                    "violations": sum(1 for c in c_batch if c < 1.0)})
        print(f"  step {step:3d} | q={q_mean:.3f} | C={c_mean:.3f} | lambda={float(lam):.3f}"
              f" | {'BINDING' if binding else 'slack'}", file=sys.stderr)

    n_binding = sum(1 for r in log if r["binding"])
    first_half = log[:max(1, len(log) // 2)]
    second_half = log[max(1, len(log) // 2):]
    report = {
        "model": MODEL, "method": "LoRA REINFORCE + KL + Lagrangian dual ascent",
        "tau": args.tau, "steps": len(log), "batch_size": args.batch_size,
        "binding_steps": n_binding,
        "binding_pct": round(100 * n_binding / len(log), 1) if log else 0.0,
        "max_lambda": round(max((r["lambda"] for r in log), default=0.0), 4),
        "final_lambda": round(log[-1]["lambda"], 4) if log else 0.0,
        "compliance_first_half": round(statistics.mean(r["compliance"] for r in first_half), 4),
        "compliance_second_half": round(statistics.mean(r["compliance"] for r in second_half), 4),
        "quality_first_half": round(statistics.mean(r["quality"] for r in first_half), 4),
        "quality_second_half": round(statistics.mean(r["quality"] for r in second_half), 4),
        "total_violations": sum(r["violations"] for r in log),
        "minutes": round((time.time() - t0) / 60, 1),
        "log": log,
    }
    report["constraint_engaged"] = report["max_lambda"] > 0 and n_binding > 0
    (OUT_DIR / "training_log.json").write_text(json.dumps(report, indent=2))

    L = ["=" * 74, "LAGRANGIAN CONSTRAINED PPO WITH A BINDING CONSTRAINT", "=" * 74, "",
         f"policy: {MODEL} (LoRA)   tau = {args.tau}   steps = {len(log)}"
         f"   batch = {args.batch_size}",
         f"runtime: {report['minutes']} min", "",
         f"  binding steps        {n_binding}/{len(log)} ({report['binding_pct']}%)",
         f"  max lambda           {report['max_lambda']}",
         f"  final lambda         {report['final_lambda']}",
         f"  total violations     {report['total_violations']}", "",
         f"  compliance  first half {report['compliance_first_half']:.3f}"
         f"  ->  second half {report['compliance_second_half']:.3f}",
         f"  quality     first half {report['quality_first_half']:.3f}"
         f"  ->  second half {report['quality_second_half']:.3f}", "",
         ("  CONSTRAINT ENGAGED: lambda left zero and the dual update did work."
          if report["constraint_engaged"] else
          "  Constraint never bound; the run degenerates to single-objective again."), ""]
    (OUT_DIR / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
