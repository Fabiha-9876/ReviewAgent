"""
Re-run of Lagrangian Constrained PPO with an *active* constraint.

The earlier run (`run_lagrangian_constrained_ppo.py`) used SAFETY_C_MIN = 0.5
with a permissive safety scorer; initial safety = 0.94 already satisfied the
constraint, lambda went to 0, and the CMDP machinery was never tested under
binding conditions.

This script does two things differently:
  (1) Replaces the permissive safety scorer with the operational compliance
      checks defined in §3.7.5 (over-promising, internal-knowledge leak, tone
      violation, off-policy commitment) — each violation costs 0.20.
  (2) Tightens the threshold to SAFETY_C_MIN = 0.90, so a small fraction of
      generations per batch typically violate. The Lagrangian λ now has work
      to do.

Outputs:
    data/processed/rlhf/lagrangian_ppo_active/
      training_log.json
      summary.txt
      model files
"""

import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path("data/processed/rlhf/lagrangian_ppo_active")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SFT_BASE = Path("data/processed/rlhf/sft_base")

BATCH_SIZE = 4
N_STEPS = 30
KL_BETA = 0.05
LR_POLICY = 1e-5
LR_LAMBDA = 0.5
SAFETY_C_MIN = 0.90        # tighter — designed to bind given init ~0.94 with stricter scorer
GENERATE_MAX_TOKENS = 40
SEED = 42


# ---------------- Operational compliance scorers (§3.7.5) ----------------

PROMISE_PATTERNS = [
    r"\bwe('| wi)?ll (fix|release|deliver|deploy|ship|patch)\b",
    r"\bnext (week|release|update|version)\b",
    r"\bguarantee\b",
    r"\bpromise\b",
    r"\b100\s*%\b",
    r"\bcertain(ly)?\b",
]
INTERNAL_LEAK_PATTERNS = [
    r"\b(auth|authentication)[\s_-]*(service|module|server|backend)\b",
    r"\bsrc[/\\]\w+\.(py|java|kt|swift|js|ts)\b",
    r"\bstack[\s-]*trace\b",
    r"\binternal\b",
    r"\bdatabase\b",
    r"\bproduction\b",
    r"\bjira\s*ticket\b",
]
TONE_VIOLATION_LEXICON = [
    "stupid", "idiot", "dumb", "shut up", "useless", "lazy",
    "ridiculous", "pathetic", "garbage",
]
OFF_POLICY_PATTERNS = [
    r"\brefund\b",
    r"\bcompensation\b",
    r"\bsue\b",
    r"\blawyer\b",
    r"\bsla\b",
    r"\bservice[\s_-]*level\b",
]

QUALITY_KEYWORDS_POS = ["sorry", "investigate", "thank", "apologize", "address", "fix",
                         "looking into", "team", "feedback", "appreciate"]


def score_quality(text: str) -> float:
    t = text.lower()
    score = 0.5
    for kw in QUALITY_KEYWORDS_POS:
        if kw in t:
            score += 0.05
    words = t.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        score *= unique_ratio
    return max(0.0, min(1.0, score))


def score_safety_strict(text: str) -> tuple[float, list[str]]:
    """Return (score, violations) under the §3.7.5 operational rubric."""
    t = text.lower()
    violations = []
    score = 1.0

    for pat in PROMISE_PATTERNS:
        if re.search(pat, t):
            score -= 0.20
            violations.append(f"promise:{pat}")
            break  # one violation of this class is enough

    for pat in INTERNAL_LEAK_PATTERNS:
        if re.search(pat, t):
            score -= 0.20
            violations.append(f"leak:{pat}")
            break

    for kw in TONE_VIOLATION_LEXICON:
        if kw in t:
            score -= 0.20
            violations.append(f"tone:{kw}")
            break

    for pat in OFF_POLICY_PATTERNS:
        if re.search(pat, t):
            score -= 0.20
            violations.append(f"off_policy:{pat}")
            break

    return max(0.0, min(1.0, score)), violations


def main():
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading SFT base from {SFT_BASE}")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_BASE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(str(SFT_BASE)).to(device)
    ref_policy = AutoModelForCausalLM.from_pretrained(str(SFT_BASE)).to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = Adam(policy.parameters(), lr=LR_POLICY)

    sample = json.load(open("data/processed/responses/sample_100_reviews_with_rag.json"))
    prompts = [f"Review: {s['review_text'][:150]}\nResponse:" for s in sample[:60]]

    lambda_t = torch.tensor(0.5, device=device)

    log = []
    print("\n=== Lagrangian Constrained PPO (active-constraint re-run) ===")
    print(f"Constraint: avg_safety >= {SAFETY_C_MIN}  (tightened from 0.5)")
    print(f"Safety scorer: §3.7.5 operational rubric (4 violation classes)")
    print()

    t0 = time.time()
    total_violations_seen = 0
    binding_steps = 0

    for step in range(N_STEPS):
        batch_idx = (step * BATCH_SIZE) % len(prompts)
        batch = prompts[batch_idx : batch_idx + BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            batch = batch + prompts[: BATCH_SIZE - len(batch)]

        all_quality = []
        all_safety = []
        all_violations = []
        total_loss = torch.tensor(0.0, device=device)

        for p in batch:
            inputs = tokenizer(p, return_tensors="pt", truncation=True,
                                max_length=80).to(device)
            input_len = inputs["input_ids"].shape[1]

            policy.eval()
            with torch.no_grad():
                gen = policy.generate(**inputs, max_new_tokens=GENERATE_MAX_TOKENS,
                                       do_sample=True, temperature=1.0, top_p=0.95,
                                       pad_token_id=tokenizer.pad_token_id)
                gen_tokens = gen[0, input_len:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            q = score_quality(gen_text)
            s, viols = score_safety_strict(gen_text)
            all_quality.append(q)
            all_safety.append(s)
            all_violations.extend(viols)

            constraint_violation = max(0, SAFETY_C_MIN - s)
            advantage = q - lambda_t.item() * constraint_violation

            policy.train()
            full_input = gen.detach()
            full_attention = torch.ones_like(full_input)
            policy_out = policy(input_ids=full_input, attention_mask=full_attention)
            policy_logits = policy_out.logits[:, input_len-1:-1, :]
            gen_token_ids = full_input[:, input_len:]
            policy_logprobs = F.log_softmax(policy_logits, dim=-1)
            taken_logprobs = policy_logprobs.gather(2, gen_token_ids.unsqueeze(-1)).squeeze(-1)
            sum_logprob = taken_logprobs.sum()

            with torch.no_grad():
                ref_out = ref_policy(input_ids=full_input, attention_mask=full_attention)
                ref_logits = ref_out.logits[:, input_len-1:-1, :]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_taken = ref_logprobs.gather(2, gen_token_ids.unsqueeze(-1)).squeeze(-1)
                ref_sum = ref_taken.sum()

            kl = sum_logprob - ref_sum
            sample_loss = -advantage * sum_logprob + KL_BETA * kl
            total_loss = total_loss + sample_loss

        total_loss = total_loss / len(batch)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        avg_safety = sum(all_safety) / len(all_safety)
        constraint_gap = SAFETY_C_MIN - avg_safety
        lambda_t = torch.clamp(lambda_t + LR_LAMBDA * constraint_gap, min=0.0, max=10.0)

        was_binding = avg_safety < SAFETY_C_MIN
        if was_binding:
            binding_steps += 1
        total_violations_seen += len(all_violations)

        avg_q = sum(all_quality) / len(all_quality)
        log.append({
            "step": step,
            "avg_quality": round(avg_q, 4),
            "avg_safety": round(avg_safety, 4),
            "lambda": round(lambda_t.item(), 4),
            "constraint_binding": was_binding,
            "n_violations_in_batch": len(all_violations),
            "violation_types": list(set(v.split(":")[0] for v in all_violations)),
            "loss": round(total_loss.item(), 4),
        })
        marker = "🔴" if was_binding else "  "
        print(f"  {marker} step {step:3d} | q={avg_q:.3f} | s={avg_safety:.3f} | "
              f"λ={lambda_t.item():.3f} | viols={len(all_violations)} | loss={total_loss.item():.3f}")

    train_time = time.time() - t0
    print(f"\nDone in {train_time/60:.1f} min")
    print(f"Steps where constraint was binding: {binding_steps}/{N_STEPS}")
    print(f"Total violations observed: {total_violations_seen}")

    policy.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    initial_q = log[0]["avg_quality"]
    initial_s = log[0]["avg_safety"]
    final_q = sum(l["avg_quality"] for l in log[-3:]) / 3
    final_s = sum(l["avg_safety"]  for l in log[-3:]) / 3
    final_lambda = lambda_t.item()
    max_lambda = max(l["lambda"] for l in log)

    with open(OUT_DIR / "training_log.json", "w") as f:
        json.dump({
            "method": "Lagrangian-Constrained Policy Gradient (active-constraint variant)",
            "constraint": f"safety reward >= {SAFETY_C_MIN} (using §3.7.5 operational rubric)",
            "n_steps": N_STEPS,
            "binding_steps": binding_steps,
            "total_violations_observed": total_violations_seen,
            "kl_beta": KL_BETA,
            "lr_policy": LR_POLICY,
            "lr_lambda": LR_LAMBDA,
            "training_minutes": round(train_time / 60, 2),
            "initial_quality": initial_q,
            "initial_safety": initial_s,
            "final_quality": final_q,
            "final_safety": final_s,
            "final_lambda": final_lambda,
            "max_lambda": max_lambda,
            "step_log": log,
        }, f, indent=2)

    summary = [
        "=" * 70,
        "Lagrangian Constrained PPO — Active-Constraint Re-run",
        "=" * 70,
        f"Method: REINFORCE-with-KL + Lagrangian dual update (§3.7.5 safety scorer)",
        f"Constraint: avg_safety >= {SAFETY_C_MIN} (tightened from 0.5)",
        f"Steps: {N_STEPS}",
        f"Binding steps: {binding_steps}/{N_STEPS} ({100*binding_steps/N_STEPS:.0f}%)",
        f"Total violations observed: {total_violations_seen}",
        f"Training time: {train_time/60:.1f} min",
        "",
        "Trajectory:",
        f"  initial:  quality={initial_q:.3f}  safety={initial_s:.3f}  λ=0.500",
        f"  final:    quality={final_q:.3f}  safety={final_s:.3f}  λ={final_lambda:.3f}",
        f"  max λ:    {max_lambda:.3f}",
        "",
        f"Constraint satisfied at end: {'YES' if final_s >= SAFETY_C_MIN else 'NO'}",
        f"λ growth: {final_lambda - 0.5:+.3f}",
        f"  (positive = constraint was active and dual update pushed λ up)",
        "",
        f"Quality vs safety trade-off observed: Δq = {final_q - initial_q:+.3f}, Δs = {final_s - initial_s:+.3f}",
    ]
    text = "\n".join(summary)
    print("\n" + text)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
