"""LLM-as-judge Y/P/N purity audit on the 605 KG-hierarchical clusters,
using local Qwen2.5-3B-Instruct as judge (no API key needed).

Mirrors audit_hierarchical_cluster_purity_llm.py but uses a local model.
Same rubric (Y/P/N) and the same 50-cluster stratified sample as flat-194 audit.

Output: data/processed/kg_hierarchical/llm_judge_purity_audit_qwen.json
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
CLUSTERS_PATH = BASE / "data/processed/kg_hierarchical/hierarchical_clusters.json"
OUT_PATH = BASE / "data/processed/kg_hierarchical/llm_judge_purity_audit_qwen.json"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SEED = 42

SYSTEM_PROMPT = """You are an expert software-engineering annotator auditing the coherence of an app-review cluster.

You will be given 5 review excerpts and the cluster's aspect label. Judge whether the 5 reviews share the same SUB-THEME within that aspect.

Output exactly one of: Y, P, N
  Y = all 5 share the sub-theme
  P = 3 or 4 of 5 share the sub-theme (partial coherence)
  N = fewer than 3 share the sub-theme (incoherent)

Then on a new line, give a one-sentence justification (under 25 words).

Format your response EXACTLY as:
VERDICT: <Y|P|N>
REASON: <one sentence>"""

USER_TEMPLATE = """Cluster aspect: {aspect}
Cluster sub-aspect ID: {cluster_id}
Cluster issue-type (V5 classifier majority): {issue_type}

5 representative reviews:
1. {r1}
2. {r2}
3. {r3}
4. {r4}
5. {r5}

What is your Y/P/N verdict on whether these 5 reviews share the same sub-theme within the aspect "{aspect}"?"""


def parse_verdict(text: str):
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    verdict = ""
    reason = ""
    for ln in lines:
        if ln.upper().startswith("VERDICT:"):
            v = ln.split(":", 1)[1].strip().upper()
            if v.startswith("Y"):
                verdict = "Y"
            elif v.startswith("P"):
                verdict = "P"
            elif v.startswith("N"):
                verdict = "N"
        elif ln.upper().startswith("REASON:"):
            reason = ln.split(":", 1)[1].strip()
    return verdict, reason


def stratified_sample(clusters, n, seed):
    by_type = defaultdict(list)
    for c in clusters:
        by_type[c.get("issue_type", "unknown")].append(c)
    rng = random.Random(seed)
    types = sorted(by_type)
    quota = max(1, n // len(types))
    sample = []
    for t in types:
        pool = by_type[t]
        rng.shuffle(pool)
        sample.extend(pool[:quota])
    rng.shuffle(sample)
    return sample[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    clusters = json.load(open(CLUSTERS_PATH))
    print(f"Loaded {len(clusters)} clusters", file=sys.stderr)

    # Filter: must have >= 5 reps; many KG clusters have exactly 3 reps.
    # Some clusters store 3 reps only; we'll need to fall back to padding or sampling fewer.
    # Check rep distribution first.
    rep_counts = defaultdict(int)
    for c in clusters:
        rep_counts[len(c.get("representative_reviews", []))] += 1
    print(f"Rep count distribution: {dict(rep_counts)}", file=sys.stderr)

    # If most have 3 reps, switch the rubric to "3 of 3" Y/P/N.
    # (Y = all share; P = 2 of 3 share; N = 0-1 share)
    sample = stratified_sample(clusters, args.n, SEED)
    print(f"Sampled {len(sample)} clusters", file=sys.stderr)

    # Load Qwen
    print(f"Loading {MODEL_ID}", file=sys.stderr)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device).eval()
    print(f"  device: {device}", file=sys.stderr)

    # If most clusters have 3 reps, adjust template
    use_three = max(rep_counts, key=rep_counts.get) <= 3
    if use_three:
        SYS = SYSTEM_PROMPT.replace(
            "5 review excerpts", "3 review excerpts"
        ).replace(
            "Y = all 5 share the sub-theme",
            "Y = all 3 share the sub-theme"
        ).replace(
            "P = 3 or 4 of 5 share the sub-theme (partial coherence)",
            "P = 2 of 3 share the sub-theme (partial coherence)"
        ).replace(
            "N = fewer than 3 share the sub-theme (incoherent)",
            "N = at most 1 shares the sub-theme (incoherent)"
        )
        USR = """Cluster aspect: {aspect}
Cluster sub-aspect ID: {cluster_id}
Cluster issue-type (V5 classifier majority): {issue_type}

3 representative reviews:
1. {r1}
2. {r2}
3. {r3}

What is your Y/P/N verdict on whether these reviews share the same sub-theme within the aspect "{aspect}"?"""
    else:
        SYS = SYSTEM_PROMPT
        USR = USER_TEMPLATE

    results = []
    counts = {"Y": 0, "P": 0, "N": 0, "error": 0}
    t0 = time.time()
    for i, c in enumerate(sample, 1):
        reps = c.get("representative_reviews", [])
        if (use_three and len(reps) < 3) or (not use_three and len(reps) < 5):
            counts["error"] += 1
            continue
        fmt = {
            "aspect": c.get("aspect", ""),
            "cluster_id": c.get("cluster_id", ""),
            "issue_type": c.get("issue_type", ""),
            "r1": reps[0][:300],
            "r2": reps[1][:300],
            "r3": reps[2][:300],
        }
        if not use_three:
            fmt["r4"] = reps[3][:300]
            fmt["r5"] = reps[4][:300]
        prompt = USR.format(**fmt)

        # Apply chat template
        messages = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt").to(device)
        with torch.inference_mode():
            out_ids = model.generate(
                **enc, max_new_tokens=80, do_sample=False, temperature=0.0,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        verdict, reason = parse_verdict(gen)
        if verdict in counts:
            counts[verdict] += 1
        else:
            counts["error"] += 1

        results.append({
            "cluster_id": c["cluster_id"],
            "aspect": c.get("aspect"),
            "issue_type": c.get("issue_type"),
            "review_count": c.get("review_count"),
            "verdict": verdict,
            "reason": reason,
            "raw": gen.strip()[:400],
        })
        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(sample)}] tally Y={counts['Y']} P={counts['P']} N={counts['N']} err={counts['error']} ({elapsed:.1f}s)",
                  file=sys.stderr)

    n_judged = counts["Y"] + counts["P"] + counts["N"]
    weighted_purity = ((counts["Y"] + 0.5 * counts["P"]) / n_judged) if n_judged else 0.0

    per_class = defaultdict(lambda: {"Y": 0, "P": 0, "N": 0, "n": 0})
    for r in results:
        v = r.get("verdict")
        if v in ("Y", "P", "N"):
            t = r.get("issue_type", "unknown") or "unknown"
            per_class[t][v] += 1
            per_class[t]["n"] += 1

    out = {
        "method": "qwen2.5_3b_llm_as_judge_y_p_n_audit",
        "judge_model": MODEL_ID,
        "n_sampled": len(sample),
        "n_judged": n_judged,
        "n_errors": counts["error"],
        "overall_counts": {k: counts[k] for k in ("Y", "P", "N")},
        "weighted_purity": weighted_purity,
        "per_class": {t: dict(d) for t, d in per_class.items()},
        "sample_seed": SEED,
        "rep_count_distribution": dict(rep_counts),
        "rubric_uses_three_reps": use_three,
        "comparison_with_flat": {
            "flat_50_cluster_lead_author_audit": 0.66,
            "hierarchical_qwen_judge": weighted_purity,
            "n_hierarchical": n_judged,
        },
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nWeighted Y/P/N purity (KG hierarchical, Qwen-judge): {weighted_purity:.3f} on {n_judged} clusters", file=sys.stderr)
    print(f"  Y={counts['Y']} P={counts['P']} N={counts['N']} err={counts['error']}", file=sys.stderr)
    print(f"Saved -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
