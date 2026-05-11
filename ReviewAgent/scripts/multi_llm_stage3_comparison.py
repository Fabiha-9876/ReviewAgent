"""
Multi-LLM Stage 3 IssueSpec generation comparison (Reviewer Gap #19).

Re-generate IssueSpecs on a 20-cluster subset using a *local* LLM
(Qwen2.5-3B-Instruct, Apache-2.0, no API needed) and compare against the
existing Claude Opus 4.7 specs on:
  - structural completeness (loose + strict §3.8.1.x)
  - bugs populating steps_to_reproduce
  - features populating user_story
  - description length
  - field-level cross-LLM agreement

Outputs:
    data/processed/issue_specs/specs_qwen2_5_3b.json
    data/processed/issue_specs/multi_llm_comparison.json
    data/processed/issue_specs/multi_llm_comparison_summary.txt
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("data/processed/issue_specs")
CLAUDE_FILE = BASE / "specs_with_taxonomy.json"
SAMPLE_FILE = BASE / "sample_100_clusters.json"
OUT_QWEN = BASE / "specs_qwen2_5_3b.json"
OUT_CMP  = BASE / "multi_llm_comparison.json"
OUT_SUM  = BASE / "multi_llm_comparison_summary.txt"

PROMPT = """You are a software-engineering triage expert. Convert the following app-review cluster into a structured issue specification.

Cluster aspect: {aspect}
Issue type: {issue_type}
Sample reviews from this cluster:
{reviews}

Produce a JSON object with these fields (populate every required field for the issue type):

For bug_report: title (short), description (~30+ words), steps_to_reproduce (list of >=3 concrete steps with action verbs), expected_behavior (>=8 words), actual_behavior (>=8 words), severity (P0|P1|P2|P3), affected_component (>=2 words, specific not generic).

For feature_request: title, description, user_story (must use "As a ... I want ... so that ..." format), acceptance_criteria (list of >=3 concrete items each >=8 words), severity, affected_component.

For performance: title, description, nfr_category (one of: speed, battery, memory, responsiveness, scalability), severity, affected_component.

For usability: title, description, nielsen_heuristic (one of Nielsen's 10), severity, affected_component.

For compatibility: title, description, device_os_matrix (dict mapping device names to OS version lists), severity, affected_component.

Output ONLY a valid JSON object, no preamble or postscript."""


def parse_json_from_text(text):
    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try fixing trailing commas
        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Claude specs and cluster sample
    with open(CLAUDE_FILE) as f:
        claude_specs = json.load(f)
    with open(SAMPLE_FILE) as f:
        clusters = json.load(f)
    cluster_by_id = {c["cluster_id"]: c for c in clusters}

    # Take first n_clusters that have Claude specs
    selected = []
    for cs in claude_specs[:args.n_clusters]:
        cid = cs["cluster_id"]
        if cid in cluster_by_id:
            selected.append((cs, cluster_by_id[cid]))

    print(f"Will generate Qwen specs for {len(selected)} clusters")

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    model.eval()

    qwen_specs = []
    n_parse_fails = 0
    t0 = time.time()
    for i, (claude_spec, cluster) in enumerate(selected):
        cid = claude_spec["cluster_id"]
        itype = claude_spec.get("issue_type") or cluster.get("issue_type") or "bug_report"
        aspect = cluster.get("auto_name") or cluster.get("aspect") or itype
        reviews = cluster.get("representative_reviews", [])[:5] or cluster.get("first_5_review_texts", [])[:5]
        review_block = "\n".join(f"- {r[:200]}" for r in reviews if isinstance(r, str))

        prompt = PROMPT.format(aspect=aspect, issue_type=itype, reviews=review_block)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        out = tokenizer.decode(new_tokens, skip_special_tokens=True)

        parsed = parse_json_from_text(out)
        if parsed is None:
            n_parse_fails += 1
            spec = {"cluster_id": cid, "issue_type": itype, "_parse_failure": True, "_raw": out[:500]}
        else:
            spec = {"cluster_id": cid, "issue_type": itype, **parsed, "condition": "qwen2_5_3b"}
        qwen_specs.append(spec)

        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (len(selected) - i - 1)
        print(f"  {i+1}/{len(selected)} | {cid} | {itype} | parse_ok={parsed is not None} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  (parse failures: {n_parse_fails}/{len(selected)})")

    OUT_QWEN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_QWEN, "w") as f:
        json.dump(qwen_specs, f, indent=2)
    print(f"Saved Qwen specs: {OUT_QWEN}")

    # ---- Compute structural-completeness comparison ----
    import sys
    sys.path.insert(0, "scripts")
    from recompute_content_validity import (
        is_strict_nonempty, REQUIRED, strict_completeness, loose_completeness,
    )

    def per_field_strict(spec, itype):
        req = REQUIRED.get(itype, [])
        return {f: is_strict_nonempty(f, spec.get(f)) for f in req}

    cmp = {"per_cluster": [], "claude_aggregates": {}, "qwen_aggregates": {}}
    bug_steps = {"claude": [], "qwen": []}
    feat_us = {"claude": [], "qwen": []}
    loose_c = {"claude": [], "qwen": []}
    strict_c = {"claude": [], "qwen": []}

    for (cs, _), qs in zip(selected, qwen_specs):
        itype = cs.get("issue_type") or "bug_report"
        c_loose, _, _ = loose_completeness(cs)
        q_loose, _, _ = loose_completeness(qs)
        c_strict, _, _ = strict_completeness(cs)
        q_strict, _, _ = strict_completeness(qs)
        loose_c["claude"].append(c_loose)
        loose_c["qwen"].append(q_loose)
        strict_c["claude"].append(c_strict)
        strict_c["qwen"].append(q_strict)

        if itype == "bug_report":
            bug_steps["claude"].append(1 if is_strict_nonempty("steps_to_reproduce", cs.get("steps_to_reproduce")) else 0)
            bug_steps["qwen"].append(  1 if is_strict_nonempty("steps_to_reproduce", qs.get("steps_to_reproduce")) else 0)
        if itype == "feature_request":
            feat_us["claude"].append(1 if is_strict_nonempty("user_story", cs.get("user_story")) else 0)
            feat_us["qwen"].append(  1 if is_strict_nonempty("user_story", qs.get("user_story")) else 0)

        cmp["per_cluster"].append({
            "cluster_id": cs["cluster_id"],
            "issue_type": itype,
            "claude_loose": c_loose, "claude_strict": c_strict,
            "qwen_loose":   q_loose, "qwen_strict":   q_strict,
            "claude_per_field_strict": per_field_strict(cs, itype),
            "qwen_per_field_strict":   per_field_strict(qs, itype),
        })

    import numpy as np
    cmp["claude_aggregates"] = {
        "n": len(loose_c["claude"]),
        "loose_fill_mean":  float(np.mean(loose_c["claude"])),
        "strict_fill_mean": float(np.mean(strict_c["claude"])),
        "bugs_with_strict_steps_pct":  100*np.mean(bug_steps["claude"]) if bug_steps["claude"] else None,
        "feats_with_strict_userstory_pct": 100*np.mean(feat_us["claude"]) if feat_us["claude"] else None,
        "n_bugs": len(bug_steps["claude"]),
        "n_feats": len(feat_us["claude"]),
    }
    cmp["qwen_aggregates"] = {
        "n": len(loose_c["qwen"]),
        "loose_fill_mean":  float(np.mean(loose_c["qwen"])),
        "strict_fill_mean": float(np.mean(strict_c["qwen"])),
        "bugs_with_strict_steps_pct":  100*np.mean(bug_steps["qwen"]) if bug_steps["qwen"] else None,
        "feats_with_strict_userstory_pct": 100*np.mean(feat_us["qwen"]) if feat_us["qwen"] else None,
        "n_parse_failures": n_parse_fails,
    }

    # Field-level cross-LLM agreement (% of fields where both LLMs make same fill/no-fill judgment)
    n_fields_compared = 0
    n_agree = 0
    for entry in cmp["per_cluster"]:
        for f in entry["claude_per_field_strict"]:
            if f in entry["qwen_per_field_strict"]:
                n_fields_compared += 1
                if entry["claude_per_field_strict"][f] == entry["qwen_per_field_strict"][f]:
                    n_agree += 1
    cmp["field_level_agreement_pct"] = 100.0 * n_agree / max(n_fields_compared, 1)
    cmp["n_fields_compared"] = n_fields_compared

    with open(OUT_CMP, "w") as f:
        json.dump(cmp, f, indent=2)
    print(f"Saved comparison: {OUT_CMP}")

    # ---- Summary ----
    summary = [
        "=" * 72,
        "Multi-LLM Stage 3 IssueSpec comparison",
        "=" * 72,
        f"Compared models: Claude Opus 4.7 (existing) vs Qwen2.5-3B-Instruct (new local run)",
        f"Subset: {len(selected)} clusters (first {args.n_clusters} of the 100-cluster headline sample)",
        f"Qwen parse failures: {n_parse_fails}/{len(selected)}",
        "",
        f"{'metric':<42} {'Claude Opus 4.7':>17} {'Qwen2.5-3B':>14}",
        "-" * 74,
    ]

    def row(name, c, q, fmt="{:.3f}"):
        cv = fmt.format(c) if c is not None else "n/a"
        qv = fmt.format(q) if q is not None else "n/a"
        return f"{name:<42} {cv:>17} {qv:>14}"

    ca = cmp["claude_aggregates"]; qa = cmp["qwen_aggregates"]
    summary.append(row("template-fill rate (loose)", ca["loose_fill_mean"], qa["loose_fill_mean"]))
    summary.append(row("template-fill rate (STRICT §3.8.1.x)", ca["strict_fill_mean"], qa["strict_fill_mean"]))
    summary.append(row(f"bugs with strict steps_to_reproduce % (n={ca['n_bugs']})",
                        ca["bugs_with_strict_steps_pct"], qa["bugs_with_strict_steps_pct"], "{:.1f}"))
    summary.append(row(f"features with strict user_story % (n={ca['n_feats']})",
                        ca["feats_with_strict_userstory_pct"], qa["feats_with_strict_userstory_pct"], "{:.1f}"))
    summary.append("")
    summary.append(f"Field-level cross-LLM agreement (strict fill judgment): "
                    f"{cmp['field_level_agreement_pct']:.1f}% across {cmp['n_fields_compared']} field-comparisons")

    text = "\n".join(summary)
    print("\n" + text)
    with open(OUT_SUM, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
