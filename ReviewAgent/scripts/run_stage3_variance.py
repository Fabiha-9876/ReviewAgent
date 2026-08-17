"""
Run-to-run variance of Stage-3 spec generation (reviewer-requested).

Every LLM-dependent number in the paper comes from a single generation run, so the paper
could not say how stable those numbers are. This script repeats Stage-3 generation k times
on the same clusters with the same prompt and a fixed sampling temperature, then reports
the spread of the two metrics the paper reports for Stage 3:

    strict template-fill rate   (the Section 5.1 criterion)
    SpecCov                     (the extractive-coverage faithfulness score)

The generator here is the local Qwen2.5-3B-Instruct, not the headline Claude run, because
repeating the headline run needs API budget. What transfers is the magnitude of run-to-run
noise for a decoder-only model on this prompt, which is what a reader needs in order to
know whether small differences between conditions are meaningful.

Output
    data/processed/ablations/stage3_variance.json
    data/processed/ablations/stage3_variance.txt

Usage
    python3 scripts/run_stage3_variance.py [--runs 3] [--n-clusters 15] [--temperature 0.7]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from speccov import speccov_score  # noqa: E402

SPECS = REPO / "data/processed/issue_specs"
OUT = REPO / "data/processed/ablations/stage3_variance.json"
OUT_TXT = OUT.with_suffix(".txt")
MODEL = "Qwen/Qwen2.5-3B-Instruct"

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

REQUIRED = {
    "bug_report": ["title", "description", "steps_to_reproduce", "expected_behavior",
                   "actual_behavior", "severity", "affected_component"],
    "feature_request": ["title", "description", "user_story", "acceptance_criteria",
                        "severity", "affected_component"],
    "performance": ["title", "description", "nfr_category", "severity", "affected_component"],
    "usability": ["title", "description", "nielsen_heuristic", "severity", "affected_component"],
    "compatibility": ["title", "description", "device_os_matrix", "severity",
                      "affected_component"],
}


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    body = text[start:end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", body))
        except json.JSONDecodeError:
            return None


def strict_fill(spec, issue_type):
    """Section 5.1 strict criterion: every required field present and substantive."""
    if not spec:
        return 0.0
    ok = 0
    fields = REQUIRED.get(issue_type, [])
    for f in fields:
        v = spec.get(f)
        if not v:
            continue
        if f == "steps_to_reproduce":
            steps = v if isinstance(v, list) else [v]
            if len(steps) >= 3 and all(len(str(s).split()) >= 2 for s in steps[:3]):
                ok += 1
        elif f == "acceptance_criteria":
            items = v if isinstance(v, list) else [v]
            if len(items) >= 3 and all(len(str(s).split()) >= 8 for s in items[:3]):
                ok += 1
        elif f == "user_story":
            s = str(v).lower()
            if "as a" in s and "i want" in s and ("so that" in s or "so i" in s):
                ok += 1
        elif f == "description":
            if len(str(v).split()) >= 30:
                ok += 1
        elif f in ("expected_behavior", "actual_behavior"):
            if len(str(v).split()) >= 8:
                ok += 1
        elif f == "affected_component":
            if len(str(v).split()) >= 2:
                ok += 1
        else:
            if str(v).strip():
                ok += 1
    return ok / len(fields) if fields else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-clusters", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    args = ap.parse_args()

    clusters = json.load(open(SPECS / "sample_100_clusters.json"))
    claude = {s["cluster_id"]: s for s in json.load(open(SPECS / "specs_with_taxonomy.json"))}
    by_type, selected = {}, []
    for c in clusters:
        if c["cluster_id"] not in claude:
            continue
        t = c.get("issue_type")
        by_type.setdefault(t, []).append(c)
    per_type = max(1, args.n_clusters // max(1, len(by_type)))
    for t, lst in sorted(by_type.items()):
        selected.extend(lst[:per_type])
    selected = selected[:args.n_clusters]
    print(f"{len(selected)} clusters across {len(by_type)} issue types", file=sys.stderr)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL} on {device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device).eval()

    runs = []
    for r in range(args.runs):
        torch.manual_seed(1000 + r)
        fills, covs, parse_fail = [], [], 0
        t0 = time.time()
        for i, cl in enumerate(selected, 1):
            reviews = "\n".join(f"- {x}" for x in (cl.get("first_5_review_texts")
                                                   or cl.get("representative_reviews") or [])[:5])
            prompt = PROMPT.format(aspect=cl.get("auto_name", ""),
                                   issue_type=cl.get("issue_type", ""), reviews=reviews)
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
            enc = tok(chat, return_tensors="pt").to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=args.temperature, top_p=0.95,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            spec = parse_json(text)
            if spec is None:
                parse_fail += 1
                fills.append(0.0)
                continue
            spec = dict(spec)
            spec["issue_type"] = cl.get("issue_type")
            fills.append(strict_fill(spec, cl.get("issue_type")))
            covs.append(speccov_score(spec, cl))
            print(f"  run {r+1}/{args.runs}  {i}/{len(selected)}", file=sys.stderr, end="\r")
        runs.append({"run": r + 1, "seed": 1000 + r,
                     "mean_strict_fill": round(sum(fills) / len(fills), 4),
                     "mean_speccov": round(sum(covs) / len(covs), 4) if covs else None,
                     "parse_failures": parse_fail, "seconds": round(time.time() - t0, 1)})
        print(f"  run {r+1}: fill={runs[-1]['mean_strict_fill']:.3f} "
              f"speccov={runs[-1]['mean_speccov']}  "
              f"parse_fail={parse_fail}  ({runs[-1]['seconds']:.0f}s)", file=sys.stderr)

    def spread(key):
        vals = [r[key] for r in runs if r[key] is not None]
        return {"values": vals, "mean": round(statistics.mean(vals), 4),
                "sd": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "range": round(max(vals) - min(vals), 4)}

    report = {"model": MODEL, "runs": args.runs, "n_clusters": len(selected),
              "temperature": args.temperature, "per_run": runs,
              "strict_fill": spread("mean_strict_fill"), "speccov": spread("mean_speccov")}
    OUT.write_text(json.dumps(report, indent=2))

    L = ["=" * 74, "STAGE-3 RUN-TO-RUN VARIANCE", "=" * 74, "",
         f"model: {MODEL}   temperature: {args.temperature}",
         f"{len(selected)} clusters, regenerated {args.runs} times with different seeds", "",
         f"  {'run':>4s}{'strict fill':>14s}{'SpecCov':>11s}{'parse fails':>13s}"]
    for r in runs:
        L.append(f"  {r['run']:>4d}{r['mean_strict_fill']:>14.3f}"
                 f"{(r['mean_speccov'] if r['mean_speccov'] is not None else float('nan')):>11.3f}"
                 f"{r['parse_failures']:>13d}")
    L += ["",
          f"  strict fill: mean {report['strict_fill']['mean']:.3f}, "
          f"sd {report['strict_fill']['sd']:.3f}, range {report['strict_fill']['range']:.3f}",
          f"  SpecCov    : mean {report['speccov']['mean']:.3f}, "
          f"sd {report['speccov']['sd']:.3f}, range {report['speccov']['range']:.3f}", ""]
    OUT_TXT.write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
