"""
Cross-LLM frontier replication for Stage 3 IssueSpec generation.

Closes §5.5 single-LLM dependence by running Stage 3 "LLM with taxonomy" on
three additional frontier models alongside the Claude Opus 4.7 headline:
  - GPT-4o            (proprietary, OpenAI API)
  - Llama-3.1-70B     (open, Together AI; openai-compatible API)
  - Gemini-1.5-Pro    (proprietary, Google API)  [optional]

Uses the same prompt template as scripts/multi_llm_stage3_comparison.py (the
working Qwen run), and the same 100-cluster sample at
  data/processed/issue_specs/sample_100_clusters.json
already used to compare Claude vs Qwen.

Cost
----
  GPT-4o:       ~$3-5  for 100 clusters (~$0.50 for 15)
  Llama-3.1-70B via Together: ~$0.50-1.00 for 100
  Gemini-1.5-Pro: ~$1-2 for 100

Usage
-----
  export OPENAI_API_KEY=sk-...
  export TOGETHER_API_KEY=...
  export GEMINI_API_KEY=...               # optional
  python scripts/multi_llm_stage3_frontier.py --models gpt4o llama3-70b
  python scripts/multi_llm_stage3_frontier.py --n 15        # match the Qwen sub-sample

Outputs
-------
  data/processed/issue_specs/specs_gpt4o.json
  data/processed/issue_specs/specs_llama3_70b.json
  data/processed/issue_specs/specs_gemini.json
  data/processed/issue_specs/frontier_summary.json

Next step: run scripts/recompute_content_validity.py to extend Table 4.2.y.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path("data/processed/issue_specs")
SAMPLE_FILE = BASE / "sample_100_clusters.json"
CLAUDE_FILE = BASE / "specs_with_taxonomy.json"  # for cluster_id alignment

MODELS = {
    "gpt4o":         {"provider": "openai",   "model": "gpt-4o-2024-11-20",                       "out": "specs_gpt4o.json"},
    "llama3-70b":    {"provider": "together", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "out": "specs_llama3_70b.json"},
    "gemini":        {"provider": "google",   "model": "gemini-1.5-pro-002",                      "out": "specs_gemini.json"},
    "llama-groq":    {"provider": "groq",     "model": "llama-3.3-70b-versatile",                 "out": "specs_llama_groq.json"},
    "gemini-flash":  {"provider": "google",   "model": "gemini-2.0-flash",                        "out": "specs_gemini_flash.json"},
}

# Identical to scripts/multi_llm_stage3_comparison.py PROMPT (kept in sync intentionally).
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


def parse_json_from_text(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def build_prompt(cluster: dict) -> str:
    reviews = cluster.get("first_5_review_texts") or cluster.get("representative_reviews") or []
    rev_block = "\n".join(f"- {r}" for r in reviews[:5])
    aspect = cluster.get("auto_name", "").split(":")[-1].strip() if ":" in cluster.get("auto_name", "") else cluster.get("auto_name", "")
    return PROMPT.format(aspect=aspect, issue_type=cluster.get("issue_type", ""), reviews=rev_block)


def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    return r.choices[0].message.content or ""


def call_together(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    return r.choices[0].message.content or ""


def call_groq(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    return r.choices[0].message.content or ""


def call_gemini(prompt: str, model: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    m = genai.GenerativeModel(model)
    r = m.generate_content(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 1500})
    return r.text or ""


CALLERS = {"openai": call_openai, "together": call_together, "google": call_gemini, "groq": call_groq}


def run_one_model(model_key: str, clusters: list, n: int | None) -> dict:
    cfg = MODELS[model_key]
    out_path = BASE / cfg["out"]
    use = clusters if n is None else clusters[:n]
    out_specs = []
    for i, c in enumerate(use, 1):
        prompt = build_prompt(c)
        try:
            text = CALLERS[cfg["provider"]](prompt, cfg["model"])
        except Exception as e:
            print(f"  [{i}/{len(use)}] {model_key} error on {c.get('cluster_id')}: {e}", file=sys.stderr)
            time.sleep(3); continue
        spec = parse_json_from_text(text)
        out_specs.append({
            "cluster_id": c.get("cluster_id"),
            "issue_type": c.get("issue_type"),
            "model": cfg["model"],
            "spec_json": spec,
            "spec_raw": text,
        })
        if i % 10 == 0:
            json.dump(out_specs, open(out_path, "w"), indent=2)
            print(f"  [{i}/{len(use)}] {model_key} checkpointed -> {out_path.name}", file=sys.stderr)
        time.sleep(0.4)
    json.dump(out_specs, open(out_path, "w"), indent=2)
    return {"model_key": model_key, "model": cfg["model"], "n_specs": len(out_specs), "out_path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=["llama-groq", "gemini-flash"])
    ap.add_argument("--n", type=int, help="cluster count (default: all 100)")
    args = ap.parse_args()

    if not SAMPLE_FILE.exists():
        print(f"Missing input: {SAMPLE_FILE}", file=sys.stderr); return 1
    clusters = json.load(open(SAMPLE_FILE))
    print(f"Loaded {len(clusters)} clusters from {SAMPLE_FILE}", file=sys.stderr)

    summary = {"models_run": [], "params": {"n": args.n or "all"}}
    for m in args.models:
        provider = MODELS[m]["provider"]
        key_env = {"openai": "OPENAI_API_KEY", "together": "TOGETHER_API_KEY", "google": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}[provider]
        if not os.environ.get(key_env):
            print(f"  Skipping {m}: {key_env} not set", file=sys.stderr); continue
        info = run_one_model(m, clusters, args.n)
        summary["models_run"].append(info)

    out_summary = BASE / "frontier_summary.json"
    json.dump(summary, open(out_summary, "w"), indent=2)
    print(f"\nModels run: {[s['model_key'] for s in summary['models_run']]}", file=sys.stderr)
    print(f"Summary -> {out_summary}", file=sys.stderr)
    if not summary["models_run"]:
        print("\nNo models executed. Set at least one of: GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, TOGETHER_API_KEY", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
