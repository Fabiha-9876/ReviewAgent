"""
Tier 1.4 baseline: LLM with chain-of-thought prompt, NO taxonomy grounding.

Why: A reviewer will ask whether the LLM-with-taxonomy win in Table 1
(completeness 1.00 vs free-form 0.619) is just "we used a better prompt"
rather than "structured templates help." The current `LLM free-form`
condition uses a single sentence prompt; that is a weak baseline.

This script gives the LLM the same clusters but with a competitive
2024-era chain-of-thought prompt: explicit reasoning steps, asks the
model to think about issue type, severity, reproduction steps, and
acceptance criteria — but does NOT pin those to Zimmermann / ISO 25010 /
Nielsen / user-story templates.

If the taxonomy-grounded condition still wins on structural completeness
against this competitive CoT baseline, the win is real. If the gap
collapses, the contribution is the prompt, not the taxonomy.

Output:
    data/processed/issue_specs/specs_llm_cot_no_taxonomy.json

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    cd /Users/fabihajalal/Desktop/Review\\ Agent/ReviewAgent
    python paper/experiments/exp_1_4_llm_cot_no_taxonomy.py

Expected runtime: ~5-8 minutes for the 60 bug+feature clusters at
~5s per call (longer because CoT outputs are bigger).
Expected cost (Claude Opus 4.7): roughly $3-5 for 60 clusters.

After running, evaluate the output specs against the existing rubric used
for Table 1 (completeness, desc.\\ words, bugs w/ steps). Add as a column
to Table~\\ref{tab:rq1}; the column to add is "LLM CoT (no taxonomy)".
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

REPO_ROOT = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent")
INPUT = REPO_ROOT / "data/processed/issue_specs/sample_100_clusters.json"
OUTPUT = REPO_ROOT / "data/processed/issue_specs/specs_llm_cot_no_taxonomy.json"

PROVIDER = os.getenv("REVIEWAGENT_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("REVIEWAGENT_LLM_MODEL", "claude-opus-4-7")

SYSTEM_PROMPT = """You are an expert software engineer triaging clusters of user app reviews
into structured issue reports for a developer audience. You write issues
that GitHub maintainers would accept without revision.

When you see a cluster of related reviews, work through these steps before
writing the spec:
  1. Identify the dominant issue type. Is this a bug, a feature request,
     a performance complaint, a usability problem, or a compatibility
     issue? State your reasoning.
  2. Determine severity. How blocking is this for affected users? P0
     (cannot use the app), P1 (major friction), P2 (workaround exists),
     P3 (polish). Justify your choice.
  3. Identify the affected component. Which subsystem of the app is
     implicated? Be specific (not "the app" — e.g., "background sync
     scheduler" or "notification permission flow").
  4. For bugs: produce reproduction steps, expected behavior, and actual
     behavior. For features: produce a user story and acceptance criteria.
     For performance/usability/compatibility: produce the relevant
     diagnostics (resource type, heuristic violated, device/OS matrix).
  5. Write a concise title (under 12 words) and a 2-4 sentence
     description.

Output a single JSON object with these fields:
  title, issue_type, severity, affected_component, description,
  steps_to_reproduce (list, may be empty), expected_behavior,
  actual_behavior, user_story, acceptance_criteria (list).

Only output the JSON object. No prose before or after."""


def build_user_prompt(cluster: dict) -> str:
    reviews = cluster.get("reviews", []) or cluster.get("sample_reviews", [])
    cluster_name = cluster.get("auto_name") or cluster.get("name") or "(unnamed)"
    issue_type_hint = cluster.get("issue_type", "")
    parts = [f"## Cluster: {cluster_name}"]
    if issue_type_hint:
        parts.append(f"## Predominant issue type (hint): {issue_type_hint}")
    parts.append("\n## Sample reviews from this cluster:")
    for i, r in enumerate(reviews[:5], 1):
        text = r.get("text") or r.get("review_text") or str(r)
        parts.append(f"{i}. {text[:400]}")
    parts.append(
        "\n\nThink step-by-step (steps 1-5 above), then output the JSON spec."
    )
    return "\n".join(parts)


def parse_json_from_response(text: str) -> dict:
    """Extract the JSON object from the LLM's response."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[len("json"):].lstrip()
    # Find the first { and the matching }
    try:
        start = text.index("{")
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    except (ValueError, json.JSONDecodeError) as e:
        return {"_parse_error": str(e), "_raw": text}
    return {"_parse_error": "no_json_found", "_raw": text}


async def main() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.common.llm_client import LLMClient

    client = LLMClient(provider=PROVIDER, model=MODEL)

    with INPUT.open() as f:
        clusters = json.load(f)

    # Restrict to bug + feature clusters to match Table 1's bug+feature subset.
    bug_feat = [
        c for c in clusters
        if c.get("issue_type") in ("bug_report", "feature_request", "bug", "feature")
    ]
    if not bug_feat:
        bug_feat = clusters  # fall back if the issue_type field is named differently

    results = []
    for cluster in bug_feat:
        user_prompt = build_user_prompt(cluster)
        text = await client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
        spec = parse_json_from_response(text)
        results.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "issue_type_hint": cluster.get("issue_type"),
                "spec": spec,
                "raw_response": text,
                "model": MODEL,
                "provider": PROVIDER,
                "condition": "llm_cot_no_taxonomy",
            }
        )
        print(f"[{len(results):3d}/{len(bug_feat)}] generated for cluster_id={cluster.get('cluster_id')}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(results)} specs to {OUTPUT}")
    print(
        "\nNext step: score these specs with the same rubric used for the existing "
        "Table 1 columns (completeness across the 14 template fields, desc. words, "
        "bugs w/ steps coverage). Add a new column 'LLM CoT (no taxonomy)' to "
        "paper/build/main.tex Table~\\ref{tab:rq1}."
    )


if __name__ == "__main__":
    asyncio.run(main())
