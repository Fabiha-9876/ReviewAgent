"""
Tier 1.2 baseline: LLM-generated responses with RAG but WITHOUT IssueSpec.

Why: The existing `responses_reviewagent_no_spec.json` was produced by a
deterministic template composer (scripts/generate_reviewagent_no_spec.py).
That makes the headline 4.62 (full) vs 2.26 (no_spec) gap conflate two
effects: (a) the IssueSpec grounding, and (b) the LLM-vs-template generator.

This script produces a fair LLM counterpart for the no_spec condition by
calling Claude Opus 4.7 via src/common/llm_client.py with:
  - the same review text
  - the same RAG context (3 retrieved past dev responses)
  - NO IssueSpec
on the SAME 100-review test set used for Stage 4b human eval.

Output is JSONL ready for the human-rating UI to consume in a fifth blinded
condition `reviewagent_no_spec_LLM`. After rating, append the row to
Table~\\ref{tab:rq2}; the gap between `reviewagent_full` and the new column
is the clean estimate of IssueSpec contribution alone.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    cd /Users/fabihajalal/Desktop/Review\\ Agent/ReviewAgent
    python paper/experiments/exp_1_2_llm_no_spec.py

Expected runtime: ~3-5 minutes for 100 reviews at ~2s per call.
Expected cost (Claude Opus 4.7): roughly $1-2 for the full 100 reviews.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

REPO_ROOT = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent")
INPUT = REPO_ROOT / "data/processed/responses/sample_100_reviews_with_rag.json"
OUTPUT = REPO_ROOT / "data/processed/responses/responses_reviewagent_no_spec_LLM.json"

# Provider / model. Override via env if needed.
PROVIDER = os.getenv("REVIEWAGENT_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("REVIEWAGENT_LLM_MODEL", "claude-opus-4-7")

SYSTEM_PROMPT = """You are a customer support specialist for a mobile app. Generate a helpful,
empathetic, and specific response to a user's app review.

Guidelines:
- Reference the SPECIFIC issue the user is describing (not a generic response)
- Be empathetic — acknowledge the user's frustration
- If a fix or workaround exists in the reference responses, mention it concretely
- Do NOT make promises you can't keep (e.g., "will be fixed in the next update" unless confirmed)
- Do NOT leak internal information (code details, team names, internal tools)
- Suggest concrete next steps for the user (update the app, try a workaround, contact support)
- Keep the response concise (3-5 sentences)
- Maintain a professional but warm tone"""


def build_user_prompt(review_text: str, rating: int | str, rag_responses: list[str]) -> str:
    sections = [
        f"## User Review (Rating: {rating}/5):",
        f'"{review_text}"',
    ]
    if rag_responses:
        sections.append("\n## Reference Information (similar past developer responses):")
        for i, r in enumerate(rag_responses[:3], 1):
            sections.append(f"- [past_response_{i}]: {r[:300]}")
    sections.append(
        "\n\nGenerate a helpful, empathetic response to this user's review. "
        "Reference the specific problem the user is describing."
    )
    return "\n".join(sections)


def extract_rag_responses(item: dict) -> list[str]:
    """Pull dev-response text from RAG entries, mirroring the existing pipeline."""
    out: list[str] = []
    rag = item.get("rag_context", {}) or {}
    for key in ("similar_past_responses", "past_responses"):
        for hit in rag.get(key, []) or []:
            text = hit.get("response") or hit.get("text") or ""
            if text:
                out.append(text)
    return out[:3]


async def main() -> None:
    # Lazy-import so this script is readable without the project installed.
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.common.llm_client import LLMClient

    client = LLMClient(provider=PROVIDER, model=MODEL)

    with INPUT.open() as f:
        reviews = json.load(f)

    results = []
    for item in reviews:
        rag = extract_rag_responses(item)
        user_prompt = build_user_prompt(
            review_text=item["review_text"],
            rating=item.get("rating", "?"),
            rag_responses=rag,
        )
        text = await client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=512,
        )
        results.append(
            {
                "review_index": item["review_index"],
                "cluster_id": item.get("cluster_id"),
                "review_text": item["review_text"],
                "condition": "reviewagent_no_spec_LLM",
                "response": text.strip(),
                "rag_sources_used": [f"past_response_{i}" for i in range(1, len(rag) + 1)],
                "issue_spec_used": False,
                "model": MODEL,
                "provider": PROVIDER,
            }
        )
        print(f"[{len(results):3d}/100] generated for review_index={item['review_index']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(results)} responses to {OUTPUT}")
    print(
        "\nNext step: add this file as a fifth blinded condition (E) in the "
        "human-rating sheet, re-run human evaluation on conditions A-E, then "
        "append a new row to paper/build/main.tex Table~\\ref{tab:rq2}."
    )


if __name__ == "__main__":
    asyncio.run(main())
