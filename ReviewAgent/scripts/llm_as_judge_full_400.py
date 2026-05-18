"""
LLM-as-judge as a third rater across the full 400-row Stage 4b evaluation.

Lead author already rated all 400 (no_spec, full) pairs (data/processed/responses/pairwise_ratings_human.json).
A second human rater scored 30 of those 400 (labmate sub-eval, §4.3.4 of the paper).
This script adds an LLM judge (Claude Opus 4.7) as a third rater across all 400 pairs, enabling:
  - 3-rater Krippendorff's α on the full set
  - Pairwise Cohen's κ for each pair of raters
  - Cross-check that the +2.36 headline is robust to rater identity

Methodology
-----------
Same rubric the human raters used:
  - quality (1-5)
  - helpful (Y/N)
  - preference (A or B)
Blinded: the LLM does not see which response is from `no_spec` vs `full`; A/B labels are shuffled per pair.

Cost
----
400 prompts × ~600 tokens each ≈ $4-8 in Claude Opus 4.7 API cost.

Usage
-----
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/llm_as_judge_full_400.py
  python scripts/llm_as_judge_full_400.py --resume   # skip already-rated pairs

After it completes, run scripts/compute_3rater_krippendorff.py to produce the agreement statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import anthropic

PAIRS_PATH = Path("data/processed/responses/pairwise_ratings_human.json")
NO_SPEC_PATH = Path("data/processed/responses/responses_reviewagent_no_spec.json")
FULL_PATH = Path("data/processed/responses/responses_reviewagent_full.json")
REVIEWS_PATH = Path("data/processed/responses/sample_100_reviews_with_rag.json")
OUT_PATH = Path("data/processed/responses/llm_as_judge_full_400.json")
MODEL = "claude-opus-4-7"
SEED = 17

SYSTEM = """You are an expert developer-relations annotator rating an app-review response.

You will see one user app review and two candidate developer responses (A and B). Rate each response on the same rubric the human raters used:

  quality (1=very poor, 2=poor, 3=okay, 4=good, 5=excellent)
  helpful (Y / N): does the response address the user's actual concern in a way that would plausibly help them?
  preference: which response is better overall, A or B? If genuinely tied, say "tie".

Respond EXACTLY in this format:

A_QUALITY: <1-5>
A_HELPFUL: <Y|N>
B_QUALITY: <1-5>
B_HELPFUL: <Y|N>
PREFERENCE: <A|B|tie>
REASON: <one sentence under 25 words>
"""

USER_TEMPLATE = """Review:
{review}

Response A:
{response_a}

Response B:
{response_b}

Rate each response and pick a preference. Format your answer exactly as specified.
"""


def parse_response(text: str) -> dict:
    out = {"a_quality": None, "a_helpful": None, "b_quality": None, "b_helpful": None,
           "preference": None, "reason": None, "raw": text}
    for ln in text.strip().splitlines():
        u = ln.strip()
        if not u or ":" not in u:
            continue
        key, val = u.split(":", 1)
        key = key.strip().upper()
        val = val.strip()
        if key == "A_QUALITY":
            try: out["a_quality"] = int(val[0])
            except: pass
        elif key == "A_HELPFUL":
            out["a_helpful"] = "Y" if val.upper().startswith("Y") else "N"
        elif key == "B_QUALITY":
            try: out["b_quality"] = int(val[0])
            except: pass
        elif key == "B_HELPFUL":
            out["b_helpful"] = "Y" if val.upper().startswith("Y") else "N"
        elif key == "PREFERENCE":
            v = val.lower()
            out["preference"] = "A" if v.startswith("a") else ("B" if v.startswith("b") else "tie")
        elif key == "REASON":
            out["reason"] = val
    return out


def load_responses() -> dict:
    """Index responses by review_index, returning {review_idx: {'no_spec': ..., 'full': ...}}."""
    nspec = {r["review_index"]: r for r in json.load(open(NO_SPEC_PATH))}
    full = {r["review_index"]: r for r in json.load(open(FULL_PATH))}
    return {"no_spec": nspec, "full": full}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="skip pairs already in OUT_PATH")
    ap.add_argument("--limit", type=int, help="cap number of pairs (debug)")
    args = ap.parse_args()

    if not PAIRS_PATH.exists():
        print(f"Missing: {PAIRS_PATH}", file=sys.stderr)
        return 1

    pairs = json.load(open(PAIRS_PATH))
    resp = load_responses()
    reviews = {}
    if REVIEWS_PATH.exists():
        for r in json.load(open(REVIEWS_PATH)):
            reviews[r.get("review_index", r.get("idx"))] = r.get("review", r.get("text", ""))

    done = {}
    if args.resume and OUT_PATH.exists():
        for r in json.load(open(OUT_PATH)).get("results", []):
            done[r["pair_index"]] = r

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr); return 1
    client = anthropic.Anthropic(api_key=key)

    rng = random.Random(SEED)
    results = list(done.values())

    n_to_do = len(pairs) if not args.limit else min(args.limit, len(pairs))
    todo = [(i, p) for i, p in enumerate(pairs[:n_to_do]) if i not in done]
    print(f"Pairs: {len(pairs)}, already done: {len(done)}, to rate: {len(todo)}", file=sys.stderr)

    for n, (i, p) in enumerate(todo, 1):
        ridx = p["review_index"]
        review_text = reviews.get(ridx, "")
        if not review_text:
            # fall back: use review embedded in the no_spec response file
            nsr = resp["no_spec"].get(ridx, {})
            review_text = nsr.get("review", nsr.get("input", ""))

        # blind shuffle: which condition is A vs B?
        a_is_full = rng.random() < 0.5
        a_cond = "full" if a_is_full else "no_spec"
        b_cond = "no_spec" if a_is_full else "full"
        a_resp = resp[a_cond].get(ridx, {}).get("response", "")
        b_resp = resp[b_cond].get(ridx, {}).get("response", "")
        if not a_resp or not b_resp:
            print(f"  [{n}/{len(todo)}] missing response for review {ridx}, skipping", file=sys.stderr)
            continue

        prompt = USER_TEMPLATE.format(review=review_text[:1500], response_a=a_resp[:1500], response_b=b_resp[:1500])
        try:
            r = client.messages.create(model=MODEL, max_tokens=200, system=SYSTEM,
                                        messages=[{"role": "user", "content": prompt}])
            text = r.content[0].text
        except Exception as e:
            print(f"  [{n}/{len(todo)}] API error: {e}", file=sys.stderr)
            time.sleep(3)
            continue
        parsed = parse_response(text)
        # Un-blind: re-key A/B back to the condition names
        unblinded = {
            "pair_index": i,
            "review_index": ridx,
            "a_condition": a_cond,
            "b_condition": b_cond,
            f"{a_cond}_quality": parsed["a_quality"],
            f"{a_cond}_helpful": parsed["a_helpful"],
            f"{b_cond}_quality": parsed["b_quality"],
            f"{b_cond}_helpful": parsed["b_helpful"],
            "preference_blind": parsed["preference"],
            "preference_condition": (a_cond if parsed["preference"] == "A" else (b_cond if parsed["preference"] == "B" else "tie")),
            "reason": parsed["reason"],
            "raw": parsed["raw"],
        }
        results.append(unblinded)
        if n % 25 == 0:
            json.dump({"model": MODEL, "n_rated": len(results), "results": results}, open(OUT_PATH, "w"), indent=2)
            print(f"  [{n}/{len(todo)}] checkpointed", file=sys.stderr)
        time.sleep(0.4)

    json.dump({"model": MODEL, "n_rated": len(results), "results": results}, open(OUT_PATH, "w"), indent=2)
    print(f"Done. {len(results)} pairs rated -> {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
