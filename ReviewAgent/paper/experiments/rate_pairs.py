"""
Blinded pairwise rating CLI for the LLM-with-spec vs LLM-no-spec comparison.

For each of 100 reviews, presents both responses (A/B random order, blinded)
and collects:
  - quality 1-5 for each
  - helpful Y/N for each
  - which-is-better (A / B / equal)

Saves progress after every rating so you can quit and resume with `python rate_pairs.py`.

Run:
    cd /Users/fabihajalal/Desktop/Review\\ Agent/ReviewAgent
    python paper/experiments/rate_pairs.py

Estimated time: 30-60 min for 100 pairs (~20-30 sec each).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent")
NO_SPEC = REPO / "data/processed/responses/responses_reviewagent_no_spec_LLM.json"
WITH_SPEC = REPO / "data/processed/responses/responses_reviewagent_full_LLM.json"
OUT = REPO / "data/processed/responses/pairwise_ratings_human.json"
SEED = 1337


def load_pairs():
    no_spec = {r["review_index"]: r for r in json.load(NO_SPEC.open())}
    with_spec = {r["review_index"]: r for r in json.load(WITH_SPEC.open())}
    rng = random.Random(SEED)
    pairs = []
    for ri in sorted(no_spec.keys()):
        if ri not in with_spec:
            continue
        # randomize which is A and which is B
        order = ["no_spec", "with_spec"]
        rng.shuffle(order)
        pairs.append(
            {
                "review_index": ri,
                "review_text": no_spec[ri]["review_text"],
                "A_condition": order[0],
                "A_response": no_spec[ri]["response"] if order[0] == "no_spec" else with_spec[ri]["response"],
                "B_condition": order[1],
                "B_response": no_spec[ri]["response"] if order[1] == "no_spec" else with_spec[ri]["response"],
            }
        )
    return pairs


def load_progress():
    if OUT.exists():
        return {r["review_index"]: r for r in json.load(OUT.open())}
    return {}


def save_progress(ratings: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(ratings.values(), key=lambda r: r["review_index"])
    with OUT.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def prompt_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  please enter a number {lo}-{hi}")


def prompt_yn(prompt: str) -> str:
    while True:
        s = input(prompt).strip().lower()
        if s in ("y", "yes"):
            return "Y"
        if s in ("n", "no"):
            return "N"
        print("  please enter Y or N")


def prompt_choice(prompt: str, options: list[str]) -> str:
    opts = "/".join(options)
    while True:
        s = input(f"{prompt} [{opts}]: ").strip().upper()
        if s in [o.upper() for o in options]:
            return s
        print(f"  please enter one of {opts}")


def main():
    pairs = load_pairs()
    ratings = load_progress()
    print(f"\n=== Blinded pairwise rating: LLM-with-IssueSpec vs LLM-no-IssueSpec ===")
    print(f"Total pairs: {len(pairs)}; already rated: {len(ratings)}; remaining: {len(pairs) - len(ratings)}")
    print("Quality scale: 1=generic/unhelpful, 2=vague, 3=basic, 4=specific+empathetic, 5=excellent")
    print("Type Q at any prompt to quit and resume later.\n")

    try:
        for i, pair in enumerate(pairs, 1):
            ri = pair["review_index"]
            if ri in ratings:
                continue
            print("=" * 80)
            print(f"[{i}/{len(pairs)}]  review_index={ri}")
            print(f"\nUSER REVIEW:\n  {pair['review_text']}\n")
            print(f"--- Response A ---\n{pair['A_response']}\n")
            print(f"--- Response B ---\n{pair['B_response']}\n")

            try:
                qa = prompt_int("  Response A quality (1-5): ", 1, 5)
                ha = prompt_yn("  Response A helpful? (Y/N): ")
                qb = prompt_int("  Response B quality (1-5): ", 1, 5)
                hb = prompt_yn("  Response B helpful? (Y/N): ")
                pref = prompt_choice("  Which is better overall?", ["A", "B", "EQUAL"])
            except (KeyboardInterrupt, EOFError):
                print("\n\n[saved progress; rerun the script to resume]")
                save_progress(ratings)
                return

            ratings[ri] = {
                "review_index": ri,
                "A_condition": pair["A_condition"],
                "B_condition": pair["B_condition"],
                "A_quality": qa,
                "A_helpful": ha,
                "B_quality": qb,
                "B_helpful": hb,
                "preferred": pref,
            }
            save_progress(ratings)
            print(f"  saved.\n")

        print("=" * 80)
        print(f"All {len(pairs)} pairs rated. Output: {OUT}")
        print("Tell Claude to wire the results into the paper.")

    except (KeyboardInterrupt, EOFError):
        save_progress(ratings)
        print("\n[progress saved; rerun the script to resume]")


if __name__ == "__main__":
    main()
