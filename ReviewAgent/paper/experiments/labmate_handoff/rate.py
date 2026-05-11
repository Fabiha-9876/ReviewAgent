"""
Blinded paired rating tool. ~15-30 minutes for 30 pairs.

Run from the folder you received:
    python3 rate.py

Saves your ratings to ratings.json after every entry.
You can quit any time with Ctrl+C and resume by rerunning.
"""

from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
PAIRS = HERE / "pairs_30.json"
OUT = HERE / "ratings.json"


def load():
    pairs = json.load(PAIRS.open())
    done = {}
    if OUT.exists():
        for r in json.load(OUT.open()):
            done[r["review_index"]] = r
    return pairs, done


def save(done):
    rows = sorted(done.values(), key=lambda r: r["review_index"])
    with OUT.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def ask_int(prompt, lo, hi):
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  please enter a number {lo}-{hi}")


def ask_yn(prompt):
    while True:
        s = input(prompt).strip().lower()
        if s in ("y", "yes"):
            return "Y"
        if s in ("n", "no"):
            return "N"
        print("  please enter Y or N")


def ask_choice(prompt, options):
    opts = "/".join(options)
    while True:
        s = input(f"{prompt} [{opts}]: ").strip().upper()
        if s in [o.upper() for o in options]:
            return s
        print(f"  please enter one of {opts}")


def main():
    pairs, done = load()
    print(f"\n=== Blinded paired rating ===")
    print(f"Total pairs: {len(pairs)}; already rated: {len(done)}; remaining: {len(pairs)-len(done)}")
    print("Quality scale: 1=generic, 2=vague, 3=basic, 4=specific+empathetic, 5=excellent")
    print("Read review first, then both responses, then rate each on its own merits.\n")

    try:
        for i, p in enumerate(pairs, 1):
            ri = p["review_index"]
            if ri in done:
                continue
            print("=" * 80)
            print(f"[{i}/{len(pairs)}]  pair_id={ri}\n")
            print(f"USER REVIEW:\n  {p['review_text']}\n")
            print(f"--- Response A ---\n{p['A_response']}\n")
            print(f"--- Response B ---\n{p['B_response']}\n")

            try:
                qa = ask_int("  Response A quality (1-5): ", 1, 5)
                ha = ask_yn("  Response A helpful? (Y/N): ")
                qb = ask_int("  Response B quality (1-5): ", 1, 5)
                hb = ask_yn("  Response B helpful? (Y/N): ")
                pref = ask_choice("  Which is better overall?", ["A", "B", "EQUAL"])
            except (KeyboardInterrupt, EOFError):
                print("\n\n[saved progress; rerun to resume]")
                save(done)
                return

            done[ri] = {
                "review_index": ri,
                "A_quality": qa, "A_helpful": ha,
                "B_quality": qb, "B_helpful": hb,
                "preferred": pref,
            }
            save(done)
            print("  saved.\n")

        print("=" * 80)
        print(f"All {len(pairs)} pairs rated. Output saved to: {OUT}")
        print("Send ratings.json back to the person who shared this with you.")

    except (KeyboardInterrupt, EOFError):
        save(done)
        print("\n[progress saved; rerun to resume]")


if __name__ == "__main__":
    main()
