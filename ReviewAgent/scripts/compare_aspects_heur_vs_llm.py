"""
Cross-validate heuristic aspects against local-LLM aspects on the same sample.

For each review where both methods produced aspects, compute:
  - exact overlap        (set intersection on normalized strings)
  - substring overlap    (aspect from one set appears inside any aspect from the other)
  - precision/recall/F1  treating LLM as the gold-standard

Output:
    data/processed/aspect_comparison/comparison.json
    data/processed/aspect_comparison/per_review.csv
    Console summary with paper-grade numbers.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


def normalize(asp: str) -> str:
    a = asp.lower().strip()
    a = re.sub(r"^(the|a|an|my|your|this|that)\s+", "", a)
    a = re.sub(r"\s+", " ", a)
    return a.strip(" .!?,")


def substring_match(a: str, others: set[str]) -> bool:
    """True if `a` is contained in any of `others` or vice versa (>=3 chars)."""
    if a in others:
        return True
    if len(a) < 3:
        return False
    for b in others:
        if len(b) < 3:
            continue
        if a in b or b in a:
            return True
    return False


def main():
    out_dir = Path("data/processed/aspect_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading heuristic aspects (215K)")
    with open("data/processed/aspects_heuristic/aspects_per_review.json") as f:
        heur = json.load(f)  # keys are str(idx)
    print(f"  {len(heur):,} reviews with heuristic aspects")

    print("Loading LLM aspects (sample)")
    with open("data/processed/aspects_local_llm/aspects_per_review.json") as f:
        llm = json.load(f)  # keys are str(idx)
    print(f"  {len(llm):,} reviews with LLM aspects")

    # Find reviews present in both
    common_keys = sorted(set(heur.keys()) & set(llm.keys()), key=int)
    print(f"  reviews in both: {len(common_keys):,}")

    # Per-review metrics
    rows = []
    p_sum = r_sum = f1_sum = 0.0
    p_sub_sum = r_sub_sum = f1_sub_sum = 0.0
    n_with_overlap = 0
    n_zero_overlap = 0

    aspect_pairs = Counter()  # (heur_aspect, llm_aspect) co-occurrences

    for k in common_keys:
        h_set = {normalize(a) for a in heur[k] if a.strip()}
        l_set = {normalize(a) for a in llm[k] if a.strip()}
        if not h_set or not l_set:
            continue

        # Exact overlap
        inter = h_set & l_set
        precision = len(inter) / len(h_set) if h_set else 0
        recall    = len(inter) / len(l_set) if l_set else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        # Substring-tolerant overlap
        h_match_sub = sum(1 for a in h_set if substring_match(a, l_set))
        l_match_sub = sum(1 for a in l_set if substring_match(a, h_set))
        precision_sub = h_match_sub / len(h_set)
        recall_sub    = l_match_sub / len(l_set)
        f1_sub        = 2 * precision_sub * recall_sub / (precision_sub + recall_sub) if (precision_sub + recall_sub) else 0

        if inter:
            n_with_overlap += 1
        else:
            n_zero_overlap += 1

        # Aspect pairs (top heur aspect ↔ top llm aspect)
        for ha in h_set:
            for la in l_set:
                if ha != la and (ha in la or la in ha):
                    aspect_pairs[(ha, la)] += 1

        p_sum += precision
        r_sum += recall
        f1_sum += f1
        p_sub_sum += precision_sub
        r_sub_sum += recall_sub
        f1_sub_sum += f1_sub

        rows.append({
            "idx": k,
            "n_heur": len(h_set),
            "n_llm": len(l_set),
            "exact_overlap": len(inter),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "precision_sub": round(precision_sub, 3),
            "recall_sub": round(recall_sub, 3),
            "f1_sub": round(f1_sub, 3),
            "heur_aspects": "|".join(sorted(h_set)),
            "llm_aspects": "|".join(sorted(l_set)),
        })

    n = len(rows)
    summary = {
        "n_reviews_compared": n,
        "n_with_any_overlap": n_with_overlap,
        "n_zero_overlap":     n_zero_overlap,
        "exact": {
            "macro_precision": round(p_sum / n, 4) if n else 0,
            "macro_recall":    round(r_sum / n, 4) if n else 0,
            "macro_f1":        round(f1_sum / n, 4) if n else 0,
        },
        "substring_tolerant": {
            "macro_precision": round(p_sub_sum / n, 4) if n else 0,
            "macro_recall":    round(r_sub_sum / n, 4) if n else 0,
            "macro_f1":        round(f1_sub_sum / n, 4) if n else 0,
        },
        "top_aspect_pairs_substring": [
            {"heur": h, "llm": l, "count": c}
            for (h, l), c in aspect_pairs.most_common(30)
        ],
    }

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_review.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Console summary
    print("\n" + "=" * 70)
    print("HEURISTIC vs LOCAL-LLM ASPECT COMPARISON")
    print("=" * 70)
    print(f"Reviews compared: {n:,}")
    print(f"  with at least one exact overlap: {n_with_overlap:,}  ({100*n_with_overlap/n:.1f}%)")
    print(f"  zero exact overlap:              {n_zero_overlap:,}")
    print()
    print("Treating LLM as gold-standard:")
    print(f"  EXACT match  — P={summary['exact']['macro_precision']:.3f}  "
          f"R={summary['exact']['macro_recall']:.3f}  F1={summary['exact']['macro_f1']:.3f}")
    print(f"  SUBSTRING    — P={summary['substring_tolerant']['macro_precision']:.3f}  "
          f"R={summary['substring_tolerant']['macro_recall']:.3f}  F1={summary['substring_tolerant']['macro_f1']:.3f}")
    print()
    print("Top 15 substring-aligned aspect pairs (heuristic → LLM):")
    for pair in summary["top_aspect_pairs_substring"][:15]:
        print(f"  {pair['count']:>4}  {pair['heur']:30s}  →  {pair['llm']}")
    print(f"\nOutputs: {out_dir}/")


if __name__ == "__main__":
    main()
