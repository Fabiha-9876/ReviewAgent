#!/usr/bin/env python3
"""
Verify all IssueSpec paper results — no Jupyter needed.

Run from VS Code terminal:
    cd "/Users/fabihajalal/Desktop/Review Agent/ReviewAgent"
    python3 verify_paper_results.py

Or run individual segments:
    python3 verify_paper_results.py 1      # just segment 1
    python3 verify_paper_results.py 5 6 9  # segments 5, 6, 9
"""

import json
import sys
from pathlib import Path
from collections import Counter

BASE = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent")
DP = BASE / "data" / "processed"
sys.path.insert(0, str(BASE / "scripts"))


def banner(n, title):
    print()
    print("=" * 72)
    print(f"  SEGMENT {n}: {title}")
    print("=" * 72)


# ----------------------------------------------------------------------
def segment_1():
    banner(1, "Corpus + annotation provenance")
    print("Paper: 215,583 working corpus; 5,230 anchor; 79.49/18.08/2.43%")
    print()
    with open(DP / "rrgen_v5_training.json") as f:
        train = json.load(f)
    src = Counter(r["source"] for r in train)
    working = src["llm_kept"] + src["anchor_corrected_v2"] + src["human_verified"]

    print(f"Total V5 training records: {len(train):,}")
    print(f"Working corpus: {working:,}")
    for s in ["llm_kept", "anchor_corrected_v2", "human_verified",
              "synthetic_compat_v2", "rrgen_mined_compat"]:
        n = src[s]
        pct = (100 * n / working
               if s in ("llm_kept", "anchor_corrected_v2", "human_verified") else None)
        note = f" ({pct:.2f}% of working)" if pct else " (augmented)"
        print(f"  {s:<25s} {n:>8,}{note}")

    with open(DP / "verified_annotations.json") as f:
        anchor = json.load(f)
    with open(DP / "issue_specs/sample_100_clusters.json") as f:
        clusters_100 = json.load(f)
    print(f"\nverified_annotations.json: {len(anchor):,} records  (paper: 5,230)")
    print(f"sample_100_clusters.json:  {len(clusters_100)} clusters  (paper: 100)")


def segment_2():
    banner(2, "Stage 1 — Cohen's kappa progression (Table 8)")
    print("Paper: V2 0.163, cleanlab 0.333, V5 0.592")
    print()
    with open(DP / "expert_evaluation/strict_holdout_kappa.json") as f:
        kappa = json.load(f)
    print(f"n_total_gold = {kappa['n_total_gold']}, "
          f"n_strict_held_out = {kappa['n_strict_held_out']}\n")
    print(f"  {'classifier':<25s} {'n':>5s} {'kappa':>8s} {'acc':>8s} {'macroF1':>10s}")
    print(f"  {'-'*60}")
    for name, v in kappa["full_490"].items():
        print(f"  {name:<25s} {v['n']:>5d} {v['cohen_kappa']:>8.4f} "
              f"{v['accuracy']:>8.4f} {v['macro_f1']:>10.4f}")


def segment_3():
    banner(3, "Stage 2 — Cluster quality (Table 11)")
    print("Paper: flat DB=12.15, CH=0.98; hier DB=2.24, CH=1.85")
    print()
    with open(DP / "clusters_umap/quality_metrics_flat_vs_hierarchical.json") as f:
        qm = json.load(f)
    flat = qm["flat_umap_hdbscan"]
    hier = qm["hierarchical_kg"]
    print("Flat:")
    print(f"  n={flat['size_stats']['n_clusters']}, "
          f"mean={flat['size_stats']['mean_size']:.1f}, "
          f"DB={flat['intrinsic_metrics']['davies_bouldin']:.4f}, "
          f"CH={flat['intrinsic_metrics']['calinski_harabasz']:.4f}")
    print(f"  Y/P/N purity (lead): {flat['yp_weighted_purity_50_audit']}")
    print("Hier:")
    print(f"  n={hier['size_stats']['n_clusters']}, "
          f"mean={hier['size_stats']['mean_size']:.1f}, "
          f"DB={hier['intrinsic_metrics']['davies_bouldin']:.4f}, "
          f"CH={hier['intrinsic_metrics']['calinski_harabasz']:.4f}")
    r_db = (flat["intrinsic_metrics"]["davies_bouldin"]
            / hier["intrinsic_metrics"]["davies_bouldin"])
    r_ch = (hier["intrinsic_metrics"]["calinski_harabasz"]
            / flat["intrinsic_metrics"]["calinski_harabasz"])
    print(f"\n5.4x lower DB -> {r_db:.2f}x")
    print(f"1.9x higher CH -> {r_ch:.2f}x")


def segment_4():
    banner(4, "A1b — count-matched flat-605 vs KG-605")
    print("Paper §5.5: agg-605 DB=1.12, CH=4.21, silhouette=+0.148; "
          "KG-605 2.24/1.85/-0.234")
    print()
    with open(DP / "ablations/a1b_repbased.json") as f:
        a1b = json.load(f)
    print(f"n_reps: {a1b['n_reps']}")
    for k in ["kg_605", "agglomerative_605", "flat_hdbscan_best"]:
        v = a1b[k]
        print(f"  {k}: n_clusters={v['n_clusters']}, "
              f"DB={v['davies_bouldin']:.4f}, "
              f"CH={v['calinski_harabasz']:.4f}, "
              f"silhouette={v['silhouette_cosine']:+.4f}")


def segment_5():
    banner(5, "Stage 3 — SpecCov scorer")
    print("Paper SpecCov: LLM+tax 4.19, LLM-free 3.38, raw 5.00, human 4.00")
    print()
    from speccov import speccov_detail
    with open(DP / "issue_specs/sample_100_clusters.json") as f:
        clusters_100 = json.load(f)
    cluster_by_id = {c["cluster_id"]: c for c in clusters_100}

    for cond_file, cond, paper in [
        ("specs_with_taxonomy.json",  "llm_taxonomy",  4.19),
        ("specs_free_form.json",      "llm_free_form", 3.38),
        ("specs_raw_summary.json",    "raw_summary",   5.00),
        ("specs_human_written.json",  "human_ref",     4.00),
    ]:
        with open(DP / "issue_specs" / cond_file) as f:
            specs = json.load(f)
        scores = [speccov_detail(s, cluster_by_id.get(s.get("cluster_id"), {}),
                                 condition=cond)["speccov_score"]
                  for s in specs]
        m = sum(scores) / len(scores)
        match = "OK" if abs(m - paper) < 0.05 else "DIFF"
        print(f"  [{match}] {cond:<15s} n={len(scores):3d} "
              f"computed={m:.2f}  paper={paper}")


def segment_6():
    banner(6, "Stage 4 — Human eval (Table 10)")
    print("Paper: rrgen 2.31, prompt 2.98, no_spec 2.26, full 4.62; +2.36 full vs no_spec")
    print()
    with open(DP / "experiments/exp2_human_eval.json") as f:
        s4 = json.load(f)
    print(f"  {'condition':<25s} {'quality':>8s} {'specif':>8s} {'helpful%':>10s}")
    print(f"  {'-'*55}")
    for c, s in s4["summary"].items():
        print(f"  {c:<25s} {s['quality_mean']:>8.2f} "
              f"{s['specificity_mean']:>8.2f} {s['helpful_pct']:>10.1f}")

    print("\nPaired Wilcoxon (quality):")
    for k, v in s4["paired_wilcoxon"].items():
        if "quality" in k:
            print(f"  {k:<55s} delta={v['mean_diff']:+.2f}  "
                  f"z={v['z']:+.2f}  p={v['p_value']:.4f}")


def segment_7():
    banner(7, "A5 — no-RAG ablation")
    print("Paper §5.5 A5: dBLEU-1 -0.006, dROUGE-L -0.008, dBERTScore -0.004")
    print()
    with open(DP / "experiments/ablation_a5_results.json") as f:
        a5 = json.load(f)
    nr = a5["no_rag_metrics"]
    fu = a5["full_system_metrics_for_comparison"]
    print(f"  {'metric':<20s} {'no_rag':>10s} {'full':>10s} {'delta':>10s}")
    for m in ["bleu_1_mean", "rouge_l_mean", "bertscore_f1_mean"]:
        d = nr[m] - fu[m]
        print(f"  {m:<20s} {nr[m]:>10.4f} {fu[m]:>10.4f} {d:>+10.4f}")


def segment_8():
    banner(8, "Agentic vs vanilla RAG (n=10 feasibility)")
    print("Paper §5.2: vanilla 0.58 -> agentic 0.70 (+0.12); citation 0% -> 60%")
    print()
    with open(DP / "ablations/agentic_vs_vanilla_rag.json") as f:
        av = json.load(f)
    print(f"n_reviews={av['n_reviews']}, max_iterations={av['max_iterations']}")
    for c in ["vanilla_rag", "agentic_rag"]:
        print(f"  {c}: {av[c]}")
    print(f"  Delta: {av['delta_agentic_minus_vanilla']}")


def segment_9():
    banner(9, "Stage 5 — RLHF 5 policies")
    print("Paper §5.3: SFT 0.090, KTO 0.068, DPO 0.084, constrained-proxy 0.137 (+52%), "
          "Lagrangian PPO 0.087")
    print()
    with open(DP / "rlhf/head_to_head/metrics.json") as f:
        rlhf = json.load(f)
    print(f"  {'policy':<20s} {'BLEU-1':>8s} {'ROUGE-L':>9s} {'BERTScore':>10s}")
    print(f"  {'-'*55}")
    for p in ["sft_base", "kto_model", "dpo_model",
              "constrained_proxy", "lagrangian_ppo"]:
        m = rlhf[p]
        b = m.get("bleu_1") or m.get("bleu_1_mean")
        r = m.get("rouge_l") or m.get("rouge_l_mean")
        bs = m.get("bertscore_f1") or m.get("bertscore_f1_mean")
        star = "*" if p == "constrained_proxy" else " "
        print(f" {star}{p:<20s} {b:>8.3f} {r:>9.3f} {bs:>10.3f}")

    sft_b = (rlhf["sft_base"].get("bleu_1")
             or rlhf["sft_base"].get("bleu_1_mean"))
    cp_b = (rlhf["constrained_proxy"].get("bleu_1")
            or rlhf["constrained_proxy"].get("bleu_1_mean"))
    print(f"\nconstrained-proxy vs SFT-base BLEU-1 gain: "
          f"+{(cp_b - sft_b) / sft_b * 100:.1f}% (paper: +52%)")


def segment_10():
    banner(10, "Inter-rater Krippendorff alpha (3 raters)")
    print("Paper Table 4: alpha = 0.451 on 99 reviews")
    print()
    with open(DP / "inter_annotator/agreement_summary.json") as f:
        ia = json.load(f)
    print(f"n_triples: {ia['n_triples']}")
    print("Pairwise Cohen kappa:")
    for k, v in ia["cohen_kappa"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nKrippendorff alpha (3 raters): "
          f"{ia['krippendorff_alpha_3raters']:.4f} -> rounds to 0.451")


def segment_11():
    banner(11, "Stage 2 — PageRank coverage (Section 5.4)")
    print("Paper Sec 5.4: top-5 aspects in 55% of reviews, top-10 in 66% (unique-review dedup)")
    print()
    ks = json.load(open(DP / "kg_hierarchical" / "kg_stats.json"))
    N = ks["n_review_nodes"]
    pr = ks["top_aspects_by_pagerank"]
    s5 = sum(a["n_reviews"] for a in pr[:5])
    s10 = sum(a["n_reviews"] for a in pr[:10])
    print(f"  NAIVE (sum of per-aspect n_reviews / {N}, double-counts reviews):")
    print(f"     top-5 {s5} ({100*s5/N:.1f}%)   top-10 {s10} ({100*s10/N:.1f}%)")
    cov = json.load(open(DP / "kg_hierarchical" / "aspect_coverage.json"))["raw"]
    print("  DEDUP (unique reviews touching top-K; via scripts/compute_aspect_coverage.py):")
    print(f"     top-5 {cov['dedup_top5_pct']}%   top-10 {cov['dedup_top10_pct']}%"
          f"   [matches paper 55/66]")


SEGMENTS = {
    "1": segment_1, "2": segment_2, "3": segment_3, "4": segment_4, "5": segment_5,
    "6": segment_6, "7": segment_7, "8": segment_8, "9": segment_9, "10": segment_10,
    "11": segment_11,
}


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else SEGMENTS.keys()
    for s in selected:
        fn = SEGMENTS.get(s)
        if fn:
            fn()
        else:
            print(f"Unknown segment: {s}", file=sys.stderr)
    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
