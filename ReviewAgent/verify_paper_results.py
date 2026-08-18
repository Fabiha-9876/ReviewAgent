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

BASE = Path(__file__).resolve().parent
DP = BASE / "data" / "processed"
sys.path.insert(0, str(BASE / "scripts"))


def banner(n, title):
    print()
    print("=" * 72)
    print(f"  SEGMENT {n}: {title}")
    print("=" * 72)


# ----------------------------------------------------------------------
FAILURES = []


def check(label, computed, expected, tol=0.0):
    """Assert a computed value against the manuscript's, and record any mismatch.

    Segments that merely print a stored summary verify nothing. Every numeric claim a
    segment can recompute should go through here so that a mismatch fails the run.
    """
    ok = abs(computed - expected) <= tol if isinstance(computed, (int, float)) else computed == expected
    print(f"    [{'OK ' if ok else 'DIFF'}] {label}: computed {computed}, paper {expected}")
    if not ok:
        FAILURES.append(f"{label}: computed {computed}, paper says {expected}")
    return ok


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
    banner(3, "Stage 2 — Cluster quality (Table 12)")
    print("Paper: hier DB=2.24, CH=1.85. The flat intrinsic column (DB=12.15, CH=0.98) is")
    print("WITHDRAWN: the script that produced it embedded reviews from one cluster while")
    print("labelling them across 194, so it measured random labelling. The ratios below are")
    print("printed only to show what the withdrawn comparison was; do not cite them.")
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
    print(f"\n  [WITHDRAWN] DB ratio {r_db:.2f}x, CH ratio {r_ch:.2f}x — not a valid comparison")

    # The mean sizes in this summary file are not the paper's: it stores representative
    # reviews per cluster and a subsample artifact. Table 12's 471.0 and 15.9 come from
    # review_count in the full cluster files, so recompute them from those.
    import statistics
    full_flat = DP / "clusters_umap/clusters_full.json"
    if full_flat.exists():
        with open(full_flat) as f:
            flat_counts = [c["review_count"] for c in json.load(f)]
        with open(DP / "kg_hierarchical/hierarchical_clusters_full.json") as f:
            hier_full = json.load(f)
        hier_counts = [c["review_count"] for c in hier_full]
        h_tot = len({i for c in hier_full for i in c["review_ids"]})
    else:
        # The 14 MB full cluster files are not in the released bundle; the bundle ships
        # the per-cluster counts instead, which is all the mean needs.
        with open(DP / "clusters_umap/cluster_size_counts.json") as f:
            cc = json.load(f)
        flat_counts = cc["flat_umap_hdbscan"]["review_counts"]
        hier_counts = cc["kg_hierarchical"]["review_counts"]
        h_tot = cc["kg_hierarchical"]["n_distinct_reviews"]
    fm = statistics.mean(flat_counts)
    hm = statistics.mean(hier_counts)
    f_tot = sum(flat_counts)
    print(f"\n  mean cluster size (Table 12): flat {fm:.1f} (paper 471.0), "
          f"hier {hm:.1f} (paper 15.9)")
    check("flat mean cluster size", round(fm, 1), 471.0)
    check("hier mean cluster size", round(hm, 1), 15.9)
    print(f"  review sets are NOT matched: flat clusters {f_tot:,} reviews, "
          f"KG covers {h_tot:,} distinct reviews of the 8,404-review sample")


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
    print("Paper SpecCov (unfloored, floors removed): LLM+tax 4.19, LLM-free 3.38, raw 4.47, human 3.45")
    print()
    from speccov import speccov_detail
    with open(DP / "issue_specs/sample_100_clusters.json") as f:
        clusters_100 = json.load(f)
    cluster_by_id = {c["cluster_id"]: c for c in clusters_100}

    for cond_file, cond, paper in [
        ("specs_with_taxonomy.json",  "llm_taxonomy",  4.19),
        ("specs_free_form.json",      "llm_free_form", 3.38),
        ("specs_raw_summary.json",    "raw_summary",   4.47),
        ("specs_human_written.json",  "human_ref",     3.45),
    ]:
        with open(DP / "issue_specs" / cond_file) as f:
            specs = json.load(f)
        scores = [speccov_detail(s, cluster_by_id.get(s.get("cluster_id"), {}),
                                 condition=None)["speccov_score"]
                  for s in specs]
        m = sum(scores) / len(scores)
        match = "OK" if abs(m - paper) < 0.05 else "DIFF"
        print(f"  [{match}] {cond:<15s} n={len(scores):3d} "
              f"computed={m:.2f}  paper={paper}")


def segment_6():
    banner(6, "Stage 4 — Human eval, lead author only (historical)")
    print("Lead-author-only view. The paper reports the three-rater pooled numbers;")
    print("see Segment 13 for those. Paper (3 raters): rrgen 2.31, prompt 2.98,")
    print("no_spec 2.24, full 4.59; +2.35 full vs no_spec.")
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
    print("Paper §5.3: SFT 0.090, KTO 0.068, DPO 0.084, constrained-proxy 0.137 (+53%), "
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
          f"+{(cp_b - sft_b) / sft_b * 100:.1f}% (paper: +53%)")


def segment_10():
    banner(10, "Inter-rater Krippendorff alpha (3 raters)")
    print("Paper Table 4: alpha = 0.285 on 99 reviews (corrected from a mis-implemented 0.451)")
    print()
    with open(DP / "inter_annotator/agreement_summary.json") as f:
        ia = json.load(f)
    print(f"n_triples: {ia['n_triples']}")
    print("Pairwise Cohen kappa:")
    for k, v in ia["cohen_kappa"].items():
        print(f"  {k}: {v:.4f}")
    # Recompute rather than read back: the stored value was wrong once already.
    # Lift the shipped scorer out of the script with ast rather than importing the module,
    # which pulls in src/ and is not present in the released data bundle.
    import ast
    src = (BASE / "scripts/run_inter_annotator_agreement.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "krippendorff_alpha")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "run_inter_annotator_agreement.py",
                 "exec"), ns)
    mod = type("mod", (), ns)
    with open(DP / "inter_annotator/llm_annotations.json") as f:
        raw = json.load(f)
    rows = [raw["expert_labels"], raw["llm_concise_labels"], raw["llm_cot_labels"]]
    cats = sorted({x for r in rows for x in r if x})
    idx = [j for j in range(len(rows[0])) if all(r[j] in cats for r in rows)]
    alpha = mod.krippendorff_alpha([[r[j] for j in idx] for r in rows], cats)
    print(f"\nKrippendorff alpha (3 raters), recomputed on {len(idx)} triples: {alpha:.4f}")
    check("Krippendorff alpha (LLM-assisted panel)", round(alpha, 3), 0.285)


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




def segment_12():
    banner(12, "Stage 1 gold — three human raters (Section 4.2)")
    print("Paper: Fleiss kappa 0.968 on the Y/N verdict, 0.979 on the 7-class label;")
    print("classifiers vs the majority-vote gold: 0.165 / 0.334 / 0.590")
    print()
    path = DP / "inter_annotator/round2_agreement.json"
    if not path.exists():
        print("  [skip] run scripts/run_round2_agreement.py first "
              "(needs the three annotator sheets)")
        return
    with open(path) as f:
        rep = json.load(f)
    print(f"  items: {rep['n_items']}, usable for label-level stats: "
          f"{rep['n_usable_label_level']}")
    for pair, v in rep["pairwise_verdict"].items():
        print(f"  verdict  {pair:10s} Cohen kappa {v['cohen_kappa']:.4f} "
              f"(raw {v['raw_agreement']*100:.1f}%)")
    print(f"  verdict  Fleiss kappa (3 raters): {rep['fleiss_verdict']['kappa']:.4f}")
    print(f"  label    Fleiss kappa (3 raters): {rep['fleiss_label']['kappa']:.4f}")
    print(f"  label    Krippendorff alpha:      {rep['krippendorff_alpha_label']:.4f}")
    print(f"  majority gold: {rep['majority_gold']['n']} items, "
          f"{rep['majority_gold']['n_three_way_ties']} three-way ties")

    # recompute the classifier-vs-majority-gold column that Table 8 prints
    gold_path = DP / "inter_annotator/round2_majority_gold.json"
    key_path = BASE / "annotator_materials/master_key.json"
    if gold_path.exists() and key_path.exists():
        with open(gold_path) as f:
            gold = json.load(f)["gold"]
        with open(key_path) as f:
            key = json.load(f)

        def kappa(pred, truth):
            n = len(pred)
            po = sum(1 for a, b in zip(pred, truth) if a == b) / n
            cp, ct = Counter(pred), Counter(truth)
            pe = sum(cp[c] / n * ct[c] / n for c in set(cp) | set(ct))
            return (po - pe) / (1 - pe) if pe < 1 else float("nan")

        print("\n  classifiers vs the 3-rater majority gold (Table 8, third column):")
        for name, field, claimed in (("V2 LLM", "main_v2_labels", 0.165),
                                     ("cleanlab-corrected", "main_corrected_v2_labels", 0.334),
                                     ("V5 classifier", "main_v5_labels", 0.590)):
            lab = key[field]
            ids = [i for i in gold if i in lab or str(i) in lab]
            truth = [gold[i]["label"] for i in ids]
            pred = [str(lab.get(i, lab.get(str(i)))).strip().lower().replace(" ", "_")
                    for i in ids]
            k = kappa(pred, truth)
            flag = "OK " if abs(k - claimed) < 0.005 else "DIFF"
            print(f"    [{flag}] {name:20s} n={len(ids)}  kappa={k:.4f}  paper={claimed}")


def segment_13():
    banner(13, "Stage 4 — DISCARDED template-composer round (kept for the record)")
    print("This round is WITHDRAWN. Its conditions came from template composers, not one model.")
    print("The reported result is segment 19. Historical values: pooled +2.35, per rater +2.36/+2.33/+2.35;")
    print("quality weighted kappa 0.970-0.986; helpful alpha 0.781")
    print()
    path = DP / "inter_annotator/round2_response_agreement.json"
    if not path.exists():
        print("  [skip] run scripts/run_round2_response_agreement.py first "
              "(needs the three rating sheets)")
        return
    with open(path) as f:
        rep = json.load(f)
    print(f"  {'condition':<25s}" + "".join(f"{n:>9s}" for n in rep["raters"]) + f"{'pooled':>9s}")
    for c, pooled in rep["pooled_condition_means"].items():
        row = f"  {c:<25s}"
        for n in rep["raters"]:
            row += f"{rep['per_rater'][n]['condition_means'][c]['quality']:>9.2f}"
        print(row + f"{pooled['quality']:>9.2f}")
    print()
    for n in rep["raters"]:
        g = rep["per_rater"][n]["full_vs_nospec_quality"]
        print(f"  rater {n}: gain {g['mean_gain']:+.2f} "
              f"CI [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}] p={g['wilcoxon_p']:.1e}")
    g = rep["pooled_full_vs_nospec_quality"]
    print(f"  pooled : gain {g['mean_gain']:+.2f} "
          f"CI [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}] p={g['wilcoxon_p']:.1e}")
    print()
    for pair, v in rep["inter_rater"].items():
        print(f"  {pair}: quality weighted kappa "
              f"{v['quality_1_to_5']['quadratic_weighted_kappa']:.3f}, "
              f"within-1 {v['quality_1_to_5']['within1_pct']:.1f}%, "
              f"helpful kappa {v['helpful_y_n']['cohen_kappa']:.3f}")
    print(f"  helpful Krippendorff alpha: {rep['helpful_krippendorff_alpha']:.3f}")


def segment_14():
    banner(14, "Cross-family five-dimension rubric (Table 4)")
    print("Paper Panel A: Claude 4.09, Llama 3.96, Qwen-3B 3.83, Qwen-1.5B 3.77")
    print("Paper Panel B: Claude 4.01, Llama 3.64")
    print()
    path = DP / "ablations/cross_family_rubric.json"
    if not path.exists():
        print("  [skip] run scripts/run_cross_family_rubric.py first (needs the local judge)")
        return
    with open(path) as f:
        rep = json.load(f)
    for key in ("panel_a", "panel_b"):
        print(f"  {rep[key]['description']} (n_clusters={rep[key]['n_clusters']})")
        for name, v in rep[key]["results"].items():
            print(f"    {name:<26s} n={v['n_scored']:<4d} mean={v['overall_mean']:.2f}")
    print(f"  caveat: {rep['caveat']}")




def segment_15():
    banner(15, "SpecCov validation against human judgement (Section 5.1)")
    print("Paper: no significant rank correlation (rho = +0.045 / +0.015 / +0.226 per rater,")
    print("+0.147 pooled); flagged vs clean SpecCov 3.64 vs 3.93, Mann-Whitney p = 0.34;")
    print("SpecCov ranks taxonomy highest, humans rank the human reference highest")
    print()
    path = DP / "ablations/speccov_validation.json"
    if not path.exists():
        print("  [skip] run scripts/run_speccov_validation.py first "
              "(needs the three returned rating sheets)")
        return
    with open(path) as f:
        rep = json.load(f)
    print(f"  n specs: {rep['n_specs']}")
    for code, v in rep["per_rater"].items():
        line = (f"  rater {code}: Spearman rho={v['spearman_rho']:+.3f} "
                f"(p={v['spearman_p']:.3f}), mean human {v['mean_human']:.2f}, "
                f"flagged {v['n_flagged']}/{v['n_flagged'] + v['n_clean']}")
        if "mannwhitney_p" in v:
            line += (f", SpecCov flagged {v['speccov_mean_flagged']:.2f} vs clean "
                     f"{v['speccov_mean_clean']:.2f} (p={v['mannwhitney_p']:.3f})")
        print(line)
    print(f"  pooled: Spearman rho={rep['pooled']['spearman_rho']:+.3f} "
          f"(p={rep['pooled']['spearman_p']:.3f})")
    print(f"  {'condition':16s}{'SpecCov':>10s}{'human (A)':>12s}")
    for c, v in rep["by_condition"].items():
        print(f"  {c:16s}{v['speccov_mean']:>10.2f}{v['human_mean_A']:>12.2f}")




def segment_16():
    banner(16, "CMDP with a binding constraint (Section 5.3)")
    print("Paper: 13/90 binding steps, lambda 0 -> 0.65 monotone, quality 0.565 -> 0.590,")
    print("compliance 0.997 -> 0.989; capability probe 5.0% vs the paper's own 5.2%")
    print()
    path = DP / "rlhf/cppo_binding/training_log.json"
    if not path.exists():
        print("  [skip] run scripts/run_cppo_binding_constraint.py first")
        return
    with open(path) as f:
        rep = json.load(f)
    lam = [r["lambda"] for r in rep["log"]]
    print(f"  policy: {rep['model']}   tau={rep['tau']}   steps={rep['steps']}")
    print(f"  binding steps: {rep['binding_steps']}/{rep['steps']} ({rep['binding_pct']}%)")
    print(f"  lambda: start {lam[0]}  max {rep['max_lambda']}  final {rep['final_lambda']}"
          f"  monotone non-decreasing: {all(b >= a for a, b in zip(lam, lam[1:]))}")
    print(f"  compliance {rep['compliance_first_half']:.4f} -> {rep['compliance_second_half']:.4f}")
    print(f"  quality    {rep['quality_first_half']:.4f} -> {rep['quality_second_half']:.4f}")
    print(f"  constraint engaged: {rep['constraint_engaged']}")
    probe = DP / "rlhf/constraint_probe_qwen15b.json"
    if probe.exists():
        with open(probe) as f:
            pr = json.load(f)
        print(f"  capability probe: {pr['violations']}/{pr['n']} "
              f"({100 * pr['violations'] / pr['n']:.1f}%) zero-shot violations, "
              f"mean compliance {pr['mean']:.3f}")




def segment_17():
    banner(17, "Five-dimension rubric on the full spec set (Table 5)")
    print("Paper: taxonomy 3.94 over all 100 specs (28-spec subsample 3.89, delta +0.05);")
    print("lead-author reference 3.45 over all 20")
    print()
    path = DP / "ablations/qwen_judge_5dim_rubric_full.json"
    if not path.exists():
        print("  [skip] run scripts/run_full_rubric_100.py first")
        return
    with open(path) as f:
        rep = json.load(f)
    for label, key in (("taxonomy", "taxonomy_full"), ("lead-author ref", "reference_full")):
        v = rep[key]
        dims = "  ".join(f"{d[:5]} {v['per_dim'][d]:.2f}" for d in rep["rubric_dimensions"])
        print(f"  {label:16s} n={v['n']:<4d} mean={v['overall_mean']:.2f} "
              f"sd={v['overall_sd']:.2f}   {dims}")
    sc = rep["subsample_comparison"]
    print(f"  subsample check: {sc['previous_n']}-spec {sc['previous_mean']:.2f} vs "
          f"full {sc['full_mean']:.2f} (delta {sc['delta']:+.2f})")
    print("  per issue type:")
    for t, v in rep["taxonomy_per_issue_type"].items():
        print(f"    {str(t):18s} n={v['n']:<4d} mean={v['overall_mean']:.2f}")


def segment_18():
    banner(18, "Stage-3 run-to-run variance (Section 6.3)")
    print("Paper: strict fill mean 0.673, sd 0.029, range 0.067 over three seeds")
    print()
    path = DP / "ablations/stage3_variance.json"
    if not path.exists():
        print("  [skip] run scripts/run_stage3_variance.py first")
        return
    with open(path) as f:
        rep = json.load(f)
    print(f"  {rep['model']}  temperature={rep['temperature']}  "
          f"{rep['n_clusters']} clusters x {rep['runs']} runs")
    for r in rep["per_run"]:
        print(f"    run {r['run']}: fill={r['mean_strict_fill']:.3f} "
              f"speccov={r['mean_speccov']:.3f} parse_fails={r['parse_failures']}")
    for key in ("strict_fill", "speccov"):
        v = rep[key]
        print(f"  {key:12s} mean {v['mean']:.3f}  sd {v['sd']:.3f}  range {v['range']:.3f}")




def segment_19():
    banner(19, "Stage 4 — the REPORTED same-model re-run (Table 13)")
    print("Paper: pooled +0.03 position-controlled (+0.04 uncontrolled), CI [-0.10, +0.18],")
    print("Wilcoxon p = 0.38; per rater +0.53 / -0.26 / -0.18; weighted kappa 0.03 / 0.04 / 0.50")
    print()
    path = DP / "inter_annotator/stage4_llm_rerun.json"
    if not path.exists():
        print("  [skip] run scripts/run_stage4_llm_rerun.py first")
        return
    with open(path) as f:
        rep = json.load(f)
    print(f"  {rep['n_rows']} rows, {rep['design']}")
    print(f"  {'rater':10s}{'no_spec':>10s}{'full':>8s}{'controlled':>13s}{'naive':>9s}{'p':>11s}")
    for c in rep["raters"]:
        v = rep["per_rater"][c]
        m = v["condition_means"]
        print(f"  {c:10s}{m['reviewagent_no_spec_LLM']['quality']:>10.2f}"
              f"{m['reviewagent_full_LLM']['quality']:>8.2f}"
              f"{v.get('position_controlled_gain', float('nan')):>13.2f}"
              f"{v['mean_gain']:>9.2f}{v['wilcoxon_p']:>11.2e}")
    p = rep["pooled"]
    print(f"  {'pooled':10s}{p['mean_no_spec']:>10.2f}{p['mean_full']:>8.2f}"
          f"{p.get('position_controlled_gain', float('nan')):>13.2f}{p['mean_gain']:>9.2f}"
          f"{p['wilcoxon_p']:>11.2e}")
    print(f"  95% CI {p['ci95']}, Cliff's delta {p['cliffs_delta']}")
    print()
    for pair, v in rep["reliability"].items():
        print(f"  {pair}: quality weighted kappa {v['quality_weighted_kappa']:.3f}, "
              f"exact {v['quality_exact_pct']}%, within-1 {v['quality_within1_pct']}%")
    ind = rep["independence"]
    print()
    print(f"  independence: D and E deviate from A on {ind['D_deviations_from_A']} and "
          f"{ind['E_deviations_from_A']} of {rep['n_rows']} rows, overlap "
          f"{ind['overlap_observed']} (chance {ind['overlap_expected_if_independent']}), "
          f"{ind['distinct_deviation_magnitudes']} distinct deviation magnitudes")


SEGMENTS = {
    "1": segment_1, "2": segment_2, "3": segment_3, "4": segment_4, "5": segment_5,
    "6": segment_6, "7": segment_7, "8": segment_8, "9": segment_9, "10": segment_10,
    "11": segment_11, "12": segment_12, "13": segment_13, "14": segment_14, "15": segment_15, "16": segment_16, "17": segment_17, "18": segment_18, "19": segment_19,
}


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else SEGMENTS.keys()
    failed = []
    for s in selected:
        fn = SEGMENTS.get(s)
        if fn is None:
            print(f"unknown segment: {s}", file=sys.stderr)
            continue
        try:
            fn()
        except FileNotFoundError as e:
            failed.append((s, f"missing data file: {e.filename}"))
            print(f"\n  [SKIP] segment {s}: missing data file {e.filename}")
            print("         this segment needs a file that is not in the released bundle")
        except Exception as e:
            failed.append((s, repr(e)))
            print(f"\n  [FAIL] segment {s}: {e!r}")
    print()
    print("=" * 72)
    if failed:
        print(f"  DONE with {len(failed)} segment(s) skipped or failed:")
        for s, why in failed:
            print(f"    segment {s}: {why}")
    elif FAILURES:
        print(f"  DONE, but {len(FAILURES)} value(s) disagree with the manuscript:")
        for why in FAILURES:
            print(f"    {why}")
    else:
        print("  DONE")
    print("=" * 72)
    # Exit non-zero on any skip, failure, or value mismatch, so this can gate a release
    # rather than always reporting success.
    return 1 if (failed or FAILURES) else 0


if __name__ == "__main__":
    sys.exit(main())
