"""
Generate updated paper figures incorporating Cohen κ progression and
human evaluation results.

Adds 4 new figures to the existing 7:
    fig08_kappa_progression.png         Cohen κ vs expert (V2 → corrected → V5)
    fig09_human_eval.png                Human eval scores per condition (Exp 2)
    fig10_paired_wilcoxon.png           Quality difference + significance
    fig11_curation_outcome.png          Cluster purity 0.66 → 0.81 + verdict breakdown
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig08_kappa_progression():
    """Cohen κ progression vs expert."""
    rep = json.load(open("annotator_materials/gold_standard_results.json"))
    classifiers = ["V2 LLM\noriginal", "cleanlab\ncorrected", "V5\nclassifier"]
    kappas = [
        rep["classifier_vs_expert"]["V2_LLM_original"]["cohen_kappa"],
        rep["classifier_vs_expert"]["corrected_v2_cleanlab"]["cohen_kappa"],
        rep["classifier_vs_expert"]["V5_classifier"]["cohen_kappa"],
    ]
    accs = [
        rep["classifier_vs_expert"]["V2_LLM_original"]["accuracy"],
        rep["classifier_vs_expert"]["corrected_v2_cleanlab"]["accuracy"],
        rep["classifier_vs_expert"]["V5_classifier"]["accuracy"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # κ progression
    colors = ["#909090", "#3D7CB7", "#E07A5F"]
    bars = ax1.bar(classifiers, kappas, color=colors)
    for bar, val in zip(bars, kappas):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                 ha="center", fontsize=11, fontweight="bold")

    # Landis & Koch interpretation bands
    ax1.axhspan(0.0, 0.20, alpha=0.08, color='red',    label='slight (<0.20)')
    ax1.axhspan(0.20, 0.40, alpha=0.08, color='orange',label='fair (0.20-0.40)')
    ax1.axhspan(0.40, 0.60, alpha=0.08, color='gold',  label='moderate (0.40-0.60)')
    ax1.axhspan(0.60, 0.80, alpha=0.08, color='green', label='substantial (0.60-0.80)')

    ax1.set_ylim(0, 0.85)
    ax1.set_ylabel("Cohen's κ vs expert (n=490)", fontsize=11)
    ax1.set_title("Cohen κ progression through the correction pipeline", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Accuracy comparison
    bars2 = ax2.bar(classifiers, accs, color=colors)
    for bar, val in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val*100:.1f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.85)
    ax2.set_ylabel("Accuracy vs expert (n=490)", fontsize=11)
    ax2.set_title("Classification accuracy progression", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_kappa_progression.png", dpi=150)
    plt.close()
    print("  saved fig08_kappa_progression.png")


def fig09_human_eval():
    """Human evaluation scores per condition (Stage 4b)."""
    he = json.load(open("data/processed/experiments/exp2_human_eval.json"))["summary"]

    conds = ["rrgen_baseline", "prompt_baseline", "reviewagent_no_spec", "reviewagent_full"]
    short = ["RRGen\nbaseline", "Core\nbaseline", "ReviewAgent\nno-spec", "ReviewAgent\n+ spec (full)"]
    quality = [he[c]["quality_mean"] for c in conds]
    quality_std = [he[c]["quality_std"] for c in conds]
    specificity = [he[c]["specificity_mean"] for c in conds]
    helpful_pct = [he[c]["helpful_pct"] for c in conds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    x = np.arange(len(conds))
    width = 0.35
    colors_q = ["#909090", "#3D7CB7", "#F2CC8F", "#E07A5F"]

    # Left: quality + specificity (1-5 scale, with error bars)
    ax1.bar(x - width/2, quality, width, yerr=quality_std, label="Quality (1-5)",
            color=colors_q, edgecolor="black", linewidth=0.5, capsize=4)
    ax1.bar(x + width/2, specificity, width, label="Specificity (1-5)",
            color=colors_q, alpha=0.5, edgecolor="black", linewidth=0.5)
    for i, (q, s) in enumerate(zip(quality, specificity)):
        ax1.text(i - width/2, q + 0.15, f"{q:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=10)
    ax1.set_ylabel("Mean rating (1-5)", fontsize=11)
    ax1.set_ylim(0, 6)
    ax1.set_title("Human evaluation: quality + specificity (n=100 per condition)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Right: helpful%
    bars = ax2.bar(short, helpful_pct, color=colors_q, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, helpful_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.0f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Helpful (Y) %", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.set_title("Human eval: would this response help the user?", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_human_eval.png", dpi=150)
    plt.close()
    print("  saved fig09_human_eval.png")


def fig10_paired_wilcoxon():
    """Paired Wilcoxon — quality difference + p-values."""
    he = json.load(open("data/processed/experiments/exp2_human_eval.json"))
    pw = he["paired_wilcoxon"]

    pairs = [
        ("ReviewAgent_full vs no_spec", "reviewagent_full_vs_reviewagent_no_spec_quality"),
        ("ReviewAgent_full vs core",    "reviewagent_full_vs_prompt_baseline_quality"),
        ("ReviewAgent_full vs RRGen",   "reviewagent_full_vs_rrgen_baseline_quality"),
        ("no_spec vs core",             "reviewagent_no_spec_vs_prompt_baseline_quality"),
        ("no_spec vs RRGen",            "reviewagent_no_spec_vs_rrgen_baseline_quality"),
        ("core vs RRGen",               "prompt_baseline_vs_rrgen_baseline_quality"),
    ]
    labels = [p[0] for p in pairs]
    deltas = [pw[p[1]]["mean_diff"] for p in pairs]
    p_vals = [pw[p[1]]["p_value"] for p in pairs]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#E07A5F" if d > 0 else "#3D7CB7" for d in deltas]
    bars = ax.barh(labels, deltas, color=colors, edgecolor="black", linewidth=0.5)
    for bar, d, p in zip(bars, deltas, p_vals):
        sig = "***" if p and p < 0.001 else ("**" if p and p < 0.01 else ("*" if p and p < 0.05 else "n.s."))
        x_pos = bar.get_width() + 0.05 if d > 0 else bar.get_width() - 0.05
        ha = "left" if d > 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"Δ={d:+.2f}  {sig}", va="center", ha=ha, fontsize=10, fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Quality difference (paired Wilcoxon)", fontsize=11)
    ax.set_title("Pairwise comparisons (n=100 paired observations each)\n*** p<0.001  ** p<0.01  * p<0.05", fontsize=11)
    ax.set_xlim(-1.5, 3.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_paired_wilcoxon.png", dpi=150)
    plt.close()
    print("  saved fig10_paired_wilcoxon.png")


def fig11_curation_outcome():
    """Cluster purity: pre-curation vs post-curation + verdict pie."""
    base = json.load(open("data/processed/clusters_umap/cluster_validation_score.json"))
    cur = json.load(open("data/processed/clusters_umap/cluster_curation_outcome.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: purity progression
    stages = ["Initial\n(50-cluster\nvalidation)", "After\ncuration\n(100-cluster)"]
    purity = [base["overall_purity"], cur["curation_purity_score"]]
    colors = ["#909090", "#81B29A"]
    bars = ax1.bar(stages, purity, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, purity):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                 ha="center", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cluster purity (Y=1, P=0.5, N=0)", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Cluster purity before vs after lead-author curation", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # Right: verdict breakdown pie
    actions = cur["actions"]
    sizes = [actions.get("Keep", 0), actions.get("Rename", 0),
             actions.get("Merge", 0), actions.get("Split", 0)]
    pie_colors = ["#81B29A", "#3D7CB7", "#F2CC8F", "#E07A5F"]
    wedges, texts, autotexts = ax2.pie(sizes,
            labels=[f"Keep\n({sizes[0]})", f"Rename\n({sizes[1]})",
                    f"Merge\n({sizes[2]})", f"Split\n({sizes[3]})"],
            colors=pie_colors, autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 10})
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Curation verdicts (n=100 clusters)", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_curation_outcome.png", dpi=150)
    plt.close()
    print("  saved fig11_curation_outcome.png")


if __name__ == "__main__":
    print(f"Generating updated figures in {FIG_DIR}/")
    fig08_kappa_progression()
    fig09_human_eval()
    fig10_paired_wilcoxon()
    fig11_curation_outcome()
    print(f"\nDone. Total figures now in {FIG_DIR}/:")
    for f in sorted(FIG_DIR.glob("fig*.png")):
        print(f"  {f.name}")
