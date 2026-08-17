"""Regenerate the result figures used in the main text, from the saved result files.

Three figures are written into paper/IssueSpec/ :

  fig_kappa_progression.png   Stage-1 classifier recovery, now with the three-rater
                              majority-vote gold plotted next to the lead-author gold.
  fig_human_eval.png          Stage-4 four-condition human evaluation. Condition labels
                              follow the paper's IssueSpec naming, not the old
                              ReviewAgent naming.
  fig_cluster_purity.png      Y/P/N cluster-purity audit, read from the saved judge
                              output for both the flat baseline and the KG hierarchy.
                              The previous version of this figure carried hand-entered
                              counts that did not match the audit files.

Usage
    python3 paper/build_v2/generate_paper_figures_v3.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "paper/IssueSpec"
FLAT_AUDIT = REPO / "data/processed/clusters_umap/llm_judge_purity_audit_qwen.json"
KG_AUDIT = REPO / "data/processed/kg_hierarchical/llm_judge_purity_audit_qwen.json"
AGREEMENT = REPO / "data/processed/inter_annotator/round2_agreement.json"

GREY, BLUE, ORANGE = "#9CA3AF", "#3B82F6", "#D97706"
GREEN, AMBER, RED = "#10B981", "#FBBF24", "#EF4444"


def fig_kappa():
    stages = ["V2 LLM\noriginal", "cleanlab\ncorrected", "V5\nclassifier"]
    lead = [0.163, 0.333, 0.592]
    panel = [0.165, 0.334, 0.590]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    bands = [(0.00, 0.20, "#FDF2F8", "slight (<0.20)"),
             (0.20, 0.40, "#FEFCE8", "fair (0.20-0.40)"),
             (0.40, 0.60, "#FFF7ED", "moderate (0.40-0.60)"),
             (0.60, 0.80, "#ECFDF5", "substantial (0.60-0.80)")]
    for lo, hi, colour, label in bands:
        ax.axhspan(lo, hi, color=colour, label=label, zorder=0)

    x = np.arange(len(stages))
    w = 0.36
    b1 = ax.bar(x - w / 2, lead, w, color=[GREY, BLUE, ORANGE],
                edgecolor="#111827", linewidth=1.0, label="vs lead-author gold", zorder=3)
    b2 = ax.bar(x + w / 2, panel, w, color=[GREY, BLUE, ORANGE], alpha=0.55,
                hatch="//", edgecolor="#111827", linewidth=1.0,
                label="vs 3-rater majority gold", zorder=3)
    for bars, vals in ((b1, lead), (b2, panel)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Cohen's $\\kappa$ vs human gold", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.set_title("Stage-1 recovery is not an artifact of one annotator", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # right panel: three-rater agreement on the gold itself
    rep = json.loads(AGREEMENT.read_text())
    names = ["A vs D", "A vs E", "D vs E", "Fleiss\n(3 raters)"]
    pv = rep["pairwise_verdict"]
    vals = [pv["A_vs_D"]["cohen_kappa"], pv["A_vs_E"]["cohen_kappa"],
            pv["D_vs_E"]["cohen_kappa"], rep["fleiss_verdict"]["kappa"]]
    bars = ax2.bar(names, vals, color=[BLUE, BLUE, BLUE, GREEN],
                   edgecolor="#111827", linewidth=1.0)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.axhline(0.81, color="#1F2937", linestyle="--", linewidth=1.0,
                label="almost-perfect threshold (0.81)")
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("$\\kappa$ on the Y/N verdict", fontsize=10)
    ax2.set_title("Agreement among the three human raters (n=490)", fontsize=10.5)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT / "fig_kappa_progression.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote fig_kappa_progression.png")


def fig_human_eval():
    conds = ["rrgen\nbaseline", "prompt\nbaseline",
             "RAG,\nno IssueSpec", "RAG +\nIssueSpec"]
    # pooled over the three human raters, see round2_response_agreement.json
    quality = [2.31, 2.98, 2.24, 4.59]
    specificity = [2.31, 2.96, 2.27, 4.59]
    helpful = [25, 83, 36, 91]
    qual_err = [0.61, 0.71, 0.58, 0.34]
    spec_err = [0.66, 0.74, 0.61, 0.40]
    colours = [GREY, BLUE, AMBER, RED]

    fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    x = np.arange(len(conds))
    w = 0.36
    b1 = ax_q.bar(x - w / 2, quality, w, yerr=qual_err, capsize=4, color="#4B5563",
                  edgecolor="#111827", linewidth=1.0, label="Quality (1-5)")
    ax_q.bar(x + w / 2, specificity, w, yerr=spec_err, capsize=4, color="#D1D5DB",
             edgecolor="#111827", linewidth=1.0, label="Specificity (1-5)")
    for bar, v in zip(b1, quality):
        ax_q.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_q.annotate("", xy=(3 - w / 2, 4.59), xytext=(2 - w / 2, 2.24),
                  arrowprops=dict(arrowstyle="->", color="#DC2626", linewidth=1.6))
    ax_q.text(2.5, 3.6, "$+2.35$\n($p<0.001$)", ha="center", fontsize=9.5,
              color="#DC2626", fontweight="bold")
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(conds, fontsize=9)
    ax_q.set_ylabel("Mean rating (1-5)", fontsize=10)
    ax_q.set_ylim(0, 6)
    ax_q.set_title("Adding the IssueSpec is what moves quality (3 raters, n=100/condition)",
                   fontsize=10.5)
    ax_q.legend(loc="upper left", fontsize=9)
    ax_q.grid(axis="y", alpha=0.25)

    bars = ax_h.bar(conds, helpful, color=colours, edgecolor="#111827", linewidth=1.0)
    for bar, v in zip(bars, helpful):
        ax_h.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v}%",
                  ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_h.set_ylabel("Rated helpful (Y), %", fontsize=10)
    ax_h.set_ylim(0, 105)
    ax_h.set_title("Would this response help the user?", fontsize=10.5)
    ax_h.tick_params(axis="x", labelsize=9)
    ax_h.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT / "fig_human_eval.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote fig_human_eval.png")


def fig_cluster_purity():
    flat = json.loads(FLAT_AUDIT.read_text())
    kg = json.loads(KG_AUDIT.read_text())
    classes = ["performance", "bug_report", "feature_request",
               "usability", "compatibility"]

    def purity(entry):
        n = entry["Y"] + entry["P"] + entry["N"]
        return (entry["Y"] + 0.5 * entry["P"]) / n if n else float("nan")

    fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    x = np.arange(len(classes))
    w = 0.38
    flat_p = [purity(flat["per_class"][c]) for c in classes]
    kg_p = [purity(kg["per_class"][c]) for c in classes]
    b1 = ax_p.bar(x - w / 2, flat_p, w, color=BLUE, edgecolor="#111827",
                  linewidth=1.0, label=f"flat-194 (overall {flat['weighted_purity']:.3f})")
    b2 = ax_p.bar(x + w / 2, kg_p, w, color=ORANGE, edgecolor="#111827",
                  linewidth=1.0, label=f"KG-605 (overall {kg['weighted_purity']:.3f})")
    for bars, vals in ((b1, flat_p), (b2, kg_p)):
        for bar, v in zip(bars, vals):
            ax_p.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(classes, rotation=18, fontsize=9)
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("Weighted purity (Y=1, P=0.5, N=0)", fontsize=10)
    ax_p.set_title("Sub-cluster purity drops at the KG's finer granularity", fontsize=10.5)
    ax_p.legend(loc="upper right", fontsize=8.5)
    ax_p.grid(axis="y", alpha=0.25)

    labels, ys, ps, ns = [], [], [], []
    for src, tag in ((flat, "flat"), (kg, "KG")):
        c = src["overall_counts"]
        labels.append(f"{tag}\n(n={c['Y'] + c['P'] + c['N']})")
        ys.append(c["Y"]); ps.append(c["P"]); ns.append(c["N"])
    ax_b.bar(labels, ys, color=GREEN, edgecolor="#111827", linewidth=0.8,
             label="Y (all reps share the sub-theme)")
    ax_b.bar(labels, ps, bottom=ys, color=AMBER, edgecolor="#111827", linewidth=0.8,
             label="P (partial)")
    ax_b.bar(labels, ns, bottom=[y + p for y, p in zip(ys, ps)], color=RED,
             edgecolor="#111827", linewidth=0.8, label="N (incoherent)")
    for i, (y, p, n) in enumerate(zip(ys, ps, ns)):
        for base, v, txt in ((0, y, y), (y, p, p), (y + p, n, n)):
            if v:
                ax_b.text(i, base + v / 2, str(txt), ha="center", va="center",
                          fontsize=10, fontweight="bold", color="#111827")
    ax_b.set_ylabel("Number of audited clusters", fontsize=10)
    ax_b.set_title("Verdict breakdown, judge audit", fontsize=10.5)
    ax_b.legend(loc="upper left", fontsize=8.5)
    ax_b.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT / "fig_cluster_purity.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote fig_cluster_purity.png")
    return flat, kg


def main():
    print(f"Writing figures into {OUT}")
    fig_kappa()
    fig_human_eval()
    flat, kg = fig_cluster_purity()
    print("\nPurity audit numbers now plotted (from the saved judge output):")
    for tag, src in (("flat-194", flat), ("KG-605", kg)):
        c = src["overall_counts"]
        print(f"  {tag}: Y={c['Y']} P={c['P']} N={c['N']} "
              f"weighted purity={src['weighted_purity']}")


if __name__ == "__main__":
    main()
