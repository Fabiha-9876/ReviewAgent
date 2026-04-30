"""
Generate paper-grade figures from existing artifacts.

Reads eval_metrics.json, correction_stats.json, cluster_validation_score.json,
relabel_stats.json, and produces matplotlib figures (PNG) suitable for a paper.

Output directory: figures/

Figures produced:
    fig01_classifier_progression.png      V1→V5 macro F1 + compat F1 trajectory
    fig02_class_distribution.png          V2 LLM vs corrected_v2 vs V5 (stacked bar)
    fig03_correction_volume.png           V1 vs V2 cleanlab corrections
    fig04_v5_agreement_matrix.png         V2 / corrected / V5 three-way agreement
    fig05_cluster_purity.png              per-class cluster purity bar chart
    fig06_cluster_size_dist.png           cluster size histogram per class
    fig07_per_class_f1.png                V3 vs V4 vs V5 per-class F1 (own-test)
"""

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["bug_report", "feature_request", "performance", "usability",
          "compatibility", "praise", "other"]

# Brand-ish color palette (consistent across figures)
COLORS = {
    "bug_report": "#E07A5F",
    "feature_request": "#3D7CB7",
    "performance": "#F2CC8F",
    "usability": "#81B29A",
    "compatibility": "#9B59B6",
    "praise": "#F4A261",
    "other": "#909090",
}


def fig01_classifier_progression():
    """V1→V5 classifier progression — macro F1 and compat F1."""
    versions = ["V1", "V2", "V3", "V4", "V5"]
    macro_f1 = [0.799, 0.856, 0.808, 0.711, 0.813]
    compat_f1 = [1.00, 0.47, 0.667, 0.00, 0.74]
    compat_support = [10, 13, 4, 1, 31]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(versions))

    bars1 = ax1.bar(x - 0.2, macro_f1, 0.4, label="macro F1 (test)", color="#3D7CB7")
    bars2 = ax1.bar(x + 0.2, compat_f1, 0.4, label="compatibility F1", color="#9B59B6")

    for bar, val in zip(bars1, macro_f1):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                 ha="center", fontsize=9)
    for bar, val, sup in zip(bars2, compat_f1, compat_support):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.2f}\n(n={sup})", ha="center", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, fontsize=11)
    ax1.set_ylabel("F1 score", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Classifier progression: macro F1 vs compatibility F1\n"
                  "(each model on its own held-out test set)", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_classifier_progression.png", dpi=150)
    plt.close()
    print("  saved fig01_classifier_progression.png")


def fig02_class_distribution():
    """V2 LLM vs corrected_v2 vs V5 — 7-class distribution side-by-side."""
    s = json.load(open("data/processed/rrgen_v5_relabeled/relabel_stats.json"))
    dists = s["label_distributions"]

    x = np.arange(len(LABELS))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))
    v2  = [dists["v2_llm"].get(l, 0) for l in LABELS]
    cor = [dists["corrected_v2"].get(l, 0) for l in LABELS]
    v5  = [dists["v5"].get(l, 0) for l in LABELS]

    ax.bar(x - width, v2,  width, label="V2 LLM original",  color="#909090")
    ax.bar(x,         cor, width, label="cleanlab corrected", color="#3D7CB7")
    ax.bar(x + width, v5,  width, label="V5 prediction",     color="#E07A5F")

    for i, (a, b, c) in enumerate(zip(v2, cor, v5)):
        ax.text(i - width, a + 1500, f"{a:,}", ha="center", fontsize=8, rotation=90)
        ax.text(i,         b + 1500, f"{b:,}", ha="center", fontsize=8, rotation=90)
        ax.text(i + width, c + 1500, f"{c:,}", ha="center", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=20, fontsize=10)
    ax.set_ylabel("Number of reviews (out of 215,583)", fontsize=11)
    ax.set_ylim(0, 95000)
    ax.set_title("Label distribution evolution: LLM → cleanlab corrections → V5", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_class_distribution.png", dpi=150)
    plt.close()
    print("  saved fig02_class_distribution.png")


def fig03_correction_volume():
    """V1 vs V2 cleanlab corrections."""
    v1 = json.load(open("data/processed/rrgen_corrected/correction_stats.json"))
    v2 = json.load(open("data/processed/rrgen_corrected_v2/correction_stats.json"))

    cats = ["llm_kept", "anchor_corrected", "human_verified"]
    v1_counts = [v1["sources"].get("llm_kept", 0),
                 v1["sources"].get("anchor_corrected", 0),
                 v1["sources"].get("human_verified", 0)]
    v2_counts = [v2["sources"].get("llm_kept", 0),
                 v2["sources"].get("anchor_corrected_v2", 0),
                 v2["sources"].get("human_verified", 0)]

    x = np.arange(len(cats))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, v1_counts, width, label="V1 (TF-IDF anchor)", color="#909090")
    ax.bar(x + width / 2, v2_counts, width, label="V2 (RoBERTa anchor)", color="#3D7CB7")

    for i, (a, b) in enumerate(zip(v1_counts, v2_counts)):
        ax.text(i - width / 2, a + 2000, f"{a:,}", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 2000, f"{b:,}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["unchanged (LLM kept)", "anchor corrected", "human verified"],
                       fontsize=10)
    ax.set_ylabel("Number of reviews", fontsize=11)
    ax.set_title("Cleanlab correction outcomes: V1 (TF-IDF) vs V2 (RoBERTa)\n"
                 "215,583 RRGen reviews", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_correction_volume.png", dpi=150)
    plt.close()
    print("  saved fig03_correction_volume.png")


def fig04_v5_agreement():
    """3-way agreement: V2 ↔ V5, V5 ↔ corrected, all three."""
    s = json.load(open("data/processed/rrgen_v5_relabeled/relabel_stats.json"))
    a = s["agreement"]
    v = s["v2_correction_validation"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: agreement %
    keys = ["V2 ↔ V5", "V5 ↔ corrected", "all three"]
    vals = [a["v2_v5"]["pct"], a["v5_corrected"]["pct"], a["all_three"]["pct"]]
    bars = ax1.bar(keys, vals, color=["#909090", "#3D7CB7", "#E07A5F"])
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
                 ha="center", fontsize=10)
    ax1.set_ylabel("% of 215,583 rows in agreement", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.set_title("Three-way classifier agreement", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # Right: V5 as third opinion on V2 corrections
    keys2 = ["V5 supports\ncorrection", "V5 supports\norig V2 LLM", "V5 third\nopinion"]
    vals2 = [v["v5_supports_the_correction"], v["v5_supports_original_v2"],
             v["v5_third_opinion"]]
    pcts2 = [100 * x / sum(vals2) for x in vals2]
    bars2 = ax2.bar(keys2, vals2, color=["#3D7CB7", "#909090", "#E07A5F"])
    for bar, val, pct in zip(bars2, vals2, pcts2):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 200,
                 f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=9)
    ax2.set_ylabel(f"# of cleanlab-corrected rows (n={sum(vals2):,})", fontsize=11)
    ax2.set_title("V5 as independent third opinion on cleanlab corrections", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_v5_agreement_matrix.png", dpi=150)
    plt.close()
    print("  saved fig04_v5_agreement_matrix.png")


def fig05_cluster_purity():
    """Cluster purity bar chart per issue type."""
    s = json.load(open("data/processed/clusters_umap/cluster_validation_score.json"))
    pc = s["per_class"]

    classes = ["performance", "bug_report", "feature_request", "compatibility", "usability"]
    purity = [pc[c]["purity"] for c in classes]
    n_total = [pc[c]["n"] for c in classes]
    y_count = [pc[c]["Y"] for c in classes]
    p_count = [pc[c]["P"] for c in classes]
    n_count = [pc[c]["N"] for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: weighted purity
    bars = ax1.bar(classes, purity, color=[COLORS[c] for c in classes])
    for bar, val, n in zip(bars, purity, n_total):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.2f}\n(n={n})", ha="center", fontsize=9)
    ax1.axhline(y=s["overall_purity"], linestyle="--", color="black", alpha=0.5,
                label=f"overall: {s['overall_purity']:.2f}")
    ax1.set_ylabel("Purity (Y=1, P=0.5, N=0)", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Cluster purity per issue type (50 clusters)", fontsize=12)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")

    # Right: stacked Y/P/N counts
    width = 0.6
    ax2.bar(classes, y_count, width, label="Y (coherent)", color="#81B29A")
    ax2.bar(classes, p_count, width, bottom=y_count, label="P (partial)", color="#F2CC8F")
    ax2.bar(classes, n_count, width, bottom=[y + p for y, p in zip(y_count, p_count)],
            label="N (incoherent)", color="#E07A5F")
    ax2.set_ylabel("Number of clusters", fontsize=11)
    ax2.set_title("Cluster verdict breakdown by class", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_cluster_purity.png", dpi=150)
    plt.close()
    print("  saved fig05_cluster_purity.png")


def fig06_cluster_size_dist():
    """Cluster size histogram per class."""
    clusters = json.load(open("data/processed/clusters_umap/clusters_summary.json"))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
    classes = ["bug_report", "feature_request", "performance", "usability", "compatibility"]

    for ax, cls in zip(axes, classes):
        sizes = [c["review_count"] for c in clusters if c["issue_type"] == cls]
        if sizes:
            ax.hist(sizes, bins=15, color=COLORS[cls], edgecolor="black", linewidth=0.5)
            ax.set_title(f"{cls}\n({len(sizes)} clusters, "
                         f"avg {int(np.mean(sizes))})", fontsize=10)
            ax.set_xlabel("cluster size", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("# clusters", fontsize=10)
    fig.suptitle("Cluster size distribution by issue type (UMAP+HDBSCAN, 194 clusters total)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_cluster_size_dist.png", dpi=150)
    plt.close()
    print("  saved fig06_cluster_size_dist.png")


def fig07_per_class_f1():
    """V3 vs V4 vs V5 per-class F1 (own test sets)."""
    paths = {
        "V3": "models/stage1_classifier_v3/eval_metrics.json",
        "V4": "models/stage1_classifier_v4/eval_metrics.json",
        "V5": "models/stage1_classifier_v5/eval_metrics.json",
    }
    f1_data = {}
    sup_data = {}
    for v, p in paths.items():
        m = json.load(open(p))
        rep = m["test_classification_report"]
        f1_data[v] = [rep[l]["f1-score"] for l in LABELS]
        sup_data[v] = [int(rep[l]["support"]) for l in LABELS]

    x = np.arange(len(LABELS))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, f1_data["V3"], width, label="V3 (cleanlab v1)", color="#909090")
    ax.bar(x,         f1_data["V4"], width, label="V4 (cleanlab v2)", color="#3D7CB7")
    ax.bar(x + width, f1_data["V5"], width, label="V5 (V4+compat aug, production)",
           color="#E07A5F")

    for i, l in enumerate(LABELS):
        for offset, v in zip([-width, 0, width], ["V3", "V4", "V5"]):
            val = f1_data[v][i]
            sup = sup_data[v][i]
            ax.text(i + offset, val + 0.02, f"{val:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("F1 (own held-out test set)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 — V3 vs V4 vs V5 (each on its own test set)\n"
                 "Note: each model's test set differs; use this for trend, not strict comparison",
                 fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_per_class_f1.png", dpi=150)
    plt.close()
    print("  saved fig07_per_class_f1.png")


if __name__ == "__main__":
    print(f"Generating figures in {FIG_DIR}/")
    fig01_classifier_progression()
    fig02_class_distribution()
    fig03_correction_volume()
    fig04_v5_agreement()
    fig05_cluster_purity()
    fig06_cluster_size_dist()
    fig07_per_class_f1()
    print(f"\nDone. {len(list(FIG_DIR.glob('fig*.png')))} figures in {FIG_DIR}/")
