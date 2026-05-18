"""Generate Figure: Cluster purity audit (per issue type + verdict breakdown)."""

import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["performance", "bug_report", "feature_request",
           "compatibility", "usability"]
N_CLUSTERS = [10, 15, 15, 4, 6]
PURITY = [0.80, 0.67, 0.63, 0.62, 0.50]
OVERALL = 0.66

# Y / P / N counts per class (approximate from purity_w and n)
# purity_w = (|Y| + 0.5|P|) / n, so for each class we infer (Y, P, N):
COUNTS = {
    "performance":     {"Y": 7, "P": 2, "N": 1},
    "bug_report":      {"Y": 6, "P": 8, "N": 1},
    "feature_request": {"Y": 6, "P": 7, "N": 2},
    "compatibility":   {"Y": 2, "P": 1, "N": 1},
    "usability":       {"Y": 1, "P": 4, "N": 1},
}

COLORS = ["#FBBF24", "#EF4444", "#3B82F6", "#8B5CF6", "#10B981"]

fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# === Left: Purity per issue type ===
bars = ax_p.bar(CLASSES, PURITY, color=COLORS,
                edgecolor="#111827", linewidth=1.0)
for bar, val, n in zip(bars, PURITY, N_CLUSTERS):
    ax_p.text(bar.get_x() + bar.get_width() / 2, val + 0.018,
              f"{val:.2f}", ha="center", va="bottom",
              fontsize=10, fontweight="bold")
    ax_p.text(bar.get_x() + bar.get_width() / 2, val / 2,
              f"(n={n})", ha="center", va="center",
              fontsize=9, color="#1F2937")
ax_p.axhline(OVERALL, color="#1F2937", linestyle="--", linewidth=1.0,
             label=f"overall: {OVERALL}")
ax_p.set_ylim(0, 1.0)
ax_p.set_ylabel("Purity (Y=1, P=0.5, N=0)", fontsize=10)
ax_p.set_title("Cluster purity per issue type (50 clusters)", fontsize=10.5)
ax_p.legend(loc="upper right", fontsize=9)
ax_p.tick_params(axis="x", rotation=20, labelsize=9)
ax_p.grid(axis="y", alpha=0.25)

# === Right: Stacked Y/P/N breakdown per class ===
y_vals = [COUNTS[c]["Y"] for c in CLASSES]
p_vals = [COUNTS[c]["P"] for c in CLASSES]
n_vals = [COUNTS[c]["N"] for c in CLASSES]

ax_b.bar(CLASSES, y_vals, color="#10B981", edgecolor="#111827",
         linewidth=0.8, label="Y (coherent)")
ax_b.bar(CLASSES, p_vals, bottom=y_vals, color="#FBBF24",
         edgecolor="#111827", linewidth=0.8, label="P (partial)")
ax_b.bar(CLASSES, n_vals,
         bottom=[y + p for y, p in zip(y_vals, p_vals)],
         color="#EF4444", edgecolor="#111827", linewidth=0.8,
         label="N (incoherent)")
ax_b.set_ylabel("Number of clusters", fontsize=10)
ax_b.set_title("Cluster verdict breakdown by class", fontsize=10.5)
ax_b.legend(loc="upper right", fontsize=9)
ax_b.tick_params(axis="x", rotation=20, labelsize=9)
ax_b.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("fig_cluster_purity.png", dpi=180,
            bbox_inches="tight", facecolor="white")
print("Saved fig_cluster_purity.png")
