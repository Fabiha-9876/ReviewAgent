"""Generate Figure: Human evaluation across the four response-generation conditions."""

import numpy as np
import matplotlib.pyplot as plt

CONDS = ["RRGen\nbaseline", "Core\nbaseline",
         "ReviewAgent\nno-spec", "ReviewAgent\n+ spec (full)"]
QUALITY = [2.31, 2.98, 2.26, 4.62]
SPECIFICITY = [2.31, 2.95, 2.30, 4.55]
HELPFUL_PCT = [19, 84, 31, 92]
COLORS = ["#9CA3AF", "#3B82F6", "#FBBF24", "#DC2626"]

QUAL_ERR = [0.61, 0.71, 0.58, 0.34]
SPEC_ERR = [0.66, 0.74, 0.61, 0.40]

fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# === Left: Quality + Specificity (1-5) grouped bars ===
x = np.arange(len(CONDS))
w = 0.36
b1 = ax_q.bar(x - w / 2, QUALITY, w, yerr=QUAL_ERR, capsize=4,
              color="#4B5563", edgecolor="#111827",
              linewidth=1.0, label="Quality (1-5)")
b2 = ax_q.bar(x + w / 2, SPECIFICITY, w, yerr=SPEC_ERR, capsize=4,
              color="#D1D5DB", edgecolor="#111827",
              linewidth=1.0, label="Specificity (1-5)")
for bar, val in zip(b1, QUALITY):
    ax_q.text(bar.get_x() + bar.get_width() / 2, val + 0.25,
              f"{val:.2f}", ha="center", va="bottom",
              fontsize=10, fontweight="bold")
ax_q.set_xticks(x)
ax_q.set_xticklabels(CONDS, fontsize=9)
ax_q.set_ylabel("Mean rating (1-5)", fontsize=10)
ax_q.set_ylim(0, 6)
ax_q.set_title("Human evaluation: quality + specificity (n=100 per condition)",
               fontsize=10.5)
ax_q.legend(loc="upper left", fontsize=9)
ax_q.grid(axis="y", alpha=0.25)

# === Right: Helpful % bar ===
bars = ax_h.bar(CONDS, HELPFUL_PCT, color=COLORS, edgecolor="#111827",
                linewidth=1.0)
for bar, val in zip(bars, HELPFUL_PCT):
    ax_h.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
              f"{val}%", ha="center", va="bottom",
              fontsize=11, fontweight="bold")
ax_h.set_ylabel("Helpful (Y) %", fontsize=10)
ax_h.set_ylim(0, 105)
ax_h.set_title("Human eval: would this response help the user?", fontsize=10.5)
ax_h.tick_params(axis="x", labelsize=9)
ax_h.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("fig_human_eval.png", dpi=180,
            bbox_inches="tight", facecolor="white")
print("Saved fig_human_eval.png")
