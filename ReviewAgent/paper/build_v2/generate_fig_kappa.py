"""Generate Figure: Cohen's kappa progression + classification accuracy."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASSIFIERS = ["V2 LLM\noriginal", "cleanlab\ncorrected", "V5\nclassifier"]
KAPPA = [0.163, 0.333, 0.592]
ACC = [0.301, 0.442, 0.650]
COLORS = ["#9CA3AF", "#3B82F6", "#D97706"]
BANDS = [
    (0.0,  0.20, "#FEE2E2", "slight (<0.20)"),
    (0.20, 0.40, "#FEF3C7", "fair (0.20-0.40)"),
    (0.40, 0.60, "#FED7AA", "moderate (0.40-0.60)"),
    (0.60, 0.80, "#D1FAE5", "substantial (0.60-0.80)"),
]

fig, (ax_k, ax_a) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

# === Left: kappa progression with Landis-Koch bands ===
for lo, hi, col, _ in BANDS:
    ax_k.axhspan(lo, hi, facecolor=col, alpha=0.4, zorder=0)
bars = ax_k.bar(CLASSIFIERS, KAPPA, color=COLORS, edgecolor="#1F2937",
                linewidth=1.2, zorder=2)
for bar, val in zip(bars, KAPPA):
    ax_k.text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.3f}",
              ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_k.set_ylim(0, 0.85)
ax_k.set_ylabel("Cohen's $\\kappa$ vs expert (n=490)", fontsize=10)
ax_k.set_title("Cohen $\\kappa$ progression through the correction pipeline",
               fontsize=11)
ax_k.tick_params(labelsize=9)
patches = [mpatches.Patch(color=col, label=label, alpha=0.5)
           for _, _, col, label in BANDS]
ax_k.legend(handles=patches, loc="upper left", fontsize=7.5, framealpha=0.9)

# === Right: classification accuracy ===
bars = ax_a.bar(CLASSIFIERS, ACC, color=COLORS, edgecolor="#1F2937",
                linewidth=1.2)
for bar, val in zip(bars, ACC):
    ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
              f"{val * 100:.1f}%", ha="center", va="bottom",
              fontsize=11, fontweight="bold")
ax_a.set_ylim(0, 0.85)
ax_a.set_ylabel("Accuracy vs expert (n=490)", fontsize=10)
ax_a.set_title("Classification accuracy progression", fontsize=11)
ax_a.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("fig_kappa_progression.png", dpi=180,
            bbox_inches="tight", facecolor="white")
print("Saved fig_kappa_progression.png")
