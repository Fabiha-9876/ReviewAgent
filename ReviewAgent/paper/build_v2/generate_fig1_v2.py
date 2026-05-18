"""Generate Figure 1 (IssueSpec architecture) -- v3: clearer, no overlap.

Larger fonts (10-13), more spacing, simplified text per box.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

COLORS = {
    "stage":         "#EFF3F7",
    "stage_edge":    "#334155",
    "substep":       "#FAFBFC",
    "substep_edge":  "#64748B",
    "template":      "#FBF6EC",
    "template_edge": "#92400E",
    "source":        "#F1F8F3",
    "source_edge":   "#166534",
    "trainer":       "#FBEFF4",
    "trainer_edge":  "#9F1239",
    "hitl":          "#FEFAEC",
    "hitl_edge":     "#A16207",
    "arrow":         "#1F2937",
    "feedback":      "#B91C1C",
    "text":          "#0F172A",
    "subtext":       "#475569",
    "label":         "#64748B",
}
FONT = "DejaVu Sans"


def rounded_box(ax, x, y, w, h, fill, edge, lw=1.2, ls="-", radius=0.04):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls,
    ))


def arrow(ax, x0, y0, x1, y1, color=None, lw=2.0, style="-|>", mut=16):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=mut,
        color=color or COLORS["arrow"], linewidth=lw,
    ))


def t(ax, x, y, text, size=10, color=None, ha="center", va="center",
      weight="normal", style="normal"):
    ax.text(x, y, text, fontsize=size, color=color or COLORS["text"],
            ha=ha, va=va, fontweight=weight, style=style, family=FONT)


def title(ax, text):
    ax.text(0.01, 1.02, text, transform=ax.transAxes, fontsize=15,
            fontweight="bold", color=COLORS["text"], va="bottom", ha="left",
            family=FONT)


# ============================================================
# (a) Overall pipeline (top, full width)
# ============================================================
def draw_a(ax):
    ax.set_xlim(0, 120); ax.set_ylim(0, 18); ax.axis("off")
    title(ax, "(a) Overall pipeline")

    stages = [
        ("Reviews",       "215,583 raw"),
        ("Intake",        "classify + ABSA + entities"),
        ("KG",            "3-layer + PageRank"),
        ("IR Translation","5 templates"),
        ("RAG Response",  "5 sources + IR"),
        ("CMDP-RLHF",     "quality + compliance"),
    ]
    n = len(stages)
    w = 17.0; gap = 2.5
    total = n * w + (n - 1) * gap
    x0 = (120 - total) / 2
    y = 5; h = 7

    centers = []
    for i, (name, sub) in enumerate(stages):
        x = x0 + i * (w + gap)
        is_first = (i == 0)
        fill = COLORS["substep"] if is_first else COLORS["stage"]
        edge = COLORS["substep_edge"] if is_first else COLORS["stage_edge"]
        rounded_box(ax, x, y, w, h, fill, edge)
        t(ax, x + w/2, y + h - 2.2, name, size=12, weight="bold")
        t(ax, x + w/2, y + 2.2, sub, size=10, color=COLORS["subtext"])
        centers.append(x + w/2)

    for i in range(n - 1):
        x_a = x0 + (i + 1) * w + i * gap
        x_b = x0 + (i + 1) * w + (i + 1) * gap
        arrow(ax, x_a, y + h/2, x_b, y + h/2)

    fx0 = centers[-1]; fx1 = centers[1]; fy = y - 2.5
    ax.plot([fx0, fx0], [y, fy], color=COLORS["feedback"], lw=1.8,
            linestyle=(0, (5, 3)))
    ax.plot([fx0, fx1], [fy, fy], color=COLORS["feedback"], lw=1.8,
            linestyle=(0, (5, 3)))
    arrow(ax, fx1, fy, fx1, y, color=COLORS["feedback"], lw=1.8, mut=14)
    t(ax, (fx0 + fx1) / 2, fy - 1.0,
      "feedback: Stage 5 corrections retrain Stage 1",
      size=10, color=COLORS["feedback"], style="italic")


# ============================================================
# (b) Intake + Knowledge Graph
# ============================================================
def draw_b(ax):
    ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.axis("off")
    title(ax, "(b) Intake and Knowledge Graph")

    rounded_box(ax, 4, 38, 42, 18, COLORS["stage"], COLORS["stage_edge"])
    t(ax, 25, 53, "Intake", size=14, weight="bold")
    subs = [
        "RoBERTa classifier (Maalej 7-class)",
        "ABSA: aspect-sentiment pairs",
        "Entity extractor (devices, OS, features)",
        "Verified-anchor noise correction",
    ]
    for i, s in enumerate(subs):
        sy = 49 - i * 2.5
        rounded_box(ax, 6, sy - 0.9, 30, 1.8,
                    COLORS["substep"], COLORS["substep_edge"], lw=1.0)
        t(ax, 21, sy, s, size=10)

    rounded_box(ax, 38, 42, 8, 7, COLORS["hitl"], COLORS["hitl_edge"],
                ls=(0, (5, 3)))
    t(ax, 42, 47, "HITL-1", size=11, weight="bold", color="#854D0E")
    t(ax, 42, 44, "classify\nverify", size=9, color="#854D0E")

    arrow(ax, 25, 38, 25, 33, mut=18)
    t(ax, 27, 35.5, "labeled review + aspects + entities",
      size=9.5, color=COLORS["label"], style="italic", ha="left")

    rounded_box(ax, 4, 6, 42, 26, COLORS["stage"], COLORS["stage_edge"])
    t(ax, 25, 29.5, "Knowledge Graph (3 layers)", size=14, weight="bold")

    rounded_box(ax, 6, 22, 38, 4.5, COLORS["substep"], COLORS["substep_edge"], lw=1.0)
    t(ax, 25, 25, "Layer 1: review-aspect-entity graph", size=11, weight="bold")
    t(ax, 25, 23, "Hu-Liu aspect mining + dependency parse",
      size=9, color=COLORS["subtext"])

    rounded_box(ax, 6, 14.5, 38, 4.5, COLORS["substep"], COLORS["substep_edge"], lw=1.0)
    t(ax, 25, 17.5, "Layer 2: hierarchical clustering", size=11, weight="bold")
    t(ax, 25, 15.5, "Sentence-BERT + UMAP + HDBSCAN  ->  605 sub-clusters",
      size=9, color=COLORS["subtext"])

    rounded_box(ax, 6, 7, 38, 4.5, COLORS["substep"], COLORS["substep_edge"], lw=1.0)
    t(ax, 25, 10, "Layer 3: PageRank prioritization", size=11, weight="bold")
    t(ax, 25, 8, "Top-5 aspects cover 74% of complaints",
      size=9, color=COLORS["subtext"])

    arrow(ax, 25, 22, 25, 19, mut=14)
    arrow(ax, 25, 14.5, 25, 11.5, mut=14)
    arrow(ax, 25, 7, 25, 3, mut=16)
    t(ax, 27, 3.5, "prioritized clusters", size=9.5, weight="bold",
      color=COLORS["text"], ha="left")
    t(ax, 27, 1.5, "-> IR Translation",
      size=9, color=COLORS["label"], style="italic", ha="left")


# ============================================================
# (c) IR Translation + 5 templates
# ============================================================
def draw_c(ax):
    ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.axis("off")
    title(ax, "(c) IR Translation with 5 templates")

    # LLM agent box
    rounded_box(ax, 10, 48, 30, 8, COLORS["stage"], COLORS["stage_edge"])
    t(ax, 25, 53.5, "LLM agent (Stage 3)", size=13, weight="bold")
    t(ax, 25, 50.5, "reads cluster, routes by class,\nfills template",
      size=10, color=COLORS["subtext"])

    # HITL-2
    rounded_box(ax, 44, 49, 12, 6, COLORS["hitl"], COLORS["hitl_edge"],
                ls=(0, (5, 3)))
    t(ax, 50, 53.5, "HITL-2", size=11, weight="bold", color="#854D0E")
    t(ax, 50, 51, "5-dim rubric", size=9, color="#854D0E")

    # Templates row -- shorter boxes (height 13 instead of 23)
    templates = [
        ("bug",       "Zimmermann",      "title, severity,\nsteps, component"),
        ("feature",   "Cohn user-story", "As-a / I-want /\nSo-that, BDD"),
        ("perf",      "ISO/IEC 25010",   "speed, battery,\nmemory"),
        ("usability", "Nielsen-10",      "visibility,\nuser control"),
        ("compat",    "Device-OS",       "devices x OS\nmatrix"),
    ]
    tw = 10.5; gap = 0.5
    total = len(templates) * tw + (len(templates) - 1) * gap
    tx0 = (60 - total) / 2
    ty = 28; th = 14   # shorter boxes
    template_tops = []

    for i, (cls, name, fields) in enumerate(templates):
        x = tx0 + i * (tw + gap)
        rounded_box(ax, x, ty, tw, th, COLORS["template"],
                    COLORS["template_edge"])
        t(ax, x + tw/2, ty + th - 1.6, cls, size=10, color="#7C2D12",
          style="italic", weight="bold")
        t(ax, x + tw/2, ty + th - 4.0, name, size=10.5, weight="bold",
          color="#7C2D12")
        t(ax, x + tw/2, ty + 2.2, fields, size=9, color="#7C2D12")
        template_tops.append((x + tw/2, ty + th))

    # Direct fan-out arrows from LLM agent bottom (y=48) to each template top
    # No intermediate "route by class" label - keep arrows clean
    llm_bottom_y = 48
    for cx, cy in template_tops:
        # Start point on LLM box bottom -- offset slightly per arrow for clarity
        # All start at same point (LLM box bottom center), fan out to templates
        arrow(ax, 25, llm_bottom_y, cx, cy, mut=12, lw=1.3)

    # "route by issue class" label -- positioned to the SIDE of the arrows
    t(ax, 50, 45, "route by\nissue class", size=9, style="italic",
      color=COLORS["label"], ha="center")

    # SpecCov
    rounded_box(ax, 4, 14, 52, 7, COLORS["substep"], COLORS["substep_edge"], lw=1.0)
    t(ax, 30, 19, "SpecCov: extractive-coverage faithfulness check",
      size=11, weight="bold")
    t(ax, 30, 16, "validates each spec against its source cluster",
      size=9, color=COLORS["subtext"])

    # Arrow from templates row down to SpecCov
    arrow(ax, 30, 28, 30, 21, mut=14)

    # Arrow from SpecCov to output
    arrow(ax, 30, 14, 30, 8, mut=16)
    t(ax, 30, 5, "validated IssueSpec", size=10, weight="bold",
      color=COLORS["text"])
    t(ax, 30, 3, "-> RAG generator",
      size=9, color=COLORS["label"], style="italic")


# ============================================================
# (d) RAG Sources
# ============================================================
def draw_d(ax):
    ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.axis("off")
    title(ax, "(d) RAG response generation with 5 input sources")

    sources = [
        ("App\nchangelogs",       "60 docs",      "app updates"),
        ("FAQ /\nhelp docs",      "40 docs",      "app Q&A"),
        ("Past review\nresponse", "10,000 pairs", "RRGen corpus"),
        ("Semantically\nsimilar", "5,000 pairs",  "embedding"),
        ("Validated\nIssueSpec",  "per-query",    "from Stage 3"),
    ]
    # Single row at top
    n = len(sources)
    sw = 10.5; sh = 10; gap = 0.6
    total = n * sw + (n - 1) * gap
    sx0 = (60 - total) / 2
    y_top = 38

    centers = []
    for i, (name, count, note) in enumerate(sources):
        x = sx0 + i * (sw + gap)
        rounded_box(ax, x, y_top, sw, sh,
                    COLORS["source"], COLORS["source_edge"])
        t(ax, x + sw/2, y_top + sh - 2.0, name, size=10, weight="bold",
          color="#14532D")
        t(ax, x + sw/2, y_top + sh - 5.2, count, size=9, color="#15803D",
          weight="bold", style="italic")
        t(ax, x + sw/2, y_top + 1.5, note, size=8, color="#15803D")
        centers.append(x + sw/2)

    # RAG retriever (centered, below sources)
    rag_x = 14; rag_y = 14; rag_w = 32; rag_h = 8
    rounded_box(ax, rag_x, rag_y, rag_w, rag_h,
                COLORS["stage"], COLORS["stage_edge"])
    t(ax, 30, rag_y + 5, "RAG retriever + LLM", size=13, weight="bold")
    t(ax, 30, rag_y + 2.5, "ChromaDB index (15,100 docs total)",
      size=10, color=COLORS["subtext"])

    # Straight-down arrows from each source to RAG (no crossings now)
    rag_top = rag_y + rag_h
    for cx in centers:
        # If source center is outside RAG box width, angle the arrow inward
        if cx < rag_x:
            target_x = rag_x + 2
        elif cx > rag_x + rag_w:
            target_x = rag_x + rag_w - 2
        else:
            target_x = cx
        arrow(ax, cx, y_top, target_x, rag_top, lw=1.2, mut=10)

    # HITL-3
    rounded_box(ax, 48, 14, 10, 8, COLORS["hitl"], COLORS["hitl_edge"],
                ls=(0, (5, 3)))
    t(ax, 53, 19.5, "HITL-3", size=11, weight="bold", color="#854D0E")
    t(ax, 53, 16.5, "dual-obj\nfeedback", size=9, color="#854D0E")
    # Arrow from RAG -> HITL-3
    arrow(ax, rag_x + rag_w, rag_y + rag_h/2, 48, 18, lw=1.2, mut=10)

    # Output
    arrow(ax, 30, rag_y, 30, 7, mut=16)
    t(ax, 30, 4, "spec-aware response", size=10, weight="bold",
      color=COLORS["text"])
    t(ax, 30, 2, "-> user / CMDP feedback",
      size=9, color=COLORS["label"], style="italic")


# ============================================================
# (e) CMDP-RLHF Trainer
# ============================================================
def draw_e(ax):
    ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.axis("off")
    title(ax, "(e) CMDP-grounded RLHF trainer chain")

    # Trainer chain at top, evenly spaced
    trainers = [
        ("KTO",  "preference-based\n(Kahneman-Tversky)"),
        ("DPO",  "direct preference\noptimization"),
        ("CPPO", "Lagrangian-\nconstrained PPO"),
    ]
    tw = 14; th = 10; gap = 4
    total = 3 * tw + 2 * gap
    tx0 = (60 - total) / 2  # = 4
    ty = 48

    centers = []
    for i, (name, sub) in enumerate(trainers):
        x = tx0 + i * (tw + gap)
        rounded_box(ax, x, ty, tw, th, COLORS["trainer"],
                    COLORS["trainer_edge"])
        t(ax, x + tw/2, ty + th - 2.5, name, size=12, weight="bold",
          color="#831843")
        t(ax, x + tw/2, ty + 2.8, sub, size=9, color="#831843")
        centers.append(x + tw/2)

    # Horizontal arrows between trainers
    for i in range(2):
        xa = tx0 + (i + 1) * tw + i * gap
        xb = tx0 + (i + 1) * tw + (i + 1) * gap
        arrow(ax, xa, ty + th/2, xb, ty + th/2, mut=14)

    cppo_x = centers[2]
    cppo_bottom = ty

    # R, C placed side-by-side, centered horizontally under the chain
    rb_w = 22; rb_h = 10; rb_gap = 2
    total_rc = 2 * rb_w + rb_gap
    rb_x0 = (60 - total_rc) / 2  # = 7
    r_y = 26

    # R box
    r_x = rb_x0
    rounded_box(ax, r_x, r_y, rb_w, rb_h,
                COLORS["substep"], COLORS["substep_edge"], lw=1.0)
    t(ax, r_x + rb_w/2, r_y + rb_h - 2.5, "Quality reward (R)",
      size=11, weight="bold")
    t(ax, r_x + rb_w/2, r_y + 3.2,
      "helpfulness, specificity,\nempathy, accuracy",
      size=9, color=COLORS["subtext"])

    # C box
    c_x = rb_x0 + rb_w + rb_gap
    rounded_box(ax, c_x, r_y, rb_w, rb_h,
                COLORS["substep"], COLORS["substep_edge"], lw=1.0)
    t(ax, c_x + rb_w/2, r_y + rb_h - 2.5, "Compliance constraints (C)",
      size=11, weight="bold")
    t(ax, c_x + rb_w/2, r_y + 3.2,
      "truthful, no leakage,\nprofessional, legal-safe",
      size=9, color=COLORS["subtext"])

    # Orthogonal "rake" routing: R and C send verticals up to a shared
    # horizontal bar, which extends right under CPPO and joins via a
    # single vertical arrow into CPPO bottom. Clean, no diagonals.
    r_top_x = r_x + rb_w/2     # 18
    c_top_x = c_x + rb_w/2     # 42
    bar_y = 42                  # horizontal bar level

    # Vertical from R top up to bar
    ax.plot([r_top_x, r_top_x], [r_y + rb_h, bar_y],
            color=COLORS["arrow"], lw=1.4, solid_capstyle="butt")
    # Vertical from C top up to bar
    ax.plot([c_top_x, c_top_x], [r_y + rb_h, bar_y],
            color=COLORS["arrow"], lw=1.4, solid_capstyle="butt")
    # Horizontal bar from R top column extending right to CPPO column
    ax.plot([r_top_x, cppo_x], [bar_y, bar_y],
            color=COLORS["arrow"], lw=1.4, solid_capstyle="butt")
    # Single vertical arrow from bar up into CPPO bottom
    arrow(ax, cppo_x, bar_y, cppo_x, cppo_bottom,
          color=COLORS["arrow"], lw=1.4, mut=12)

    # CMDP objective formula, between R/C panel and aligned generator
    t(ax, 30, 21,
      r"$\max_\theta\ \mathbb{E}[R]\ \ \mathrm{s.t.}\ \ \mathbb{E}[C] \leq \tau$",
      size=12, color=COLORS["text"])
    t(ax, 30, 18.5, "CMDP objective",
      size=9, color=COLORS["label"], style="italic")

    # Straight-down "trained theta" arrow from objective to aligned generator
    arrow(ax, 30, 17, 30, 13.5, lw=1.5, mut=14)
    t(ax, 31, 15, r"trained $\theta$", size=9,
      color=COLORS["subtext"], style="italic", ha="left")

    # Aligned response generator (centered, bottom)
    rounded_box(ax, 10, 4, 40, 8, COLORS["stage"], COLORS["stage_edge"])
    t(ax, 30, 9, "Aligned response generator", size=12, weight="bold")
    t(ax, 30, 6, "deployed for spec-aware replies",
      size=9, color=COLORS["subtext"], style="italic")

    # Feedback loop on LEFT side: aligned gen left edge -> up -> KTO left edge
    fb_x = 2.0
    ax.plot([10, fb_x], [8, 8],
            color=COLORS["feedback"], lw=1.3, linestyle=(0, (4, 2)))
    ax.plot([fb_x, fb_x], [8, ty + th/2],
            color=COLORS["feedback"], lw=1.3, linestyle=(0, (4, 2)))
    arrow(ax, fb_x, ty + th/2, tx0, ty + th/2,
          color=COLORS["feedback"], lw=1.3, mut=12)
    t(ax, fb_x + 0.7, 30, "preference\nfeedback",
      size=8.5, color=COLORS["feedback"], style="italic", ha="left")


def main():
    # 3-row layout: pipeline overview (a) spans both columns at top,
    # then (b)(c) and (d)(e) in 2x2 grid below.
    fig = plt.figure(figsize=(28, 20), dpi=180, facecolor="white")
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.28, 1.0, 1.0],
                  hspace=0.22, wspace=0.10,
                  left=0.02, right=0.98, top=0.96, bottom=0.03)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    draw_a(ax_a)
    draw_b(ax_b)
    draw_c(ax_c)
    draw_d(ax_d)
    draw_e(ax_e)

    out = "fig1_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
