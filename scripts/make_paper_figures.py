#!/usr/bin/env python3
"""
Paper figures for the extended abstract.

Figure 1 — Layer importance heatmap (ΔWER per layer × language)
Figure 2 — Pruning sweep: zero-shot degradation + post-distillation recovery
Figure 3 — Distillation recovery trajectory (k=6)
Figure 4 — k=7 zero-shot sweep: severity of each candidate 7th layer

Run from the repo root:
    uv run python tools/make_paper_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

OUT  = Path("results/paper")
OUT.mkdir(exist_ok=True)

LANGS = {
    "da_dk": "Danish",
    "en_us": "English",
    "de_de": "German",
    "fr_fr": "French",
}

BLUE   = "#2166AC"
ORANGE = "#D6604D"
GREEN  = "#4DAC26"
GREY   = "#888888"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "figure.dpi":         200,
    "pdf.fonttype":       42,   # editable text in Illustrator/Inkscape
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open("results/delta_wers.json") as f:
    delta_data = json.load(f)

with open("results/prune_sweep_delta_wer.json") as f:
    sweep = json.load(f)

with open("results/distillation_log_5_6_7_9_11_12.json") as f:
    distlog = json.load(f)

# ---------------------------------------------------------------------------
# Figure 1 — Layer importance heatmap
# ---------------------------------------------------------------------------
langs   = list(LANGS.keys())
n_layers = len(delta_data["delta_wers"][langs[0]])
matrix  = np.array([[delta_data["delta_wers"][l][i] for i in range(n_layers)]
                     for l in langs])   # (n_langs, 32)

# Clip extreme values for colour scale readability
vmax = np.percentile(np.abs(matrix), 95)
vmin = -vmax

fig, ax = plt.subplots(figsize=(10, 2.8))
im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)

ax.set_yticks(range(len(langs)))
ax.set_yticklabels([LANGS[l] for l in langs])
ax.set_xlabel("Encoder layer index")
ax.set_xticks(range(0, n_layers, 2))
ax.set_xticklabels(range(0, n_layers, 2), fontsize=9)

# Mark the removed layers
removed = [5, 6, 7, 9, 11, 12]
for idx in removed:
    ax.axvline(idx, color=ORANGE, linewidth=1.5, alpha=0.7)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Rel. ΔWER", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title("Layer importance (ΔWER when layer removed). Orange lines = pruned layers.")
fig.tight_layout()
fig.savefig(OUT / "fig1_layer_heatmap.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_layer_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved fig1_layer_heatmap")

# ---------------------------------------------------------------------------
# Figure 2 — Pruning sweep + post-distillation recovery
# ---------------------------------------------------------------------------
sweep_ks    = [s["k"] for s in sweep["steps"]]
sweep_means = [
    sum(s["rel_delta"][l] for l in langs if l in s["rel_delta"]) /
    sum(1 for l in langs if l in s["rel_delta"])
    for s in sweep["steps"]
]

# Post-distillation for k=6 (from distillation log)
post_k6 = sum(v["rel_delta"] for v in distlog["final"].values()) / len(distlog["final"])

ks = sweep_ks[1:]   # skip k=0
zs = [sweep_means[i] for i, k in enumerate(sweep_ks) if k > 0]

fig, ax = plt.subplots(figsize=(6, 3.8))

bars_zs = ax.bar([k - 0.18 for k in ks], zs,   width=0.32, color=ORANGE, alpha=0.85, label="Zero-shot")
bars_pd = ax.bar([6 + 0.18],              [post_k6], width=0.32, color=BLUE,   alpha=0.85, label="After distillation (k=6)")

ax.axvline(6.5, color=GREY, linestyle="--", linewidth=1.0, alpha=0.8)
ax.text(6.65, ax.get_ylim()[1] * 0.85, "cliff", color=GREY, fontsize=9, style="italic")

ax.set_xlabel("Layers removed (k)")
ax.set_ylabel("Mean relative WER increase")
ax.set_xticks(ks)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"+{x:.0%}" if x >= 0 else f"{x:.0%}"))
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(bottom=0)
ax.set_title("Pruning degrades gracefully to k=6, then collapses")

fig.tight_layout()
fig.savefig(OUT / "fig2_pruning_sweep.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_pruning_sweep.png", bbox_inches="tight")
plt.close()
print("Saved fig2_pruning_sweep")

# ---------------------------------------------------------------------------
# Figure 3 — Distillation recovery trajectory
# ---------------------------------------------------------------------------
steps = [0] + [s["step"] for s in distlog["steps"]]

# Compute per-step mean using only the new 4 languages (filter if needed)
def mean_over_langs(wer_dict):
    vals = [v["rel_delta"] for k, v in wer_dict.items() if k in langs]
    if not vals:   # fall back to all langs if new ones not present yet
        vals = [v["rel_delta"] for v in wer_dict.values()]
    return sum(vals) / len(vals)

means  = [mean_over_langs(distlog["zero_shot"])]
means += [mean_over_langs(s["wer"]) for s in distlog["steps"]]

fig, ax = plt.subplots(figsize=(6, 3.8))

ax.plot(steps, means, color=BLUE, linewidth=2.2, marker="o", markersize=5, zorder=3)
ax.fill_between(steps, means, alpha=0.08, color=BLUE)

ax.annotate(f"+{means[0]:.0%}",
            xy=(steps[0], means[0]), xytext=(100, means[0] + 0.012),
            fontsize=9, color=ORANGE,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0))
ax.annotate(f"+{means[-1]:.0%}",
            xy=(steps[-1], means[-1]), xytext=(steps[-1] - 400, means[-1] + 0.018),
            fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0))

ax.axhline(0, color=GREY, linestyle=":", linewidth=1, alpha=0.5)
ax.set_xlabel("Distillation steps")
ax.set_ylabel("Mean relative WER increase")
ax.set_title("Label-free distillation recovers pruning degradation (~26 min)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"+{x:.0%}" if x >= 0 else f"{x:.0%}"))
ax.set_xlim(-50, steps[-1] + 100)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(OUT / "fig3_distillation_recovery.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_distillation_recovery.png", bbox_inches="tight")
plt.close()
print("Saved fig3_distillation_recovery")

# ---------------------------------------------------------------------------
# Figure 4 — k=7 zero-shot sweep
# ---------------------------------------------------------------------------
with open("results/k7_zero_shot_sweep.json") as f:
    k7sweep = json.load(f)

# Build sorted list: k=6 reference first, then k=7 candidates sorted by degradation
k6_pp = k7sweep["k6_reference"]["mean_delta_pp"]
candidates = sorted(k7sweep["candidates"], key=lambda x: x["mean_delta_pp"])

labels  = ["k=6\n(base)"] + [f"+L{c['seventh_layer']}" for c in candidates]
values  = [k6_pp]         + [c["mean_delta_pp"] for c in candidates]
colors  = [BLUE] + [ORANGE] * len(candidates)

fig, ax = plt.subplots(figsize=(8, 4.0))

bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, width=0.6)

# Horizontal reference line at k=6 level
ax.axhline(k6_pp, color=BLUE, linestyle="--", linewidth=1.2, alpha=0.7)

# Shade the "safe" zone (below 2× k=6 degradation)
ax.axhspan(0, k6_pp * 2, color=BLUE, alpha=0.04)

# Annotate the jump from k=6 to best k=7
best_k7_pp = candidates[0]["mean_delta_pp"]
best_k7_x  = 1   # index in bars
ratio = best_k7_pp / k6_pp
ax.annotate(
    f"×{ratio:.1f}",
    xy=(best_k7_x, best_k7_pp / 2),
    fontsize=9, color=ORANGE, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ORANGE, lw=0.8),
)

# Value labels on bars (skip catastrophic ones to avoid clutter)
for i, (bar, val) in enumerate(zip(bars, values)):
    if val < 60:
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"+{val:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333333")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean WER increase (pp)")
ax.set_title("Any 7th layer removed causes ≥2.6× more degradation than k=6")
ax.set_ylim(bottom=0)

# Clip y-axis to show structure in the interesting range; note clipped bars
clip_at = 70
clipped = [(i, v) for i, v in enumerate(values) if v > clip_at]
if clipped:
    ax.set_ylim(top=clip_at)
    for i, v in clipped:
        ax.text(i, clip_at - 2, f"+{v:.0f}pp", ha="center", va="top",
                fontsize=7.5, color=ORANGE, style="italic")
        # Draw a small arrow to indicate the bar is clipped
        ax.annotate("", xy=(i, clip_at), xytext=(i, clip_at - 6),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, alpha=0.85, label="k=6 (optimal)"),
                   Patch(color=ORANGE, alpha=0.85, label="k=7 candidates")],
          frameon=False, fontsize=9, loc="upper left")

fig.tight_layout()
fig.savefig(OUT / "fig4_k7_sweep.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_k7_sweep.png", bbox_inches="tight")
plt.close()
print("Saved fig4_k7_sweep")

print(f"\nAll figures saved to {OUT}/")
