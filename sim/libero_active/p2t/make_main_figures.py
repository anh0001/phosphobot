"""P2T MAIN figures (paper-plan): Fig 1 hero (data arrows) + Fig 3 basin diagnostic.
Palette/chrome per dataviz reference. Usage: .venv/bin/python p2t/make_main_figures.py"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, RED, GREEN = "#2a78d6", "#e34948", "#1baf7a"
FIGS = Path("p2t/figs"); FIGS.mkdir(exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "savefig.facecolor": SURFACE})

# ================= Fig 1 HERO — schematic (left) + data arrows (right) =================
fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={"width_ratios": [1, 1.15]})

# LEFT: schematic of the readout on the u-axis
axl.set_xlim(-0.4, 1.4); axl.set_ylim(-1.0, 1.2); axl.axis("off")
axl.set_title("The reach readout", fontsize=10, color=INK, loc="left")
# u axis
axl.annotate("", xy=(1.25, 0), xytext=(-0.25, 0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
axl.text(1.28, 0, "u", fontsize=10, va="center", color=INK2)
axl.scatter([0], [0], marker="X", s=130, color=INK, zorder=5)
axl.text(0, -0.22, "object (u=0)", ha="center", fontsize=8, color=INK)
axl.scatter([1], [0], marker="P", s=130, color=MUTED, zorder=5)
axl.text(1, -0.22, "canonical (u=1)", ha="center", fontsize=8, color=MUTED)
# wide grasp basin around object
from matplotlib.patches import Ellipse
axl.add_patch(Ellipse((0, 0), 1.7, 0.5, facecolor=GREEN, alpha=0.12, edgecolor="none", zorder=0))
axl.text(0.0, 0.36, "grasp basin", ha="center", fontsize=7.5, color="#0a7a52")
# two reaches
axl.annotate("", xy=(0.08, 0.62), xytext=(1.0, 0.62), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
axl.text(0.54, 0.74, "grounded reach (u→0)", ha="center", fontsize=8, color=BLUE)
axl.annotate("", xy=(0.95, -0.62), xytext=(1.0, -0.62), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
axl.scatter([0.95], [-0.62], s=40, color=RED, zorder=6)
axl.text(0.62, -0.82, "anchored reach (u→1)\nsucceeds via wide basin", ha="center", fontsize=8, color=RED)

# RIGHT: DATA — arrows in (displaced success, u_succ) space
axr.set_title("Same intervention, opposite mechanism", fontsize=10, color=INK, loc="left")
axr.axhspan(-0.1, 0.5, color=GREEN, alpha=0.07, zorder=0)
axr.text(0.415, 0.02, "object-near reaches", fontsize=7.4, color="#0a7a52", va="bottom", ha="right")
axr.text(0.415, 1.02, "off-object (anchored) reaches", fontsize=7.4, color=MUTED, va="bottom", ha="right")
# spatial arrow: base (30%,1.06) -> M2 (37%,0.51)
axr.annotate("", xy=(0.37, 0.51), xytext=(0.30, 1.06),
             arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
axr.scatter([0.30, 0.37], [1.06, 0.51], s=[55, 70], color=BLUE, zorder=5, edgecolors=SURFACE, linewidths=1.2)
axr.annotate("spatial base", (0.30, 1.06), textcoords="offset points", xytext=(4, 6), fontsize=7.6, color=BLUE)
axr.annotate("spatial +FT\n(basin-widen)", (0.37, 0.51), textcoords="offset points", xytext=(6, -2), fontsize=7.6, color=BLUE)
# object arrow: base (10%,0.67) -> M2 (27%,0.21)
axr.annotate("", xy=(0.27, 0.21), xytext=(0.10, 0.67),
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2))
axr.scatter([0.10, 0.27], [0.67, 0.21], s=[55, 70], color=RED, zorder=5, edgecolors=SURFACE, linewidths=1.2)
axr.annotate("object base", (0.10, 0.67), textcoords="offset points", xytext=(4, 6), fontsize=7.6, color=RED)
axr.annotate("object +FT\n(object-near)", (0.27, 0.21), textcoords="offset points", xytext=(4, -14), fontsize=7.6, color=RED)
axr.set_xlabel("Displaced task success @50 mm"); axr.set_ylabel("Steering u (successes): 0=object, 1=canonical")
axr.set_xlim(0.03, 0.45); axr.set_ylim(-0.05, 1.2)
axr.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
axr.text(0.24, 1.14, "success improves in BOTH — steering tells them apart", fontsize=7.8, color=INK, ha="center")
fig.tight_layout()
fig.savefig(FIGS / "fig_hero.png"); plt.close(fig)

# ================= Fig 3 — basin-tightness diagnostic =================
fig, ax = plt.subplots(figsize=(5.2, 3.4))
mags = [25, 50, 75, 100]
spatial = [0.64, 0.30, 0.18, 0.12]
obj_s = [0.42, 0.10, 0.01, 0.00]
ax.plot(mags, spatial, "-o", color=BLUE, lw=2, markersize=7, label="LIBERO-spatial (wide basin)", zorder=4)
ax.plot(mags, obj_s, "-D", color=RED, lw=2, markersize=6, label="LIBERO-object (tight basin)", zorder=4)
for m, s in zip(mags, spatial):
    ax.annotate(f"{s:.0%}", (m, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.4, color=BLUE)
for m, s in zip(mags, obj_s):
    ax.annotate(f"{s:.0%}", (m, s), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.4, color=RED)
ax.set_xlabel("Object displacement (mm)"); ax.set_ylabel("Base-policy displaced success")
ax.set_ylim(-0.03, 0.72); ax.set_xticks(mags)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.set_title("Basin tightness by suite: object collapses far faster",
             fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIGS / "fig_basin.png"); plt.close(fig)
print("MAIN_FIGS_DONE: fig_hero.png, fig_basin.png")
