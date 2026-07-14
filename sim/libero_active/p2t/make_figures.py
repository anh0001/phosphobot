"""P2T paper figures (round 1). Palette/chrome per dataviz reference; emphasis
form: A_gain (blue) + D_failure (red) are the subject, controls in grays with
mandatory direct labels (validated relief).

Usage: .venv/bin/python p2t/make_figures.py   -> p2t/figs/*.png (300 dpi)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- chrome (dataviz reference palette, light mode) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
COL = {"A_gain": "#2a78d6", "D_failure": "#e34948",
       "B_noaug": "#8f8d86", "C_uniform": "#aeaca4", "E_sham": "#c6c4bc"}
LABEL = {"A_gain": "A gain-targeted", "B_noaug": "B no-aug",
         "C_uniform": "C uniform", "D_failure": "D failure-targeted",
         "E_sham": "E sham"}
CONDS = ["A_gain", "B_noaug", "C_uniform", "D_failure", "E_sham"]
FIGS = Path("p2t/figs"); FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.facecolor": SURFACE,
})


def load_eval(path: str) -> dict:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "error" not in r]
    out = {"clean": [], "disp": []}
    for r in recs:
        if r["cond"] == "clean":
            out["clean"].append(bool(r["success"]))
            continue
        ep = r["endpoints"]["pregrasp"] or r["endpoints"]["closest"]
        d = np.asarray(r["d_real_xy"], dtype=np.float64)
        if ep is None or np.linalg.norm(d) < 0.01:
            continue
        E = np.asarray(ep[:2]); T = np.asarray(r["true_xyz"][:2])
        e = E - T
        dhat = d / np.linalg.norm(d)
        perp = np.array([-dhat[1], dhat[0]])
        out["disp"].append({
            "success": bool(r["success"]),
            "proj": float(np.dot(e, -d) / np.dot(d, d)),
            "u": float(np.dot(e, -dhat) / np.linalg.norm(d)),
            "v": float(np.dot(e, perp) / np.linalg.norm(d)),
        })
    return out


def base_by_mag(mag_mm: int) -> dict:
    recs = [json.loads(l) for l in Path("p2t/reach_field_retrain.jsonl").read_text().splitlines()]
    keep = [json.dumps(r) for r in recs if "error" not in r and
            (r["cond"] == "clean" or r["cond"].startswith(f"d{mag_mm}mm"))]
    tmp = FIGS / f"_base{mag_mm}.jsonl"; tmp.write_text("\n".join(keep))
    out = load_eval(str(tmp)); tmp.unlink()
    return out


DATA = {c: {"50": load_eval(f"p2t/eval_{c}.jsonl"),
            "75": load_eval(f"p2t/eval75_{c}.jsonl")} for c in CONDS}
BASE = {"50": base_by_mag(50), "75": base_by_mag(75)}


def succ(d): return float(np.mean([r["success"] for r in d["disp"]]))
def projm(d): return float(np.median([r["proj"] for r in d["disp"]]))
def cleanr(d): return float(np.mean(d["clean"])) if d["clean"] else np.nan


# ================= Fig 1: success by condition x regime =================
fig, ax = plt.subplots(figsize=(7.0, 3.4))
regimes = [("Clean\n(canonical)", lambda c: cleanr(DATA[c]["50"]), cleanr(BASE["50"])),
           ("Displaced 50 mm", lambda c: succ(DATA[c]["50"]), succ(BASE["50"])),
           ("Displaced 75 mm", lambda c: succ(DATA[c]["75"]), succ(BASE["75"]))]
W, GAP = 0.15, 0.015
for gi, (name, fn, bval) in enumerate(regimes):
    for ci, c in enumerate(CONDS):
        x = gi + (ci - 2) * (W + GAP)
        y = fn(c)
        ax.bar(x, y, W, color=COL[c], zorder=3,
               label=LABEL[c] if gi == 0 else None)
        ax.text(x, y + 0.012, f"{y:.0%}", ha="center", va="bottom",
                fontsize=7.2, color=INK)
    ax.hlines(bval, gi - 2.6 * (W + GAP), gi + 2.6 * (W + GAP),
              color=INK, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
    ax.text(gi + 2.7 * (W + GAP), bval, f"base {bval:.0%}", fontsize=7.2,
            color=INK2, va="center")
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in regimes], color=INK)
ax.set_ylim(0, 0.95); ax.set_ylabel("Task success rate")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.legend(loc="upper right", frameon=False, fontsize=7.5, ncols=2)
ax.set_title("Counterfactual fine-tuning: success by condition and regime",
             fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIGS / "fig1_success_by_condition.png"); plt.close(fig)

# ================= Fig 2: mechanism — success vs steering =================
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.axhspan(-0.1, 0.70, color="#1baf7a", alpha=0.10, zorder=0)
ax.text(0.985, 0.66, "steering acquired (round-2 gate: proj ≤ 0.70)",
        ha="right", va="top", fontsize=7.2, color="#0a7a52")
ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
ax.text(0.985, 1.015, "full anchoring (reach still goes to canonical)",
        ha="right", va="bottom", fontsize=7.2, color=MUTED)
for c in CONDS:
    for mag, marker in (("50", "o"), ("75", "D")):
        d = DATA[c][mag]
        ax.scatter(succ(d), projm(d), s=52 if marker == "o" else 40,
                   marker=marker, color=COL[c], zorder=4,
                   edgecolors=SURFACE, linewidths=1.2)
for mag, marker in (("50", "o"), ("75", "D")):
    ax.scatter(succ(BASE[mag]), projm(BASE[mag]), s=95, marker="*",
               color=INK, zorder=5, edgecolors=SURFACE, linewidths=0.8)
# selective direct labels: A/D at 50 mm; one cluster label for the 75 mm pile
for c in ("A_gain", "D_failure"):
    d = DATA[c]["50"]
    ax.annotate(f"{LABEL[c].split()[0]}@50", (succ(d), projm(d)),
                textcoords="offset points", xytext=(0, 8), ha="center",
                fontsize=7.2, color=COL[c])
u75 = float(np.mean([succ(DATA[c]["75"]) for c in CONDS]))
ax.annotate("all conditions @75 mm", (u75, 0.955),
            textcoords="offset points", xytext=(-6, -16), ha="center",
            fontsize=7.2, color=INK2)
ax.annotate("base", (succ(BASE["50"]), projm(BASE["50"])),
            textcoords="offset points", xytext=(8, 8), ha="left",
            fontsize=7.2, color=INK)
ax.set_xlabel("Displaced task success"); ax.set_ylabel("Median steering proj  (1 = anchored, 0 = tracks object)")
ax.set_xlim(0.05, 0.45); ax.set_ylim(-0.05, 1.12)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
hnd = [plt.Line2D([], [], marker="o", ls="", color=INK2, label="50 mm"),
       plt.Line2D([], [], marker="D", ls="", color=INK2, label="75 mm"),
       plt.Line2D([], [], marker="*", ls="", color=INK, markersize=10, label="base ckpt")]
ax.legend(handles=hnd, loc="center left", frameon=False, fontsize=7.5)
ax.set_title("Success moves; steering never does — the prior is sticky",
             fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIGS / "fig2_mechanism_success_vs_steering.png"); plt.close(fig)

# ================= Fig 3: realized synthesis distributions =================
fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.5), sharey=True)
buckets = [(0, 37, "25"), (37, 62, "50"), (62, 87, "75"), (87, 130, "100")]
for ax, c in zip(axes, ["A_gain", "C_uniform", "D_failure", "E_sham"]):
    recs = [json.loads(l) for l in open(f"p2t/staging/{c}/meta.jsonl")]
    ok = [r for r in recs if r.get("success") and r.get("cell") != "clean"]
    mags = np.array([np.linalg.norm(r["d_xy"]) * 1000 for r in ok])
    for i, (lo, hi, lbl) in enumerate(buckets):
        n = int(((mags >= lo) & (mags < hi)).sum())
        ax.bar(i, n, 0.66, color=COL[c], zorder=3)
        ax.text(i, n + 0.8, str(n), ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([b[2] for b in buckets])
    ax.set_title(LABEL[c], fontsize=8.5, color=INK)
    ax.set_xlabel("|d| cell (mm)", fontsize=8)
axes[0].set_ylabel("delivered episodes")
fig.suptitle("Realized synthetic-episode displacement distributions (success-filtered)",
             fontsize=10, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIGS / "fig3_synthesis_distributions.png"); plt.close(fig)

# ================= Fig 4: normalized endpoint clouds =================
panels = [("base ckpt", BASE["50"], INK),
          ("A gain-targeted", DATA["A_gain"]["50"], COL["A_gain"]),
          ("D failure-targeted", DATA["D_failure"]["50"], COL["D_failure"])]
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1), sharex=True, sharey=True)
for ax, (name, d, col) in zip(axes, panels):
    U = np.array([r["u"] for r in d["disp"]]); V = np.array([r["v"] for r in d["disp"]])
    S = np.array([r["success"] for r in d["disp"]])
    ax.scatter(U[~S], V[~S], s=13, facecolors="none", edgecolors=col,
               linewidths=0.7, alpha=0.55, zorder=3, label="failure")
    ax.scatter(U[S], V[S], s=13, color=col, alpha=0.75, zorder=4, label="success")
    ax.scatter([0], [0], marker="X", s=70, color=INK, zorder=6)
    ax.scatter([1], [0], marker="P", s=70, color=MUTED, zorder=6)
    ax.annotate("object (T)", (0, 0), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=7.2, color=INK)
    ax.annotate("canonical (C)", (1, 0), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=7.2, color=MUTED)
    med = (float(np.median(U)), float(np.median(V)))
    ax.scatter(*med, marker="o", s=52, facecolors=col, edgecolors=INK,
               linewidths=1.0, zorder=7)
    ax.set_title(f"{name}   median u={med[0]:.2f}", fontsize=8.6, color=INK)
    ax.set_xlim(-0.9, 2.2); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("u = along −d  (0 = object, 1 = canonical)", fontsize=8)
axes[0].set_ylabel("v = perpendicular (|d| units)", fontsize=8)
axes[0].legend(loc="upper left", frameon=False, fontsize=7.2)
fig.suptitle("Pre-grasp endpoints in displacement-normalized frame — 50 mm; every cloud sits on canonical",
             fontsize=10, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIGS / "fig4_endpoint_clouds.png"); plt.close(fig)

print("FIGS_DONE:", sorted(p.name for p in FIGS.glob("*.png")))

# ================= Fig 5: round-2 dose-response (the money figure) =================
def _stats_r2(path, mag):
    recs = [json.loads(l) for l in open(path)]
    disp = [r for r in recs if r.get('cond', '').startswith(f'd{mag}mm') and 'error' not in r]
    clean = [r['success'] for r in recs if r.get('cond') == 'clean']
    ps = []
    for r in disp:
        ep = r['endpoints']['pregrasp'] or r['endpoints']['closest']
        d = np.asarray(r['d_real_xy'])
        if ep is None or np.linalg.norm(d) < 0.01:
            continue
        e = np.asarray(ep[:2]) - np.asarray(r['true_xyz'][:2])
        ps.append(float(np.dot(e, -d) / np.dot(d, d)))
    return float(np.mean(clean)), float(np.median(ps))


def fig5():
    # (label, synthetic fraction %, proj@50, clean, unfrozen?)
    a_clean, a_proj = cleanr(DATA["A_gain"]["50"]), projm(DATA["A_gain"]["50"])
    m1c, m1p = _stats_r2('p2t/eval_r2_M1_ratio50.jsonl', 50)
    m2c, m2p = _stats_r2('p2t/eval_r2_M2_pure.jsonl', 50)
    m3c, m3p = _stats_r2('p2t/eval_r2_M3_plastic.jsonl', 50)
    pts = [("17%\n(round 1, A)", 17, a_proj, a_clean, False),
           ("50%\n(M1)", 50, m1p, m1c, False),
           ("100%\n(M2)", 100, m2p, m2c, False),
           ("100% +\nunfrozen (M3)", 135, m3p, m3c, True)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ax = axes[0]
    ax.axhspan(-0.05, 0.70, color="#1baf7a", alpha=0.10, zorder=0)
    ax.text(14, 0.665, "steering acquired (gate proj ≤ 0.70)", fontsize=7.2,
            color="#0a7a52", va="top")
    frozen = [p for p in pts if not p[4]]
    ax.plot([p[1] for p in frozen], [p[2] for p in frozen], "-", color="#2a78d6",
            linewidth=2, zorder=3)
    for name, x, proj, _, unf in pts:
        col = "#e34948" if unf else "#2a78d6"
        ax.scatter(x, proj, s=64, color=col, zorder=4, edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(f"{proj:.2f}", (x, proj), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.6, color=col)
    ax.set_xticks([p[1] for p in pts]); ax.set_xticklabels([p[0] for p in pts], fontsize=7.2)
    ax.set_ylim(-0.05, 1.1); ax.set_ylabel("Median steering proj @50 mm")
    ax.set_title("Steering vs counterfactual fraction", fontsize=9.5, color=INK, loc="left")
    ax = axes[1]
    ax.plot([p[1] for p in frozen], [p[3] for p in frozen], "-", color="#2a78d6", linewidth=2, zorder=3)
    for name, x, _, cl, unf in pts:
        col = "#e34948" if unf else "#2a78d6"
        ax.scatter(x, cl, s=64, color=col, zorder=4, edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(f"{cl:.0%}", (x, cl), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.6, color=col)
    ax.axhline(0.74, color=INK, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(116, 0.745, "base 74%", fontsize=7.2, color=INK2, ha="right", va="bottom")
    ax.set_xticks([p[1] for p in pts]); ax.set_xticklabels([p[0] for p in pts], fontsize=7.2)
    ax.set_ylim(0.4, 0.9); ax.set_ylabel("Clean success")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("The retention cost", fontsize=9.5, color=INK, loc="left")
    fig.suptitle("The prior is defended by the canonical data: dose-response of steering acquisition",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "fig5_dose_response.png"); plt.close(fig)


fig5()
print("fig5 added")
