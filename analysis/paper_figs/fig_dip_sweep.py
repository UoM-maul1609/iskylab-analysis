"""Main competition figure (v1.9): from runs/analysis/fig_sweep.csv (written by fig_sweep.py after
batchRuns_v2.sh v19sweep).  (a) droplets that form vs NaCl added, real chamber (walls) and sealed chamber;
(b) droplets WELAS would count (> 2.5 um after its inlet), real chamber.  Early-cloud window.
Usage: python3 fig_dip_sweep.py  ->  runs/analysis/fig_dip_sweep.png"""
import os
from pathlib import Path
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")); A = T / "analysis"
df = pd.read_csv(A / "fig_sweep.csv")
EXPS = [("Exp006", "ammonium sulfate + coarse NaCl", "#c0392b"), ("Exp018", "fulvic acid + coarse NaCl", "#8e44ad"),
        ("Exp008", "ammonium sulfate + fine NaCl", "#7f8c8d")]
XMAX = 3100
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw=dict(wspace=0.22))
for ax in (a, b):
    ax.axhspan(-100, 0, color="#f6e3e0", zorder=0); ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(0, XMAX); ax.set_ylim(-60, 80); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("NaCl particles added (cm$^{-3}$)", fontsize=11)
    ax.text(XMAX * 0.98, -57, "fewer droplets than with no NaCl", ha="right", va="bottom", fontsize=9.5, color="#a33")
for e, lab, col in EXPS:
    for walls in (True, False):
        d = df[(df.exp == e) & (df.walls == walls) & (df.nacl_cm3 <= XMAX)].sort_values("nacl_cm3")
        a.plot(d.nacl_cm3, d.d_tot, "-" if walls else "--", color=col, lw=2.6 if walls else 1.4, alpha=1 if walls else 0.75)
        x1 = d[d.scale == 1]
        a.scatter(x1.nacl_cm3, x1.d_tot, s=110, color=col if walls else "white", edgecolor="k" if walls else col, lw=1.3, zorder=5)
    d = df[(df.exp == e) & df.walls & (df.nacl_cm3 <= XMAX)].sort_values("nacl_cm3")
    b.plot(d.nacl_cm3, d.d_seen, "-", color=col, lw=2.6)
    x1 = d[d.scale == 1]; b.scatter(x1.nacl_cm3, x1.d_seen, s=110, color=col, edgecolor="k", lw=1.3, zorder=5)
r = df[(df.exp == "Exp006") & df.walls & (df.scale == 1)].iloc[0]
a.annotate(f"Exp006 sat at the bottom\nof the dip: {r.d_tot:.0f} %".replace("-", "−"), (r.nacl_cm3, r.d_tot), xytext=(40, -56),
           fontsize=10.5, weight="bold", color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))
a.text(1350, 50, "sealed chamber: no dip\nfor ammonium sulfate", fontsize=10, color="#c0392b", ha="center")
b.annotate(f"at the same point WELAS\nwould count {r.d_seen:+.0f} %".replace("-", "−"), (r.nacl_cm3, r.d_seen), xytext=(1250, 62),
           fontsize=10.5, weight="bold", color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))
a.set_ylabel("Change compared with no NaCl (%)", fontsize=11)
a.set_title("(a) Cloud droplets that form (model)", loc="left", fontsize=12, weight="bold")
b.set_title("(b) Droplets WELAS would count (model, real chamber)", loc="left", fontsize=12, weight="bold")
H = [Line2D([], [], color=c, lw=2.6, label=l) for _, l, c in EXPS] + [
     Line2D([], [], color="0.3", lw=2.6, label="real chamber (walls)"), Line2D([], [], color="0.3", lw=1.4, ls="--", label="sealed chamber (no walls)"),
     Line2D([], [], lw=0, marker="o", mfc="0.5", mec="k", ms=10, label="amount added in the experiment")]
fig.legend(handles=H, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
fig.suptitle("Coarse NaCl suppressed cloud droplets in the chamber, but the droplet counter would have seen more, not fewer",
             fontsize=13.5, weight="bold", y=1.0)
fig.savefig(A / "fig_dip_sweep.png", dpi=200, bbox_inches="tight")
