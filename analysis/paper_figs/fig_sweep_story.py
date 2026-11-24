"""Fig. 12 (v1.9): what adding coarse NaCl does in Exp006 (ammonium sulfate + coarse NaCl), from the NaCl sweep
(fig_sweep.csv, from fig_sweep.py) and the parcel check (parcel_check.csv, from parcel_check.py).
(a) droplets on each aerosol, real chamber; (b) change in all droplets: real chamber, sealed chamber, adiabatic parcel;
(c) change in droplets formed vs droplets WELAS would count, real chamber.  Early-cloud window.
Usage: python3 fig_sweep_story.py -> runs/analysis/fig_sweep_story.png"""
import os
from pathlib import Path
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]; A = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")) / "analysis"
sw = pd.read_csv(A / "fig_sweep.csv"); pc = pd.read_csv(A / "parcel_check.csv")
R = sw[(sw.exp == "Exp006") & sw.walls].sort_values("nacl_cm3"); S = sw[(sw.exp == "Exp006") & ~sw.walls].sort_values("nacl_cm3")
P = pc[pc.set.str.startswith("parcel, 1.3")].sort_values("nacl_cm3")
X1 = float(R[R.scale == 1].nacl_cm3.iloc[0]); r1 = R[R.scale == 1].iloc[0]
AS, NA, RED, INK = "#2166ac", "#e08214", "#b2182b", "#222222"
plt.rcParams.update({"font.size": 11})
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw=dict(wspace=0.28))
for ax in axs:
    ax.set_xlim(0, 3000); ax.set_xlabel("NaCl particles added (cm$^{-3}$)"); ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(X1, color="0.55", lw=1, ls=":"); 
# (a)
a = axs[0]
a.fill_between(R.nacl_cm3, 0, R.bg, color=AS, alpha=0.85, lw=0, label="on ammonium sulfate")
a.fill_between(R.nacl_cm3, R.bg, R.tot, color=NA, alpha=0.85, lw=0, label="on NaCl")
a.plot(R.nacl_cm3, R.tot, color=INK, lw=1.8)
a.set_ylim(0, 1300); a.set_ylabel("Droplets formed (cm$^{-3}$)")
a.text(X1 + 40, 1000, "amount used\nin Exp006", fontsize=10, color="0.4", va="top")
a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=10)
a.set_title("(a) The NaCl takes the droplets\nfrom the ammonium sulfate", loc="left", fontsize=12, weight="bold")
# (b)
b = axs[1]; b.axhspan(-100, 0, color="#f7e4e1", zorder=0); b.axhline(0, color="k", lw=0.8)
b.plot(R.nacl_cm3, R.d_tot, color=RED, lw=3, label="chamber with walls (model)")
b.plot(S.nacl_cm3, S.d_tot, color="0.35", lw=2, ls="--", label="sealed chamber, no walls")
b.plot(P.nacl_cm3, P.d_tot, color="#1b7837", lw=2, ls=":", label="adiabatic parcel, 1.3 m s$^{-1}$")
b.scatter([X1], [r1.d_tot], s=90, color=RED, edgecolor="k", zorder=5)
b.annotate(f"{r1.d_tot:.0f} %".replace("-", "−"), (X1, r1.d_tot), xytext=(14, -6), textcoords="offset points", color=RED, weight="bold", fontsize=12)
b.set_ylim(-60, 70); b.set_ylabel("Change in droplets formed (%)")
b.legend(frameon=False, loc="upper left", fontsize=10)
b.text(2950, -56, "fewer droplets", ha="right", va="bottom", color="#a33", fontsize=10)
b.set_title("(b) Only the walls turn this into\nfewer droplets", loc="left", fontsize=12, weight="bold")
# (c)
c = axs[2]; c.axhspan(-100, 0, color="#f7e4e1", zorder=0); c.axhline(0, color="k", lw=0.8)
c.plot(R.nacl_cm3, R.d_tot, color=RED, lw=3, label="droplets formed")
c.plot(R.nacl_cm3, R.d_seen, color=INK, lw=3, label="droplets WELAS counts")
for y, col in [(r1.d_tot, RED), (r1.d_seen, INK)]:
    c.scatter([X1], [y], s=90, color=col, edgecolor="k", zorder=5)
    c.annotate(f"{y:+.0f} %".replace("-", "−"), (X1, y), xytext=(14, -4), textcoords="offset points", color=col, weight="bold", fontsize=12)
c.set_ylim(-60, 220); c.set_ylabel("Change (%)")
c.legend(frameon=False, loc="upper left", fontsize=10)
c.set_title("(c) WELAS sees the opposite:\nmore droplets, not fewer", loc="left", fontsize=12, weight="bold")
fig.savefig(A / "fig_sweep_story.png", dpi=200, bbox_inches="tight")
