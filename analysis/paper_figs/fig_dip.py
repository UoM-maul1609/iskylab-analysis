"""Summary figure for the competition result (v1.9): change in the total number of droplets that form (on any
aerosol) when the NaCl is added, against the number of NaCl particles added.  Paired model runs (with / without the
NaCl, everything else the same), early-cloud window.  Filled: real chamber (v1.9 wall settings); open: sealed chamber.
Usage: python3 fig_dip.py  ->  runs/analysis/fig_dip.png and fig_dip.csv"""
import os, sys, csv
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
st = T / "analysis/v19_settings.txt"
if "ISKYLAB_INLET_DD2" not in os.environ and st.exists(): os.environ["ISKYLAB_INLET_DD2"] = st.read_text().split()[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
import paper_tags as PT
MIX = [("Exp006", "AS", "coarse", 703), ("Exp007", "AS", "coarse", 2065), ("Exp018", "FA", "coarse", 796),
       ("Exp008", "AS", "fine", 1028), ("Exp009", "AS", "fine", 1588), ("Exp010", "AS", "fine", 3107)]
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def nact(tag, e):
    w = win(e)
    with Dataset(T / tag / "single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None] / 1e6
        return float((nc["nliq"][:].data[s] * rho).sum(axis=(1, 2)).mean())
rows = []
for e, bg, kind, n in MIX:
    r = dict(exp=e, background=bg, nacl=kind, nacl_cm3=n, real_without=nact(PT.BG, e), real_with=nact(PT.FINAL, e),
             sealed_without=nact(PT.BG_NOBL, e), sealed_with=nact(PT.MIX_NOBL, e))
    r["real_change"] = 100 * (r["real_with"] / r["real_without"] - 1); r["sealed_change"] = 100 * (r["sealed_with"] / r["sealed_without"] - 1)
    rows.append(r)
COL = {"coarse": "#c0392b", "fine": "#5d6d7e"}
fig, ax = plt.subplots(figsize=(7.6, 5.4))
ax.axhspan(-100, 0, color="#f6e3e0", zorder=0); ax.axhline(0, color="k", lw=0.8)
ax.text(3350, -57, "fewer droplets than without NaCl", ha="right", va="bottom", fontsize=9.5, color="#a33")
for kind in ("coarse", "fine"):
    R = sorted([r for r in rows if r["nacl"] == kind and r["background"] == "AS"], key=lambda r: r["nacl_cm3"])
    x = [0] + [r["nacl_cm3"] for r in R]
    ax.plot(x, [0] + [r["real_change"] for r in R], "-", color=COL[kind], lw=2.2, zorder=2)
    ax.plot(x, [0] + [r["sealed_change"] for r in R], "--", color=COL[kind], lw=1.2, alpha=0.6, zorder=2)
for r in rows:
    mk = "o" if r["background"] == "AS" else "^"
    ax.scatter(r["nacl_cm3"], r["real_change"], s=90, marker=mk, color=COL[r["nacl"]], edgecolor="k", lw=0.6, zorder=4)
    ax.scatter(r["nacl_cm3"], r["sealed_change"], s=70, marker=mk, facecolor="white", edgecolor=COL[r["nacl"]], lw=1.4, zorder=4)
    v = round(r["real_change"]); lab = "0 %" if v == 0 else (f"{v:+d} %").replace("-", "\u2212")
    left = r["exp"] == "Exp006"
    ax.annotate(lab, (r["nacl_cm3"], r["real_change"]), xytext=(-11 if left else 10, 0 if left else (-4 if v < 0 else 5)), textcoords="offset points",
                fontsize=10, weight="bold", color=COL[r["nacl"]], va="center", ha="right" if left else "left")
ax.scatter(0, 0, s=50, color="k", zorder=5)
ax.text(1450, -38, "coarse NaCl:\na dip", fontsize=11, weight="bold", color=COL["coarse"], ha="center")
ax.text(2550, 3.5, "fine NaCl: no effect", fontsize=11, weight="bold", color=COL["fine"], ha="center", va="bottom")
ax.set_xlim(-100, 3400); ax.set_ylim(-58, 28)
ax.set_xlabel("NaCl particles added (cm$^{-3}$)", fontsize=11)
ax.set_ylabel("Change in the number of cloud droplets\nwhen the NaCl is added (%)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
from matplotlib.lines import Line2D
H = [Line2D([], [], color="0.3", lw=2.2, marker="o", mfc="0.5", mec="k", ms=8, label="model, real chamber (walls)"),
     Line2D([], [], color="0.3", lw=1.2, ls="--", marker="o", mfc="white", mec="0.3", ms=7, label="model, sealed chamber (no walls)"),
     Line2D([], [], color="0.3", lw=0, marker="o", mfc="0.5", mec="k", ms=8, label="ammonium sulfate background"),
     Line2D([], [], color="0.3", lw=0, marker="^", mfc="0.5", mec="k", ms=8, label="fulvic acid background")]
ax.legend(handles=H, frameon=False, fontsize=8.8, loc="upper left", ncol=2, bbox_to_anchor=(0, -0.13))
ax.set_title("Adding coarse NaCl can reduce the number of cloud droplets; fine NaCl does not", fontsize=11.5, weight="bold", loc="left")
OUT = T / "analysis"; fig.savefig(OUT / "fig_dip.png", dpi=200, bbox_inches="tight")
with open(OUT / "fig_dip.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
for r in rows: print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
