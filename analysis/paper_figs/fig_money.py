"""Summary figure for the competition result (v1.9): one experiment (default Exp006, AS + coarse NaCl), early-cloud
window, paired model runs.  Three pairs of bars: droplets formed (model, with walls), droplets formed (model, sealed
chamber), droplets WELAS would count (> 2.5 um after its inlet, with walls).
Usage: python3 fig_money.py [Exp006]   ->  runs/analysis/fig_money.png and fig_money.csv"""
import os, sys, csv
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
st = T / "analysis/v19_settings.txt"
if "ISKYLAB_INLET_DD2" not in os.environ and st.exists(): os.environ["ISKYLAB_INLET_DD2"] = st.read_text().split()[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
import dataAnalysis_new as DA, paper_tags as PT
E = sys.argv[1] if len(sys.argv) > 1 else "Exp006"; DD2 = DA.cfg.WELAS_INLET_DD2_UM2
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def get(tag, w):
    with Dataset(T / tag / "single_comparison" / f"output-{E}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None] / 1e6
        nw = nc["nwat"][:].data[s] * rho; nl = nc["nliq"][:].data[s] * rho; smax = 100 * (np.nanmax(nc["rh"][:].data) - 1)
        di = DA.apply_welas_inlet({"dwet": nc["dwet"][:].data[s], "mwat": nc["mwat"][:].data[s]}, DD2)["dwet"]
    act = [nl[:, k].sum(-1).mean() for k in range(nl.shape[1])]; seen = [(nw[:, k] * (di[:, k] > 2.5e-6)).sum(-1).mean() for k in range(nw.shape[1])]
    return dict(act=act + [0.0] * (2 - len(act)), seen=seen + [0.0] * (2 - len(seen)), smax=smax)
w = win(E)
R = {k: get(t, w) for k, t in [("bg", PT.BG), ("mix", PT.FINAL), ("bg0", PT.BG_NOBL), ("mix0", PT.MIX_NOBL)]}
AS, NA, GHOST, INK = "#1f6fb4", "#d9822b", "#bdbdbd", "#222222"
groups = [("Model: real chamber\n(walls hold the air near saturation)", "bg", "mix", "act"),
          ("Model: sealed chamber\n(no wall effect)", "bg0", "mix0", "act"),
          ("What WELAS would count\n(real chamber, droplets > 2.5 µm\nafter its warm inlet)", "bg", "mix", "seen")]
fig, ax = plt.subplots(figsize=(10.5, 5.6)); bw = 0.36; rows = []
for g, (title, a, b, q) in enumerate(groups):
    x0 = g * 1.25
    for j, k in enumerate([a, b]):
        x = x0 + (j - 0.5) * (bw + 0.06); v = R[k][q]
        if q == "seen":   # outline: all droplets that formed, most of which WELAS does not count
            ax.bar(x, sum(R[k]["act"]), bw, facecolor="none", edgecolor=GHOST, hatch="///", lw=0.8, zorder=1)
        ax.bar(x, v[0], bw, color=AS, zorder=2); ax.bar(x, v[1], bw, bottom=v[0], color=NA, zorder=2)
        top = max(sum(v), sum(R[k]["act"]) if q == "seen" else 0)
        ax.text(x, top + 25, f"{sum(v):.0f}" + (" counted" if q == "seen" else ""), ha="center", fontsize=10, color=INK)
        ax.text(x, -40, ["AS\nalone", "AS +\nNaCl"][j], ha="center", va="top", fontsize=10)
        rows.append(dict(group=g, run=k, quantity=q, on_AS=v[0], on_NaCl=v[1]))
    va, vb = R[a][q], R[b][q]
    if q == "act":
        ch = 100 * (vb[0] / va[0] - 1); txt = f"droplets on AS\n{ch:+.0f} %".replace("-", "−"); col = "#a33" if ch < -20 else "0.35"
    else:
        ch = 100 * (sum(vb) / sum(va) - 1); txt = f"counted droplets\n{ch:+.0f} %".replace("-", "−"); col = "#2a7a2a" if ch > 0 else "#a33"
    ax.text(x0, 1800, txt, ha="center", va="center", fontsize=12.5, weight="bold", color=col)
    ax.text(x0, 2130, title, ha="center", va="bottom", fontsize=10.5)
ax.set_xlim(-0.65, 3.15); ax.set_ylim(0, 2100); ax.set_xticks([]); ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.set_ylabel("Droplets (cm$^{-3}$), first minutes of the cloud"); ax.axhline(0, color="k", lw=0.8)
for xv in (0.625, 1.875): ax.axvline(xv, color="0.85", lw=0.8, ymax=0.97)
from matplotlib.patches import Patch
H = [Patch(color=AS, label="droplets on ammonium sulfate (AS)"), Patch(color=NA, label="droplets on the added coarse NaCl"),
     Patch(facecolor="none", edgecolor=GHOST, hatch="///", label="formed, but too small for WELAS to count")]
ax.legend(handles=H, frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
fig.suptitle("Adding coarse NaCl suppresses droplets on the AS, because the walls keep the air near saturation,\n"
             "but WELAS would count more droplets, not fewer", fontsize=12.5, weight="bold", y=1.075)
OUT = T / "analysis"; fig.savefig(OUT / "fig_money.png", dpi=200, bbox_inches="tight")
with open(OUT / "fig_money.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(E, "window", w, {k: (np.round(v["act"]), np.round(v["seen"]), round(v["smax"], 3)) for k, v in R.items()})
