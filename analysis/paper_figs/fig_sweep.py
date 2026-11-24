"""Competition sweep (v1.9, batchRuns_v2.sh v19sweep): the added NaCl scaled from 0 to 4x of the amount used in the
experiment, with wall mixing (real chamber) and without (sealed chamber).  Early-cloud window, as fig_competition_v2.
(a) change in the total number of droplets that form;  (b) change in droplets on the background aerosol;
(c) change in the droplets WELAS would count (> 2.5 um after its inlet), real chamber.
Usage: python3 fig_sweep.py  ->  runs/analysis/fig_sweep.png and fig_sweep.csv"""
import os, sys, csv
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
st = T / "analysis/v19_settings.txt"
if "ISKYLAB_INLET_DD2" not in os.environ and st.exists(): os.environ["ISKYLAB_INLET_DD2"] = st.read_text().split()[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
import dataAnalysis_new as DA, paper_tags as PT
DD2 = DA.cfg.WELAS_INLET_DD2_UM2
SCALES = [0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 1, 1.5, 2.5, 4]
EXPS = [("Exp006", "AS + coarse NaCl", "#c0392b"), ("Exp018", "fulvic acid + coarse NaCl", "#8e44ad"), ("Exp008", "AS + fine NaCl", "#5d6d7e")]
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def get(tag, e):
    p = T / tag / "single_comparison" / f"output-{e}.nc"
    if not p.exists(): return None
    w = win(e)
    with Dataset(p) as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None] / 1e6
        nl = nc["nliq"][:].data[s] * rho; nw = nc["nwat"][:].data[s] * rho
        di = DA.apply_welas_inlet({"dwet": nc["dwet"][:].data[s], "mwat": nc["mwat"][:].data[s]}, DD2)["dwet"]
        nlist = nc["nliq"][:].data  # not used further
    return dict(tot=nl.sum(axis=(1, 2)).mean(), bg=nl[:, 0].sum(-1).mean(), seen=(nw * (di > 2.5e-6)).sum(axis=(1, 2)).mean())
def nacl_cm3(e):   # number of NaCl particles added in the experiment: mode-2 number in the generated namelist x air density
    txt = (T / PT.FINAL / "single_comparison" / f"namelist-{e}.in").read_text()
    import re; v = re.search(r"n_aer1\(1:3,2:2\)\s*=\s*([^d]*?),\s*d_aer1", txt).group(1)
    nkg = sum(float(x) for x in v.split(",") if x.strip())
    with Dataset(T / PT.FINAL / "single_comparison" / f"output-{e}.nc") as nc: rho0 = float(nc["p"][0] / (287 * nc["t"][0]))
    return nkg * rho0 / 1e6
rows = []
for e, lab, col in EXPS:
    n0 = nacl_cm3(e)
    for walls, pre, base in [(True, "v19_sw_s", PT.BG), (False, "v19_sw0_s", PT.BG_NOBL)]:
        B = get(base, e)
        for s in [0] + SCALES:
            R = B if s == 0 else get(f"{pre}{int(round(s * 100)):04d}", e)
            if R is None: continue
            rows.append(dict(exp=e, walls=walls, scale=s, nacl_cm3=s * n0, tot=R["tot"], bg=R["bg"], seen=R["seen"],
                             d_tot=100 * (R["tot"] / B["tot"] - 1), d_bg=100 * (R["bg"] / B["bg"] - 1), d_seen=100 * (R["seen"] / B["seen"] - 1)))
df = pd.DataFrame(rows); OUT = T / "analysis"; df.to_csv(OUT / "fig_sweep.csv", index=False)
print(df.round(1).to_string(index=False))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), sharex=True)
panels = [("d_tot", "(a) All cloud droplets", True), ("d_bg", "(b) Droplets on the background aerosol", True), ("d_seen", "(c) Droplets WELAS would count", False)]
for a, (q, title, both) in zip(axs, panels):
    a.axhspan(-100, 0, color="#f6e3e0", zorder=0); a.axhline(0, color="k", lw=0.8)
    for e, lab, col in EXPS:
        for walls in ([True, False] if both else [True]):
            d = df[(df.exp == e) & (df.walls == walls)].sort_values("nacl_cm3")
            a.plot(d.nacl_cm3, d[q], "-" if walls else "--", color=col, lw=2.2 if walls else 1.3, alpha=1 if walls else 0.7, label=lab if walls else None)
            x1 = d[d.scale == 1]
            a.scatter(x1.nacl_cm3, x1[q], s=90, marker="o", color=col if walls else "white", edgecolor=col if not walls else "k", lw=1.2, zorder=4)
    a.set_title(title, loc="left", fontsize=11, weight="bold"); a.set_xlabel("NaCl particles added (cm$^{-3}$)")
    a.spines[["top", "right"]].set_visible(False); a.set_xscale("symlog", linthresh=30); a.set_xlim(0, None)
axs[0].set_ylabel("Change compared with no NaCl (%)")
lo = min(df[["d_tot", "d_bg"]].min().min(), -10) - 5; hi = max(df[["d_tot", "d_bg", "d_seen"]].max().max(), 10) + 5
for a in axs: a.set_ylim(max(lo, -105), hi)
from matplotlib.lines import Line2D
H = [Line2D([], [], color=c, lw=2.2, label=l) for _, l, c in EXPS] + [
     Line2D([], [], color="0.3", lw=2.2, label="real chamber (walls)"), Line2D([], [], color="0.3", lw=1.3, ls="--", label="sealed chamber (no walls)"),
     Line2D([], [], lw=0, marker="o", mfc="0.5", mec="k", ms=9, label="amount used in the experiment")]
fig.legend(handles=H, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=9.5)
fig.savefig(OUT / "fig_sweep.png", dpi=200, bbox_inches="tight")
