"""Can a different wall effect plus evaporation in the welas inlet fit droplet number AND liquid water together?
For every run and every wall-air offset (batchRuns_v2.sh refit), model droplets are shrunk by a uniform dD2 and compared
with welas as welas measures them: liquid water > 2.5 um, number > 2.5 um, number > 5 um.
One dD2 is used for all runs (same inlet); for each dD2 each run takes the offset that fits best.
Cost per run = sum over the three quantities of [log(model/welas)]^2."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EX = ["Exp005","Exp006","Exp007","Exp008","Exp009","Exp010","Exp011","Exp013","Exp014","Exp015","Exp016","Exp017","Exp018","Exp019"]
OFFS = {0.0: "baseline", -0.015: "refit_m015", -0.01: "refit_m010", -0.005: "refit_m005", 0.005: "refit_p005", 0.01: "refit_p010",
        0.015: "refit_p015", 0.02: "refit_p020", 0.025: "refit_p025", 0.03: "refit_p030"}
DDS = np.arange(0, 30.1, 1.0); Q = ["lwc25", "n25", "n5"]
vol = lambda d: np.pi / 6 * (d * 1e-4)**3 * 1e6
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}; return float(m["window_start_s"]), float(m["window_end_s"])
def welas(e):
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-d-MergedW.csv", skipinitialspace=True); cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]); f = np.nan_to_num(o[cols].values.astype(float)); lg = np.log10(dm)
    ed = np.r_[lg[0] - (lg[1] - lg[0]) / 2, (lg[1:] + lg[:-1]) / 2, lg[-1] + (lg[-1] - lg[-2]) / 2]
    return o.iloc[:, 0].values.astype(float), dm, f * np.diff(ed)
rows = []
for e in EX:
    w = win(e); to, dm, n = welas(e); so = (to >= w[0]) & (to <= w[1]); ob = n[so].mean(0)
    obs = dict(lwc25=(ob * vol(dm))[dm > 2.5].sum(), n25=ob[dm > 2.5].sum(), n5=ob[dm > 5].sum())
    for off, tag in OFFS.items():
        p = T / tag / "single_comparison" / f"output-{e}.nc"
        if not p.exists(): continue
        with Dataset(p) as nc:
            t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s]
            nw = nc["nwat"][:].data[s] * rho[:, None, None] / 1e6; dw = nc["dwet"][:].data[s] * 1e6
        for dd in DDS:
            d = np.sqrt(np.maximum(dw**2 - dd, 0)); m = d > 2.5
            r = dict(exp=e, offset=off, dD2=dd, lwc25=(nw * vol(d) * m).sum(axis=(1, 2)).mean() / obs["lwc25"],
                     n25=(nw * m).sum(axis=(1, 2)).mean() / obs["n25"], n5=(nw * (d > 5)).sum(axis=(1, 2)).mean() / obs["n5"])
            r["cost"] = sum(np.log(max(r[q], 1e-3))**2 for q in Q); rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT / "inlet_refit_all.csv", index=False)
best = df.loc[df.groupby(["exp", "dD2"]).cost.idxmin()]            # best offset per run for each common dD2
tot = best.groupby("dD2").cost.sum(); dd_best = float(tot.idxmin())
B = best[best.dD2 == dd_best].set_index("exp"); B0 = best[best.dD2 == 0].set_index("exp")
base = df[(df.offset == 0) & (df.dD2 == 0)].set_index("exp")
B.to_csv(OUT / "inlet_refit_best.csv")
print("offsets available per run:", df.groupby("exp").offset.nunique().to_dict())
print(f"\nbest common inlet shrinkage dD2 = {dd_best:.0f} um2 (total cost {tot[dd_best]:.2f}; no shrinkage, refit: {tot[0]:.2f}; baseline as published: {base.cost.sum():.2f})")
for lab, X in [("baseline, no shrinkage", base), ("refit wall air, no shrinkage", B0), (f"refit wall air + dD2={dd_best:.0f}", B)]:
    within = ((X[Q] > 0.8) & (X[Q] < 1.25)).all(axis=1).sum()
    print(f"{lab:32s} median ratios: LWC {X.lwc25.median():.2f}  N>2.5 {X.n25.median():.2f}  N>5 {X.n5.median():.2f} | runs with all three within 20%: {within} of {len(X)}")
print("\nper run at best dD2:"); print(B[["offset"] + Q].round(2).to_string())
# figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[0.8, 1.4]))
ax[0].plot(tot.index, tot.values, color="k", marker="o", ms=3); ax[0].axvline(dd_best, color="0.5", ls="--")
ax[0].set_xlabel("Inlet shrinkage applied to all runs, $\\Delta D^2$ (µm$^2$)"); ax[0].set_ylabel("Total misfit (all runs; wall air refitted)")
ax[0].set_title("(a) Which inlet shrinkage fits best?", loc="left", fontsize=10)
x = np.arange(len(EX)); cols = {"lwc25": "#2a9d8f", "n25": "#1f6fb4", "n5": "#7b52ab"}; labs = {"lwc25": "liquid water > 2.5 µm", "n25": "number > 2.5 µm", "n5": "number > 5 µm"}
for i, q in enumerate(Q):
    ax[1].scatter(x + (i - 1) * 0.22, base.loc[EX, q], marker="o", facecolor="white", edgecolor=cols[q], s=30, zorder=3)
    ax[1].scatter(x + (i - 1) * 0.22, B.loc[EX, q], marker="o", color=cols[q], s=36, zorder=4, label=labs[q])
ax[1].axhspan(0.8, 1.25, color="0.92", zorder=0); ax[1].axhline(1, color="k", lw=0.8); ax[1].set_yscale("log"); ax[1].set_ylim(0.3, 5)
ax[1].set_yticks([0.5, 1, 2, 4]); ax[1].set_yticklabels(["0.5", "1", "2", "4"]); ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set_xticks(x); ax[1].set_xticklabels([e[3:] for e in EX], fontsize=8); ax[1].set_ylabel("Model / WELAS")
ax[1].legend(frameon=False, fontsize=8, loc="upper right", title=f"filled: wall air refitted + $\\Delta D^2$={dd_best:.0f} µm$^2$\nopen: baseline, no shrinkage", title_fontsize=8)
ax[1].set_title("(b) Fit to all three quantities, each experiment", loc="left", fontsize=10)
for a in ax: a.spines[["top", "right"]].set_visible(False)
import matplotlib.ticker
fig.tight_layout(); fig.savefig(OUT / "fig_inlet_refit.png", dpi=200); plt.close(fig)
