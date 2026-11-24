"""Fig. 7 (v1.9): why the model is compared through the WELAS inlet.  All panels use the v1.9 simulations (v19_final).
(a) model/WELAS number of droplets above a counting threshold, compared directly and through the inlet (median, IQR);
(b) model/WELAS for droplets > 2.5 um, > 5 um and liquid water > 2.5 um, every experiment, directly and through the inlet;
(c) total misfit of the joint fit against the inlet shrinkage dD2 (v19_offset_all.csv).  Whole-cloud windows.
Usage: python3 fig_inlet2.py -> runs/analysis/fig_inlet2.png"""
import os, sys, csv
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")); A = T / "analysis"
st = A / "v19_settings.txt"
if "ISKYLAB_INLET_DD2" not in os.environ and st.exists(): os.environ["ISKYLAB_INLET_DD2"] = st.read_text().split()[1]
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
import dataAnalysis_new as DA
DD2 = DA.cfg.WELAS_INLET_DD2_UM2; D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EX = ["Exp005", "Exp006", "Exp007", "Exp008", "Exp009", "Exp010", "Exp011", "Exp013", "Exp014", "Exp015", "Exp016", "Exp017", "Exp018", "Exp019"]
TH = [2.5, 3, 3.5, 4, 5, 6, 7, 8]; vol = lambda d: np.pi / 6 * d**3 * 1000
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "v19_final/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
rows = []
for e in EX:
    w = win(e)
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-d-MergedW.csv", skipinitialspace=True); c = [x for x in o.columns[1:] if x.replace(".", "").isdigit()]
    dm = np.array([float(x) for x in c]); n = np.nan_to_num(o[c].values.astype(float)) * 0.00984; to = o.iloc[:, 0].values; so = (to >= w[0]) & (to <= w[1])
    ob = {th: n[so][:, dm > th].sum(1).mean() for th in TH}; ob["lwc"] = (n[so][:, dm > 2.5] * 1e6 * vol(dm[dm > 2.5] * 1e-6)).sum(1).mean()
    with Dataset(T / "v19_final/single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None]
        nw = nc["nwat"][:].data[s] * rho / 1e6; d0 = nc["dwet"][:].data[s]
        d1 = DA.apply_welas_inlet({"dwet": d0, "mwat": nc["mwat"][:].data[s]}, DD2)["dwet"]
    for lab, d in [("direct", d0), ("inlet", d1)]:
        r = dict(exp=e, how=lab, **{f"N{th}": (nw * (d > th * 1e-6)).sum(axis=(1, 2)).mean() / ob[th] for th in TH})
        r["lwc"] = (nw * 1e6 * vol(d) * (d > 2.5e-6)).sum(axis=(1, 2)).mean() / ob["lwc"]; rows.append(r)
df = pd.DataFrame(rows); df.to_csv(A / "fig_inlet2.csv", index=False)
GREY, BLUE = "#8c8c8c", "#2166ac"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1.1, 0.9], wspace=0.3))
a = ax[0]
for how, col, lab in [("direct", GREY, "compared directly"), ("inlet", BLUE, "through the inlet")]:
    q = df[df.how == how][[f"N{th}" for th in TH]]
    a.fill_between(TH, q.quantile(0.25), q.quantile(0.75), color=col, alpha=0.2, lw=0)
    a.plot(TH, q.median(), "-o", color=col, lw=2.4, ms=5, label=lab)
a.axhline(1, color="k", lw=0.8); a.set_yscale("log"); a.set_ylim(0.4, 4); a.set_yticks([0.5, 1, 2, 4]); a.set_yticklabels(["0.5", "1", "2", "4"])
a.set_xlabel("Counting threshold (µm)"); a.set_ylabel("Model / WELAS, droplets above threshold"); a.legend(frameon=False, fontsize=10)
a.set_title("(a) Through the inlet, the model agrees\nat every counting size", loc="left", fontsize=11.5, weight="bold")
b = ax[1]; Q = [("N2.5", "droplets\n> 2.5 µm"), ("N5", "droplets\n> 5 µm"), ("lwc", "liquid water\n> 2.5 µm")]
b.axhspan(1 / 1.25, 1.25, color="#e5f2e8", zorder=0); b.axhline(1, color="k", lw=0.8)
rng = np.random.default_rng(1)
for i, (k, lab) in enumerate(Q):
    for j, (how, col) in enumerate([("direct", GREY), ("inlet", BLUE)]):
        v = df[df.how == how][k].values; xx = i * 3 + j + rng.uniform(-0.18, 0.18, len(v))
        b.scatter(xx, v, s=28, color=col, alpha=0.85, edgecolor="none", zorder=3)
        b.plot([i * 3 + j - 0.3, i * 3 + j + 0.3], [np.median(v)] * 2, color=col, lw=3, zorder=4)
b.set_xticks([0.5, 3.5, 6.5]); b.set_xticklabels([l for _, l in Q]); b.set_yscale("log"); b.set_ylim(0.4, 4)
b.set_yticks([0.5, 0.8, 1, 1.25, 2, 4]); b.set_yticklabels(["0.5", "0.8", "1", "1.25", "2", "4"]); b.set_ylabel("Model / WELAS, each experiment")
b.text(8.1, 1.27, "within 25 %", fontsize=9, color="#2a7a2a", ha="right", va="bottom")
b.scatter([], [], color=GREY, label="compared directly"); b.scatter([], [], color=BLUE, label="through the inlet"); b.legend(frameon=False, fontsize=10, loc="upper right")
b.set_title("(b) Through the inlet, number and\nliquid water agree together", loc="left", fontsize=11.5, weight="bold")
c = ax[2]; of = pd.read_csv(A / "v19_offset_all.csv"); tot = of.loc[of.groupby(["exp", "dD2"]).cost.idxmin()].groupby("dD2").cost.sum()
c.plot(tot.index, tot.values, color=BLUE, lw=2.4); c.scatter([DD2], [tot.loc[DD2]], s=80, color=BLUE, edgecolor="k", zorder=5)
c.annotate(f"best: {DD2:g} µm$^2$", (DD2, tot.loc[DD2]), xytext=(0, 26), textcoords="offset points", ha="center", fontsize=10.5, weight="bold", color=BLUE)
c.axvspan(16, 40, color="#e08214", alpha=0.13, lw=0); c.text(28, tot.max() * 0.97, "estimate for the\nWELAS sampling\nline: 16–40 µm$^2$", ha="center", va="top", fontsize=9.5, color="#b35806")
c.set_xlabel("Shrinkage in the inlet, $\\Delta D^2$ (µm$^2$)"); c.set_ylabel("Misfit, all 14 experiments"); c.set_ylim(0, None)
c.set_title("(c) One shrinkage fits\nevery experiment", loc="left", fontsize=11.5, weight="bold")
from matplotlib.ticker import NullFormatter
for x in ax: x.spines[["top", "right"]].set_visible(False)
for x in ax[:2]: x.yaxis.set_minor_formatter(NullFormatter())
fig.savefig(A / "fig_inlet2.png", dpi=200, bbox_inches="tight")
d = df.set_index(["how"]); print(df.groupby("how")[["N2.5", "N5", "lwc"]].median().round(2))
