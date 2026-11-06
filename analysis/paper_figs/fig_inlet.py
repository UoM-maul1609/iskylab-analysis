"""Droplet number by size: model/WELAS ratio vs counting threshold, and a single inlet-shrinkage test."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.ticker; import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EX = ["Exp005","Exp006","Exp007","Exp008","Exp009","Exp010","Exp011","Exp013","Exp014","Exp015","Exp016","Exp017","Exp018","Exp019"]
DD2 = 12.0  # um^2
TH = np.array([2.5, 3, 3.5, 4, 5, 6, 7, 8])
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}; return float(m["window_start_s"]), float(m["window_end_s"])
def welas(e, s="d-MergedW"):
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-{s}.csv", skipinitialspace=True); cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]); f = np.nan_to_num(o[cols].values.astype(float)); lg = np.log10(dm)
    ed = np.r_[lg[0] - (lg[1] - lg[0]) / 2, (lg[1:] + lg[:-1]) / 2, lg[-1] + (lg[-1] - lg[-2]) / 2]
    return o.iloc[:, 0].values.astype(float), dm, f * np.diff(ed)
DDS = np.array([0, 2, 4, 6, 8, 10, 12, 16.0]); OP = {"n25": [], "n5": [], "q25": []}
rows = []; R = []; RC = []
for e in EX:
    w = win(e)
    with Dataset(T / "baseline/single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; nw = nc["nwat"][:].data; dw = nc["dwet"][:].data * 1e6; rho = (nc["p"][:] / (287 * nc["t"][:])).data
    s = (t >= w[0]) & (t <= w[1]); nw = nw[s] * rho[s, None, None] / 1e6; dw = dw[s]; dc = np.sqrt(np.maximum(dw**2 - DD2, 0))
    to, dm, n = welas(e); so = (to >= w[0]) & (to <= w[1]); ob = n[so].mean(0)
    no = np.array([ob[dm > x].sum() for x in TH]); nm = np.array([(nw * (dw > x)).sum(axis=(1, 2)).mean() for x in TH]); nc_ = np.array([(nw * (dc > x)).sum(axis=(1, 2)).mean() for x in TH])
    vol = lambda d: np.pi / 6 * (d * 1e-4)**3 * 1e6  # g m-3 per cm-3
    qo5 = (ob * vol(dm))[dm > 5].sum(); qm5 = (nw * vol(dw) * (dw > 5)).sum(axis=(1, 2)).mean()
    sens = {}
    for lab, sn in [("w1", "a-Welas01"), ("w2", "b-Welas02")]:
        a, b, c = welas(e, sn); sa = (a >= w[0]) & (a <= w[1]); sens[lab] = c[sa].mean(0)[b > 2.5].sum()
    R.append(nm / no); RC.append(nc_ / no)
    o25, o5, oq = ob[dm > 2.5].sum(), ob[dm > 5].sum(), (ob * vol(dm))[dm > 2.5].sum(); a, b_, c_ = [], [], []
    for dd in DDS:
        d = np.sqrt(np.maximum(dw**2 - dd, 0)); a.append((nw * (d > 2.5)).sum(axis=(1, 2)).mean() / o25)
        b_.append((nw * (d > 5)).sum(axis=(1, 2)).mean() / o5); c_.append((nw * vol(d) * (d > 2.5)).sum(axis=(1, 2)).mean() / oq)
    OP["n25"].append(a); OP["n5"].append(b_); OP["q25"].append(c_)
    rows.append(dict(exp=e, nobs25=no[0], nmod25=nm[0], nobs5=no[4], nmod5=nm[4], ratio25=nm[0] / no[0], ratio5=nm[4] / no[4], ratio25_corr=nc_[0] / no[0], ratio5_corr=nc_[4] / no[4], lwc5_ratio=qm5 / qo5, welas02_over_01=sens["w2"] / sens["w1"]))
df = pd.DataFrame(rows); df.to_csv(OUT / "inlet_offset.csv", index=False)
R = np.array(R); RC = np.array(RC)
fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.4)); ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
a = ax[0]
q1, q2, q3 = np.nanpercentile(R, [25, 50, 75], axis=0)
a.fill_between(TH, q1, q3, color="#1f6fb4", alpha=0.18, lw=0); a.plot(TH, q2, color="#1f6fb4", marker="o", ms=4)
a.axhline(1, color="k", lw=1); a.set_yscale("log"); a.set_ylim(0.25, 4); a.set_yticks([0.25, 0.5, 1, 2, 4]); a.set_yticklabels(["0.25", "0.5", "1", "2", "4"]); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel("Counting threshold (µm)"); a.set_ylabel("Model / WELAS number above threshold")
a.set_title("(a) Droplet number by size\n(median and quartiles, 14 runs)", loc="left", fontsize=10)
p_ = ax[2]
for k, col, lab, mk in [("n25", "#1f6fb4", "number > 2.5 µm", "o"), ("n5", "#7b52ab", "number > 5 µm", "s"), ("q25", "#2a9d8f", "liquid water > 2.5 µm", "D")]:
    arr = np.array(OP[k]); m1, m2, m3 = np.nanpercentile(arr, [25, 50, 75], axis=0)
    p_.fill_between(DDS, m1, m3, color=col, alpha=0.15, lw=0); p_.plot(DDS, m2, color=col, marker=mk, ms=4, label=lab)
p_.axhline(1, color="k", lw=1); p_.set_yscale("log"); p_.set_ylim(0.4, 3); p_.set_yticks([0.5, 1, 2]); p_.set_yticklabels(["0.5", "1", "2"]); p_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
p_.set_xlabel("Shrinkage applied to model droplets, $\\Delta D^2$ (µm$^2$)"); p_.set_ylabel("Model / WELAS"); p_.legend(frameon=False, fontsize=8, loc="upper right")
p_.set_title("(c) Wall settings as fitted; droplets shrunk as in\na warm inlet, then compared as WELAS measures them", loc="left", fontsize=10)
pd.DataFrame({"dD2": DDS, **{k + "_median": np.nanmedian(np.array(v), 0) for k, v in OP.items()}}).to_csv(OUT / "inlet_operator.csv", index=False)
b = ax[1]; lo, hi = 60, 3000; xx = np.array([lo, hi])
b.fill_between(xx, xx / 2, xx * 2, color="0.9", zorder=0, label="within a factor of 2"); b.plot(xx, xx, color="k", lw=1)
b.scatter(df.nobs25, df.nmod25, marker="o", facecolor="white", edgecolor="#1f6fb4", s=36, label="droplets > 2.5 µm", zorder=3)
b.scatter(df.nobs5, df.nmod5, marker="o", color="#1f6fb4", s=36, label="droplets > 5 µm", zorder=3)
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lo, hi); b.set_ylim(lo, hi); b.set_aspect("equal")
b.set_xlabel("Observed, WELAS (cm$^{-3}$)"); b.set_ylabel("Model (cm$^{-3}$)"); b.legend(frameon=False, fontsize=7.5, loc="upper left")
b.set_title("(b) Model against observations\n(no shrinkage)", loc="left", fontsize=10)
RF = OUT / "inlet_refit_all.csv"
d_ = ax[3]
if RF.exists():
    rf = pd.read_csv(RF); Qs = ["lwc25", "n25", "n5"]
    bst = rf.loc[rf.groupby(["exp", "dD2"]).cost.idxmin()]; ddb = float(bst.groupby("dD2").cost.sum().idxmin())
    Bf = bst[bst.dD2 == ddb].set_index("exp"); B0 = rf[(rf.offset == 0) & (rf.dD2 == 0)].set_index("exp")
    xx_ = np.arange(len(EX)); cc = {"lwc25": "#2a9d8f", "n25": "#1f6fb4", "n5": "#7b52ab"}; ll = {"lwc25": "liquid water > 2.5 µm", "n25": "number > 2.5 µm", "n5": "number > 5 µm"}
    for i, q in enumerate(Qs):
        d_.scatter(xx_ + (i - 1) * 0.22, B0.loc[EX, q], marker="o", facecolor="white", edgecolor=cc[q], s=24, zorder=3)
        d_.scatter(xx_ + (i - 1) * 0.22, Bf.loc[EX, q], marker="o", color=cc[q], s=32, zorder=4, label=ll[q])
    d_.axhspan(0.8, 1.25, color="0.92", zorder=0); d_.axhline(1, color="k", lw=0.8); d_.set_yscale("log"); d_.set_ylim(0.2, 6)
    d_.set_yticks([0.5, 1, 2, 4]); d_.set_yticklabels(["0.5", "1", "2", "4"]); d_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    d_.set_xticks(xx_); d_.set_xticklabels([e[3:] for e in EX], fontsize=7.5); d_.set_ylabel("Model / WELAS")
    d_.legend(frameon=False, fontsize=7.3, loc="lower center", ncol=3, title=f"open: as fitted, no shrinkage;   filled: wall air refitted, $\\Delta D^2$ = {ddb:.0f} µm$^2$", title_fontsize=7.3)
    d_.set_title("(d) Wall air refitted for each experiment, one\ninlet shrinkage for all", loc="left", fontsize=10)
for x in ax: x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_inlet.png", dpi=200); plt.close(fig)
print(df.round(2).to_string(index=False))
print("median ratio by threshold", dict(zip(TH, np.round(np.median(R, 0), 2))))
print("median corrected", dict(zip(TH, np.round(np.median(RC, 0), 2))))
print("median lwc>5 ratio %.2f; welas02/01 median %.2f range %.2f-%.2f; ratio5 range %.2f-%.2f" % (df.lwc5_ratio.median(), df.welas02_over_01.median(), df.welas02_over_01.min(), df.welas02_over_01.max(), df.ratio5.min(), df.ratio5.max()))
