"""Clearer figures for with/without wall mixing and for competition (v1.6)."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
DATA = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
OBS, WITH, WITHOUT = "#444444", "#1f6fb4", "#d9822b"
EXPS = ["Exp005","Exp006","Exp007","Exp008","Exp009","Exp010","Exp011","Exp013","Exp014","Exp015","Exp016","Exp017","Exp018","Exp019"]

def window(exp):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{exp}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def obs_n(exp, thr=2.5):
    o = pd.read_csv(DATA / f"iSKYLAB01-{exp}-2-OPC-d-MergedW.csv", skipinitialspace=True)
    cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]); n = np.nan_to_num(o[cols].values.astype(float)) * 0.00984
    return o.iloc[:, 0].values.astype(float), n[:, dm > thr].sum(1)
def mod(tag, exp):
    with Dataset(T / tag / "single_comparison" / f"output-{exp}.nc") as nc:
        t = nc["time"][:]; n = nc["nwat"][:]; d = nc["dwet"][:]; rho = nc["p"][:] / (287.0 * nc["t"][:]); nd = nc["ndrop"][:]
    big = d > 2.5e-6
    n25 = (n * big).sum(axis=(1, 2)) * rho / 1e6
    ql25 = (np.pi / 6 * 1000 * n * d**3 * big).sum(axis=(1, 2))
    return np.asarray(t), np.asarray(n25), np.asarray(nd * rho / 1e6), np.asarray(ql25)
def wmean(t, x, w): s = (t >= w[0]) & (t <= w[1]); return float(np.nanmean(x[s]))

s = pd.read_csv(T / "sensitivity_summary.csv")
rows = []
for e in EXPS:
    w = window(e); to, no = obs_n(e)
    r = {"exp": e, "obs_N25": wmean(to, no, w)}
    for tag in ("baseline", "nobl"):
        t, n25, nact, ql = mod(tag, e)
        r[f"{tag}_N25"] = wmean(t, n25, w)
        m = s[(s.tag == tag) & (s.experiment == e)].iloc[0]
        r[f"{tag}_Iq_mod"] = m.integrated_ql_model_kg_s_kg; r["Iq_obs"] = m.integrated_ql_obs_kg_s_kg
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT / "with_without_mixing.csv", index=False)

# ---------------- Figure: with vs without wall mixing
import sys as _s; _s.path.insert(0, str(REPO / "scripts")); import dataAnalysis_new as _d
_data = _d.load_all_data(["Exp013"]); _o = _d._observed_bulk_series("Exp013", _data)
fig = plt.figure(figsize=(9, 7.6)); gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2])
a0 = fig.add_subplot(gs[0, :]); ax = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
a0.plot(_o["time"] / 60, pd.Series(_o["ql_above_min"] * 1e3).rolling(15, center=True, min_periods=3).mean(), color=OBS, lw=1.1, label="observed (WELAS)")
for tag, col, name in [("nobl", WITHOUT, "model, no wall mixing"), ("baseline", WITH, "model, with wall mixing")]:
    t, n25, nact, ql = mod(tag, "Exp013"); a0.plot(t / 60, ql * 1e3, color=col, lw=1.8, label=name)
a0.set_xlim(8, 29); a0.set_xlabel("Time since start of expansion (min)"); a0.set_ylabel("$q_l$ (g kg$^{-1}$)")
a0.set_title("(a) Liquid water in Exp013 (fan on)", loc="left", fontsize=10); a0.legend(frameon=False, fontsize=8); a0.spines[["top", "right"]].set_visible(False)
for a, (mcol, ocol, lab, unit, scale) in zip(ax, [("_Iq_mod", "Iq_obs", "(b) Time-integrated liquid water, all runs", "g kg$^{-1}$ min", 1e3 / 60), ("_N25", "obs_N25", "(c) Droplets larger than 2.5 µm, all runs", "cm$^{-3}$", 1.0)]):
    o = df[ocol].values * scale
    for tag, col, mk, name in [("nobl", WITHOUT, "s", "no wall mixing"), ("baseline", WITH, "o", "with wall mixing")]:
        m = df[f"{tag}{mcol}"].values * scale
        a.scatter(o, m, color=col, marker=mk, s=34, label=name, zorder=3)
        for xi, yi, e in zip(o, m, df.exp):
            if e == "Exp005" and mcol == "_Iq_mod": a.annotate("fan off", (xi, yi), fontsize=7, color=col, xytext=(-30, 6), textcoords="offset points")
    lo = min(o.min(), df[[f"nobl{mcol}", f"baseline{mcol}"]].values.min() * scale) * 0.6; hi = max(o.max(), df[[f"nobl{mcol}", f"baseline{mcol}"]].values.max() * scale) * 1.6
    xx = np.array([lo, hi]); a.plot(xx, xx, color="k", lw=1); a.fill_between(xx, xx / 2, xx * 2, color="0.9", zorder=0, label="within a factor of 2")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_xlabel(f"Observed ({unit})"); a.set_ylabel(f"Model ({unit})"); a.set_title(lab, loc="left", fontsize=10)
    a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_nobl_scatter.png", dpi=200); plt.close(fig)

# ---------------- Competition: total activated droplets and OPC count
mix = [("Exp008", "AS\n+fine\n1028"), ("Exp009", "AS\n+fine\n1588"), ("Exp010", "AS\n+fine\n3107"),
       ("Exp006", "AS\n+coarse\n703"), ("Exp007", "AS\n+coarse\n2065"), ("Exp018", "FA\n+coarse\n796")]
ref_obs = {"AS": "Exp016", "FA": "Exp015"}   # single-aerosol runs with the same background (AS 3000, FA 3000)
def mod_n(tag, exp, thr):
    with Dataset(T / tag / "single_comparison" / f"output-{exp}.nc") as nc:
        t = nc["time"][:]; n = nc["nwat"][:]; d = nc["dwet"][:]; rho = nc["p"][:] / (287.0 * nc["t"][:])
    return np.asarray(t), np.asarray((n * (d > thr * 1e-6)).sum(axis=(1, 2)) * rho / 1e6)
crow = []
for e, lab in mix:
    w = window(e); r = {"exp": e, "lab": lab}
    for tag in ["baseline", "comp_bg", "comp_mix_nobl", "comp_bg_nobl"]:
        t, n25, nact, ql = mod(tag, e); r[f"{tag}_act"] = wmean(t, nact, w); r[f"{tag}_N25"] = wmean(t, n25, w)
    to, no = obs_n(e); r["obs_N25"] = wmean(to, no, w)
    ref = ref_obs["FA" if lab.startswith("FA") else "AS"]; to, no = obs_n(ref); r["obs_ref_N25"] = wmean(to, no, window(ref))
    to, no = obs_n(e, 5); r["obs_N5"] = wmean(to, no, w); to, no = obs_n(ref, 5); r["obs_ref_N5"] = wmean(to, no, window(ref))
    t, n5 = mod_n("baseline", e, 5); r["mod_N5"] = wmean(t, n5, w); t, n5 = mod_n("baseline", ref, 5); r["mod_ref_N5"] = wmean(t, n5, window(ref)); r["ref"] = ref
    crow.append(r)
c = pd.DataFrame(crow); c.to_csv(OUT / "competition_totals.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=False)
x = np.arange(len(mix)); w = 0.38
ch_with = 100 * (c.baseline_act / c.comp_bg_act - 1); ch_without = 100 * (c.comp_mix_nobl_act / c.comp_bg_nobl_act - 1)
ax[0].bar(x - w / 2, ch_with, w, color=WITH, label="with wall mixing"); ax[0].bar(x + w / 2, ch_without, w, color=WITHOUT, label="no wall mixing")
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_ylabel("Change in activated droplets (%)")
ax[0].set_title("(a) Model: all activated droplets,\nmixture vs background alone", loc="left", fontsize=10)
ax[0].legend(frameon=False, fontsize=8)
ch_mod5 = 100 * (c.mod_N5 / c.mod_ref_N5 - 1); ch_obs5 = 100 * (c.obs_N5 / c.obs_ref_N5 - 1)
ax[1].bar(x - w / 2, ch_mod5, w, color=WITH, label="model (with wall mixing)"); ax[1].bar(x + w / 2, ch_obs5, w, color=OBS, label="WELAS")
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("Change in droplets > 5 µm (%)")
ax[1].set_title("(b) Droplets > 5 µm, mixture vs\nsingle-aerosol experiment (measurable)", loc="left", fontsize=10); ax[1].legend(frameon=False, fontsize=8)
for a in ax:
    a.set_xticks(x); a.set_xticklabels([l for _, l in mix], fontsize=7.5); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_comp_v16.png", dpi=200); plt.close(fig)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False)); print(c.drop(columns="lab").round(0).to_string(index=False))
print("act change with:", ch_with.round(0).tolist(), " without:", ch_without.round(0).tolist())
print("N5 change model:", ch_mod5.round(0).tolist(), " obs:", ch_obs5.round(0).tolist())
