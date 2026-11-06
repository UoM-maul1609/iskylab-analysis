"""Figures for the v1.3 manuscript from the batchRuns_v2.sh output in <repo>/runs.
fig_mixtype.png : evaporation treatment (full-moving bins)
fig_comp.png    : competition counterfactuals
"""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]
T = Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
if len(sys.argv) > 1: T = Path(sys.argv[1])
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
DATA = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
INK, MUTED = "#222222", "#666666"
C = {"baseline": "#1f6fb4", "hom": "#d9822b", "p0": "#7b4ea3", "obs": "#444444", "nobl": "#b03a2e"}

def window(exp):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{exp}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])

def obs_n(exp, thr=2.5):
    o = pd.read_csv(DATA / f"iSKYLAB01-{exp}-2-OPC-d-MergedW.csv", skipinitialspace=True)
    cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]); n = np.nan_to_num(o[cols].values.astype(float)) * 0.00984
    return o.iloc[:, 0].values.astype(float), n[:, dm > thr].sum(1)

def mod_n(tag, exp, thr=2.5, per_mode=False):
    with Dataset(T / tag / "single_comparison" / f"output-{exp}.nc") as nc:
        t = nc["time"][:]; n = nc["nwat"][:]; d = nc["dwet"][:]; rho = nc["p"][:] / (287.0 * nc["t"][:])
    nm = (n * (d > thr * 1e-6)).sum(axis=2) * np.asarray(rho)[:, None] / 1e6
    return (t, nm) if per_mode else (t, nm.sum(1))

# ------------------------------------------------------------------ mixing type
s = pd.read_csv(T / "sensitivity_summary.csv")
s["Rq"] = s.integrated_ql_model_kg_s_kg / s.integrated_ql_obs_kg_s_kg
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
to, no = obs_n("Exp013"); ax[0].plot(to / 60, pd.Series(no).rolling(15, center=True, min_periods=3).mean(), color=C["obs"], lw=1.2, label="WELAS")
for tag, lab in [("baseline", "inhomogeneous, size-weighted (baseline)"), ("hom", "homogeneous ($D^2$ decrement)"), ("p0", "inhomogeneous, uniform ($p=0$)")]:
    t, n = mod_n(tag, "Exp013"); ax[0].plot(t / 60, n, color=C[tag], lw=1.5, label=lab)
w = window("Exp013"); ax[0].axvspan(w[0] / 60, w[1] / 60, color="0.9", zorder=0)
ax[0].set_xlabel("Experiment time (min)"); ax[0].set_ylabel("$N$ ($D>2.5$ µm) (cm$^{-3}$)"); ax[0].set_title("(a) Exp013", loc="left")
ax[0].legend(frameon=False, fontsize=7.5); ax[0].set_xlim(8, 29)
tags = ["baseline", "hom", "p0"]; metrics = [("ndrop_psd_nrmse", "$N$ NRMSE"), ("deff_nrmse", "$D_{\\rm eff}$ NRMSE"), ("Rq", "$R_q$")]
x = np.arange(len(metrics)); wd = 0.26
for i, tag in enumerate(tags):
    v = [s[s.tag == tag][m].median() for m, _ in metrics]
    b = ax[1].bar(x + (i - 1) * wd, v, wd, color=C[tag])
    for xx, vv in zip(x + (i - 1) * wd, v): ax[1].text(xx, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in metrics]); ax[1].set_title("(b) Medians, 14 runs", loc="left")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_mixtype.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ competition
mix = [("Exp008", "AS+fine\n1028"), ("Exp009", "AS+fine\n1588"), ("Exp010", "AS+fine\n3107"),
       ("Exp006", "AS+coarse\n703"), ("Exp007", "AS+coarse\n2065"), ("Exp018", "FA+coarse\n796"), ("Exp019", "FA+coarse\n2059")]
rows = []
for exp, lab in mix:
    w = window(exp); r = {"exp": exp}
    for tag in ["baseline", "comp_bg", "comp_mix_nobl", "comp_bg_nobl"]:
        t, nm = mod_n(tag, exp, 1.0, per_mode=True); sel = (t >= w[0]) & (t <= w[1])
        r[tag + "_bg"] = nm[sel, 0].mean(); r[tag + "_nacl"] = nm[sel, 1].mean()
        t, n25 = mod_n(tag, exp, 2.5); r[tag + "_N25"] = n25[(t >= w[0]) & (t <= w[1])].mean()
    to, no = obs_n(exp); r["obs_N25"] = no[(to >= w[0]) & (to <= w[1])].mean()
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT / "competition_table.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
x = np.arange(len(mix)); wd = 0.2
for i, (tag, lab, col, hatch) in enumerate([("comp_bg", "background alone, wall mixing", "#9ecae1", ""), ("baseline", "background in mixture, wall mixing", "#1f6fb4", ""),
                                            ("comp_bg_nobl", "background alone, no wall mixing", "#fdd0a2", ""), ("comp_mix_nobl", "background in mixture, no wall mixing", "#d9822b", "")]):
    ax[0].bar(x + (i - 1.5) * wd, df[tag + "_bg"], wd, color=col, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in mix], fontsize=7)
ax[0].set_ylabel("Background droplets $D>1$ µm (cm$^{-3}$)"); ax[0].set_title("(a) Activation of the background aerosol", loc="left"); ax[0].legend(frameon=False, fontsize=7)
wd = 0.27
ax[1].bar(x - wd, df["comp_bg_N25"], wd, color="#9ecae1", label="model, background alone")
ax[1].bar(x, df["baseline_N25"], wd, color="#1f6fb4", label="model, mixture")
ax[1].bar(x + wd, df["obs_N25"], wd, color=C["obs"], label="WELAS, mixture")
ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in mix], fontsize=7)
ax[1].set_ylabel("$N$ ($D>2.5$ µm) (cm$^{-3}$)"); ax[1].set_title("(b) Droplets seen by the OPC", loc="left"); ax[1].legend(frameon=False, fontsize=7)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_comp.png", dpi=200); plt.close(fig)
print(df.round(0).to_string(index=False)); print("figures in", OUT)
