"""Replot the model-observation panels for the paper directly from BMM output
(<repo>/runs/<tag>/single_comparison) and the iSKYLAB data, showing only the curves discussed.
Writes fig_exp013.png, fig_nobl.png, fig_fan.png, fig_sens.png to <repo>/runs/analysis."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "scripts"))
import dataAnalysis_new as d
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
OBS, MOD, MOD2, WALL, EVAP, FALL = "#555555", "#1f6fb4", "#d9822b", "#2a9d8f", "#9c6ade", "#e76f51"
EXPS = ["Exp005", "Exp006", "Exp007", "Exp008", "Exp009", "Exp010", "Exp011", "Exp013", "Exp014", "Exp015", "Exp016", "Exp017", "Exp018", "Exp019"]
data = d.load_all_data(EXPS)

def window(exp):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{exp}.csv"))}
    return float(m["window_start_s"]) / 60, float(m["window_end_s"]) / 60

def model(tag, exp):
    with Dataset(T / tag / "single_comparison" / f"output-{exp}.nc") as nc:
        m = {v: np.asarray(nc[v][:]) for v in ["time", "ql", "t", "p", "nwat", "dwet", "ndrop"]}
        for v in ["qchamber_bl", "qchamber_bl_evap", "qfall_liq"]:
            m[v] = np.asarray(nc[v][:]) if v in nc.variables else np.zeros_like(m["time"])
    rho = m["p"] / (287.0 * m["t"])
    big = m["dwet"] > 2.5e-6; n = np.where(big, m["nwat"], 0.0); dd = np.where(big, m["dwet"], 0.0)
    m["N25"] = n.sum(axis=(1, 2)) * rho / 1e6
    m2 = (n * dd**2).sum(axis=(1, 2)); m3 = (n * dd**3).sum(axis=(1, 2)); m1 = (n * dd).sum(axis=(1, 2)); m0 = n.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        m["deff"] = np.where(m2 > 0, m3 / m2 * 1e6, np.nan)
        mean = m1 / m0; sd = np.sqrt(np.maximum((n * dd**2).sum(axis=(1, 2)) / m0 - mean**2, 0))
        m["rd"] = np.where(m0 > 0, sd / mean, np.nan)
    # liquid water above 2.5 um, as compared with welas
    m["ql25"] = (np.pi / 6 * 1000 * (n * dd**3)).sum(axis=(1, 2))
    m["tmin"] = m["time"] / 60
    return m

def obs(exp):
    o = d._observed_bulk_series(exp, data); t = o["time"] / 60
    f = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion" / f"iSKYLAB01-{exp}-2-OPC-d-MergedW.csv"
    op = pd.read_csv(f, skipinitialspace=True); cols = [c for c in op.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]) * 1e-6; n = np.nan_to_num(op[cols].values.astype(float)) * 0.00984 * 1e6  # m-3
    to = op.iloc[:, 0].values / 60; big = dm > 2.5e-6
    N25 = n[:, big].sum(1) / 1e6
    m2 = (n[:, big] * dm[big]**2).sum(1); m3 = (n[:, big] * dm[big]**3).sum(1); m1 = (n[:, big] * dm[big]).sum(1); m0 = n[:, big].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        deff = np.where(N25 > 5, m3 / m2 * 1e6, np.nan); mean = m1 / m0
        rd = np.where(N25 > 5, np.sqrt(np.maximum(m2 / m0 - mean**2, 0)) / mean, np.nan)
    sm = lambda x: pd.Series(x).rolling(15, center=True, min_periods=3).mean().values
    return dict(t=t, ql=sm(o["ql_above_min"]) * 1e3, to=to, N25=sm(N25), deff=sm(deff), rd=sm(rd))

def style(ax, lab, ylabel, xlim=None):
    ax.set_title(lab, loc="left", fontsize=10); ax.set_ylabel(ylabel); ax.spines[["top", "right"]].set_visible(False)
    if xlim: ax.set_xlim(*xlim)

def shade(ax, exp):
    w = window(exp); ax.axvspan(*w, color="0.92", zorder=0)

# ---------------- Fig: Exp013 overview
exp = "Exp013"; m = model("baseline", exp); o = obs(exp); met = data[f"MeteoCPC-{exp}"]
fig, ax = plt.subplots(3, 2, figsize=(9, 8.2), sharex=True)
a = ax[0, 0]; a.plot(np.asarray(met["Time"]) / 60, met["Tgw mean"], color=OBS, label="gas (measured)"); a.plot(np.asarray(met["Time"]) / 60, met["Tww mean"], color=OBS, ls="--", label="wall (measured)")
style(a, "(a) Temperature", "T (°C)"); a.legend(frameon=False, fontsize=8)
a = ax[0, 1]; a.plot(o["t"], o["ql"], color=OBS, lw=1, label="WELAS"); a.plot(m["tmin"], m["ql25"] * 1e3, color=MOD, lw=1.6, label="model"); shade(a, exp)
style(a, "(b) Liquid water ($D>2.5$ µm)", "$q_l$ (g kg$^{-1}$)"); a.legend(frameon=False, fontsize=8)
a = ax[1, 0]; a.plot(o["to"], o["N25"], color=OBS, lw=1, label="WELAS"); a.plot(m["tmin"], m["N25"], color=MOD, lw=1.6, label="model"); a.plot(m["tmin"], m["ndrop"] * m["p"] / (287.0 * m["t"]) / 1e6, color=MOD, lw=1.2, ls="--", label="model, activated"); shade(a, exp); a.legend(frameon=False, fontsize=8)
style(a, "(c) Droplet number ($D>2.5$ µm)", "$N$ (cm$^{-3}$)")
a = ax[1, 1]; a.plot(o["to"], o["deff"], color=OBS, lw=1); a.plot(m["tmin"], m["deff"], color=MOD, lw=1.6); shade(a, exp)
style(a, "(d) Effective diameter", "$D_{\\rm eff}$ (µm)"); a.set_ylim(0, 25)
a = ax[2, 0]; a.plot(o["to"], o["rd"], color=OBS, lw=1); a.plot(m["tmin"], m["rd"], color=MOD, lw=1.6); shade(a, exp)
style(a, "(e) Relative dispersion", "$\\sigma_D/\\bar D$"); a.set_ylim(0, 0.8)
a = ax[2, 1]; a.plot(m["tmin"], m["qchamber_bl"] * 1e3, color=WALL, lw=1.8, label="net loss to walls"); a.plot(m["tmin"], m["qchamber_bl_evap"] * 1e3, color=EVAP, lw=1.8, label="cloud water evaporated by wall air"); a.plot(m["tmin"], m["qfall_liq"] * 1e3, color=FALL, lw=1.8, label="sedimentation")
style(a, "(f) Model water budget (cumulative)", "g kg$^{-1}$"); a.legend(frameon=False, fontsize=8)
for a in ax[2]: a.set_xlabel("Time since start of expansion (min)")
for a in ax.flat: a.set_xlim(0, 29)
fig.tight_layout(); fig.savefig(OUT / "fig_exp013.png", dpi=200); plt.close(fig)

# ---------------- Fig: no wall mixing
s = pd.read_csv(T / "sensitivity_summary.csv"); s["Rq"] = s.integrated_ql_model_kg_s_kg / s.integrated_ql_obs_kg_s_kg
fig = plt.figure(figsize=(9, 6.2)); a1 = fig.add_subplot(2, 2, 1); a2 = fig.add_subplot(2, 2, 2); a3 = fig.add_subplot(2, 1, 2)
mn = model("nobl", "Exp013"); mb = model("baseline", "Exp013")
a1.plot(o["t"], o["ql"], color=OBS, lw=1, label="WELAS"); a1.plot(mn["tmin"], mn["ql25"] * 1e3, color=MOD2, lw=1.6, label="model, no wall mixing"); a1.plot(mb["tmin"], mb["ql25"] * 1e3, color=MOD, lw=1.6, label="model, wall mixing")
style(a1, "(a) Exp013 liquid water", "$q_l$ (g kg$^{-1}$)", (0, 29)); a1.legend(frameon=False, fontsize=8); a1.set_xlabel("Time (min)")
a2.plot(o["to"], o["rd"], color=OBS, lw=1); a2.plot(mn["tmin"], mn["rd"], color=MOD2, lw=1.6); a2.plot(mb["tmin"], mb["rd"], color=MOD, lw=1.6)
style(a2, "(b) Exp013 relative dispersion", "$\\sigma_D/\\bar D$", (0, 29)); a2.set_ylim(0, 0.8); a2.set_xlabel("Time (min)")
x = np.arange(len(EXPS)); b0 = [s[(s.tag == "nobl") & (s.experiment == e)].Rq.iloc[0] for e in EXPS]; b1 = [s[(s.tag == "baseline") & (s.experiment == e)].Rq.iloc[0] for e in EXPS]
a3.bar(x - 0.2, b0, 0.38, color=MOD2, label="no wall mixing"); a3.bar(x + 0.2, b1, 0.38, color=MOD, label="wall mixing"); a3.axhline(1, color="k", lw=0.8)
a3.set_yscale("log"); a3.set_xticks(x); a3.set_xticklabels([e[3:] for e in EXPS]); a3.legend(frameon=False, fontsize=8, ncol=2)
style(a3, "(c) Time-integrated liquid water, model ÷ observed", "$R_q$"); a3.set_xlabel("Experiment")
fig.tight_layout(); fig.savefig(OUT / "fig_nobl.png", dpi=200); plt.close(fig)

# ---------------- Fig: fan off vs fan on
fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex="col")
for j, (exp, lab) in enumerate([("Exp005", "Exp005, fan off"), ("Exp013", "Exp013, fan on")]):
    m = model("baseline", exp); o = obs(exp)
    ax[0, j].plot(o["t"], o["ql"], color=OBS, lw=1, label="WELAS"); ax[0, j].plot(m["tmin"], m["ql25"] * 1e3, color=MOD, lw=1.6, label="model")
    style(ax[0, j], f"({'ab'[j]}) {lab}: liquid water", "$q_l$ (g kg$^{-1}$)", (0, 30)); ax[0, j].legend(frameon=False, fontsize=8)
    ax[1, j].plot(m["tmin"], m["qchamber_bl"] * 1e3, color=WALL, lw=1.8, label="net loss to walls"); ax[1, j].plot(m["tmin"], m["qchamber_bl_evap"] * 1e3, color=EVAP, lw=1.8, label="cloud water evaporated by wall air"); ax[1, j].plot(m["tmin"], m["qfall_liq"] * 1e3, color=FALL, lw=1.8, label="sedimentation")
    style(ax[1, j], f"({'cd'[j]}) {lab}: model water budget", "cumulative (g kg$^{-1}$)", (0, 30)); ax[1, j].set_xlabel("Time (min)")
    if j == 0: ax[1, j].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_fan.png", dpi=200); plt.close(fig)

# ---------------- Fig: process sensitivity
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
b = [s[(s.tag == "baseline") & (s.experiment == e)].Rq.iloc[0] for e in EXPS]
f0 = [100 * (s[(s.tag == "nofallout") & (s.experiment == e)].Rq.iloc[0] / bb - 1) for e, bb in zip(EXPS, b)]
f1 = [100 * (s[(s.tag == "nosce") & (s.experiment == e)].Rq.iloc[0] / bb - 1) for e, bb in zip(EXPS, b)]
ax[0].bar(x - 0.2, f0, 0.38, color=FALL, label="no sedimentation"); ax[0].bar(x + 0.2, f1, 0.38, color=MOD, label="no collisions or coagulation")
ax[0].set_xticks(x); ax[0].set_xticklabels([e[3:] for e in EXPS], fontsize=8); ax[0].legend(frameon=False, fontsize=8); ax[0].axhline(0, color="k", lw=0.6)
style(ax[0], "(a) Change in time-integrated liquid water", "%"); ax[0].set_xlabel("Experiment")
o = obs("Exp005"); mb = model("baseline", "Exp005"); mf = model("nofallout", "Exp005")
ax[1].plot(o["t"], o["ql"], color=OBS, lw=1, label="WELAS"); ax[1].plot(mb["tmin"], mb["ql25"] * 1e3, color=MOD, lw=1.6, label="model"); ax[1].plot(mf["tmin"], mf["ql25"] * 1e3, color=FALL, lw=1.6, label="model, no sedimentation")
style(ax[1], "(b) Exp005 (fan off)", "$q_l$ (g kg$^{-1}$)", (0, 30)); ax[1].legend(frameon=False, fontsize=8); ax[1].set_xlabel("Time (min)")
fig.tight_layout(); fig.savefig(OUT / "fig_sens.png", dpi=200); plt.close(fig)
print("written to", OUT)
