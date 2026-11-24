"""Warm runs with the combined wall closure (BMM wall-water mode 3; batchRuns_v2.sh combined / combined_score).
Mode 3 = measured RH0 + the mode-2 finite-rate wall vapour flux at all times + the mode-0 saturation cap in cloud.
Tags: baseline (published), wallsink_km3.0e-4 (mode 2 only, for reference), comb_liq_km* (wall water stays liquid
below 0 C), comb_ice_km* (wall water freezes below 0 C).  Same diagnostics and the same decision rule, fixed before
the runs, as wallsink_check.py:
  (a) predicted saturation within 1 min of the hygrometer in >= 9 of 12 fan-on runs,
  (b) pre-cloud RH RMS <= 1 percentage point in >= 9 of 12, and
  (c) 0.5 <= R_q <= 2 in >= 12 of 14.
Writes runs/analysis/combined_check.csv."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EX = ["Exp005","Exp006","Exp007","Exp008","Exp009","Exp010","Exp011","Exp013","Exp014","Exp015","Exp016","Exp017","Exp018","Exp019"]
FANON = [e for e in EX if e not in ("Exp005", "Exp017")]
TAGS = ["baseline", "wallsink_km3.0e-4"] + [f"comb_liq_km{k}" for k in ["0.0", "1.0e-4", "2.0e-4", "3.0e-4", "5.0e-4", "1.0e-3"]] \
       + [f"comb_ice_km{k}" for k in ["1.0e-4", "3.0e-4", "1.0e-3"]] \
       + [f"combfit_dT-{d}" for d in ["0.01", "0.02", "0.03", "0.05", "0.08", "0.12"]]
es = lambda Tc: 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))           # Pa, T in C, over liquid
def obs(e):
    m = pd.read_csv(D / f"iSKYLAB01-{e}-1-MeteoCPC.csv", skipinitialspace=True)
    t = m.iloc[:, 0].values.astype(float); p = m.iloc[:, 1].values * 100; Tg = m.iloc[:, 2].values; Td = m.iloc[:, 6].values
    rh = 100 * es(Td) / es(Tg); rhs = pd.Series(rh).rolling(15, center=True, min_periods=3).mean().values
    qt = 0.622 * es(Td) / (p - es(Td))
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-d-MergedW.csv", skipinitialspace=True); c = [x for x in o.columns[1:] if x.replace(".", "").isdigit()]
    dm = np.array([float(x) for x in c]); lg = np.log10(dm); ed = np.r_[lg[0]-(lg[1]-lg[0])/2, (lg[1:]+lg[:-1])/2, lg[-1]+(lg[-1]-lg[-2])/2]
    n = np.nan_to_num(o[c].values.astype(float)) * np.diff(ed); to = o.iloc[:, 0].values.astype(float)
    sm = pd.Series(n[:, dm > 2.5].sum(1)).rolling(30, center=True, min_periods=5).mean().values
    ton = to[np.argmax(sm > 20)]
    ok = (rhs >= 100) & (t > 0); tsat = t[np.argmax(ok)] if ok.any() else np.nan
    return dict(t=t, rh=rhs, qt=qt, tsat=tsat, ton=ton, to=to, dm=dm, n=n)
def metrics(tag, e):
    p = T / tag / "single_comparison" / f"metrics-{e}.csv"
    if not p.exists(): return None
    return {r["metric"]: r["value"] for r in csv.DictReader(open(p))}
def model(tag, e):
    p = T / tag / "single_comparison" / f"output-{e}.nc"
    if not p.exists(): return None
    with Dataset(p) as nc:
        t = nc["time"][:].data; Tk = nc["t"][:].data; pp = nc["p"][:].data; rh = nc["rh"][:].data; ql = nc["ql"][:].data
        nd = nc["ndrop"][:].data; nw = nc["nwat"][:].data; dw = nc["dwet"][:].data
    e_ = rh * es(Tk - 273.15); qv = 0.622 * e_ / (pp - e_); rho = pp / (287 * Tk)
    return dict(t=t, rh=100 * rh, qt=qv + ql, nd=nd * rho / 1e6, nw=nw * rho[:, None, None] / 1e6, dw=dw * 1e6)
rows = []
for e in EX:
    O = obs(e)
    for tag in TAGS:
        M = model(tag, e); mt = metrics(tag, e)
        if M is None or mt is None: continue
        w = (float(mt["window_start_s"]), float(mt["window_end_s"]))
        k = np.argmax(M["rh"] >= 100); msat = M["t"][k] if M["rh"][k] >= 100 else np.nan
        pre = (O["t"] > 0) & (O["t"] < O["tsat"]) & np.isfinite(O["rh"])
        rms = np.sqrt(np.nanmean((np.interp(O["t"][pre], M["t"], M["rh"]) - O["rh"][pre])**2)) if pre.any() else np.nan
        # in-cloud total-water loss rate, WELAS onset + 60 s to window end
        a, b = O["ton"] + 60, w[1]
        so = (O["t"] >= a) & (O["t"] <= b) & np.isfinite(O["qt"]); sm = (M["t"] >= a) & (M["t"] <= b)
        r_obs = -np.polyfit(O["t"][so] / 60, O["qt"][so] * 1e3, 1)[0] if so.sum() > 5 else np.nan
        r_mod = -np.polyfit(M["t"][sm] / 60, M["qt"][sm] * 1e3, 1)[0] if sm.sum() > 3 else np.nan
        s = (M["t"] >= w[0]) & (M["t"] <= w[1]); so2 = (O["to"] >= w[0]) & (O["to"] <= w[1]); ob = O["n"][so2].mean(0)
        n25 = (M["nw"][s] * (M["dw"][s] > 2.5)).sum(axis=(1, 2)).mean() / ob[O["dm"] > 2.5].sum()
        n5 = (M["nw"][s] * (M["dw"][s] > 5)).sum(axis=(1, 2)).mean() / ob[O["dm"] > 5].sum()
        rq = float(mt["integrated_ql_model_kg_s_kg"]) / float(mt["integrated_ql_obs_kg_s_kg"])
        rows.append(dict(exp=e, tag=tag, t_sat_model_min=msat / 60, t_sat_hyg_min=O["tsat"] / 60, welas_onset_min=O["ton"] / 60,
                         sat_lag_min=(msat - O["tsat"]) / 60, rh_rms_pp=rms, Rq=rq, ql_nrmse=float(mt["ql_nrmse_absolute"]),
                         N_nrmse=float(mt["ndrop_psd_nrmse"]), Deff_nrmse=float(mt["deff_nrmse"]),
                         loss_obs=r_obs, loss_mod=r_mod, loss_ratio=r_mod / r_obs, N25_ratio=n25, N5_ratio=n5, act=M["nd"][s].mean()))
df = pd.DataFrame(rows); df.to_csv(OUT / "combined_check.csv", index=False)
pd.set_option("display.width", 250)
cols = ["sat_lag_min", "rh_rms_pp", "Rq", "ql_nrmse", "N_nrmse", "Deff_nrmse", "loss_ratio", "N25_ratio", "N5_ratio"]
print("Medians over the 12 fan-on runs (0.5 K/min):")
print(df[df.exp.isin(FANON)].groupby("tag")[cols].median().round(2).reindex([t for t in TAGS if t in set(df.tag)]).to_string())
for tag in [t for t in TAGS if t.startswith("comb")]:
    W = df[df.tag == tag]
    if not len(W): continue
    F = W[W.exp.isin(FANON)]
    a_ = (F.sat_lag_min.abs() <= 1).sum(); b_ = (F.rh_rms_pp <= 1).sum(); c_ = ((W.Rq >= 0.5) & (W.Rq <= 2)).sum()
    ok = a_ >= 9 and b_ >= 9 and c_ >= 12
    print(f"\n{tag}: (a) saturation within 1 min of hygrometer {a_}/{len(F)}; (b) pre-cloud RH RMS <= 1 pp {b_}/{len(F)}; (c) 0.5 <= R_q <= 2 {c_}/{len(W)} => {'CONSISTENT' if ok else 'not consistent'}")

# Offset refit under mode 3 (km = 3e-4, default wall phase): per run, the dT_BL shift (0 to -0.12 K) with the
# lowest liquid-water NRMSE, as for the published baseline.  Then the same decision rule.
FIT = ["comb_ice_km3.0e-4"] + [f"combfit_dT-{d}" for d in ["0.01", "0.02", "0.03", "0.05", "0.08", "0.12"]]
S = df[df.tag.isin(FIT)].copy()
if S.tag.nunique() == len(FIT):
    S["shift_K"] = S.tag.map(lambda t: 0.0 if t.startswith("comb_") else -float(t.split("-")[-1]))
    B = S.loc[S.groupby("exp").ql_nrmse.idxmin()].sort_values("exp")
    B.to_csv(OUT / "combined_fit_best.csv", index=False)
    print("\nPer-run refit (mode 3, km=3e-4):")
    print(B[["exp", "shift_K", "sat_lag_min", "rh_rms_pp", "Rq", "ql_nrmse", "N_nrmse", "Deff_nrmse", "N25_ratio", "N5_ratio", "loss_ratio"]].round(2).to_string(index=False))
    F = B[B.exp.isin(FANON)]
    print("fan-on medians:", F[cols].median().round(2).to_dict())
    a_ = (F.sat_lag_min.abs() <= 1).sum(); b_ = (F.rh_rms_pp <= 1).sum(); c_ = ((B.Rq >= 0.5) & (B.Rq <= 2)).sum()
    print(f"refit: (a) {a_}/{len(F)}; (b) {b_}/{len(F)}; (c) {c_}/{len(B)} => {'CONSISTENT' if (a_>=9 and b_>=9 and c_>=12) else 'not consistent'}")
