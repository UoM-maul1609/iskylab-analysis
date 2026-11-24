"""Every number the warm paper quotes, from the v1.9 runs (after make_v19.sh has scored them).
python3 v19_numbers.py > runs/analysis/v19_numbers.txt"""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataAnalysis_new as DA, paper_tags as PT
T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")); A = T / "analysis"; D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"; DATA = D
EX = ["Exp005","Exp006","Exp007","Exp008","Exp009","Exp010","Exp011","Exp013","Exp014","Exp015","Exp016","Exp017","Exp018","Exp019"]
FAN = [e for e in EX if e not in ("Exp005", "Exp017")]; DD2 = DA.cfg.WELAS_INLET_DD2_UM2
vol = lambda d: np.pi / 6 * d**3 * 1000.0
pd.set_option("display.width", 220)
def met(tag, e):
    p = T / tag / "single_comparison" / f"metrics-{e}.csv"
    if not p.exists(): return None
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(p))}
    f = lambda k: float(m[k])
    return dict(ql=f("ql_nrmse_absolute"), Rq=f("integrated_ql_model_kg_s_kg") / f("integrated_ql_obs_kg_s_kg"), N=f("ndrop_psd_nrmse"),
                Deff=f("deff_nrmse"), t0=f("window_start_s") / 60, t1=f("window_end_s") / 60,
                tot=f("integrated_ql_model_total_kg_s_kg") / f("integrated_ql_obs_kg_s_kg"))
def M(tag): return pd.DataFrame({e: met(tag, e) for e in EX if met(tag, e)}).T
def summ(tag, lab=None):
    d = M(tag)
    if not len(d): print(f"{lab or tag:28s} (no metrics)"); return d
    print(f"{lab or tag:28s} med ql {d.ql.median():.2f} Rq {d.Rq.median():.2f} N {d.N.median():.2f} Deff {d.Deff.median():.2f} | Rq {d.Rq.min():.2f}-{d.Rq.max():.2f}, x2 {((d.Rq>=.5)&(d.Rq<=2)).sum()}/{len(d)}, within 31% {((d.Rq>=0.69)&(d.Rq<=1.31)).sum()}")
    return d
print(f"=== WELAS inlet dD2 = {DD2:g} um2")
S = pd.read_csv(A / "v19_settings.txt", sep=r"\s+", header=None, skiprows=1, names=["exp", "tsat", "dT"]).set_index("exp")
F = M(PT.FINAL); print("\n=== Table 2 (per run)"); print(pd.concat([S, F[["t0", "t1", "ql", "Rq", "N", "Deff"]]], axis=1).round(3).to_string())
print("\n=== Table 3 / Sect 4.2-4.5 medians")
for t in [PT.FINAL, PT.NOBL, PT.HOM, PT.P0, PT.CHENLAMB, PT.MOVINGCENTRE, PT.FIXED, "wallsink_km3.0e-4_in", "wallsink_km1.0e-3_in", "wallsink_km3.0e-3_in", "comb_ice_km3.0e-4_in"]: summ(t)
N_ = M(PT.NOBL); H, P = M(PT.HOM), M(PT.P0)
print(f"nobl fan-on Rq {N_.loc[FAN].Rq.min():.1f}-{N_.loc[FAN].Rq.max():.1f} (median {N_.loc[FAN].Rq.median():.1f}); Exp005 {N_.loc['Exp005'].Rq:.2f}; Exp013 {N_.loc['Exp013'].Rq:.1f}")
print(f"baseline better N than hom in {(F.N<H.N).sum()}, than p0 in {(F.N<P.N).sum()}; Deff better than hom in {(F.Deff<H.Deff).sum()}")
print(f"final: ql {F.ql.min():.2f}-{F.ql.max():.2f}, N {F.N.min():.2f}-{F.N.max():.2f}, chamber total liquid / WELAS (fan-on median) {F.loc[FAN].tot.median():.2f}")
for t, lab in [(PT.NOFALLOUT, "no sedimentation"), (PT.NOSCE, "no collisions")]:
    d = M(t)
    if len(d): ch = 100 * (d.Rq / F.Rq - 1); print(f"{lab}: change in integrated liquid water {ch.round(0).to_dict()}")
print("\n=== Onset (v19_onset.csv: chosen cloud-formation time)")
on = pd.read_csv(A / "v19_onset.csv"); b = on.loc[on.groupby("exp").lag_min.apply(lambda x: x.abs().idxmin())]
print(b.round(2).to_string(index=False)); print(f"lag model-WELAS onset: {b.lag_min.min():.2f} to {b.lag_min.max():.2f} min, median {b.lag_min.median():.2f}")
print("\n=== Inlet refit (v19_offset_all.csv)")
of = pd.read_csv(A / "v19_offset_all.csv"); bst = of.loc[of.groupby(["exp", "dD2"]).cost.idxmin()]; tot = bst.groupby("dD2").cost.sum()
B = bst[bst.dD2 == DD2].set_index("exp"); Q = ["lwc25", "n25", "n5"]
print(f"misfit at dD2=0 {tot.loc[0]:.2f}, 14 {tot.loc[14]:.2f}, best {tot.min():.2f} at {tot.idxmin():g}; within 10% of best: {tot[tot<=1.1*tot.min()].index.min():g}-{tot[tot<=1.1*tot.min()].index.max():g}")
print("median ratios", B[Q].median().round(2).to_dict(), "| all within x1.5:", ((B[Q] > 1/1.5) & (B[Q] < 1.5)).all(axis=1).sum(), "| within 20-25%:", ((B[Q] > 0.8) & (B[Q] < 1.25)).all(axis=1).sum())
print("per-run best dD2:", of.loc[of.groupby("exp").cost.idxmin()].set_index("exp").dD2.to_dict())
print("\n=== Budgets")
for e in ["Exp005", "Exp013"]:
    with Dataset(T / PT.FINAL / "single_comparison" / f"output-{e}.nc") as nc:
        rho = (nc["p"][:] / (287 * nc["t"][:])).data; di = DA.apply_welas_inlet({"dwet": nc["dwet"][:].data, "mwat": nc["mwat"][:].data}, DD2)["dwet"]
        q = (nc["nwat"][:].data * vol(di) * (di > 2.5e-6)).sum(axis=(1, 2))
        print(f"{e}: walls {nc['qchamber_bl'][-1]*1e3:.2f}, evaporated by wall air {nc['qchamber_bl_evap'][-1]*1e3:.2f}, sedimentation {nc['qfall_liq'][-1]*1e3:.2f} g/kg; peak liquid seen by WELAS {q.max()*1e3:.3f} g/kg")
os.environ["BUDGET_TAG"] = PT.FINAL
exec(open(REPO / "analysis/wall_water_check/wall.py").read().split("rows=[]")[0])
R_ = []
for e in ["006","007","008","009","010","011","013","014","015","016","018","019"]:
    t, p, Tg, Tw, qm, lw = load(e); ok = np.isfinite(qm); above = lw > 1e-5
    idx = next(i for i in range(len(t)) if above[i:i+60].all()); w = ok & (t > t[idx] + 60); ro = -np.polyfit(t[w], qm[w], 1)[0] * 60e3
    with Dataset(T / PT.FINAL / "single_comparison" / f"output-Exp{e}.nc") as nc:
        tm = nc["time"][:].data; rem = nc["qchamber_bl"][:].data + nc["qfall_liq"][:].data
    wm = (tm > t[idx] + 60) & (tm <= t[w].max()); rm = np.polyfit(tm[wm], rem[wm], 1)[0] * 60e3; R_.append((ro, rm, rm / ro))
R_ = np.array(R_); print(f"in-cloud loss: obs {R_[:,0].min():.3f}-{R_[:,0].max():.3f}, model {R_[:,1].min():.3f}-{R_[:,1].max():.3f}, ratio {R_[:,2].min():.2f}-{R_[:,2].max():.2f} (median {np.median(R_[:,2]):.2f})")
print("\n=== Without wall mixing: inlet shrinkage needed to match WELAS liquid water")
out = []
for e in FAN:
    m = met(PT.FINAL, e); a, b_ = m["t0"] * 60, m["t1"] * 60
    o = pd.read_csv(DATA / f"iSKYLAB01-{e if str(e).startswith('Exp') else 'Exp' + str(e)}-2-OPC-d-MergedW.csv", skipinitialspace=True); c = [x for x in o.columns[1:] if x.replace(".", "").isdigit()]
    dm = np.array([float(x) for x in c]) * 1e-6; n = np.nan_to_num(o[c].values.astype(float)) * 0.00984 * 1e6; to = o.iloc[:, 0].values; so = (to >= a) & (to <= b_)
    lwo = (n[so][:, dm > 2.5e-6] * vol(dm[dm > 2.5e-6])).sum(1).mean()
    with Dataset(T / PT.NOBL / "single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= a) & (t <= b_); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None]
        nw = nc["nwat"][:].data[s] * rho; dw = nc["dwet"][:].data[s]; mw = nc["mwat"][:].data[s]
    raw = (nw * vol(dw)).sum(axis=(1, 2)).mean()
    for dd in range(0, 401, 2):
        di = DA.apply_welas_inlet({"dwet": dw, "mwat": mw}, dd)["dwet"]
        if (nw * vol(di) * (di > 2.5e-6)).sum(axis=(1, 2)).mean() <= lwo: out.append((dd, 1 - lwo / raw)); break
o_ = np.array(out); print(f"dD2 needed {o_[:,0].min():.0f}-{o_[:,0].max():.0f} um2; WELAS would miss {o_[:,1].min():.2f}-{o_[:,1].max():.2f} of the liquid water")
print("\n=== Competition (early-cloud windows) - see also competition_v2*.csv from fig_competition_v2.py")
for f in ["competition_v2.csv", "competition_v2_sens.csv", "competition_v2_visibility.csv"]:
    if (A / f).exists(): print(f); print(pd.read_csv(A / f).round(1).to_string(index=False))
for e in ["Exp006", "Exp007", "Exp018", "Exp008", "Exp009", "Exp010"]:
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}; w = (float(m["window_start_s"]), float(m["window_end_s"]))
    v = {}
    for tag in [PT.BG, PT.FINAL, PT.BG_NOBL, PT.MIX_NOBL]:
        with Dataset(T / tag / "single_comparison" / f"output-{e}.nc") as nc:
            t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None] / 1e6
            nl = nc["nliq"][:].data[s] * rho; v[tag] = [nl[:, k].sum(-1).mean() for k in range(nl.shape[1])]
    print(f"{e}: activated, wall mixing: alone {v[PT.BG][0]:.0f}, mix {sum(v[PT.FINAL]):.0f} (on background {v[PT.FINAL][0]:.0f}, on NaCl {v[PT.FINAL][1]:.0f}); "
          f"no wall mixing: alone {v[PT.BG_NOBL][0]:.0f}, mix {sum(v[PT.MIX_NOBL]):.0f}")
