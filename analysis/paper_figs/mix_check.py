"""Mixed-evaporation test (batchRuns_v2.sh v19mix / v19mixall): scores against WELAS for the smallest-first baseline
(v19_final), homogeneous (v19_hom) and mixed runs, plus Exp013 relative dispersion early (13-15 min) and late (20-25 min).
Usage: python3 mix_check.py"""
import os, sys, csv, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
os.environ.setdefault("ISKYLAB_INLET_DD2", (T / "analysis/v19_settings.txt").read_text().split()[1])
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, pandas as pd
from netCDF4 import Dataset
import dataAnalysis_new as DA
DD2 = DA.cfg.WELAS_INLET_DD2_UM2
def met(tag, e):
    p = T / tag / "single_comparison" / f"metrics-{e}.csv"
    if not p.exists(): return None
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(p))}
    return dict(ql=float(m["ql_above_min_nrmse_absolute"]), Rq=float(m["integrated_ql_model_kg_s_kg"]) / float(m["integrated_ql_obs_kg_s_kg"]),
                N=float(m["ndrop_psd_nrmse"]), Deff=float(m["deff_nrmse"]), RD=float(m["rel_disp_nrmse"]))
def rd(tag, win):
    with Dataset(T / tag / "single_comparison/output-Exp013.nc") as nc:
        t = nc["time"][:].data / 60; s = (t >= win[0]) & (t <= win[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None]
        n = (nc["nwat"][:].data[s] * rho).reshape(s.sum(), -1)
        d = DA.apply_welas_inlet({"dwet": nc["dwet"][:].data[s], "mwat": nc["mwat"][:].data[s]}, DD2)["dwet"].reshape(s.sum(), -1)
    m = d > 2.5e-6; n0 = (n * m).sum(-1); mu = (n * d * m).sum(-1) / n0; v = (n * d * d * m).sum(-1) / n0 - mu**2
    return np.nanmean(np.sqrt(np.maximum(v, 0)) / mu), np.nanmean(mu) * 1e6
tags = ["v19_final", "v19_hom"] + sorted(Path(p).name for p in glob.glob(str(T / "v19_mixf*"))) + sorted(Path(p).name for p in glob.glob(str(T / "v19_mixall*")))
print("Exp013 (WELAS: rd 0.30 early, 0.26 late; mean D late 12.0 um)")
for tag in tags:
    if not (T / tag / "single_comparison/output-Exp013.nc").exists(): continue
    m = met(tag, "Exp013"); (a, _), (b, mu) = rd(tag, (13, 15)), rd(tag, (20, 25))
    print(f"  {tag:16s} rd early {a:.2f} late {b:.2f}, mean D late {mu:.1f} | ql {m['ql']:.2f} Rq {m['Rq']:.2f} N {m['N']:.2f} Deff {m['Deff']:.2f} RD {m['RD']:.2f}")
EX = ["Exp005", "Exp006", "Exp007", "Exp008", "Exp009", "Exp010", "Exp011", "Exp013", "Exp014", "Exp015", "Exp016", "Exp017", "Exp018", "Exp019"]
print("medians over the experiments available")
for tag in tags:
    M = [met(tag, e) for e in EX]; M = [m for m in M if m]
    if len(M) < 5: continue
    d = pd.DataFrame(M); print(f"  {tag:16s} n={len(M):2d} ql {d.ql.median():.2f} Rq {d.Rq.median():.2f} ({d.Rq.min():.2f}-{d.Rq.max():.2f}) N {d.N.median():.2f} Deff {d.Deff.median():.2f} RD {d.RD.median():.2f}")
