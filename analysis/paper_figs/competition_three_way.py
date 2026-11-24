"""Each mixture simulated three ways with the same chamber settings (v1.9): both aerosols (v19_final), background only
(v19_bg) and NaCl only (v19_nacl).  Misfit against WELAS over the whole cloud (window from v19_final metrics):
sum of ln(model/WELAS)^2 for droplets > 2.5 um, > 5 um and liquid water > 2.5 um, model seen through the inlet.
Usage: python3 competition_three_way.py -> runs/analysis/competition_three_way.csv"""
import os, sys, csv
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
st = T / "analysis/v19_settings.txt"
if "ISKYLAB_INLET_DD2" not in os.environ and st.exists(): os.environ["ISKYLAB_INLET_DD2"] = st.read_text().split()[1]
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, pandas as pd
from netCDF4 import Dataset
import dataAnalysis_new as DA
D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"; DD2 = DA.cfg.WELAS_INLET_DD2_UM2
vol = lambda d: np.pi / 6 * d**3 * 1000
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "v19_final/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def mod(tag, e, w):
    with Dataset(T / tag / "single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None]
        nw = nc["nwat"][:].data[s] * rho; di = DA.apply_welas_inlet({"dwet": nc["dwet"][:].data[s], "mwat": nc["mwat"][:].data[s]}, DD2)["dwet"]
    return np.array([(nw * (di > 2.5e-6)).sum(axis=(1, 2)).mean() / 1e6, (nw * (di > 5e-6)).sum(axis=(1, 2)).mean() / 1e6,
                     (nw * vol(di) * (di > 2.5e-6)).sum(axis=(1, 2)).mean() * 1e3])
def obs(e, w):
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-d-MergedW.csv", skipinitialspace=True); c = [x for x in o.columns[1:] if x.replace(".", "").isdigit()]
    dm = np.array([float(x) for x in c]); n = np.nan_to_num(o[c].values.astype(float)) * 0.00984; t = o.iloc[:, 0].values
    s = (t >= w[0]) & (t <= w[1]); big = dm > 2.5
    return np.array([n[s][:, big].sum(1).mean(), n[s][:, dm > 5].sum(1).mean(), (n[s][:, big] * 1e6 * vol(dm[big] * 1e-6)).sum(1).mean() * 1e3])
rows = []
for e in ["Exp006", "Exp007", "Exp018", "Exp019", "Exp008", "Exp009", "Exp010"]:
    w = win(e); O = obs(e, w)
    for lab, tag in [("both", "v19_final"), ("no NaCl", "v19_bg"), ("NaCl only", "v19_nacl")]:
        M = np.maximum(mod(tag, e, w), 1e-3); r = 100 * (M / O - 1)
        rows.append(dict(exp=e, run=lab, N25=r[0], N5=r[1], LWC=r[2], misfit=float(np.sum(np.log(M / O)**2))))
df = pd.DataFrame(rows); df.to_csv(T / "analysis/competition_three_way.csv", index=False); print(df.round(3).to_string(index=False))
