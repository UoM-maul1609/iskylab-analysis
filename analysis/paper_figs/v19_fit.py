"""v1.9 fit of the warm runs (see batchRuns_v2.sh, sets v19onset / v19offset / v19final).

  python3 v19_fit.py onset   after ./batchRuns_v2.sh v19onset
      For each run, the cloud-formation time (shifted 0-2.5 min earlier than the original) whose modelled liquid water,
      seen through the WELAS inlet, first exceeds 5 % of its peak closest to the time WELAS does.
      Writes runs/analysis/v19_tsat.txt (exp, cloud-formation time [min], start offset dT0 [K]) and v19_onset.csv.
  python3 v19_fit.py offset  after ./batchRuns_v2.sh v19offset
      Wall-air offset per run and one inlet shrinkage dD2 for all runs, fitted jointly over the whole cloud
      (onset to end; cost = sum over liquid water > 2.5 um, number > 2.5 um and > 5 um of [ln(model/WELAS)]^2).
      Writes runs/analysis/v19_settings.txt (line 'dD2 <value>', then exp tsat dT), v19_offset_all.csv and
      inlet_refit_all_v19.csv (for fig_inlet.py).
"""
import os, sys, re
from pathlib import Path
import numpy as np, pandas as pd
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "scripts"))
import dataAnalysis_new as DA
T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")); OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EX = ["Exp005","Exp006","Exp007","Exp008","Exp009","Exp010","Exp011","Exp013","Exp014","Exp015","Exp016","Exp017","Exp018","Exp019"]
vol = lambda d: np.pi / 6 * d**3 * 1000.0

def batch_tables():
    s = (REPO / "scripts/batchRuns_v2.sh").read_text()
    runs = re.search(r"read -r -d '' RUNS <<'TABLE'\n(.*?)\nTABLE", s, re.S).group(1)
    dt0 = re.search(r"read -r -d '' V19_DT0 <<'TABLE'\n(.*?)\nTABLE", s, re.S).group(1)
    R = {l.split()[0]: float(l.split()[1]) for l in runs.splitlines() if l.strip()}
    D0 = {l.split()[0]: float(l.split()[1]) for l in dt0.splitlines() if l.strip()}
    rel = [float(x) for x in re.search(r'V19_OFFSETS="([^"]*)"', s).group(1).split()]
    sh = [float(x) for x in re.search(r'V19_ONSET_SHIFTS="([^"]*)"', s).group(1).split()]
    return R, D0, rel, sh
def tag(x): return ("m%03d" % round(-x * 1000)) if x < 0 else ("p%03d" % round(x * 1000))

def welas(e):
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-d-MergedW.csv", skipinitialspace=True)
    c = [x for x in o.columns[1:] if x.replace(".", "").isdigit()]
    dm = np.array([float(x) for x in c]) * 1e-6; n = np.nan_to_num(o[c].values.astype(float)) * 0.00984 * 1e6
    return o.iloc[:, 0].values.astype(float), dm, n
def onset(t, x, f=0.05):
    x = pd.Series(x).rolling(30 if len(x) > 1000 else 3, center=True, min_periods=1).mean().values
    i = np.where(x > f * np.nanmax(x))[0]; return t[i[0]] if len(i) else np.nan
def model(tag_, e):
    p = T / tag_ / "single_comparison" / f"output-{e}.nc"
    if not p.exists(): return None
    with Dataset(p) as nc:
        return dict(t=nc["time"][:].data, rho=(nc["p"][:] / (287 * nc["t"][:])).data, nw=nc["nwat"][:].data,
                    dw=nc["dwet"][:].data, mw=nc["mwat"][:].data)
def seen(m, dd2, s=None):
    s = slice(None) if s is None else s
    di = DA.apply_welas_inlet({"dwet": m["dw"][s], "mwat": m["mw"][s]}, dd2)["dwet"]; nw = m["nw"][s] * m["rho"][s][:, None, None]
    return (nw * vol(di) * (di > 2.5e-6)).sum(axis=(1, 2)), (nw * (di > 2.5e-6)).sum(axis=(1, 2)), (nw * (di > 5e-6)).sum(axis=(1, 2))

def do_onset():
    R, D0, rel, shifts = batch_tables(); dd2 = DA.cfg.WELAS_INLET_DD2_UM2; rows = []
    for e in EX:
        to, dm, n = welas(e); big = dm > 2.5e-6; lo = (n[:, big] * vol(dm[big])).sum(1); t_obs = onset(to, lo)
        for s in shifts:
            m = model(f"v19_on_{tag(s)}", e)
            if m is None: continue
            lw, _, _ = seen(m, dd2); rows.append(dict(exp=e, shift_min=s, tsat=R[e] - s, welas_onset_min=t_obs / 60, model_onset_min=onset(m["t"], lw) / 60))
    if not rows: sys.exit("no model output found in runs/v19_on_*/single_comparison - check scripts/failed_runs.txt and scripts/logs/")
    df = pd.DataFrame(rows); df["lag_min"] = df.model_onset_min - df.welas_onset_min; df.to_csv(OUT / "v19_onset.csv", index=False)
    best = df.loc[df.groupby("exp").lag_min.apply(lambda x: x.abs().idxmin())].set_index("exp")
    with open(OUT / "v19_tsat.txt", "w") as f:
        for e in EX: f.write(f"{e} {best.loc[e, 'tsat']:g} {D0[e]:g}\n")
    print(best[["shift_min", "tsat", "welas_onset_min", "model_onset_min", "lag_min"]].round(2).to_string())
    print("written", OUT / "v19_tsat.txt", "- next: ./batchRuns_v2.sh v19offset")

def do_offset():
    R, D0, rel, _ = batch_tables(); ts = pd.read_csv(OUT / "v19_tsat.txt", sep=r"\s+", header=None, names=["exp", "tsat", "dT0"]).set_index("exp")
    DDS = np.arange(0, 41, 1.0); rows = []
    for e in EX:
        a, b = DA._full_experiment_window(e)
        to, dm, n = welas(e); so = (to >= a) & (to <= b); big = dm > 2.5e-6
        O = dict(lw=(n[so][:, big] * vol(dm[big])).sum(1).mean(), n25=n[so][:, big].sum(1).mean(), n5=n[so][:, dm > 5e-6].sum(1).mean())
        for r in rel:
            m = model(f"v19_off_{tag(r)}", e)
            if m is None: continue
            s = (m["t"] >= a) & (m["t"] <= b)
            for dd in DDS:
                lw, n25, n5 = seen(m, dd, s)
                q = dict(lwc25=lw.mean() / O["lw"], n25=n25.mean() / O["n25"], n5=n5.mean() / O["n5"])
                rows.append(dict(exp=e, rel=r, offset=round(ts.loc[e, "dT0"] + r, 4), dD2=dd, **q,
                                 cost=sum(np.log(max(v, 1e-3))**2 for v in q.values())))
    if not rows: sys.exit("no model output found in runs/v19_off_*/single_comparison - check scripts/failed_runs.txt and scripts/logs/")
    df = pd.DataFrame(rows); df.to_csv(OUT / "v19_offset_all.csv", index=False)
    df.rename(columns={"rel": "offset_rel"}).assign(offset=lambda x: x.offset_rel).to_csv(OUT / "inlet_refit_all_v19.csv", index=False)
    best = df.loc[df.groupby(["exp", "dD2"]).cost.idxmin()]; tot = best.groupby("dD2").cost.sum(); ddb = float(tot.idxmin())
    B = best[best.dD2 == ddb].set_index("exp")
    with open(OUT / "v19_settings.txt", "w") as f:
        f.write(f"dD2 {ddb:g}\n")
        for e in EX: f.write(f"{e} {ts.loc[e, 'tsat']:g} {B.loc[e, 'offset']:g}\n")
    print(f"best common inlet shrinkage dD2 = {ddb:g} um2; misfit by dD2:"); print(tot.round(2).to_string())
    print(B[["offset", "rel", "lwc25", "n25", "n5"]].round(3).to_string())
    edge = B.index[(B.rel <= min(rel)) | (B.rel >= max(rel))].tolist()
    if edge: print("WARNING: offset at the edge of the scanned range for", edge, "- widen V19_OFFSETS for these runs")
    print("written", OUT / "v19_settings.txt", "- next: ./batchRuns_v2.sh v19final")

if __name__ == "__main__":
    {"onset": do_onset, "offset": do_offset}[sys.argv[1]]()
