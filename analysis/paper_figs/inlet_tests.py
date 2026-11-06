"""Tests of the WELAS-inlet hypothesis: time dependence, two sensors, and the size-distribution shape."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EX = ["Exp005","Exp006","Exp007","Exp008","Exp009","Exp010","Exp011","Exp013","Exp014","Exp015","Exp016","Exp017","Exp018","Exp019"]
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}; return float(m["window_start_s"]), float(m["window_end_s"])
def welas(e, s="d-MergedW"):
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-{s}.csv", skipinitialspace=True); cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]); f = o[cols].values.astype(float); f = np.nan_to_num(f)
    lg = np.log10(dm); e_ = np.r_[lg[0]-(lg[1]-lg[0])/2, (lg[1:]+lg[:-1])/2, lg[-1]+(lg[-1]-lg[-2])/2]
    return o.iloc[:, 0].values.astype(float), dm, f, np.diff(e_), e_   # dN/dlogD
def model(e):
    with Dataset(T / "baseline/single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; nw = nc["nwat"][:].data; dw = nc["dwet"][:].data; rho = (nc["p"][:] / (287 * nc["t"][:])).data; tg = nc["t"][:].data
    return t, nw * rho[:, None, None] / 1e6, dw * 1e6, tg   # cm-3 per bin, um
def nabove(nw, dw, th, dd2=0.0):
    d = np.sqrt(np.maximum(dw**2 - dd2, 0)); return (nw * (d > th)).sum(axis=(1, 2))
def psd(nw, dw, edges, dd2):
    d = np.sqrt(np.maximum(dw**2 - dd2, 0)); out = np.zeros(len(edges)-1)
    ok = d > 0; h, _ = np.histogram(np.log10(d[ok]), bins=edges, weights=nw[ok]); return h
rows = []; ts_rows = []
for e in EX:
    w = win(e); tm, nw, dw, tg = model(e)
    to, dm, f, dlg, edges = welas(e)
    nobs_t = (f[:, dm > 2.5] * dlg[dm > 2.5]).sum(1)
    nobs_s = pd.Series(nobs_t).rolling(30, center=True, min_periods=5).mean().values   # 30 s mean (1 Hz data)
    # 1) time dependence of the offset through the whole cloud
    for i, t in enumerate(tm):
        no = np.interp(t, to, nobs_s)
        if no < 30 or not np.isfinite(no): continue
        ths = np.linspace(1, 15, 141); v = np.array([(nw[i] * (dw[i] > x)).sum() for x in ths])
        if v[0] < no: continue
        ds = float(np.interp(-no, -v, ths)); ts_rows.append(dict(exp=e, t=t, tgas=tg[i]-273.15, nobs=no, dstar=ds, dd2=ds**2 - 6.25))
    # 2) two sensors separately (window means)
    s = (tm >= w[0]) & (tm <= w[1]); r = dict(exp=e)
    for lab, sens in [("w1", "a-Welas01"), ("w2", "b-Welas02"), ("m", "d-MergedW")]:
        to2, dm2, f2, dl2, _ = welas(e, sens); so = (to2 >= w[0]) & (to2 <= w[1])
        no = (f2[so][:, dm2 > 2.5] * dl2[dm2 > 2.5]).sum(1).mean()
        ths = np.linspace(1, 15, 281); v = np.array([nabove(nw[s], dw[s], x).mean() for x in ths])
        r["N_" + lab] = no; r["dstar_" + lab] = float(np.interp(-no, -v, ths))
    # 3) shape: fit dd2 to the window-mean PSD above 2.5 um (log misfit), and compare a constant shrink factor
    so = (to >= w[0]) & (to <= w[1]); obs = (f[so] * dlg).mean(0); sel = (dm > 2.5) & (dm < 20) & (obs > 0.05)
    def misfit(mod): m = mod[sel]; return np.sqrt(np.mean((np.log10(np.maximum(m, 1e-3)) - np.log10(obs[sel]))**2))
    mpsd = lambda dd2: np.mean([psd(nw[i], dw[i], edges, dd2) for i in np.where(s)[0]], axis=0)
    grid = np.arange(0, 40.1, 1.0); mf = [misfit(mpsd(g)) for g in grid]
    r["dd2_shape"] = float(grid[int(np.argmin(mf))]); r["misfit_raw"] = mf[0]; r["misfit_best"] = min(mf); r["misfit_12"] = mf[12]
    def mfac(fa):
        h = np.mean([np.histogram(np.log10(np.maximum(dw[i]*fa, 1e-6)).ravel(), bins=edges, weights=nw[i].ravel())[0] for i in np.where(s)[0]], axis=0); return misfit(h)
    fg = np.arange(0.4, 1.01, 0.02); mfa = [mfac(x) for x in fg]; r["fac_shape"] = float(fg[int(np.argmin(mfa))]); r["misfit_fac"] = min(mfa)
    # mode diameter (by number, D>2.5) obs / model raw / model dd2=12
    def mode(h): hh = h.copy(); hh[dm <= 2.5] = 0; return float(dm[np.argmax(hh)])
    r["mode_obs"] = mode(obs); r["mode_raw"] = mode(mpsd(0)); r["mode_12"] = mode(mpsd(12))
    r["dd2_count"] = r["dstar_m"]**2 - 6.25
    rows.append(r); print(e, {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(OUT / "inlet_tests.csv", index=False); pd.DataFrame(ts_rows).to_csv(OUT / "inlet_dstar_time.csv", index=False)
