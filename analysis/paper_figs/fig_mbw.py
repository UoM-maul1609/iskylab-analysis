"""Fig. 9 (v1.9): the hygrometer measures the wall loss.  (a) Exp013 total water: hygrometer (MBW dew point), model with
the fan on (v19_final) and off (v19_fanoff, exchange time 80 s), and saturation at the gas and wall temperature.
(b) Rate at which total water falls once cloud has formed, every fan-on experiment including Exp017: hygrometer,
model fan on, model fan off.  Rates fitted from 1 min after the WELAS liquid water is continuously present until it ends.
Usage: python3 fig_mbw.py -> runs/analysis/fig_mbw.png and mbw_rates.csv"""
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")); A = T / "analysis"
exec(open(REPO / "analysis/wall_water_check/wall.py").read().split("rows=[]")[0])
EXPS = ["006", "007", "008", "009", "010", "011", "013", "014", "015", "016", "017", "018", "019"]
def model_qt(tag, e):
    p = T / tag / "single_comparison" / f"output-Exp{e}.nc"
    if not p.exists(): return None
    with Dataset(p) as nc:
        tm = nc["time"][:].data; Tm = nc["t"][:].data.ravel(); pm = nc["p"][:].data.ravel(); rh = nc["rh"][:].data.ravel(); ql = nc["ql"][:].data.ravel()
    return tm, rh * qs(Tm, pm) + ql
def window(t, lw):
    above = lw > 1e-5; n = 60 if above.sum() > 200 else 20
    i0 = next(i for i in range(len(t)) if above[i:i + n].all()); i1 = np.where(above)[0].max()
    return t[i0] + 60, t[i1]
def rate(t, q, a, b):
    w = np.isfinite(q) & (t >= a) & (t <= b); return -np.polyfit(t[w], q[w], 1)[0] * 60e3 if w.sum() > 5 else np.nan
rows = []
for e in EXPS:
    t, p, Tg, Tw, qm, lw = load(e); a, b = window(t, lw); r = dict(exp=e, obs=rate(t, qm, a, b), t0=a / 60, t1=b / 60)
    for k, tag in [("fan_on", "v19_final"), ("fan_off", "v19_fanoff")]:
        m = model_qt(tag, e); r[k] = rate(m[0], m[1], a, b) if m else np.nan
        if m:
            w_ = np.isfinite(qm) & (t >= a) & (t <= b); r["rms_" + k] = np.sqrt(np.mean((np.interp(t[w_], m[0], m[1]) - qm[w_])**2)) * 1e3
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(A / "mbw_rates.csv", index=False); print(df.round(3).to_string(index=False))
f = df[df.exp != "017"]; print("rms fan on %.3f-%.3f (median %.3f), fan off %.3f-%.3f (median %.3f); fan off worse in %d of %d" % (df.rms_fan_on.min(), df.rms_fan_on.max(), df.rms_fan_on.median(), df.rms_fan_off.min(), df.rms_fan_off.max(), df.rms_fan_off.median(), (df.rms_fan_off > df.rms_fan_on).sum(), len(df)))
print("0.5 K/min runs: obs %.3f-%.3f (median %.3f); fan on/obs %.2f-%.2f (median %.2f); fan off/obs median %.2f"
      % (f.obs.min(), f.obs.max(), f.obs.median(), (f.fan_on / f.obs).min(), (f.fan_on / f.obs).max(), (f.fan_on / f.obs).median(), (f.fan_off / f.obs).median()))
BLK, ON, OFF = "#222222", "#2166ac", "#e08214"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.25))
t, p, Tg, Tw, qm, lw = load("013"); tmin = t / 60
a1.plot(tmin, qm * 1e3, color=BLK, lw=2.2, label="total water, hygrometer")
for tag, col, ls, lab in [("v19_final", ON, "-", "total water, model, fan on"), ("v19_fanoff", OFF, "--", "total water, model, fan off")]:
    m = model_qt(tag, "013")
    if m: a1.plot(m[0] / 60, m[1] * 1e3, color=col, lw=2, ls=ls, label=lab)
a1.plot(tmin, qs(Tg, p) * 1e3, color="0.55", lw=1.2, label="saturation at gas temperature")
a1.plot(tmin, qs(Tw, p) * 1e3, color="0.55", lw=1.2, ls=":", label="saturation at wall temperature")
a, b = window(t, lw); a1.axvspan(a / 60, b / 60, color="0.93", zorder=0)
a1.set_xlim(0, 29); a1.set_xlabel("Time since start of expansion (min)"); a1.set_ylabel("Water mixing ratio (g kg$^{-1}$)")
a1.legend(frameon=False, fontsize=9, loc="lower left"); a1.spines[["top", "right"]].set_visible(False)
a1.set_title("(a) Exp013: total water follows the model with the fan on", loc="left", fontsize=11.5, weight="bold")
x = np.arange(len(df)); w = 0.38
a2.bar(x - w / 2, df.rms_fan_on, w, color=ON, label="model, fan on"); a2.bar(x + w / 2, df.rms_fan_off, w, color=OFF, label="model, fan off")
a2.set_xticks(x); a2.set_xticklabels(df.exp, fontsize=9); a2.set_xlabel("Experiment")
a2.set_ylabel("RMS difference from hygrometer (g kg$^{-1}$)"); a2.spines[["top", "right"]].set_visible(False)
a2.legend(frameon=False, fontsize=9.5, loc="upper left")
a2.set_title("(b) Total water in cloud: model minus hygrometer", loc="left", fontsize=11.5, weight="bold")
fig.savefig(A / "fig_mbw.png", dpi=200, bbox_inches="tight")
