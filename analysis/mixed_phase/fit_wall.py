"""Fit a wall setting for the cold runs against the MBW total water first, then
check droplets and liquid water (WELAS operator).

For each run and tag:
  qt_rmse : RMS of (model total water - MBW total water) / MBW, over t = 0 .. end
            (model total water = vapour + liquid + ice in the air)
  qt_end  : model / MBW total water at the end of the expansion
  Smax    : maximum supersaturation (%)
  Nd/obs, LWC/obs : window means t_sat .. t_sat + 600 s (WELAS operator)

Usage:
  python3 fit_wall.py                       # km scan (mp_km01 ... mp_km10)
  python3 fit_wall.py mp_off_p000 mp_km02   # any list of tags
"""
import csv
import sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TAGS = sys.argv[1:] or ["mp_km01", "mp_km02", "mp_km05", "mp_km10"]
_argv, sys.argv = sys.argv, sys.argv[:1]   # compare_mp reads argv[1] as RUNS_DIR
import compare_mp as cmp  # noqa: E402
sys.argv = _argv
TS = HERE.parents[1] / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"


def ew(tc):
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _f(s):
    try:
        x = float(s)
        return x if np.isfinite(x) else np.nan
    except ValueError:
        return np.nan


def mbw(exp):
    rows = list(csv.reader(open(TS / f"iSKYLAB01-{exp}-1-MeteoCPC.csv")))
    h = [x.strip() for x in rows[0]]
    c = lambda p: [i for i, x in enumerate(h) if x.startswith(p)][0]
    a = np.array([[_f(r[0]), _f(r[c("Pressure")]), _f(r[c("TDew")])] for r in rows[1:]])
    p = a[:, 1] * 100.0
    e = ew(a[:, 2])
    return a[:, 0], 0.622 * e / (p - e)


def model_qt(tag, exp):
    f = cmp.RUNS / tag / "single_comparison" / f"output-{exp}.nc"
    if not f.exists():
        return None
    v = Dataset(f).variables
    tc = v["t"][:] - 273.15
    p = v["p"][:]
    e = v["rh"][:] * ew(tc)
    qv = 0.622 * e / (p - e)
    return np.asarray(v["time"][:]), np.asarray(qv + v["ql"][:] + v["qi"][:])


def main():
    summary = {}
    for exp in cmp.TSAT:
        o = cmp.obs(exp)
        td, _ = cmp.windows(o, exp)
        sel = np.isin(o["t"], td)
        ond, olwc = np.nanmean(o["nd"][sel]), np.nanmean(o["lwc"][sel])
        # WELAS also counts interstitial aerosol > 0.39 um: subtract the
        # pre-cloud level (t_sat - 4 .. - 1 min) to get cloud droplets only.
        pre = (o["t"] > cmp.TSAT[exp] - 240) & (o["t"] < cmp.TSAT[exp] - 60)
        ond = ond - np.nanmean(o["nd"][pre])
        tm_, qm = mbw(exp)
        print(f"== {exp}: obs Nd above pre-cloud {ond:.0f} cm-3, LWC {olwc:.3f} g m-3, MBW total water {qm[np.argmin(abs(tm_))]*1e3:.2f} -> {np.nanmean(qm[-60:])*1e3:.2f} g/kg")
        print(f"   {'tag':14s} {'qt_rmse':>7s} {'qt_end':>6s} {'Smax%':>6s} {'Nd/obs':>7s} {'LWC/obs':>8s}")
        for tag in TAGS:
            q = model_qt(tag, exp)
            m = cmp.model(tag, exp)
            if q is None or m is None:
                continue
            t, qt = q
            s = (tm_ >= 0) & (tm_ <= t[-1]) & np.isfinite(qm)
            r = np.interp(tm_[s], t, qt) / qm[s] - 1.0
            rmse = float(np.sqrt(np.mean(r ** 2)))
            end = float(np.interp(tm_[s][-1], t, qt) / qm[s][-1])
            nd = float(np.mean(np.interp(td, m["t"], m["nd"])))
            lwc = cmp.wmean(m, "lwc", 0, 0, td)
            print(f"   {tag:14s} {rmse:7.3f} {end:6.2f} {100*m['smax']:6.2f} {nd/ond:7.2f} {lwc/olwc:8.1f}")
            summary.setdefault(exp, []).append((rmse, tag, nd / ond, lwc / olwc))
    print("\nBest tag per run by MBW total water:")
    for exp, rows in summary.items():
        rmse, tag, r, l = min(rows)
        print(f"   {exp}: {tag:14s} qt_rmse {rmse:.3f}  Nd/obs {r:.2f}  LWC/obs {l:.1f}")


if __name__ == "__main__":
    main()
