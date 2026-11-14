"""Compare mixed-phase BMM runs with WELAS (KIT merged file: NDrop 0.3-20 um,
NIce > 20 um, LWC) and compute the simulated competition effect from pairs.

Windows (relative to the model saturation time t_sat):
  droplets: t_sat .. t_sat + 600 s
  ice:      t_sat + 300 s .. end of expansion
Means are plain time means of 1 s WELAS data and model output interpolated to
the WELAS times.  WELAS observation operator (warm paper): droplets shrink in the
inlet by dD^2 = 14 um^2; model "NDrop" counts liquid with sqrt(D^2 - 14 um^2) in
0.3-20 um; model "NIce" counts every particle > 20 um (ice by Dmax, liquid by D),
as KIT's size threshold does.  Sublimation of ice in the inlet is ignored.
The BMM ndrop diagnostic is not used (it returns 0 in the NaCl-rich runs).

Usage: python3 compare_mp.py [RUNS_DIR]
"""
import csv, sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve()
DATA = HERE.parents[2] / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[3] / "iSKYLAB01/runs"
TSAT = {"Exp021": 720, "Exp022": 840, "Exp023": 780, "Exp025": 600, "Exp026": 720, "Exp027": 840}
MIX = ["Exp022", "Exp023", "Exp026", "Exp027"]


def _f(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except ValueError:
        return np.nan


def obs(exp):
    rows = list(csv.reader(open(DATA / f"iSKYLAB01-{exp}-2-OPC-d-MergedW.csv")))
    h = [x.strip() for x in rows[0]]
    col = lambda p: [i for i, x in enumerate(h) if x.startswith(p)][0]
    a = np.array([[_f(r[0]), _f(r[col("NDrop")]), _f(r[col("NIce")]), _f(r[col("LWC")])] for r in rows[1:]])
    return {"t": a[:, 0], "nd": a[:, 1], "ni": a[:, 2], "lwc": a[:, 3]}


DD2 = 14e-12   # m2, fitted WELAS inlet shrinkage (warm paper)


def model(tag, exp):
    f = RUNS / tag / "single_comparison" / f"output-{exp}.nc"
    if not f.exists():
        return None
    v = Dataset(f).variables
    rhoa = np.asarray(v["p"][:] / (287.05 * v["t"][:]))
    dw, nw = np.asarray(v["dwet"][:]), np.asarray(v["nwat"][:])
    di, ni = np.asarray(v["dmaxice"][:]), np.asarray(v["nicem"][:])
    ax = tuple(range(1, dw.ndim))
    dobs = np.sqrt(np.clip(dw ** 2 - DD2, 0.0, None))
    nd = np.sum(nw * ((dobs > 0.3e-6) & (dobs <= 20e-6)), axis=ax)
    nbig = np.sum(nw * (dw > 20e-6) * (dw < 1e-2), axis=ax) + np.sum(ni * (di > 20e-6) * (di < 1e-2), axis=ax)
    return {"t": np.asarray(v["time"][:]),
            "nd": nd * rhoa / 1e6,
            "ni": nbig * rhoa / 1e6,
            "nice_all": np.asarray(v["nice"][:]) * rhoa / 1e6,
            "lwc": np.asarray(v["ql"][:]) * rhoa * 1e3,
            "smax": float(np.max(v["rh"][:]) - 1.0)}


def wmean(d, key, t0, t1, tgrid=None):
    if tgrid is None:
        s = (d["t"] >= t0) & (d["t"] <= t1)
        return float(np.nanmean(d[key][s]))
    return float(np.nanmean(np.interp(tgrid, d["t"], d[key])))


def windows(o, exp):
    ts = TSAT[exp]; tend = np.nanmax(o["t"])
    sd = (o["t"] >= ts) & (o["t"] <= ts + 600) & np.isfinite(o["nd"])
    si = (o["t"] >= ts + 300) & (o["t"] <= tend) & np.isfinite(o["ni"])
    return o["t"][sd], o["t"][si]


def summary(tags, exps):
    print(f"{'run':10s} {'exp':7s} {'Smax%':>6s} {'Nd mod/obs':>16s} {'ratio':>6s} {'Ni mod/obs':>14s} {'ratio':>6s} {'LWC mod/obs':>14s}")
    for exp in exps:
        o = obs(exp); td, ti = windows(o, exp)
        ond, oni, olwc = np.nanmean(o["nd"][np.isin(o["t"], td)]), np.nanmean(o["ni"][np.isin(o["t"], ti)]), np.nanmean(o["lwc"][np.isin(o["t"], td)])
        for tag in tags:
            m = model(tag, exp)
            if m is None:
                continue
            nd, ni, lwc = wmean(m, "nd", 0, 0, td), wmean(m, "ni", 0, 0, ti), wmean(m, "lwc", 0, 0, td)
            print(f"{tag:10s} {exp:7s} {100*m['smax']:6.2f} {nd:7.1f}/{ond:7.1f} {nd/ond:6.2f} {ni:6.3f}/{oni:6.3f} {ni/oni:6.2f} {lwc:6.3f}/{olwc:6.3f}")
        print()


def competition(base="mp_base", dust="mp_dust"):
    print(f"[{base} vs {dust}]")
    print("Simulated competition (paired runs, same dust): change in total ice number from adding NaCl\n(Nd here = WELAS-operator droplets; Ni = all model ice)")
    print(f"{'exp':7s} {'Nd dust-only':>12s} {'Nd mix':>8s} {'Ni dust-only':>12s} {'Ni mix':>8s} {'dNi %':>7s}")
    for exp in MIX:
        o = obs(exp); td, ti = windows(o, exp)
        a, b = model(dust, exp), model(base, exp)
        if a is None or b is None:
            continue
        nia, nib = wmean(a, "nice_all", 0, 0, ti), wmean(b, "nice_all", 0, 0, ti)
        print(f"{exp:7s} {wmean(a,'nd',0,0,td):12.1f} {wmean(b,'nd',0,0,td):8.1f} {nia:12.3f} {nib:8.3f} {100*(nib/nia-1):+7.1f}")


if __name__ == "__main__":
    summary(["mp_base", "mp_aT1", "mp_nobl", "mp_wall2", "mp_wet1", "mp_tau20", "mp_base_cap"], ["Exp021", "Exp025"])
    summary(["mp_base"], ["Exp022", "Exp023", "Exp026", "Exp027"])
    summary(["mp_base", "mp_nf"], ["Exp021", "Exp025"])
    summary(["mp_a033"], ["Exp021", "Exp022", "Exp023", "Exp025", "Exp026", "Exp027"])
    competition()
    competition("mp_wet1", "mp_wet1dust")
    for a, b in (("mp_wet1nf", "mp_wet1nfdust"), ("mp_x001", "mp_x001dust"), ("mp_x1", "mp_x1dust")):
        competition(a, b)
