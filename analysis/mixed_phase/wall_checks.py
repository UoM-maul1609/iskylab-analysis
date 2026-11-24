"""Checks 1-3 of the warm-paper feedback for the cold runs (17 Nov 2026).
Runs: final wk3c mode-1 configuration of Exp021 (+0.05 K), Exp025 (+0.15 K), Exp027 (0.00 K),
rerun with a Linux BMM build at km = 3e-4 (base), 0 (k0) and 9e-3 m/s (k30), all else fixed."""
import os, sys
os.environ["ISKYLAB_TSAT"] = "hygrometer"
sys.argv = [sys.argv[0], "/home/claude/chk/runs"]
import numpy as np
from netCDF4 import Dataset
import paper_figs as P

EX = ["Exp021", "Exp025", "Exp027"]
TAGS = [("k0", "km = 0"), ("base", "km = 3e-4"), ("k30", "km = 9e-3")]

def es_w(T): return P.ew(T)
def es_i(T): return P.ei(T)

def qt_obs(o):
    """Total-water mixing ratio from the MBW dew point (kg/kg)."""
    e = P.ew(o["tdew"] + 273.15)
    return 0.622 * e / (o["p"] * 100 - e)

def raw(tag, e):
    v = Dataset(f"/home/claude/chk/out-{tag}-{e}.nc").variables
    g = lambda k: np.asarray(v[k][:])
    return {k: g(k) for k in ("time", "t", "p", "rh", "ql", "qi", "chamber_wall_vapour_flux", "chamber_wall_rh",
                              "qchamber_bl", "qfall_liq", "qfall_ice", "qwall_liq", "qwall_ice",
                              "qchamber_wall_ice_dep", "qchamber_wall_ice_subl")}

out = []
for e in EX:
    o = P.obs(e)
    ts = P.TSAT[e]
    tgl = None
    s = P.smooth(o["lwc"], 61); after = o["t"] >= ts - 120
    imax = int(np.nanargmax(np.where(after, s, -np.inf)))
    later = np.where((np.arange(len(s)) > imax) & (s < 0.1 * s[imax]))[0]
    tgl = float(o["t"][later[0]])
    pre = (o["t"] >= 60) & (o["t"] <= ts)
    T = o["tg"] + 273.15; Tw = o["tw"] + 273.15
    e_air = P.ew(o["tdew"] + 273.15)                     # total-water vapour pressure (pre-cloud = vapour)
    wall_excess = 100 * (e_air / P.ei(Tw) - 1)
    print(f"\n===== {e}: t_sat {ts} s, observed end of liquid {tgl:.0f} s ({np.interp(tgl, o['t'], o['tg']):.1f} C)")
    print(f"Check 3: wall T {np.nanmax(o['tw'][o['t']>=0]):.1f} to {np.nanmin(o['tw']):.1f} C (always below 0 C -> ice wall); "
          f"wall - gas: pre-cloud {np.nanmean((o['tw']-o['tg'])[pre]):+.2f} K, in cloud {np.nanmean((o['tw']-o['tg'])[(o['t']>=ts)&(o['t']<=tgl)]):+.2f} K")
    print(f"  observed vapour over the ice wall before the cloud: excess {np.nanpercentile(wall_excess[pre],5):+.0f} to {np.nanpercentile(wall_excess[pre],95):+.0f} % "
          f"(at t=60 s {np.interp(60,o['t'],wall_excess):+.0f} %, at t_sat {np.interp(ts,o['t'],wall_excess):+.0f} %)")
    qo = qt_obs(o)
    q0 = float(np.interp(0, o["t"], P.smooth(qo, 31)))
    print(f"  observed q_t: change 0 -> t_sat {100*(np.interp(ts,o['t'],P.smooth(qo,31))/q0-1):+.1f} %, "
          f"t_sat -> end of liquid {100*(np.interp(tgl,o['t'],P.smooth(qo,31))/np.interp(ts,o['t'],P.smooth(qo,31))-1):+.1f} %")
    for tag, lab in TAGS:
        m = P.model(tag, e); r = raw(tag, e)
        rhw = np.interp(o["t"][pre], m["t"], m["rhw_tot"])
        rms = float(np.sqrt(np.nanmean((rhw - o["rhw"][pre]) ** 2)))
        k = np.where(m["rhw_tot"] >= 100)[0]; tsm = float(m["t"][k[0]]) if len(k) else np.nan
        qtm = r["rh"] * 0 + 0
        # model q_t = qv + ql + qi ; qv from rh over water
        es = P.ew(r["t"]); qv = 0.622 * r["rh"] * es / (r["p"] - r["rh"] * es)
        qtm = qv + r["ql"] + r["qi"]
        qtm0 = qtm[0]
        dq_pre = 100 * (np.interp(ts, r["time"], qtm) / qtm0 - 1)
        dq_cld = 100 * (np.interp(tgl, r["time"], qtm) / np.interp(ts, r["time"], qtm) - 1)
        prew = r["time"] <= ts
        J = r["chamber_wall_vapour_flux"]
        # budget of the in-cloud loss (t_sat -> end of liquid), as % of q_t at t_sat
        qs = np.interp(ts, r["time"], qtm)
        d = lambda key: 100 * (np.interp(tgl, r["time"], r[key]) - np.interp(ts, r["time"], r[key])) / qs
        wall = d("qchamber_wall_ice_dep") - d("qchamber_wall_ice_subl")
        fall = d("qfall_liq") + d("qfall_ice")
        dep = d("qwall_liq") + d("qwall_ice")
        welas = (o["t"] >= ts) & (o["t"] <= ts + 600)
        lw_m = P.mmean(m, "lwc_welas", o["t"][welas]); lw_o = float(np.nanmean(o["lwc"][welas]))
        rhwall_pre = r["chamber_wall_rh"][prew & (r["time"] >= 60)]
        print(f"  [{lab:10s}] pre-cloud RMS {rms:.2f} pp; TW at t_sat {np.interp(ts, m['t'], m['rhw_tot']):.1f} % (obs {np.interp(ts,o['t'],o['rhw']):.1f}); "
              f"model saturation {tsm:.0f} s; q_t 0->t_sat {dq_pre:+.1f} %; RH over wall pre-cloud {rhwall_pre.min():.2f}-{rhwall_pre.max():.2f}; "
              f"J pre-cloud mean {np.mean(J[prew & (r['time']>=60)])*1e6:+.2f} mg m-2 s-1")
        print(f"  {'':12s} in cloud (t_sat -> end of liquid): q_t {dq_cld:+.1f} % (obs above); of which wall flux {-wall:+.1f} %, fall-out {-fall:+.1f} %, wall deposition of particles {-dep:+.1f} %; "
              f"WELAS-visible LWC first 10 min {lw_m:.4f} vs {lw_o:.4f} g m-3; model glaciation {P.glaciation_T(m['t'], m['tg'], m['lwc'], ts):.1f} C")
