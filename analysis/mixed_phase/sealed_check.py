"""Sealed-chamber check for the mixed-phase paper (17 Nov 2026).
Exp022 and Exp027, both freezing modes, mixture and NaCl removed, final wk3c settings, rerun
(Linux BMM build) with the walls (as wk3c) and sealed (chamber_bl_mix = 0 and
chamber_wall_vapour_transfer_velocity = 0).  Prints peak supersaturation, the dust surface in
droplets, the paired ratio P before glaciation and the dust still airborne at the end.
Usage: python3 sealed_check.py <dir with out-{walls,sealed}-m{0,1}-{mix,dust}-Exp02{2,7}.nc>"""
import sys
import numpy as np
from netCDF4 import Dataset

D = sys.argv[1] if len(sys.argv) > 1 else "."
TS = {"Exp022": (738, 1467), "Exp027": (552, 1551)}   # hygrometer saturation, observed end of liquid (s)


def darea(dmax, phi):
    phi = np.where((phi > 0) & np.isfinite(phi), phi, 1.0)
    phi = np.where(abs(phi - 1) < 1e-4, 1.0001, phi)
    if np.all(phi > 1):
        c = dmax / 2; a = c / phi; e = np.sqrt(1 - 1 / phi ** 2)
        S = 2 * np.pi * a ** 2 * (1 + (c / (a * e)) * np.arcsin(e))
    else:
        a = dmax / 2; e = np.sqrt(1 - phi ** 2)
        S = 2 * np.pi * a ** 2 * (1 + ((1 - e ** 2) / e) * np.arctanh(e))
    return np.sqrt(S / np.pi)


def load(f):
    v = Dataset(f).variables
    g = lambda k: np.asarray(v[k][:])
    t, tg, rho = g("time"), g("t") - 273.15, g("p") / (287.05 * g("t"))
    ni, dm, phi = g("nicem"), g("dmaxice"), g("phi")
    da = np.empty_like(dm)
    for i in range(len(t)):
        da[i] = darea(dm[i], np.full(dm[i].shape, phi[i]))
    n15 = np.sum(ni * (da > 15e-6) * (da <= 218e-6), axis=(1, 2)) * rho / 1e6
    nw, dw, ma = g("nwat"), g("dwet"), g("maer")

    def act(T):
        i = int(np.argmin(abs(tg - T))); n = nw[i, 0]; md = ma[i, :, 0, :].sum(0)
        dd = (6 * md / (np.pi * 2600)) ** (1 / 3)   # maer is mass per particle (kg)
        return float(np.sum(n * dd ** 2 * (dw[i, 0] > 2e-6)) / max(np.sum(n * dd ** 2), 1e-30))
    dust = np.sum(nw[:, 0, :], axis=1) + np.sum(ni[:, 0, :], axis=1)
    return dict(t=t, rh=g("rh"), n15=n15, act=act, ret=dust[-1] / dust[0])


for e, (ts, tgl) in TS.items():
    for m in (0, 1):
        for w in ("walls", "sealed"):
            a, b = load(f"{D}/out-{w}-m{m}-mix-{e}.nc"), load(f"{D}/out-{w}-m{m}-dust-{e}.nc")
            sel = lambda x: (x["t"] >= ts) & (x["t"] <= tgl)
            mi, di = np.mean(a["n15"][sel(a)]), np.mean(b["n15"][sel(b)])
            print(f"{e} mode {m} {w:6s}: Smax {100 * (a['rh'][a['t'] >= ts - 120].max() - 1):5.2f} %; dust surface in drops "
                  f"-21/-23 C: mix {a['act'](-21):.2f}/{a['act'](-23):.2f}, NaCl removed {b['act'](-21):.2f}/{b['act'](-23):.2f}; "
                  f"P {mi / di if di > 0 else float('nan'):.2f}; dust airborne {a['ret']:.2f}/{b['ret']:.2f}")
