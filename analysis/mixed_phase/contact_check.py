"""How often does unactivated (interstitial) dust meet a droplet?  Brownian coagulation
(Fuchs) between interstitial dust and dilute droplets, from the final mode-1 model runs,
integrated from t_sat.  Compared with the observed ice the model does not explain.
Lower bound on contacts: phoresis, turbulence and gravity neglected (all small for
sub-micron dust near water saturation)."""
import os, sys
os.environ["ISKYLAB_TSAT"] = "hygrometer"
sys.argv = [sys.argv[0], "/home/claude/chk/runs"]
import numpy as np
from netCDF4 import Dataset
import paper_figs as P

U = "/mnt/user-data/uploads/iSKYLAB01/runs"
F = {"Exp021": "/home/claude/chk/out-base-Exp021.nc", "Exp025": "/home/claude/chk/out-base-Exp025.nc",
     "Exp027": "/home/claude/chk/out-base-Exp027.nc", "Exp022": f"{U}/wk3c_m1_p000/single_comparison/output-Exp022.nc",
     "Exp023": f"{U}/wk3c_m1_p050/single_comparison/output-Exp023.nc", "Exp026": f"{U}/wk3c_m1_m050/single_comparison/output-Exp026.nc"}
kB = 1.380649e-23


def darea(dmax, phi):
    phi = np.where((phi > 0) & np.isfinite(phi), phi, 1.0); phi = np.where(abs(phi - 1) < 1e-4, 1.0001, phi)
    if np.all(phi > 1):
        c = dmax / 2; a = c / phi; e = np.sqrt(1 - 1 / phi ** 2); S = 2 * np.pi * a ** 2 * (1 + (c / (a * e)) * np.arcsin(e))
    else:
        a = dmax / 2; e = np.sqrt(1 - phi ** 2); S = 2 * np.pi * a ** 2 * (1 + ((1 - e ** 2) / e) * np.arctanh(e))
    return np.sqrt(S / np.pi)


def fuchs(d1, d2, T, p):
    """Brownian coagulation kernel (m3 s-1), Fuchs interpolation (Seinfeld & Pandis eq. 13.50)."""
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    lam = mu / p * np.sqrt(np.pi * 287.05 * T / 2)
    def props(d, rho):
        Kn = 2 * lam / d
        Cc = 1 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))
        D = kB * T * Cc / (3 * np.pi * mu * d)
        m = rho * np.pi / 6 * d ** 3
        c = np.sqrt(8 * kB * T / (np.pi * m))
        l = 8 * D / (np.pi * c)
        g = ((d + l) ** 3 - (d ** 2 + l ** 2) ** 1.5) / (3 * d * l) - d
        return D, c, g
    D1, c1, g1 = props(d1, 2600.0); D2, c2, g2 = props(d2, 1000.0)
    dd = d1 + d2
    beta = 1 / (dd / (dd + 2 * np.sqrt(g1 ** 2 + g2 ** 2)) + 8 * (D1 + D2) / (np.sqrt(c1 ** 2 + c2 ** 2) * dd))
    return 2 * np.pi * (D1 + D2) * dd * beta


def run(e):
    v = Dataset(F[e]).variables
    g = lambda k: np.asarray(v[k][:])
    t, T, p = g("time"), g("t"), g("p"); rho = p / (287.05 * T)
    nw, dw, ma = g("nwat"), g("dwet"), g("maer")          # per kg air
    ni, dm, phi = g("nicem"), g("dmaxice"), g("phi")
    nmode = nw.shape[1]
    rate = np.zeros(len(t))
    ninter = np.zeros(len(t))
    for i in range(len(t)):
        # dry diameters
        ddry = []
        for m in range(nmode):
            n = nw[i, m]; md = ma[i, :, m, :].sum(0)  # per-particle mass (kg)
            ddry.append((6 * md / (np.pi * 2600)) ** (1 / 3))
        # interstitial dust: mode 0, wet D < 2 um
        n0, d0 = nw[i, 0] * rho[i], dw[i, 0]
        inter = (n0 > 0) & (d0 < 2e-6) & (d0 > 0)
        ninter[i] = n0[inter].sum()
        # collectors: dilute drops of any mode (wet/dry > 4 and wet D > 1 um), dust droplets included
        coll_n, coll_d = [], []
        for m in range(nmode):
            nn, d = nw[i, m] * rho[i], dw[i, m]
            ok = (nn > 0) & (d > 1e-6) & (d > 4 * np.maximum(ddry[m], 1e-9))
            coll_n.append(nn[ok]); coll_d.append(d[ok])
        cn, cd = np.concatenate(coll_n), np.concatenate(coll_d)
        if inter.any() and len(cn):
            K = fuchs(d0[inter][:, None], cd[None, :], T[i], p[i])
            rate[i] = np.sum(K * n0[inter][:, None] * cn[None, :])
    cum = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    da = np.empty_like(dm)
    for i in range(len(t)):
        da[i] = darea(dm[i], np.full(dm[i].shape, phi[i]))
    ice = np.sum(ni * (da > 15e-6) * (da <= 218e-6), axis=(1, 2)) * rho / 1e6
    return t, T - 273.15, cum / 1e6, ninter / 1e6, ice, rate / 1e6


print("Contacts of interstitial dust with dilute droplets since t_sat (cm-3), vs observed ice the model (mode 1) does not explain")
print("exp     T(C)  interstitial dust  contacts  | obs ice  model ice  excess | efficiency needed")
for e in ["Exp021", "Exp022", "Exp023", "Exp025", "Exp026", "Exp027"]:
    o = P.obs(e); ts = P.TSAT[e]; oi = P.smooth(o["ni"], 61)
    t, tg, cum, nint, ice, rate = run(e)
    c0 = np.interp(ts, t, cum)
    for Tq in (-22, -23, -24):
        k = np.where((o["t"] >= ts) & (o["tg"] <= Tq))[0]
        if not len(k): continue
        tq = o["t"][k[0]]
        ob = float(oi[k[0]]); mo = float(np.interp(tq, t, ice)); c = float(np.interp(tq, t, cum) - c0)
        ex = ob - mo
        print(f"{e}  {Tq:4d}  {np.interp(tq, t, nint):9.1f}  {c:9.3f}  | {ob:6.3f}  {mo:6.3f}  {ex:+6.3f} | {ex / c if c > 0 and ex > 0 else float('nan'):.2f}")


def contact_area(e):
    """Cumulative dust surface brought into contact with dilute droplets (um2 cm-3)."""
    v = Dataset(F[e]).variables
    g = lambda k: np.asarray(v[k][:])
    t, T, p = g("time"), g("t"), g("p"); rho = p / (287.05 * T)
    nw, dw, ma = g("nwat"), g("dwet"), g("maer")
    rate = np.zeros(len(t)); dmean = np.zeros(len(t))
    for i in range(len(t)):
        ddry = []
        for m in range(nw.shape[1]):
            n = nw[i, m]; md = ma[i, :, m, :].sum(0)  # per-particle mass (kg)
            ddry.append((6 * md / (np.pi * 2600)) ** (1 / 3))
        n0, d0 = nw[i, 0] * rho[i], dw[i, 0]
        inter = (n0 > 0) & (d0 < 2e-6) & (d0 > 0)
        cn, cd = [], []
        for m in range(nw.shape[1]):
            nn, d = nw[i, m] * rho[i], dw[i, m]
            ok = (nn > 0) & (d > 1e-6) & (d > 4 * np.maximum(ddry[m], 1e-9))
            cn.append(nn[ok]); cd.append(d[ok])
        cn, cd = np.concatenate(cn), np.concatenate(cd)
        if inter.any() and len(cn):
            K = fuchs(d0[inter][:, None], cd[None, :], T[i], p[i])
            r = K * n0[inter][:, None] * cn[None, :]
            A = np.pi * ddry[0][inter] ** 2
            rate[i] = np.sum(r * A[:, None])
            dmean[i] = np.sum(r * ddry[0][inter][:, None]) / max(np.sum(r), 1e-300)
    cum = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    return t, cum, dmean


print("\nSurface-weighted: n_s that contacts would need (excess / contacted dust surface) vs immersion n_s in the model")
B = {"SDSA01": (-0.7640, 4.333, -27.75), "ATD03": (-1.5437, -12.453, -24.75)}
for e in ["Exp022", "Exp023"]:
    o = P.obs(e); ts = P.TSAT[e]; oi = P.smooth(o["ni"], 61)
    t, tg, cum, nint, ice, rate = run(e)
    ta, ca, dmean = contact_area(e)
    c0 = np.interp(ts, ta, ca)
    b, a, tc = B[P.DUST[e]]
    for Tq in (-23, -24):
        k = np.where((o["t"] >= ts) & (o["tg"] <= Tq))[0][0]; tq = o["t"][k]
        ex = float(oi[k]) - float(np.interp(tq, t, ice))
        area = (np.interp(tq, ta, ca) - c0)          # m2 m-3
        ns_need = ex * 1e6 / area
        ns_imm = np.exp(b * max(Tq, tc) + a)
        print(f"{e} {Tq}: mean diameter of contacting dust {1e6*np.interp(tq, ta, dmean):.2f} um; contacted surface {area*1e6:.3f} um2 cm-3; "
              f"n_s needed {ns_need:.2e} m-2 = {ns_need/ns_imm:.0f} x immersion ({ns_imm:.1e}); "
              f"equivalent warm shift {np.log(ns_need/ns_imm)/(-b):.1f} K")
