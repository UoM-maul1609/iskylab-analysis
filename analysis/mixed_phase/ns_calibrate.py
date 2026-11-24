"""Calibrate n_s(T) inside the model so that fall-out is treated the same way in the
calibration and in the test (mixed-phase paper, step 3 of the redo, 16 Nov 2026).

Problem: n_s was fitted to the OBSERVED AIRBORNE ice divided by the dust surface,
and used in the model as if it were all the ice ever formed.  The model then
removes crystals by fall-out, so its airborne ice is 3-5 times too low.

Method (per dust and per freezing mode, on the dust-alone run at its chosen offset):
  f(T)        = model airborne ice / (airborne + fallen ice)        [from the run]
  n_s,cum(T)  = n_s,obs(T) / f(T)                                   [observed, corrected]
  DEFAULT (--scale): keep the observed slope b and multiply n_s by one factor
  F = median over the fitted range of 1/f (total ice formed / airborne ice), i.e.
  a -> a + ln F.  Robust to single bins where the model has just glaciated.
  --refit: refit ln n_s,cum = b*T + a over the same temperature range instead;
  colder than the fitted range n_s is held constant (the most active sites are
  used up; the model removes dust that has already frozen).
Test of that choice: the corrected n_s colder than the fitted range is printed with
a fitted cold-side slope.  A slope near zero supports holding n_s constant.
One pass is usually enough; check with the wkcal runs and repeat if needed
(the script reads the current coefficients from paper/ns_coef_<prefix>.txt if
it exists, so a second pass starts from the first).

Usage:  ISKYLAB_TSAT=hygrometer python3 ns_calibrate.py [--prefix wk3]
Reads:  paper/refit_choice_<prefix>_constrained.txt, figs/ns_observed.csv
Writes: paper/ns_coef_<prefix>.txt   (mode dust b a tc cold)
"""
import csv
import sys

import numpy as np
from netCDF4 import Dataset

REFIT = "--refit" in sys.argv
if REFIT:
    sys.argv.remove("--refit")
if "--scale" in sys.argv:
    sys.argv.remove("--scale")

PREFIX = "wk3"
if "--prefix" in sys.argv:
    i = sys.argv.index("--prefix")
    PREFIX = sys.argv[i + 1]
    del sys.argv[i:i + 2]
import paper_figs as P  # noqa: E402

FIT = {"SDSA01": ("Exp021", -27.75, -20.75, 0.0, -0.7640, 2.634),
       "ATD03": ("Exp025", -24.75, -21.75, 0.0, -1.5437, -13.511)}


def tagof(prefix, mode, off):
    v = round(off * 1000)
    return f"{prefix}_m{mode}_{'m' if v < 0 else 'p'}{abs(v):03d}"


def main():
    choice = {}
    for line in open(P.OUT / f"refit_choice_{PREFIX}_constrained.txt"):
        if line.strip() and not line.startswith("#"):
            e, m, off = line.split()
            choice[(e, int(m))] = float(off)
    ns = {}
    for r in csv.DictReader(open(P.HERE / "figs" / "ns_observed.csv")):
        if r["dilution"] == "cpc":
            ns.setdefault(r["exp"], {})[float(r["T_mid_C"])] = P._num(r["ns_dust_m2"])
    out = []
    for dust, (e, tc, tw, cold, b0, a0) in FIT.items():
        for mode in (0, 1):
            tag = tagof(PREFIX, mode, choice[(e, mode)])
            v = Dataset(P.RUNS / tag / "single_comparison" / f"output-{e}.nc").variables
            rho = np.asarray(v["p"][:] / (287.05 * v["t"][:]))
            tg = np.asarray(v["t"][:]) - 273.15
            air = np.asarray(v["nice"][:]) * rho
            fall = np.asarray(v["nfall_ice"][:]) * rho
            T, y = [], []
            print(f"\n{dust} mode {mode} ({tag}):")
            print(f"  {'T':>6s} {'ns obs':>9s} {'airborne frac':>13s} {'ns cum':>9s} {'old fit':>9s}")
            for t in sorted(ns[e]):
                if tc <= t <= tw and np.isfinite(ns[e][t]) and ns[e][t] > 0:
                    k = np.argmin(abs(tg - t))
                    f = air[k] / max(air[k] + fall[k], 1e-30)
                    if not (0.02 < f <= 1.0):
                        continue
                    T.append(t)
                    y.append(np.log(ns[e][t] / f))
                    print(f"  {t:6.2f} {ns[e][t]:9.2e} {f:13.2f} {ns[e][t] / f:9.2e} {np.exp(b0 * t + a0):9.2e}")
            b, a = np.polyfit(T, y, 1)
            corr = np.array([yy - np.log(ns[e][tt]) for tt, yy in zip(T, y)])   # ln(1/f)
            F = float(np.exp(np.median(corr)))
            if not REFIT:
                b, a = b0, a0 + np.log(F)
            print(f"  fall-out correction factor F (median 1/f) = {F:.2f}; "
                  f"{'refit' if REFIT else 'scaled'} fit: ln ns = {b:.4f} T + {a:.3f}")
            # cold side: corrected ns colder than the fitted range
            Tc_, yc = [], []
            for t in sorted(ns[e]):
                if -30 <= t < tc and np.isfinite(ns[e][t]) and ns[e][t] > 0:
                    k = np.argmin(abs(tg - t))
                    f = air[k] / max(air[k] + fall[k], 1e-30)
                    if 0.02 < f <= 1.0:
                        Tc_.append(t)
                        yc.append(np.log(ns[e][t] / f))
            if len(Tc_) >= 3:
                bc = np.polyfit(Tc_, yc, 1)[0]
                lvl = np.exp(b * tc + a)
                print(f"  colder than {tc} C: corrected ns {np.exp(min(yc)):.2e}-{np.exp(max(yc)):.2e} "
                      f"(fit level at {tc} C: {lvl:.2e}); fitted cold slope {-bc:+.3f} K-1 "
                      f"(published fit {dict(SDSA01=0.517, ATD03=0.380)[dust]:.3f}; 0 = held constant)")
            print(f"  was: ln ns = {b0:.4f} T + {a0:.3f}")
            out.append((mode, dust, b, a, tc, cold))
    with open(P.OUT / f"ns_coef_{PREFIX}.txt", "w") as fh:
        fh.write("# mode dust b a tc cold   (ns_calibrate.py; ln ns = b*Tc + a down to tc, then slope cold)\n")
        for mode, dust, b, a, tc, cold in out:
            fh.write(f"{mode} {dust} {b:.4f} {a:.3f} {tc} {cold}\n")
    print("\nwritten", P.OUT / f"ns_coef_{PREFIX}.txt")


if __name__ == "__main__":
    main()
