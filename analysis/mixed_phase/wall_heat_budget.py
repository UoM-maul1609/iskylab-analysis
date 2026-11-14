"""Gas heat budget in the dry (pre-cloud) phase: is the measured Twall - Tgas
the temperature contrast that heats/cools the gas?

    dTg/dt = (R/cp) (Tg/p) dp/dt  +  (Tw - Tg)/tau_h  (+ other heating, e.g. fan)

The residual from the dry adiabat is regressed on the measured Tw - Tg
(through the origin) between t = 120 s and 60 s before cloud formation.
If the BL model is right, alpha_T = tau_BL / tau_h (tau_BL = 10 s).

Usage: python3 wall_heat_budget.py   (run from anywhere; uses Datasets-V2)
"""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
TSAT_MIN = {"Exp006": 12, "Exp007": 12, "Exp008": 12, "Exp009": 12, "Exp010": 11,
            "Exp011": 13, "Exp013": 11, "Exp014": 10.5, "Exp015": 10.75, "Exp016": 10.6,
            "Exp019": 11, "Exp021": 12, "Exp022": 14, "Exp023": 13, "Exp025": 10,
            "Exp026": 12, "Exp027": 14}
R_CP = 287.05 / 1005.0
TAU_BL = 10.0


def smooth(x, n=61):
    return np.convolve(x, np.ones(n) / n, "same")


def main():
    print("exp     dry_s  Tw-Tg[K]  heating[K/min]  tau_h[s]  r      alpha_T=tau_BL/tau_h")
    for exp, tsat in TSAT_MIN.items():
        rows = list(csv.reader(open(ROOT / f"iSKYLAB01-{exp}-1-MeteoCPC.csv")))
        a = np.array([[float(x[0]), float(x[1]), float(x[2]), float(x[4])]
                      for x in rows[1:] if all(x[i].strip() for i in (1, 2, 4))])
        t, p, tg, tw = a.T
        p, tg, tw = smooth(p * 100.0), smooth(tg + 273.15), smooth(tw + 273.15)
        resid = np.gradient(tg, t) - R_CP * tg / p * np.gradient(p, t)
        d = tw - tg
        s = (t > 120.0) & (t < tsat * 60.0 - 60.0)
        tau_h = np.sum(d[s] ** 2) / np.sum(d[s] * resid[s])
        r = np.corrcoef(d[s], resid[s])[0, 1]
        print(f"{exp}  {s.sum():5d}  {d[s].mean():+7.2f}  {resid[s].mean()*60:+12.3f}  "
              f"{tau_h:8.0f}  {r:+.2f}   {TAU_BL/tau_h:+.2f}")


if __name__ == "__main__":
    main()
