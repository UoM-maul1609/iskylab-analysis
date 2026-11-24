"""Why does the sealed chamber show no dip for AS + coarse NaCl when an adiabatic parcel does?  (batchRuns_v2.sh v19parcel)
Exp006 aerosol, NaCl scaled.  Droplets formed (all aerosol) averaged 210-570 s after each run's own cloud formation
(RH first > 100 %), the same offset as the early-cloud window in Exp006.  Change relative to no NaCl.
Usage: python3 parcel_check.py -> runs/analysis/parcel_check.csv and parcel_check.png"""
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")); A = T / "analysis"
SC = [0, 0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 1, 1.5, 2.5, 4]; N0 = 749.0   # Exp006 NaCl added (cm-3) at scale 1
SETS = [("parcel, 1.3 m/s (0.76 K/min)", "v19_pc130_s{:04d}", "v19_pc130_s0000", "#1a9850", "-"),
        ("parcel, 0.85 m/s (0.50 K/min)", "v19_pc085_s{:04d}", "v19_pc085_s0000", "#66bd63", "--"),
        ("sealed chamber (measured T, p)", "v19_sw0_s{:04d}", "v19_bg_nobl", "#c0392b", "--"),
        ("real chamber (walls)", "v19_sw_s{:04d}", "v19_bg", "#c0392b", "-")]
def nd(tag):
    p = T / tag / "single_comparison" / "output-Exp006.nc"
    if not p.exists(): return None
    with Dataset(p) as nc:
        t = nc["time"][:].data; rh = nc["rh"][:].data.ravel(); rho = (nc["p"][:] / (287 * nc["t"][:])).data.ravel()
        n = nc["nliq"][:].data.sum(axis=(1, 2)) * rho / 1e6
    i = np.argmax(rh > 1.0); s = (t >= t[i] + 210) & (t <= t[i] + 570)
    return dict(n=n[s].mean(), smax=100 * (rh.max() - 1), t0=t[i])
rows = []
for lab, pat, base, col, ls in SETS:
    B = nd(base)
    if B is None: continue
    for s in SC:
        R = B if s == 0 else nd(pat.format(int(round(s * 100))))
        if R is None: continue
        rows.append(dict(set=lab, scale=s, nacl_cm3=s * N0, n=R["n"], smax=R["smax"], onset_s=R["t0"], d_tot=100 * (R["n"] / B["n"] - 1)))
df = pd.DataFrame(rows); df.to_csv(A / "parcel_check.csv", index=False); print(df.round(2).to_string(index=False))
fig, ax = plt.subplots(figsize=(7.5, 5)); ax.axhspan(-100, 0, color="#f6e3e0"); ax.axhline(0, color="k", lw=0.8)
for lab, pat, base, col, ls in SETS:
    d = df[df.set == lab]
    if len(d): ax.plot(d.nacl_cm3, d.d_tot, ls, color=col, lw=2.2, marker="o", ms=4, label=lab)
ax.axvline(N0, color="0.6", lw=0.8, ls=":"); ax.text(N0, ax.get_ylim()[1], " Exp006", va="top", fontsize=9, color="0.4")
ax.set_xlabel("NaCl particles added (cm$^{-3}$)"); ax.set_ylabel("Change in droplets formed (%)"); ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False); ax.set_title("AS + coarse NaCl (Exp006 aerosol): where does the dip go?", loc="left", weight="bold")
fig.savefig(A / "parcel_check.png", dpi=160, bbox_inches="tight")
