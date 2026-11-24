"""Fig. 11: mean droplet size distribution in Exp013, early and late in the cloud: WELAS, the model as computed,
and the model seen through the WELAS inlet (dD2 = cfg.WELAS_INLET_DD2_UM2).  Writes runs/analysis/fig_psd.png."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "scripts"))
import dataAnalysis_new as DA
sys.path.insert(0, str(Path(__file__).resolve().parent)); import paper_tags as PT
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
OUT = T / "analysis"; DATA = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EXP = "Exp013"; WINS = [(13.0, 15.0, "(a) Early in the cloud: 13–15 min"), (20.0, 25.0, "(b) Late: 20–25 min")]
o = pd.read_csv(DATA / f"iSKYLAB01-{EXP}-2-OPC-d-MergedW.csv", skipinitialspace=True)
cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]; dm = np.array([float(c) for c in cols])
fo = np.nan_to_num(o[cols].values.astype(float)); to = o.iloc[:, 0].values / 60          # dN/dlogD (cm-3)
lg = np.log10(dm); edw = 10**np.r_[lg[0] - (lg[1] - lg[0]) / 2, (lg[1:] + lg[:-1]) / 2, lg[-1] + (lg[-1] - lg[-2]) / 2]
ed = np.logspace(np.log10(0.8), np.log10(40), 25); dc = np.sqrt(ed[1:] * ed[:-1])          # common grid, both sides
def opsd(s):   # WELAS counts per original bin, re-assigned to the common grid by bin centre
    cnt = (fo[s] * np.diff(np.log10(edw))).mean(0)
    return np.histogram(dm, bins=ed, weights=cnt)[0] / np.diff(np.log10(ed))
with Dataset(T / PT.FINAL / "single_comparison" / f"output-{EXP}.nc") as nc:
    tm = nc["time"][:] / 60; n = np.asarray(nc["nwat"][:]); d = np.asarray(nc["dwet"][:]) * 1e6; mw = np.asarray(nc["mwat"][:])
    rho = np.asarray(nc["p"][:] / (287.0 * nc["t"][:]))
din = DA.apply_welas_inlet({"dwet": d * 1e-6, "mwat": mw}, DA.cfg.WELAS_INLET_DD2_UM2)["dwet"] * 1e6
def mpsd(dd, s):
    h = np.array([np.histogram(dd[i].ravel(), bins=ed, weights=n[i].ravel())[0] for i in np.where(s)[0]]) * rho[s][:, None] / 1e6
    return h.mean(0) / np.diff(np.log10(ed))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, (t0, t1, lab) in zip(ax, WINS):
    so = (to >= t0) & (to <= t1); sm = (tm >= t0) & (tm <= t1)
    a.step(dc, opsd(so), where="mid", color="#333333", lw=1.8, label="WELAS")
    a.step(dc, mpsd(d, sm), where="mid", color="#1f6fb4", lw=1.2, ls="--", label="model, as computed")
    a.step(dc, mpsd(din, sm), where="mid", color="#1f6fb4", lw=2.0, label="model, seen through the WELAS inlet")
    a.axvline(2.5, color="0.6", lw=0.8, ls=":"); a.set_xscale("log"); a.set_xlim(0.8, 40)
    a.set_title(lab, loc="left", fontsize=10); a.set_xlabel("Droplet diameter (µm)"); a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("d$N$/dlog$_{10}D$ (cm$^{-3}$)"); ax[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.32, 1.0))
for a in ax: a.text(2.4, a.get_ylim()[1] * 0.6, "2.5 µm threshold", fontsize=7, color="0.4", rotation=90, ha="right", va="center")
fig.tight_layout(); fig.savefig(OUT / "fig_psd.png", dpi=200); print("written")
