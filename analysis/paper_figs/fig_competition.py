"""Competition figure.
(a) Paired simulations (same experiment with and without the NaCl): droplets on the background aerosol and on the NaCl.
(b) Droplets > 5 um: measured mixture vs background-only experiment, the same comparison in the model, and the NaCl effect alone (paired).
(c) Robustness: change in total droplets in the sensitivity simulations that reproduce the liquid water.
Needs baseline, comp_bg, Exp015/016 baseline runs, and competition_robust.csv (competition_robust.py)."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT = T / "analysis"; D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
EXPS = [("Exp006", "AS + coarse NaCl 703", "Exp016"), ("Exp007", "AS + coarse NaCl 2065", "Exp016"), ("Exp018", "FA + coarse NaCl 796", "Exp015"),
        ("Exp008", "AS + fine NaCl 1028", "Exp016"), ("Exp009", "AS + fine NaCl 1588", "Exp016"), ("Exp010", "AS + fine NaCl 3107", "Exp016")]
BG, NA, OBS, MOD, PAIR = "#1f6fb4", "#d9822b", "#333333", "#7fb3e0", "#d9822b"
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}; return float(m["window_start_s"]), float(m["window_end_s"])
def sim(tag, e, w):
    with Dataset(T / tag / "single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None] / 1e6
        nl = nc["nliq"][:].data[s] * rho; nw = nc["nwat"][:].data[s] * rho; dw = nc["dwet"][:].data[s] * 1e6
    return nl[:, 0].sum(-1).mean(), nl[:, 1].sum(-1).mean(), (nw * (dw > 5)).sum(axis=(1, 2)).mean()
def obs5(e, w):
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-d-MergedW.csv", skipinitialspace=True); cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]); n = np.nan_to_num(o[cols].values.astype(float)) * 0.00984; t = o.iloc[:, 0].values
    s = (t >= w[0]) & (t <= w[1]); return n[s][:, dm > 5].sum(1).mean()
rows = []
for e, lab, ref in EXPS:
    w = win(e); wr = win(ref)
    mb, mn, m5 = sim("baseline", e, w); bb, bn, b5 = sim("comp_bg", e, w); _, _, r5 = sim("baseline", ref, wr)
    rows.append(dict(exp=e, lab=lab, ref=ref, mix_bg=mb, mix_nacl=mn, alone_bg=bb, total_change=100 * ((mb + mn) / bb - 1),
                     obs_change5=100 * (obs5(e, w) / obs5(ref, wr) - 1), mod_change5=100 * (m5 / r5 - 1), pair_change5=100 * (m5 / b5 - 1)))
df = pd.DataFrame(rows); df.to_csv(OUT / "competition_figure_data.csv", index=False); print(df.round(0).to_string(index=False))
rob = pd.read_csv(OUT / "competition_robust.csv")
fig, ax = plt.subplots(1, 3, figsize=(13.0, 6.0), gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.9]))
y = np.arange(len(EXPS))[::-1].astype(float); y[3:] -= 0.6
# (a) paired simulations: stacked droplets by aerosol
a = ax[0]; h = 0.36
for yi, r in zip(y, df.itertuples()):
    a.barh(yi + 0.2, r.alone_bg, h, color=BG)
    a.barh(yi - 0.2, r.mix_bg, h, color=BG); a.barh(yi - 0.2, r.mix_nacl, h, left=r.mix_bg, color=NA)
    a.text(r.alone_bg + 25, yi + 0.2, "without NaCl", va="center", fontsize=7, color="0.35")
    a.text(r.mix_bg + r.mix_nacl + 25, yi - 0.2, f"with NaCl: total {r.total_change:+.0f} %", va="center", fontsize=7.5,
           color="k", weight="bold" if r.total_change < -5 else "normal")
a.set_yticks(y); a.set_yticklabels(df.lab, fontsize=8.5)
for t, col in zip(a.get_yticklabels(), ["k"] * 6): t.set_color(col)
a.set_xlim(0, 2600); a.set_xlabel("Droplets in the model (cm$^{-3}$)")
a.bar(0, 0, color=BG, label="on the background aerosol"); a.bar(0, 0, color=NA, label="on the NaCl"); a.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=2)
a.set_title("(a) Competition in the model: the same experiment\nsimulated without and with the NaCl", loc="left", fontsize=9.5)
# (b) droplets > 5 um
b = ax[1]; h = 0.26
for yi, r in zip(y, df.itertuples()):
    b.barh(yi + h, r.obs_change5, h, color=OBS, label="measured: mixture vs background-only experiment" if r.exp == "Exp006" else None)
    b.barh(yi, r.mod_change5, h, color=MOD, label="model: the same comparison" if r.exp == "Exp006" else None)
    b.barh(yi - h, r.pair_change5, h, color=PAIR, label="model: effect of the NaCl alone (as in a)" if r.exp == "Exp006" else None)
b.axvline(0, color="k", lw=0.8); b.set_yticks(y); b.set_yticklabels([]); b.set_xlim(-10, 200)
b.set_xlabel("Change in droplets larger than 5 µm (%)"); b.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(-0.02, -0.13))
b.text(198, (y[2] + y[3]) / 2, "fine NaCl: the increases come from other\ndifferences between the experiments;\nthe NaCl alone has no effect (orange)", ha="right", va="center", fontsize=7.5, color="0.25")
b.set_title("(b) Test against the measurements:\ndroplets larger than 5 µm", loc="left", fontsize=9.5)
# (c) sensitivity simulations
c = ax[2]; c.axvspan(-100, 0, color="#e8f0f8", zorder=0); c.axvline(0, color="k", lw=0.8)
MK = {"baseline": ("D", "baseline"), "hom": ("^", "homogeneous evaporation"), "p0": ("s", "evaporation of all sizes ($p$=0)"),
      "chenlamb": ("h", "Chen–Lamb bin scheme"), "dTwarm": ("*", "wall-conditioned air 0.03 K warmer")}
seen = set()
for yi, r in zip(y, df.itertuples()):
    d = rob[(rob.exp == r.exp) & np.isfinite(rob.change_pct) & (rob.Rq_mix >= 0.5) & (rob.Rq_mix <= 2)]
    if len(d) == 0: d = pd.DataFrame([dict(variant="baseline", change_pct=r.total_change)])
    for q in d.itertuples():
        m, l = MK[q.variant]
        c.scatter(q.change_pct, yi, marker=m, s=70 if q.variant == "baseline" else 55, color="k" if q.variant == "baseline" else "0.6", edgecolor="0.2", lw=0.5, zorder=3, label=None if l in seen else l); seen.add(l)
c.set_yticks(y); c.set_yticklabels([]); c.set_xlim(-70, 30); c.set_xlabel("Change in total droplets (%)")
c.text(-3, y[0] + 0.75, "← fewer", ha="right", fontsize=8); c.text(3, y[0] + 0.75, "more →", ha="left", fontsize=8)
c.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=1, title="all reproduce the measured liquid water", title_fontsize=7.8, alignment="left")
c.set_title("(c) Total droplets in the\nsensitivity simulations", loc="left", fontsize=9.5)
for x in ax: x.set_ylim(y[-1] - 0.6, y[0] + 1.1); x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.subplots_adjust(bottom=0.3); fig.savefig(OUT / "fig_competition.png", dpi=200); plt.close(fig)
