"""Figure: ice at matched gas temperature (17 Nov 2026).  Uses paper/matched_T.json
written by matched_T.py.  Top: observed ice per unit dust surface.  Bottom: observed /
modelled ice (mode 1, area-sized 15-218 um); the dust-alone run sets the reference."""
import json

import matplotlib.pyplot as plt
import numpy as np

import paper_figs as P

J = json.load(open(P.OUT / "matched_T.json"))
A = J["area"]
LAB = {"Exp021": "dust alone (Exp021)", "Exp022": "+ NaCl (Exp022)", "Exp023": "+ more NaCl (Exp023)",
       "Exp025": "dust alone (Exp025)", "Exp026": "+ NaCl (Exp026)", "Exp027": "+ more NaCl (Exp027)"}
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
for j, (name, exps) in enumerate((("South African soil dust (SDSA01)", ["Exp021", "Exp022", "Exp023"]),
                                  ("Arizona test dust (ATD03)", ["Exp025", "Exp026", "Exp027"]))):
    for e in exps:
        pr = np.array(A["profiles"][e])
        ok = pr[:, 1] > 0.02
        c = P.C_RUN[e]
        ls = "-" if e in P.MIX else "-"
        lw = 2.0 if e not in P.MIX else 1.4
        axs[0, j].plot(pr[ok, 0], pr[ok, 3] * 1e2, ls, color=c, lw=lw, marker="o", ms=2.5, label=LAB[e])
        axs[1, j].plot(pr[ok, 0], pr[ok, 1] / pr[ok, 2], ls, color=c, lw=lw, marker="o", ms=2.5)
        if e in P.MIX:
            k = exps.index(e)
            axs[1, j].text(0.03, 0.97 - 0.07 * k, f"{LAB[e].split(' (')[0]}: ×{A['E'][e]:.1f} relative to dust alone",
                           transform=axs[1, j].transAxes, va="top", fontsize=6.5, color=c)
    axs[0, j].set_title(name, fontsize=8, loc="left")
    axs[0, j].set_yscale("log")
    axs[1, j].set_yscale("log")
    axs[1, j].axhline(1, color=P.MUTED, lw=0.6, zorder=0)
    axs[1, j].set_xlabel("Gas temperature (°C)")
    axs[0, j].legend(fontsize=6.5, loc="upper left" if j == 0 else "lower right")
for ax in axs.flat:
    ax.set_xlim(-20.5, -26.5)
axs[0, 0].set_ylabel("Observed ice per dust surface\n(cm$^{-3}$ per 100 µm$^2$ cm$^{-3}$)")
axs[1, 0].set_ylabel("Observed / modelled ice\n(model: dust freezes without activating)")
axs[1, 0].set_ylim(0.1, 30)
axs[1, 1].set_ylim(0.1, 30)
for ax, l in zip(axs.flat, "abcd"):
    ax.text(0.99, 0.97, l, transform=ax.transAxes, ha="right", va="top", fontweight="bold")
fig.tight_layout()
P.save(fig, "f_matched_T")
print("saved", P.FIG / "f_matched_T.png")
