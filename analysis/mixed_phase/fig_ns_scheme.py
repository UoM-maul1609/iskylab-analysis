"""Schematic of how BMM stores the freezing temperatures of the dust (17 Nov 2026, v2).

One bin of 2 um SDSA01 particles, n_s as used in the model (mode 1, held constant colder than
-27.75 C).  For display the freezing temperatures are grouped into 12 classes of 1 K from -18 to
-30 C (the model uses 100 classes of 0.12 K).
(a) What the bin stores when the aerosol is made: the share of its particles in each class.
(b) Gas at -27 C, particles eligible to freeze throughout: every class warmer than the gas has
    frozen and left the dust; colder classes are still stored.
(c) Gas at -27 C, particles not eligible until now: the classes warmer than the gas are still
    stored ("waiting"), and freeze in the time step in which the particles become eligible.
(d) Fraction of the bin frozen against gas temperature, eligible throughout, eligible only from
    -27 C with stored freezing temperatures (BMM), and a scheme that remembers only the lowest
    temperature reached.
Writes paper/figs/F06b_ns_scheme.pdf|png.  Usage: python3 fig_ns_scheme.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_figs as P

INK, MUTED = P.INK, P.MUTED
C_STORE, C_ICE, C_WAIT = "#b9b9b4", "#2a78d6", "#eb6834"
B, A, TC = -0.7640, 4.333, -27.75          # SDSA01, mode 1 (paper/ns_coef_wk3.txt)
D = 2e-6
AREA = np.pi * D ** 2
EDGES = np.arange(-18.0, -30.01, -1.0)     # 12 display classes
T_GAS = -27.0


def ns(t):
    t = np.asarray(t, float)
    return np.where(t > -18, 0.0, np.exp(B * np.maximum(t, TC) + A))


cum = 1 - np.exp(-AREA * ns(EDGES))        # share with freezing T warmer than each edge
share = np.diff(cum)                        # share in each 1 K class
mid = 0.5 * (EDGES[:-1] + EDGES[1:])
warmer = EDGES[1:] >= T_GAS                 # class lies entirely warmer than the gas
YM = share.max() * 100

fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.62, wspace=0.32)


def bars(ax, colours, title, letter):
    for x, h, c in zip(mid, share * 100, colours):
        ax.bar(x, h, width=0.85, color=c, lw=0, hatch="////" if c == C_WAIT else None,
               edgecolor="white" if c == C_WAIT else None)
    ax.set_xlim(-17.5, -30.5)
    ax.set_ylim(0, YM * 1.3)
    ax.set_xlabel("Freezing temperature (°C)")
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.4, axis="y")


ax = fig.add_subplot(gs[0, 0])
bars(ax, [C_STORE] * len(mid), "Stored when the aerosol is made", "a")
ax.set_ylabel("Particles in the bin (%)")
ax.text(-18.2, YM * 1.25, "1 − exp[−A n$_s$(T)],\nsplit into classes", fontsize=6, va="top")

for col, (cols, title, letter, note, ncol) in enumerate((
        ([C_ICE if w else C_STORE for w in warmer], "Gas at −27 °C, eligible", "b", "frozen:\nnow ice", C_ICE),
        ([C_WAIT if w else C_STORE for w in warmer], "Gas at −27 °C, not yet eligible", "c",
         "waiting: freeze\nas soon as eligible", C_WAIT)), start=1):
    ax = fig.add_subplot(gs[0, col])
    bars(ax, cols, title, letter)
    ax.axvline(T_GAS, color=INK, lw=0.8, ls="--")
    ax.text(T_GAS + 0.25, YM * 1.25, "gas", fontsize=6, ha="right", va="top")
    ax.text(-18.2, YM * 1.25, note, fontsize=6, color=ncol, va="top")
    ax.text(T_GAS - 0.25, YM * 1.25, "still\nstored", fontsize=6, color=MUTED, ha="left", va="top")

ax = fig.add_subplot(gs[1, :2])
T = np.linspace(-18, -30, 600)
always = 1 - np.exp(-AREA * ns(T))
stored = np.where(T > T_GAS, 0.0, always)
tmin = np.where(T > T_GAS, 0.0, 1 - np.exp(-AREA * (ns(T) - ns(T_GAS))))
ax.axvspan(-17.5, T_GAS, color=C_WAIT, alpha=0.08, lw=0)
ax.text(-18.0, 0.97, "not eligible (c)", fontsize=6.5, va="top", color=C_WAIT)
ax.plot(T, always, "-", color=C_ICE, lw=1.4, label="eligible throughout (b)")
ax.plot(T, stored, "-", color=C_WAIT, lw=2.0, label="eligible from −27 °C, freezing temperatures stored (BMM)")
ax.plot(T, tmin, "--", color=INK, lw=1.2, label="eligible from −27 °C, only the lowest temperature remembered")
x0 = -29.2
y1 = float(np.interp(x0, T[::-1], stored[::-1])); y0 = float(np.interp(x0, T[::-1], tmin[::-1]))
ax.annotate("", xy=(x0, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
ax.text(x0 - 0.15, 0.5 * (y0 + y1), "lost", fontsize=6.5, ha="left", va="center")
ax.set_xlim(-17.5, -30.5)
ax.set_ylim(0, 1)
ax.set_xlabel("Gas temperature (°C)")
ax.set_ylabel("Fraction of the bin frozen")
ax.set_title("(d) Fraction of the bin frozen as the gas cools", loc="left", fontsize=7.5)
ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.86))
ax.grid(alpha=0.25, lw=0.4)

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.set_title("What BMM keeps for each dust bin", loc="left", fontsize=7.5)
ax.text(0.0, 0.9,
        "• the share of its particles in each\n  freezing-temperature class (a)\n\n"
        "• moved with the particles: growth,\n  collisions, fall-out, wall loss\n\n"
        "• a class freezes when the gas is\n  colder and the particles are eligible;\n  it then leaves the dust (b)\n\n"
        "• classes passed while not eligible\n  wait, and freeze at once later (c, d)",
        fontsize=6.3, va="top", transform=ax.transAxes)

fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.1)
P.save(fig, "F06b_ns_scheme")
print("saved", P.FIG / "F06b_ns_scheme.png")
