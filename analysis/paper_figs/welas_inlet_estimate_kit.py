"""WELAS inlet shrinkage with the sampling-line geometry from KIT (R. Wagner, 24 Nov 2026): vertical line from the
bottom of the cloud vessel; ~10 cm inside the vessel (at chamber temperature), ~50 cm through the outer vacuum vessel
(wall temperature between chamber and room: sensors Tgv0_a/b there read 15-55 % of the way from chamber to room
temperature), then ~15 cm in the lab at room temperature before the sensor. Flow 5 L/min. Same physics as
welas_inlet_estimate.py (laminar-pipe warming, Nu = 3.66; dD2/dt = 8 G s)."""
import numpy as np
from welas_inlet_estimate import G, es, Ra
def run(Tc, f_vac, Troom=21.5, bore=4.57e-3, flow_lpm=5.0, p=88000.0, segs=((0.10, 0.0), (0.50, None), (0.15, 1.0)), dt=1e-4):
    Tc += 273.15; Tr = Troom + 273.15; v = flow_lpm / 60000 / (np.pi * bore**2 / 4)
    rho = p / (Ra * Tc); tau = rho * 1005 * bore**2 / (4 * 3.66 * 0.024); qv = 0.622 * es(Tc) / p
    T = Tc; dd2 = 0.0; ttot = 0.0
    for length, f in segs:
        f = f_vac if f is None else f; Tw = Tc + f * (Tr - Tc); t = 0.0; tres = length / v
        while t < tres:
            T += (Tw - T) / tau * dt; s = qv * p / 0.622 / es(T) - 1
            if s < 0: dd2 += -8 * G(T, p) * s * dt
            t += dt
        ttot += tres
    return ttot, T - Tc, dd2 * 1e12
if __name__ == "__main__":
    for bore in (4.57e-3, 6.35e-3):
        for f in (0.15, 0.35, 0.55):
            for Tc in (-2.5,):
                tr, dT, d = run(Tc, f, bore=bore)
                print(f"bore {bore*1e3:.2f} mm, vacuum-vessel wall {int(f*100)} % to room: residence {tr:.3f} s, warming {dT:.1f} K, dD2 {d:.1f} um2")
    print("only the 15 cm room-temperature section, 4.57 mm:", round(run(-2.5, 0.0, segs=((0.15, 1.0),))[2], 1))
