"""Estimate of droplet shrinkage (Delta D^2) in the WELAS sampling line.
Air from the chamber (saturated at Tchamber) flows through a tube whose wall is at (a fraction of) room temperature;
the air warms with the laminar-pipe heating timescale tau = rho cp D^2 / (4 Nu k), Nu = 3.66, and droplets evaporate
at dD^2/dt = 8 G s (Rogers & Yau G).  Vapour added by evaporating droplets is neglected (|s| ~ 0.6 >> 0.01).
Assumptions to replace with KIT values: flow rate, tube bore, tube length, tube-wall temperature, room temperature."""
import numpy as np
Rv, Ra, L, rw = 461.5, 287.0, 2.5e6, 1000.0
es = lambda T: 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
def G(T, p):
    Dv = 2.11e-5 * (T / 273.15)**1.94 * (101325 / p); K = 2.4e-2
    return 1 / ((L / (Rv * T) - 1) * L * rw / (K * T) + rw * Rv * T / (Dv * es(T)))
def shrink(Tc, Troom=21.5, bore=4.57e-3, length=0.4, flow_lpm=5.0, p=88000.0, wall_frac=1.0, dt=1e-4):
    Tc += 273.15; Tw = Tc + wall_frac * (Troom + 273.15 - Tc)
    rho = p / (Ra * Tc); v = flow_lpm / 60000 / (np.pi * bore**2 / 4); tres = length / v
    tau = rho * 1005 * bore**2 / (4 * 3.66 * 0.024); qv = 0.622 * es(Tc) / p; T = Tc; dd2 = 0.0; t = 0.0
    while t < tres:
        T += (Tw - T) / tau * dt; s = qv * p / 0.622 / es(T) - 1
        if s < 0: dd2 += -8 * G(T, p) * s * dt
        t += dt
    return dict(residence_s=tres, warming_K=T - Tc, s_end=s, dD2_um2=dd2 * 1e12)
if __name__ == "__main__":
    for bore in (4.57e-3, 6.35e-3):
        for length in (0.3, 0.4, 0.5):
            r = shrink(-2.5, bore=bore, length=length)
            print(f"bore {bore*1e3:.1f} mm, {length:.1f} m, 5 L/min: residence {r['residence_s']:.3f} s, warming {r['warming_K']:.1f} K, dD2 {r['dD2_um2']:.1f} um2")
    for q in (2, 5, 10): print(f"flow {q} L/min (4.6 mm, 0.4 m): dD2 {shrink(-2.5, flow_lpm=q)['dD2_um2']:.1f} um2")
    for f in (0.25, 0.5, 1.0): print(f"tube wall {int(f*100)} % of the way to room T: dD2 {shrink(-2.5, wall_frac=f)['dD2_um2']:.1f} um2")
