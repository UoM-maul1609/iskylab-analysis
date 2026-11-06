import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, os, re, json
from scipy.optimize import least_squares
D=REPO+"/iSKYLAB-data/Datasets-V2/Initial-PNSD"
OUT=OUTDIR+"/analysis"; os.makedirs(OUT,exist_ok=True)
def f(x):
    try: return float(x)
    except: return np.nan
def ln(D,N,dm,s): return N/(np.sqrt(2*np.pi)*s)*np.exp(-(np.log(D/dm))**2/(2*s*s))*np.log(10)   # dN/dlog10D
res={}
import pandas as pd
met=REPO+"/iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
for e in ["005","006","007","008","009","010","011","013","014","015","016","017","018","019"]:
    L=[l.rstrip("\n").split(",") for l in open(f"{D}/iSKYLAB01-Exp{e}-3-InitialPNSD-Mrg.csv")]
    d={re.sub(r"\s+"," ",r[0].strip()):np.array([f(x) for x in r[1:]]) for r in L}
    m=min(len(v) for v in d.values()); d={k:v[:m] for k,v in d.items()}
    Dv=d["Dve (um)"]*1e-6; dl=d["dlogDve"]
    mm=pd.read_csv(f"{met}/iSKYLAB01-Exp{e}-1-MeteoCPC.csv",skipinitialspace=True); mm.columns=[c.strip() for c in mm.columns]
    pre=mm[mm["Time (Sec)"]<0]; rho=pre["Pressure (hPa)"].mean()*100/(287*(pre["Tgw_mean (degC)"].mean()+273.15))
    res[e]={"rho_air":round(rho,4)}
    for sp in ["AS","NaCl","FA","Total"]:
        k=[kk for kk in d if kk.startswith(sp+" ") and "#/cc" in kk]
        if not k: continue
        n=np.nan_to_num(d[k[0]]); 
        if n.sum()==0: continue
        if sp=="Total" and any(x in res[e] for x in ("AS","NaCl","FA")): continue
        name=sp if sp!="Total" else {"005":"AS","011":"AS","016":"AS","017":"AS","013":"NaCl","014":"NaCl","015":"FA"}[e]
        ok=(Dv>=10e-9)
        Ntot=(n*dl)[Dv>=10e-9].sum()
        best=None
        for nm in (1,2,3):
            dms=np.exp(np.linspace(np.log(40e-9),np.log(300e-9),nm))
            x0=np.concatenate([[Ntot/nm]*nm,np.log(dms),[0.5]*nm])
            def model(x): return sum(ln(Dv,x[i],np.exp(x[nm+i]),x[2*nm+i]) for i in range(nm))
            r=least_squares(lambda x: (np.log1p(model(x)[ok])-np.log1p(n[ok])),x0,bounds=([0]*nm+[np.log(15e-9)]*nm+[0.05]*nm,[Ntot*3]*nm+[np.log(5e-6)]*nm+[1.5]*nm))
            cost=np.sqrt(np.mean(r.fun**2))
            if best is None or cost<best[0]*0.8: best=(cost,nm,r.x)
        cost,nm,x=best
        modes=[dict(N_cm3=round(x[i],1),N_kg=float("%.4g"%(x[i]*1e6/rho)),d_m=float("%.4g"%np.exp(x[nm+i])),ln_sigma=round(x[2*nm+i],3)) for i in range(nm)]
        res[e][name]=dict(N_meas_cm3=round(Ntot,1),N_gt0p3_cm3=round(float((n*dl)[Dv>=0.3e-6].sum()),1),rms_log=round(cost,3),modes=modes)
json.dump(res,open(OUT+"/measured_psd_fits.json","w"),indent=1)
for e,v in res.items():
    print(e, {k:(vv["N_meas_cm3"],[(m_["N_cm3"],round(m_["d_m"]*1e9),m_["ln_sigma"]) for m_ in vv["modes"]],vv["rms_log"]) for k,vv in v.items() if k!="rho_air"})
