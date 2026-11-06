import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, pandas as pd
from scipy.optimize import least_squares
exec(open(_os.path.join(REPO,'analysis','wall_water_check','wall.py')).read().split("rows=[]")[0])
rows=[]
for e in ["006","007","008","009","010","011","013","014","015","016","018","019"]:
    t,p,Tg,Tw,qm,lw=load(e); ok=np.isfinite(qm)
    above=lw>1e-5; idx=next(i for i in range(len(t)) if above[i:i+60].all()); on=t[idx]
    sel=ok&(t>=0)&(t<=on)
    tt,qq=t[sel],qm[sel]; q0=np.nanmean(qm[t<0])
    def sim(par):
        d,lk=par; k=np.exp(lk); q=q0; out=[]
        for j,ti in enumerate(t[(t>=0)&(t<=on)]):
            i=np.searchsorted(t,ti); 
            q-=k*max(q-qs(Tw[i]+d,p[i]),0)*1.0; out.append(q)
        return np.array(out)[ok[(t>=0)&(t<=on)]]
    r=least_squares(lambda par: (sim(par)-qq)*1e3,[0.0,np.log(1/300)],bounds=([-2,np.log(1e-5)],[2,np.log(1)]))
    d,lk=r.x
    rows.append(dict(exp=e,onset=round(on/60,1),delta_K=round(d,2),tau_s=round(1/np.exp(lk)),rms_gkg=round(np.sqrt(np.mean(r.fun**2)),3)))
df=pd.DataFrame(rows); print(df.to_string(index=False)); print(df[["delta_K","tau_s"]].agg(["mean","std","median"]))
