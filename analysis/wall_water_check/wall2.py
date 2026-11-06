import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, pandas as pd
exec(open(_os.path.join(REPO,'analysis','wall_water_check','wall.py')).read().split("rows=[]")[0])
tsat={"005":8,"006":12,"007":12,"008":12,"009":12,"010":11,"011":13,"013":11,"014":10.5,"015":10.75,"016":10.6,"017":1.25,"018":7,"019":11}
def emul(t,p,Tg,Tw,q0,off,tau):
    q=q0; qs_=[]; on=np.nan; L_=[q0]
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]; qsg=qs(Tg[k],p[k]); qv=min(q,qsg)
        if np.isnan(on) and q>=qsg: on=t[k]
        q-=(1-np.exp(-dt/tau))*bl_loss(qv,Tw[k]+off,p[k]); L_.append(q)
    return np.array(L_),on
rows=[]
for e,(off_p,tau) in paper.items():
    if e in ("005","017"): continue
    t,p,Tg,Tw,qm,lw=load(e); ok=np.isfinite(qm); q0=np.nanmean(qm[t<0])
    # observed onset: welas ql >0.01 g/kg sustained 60 s
    above=lw>1e-5; idx=next(i for i in range(len(t)) if above[i:i+60].all())
    on_obs=t[idx]
    # obs loss before onset, and in-cloud rate from onset+60 to last MBW
    pre=q0-np.nanmean(qm[(t>on_obs-30)&(t<=on_obs)&ok])
    w=ok&(t>on_obs+60)
    a=np.polyfit(t[w],qm[w],1)[0]*60*1e3   # g/kg/min observed total-water slope in cloud
    # emulator in-cloud slope: qs(Tg) based loss rate, per min
    def rate(off):
        r=[bl_loss(qs(Tg[k],p[k]),Tw[k]+off,p[k])/tau for k in np.where(w)[0]]
        return np.mean(r)*60*1e3
    # observed in-cloud loss rate = -(dqtot/dt) ; note dqs/dt also... qtot change is pure wall loss
    obs_rate=-a
    from scipy.optimize import brentq
    off_fit=brentq(lambda o: rate(o)-obs_rate,-2,0.26*0+0.6)
    Lp,on_p=emul(t,p,Tg,Tw,q0,off_p,tau); Lf,on_f=emul(t,p,Tg,Tw,q0,off_fit,tau)
    Lp_pre=q0-np.interp(on_obs,t,Lp)
    rows.append(dict(exp=e,on_obs=round(on_obs/60,1),tsat_BMM=tsat[e],on_noBL=round(t[np.argmax(q0>=qs(Tg,p))]/60,1),on_emul_paper=round(on_p/60,1),on_emul_fit=round(on_f/60,1),
        pre_loss_obs=round(pre*1e3,2),pre_loss_emul_paper=round(Lp_pre*1e3,2),incloud_rate_obs=round(obs_rate,3),rate_paper=round(rate(off_p),3),dT_paper=off_p,dT_fit_incloud=round(off_fit,3),TgTw=round(np.mean((Tg-Tw)[w]),2)))
df=pd.DataFrame(rows); pd.set_option("display.width",300); print(df.to_string(index=False))
print(df[["dT_fit_incloud","TgTw"]].describe().loc[["mean","std"]])
