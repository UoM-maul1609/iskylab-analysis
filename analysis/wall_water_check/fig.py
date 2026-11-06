import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
exec(open(_os.path.join(REPO,'analysis','wall_water_check','wall.py')).read().split("rows=[]")[0])
out=OUTDIR+"/analysis"; os.makedirs(out,exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
t,p,Tg,Tw,qm,lw=load("013"); m=t/60
ax[0].plot(m,qm*1e3,'k',lw=1.5,label="MBW total water")
ax[0].plot(m,qs(Tg,p)*1e3,color='C0',label="q$_s$(T$_{gas}$)")
ax[0].plot(m,qs(Tw,p)*1e3,color='C1',ls='--',label="q$_s$(T$_{wall}$)")
ax[0].plot(m,(qs(Tg,p)+lw)*1e3,color='C2',lw=0.8,label="q$_s$(T$_{gas}$) + WELAS q$_l$")
ax[0].axhline(np.nanmean(qm[t<0])*1e3,color='grey',lw=0.8,ls=':',label="initial water")
ax[0].set_xlabel("Time (min)"); ax[0].set_ylabel("Water (g kg$^{-1}$)"); ax[0].set_title("(a) Exp013: total water from MBW"); ax[0].legend(fontsize=7,frameon=False)
rates=[]; exps=["006","007","008","009","010","011","013","014","015","016","018","019"]
for e in exps:
    t,p,Tg,Tw,qm,lw=load(e); ok=np.isfinite(qm); above=lw>1e-5
    idx=next(i for i in range(len(t)) if above[i:i+60].all()); on=t[idx]; w=ok&(t>on+60)
    rates.append(-np.polyfit(t[w],qm[w],1)[0]*60*1e3)
ax[1].bar(exps,rates,color='C0'); ax[1].axhline(np.median(rates),color='k',ls='--',lw=0.8)
ax[1].set_ylabel("In-cloud total-water loss (g kg$^{-1}$ min$^{-1}$)"); ax[1].set_title("(b) Wall loss rate, fan-on runs (median %.3f)"%np.median(rates)); ax[1].tick_params(axis='x',rotation=90)
fig.tight_layout(); fig.savefig(out+"/wall_water_check.png",dpi=150); print(out, np.round(rates,3))
