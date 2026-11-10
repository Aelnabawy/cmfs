import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
D=pd.read_pickle('/home/claude/sim/clinical_summary.pkl'); DCA,SUM=D['DCA'],D['SUM']
fig,ax=plt.subplots(2,2,figsize=(8.6,7.2)); col={'LEFT':'#2E6FAC','RIGHT':'#C0504D'}; ttl={'LEFT':'left-hemisphere strokes\n(cognitive tier)','RIGHT':'right-hemisphere strokes\n(three tiers)'}
for j,s in enumerate(['LEFT','RIGHT']):
    d=DCA[s]; p,y=d['pbar'],d['y']; a=ax[0,j]
    a.plot([0,1],[0,1],ls='--',c='grey',lw=1,label='Perfect calibration')
    q=pd.qcut(p,4,labels=False,duplicates='drop')
    for g in np.unique(q):
        m=q==g; mp,ob,n=p[m].mean(),y[m].mean(),m.sum(); z=1.96; cen=(ob+z*z/(2*n))/(1+z*z/n); hw=z*np.sqrt(ob*(1-ob)/n+z*z/(4*n*n))/(1+z*z/n)
        a.errorbar(mp,ob,yerr=[[ob-(cen-hw)],[(cen+hw)-ob]],fmt='o',c=col[s],ms=6,capsize=3,label='Observed, quartiles of risk (95% CI)' if g==0 else None)
    lw=lowess(y,p,frac=0.8,it=0); a.plot(lw[:,0],lw[:,1],c=col[s],lw=1.5,alpha=0.7,label='Smoothed')
    sl=SUM[s]['cal_slope'][0]; br=SUM[s]['Brier'][0]
    a.set(xlim=(0,1),ylim=(-0.05,1.05),xlabel='Predicted probability of stroke',ylabel='Observed proportion with stroke',title=f'{"AB"[j]}. Calibration, {ttl[s]}')
    a.text(0.97,0.05,f'Calibration slope {sl:.2f}\nBrier score {br:.3f}',ha='right',va='bottom',fontsize=8,bbox=dict(fc='white',ec='lightgrey'))
    a.legend(fontsize=7,loc='upper left',frameon=False)
    b=ax[1,j]; th=d['th']
    b.plot(th,d['model'],c=col[s],lw=2,label='CMFS design'); b.plot(th,d['all'],c='black',lw=1,ls=':',label='Treat all as stroke'); b.axhline(0,c='grey',lw=1,label='Treat none')
    b.set(xlim=(0.05,0.6),ylim=(-0.05,max(0.32,d['prev']+0.02)),xlabel='Threshold probability',ylabel='Net benefit',title=f'{"CD"[j]}. Decision curve, {ttl[s]}')
    b.legend(fontsize=7,frameon=False); b.text(0.59,max(0.32,d['prev']+0.02)*0.93,f'Sample prevalence {d["prev"]:.2f}',ha='right',fontsize=7.5,color='dimgrey')
for a in ax.flat: a.spines[['top','right']].set_visible(False)

fig.tight_layout(); fig.savefig('/mnt/user-data/outputs/CMFS_clinical_validity.png',dpi=300); fig.savefig('/mnt/user-data/outputs/CMFS_clinical_validity.pdf'); print('saved')
