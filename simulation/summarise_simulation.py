"""Summarise the simulation study (Section 5.8): tables and Figure 8."""
import pickle, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='/home/claude/simstudy'; R=pd.concat([pd.DataFrame(pickle.load(open(f,'rb'))) for f in glob.glob(f'{D}/res_*.pkl')],ignore_index=True)
SHORT={'CMFS (tiered causal mediation)':'CMFS','Single-mediator screening (HIMA-style)':'HIMA-style','Markov boundary (IAMB)':'Markov boundary','Association (Mann-Whitney)':'Association','LASSO':'LASSO','mRMR':'mRMR','Random forest importance':'Random forest'}
R['m']=R.method.map(SHORT).fillna(R.method); M=list(SHORT.values())
dec=['selected: lesion marker','selected: outcome-only','selected: measured-confounder proxy','selected: unmeasured-confounder proxy','selected: collider','selected: noise']
def ci(v): v=np.asarray(v,float); v=v[~np.isnan(v)]; return v.mean(),1.96*v.std(ddof=1)/np.sqrt(len(v))
rows=[]
for s,lbl in [('n16_40','16 vs 40 (CVES-like)'),('n50_50','50 vs 50'),('n150_150','150 vs 150')]:
    for m in M:
        x=R[(R.scenario==s)&(R.m==m)]; r=dict(sample=lbl,method=m,reps=len(x))
        for k in ['mediator recall','mediator precision','AUC']+dec: r[k]=x[k].mean()
        rows.append(r)
MAIN=pd.DataFrame(rows)
SENS=[]
for s,lbl in [('n50_50','Reference (50 vs 50)'),('weak_50','Weak mediation'),('strongU_50','Strong unmeasured confounding')]:
    for m in M:
        x=R[(R.scenario==s)&(R.m==m)]; SENS.append(dict(scenario=lbl,method=m,**{k:x[k].mean() for k in ['mediator recall','AUC','selected: unmeasured-confounder proxy','selected: lesion marker','selected: collider']}))
for m in R[R.scenario=='order_50'].m.unique():
    x=R[(R.scenario=='order_50')&(R.m==m)]; SENS.append(dict(scenario='Tier order',method=m,**{k:x[k].mean() for k in ['mediator recall','AUC','selected: unmeasured-confounder proxy','selected: lesion marker','selected: collider']}))
SENS=pd.DataFrame(SENS)
pd.set_option('display.width',250); print(MAIN.round(3).to_string(index=False)); print(); print(SENS.round(3).to_string(index=False))
# ---------- Figure 8 ----------
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5}); col={'CMFS':'#C0504D','HIMA-style':'#E8A33D','Markov boundary':'#7F7F7F','Association':'#2E6FAC','LASSO':'#4BACC6','mRMR':'#8064A2','Random forest':'#9BBB59'}
fig,ax=plt.subplots(2,2,figsize=(9.4,7.4)); xs=['16 vs 40 (CVES-like)','50 vs 50','150 vs 150']
for m in M:
    y=[MAIN[(MAIN['sample']==s)&(MAIN.method==m)]['mediator recall'].values[0] for s in xs]; ax[0,0].plot(range(3),y,'-o',c=col[m],lw=2.4 if m=='CMFS' else 1.3,ms=4,label=m,zorder=3 if m=='CMFS' else 2)
    y=[MAIN[(MAIN['sample']==s)&(MAIN.method==m)]['AUC'].values[0] for s in xs]; ax[1,0].plot(range(3),y,'-o',c=col[m],lw=2.4 if m=='CMFS' else 1.3,ms=4,label=m,zorder=3 if m=='CMFS' else 2)
for a,t,yl in [(ax[0,0],'A. Recovery of the true mediators','Share of the 6 true mediators selected'),(ax[1,0],'C. Classification on independent test data','AUC')]:
    a.set_xticks(range(3)); a.set_xticklabels(['16 vs 40\n(CVES-like)','50 vs 50','150 vs 150']); a.set_title(t,loc='left',fontsize=9.5); a.set_ylabel(yl); a.set_xlabel('Strokes vs controls')
ax[0,0].set_ylim(0,1.15); ax[0,0].legend(fontsize=7,frameon=False,ncol=2,loc='upper left')
H=MAIN[MAIN['sample']=='50 vs 50'].set_index('method').loc[M,dec].values; im=ax[0,1].imshow(H,cmap='Reds',vmin=0,vmax=1,aspect='auto')
for i in range(H.shape[0]):
    for j in range(H.shape[1]): ax[0,1].text(j,i,f'{100*H[i,j]:.0f}',ha='center',va='center',fontsize=7.5,color='white' if H[i,j]>0.6 else 'black')
ax[0,1].set_xticks(range(len(dec))); ax[0,1].set_xticklabels(['Lesion\nmarker','Outcome\nonly','Measured\nconfounder\nproxy','Unmeasured\nconfounder\nproxy','Collider','Noise'],fontsize=6.8); ax[0,1].set_yticks(range(len(M))); ax[0,1].set_yticklabels(M)
ax[0,1].set_title('B. Decoys selected (% of each type, 50 vs 50)',loc='left',fontsize=9.5)
S=SENS[SENS.method.str.startswith('CMFS')].copy(); S['lab']=np.where(S.scenario=='Tier order',S.method.str.replace('CMFS, ',''),S.scenario)
order=['Reference (50 vs 50)','Weak mediation','Strong unmeasured confounding','correct tier order','cognitive and gait swapped','tier order reversed']; S=S.set_index('lab').loc[[o for o in order if o in S.lab.values]]
yy=np.arange(len(S)); ax[1,1].barh(yy-0.2,S['mediator recall'],0.38,color='#C0504D',label='True mediators selected'); ax[1,1].barh(yy+0.2,S['selected: unmeasured-confounder proxy'],0.38,color='#7F7F7F',label='Unmeasured-confounder proxies selected')
ax[1,1].set_yticks(yy); ax[1,1].set_yticklabels([o[0].upper()+o[1:] for o in S.index]); ax[1,1].invert_yaxis(); ax[1,1].set_xlim(0,1); ax[1,1].set_xlabel('Share selected'); ax[1,1].legend(fontsize=7,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=2); ax[1,1].set_title('D. CMFS under harder conditions (50 vs 50)',loc='left',fontsize=9.5)
for a in [ax[0,0],ax[1,0],ax[1,1]]: a.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('/mnt/user-data/outputs/CMFS_simulation_study.png',dpi=300); fig.savefig('/mnt/user-data/outputs/CMFS_simulation_study.pdf')
from openpyxl import load_workbook
out='/mnt/user-data/outputs/CMFS_simulation_study.xlsx'
notes=['Simulation study (Section 5.8). Data shaped like CVES: feature correlations copied from the real cerebrovascular (45), cognitive (16) and gait (12) tiers; case-control sampling from a population in which age, hypertension and smoking affect stroke and age, BMI and smoking affect walking speed.',
'Planted in every tier: 2 true mediators (with a cerebrovascular -> cognitive -> gait chain), 1-2 lesion markers, 1 outcome-only predictor, 1 measured-confounder proxy (age), 1 unmeasured-confounder proxy (hidden factor affecting stroke), 1 collider (stroke -> feature <- walking speed); remaining features are correlated noise.',
'Effects (reference): stroke -> mediator 0.5 SD, mediator -> walking speed -0.04 m/s per SD, chain 0.5, stroke -> lesion marker 0.8 SD, direct effect -0.10 m/s, residual SD 0.12 m/s. Weak mediation: 0.3 SD and -0.025 m/s. Strong unmeasured confounding: hidden factor effect on stroke 1.5 (log-odds per SD) and on its proxy 1.2 SD.',
'200 replicates per scenario; every method selects up to 11 features exactly as in the robustness benchmark; AUC from ridge classification on an independent test set of 500 strokes and 500 controls.']
with pd.ExcelWriter(out,engine='openpyxl') as w:
    pd.DataFrame({'Notes':notes}).to_excel(w,sheet_name='README',index=False); MAIN.to_excel(w,sheet_name='Main (by sample size)',index=False); SENS.to_excel(w,sheet_name='Sensitivity',index=False)
print('saved')
