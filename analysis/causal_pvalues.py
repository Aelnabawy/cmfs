import sys, warnings; warnings.filterwarnings('ignore')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
CI=pd.read_pickle('comparison_internal.pkl'); FULL=CI['FULL']
ibF=EnvImputer().fit(Xb_all[cohg]); inF=EnvImputer().fit(Xn_all[cohg]); igF=EnvImputer().fit(Gm[cohg]); drawF=np.random.default_rng(5).random(len(lab))<0.5
EXTRA=['TM: Part B Time (sec)','HVLT: Total Recall','RCFT: Copy-Raw']; rng=np.random.default_rng(2026); rows=[]
for sname,(sv,BC) in SIDES.items():
    k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]; T,X,e,names,fam=side_data(k,sv,BC,ibF.transform(Xb_all[k]),inF.transform(Xn_all[k]),igF.transform(Gm[k]),drawF[k])
    y=Y[k]; c=C5[k]; cg=np.column_stack([c,FT[k]])
    UB=crossfit_index('lda',T['B'],e,y,n_splits=4)[:,None]; UN=crossfit_index('lda',T['N'],e,y,n_splits=4)[:,None]
    tier_of=lambda f: 'G' if f.startswith('paretic ') else ('N' if f in NEURO else 'B')
    cols={'B':list(BC),'N':list(NEURO),'G':['paretic '+g for g in GN]}
    def design(t):
        cc=cg if t=='G' else c; up=[] if t=='B' else ([UB] if t=='N' else [UB,UN]); return cc,up
    def paths(t,j,idx=None):
        cc,up=design(t); xv=T[t][:,j]; i=np.arange(len(e)) if idx is None else idx
        A=sm.add_constant(np.column_stack([e[i],cc[i]])); z=(xv[i]-xv[i].mean())/xv[i].std(); oa=sm.OLS(z,A).fit()
        B=sm.add_constant(np.column_stack([z,e[i],cc[i]]+[u[i] for u in up])); ob=sm.OLS(y[i],B).fit()
        return oa,ob
    # q-values: joint-significance p over every candidate in the tier
    Q={}
    for t in 'BNG':
        pj=[]
        for j in range(T[t].shape[1]):
            oa,ob=paths(t,j); pj.append(max(oa.pvalues[1],ob.pvalues[1]))
        q=multipletests(pj,method='fdr_bh')[1]; Q.update({cols[t][j]:q[j] for j in range(len(pj))})
    tot=sm.OLS(y,sm.add_constant(np.column_stack([e,c]))).fit().params[1]
    panel=[str(f) for f in FULL[(sname,'CMFS (tiered causal mediation)')]]+[f for f in EXTRA if f not in map(str,FULL[(sname,'CMFS (tiered causal mediation)')])]
    for f in panel:
        t=tier_of(f); j=cols[t].index(f); oa,ob=paths(t,j); a,b=oa.params[1],ob.params[1]; sa,sb=oa.bse[1],ob.bse[1]; ab=a*b
        zs=ab/np.sqrt(b**2*sa**2+a**2*sb**2); ps=2*norm.sf(abs(zs)); Bs=[]
        i1,i0=np.where(e==1)[0],np.where(e==0)[0]
        for _ in range(2000):
            i=np.r_[rng.choice(i1,len(i1)),rng.choice(i0,len(i0))]; oa_,ob_=paths(t,j,i); Bs.append(oa_.params[1]*ob_.params[1])
        Bs=np.array(Bs); pb=min(1.0,2*min((Bs<=0).mean(),(Bs>=0).mean()))
        rows.append(dict(side=sname,tier={'B':'cerebrovascular','N':'cognitive','G':'paretic gait'}[t],feature=f,in_CMFS_panel=f in map(str,FULL[(sname,'CMFS (tiered causal mediation)')]),
                         a_stroke_to_feature_SD=a,p_a=oa.pvalues[1],b_feature_to_speed_ms_per_SD=b,p_b=ob.pvalues[1],indirect_ms=ab,ci_lo=np.percentile(Bs,2.5),ci_hi=np.percentile(Bs,97.5),
                         share_of_total=ab/tot,p_joint=max(oa.pvalues[1],ob.pvalues[1]),p_sobel=ps,p_bootstrap=pb,q_joint_FDR_in_tier=Q[f]))
R=pd.DataFrame(rows); R.to_csv('causal_pvalues.csv',index=False); pd.set_option('display.width',260); pd.set_option('display.max_colwidth',34)
print(R[['side','tier','feature','in_CMFS_panel','a_stroke_to_feature_SD','p_a','b_feature_to_speed_ms_per_SD','p_b','indirect_ms','ci_lo','ci_hi','share_of_total','p_joint','p_sobel','p_bootstrap','q_joint_FDR_in_tier']].round(4).to_string(index=False))
print('total effects:',{s:round(float(sm.OLS(Y[cohg[(lab[cohg]==0)|(lab[cohg]==v)]],sm.add_constant(np.column_stack([(lab[cohg[(lab[cohg]==0)|(lab[cohg]==v)]]==v).astype(float),C5[cohg[(lab[cohg]==0)|(lab[cohg]==v)]]]))).fit().params[1]),3) for s,(v,_) in SIDES.items()})
