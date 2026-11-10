import sys, time, warnings; warnings.filterwarnings('ignore')
exec(open('multi_tier.py').read().split("bcols=sorted(set(LB+RB))")[0])
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, norm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score
K_TOTAL=11; rng0=np.random.default_rng(99)
bcols=sorted(set(LB+RB)); Xb_all=Xraw[bcols].values.astype(float); Xn_all=Xraw[NEURO].values.astype(float)
# ---------------- selection methods (X: standardised 73 candidates, names, tiers) ----------------
def m_assoc(X,e,**k): return list(np.argsort([mannwhitneyu(X[e==1,j],X[e==0,j]).pvalue for j in range(X.shape[1])])[:K_TOTAL])
def m_lasso(X,e,**k):
    m=LogisticRegressionCV(Cs=10,penalty='l1',solver='liblinear',cv=4,class_weight='balanced',scoring='roc_auc',max_iter=2000).fit(X,e); c=np.abs(m.coef_[0])
    o=[j for j in np.argsort(-c) if c[j]>0][:K_TOTAL]; return o if o else list(np.argsort(-c)[:1])
def m_mrmr(X,e,**k):
    rel=mutual_info_classif(X,e,random_state=0); C=np.abs(np.corrcoef(X.T)); sel=[int(np.argmax(rel))]
    while len(sel)<K_TOTAL:
        sc=[rel[j]-C[j,sel].mean() if j not in sel else -np.inf for j in range(X.shape[1])]; sel.append(int(np.argmax(sc)))
    return sel
def m_rf(X,e,**k): return list(np.argsort(-RandomForestClassifier(500,class_weight='balanced',random_state=0,min_samples_leaf=2).fit(X,e).feature_importances_)[:K_TOTAL])
def m_hima(X,e,y=None,cc=None,**k):
    a,pa=coef_p(X,np.column_stack([e,cc]),1); b=np.array([ols_b(y,np.column_stack([X[:,j],e,cc]))[1] for j in range(X.shape[1])]); s=a*b
    el=set(np.where(pa<0.1)[0]); return [j for j in np.argsort(s) if j in el or len(el)<K_TOTAL][:K_TOTAL]
def pcor_p(X,e,j,S):
    Z=np.column_stack([np.ones(len(e))]+([X[:,S]] if S else [])); rx=X[:,j]-Z@np.linalg.lstsq(Z,X[:,j],rcond=None)[0]; re=e-Z@np.linalg.lstsq(Z,e,rcond=None)[0]
    r=np.corrcoef(rx,re)[0,1]; z=np.arctanh(np.clip(r,-0.999,0.999))*np.sqrt(max(len(e)-len(S)-3,1)); return 2*norm.sf(abs(z)),abs(r)
def m_iamb(X,e,**k):
    MB=[]
    while len(MB)<K_TOTAL:
        cand=[(pcor_p(X,e,j,MB),j) for j in range(X.shape[1]) if j not in MB]; (p,r),j=max(cand,key=lambda t:t[0][1])
        if p>0.05: break
        MB.append(j)
    for j in list(MB):
        if pcor_p(X,e,j,[m for m in MB if m!=j])[0]>0.05: MB.remove(j)
    return MB if MB else [int(np.argmax([abs(np.corrcoef(X[:,j],e)[0,1]) for j in range(X.shape[1])]))]
def m_cmfs(X,e,T=None,y=None,c=None,cg=None,fams=None,**k):
    sel,_,_,_,_=run(('B','N','G'),(),T,T,e,y,c,cg,fams); nb,nn=T['B'].shape[1],T['N'].shape[1]; off={'B':0,'N':nb,'G':nb+nn}; return [off[t]+int(j) for t in ('B','N','G') for j in sel[t]]
METHODS={'CMFS (tiered causal mediation)':m_cmfs,'Single-mediator screening (HIMA-style)':m_hima,'Markov boundary (IAMB)':m_iamb,'Association (Mann-Whitney)':m_assoc,'LASSO':m_lasso,'mRMR':m_mrmr,'Random forest importance':m_rf}
def side_data(idx,sv,BC,B,N,G,draw):
    bi=[bcols.index(c_) for c_ in BC]; e=(lab[idx]==sv).astype(float)
    T={'B':B[:,bi],'N':N,'G':paretic(G,lab[idx],sv,draw)}; names=list(BC)+list(NEURO)+['paretic '+g for g in GN]
    X=np.column_stack([T['B'],T['N'],T['G']]); mu,sd=X.mean(0),X.std(0)+1e-9
    return T,(X-mu)/sd,e,names,np.array([family(x) for x in BC])
# ---------------- 1. internal nested comparison ----------------
rng=np.random.default_rng(321); SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(cohg,lab[cohg])); INT={}; SELF={}; t0=time.time()
for f,(trp,tep) in enumerate(SP):
    tr,te=cohg[trp],cohg[tep]; ib=EnvImputer().fit(Xb_all[tr]); inn=EnvImputer().fit(Xn_all[tr]); ig=EnvImputer().fit(Gm[tr])
    for sname,(sv,BC) in SIDES.items():
        ktr=tr[(lab[tr]==0)|(lab[tr]==sv)]; kte=te[(lab[te]==0)|(lab[te]==sv)]
        Ttr,Xtr,e,names,fam=side_data(ktr,sv,BC,ib.transform(Xb_all[ktr]),inn.transform(Xn_all[ktr]),ig.transform(Gm[ktr]),rng.random(len(ktr))<0.5)
        bi=[bcols.index(c_) for c_ in BC]; Bte=ib.transform(Xb_all[kte])[:,bi]; Pte=paretic(ig.transform(Gm[kte]),lab[kte],sv,rng.random(len(kte))<0.5)
        Xte_raw=np.column_stack([Bte,inn.transform(Xn_all[kte]),Pte]); Xtr_raw=np.column_stack([Ttr['B'],Ttr['N'],Ttr['G']]); mu,sd=Xtr_raw.mean(0),Xtr_raw.std(0)+1e-9; Xte=(Xte_raw-mu)/sd
        ete=(lab[kte]==sv).astype(float); y=Y[ktr]; c=C5[ktr]; cg=np.column_stack([c,FT[ktr]])
        for mn,fn in METHODS.items():
            s=fn(Xtr,e,T=Ttr,y=y,c=c,cg=cg,cc=cg,fams={'B':fam,'N':None,'G':None})
            INT.setdefault((sname,mn),[]).append(auc(Xtr[:,s],e,Xte[:,s],ete)); SELF.setdefault((sname,mn),[]).append(tuple(np.array(names)[s]))
        for r in range(3):
            s=list(rng0.choice(Xtr.shape[1],K_TOTAL,replace=False)); INT.setdefault((sname,'Random panels (null)'),[]).append(auc(Xtr[:,s],e,Xte[:,s],ete))
    if f%10==9: print(f'  fold {f+1}/50 ({time.time()-t0:.0f}s)',flush=True)
# ---------------- 2. full-data panels ----------------
FULL={}; ibF=EnvImputer().fit(Xb_all[cohg]); inF=EnvImputer().fit(Xn_all[cohg]); igF=EnvImputer().fit(Gm[cohg]); drawF=np.random.default_rng(5).random(len(lab))<0.5
for sname,(sv,BC) in SIDES.items():
    k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]; T,X,e,names,fam=side_data(k,sv,BC,ibF.transform(Xb_all[k]),inF.transform(Xn_all[k]),igF.transform(Gm[k]),drawF[k])
    for mn,fn in METHODS.items(): FULL[(sname,mn)]=list(np.array(names)[fn(X,e,T=T,y=Y[k],c=C5[k],cg=np.column_stack([C5[k],FT[k]]),cc=np.column_stack([C5[k],FT[k]]),fams={'B':fam,'N':None,'G':None})])
    FULL[(sname,'_names')]=names
pd.to_pickle(dict(INT=INT,SEL=SELF,FULL=FULL),'comparison_internal.pkl')
print(f"\n{'method':42s} {'LEFT AUC':>9s} {'stab':>5s} {'RIGHT AUC':>10s} {'stab':>5s}")
for mn in list(METHODS)+['Random panels (null)']:
    out=[]
    for s_ in SIDES:
        v=SELF.get((s_,mn)); J=np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for i,a in enumerate(v) for b in v[i+1:]]) if v else np.nan
        out+= [np.nanmean(INT[(s_,mn)]),J]
    print(f'{mn:42s} {out[0]:9.3f} {out[1]:5.2f} {out[2]:10.3f} {out[3]:5.2f}')
for k_,v in FULL.items():
    if k_[1]!='_names': print(k_,'->',v)
