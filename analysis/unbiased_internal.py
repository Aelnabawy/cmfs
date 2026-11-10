import sys, time, warnings; warnings.filterwarnings('ignore')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
CLF={'logistic':lambda: LogisticRegression(C=1.0,max_iter=2000),'linear SVM':lambda: SVC(kernel='linear',C=0.1),'random forest':lambda: RandomForestClassifier(200,min_samples_leaf=2,random_state=0),
     'kNN':lambda: KNeighborsClassifier(7),'naive Bayes':lambda: GaussianNB()}
def unit_score(Xtr,e,Xte):
    d=np.sign(Xtr[e==1].mean(0)-Xtr[e==0].mean(0)); d[d==0]=1; return Xte@d/len(d)
def clf_aucs(Xtr,e,Xte,ete):
    out={}
    for n_,mk in CLF.items():
        m=mk().fit(Xtr,e); s=m.decision_function(Xte) if hasattr(m,'decision_function') else m.predict_proba(Xte)[:,1]; out[n_]=roc_auc_score(ete,s)
    return out
CI=pd.read_pickle('comparison_internal.pkl'); SELF=CI['SEL']; METH=[m for (s,m) in CI['FULL'] if s=='LEFT' and m!='_names']
rng=np.random.default_rng(321); rngN=np.random.default_rng(77); SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(cohg,lab[cohg])); U={}; t0=time.time()
for f,(trp,tep) in enumerate(SP):
    tr,te=cohg[trp],cohg[tep]; ib=EnvImputer().fit(Xb_all[tr]); inn=EnvImputer().fit(Xn_all[tr]); ig=EnvImputer().fit(Gm[tr])
    for sname,(sv,BC) in SIDES.items():
        ktr=tr[(lab[tr]==0)|(lab[tr]==sv)]; kte=te[(lab[te]==0)|(lab[te]==sv)]
        Ttr,Xtr,e,names,fam=side_data(ktr,sv,BC,ib.transform(Xb_all[ktr]),inn.transform(Xn_all[ktr]),ig.transform(Gm[ktr]),rng.random(len(ktr))<0.5)
        bi=[bcols.index(c_) for c_ in BC]; Pte=paretic(ig.transform(Gm[kte]),lab[kte],sv,rng.random(len(kte))<0.5)
        Xte_raw=np.column_stack([ib.transform(Xb_all[kte])[:,bi],inn.transform(Xn_all[kte]),Pte]); Xtr_raw=np.column_stack([Ttr['B'],Ttr['N'],Ttr['G']]); mu,sd=Xtr_raw.mean(0),Xtr_raw.std(0)+1e-9; Xte=(Xte_raw-mu)/sd
        ete=(lab[kte]==sv).astype(float)
        if len(np.unique(ete))<2: continue
        panels={m:[names.index(x) for x in SELF[(sname,m)][f]] for m in METH}
        for r in range(3): panels[f'_null{r}']=list(rngN.choice(Xtr.shape[1],K_TOTAL,replace=False))
        for m,s in panels.items():
            key=m if not m.startswith('_null') else 'Random panels (null)'
            U.setdefault((sname,key,'unit-weighted score'),[]).append(roc_auc_score(ete,unit_score(Xtr[:,s],e,Xte[:,s])))
            for cn,v in clf_aucs(Xtr[:,s],e,Xte[:,s],ete).items(): U.setdefault((sname,key,cn),[]).append(v)
    if f%10==9: print(f'  fold {f+1}/50 ({time.time()-t0:.0f}s)',flush=True)
pd.to_pickle(U,'unbiased_internal.pkl')
R=pd.DataFrame([dict(side=k[0],method=k[1],predictor=k[2],AUC=np.mean(v)) for k,v in U.items()])
W=R.pivot_table(index='method',columns=['side','predictor'],values='AUC')
out=pd.DataFrame({('LEFT','unit-weighted'):W[('LEFT','unit-weighted score')],('LEFT','5-classifier mean'):W['LEFT'][list(CLF)].mean(1),
                  ('RIGHT','unit-weighted'):W[('RIGHT','unit-weighted score')],('RIGHT','5-classifier mean'):W['RIGHT'][list(CLF)].mean(1)})
print(out.round(3).sort_values(('RIGHT','unit-weighted'),ascending=False).to_string())
print('\nper classifier (RIGHT):'); print(W['RIGHT'][list(CLF)].round(3).to_string()); print('\nper classifier (LEFT):'); print(W['LEFT'][list(CLF)].round(3).to_string())
