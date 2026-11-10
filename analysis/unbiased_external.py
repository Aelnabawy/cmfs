import sys, warnings; warnings.filterwarnings('ignore')
exec(open('ext_compare.py').read().split("# ---- evaluation helpers ----")[0])
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
CLF={'logistic':lambda: LogisticRegression(C=1.0,max_iter=2000,class_weight='balanced'),'linear SVM':lambda: SVC(kernel='linear',C=0.1,class_weight='balanced'),
     'random forest':lambda: RandomForestClassifier(150,min_samples_leaf=3,class_weight='balanced',random_state=0),'kNN':lambda: KNeighborsClassifier(15),'naive Bayes':lambda: GaussianNB()}
rngB=np.random.default_rng(31)
def splits(y,groups,reps=2):
    if groups is None: return list(RepeatedStratifiedKFold(n_splits=5,n_repeats=reps,random_state=3).split(np.zeros(len(y)),y))
    return [s for r in range(reps) for s in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=r).split(np.zeros(len(y)),y,groups)]
def unit(df,cols):
    Z=np.column_stack([(df[c].values-np.nanmean(df[c].values))/(np.nanstd(df[c].values)+1e-9)*SIGN[c] for c in cols]); Z=np.where(np.isnan(Z),0,Z); s=Z.mean(1); y=df.y.values
    B=[roc_auc_score(y[i],s[i]) for i in [rngB.integers(0,len(y),len(y)) for _ in range(300)] if len(np.unique(y[i]))==2]
    return roc_auc_score(y,s),np.percentile(B,2.5),np.percentile(B,97.5)
def clf_mean(df,cols,groups):
    X=df[cols].values.astype(float); y=df.y.values; gr=None if groups is None else df[groups].values; res={k:[] for k in CLF}
    for tr,te in splits(y,gr):
        if len(np.unique(y[te]))<2: continue
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med); A=np.where(np.isnan(X[tr]),med,X[tr]); B=np.where(np.isnan(X[te]),med,X[te]); mu,sd=A.mean(0),A.std(0)+1e-9; A=(A-mu)/sd; B=(B-mu)/sd
        for n_,mk in CLF.items():
            m=mk().fit(A,y[tr]); s=m.decision_function(B) if hasattr(m,'decision_function') else m.predict_proba(B)[:,1]; res[n_].append(roc_auc_score(y[te],s))
    return {k:float(np.mean(v)) for k,v in res.items()}
def evaluate(panel,do_clf=True):
    out={}
    for dname,(df,groups) in EXT.items():
        cols=[c for c in dict.fromkeys(panel) if c in df.columns and df[c].notna().sum()>=10 and c in SIGN]
        if not cols: continue
        u=unit(df,cols); out[dname]=dict(features=cols,unit=u[0],unit_lo=u[1],unit_hi=u[2])
        if do_clf: out[dname]['clf']=clf_mean(df,cols,groups)
    return out
OUT={m:evaluate([str(x) for x in FULL[('LEFT',m)]]+[str(x) for x in FULL[('RIGHT',m)]]) for m in METH}
names_L=[str(x) for x in FULL[('LEFT','_names')]]; names_R=[str(x) for x in FULL[('RIGHT','_names')]]
NULL=[evaluate(list(rng.choice(names_L,11,replace=False))+list(rng.choice(names_R,11,replace=False)),do_clf=(r<10)) for r in range(40)]
pd.to_pickle(dict(OUT=OUT,NULL=NULL),'unbiased_external.pkl')
D=list(EXT); rows=[]
for m in METH:
    R=OUT[m]; row={'method':m,'covered':len(R)}
    for d in D: row[d]=R[d]['unit'] if d in R else np.nan
    row['mean unit-weighted']=np.nanmean([R[d]['unit'] for d in R]); row['mean 5-classifier']=np.nanmean([np.mean(list(R[d]['clf'].values())) for d in R])
    row['datasets above chance (unit CI>0.5)']=sum(R[d]['unit_lo']>0.5 for d in R); rows.append(row)
row={'method':'Random panels (null, 40 draws)','covered':np.mean([len(x) for x in NULL])}
for d in D: row[d]=np.mean([x[d]['unit'] for x in NULL if d in x])
row['mean unit-weighted']=np.mean([np.mean([v['unit'] for v in x.values()]) for x in NULL]); row['mean 5-classifier']=np.mean([np.mean([np.mean(list(v['clf'].values())) for v in x.values()]) for x in NULL[:10]])
row['datasets above chance (unit CI>0.5)']=np.mean([sum(v['unit_lo']>0.5 for v in x.values()) for x in NULL]); rows.append(row)
T=pd.DataFrame(rows); pd.set_option('display.width',320); pd.set_option('display.max_columns',30)
print(T.round(3).to_string(index=False)); T.to_csv('unbiased_external_table.csv',index=False)
