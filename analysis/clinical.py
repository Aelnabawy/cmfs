import sys, time, warnings; warnings.filterwarnings('ignore')
src=open('multi_tier.py').read().split("SP=list(RepeatedStratifiedKFold")[0]; exec(src)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
DESIGN={'LEFT':(('N',),('N',)),'RIGHT':(('B','N','G'),('B','G'))}   # (tier order, transformed tiers) of the published best designs
SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(cohg,lab[cohg])); rng=np.random.default_rng(321)
OUT={s:dict(ridge=np.full((10,len(lab)),np.nan),prob=np.full((10,len(lab)),np.nan),thr=np.full((10,len(lab)),np.nan)) for s in SIDES}
for f,(trp,tep) in enumerate(SP):
    r=f//5; tr,te=cohg[trp],cohg[tep]; ib=EnvImputer().fit(Xb_all[tr]); inn=EnvImputer().fit(Xn_all[tr]); ig=EnvImputer().fit(Gm[tr])
    B_tr,B_te,N_tr,N_te,G_tr,G_te=ib.transform(Xb_all[tr]),ib.transform(Xb_all[te]),inn.transform(Xn_all[tr]),inn.transform(Xn_all[te]),ig.transform(Gm[tr]),ig.transform(Gm[te])
    for sname,(sv,BC) in SIDES.items():
        bi=[bcols.index(c_) for c_ in BC]; ktr=np.where((lab[tr]==0)|(lab[tr]==sv))[0]; kte=np.where((lab[te]==0)|(lab[te]==sv))[0]
        e=(lab[tr][ktr]==sv).astype(float); y=Y[tr][ktr]; c=C5[tr][ktr]; cg=np.column_stack([c,FT[tr][ktr]])
        T_tr={'B':B_tr[ktr][:,bi],'N':N_tr[ktr],'G':paretic(G_tr[ktr],lab[tr][ktr],sv,rng.random(len(ktr))<0.5)}
        T_te={'B':B_te[kte][:,bi],'N':N_te[kte],'G':paretic(G_te[kte],lab[te][kte],sv,rng.random(len(kte))<0.5)}
        fams={'B':np.array([family(x) for x in BC]),'N':None,'G':None}; order,trans=DESIGN[sname]
        sel,F_tr,F_te,I_tr,I_te=run(order,trans,T_tr,T_te,e,y,c,cg,fams)
        A=np.column_stack(F_tr+I_tr); Bt=np.column_stack(F_te+I_te); mu,sd=A.mean(0),A.std(0)+1e-9; A,Bt=(A-mu)/sd,(Bt-mu)/sd
        idx_te=te[kte]
        OUT[sname]['ridge'][r,idx_te]=RidgeClassifierCV(alphas=[0.1,1,10,100,1000]).fit(A,e).decision_function(Bt)
        lr=LogisticRegressionCV(Cs=[0.01,0.03,0.1,0.3,1,3,10],cv=StratifiedKFold(4,shuffle=True,random_state=f),scoring='neg_log_loss',max_iter=2000).fit(A,e)
        OUT[sname]['prob'][r,idx_te]=lr.predict_proba(Bt)[:,1]; OUT[sname]['thr'][r,idx_te]=e.mean()
pd.to_pickle(OUT,'clinical_oof.pkl')
def metrics(yv,p,t):
    pred=p>=t; tp=np.sum(pred&(yv==1)); fn=np.sum(~pred&(yv==1)); tn=np.sum(~pred&(yv==0)); fp=np.sum(pred&(yv==0))
    se=tp/(tp+fn); sp=tn/(tn+fp); return dict(sens=se,spec=sp,PPV=tp/max(tp+fp,1),NPV=tn/max(tn+fn,1),LRpos=se/max(1-sp,1e-9),LRneg=(1-se)/max(sp,1e-9))
def calib(yv,p):
    lp=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    import statsmodels.api as sm
    slope=sm.Logit(yv,sm.add_constant(lp)).fit(disp=0).params[1]; inter=sm.GLM(yv,np.ones((len(yv),1)),offset=lp,family=sm.families.Binomial()).fit().params[0]
    return dict(cal_intercept=inter,cal_slope=slope,Brier=np.mean((p-yv)**2))
rngb=np.random.default_rng(11); SUM={}; DCA={}
for sname,(sv,_) in SIDES.items():
    k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]; yv=(lab[k]==sv).astype(float); O=OUT[sname]
    per=[]; 
    for r in range(10):
        p=O['prob'][r,k]; d=dict(AUC_ridge=roc_auc_score(yv,O['ridge'][r,k]),AUC_logistic=roc_auc_score(yv,p)); d.update(metrics(yv,p,O['thr'][r,k])); d.update(calib(yv,p)); per.append(d)
    P=pd.DataFrame(per).mean(); pbar=np.nanmean(O['prob'][:,k],0); tbar=np.nanmean(O['thr'][:,k],0)
    boots=[]
    for _ in range(2000):
        i=rngb.integers(0,len(yv),len(yv))
        if len(np.unique(yv[i]))<2: continue
        d=dict(AUC_logistic=roc_auc_score(yv[i],pbar[i])); d.update(metrics(yv[i],pbar[i],tbar[i])); d.update(calib(yv[i],pbar[i])); boots.append(d)
    Bd=pd.DataFrame(boots); SUM[sname]={m:(P[m],Bd[m].quantile(.025) if m in Bd else np.nan,Bd[m].quantile(.975) if m in Bd else np.nan) for m in P.index}
    th=np.arange(0.05,0.61,0.01); prev=yv.mean()
    nb=[(np.sum((pbar>=t)&(yv==1))-np.sum((pbar>=t)&(yv==0))*t/(1-t))/len(yv) for t in th]; nb_all=[prev-(1-prev)*t/(1-t) for t in th]
    DCA[sname]=dict(th=th,model=np.array(nb),all=np.array(nb_all),prev=prev,pbar=pbar,y=yv,n=(int(yv.sum()),int((1-yv).sum())))
    print(f'\n=== {sname}-hemisphere strokes ({int(yv.sum())} vs {int((1-yv).sum())}; sample prevalence {prev:.2f}) ===')
    for m,(v,lo,hi) in SUM[sname].items(): print(f'  {m:14s} {v:.3f}' + (f'  [{lo:.3f}, {hi:.3f}]' if not np.isnan(lo) else ''))
    useful=th[(np.array(nb)>np.maximum(np.array(nb_all),0)+0.005)]
    print(f'  decision curve: net benefit above treat-all and treat-none for thresholds {useful.min():.2f}-{useful.max():.2f}' if len(useful) else '  decision curve: no threshold range with net benefit')
pd.to_pickle(dict(SUM=SUM,DCA=DCA),'clinical_summary.pkl')
