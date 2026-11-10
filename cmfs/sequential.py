"""Proposed method 2: sequential pathway selection (blood_j -> neuro_k -> outcome)."""
import sys, time; sys.path.insert(0,'.')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from collections import Counter
exec(open('run_cves.py').read().split("model = CMFS")[0].replace("stage = sys.argv[1]","stage='seq'"))
from cmfs.data import EnvImputer
E=df.stroke.values.astype(float); Yv=df.speed.values; Cm=df[['age','male','bmi','pack_years','htn']].astype(float); Cm=((Cm-Cm.mean())/Cm.std()).fillna(0).values
env=((df.enrolment-df.enrolment.mean())/df.enrolment.std()).values; Cc=np.column_stack([Cm,env]); sd_=df.side.values
B_cols=[c for c in BLOOD if df[c].isna().mean()<=0.5 and df[c].nunique()>1]; N_cols=NEURO; nb=len(B_cols)
Xall=df[B_cols+N_cols].astype(float).values
def tstat(Xm,yv,j):
    X1=np.column_stack([np.ones(len(yv)),Xm]); b=np.linalg.lstsq(X1,yv,rcond=None)[0]; r=yv-X1@b; df_=len(yv)-X1.shape[1]
    se=np.sqrt(r@r/df_*np.linalg.inv(X1.T@X1)[j,j]); return b[j],2*tdist.sf(abs(b[j]/se),df_)
def chains(Xb,Xn,e,y,c):
    rows=[]; n=len(e)
    for j in range(Xb.shape[1]):
        a1,pa1=tstat(np.column_stack([e,c]),Xb[:,j],1)
        for k in range(Xn.shape[1]):
            d,pd_=tstat(np.column_stack([Xb[:,j],e,c]),Xn[:,k],1)
            X1=np.column_stack([np.ones(n),Xb[:,j],Xn[:,k],e,c]); bb=np.linalg.lstsq(X1,y,rcond=None)[0]; r=y-X1@bb; df_=n-X1.shape[1]; V=r@r/df_*np.linalg.inv(X1.T@X1)
            b2,pb2=bb[2],2*tdist.sf(abs(bb[2]/np.sqrt(V[2,2])),df_); b1=bb[1]
            rows.append((j,k,a1*d*b2,a1*b1,max(pa1,pd_,pb2),a1,d,b2))
    return np.array(rows)
imp=EnvImputer().fit(Xall,env); XI=imp.transform(Xall,env)
groups=[('pooled',(E==0)|(E==1)),('left',(E==0)|(sd_=='L')),('right',(E==0)|(sd_=='R'))]
t0=time.time(); print(f'pairs: {nb} blood x {len(N_cols)} neuro = {nb*len(N_cols)}')
for g,mask in groups:
    k_=np.where(mask)[0]; R=chains(XI[k_][:,:nb],XI[k_][:,nb:],E[k_],Yv[k_],Cc[k_]); q=multipletests(R[:,4],method='fdr_bh')[1]
    tot=tstat(np.column_stack([E[k_],Cc[k_]]),Yv[k_],1)[0]
    T=pd.DataFrame({'blood':np.array(B_cols)[R[:,0].astype(int)],'neuro':np.array(N_cols)[R[:,1].astype(int)],'chain':R[:,2],'direct_blood':R[:,3],'p_joint':R[:,4],'q':q,'a1':R[:,5],'d':R[:,6],'b2':R[:,7]})
    T.to_csv(f'seq_scan_{g}.csv',index=False)
    print(f'\n=== {g} (n={len(k_)}, total effect {tot:+.3f} m/s) === chains q<0.1: {(q<0.1).sum()} | p_joint<0.05: {(R[:,4]<0.05).sum()} (expected by chance ~{0.05*len(R):.0f})')
    print(f'   median |chain| {np.median(np.abs(R[:,2])):.4f} vs median |direct blood| {np.median(np.abs(R[:,3])):.4f} m/s; chain larger than direct in {100*np.mean(np.abs(R[:,2])>np.abs(R[:,3])):.0f}% of pairs')
    print(T.sort_values('chain').head(6)[['blood','neuro','chain','direct_blood','a1','d','b2','p_joint','q']].round(3).to_string(index=False))
print(f'[{time.time()-t0:.0f}s]')

# ---------------- nested evaluation of sequential pathway selection
lab=np.where(E==0,0,np.where(sd_=='L',1,np.where(sd_=='R',2,-1))); coh=np.where(lab>=0)[0]
SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(coh,lab[coh]))
def auc(Xtr,ytr,Xte,yte):
    if len(np.unique(yte))<2: return np.nan
    return roc_auc_score(yte,RidgeClassifierCV(alphas=[0.1,1,10,100,1000]).fit(Xtr,ytr).decision_function(Xte))
def top_chains(Xtr_,e,y,c,k=3):
    R=chains(Xtr_[:,:nb],Xtr_[:,nb:],e,y,c); el=R[R[:,4]<0.2]; el=el if len(el)>=k else R[np.argsort(R[:,4])[:k+5]]
    return [(int(r[0]),int(r[1])) for r in el[np.argsort(el[:,2])][:k]]
res={};CH={'left':[],'right':[]}; oofB=np.full((10,len(E)),np.nan); oofN=np.full((10,len(E)),np.nan); t1=time.time()
for f,(trp,tep) in enumerate(SP):
    tr,te=coh[trp],coh[tep]; im=EnvImputer().fit(Xall[tr],env[tr]); Xtr=im.transform(Xall[tr],env[tr]); Xte=im.transform(Xall[te],env[te]); feats=set()
    for gname,gv in [('left',1),('right',2)]:
        ktr=np.where((lab[tr]==0)|(lab[tr]==gv))[0]; kte=np.where((lab[te]==0)|(lab[te]==gv))[0]; etr=(lab[tr][ktr]==gv).astype(float); ete=(lab[te][kte]==gv).astype(float)
        ch=top_chains(Xtr[ktr],etr,Yv[tr][ktr],Cc[tr][ktr]); CH[gname].append(tuple(f'{B_cols[j]} -> {N_cols[k]}' for j,k in ch))
        cols=sorted(set([j for j,_ in ch]+[nb+k for _,k in ch])); feats|=set(cols)
        res.setdefault(f'{gname}: chain features',[]).append(auc(Xtr[ktr][:,cols],etr,Xte[kte][:,cols],ete))
    cols=sorted(feats); Etr,Ete=(lab[tr]>0).astype(float),(lab[te]>0).astype(float)
    res.setdefault('all: union of chain features',[]).append(auc(Xtr[:,cols],Etr,Xte[:,cols],Ete))
    bj=[c for c in cols if c<nb]; nk=[c for c in cols if c>=nb]
    for key,cc,O in [('B',bj,oofB),('N',nk,oofN)]:
        sg=np.sign(np.linalg.lstsq(np.column_stack([np.ones(len(tr)),Etr,Cc[tr]]),Xtr[:,cc],rcond=None)[0][1]); sg[sg==0]=1
        ctr=(Xtr[:,cc]*sg).mean(1); O[f//5,te]=((Xte[:,cc]*sg).mean(1)-ctr.mean())/ctr.std()
print(f'\n=== NESTED (50 folds, {time.time()-t1:.0f}s) ===')
for k_,v in res.items(): print(f'   {k_:34s} AUC {np.nanmean(v):.3f}')
for g in CH:
    J=np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for i,a in enumerate(CH[g]) for b in CH[g][i+1:]]); c=Counter(x for t in CH[g] for x in t)
    print(f'{g}: chain stability {J:.2f}; most selected chains: '+'; '.join(f'{x} {v/50:.0%}' for x,v in c.most_common(3)))
Bi,Ni=np.nanmean(oofB,0)[coh],np.nanmean(oofN,0)[coh]; Ec,Yc,Ck=(lab[coh]>0).astype(float),Yv[coh],Cc[coh]; rng=np.random.default_rng(5)
def ser(i):
    o=lambda y,X: np.linalg.lstsq(np.column_stack([np.ones(len(i)),X]),y,rcond=None)[0]
    a1=o(Bi[i],np.column_stack([Ec[i],Ck[i]]))[1]; d=o(Ni[i],np.column_stack([Bi[i],Ec[i],Ck[i]]))[1]; bb=o(Yc[i],np.column_stack([Bi[i],Ni[i],Ec[i],Ck[i]]))
    return np.array([a1*d*bb[2],a1*bb[1]])
e=ser(np.arange(len(coh))); BB=np.array([ser(rng.integers(0,len(coh),len(coh))) for _ in range(2000)])
print(f'honest out-of-fold CHAIN effect (blood -> neuro -> speed): {e[0]:+.3f} m/s [{np.percentile(BB[:,0],2.5):+.3f}, {np.percentile(BB[:,0],97.5):+.3f}] | direct blood route of the same features: {e[1]:+.3f} [{np.percentile(BB[:,1],2.5):+.3f}, {np.percentile(BB[:,1],97.5):+.3f}]')
