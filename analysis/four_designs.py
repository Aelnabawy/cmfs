import sys, time; sys.path.insert(0,'/home/claude/cmfs')
from prep89 import *
from scipy.stats import t as tdist, mannwhitneyu
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from collections import Counter
from cmfs.data import EnvImputer
from cmfs.stats import coef_p
from cmfs.transform import fit_index, crossfit_index
sr=d['Stroke Side'].astype(str).str.upper().str.strip()
side=np.where(S==1,np.where(sr.str.contains('BILAT'),'',np.where(sr.str.match(r'^(L|LEFT)'),'L',np.where(sr.str.match(r'^(R|RIGHT)'),'R',''))),'')
lab=np.where(S==0,0,np.where(side=='L',1,np.where(side=='R',2,-1))); coh=np.where(lab>=0)[0]
C5=C.copy()   # age, sex, BMI, pack-years, hypertension (standardised)
BLOOD=[c for c in names if domain[c]=='Blood' and not c.startswith(('p_','R2_')) and 'olumn 40' not in c]
swaps=[('MCAL','MCAR'),('MFVL','MFVR'),('RCO2L','RCO2R'),('ARL','ARR'),('(Left)','(Right)'),('Left','Right')]
LB,RB=[],[]
for c in BLOOD:
    for a_,b_ in swaps:
        if a_ in c and c.replace(a_,b_) in BLOOD: LB.append(c); RB.append(c.replace(a_,b_)); break
NEURO=['TM: Part A Time (sec)','TM: Part B Time (sec)','VF: # Animals','VF: FAS Total','VF: # supermarket','HVLT: Total Recall','HVLT: Delayed Recall','RCFT: Copy-Raw','RCFT: I.R.-Raw','RCFT: D.R.-Raw','RCFT: Total Correct','DS: Total','Clock: Total Score','MMSE Total','GDS Total Response','BADS Total Responses']
SIDES={'LEFT':(1,LB),'RIGHT':(2,RB)}
def ols_b(y,X): return np.linalg.lstsq(np.column_stack([np.ones(len(y)),X]),y,rcond=None)[0]
def tp(y,X,j):
    X1=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X1,y,rcond=None)[0]; r=y-X1@b; df=len(y)-X1.shape[1]
    return b[j],2*tdist.sf(abs(b[j]/np.sqrt(r@r/df*np.linalg.inv(X1.T@X1)[j,j])),df)
def idx_fit(kind,M,e,y,for_scoring):
    if kind=='encoder': f=fit_index('pca',M,e,y); return f(M),f
    f=fit_index('lda',M,e,y); return (crossfit_index('lda',M,e,y,n_splits=4) if for_scoring else f(M)),f
def topk(score,pa,k,alpha=0.1):
    el=np.where(pa<alpha)[0]; el=el if len(el)>=k else np.argsort(pa)[:k+3]; return el[np.argsort(score[el])][:k]
def design1(Xb,Xn,e,y,c,kind,k=5):     # side blood i -> neuro index -> outcome ; rank by total effect through blood i
    NI,_=idx_fit(kind,Xn,e,y,True); a,pa=coef_p(Xb,np.column_stack([e,c]),1); tot=np.zeros(Xb.shape[1]); ch=np.zeros(Xb.shape[1])
    for j in range(Xb.shape[1]):
        d_=ols_b(NI,np.column_stack([Xb[:,j],e,c]))[1]; bb=ols_b(y,np.column_stack([Xb[:,j],NI,e,c])); tot[j]=a[j]*(bb[1]+d_*bb[2]); ch[j]=a[j]*d_*bb[2]
    return topk(tot,pa,k),tot,ch,pa
def design2(Xb,Xn,e,y,c,kind,k=3):     # blood index -> neuro j -> outcome ; rank by total effect through neuro j
    BI,_=idx_fit(kind,Xb,e,y,True); at,pa=coef_p(Xn,np.column_stack([e,c]),1); sc=np.zeros(Xn.shape[1]); ch=np.zeros(Xn.shape[1])
    a1=ols_b(BI,np.column_stack([e,c]))[1]
    for j in range(Xn.shape[1]):
        d_=ols_b(Xn[:,j],np.column_stack([BI,e,c]))[1]; b2=ols_b(y,np.column_stack([Xn[:,j],BI,e,c]))[1]; sc[j]=at[j]*b2; ch[j]=a1*d_*b2
    return topk(sc,pa,k),sc,ch,pa
def design3(Xb,Xn,e,y,c,k=3):          # all blood i x neuro j sequential paths ; rank by joint significance then chain
    rows=[]
    for i in range(Xb.shape[1]):
        a1,pa1=tp(Xb[:,i],np.column_stack([e,c]),1)
        for j in range(Xn.shape[1]):
            d_,pd_=tp(Xn[:,j],np.column_stack([Xb[:,i],e,c]),1); b2,pb2=tp(y,np.column_stack([Xn[:,j],Xb[:,i],e,c]),1)
            rows.append((i,j,a1*d_*b2,max(pa1,pd_,pb2)))
    R=np.array(rows); R=R[np.lexsort((R[:,2],R[:,3]))]; return [(int(r[0]),int(r[1]),r[2],r[3]) for r in R[:k]],R
def auc(Xtr,ytr,Xte,yte):
    if len(np.unique(yte))<2: return np.nan
    return roc_auc_score(yte,RidgeClassifierCV(alphas=[0.1,1,10,100,1000]).fit(Xtr,ytr).decision_function(Xte))
if __name__=='__main__':
    Xall=Xraw[sorted(set(LB+RB))+NEURO].values.astype(float); cols=sorted(set(LB+RB))+NEURO
    SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(coh,lab[coh])); res={}; SEL={}; t0=time.time()
    for f,(trp,tep) in enumerate(SP):
        tr,te=coh[trp],coh[tep]; imp=EnvImputer().fit(Xall[tr]); Xtr,Xte=imp.transform(Xall[tr]),imp.transform(Xall[te])
        for sname,(sv,BC) in SIDES.items():
            bi=[cols.index(c) for c in BC]; ni=[cols.index(c) for c in NEURO]
            ktr=np.where((lab[tr]==0)|(lab[tr]==sv))[0]; kte=np.where((lab[te]==0)|(lab[te]==sv))[0]; e=(lab[tr][ktr]==sv).astype(float); ete=(lab[te][kte]==sv).astype(float)
            Btr,Bte,Ntr,Nte,y,c=Xtr[ktr][:,bi],Xte[kte][:,bi],Xtr[ktr][:,ni],Xte[kte][:,ni],Y[tr][ktr],C5[tr][ktr]
            for kind in ['encoder','LDA']:
                s1,_,_,_=design1(Btr,Ntr,e,y,c,kind); _,nf=idx_fit(kind,Ntr,e,y,False); SEL.setdefault((sname,'1',kind),[]).append(tuple(np.array(BC)[s1]))
                res.setdefault((sname,'1',kind,'selected blood'),[]).append(auc(Btr[:,s1],e,Bte[:,s1],ete))
                res.setdefault((sname,'1',kind,'selected blood + neuro index'),[]).append(auc(np.column_stack([Btr[:,s1],nf(Ntr)]),e,np.column_stack([Bte[:,s1],nf(Nte)]),ete))
                s2,_,_,_=design2(Btr,Ntr,e,y,c,kind); _,bf=idx_fit(kind,Btr,e,y,False); SEL.setdefault((sname,'2',kind),[]).append(tuple(np.array(NEURO)[s2]))
                res.setdefault((sname,'2',kind,'selected neuro'),[]).append(auc(Ntr[:,s2],e,Nte[:,s2],ete))
                res.setdefault((sname,'2',kind,'selected neuro + blood index'),[]).append(auc(np.column_stack([Ntr[:,s2],bf(Btr)]),e,np.column_stack([Nte[:,s2],bf(Bte)]),ete))
            paths,_=design3(Btr,Ntr,e,y,c); SEL.setdefault((sname,'3','-'),[]).append(tuple(f'{BC[i]} -> {NEURO[j]}' for i,j,_,_ in paths))
            Bsel=sorted(set(i for i,_,_,_ in paths)); Nsel=sorted(set(j for _,j,_,_ in paths))
            res.setdefault((sname,'3','-','features of top-3 paths'),[]).append(auc(np.column_stack([Btr[:,Bsel],Ntr[:,Nsel]]),e,np.column_stack([Bte[:,Bsel],Nte[:,Nsel]]),ete))
            assoc=np.argsort([mannwhitneyu(np.column_stack([Btr,Ntr])[e==1,j],np.column_stack([Btr,Ntr])[e==0,j]).pvalue for j in range(len(bi)+len(ni))])[:5]
            res.setdefault((sname,'ref','-','association top-5 (blood+neuro)'),[]).append(auc(np.column_stack([Btr,Ntr])[:,assoc],e,np.column_stack([Bte,Nte])[:,assoc],ete))
        if f%10==9: print(f'  fold {f+1}/50 ({time.time()-t0:.0f}s)',flush=True)
    pd.to_pickle(dict(res=res,SEL=SEL),'four_designs_nested.pkl')
    print(f"\n{'side':6s} {'design':7s} {'transform':8s} {'classifier input':34s} AUC")
    for (s_,dsg,k_,inp),v in res.items(): print(f'{s_:6s} {dsg:7s} {k_:8s} {inp:34s} {np.nanmean(v):.3f}')
    print()
    for (s_,dsg,k_),v in SEL.items():
        J=np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for i,a in enumerate(v) for b in v[i+1:]])
        print(f'{s_} design {dsg} {k_}: stability {J:.2f}; top: '+'; '.join(f'{x} {c/50:.0%}' for x,c in Counter(y for t in v for y in t).most_common(3)))
