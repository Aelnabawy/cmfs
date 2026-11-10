import sys, os, time, warnings; warnings.filterwarnings('ignore')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAc
from sklearn.model_selection import StratifiedKFold
ibF=EnvImputer().fit(Xb_all[cohg]); inF=EnvImputer().fit(Xn_all[cohg]); igF=EnvImputer().fit(Gm[cohg]); drawF=np.random.default_rng(5).random(len(lab))<0.5
def cf(X,e,seed=0):
    s=np.zeros(len(e)); k=min(4,int(e.sum()),int((1-e).sum()))
    for tr,te in StratifiedKFold(k,shuffle=True,random_state=seed).split(X,e): s[te]=LDAc(solver='lsqr',shrinkage='auto').fit(X[tr],e[tr]).decision_function(X[te])
    return ((s-s.mean())/(s.std()+1e-9))[:,None]
def coef_se(y,X):
    X1=np.column_stack([np.ones(len(y)),X]); b,*_=np.linalg.lstsq(X1,y,rcond=None); r=y-X1@b; df=max(len(y)-X1.shape[1],1)
    cov=(r@r/df)*np.linalg.pinv(X1.T@X1); return b[1],np.sqrt(cov[1,1])
def prep(T,e,y,c,cg,idx,seed):
    Th={t:v[idx] for t,v in T.items()}; eh,yh,ch,cgh=e[idx],y[idx],c[idx],cg[idx]
    UB=cf(Th['B'],eh,seed); UN=cf(Th['N'],eh,seed+1)
    return Th,eh,yh,ch,cgh,{'B':[],'N':[UB],'G':[UB,UN]}
def ab_one(x,e,y,cc,up,i=None):
    i=np.arange(len(e)) if i is None else i; z=x[i]; z=(z-z.mean())/(z.std()+1e-9)
    a,sa=coef_se(z,np.column_stack([e[i],cc[i]])); b,sb=coef_se(y[i],np.column_stack([z,e[i],cc[i]]+[u[i] for u in up])); return a,sa,b,sb
SIDE=os.environ.get('SIDE','RIGHT'); sv,BC=SIDES[SIDE]; rng=np.random.default_rng(909)
k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]; T,X,e,names,fam=side_data(k,sv,BC,ibF.transform(Xb_all[k]),inF.transform(Xn_all[k]),igF.transform(Gm[k]),drawF[k])
y=Y[k]; c=C5[k]; cg=np.column_stack([c,FT[k]]); tiers=[('B',j) for j in range(T['B'].shape[1])]+[('N',j) for j in range(16)]+[('G',j) for j in range(12)]
REC={n:dict(sel=0,dir_ok=0,ci_ok=0,est=[]) for n in names}; nsel=[]; t0=time.time()
for r in range(100):
    i1,i0=rng.permutation(np.where(e==1)[0]),rng.permutation(np.where(e==0)[0]); S=np.r_[i1[:len(i1)//2],i0[:len(i0)//2]]; Cf=np.r_[i1[len(i1)//2:],i0[len(i0)//2:]]
    Ts,es,ys,cs,cgs,ups=prep(T,e,y,c,cg,S,r); Tc,ec,yc,cc_,cgc,upc=prep(T,e,y,c,cg,Cf,r+100)
    Xs=np.column_stack([Ts['B'],Ts['N'],Ts['G']]); Xs=(Xs-Xs.mean(0))/(Xs.std(0)+1e-9)
    sel=m_cmfs(Xs,es,T=Ts,y=ys,c=cs,cg=cgs,fams={'B':fam,'N':None,'G':None}); chosen=[(names[i],)+tiers[i] for i in sel]
    nsel.append(len(chosen))
    for n,t,j in chosen:
        REC[n]['sel']+=1; ccc=cgc if t=='G' else cc_; a,_,b,_=ab_one(Tc[t][:,j],ec,yc,ccc,upc[t]); est=a*b; REC[n]['est'].append(est)
        j1,j0=np.where(ec==1)[0],np.where(ec==0)[0]; bs=[]
        for _ in range(200):
            ii=np.r_[rng.choice(j1,len(j1)),rng.choice(j0,len(j0))]; a_,_,b_,_=ab_one(Tc[t][:,j],ec,yc,ccc,upc[t],ii); bs.append(a_*b_)
        REC[n]['dir_ok']+=est<0; REC[n]['ci_ok']+=np.percentile(bs,97.5)<0
    if r%25==24: print(f'  {SIDE} split {r+1}/100 ({time.time()-t0:.0f}s)',flush=True)
rows=[dict(feature=n,tier={'B':'cerebrovascular','N':'cognitive','G':'paretic gait'}[t],selected_pct=REC[n]['sel'],direction_confirmed_pct_of_selected=100*REC[n]['dir_ok']/max(1,REC[n]['sel']),
           CI_confirmed_pct_of_selected=100*REC[n]['ci_ok']/max(1,REC[n]['sel']),selected_and_CI_confirmed_pct=REC[n]['ci_ok'],
           honest_indirect_mean=np.mean(REC[n]['est']) if REC[n]['est'] else np.nan) for n,(t,j) in zip(names,tiers)]
D=pd.DataFrame(rows).sort_values(['selected_and_CI_confirmed_pct','selected_pct'],ascending=False); D.to_csv(f'split_confirm_std_{SIDE}.csv',index=False)
print(f'\n{SIDE}: {int(e.sum())} strokes vs {int((1-e).sum())} controls; halves of ~{int(e.sum())//2} strokes; standard CMFS panel of {np.mean(nsel):.0f} features selected per split')
pd.set_option('display.width',220); print(D[D.selected_pct>0].round(3).head(15).to_string(index=False))
