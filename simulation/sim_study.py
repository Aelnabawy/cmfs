import sys, os, time, pickle, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/sim'); os.chdir('/home/claude/sim')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
# ---- real tier correlation structure (right-hemisphere analysis cohort) ----
ibF=EnvImputer().fit(Xb_all[cohg]); inF=EnvImputer().fit(Xn_all[cohg]); igF=EnvImputer().fit(Gm[cohg])
sv,BC=SIDES['RIGHT']; k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]
T0,_,_,_,_=side_data(k,sv,BC,ibF.transform(Xb_all[k]),inF.transform(Xn_all[k]),igF.transform(Gm[k]),np.random.default_rng(5).random(len(k))<0.5)
def psd(M): w,V=np.linalg.eigh(np.nan_to_num(M)); w=np.clip(w,1e-3,None); A=V@np.diag(w)@V.T; d=np.sqrt(np.diag(A)); return A/np.outer(d,d)
R={t:psd(np.corrcoef(T0[t].T)) for t in 'BNG'}; L={t:np.linalg.cholesky(R[t]) for t in 'BNG'}; P={'B':45,'N':16,'G':12}
ROLES=['mediator','lesion marker','outcome-only','measured-confounder proxy','unmeasured-confounder proxy','collider','noise']
PLAN={'B':[('mediator',2),('lesion marker',2),('outcome-only',1),('measured-confounder proxy',1),('unmeasured-confounder proxy',1),('collider',1)],
      'N':[('mediator',2),('lesion marker',1),('outcome-only',1),('measured-confounder proxy',1),('unmeasured-confounder proxy',1),('collider',1)],
      'G':[('mediator',2),('lesion marker',1),('outcome-only',1),('measured-confounder proxy',1),('unmeasured-confounder proxy',1),('collider',1)]}
def generate(n1,n0,rng,a_m=0.5,b=-0.04,a_l=0.8,d=0.5,gU=0.8,lamU=0.8,ntest=0):
    N=8000; age=rng.normal(size=N); sex=rng.binomial(1,.5,N); bmi=rng.normal(size=N); pack=rng.normal(size=N); htn=rng.binomial(1,.5,N); U=rng.normal(size=N)
    p=1/(1+np.exp(-(-1.6+0.5*age+0.5*htn+0.3*pack+gU*U))); E=rng.binomial(1,p)
    i1=rng.choice(np.where(E==1)[0],n1+ntest,replace=False); i0=rng.choice(np.where(E==0)[0],n0+ntest,replace=False)
    idx=np.r_[i1[:n1],i0[:n0]]; tst=np.r_[i1[n1:],i0[n0:]] if ntest else None
    def build(ix):
        n=len(ix); e=E[ix].astype(float); C=np.column_stack([age[ix],sex[ix],bmi[ix],pack[ix],htn[ix]]); u=U[ix]
        T={t:(rng.normal(size=(n,P[t]))@L[t].T) for t in 'BNG'}; role={t:np.array(['noise']*P[t],dtype=object) for t in 'BNG'}
        pos={t:rng.permutation(P[t]) for t in 'BNG'}; slot={}
        for t in 'BNG':
            q=0
            for r,cnt in PLAN[t]:
                for c_ in range(cnt): slot[(t,r,c_)]=pos[t][q]; role[t][pos[t][q]]=r; q+=1
        m=lambda t,c_=0: slot[(t,'mediator',c_)]
        T['B'][:,m('B',0)]+=a_m*e; T['B'][:,m('B',1)]+=a_m*e
        T['N'][:,m('N',0)]+=a_m*e+d*T['B'][:,m('B',0)]; T['N'][:,m('N',1)]+=a_m*e
        T['G'][:,m('G',0)]+=a_m*e+d*T['N'][:,m('N',0)]; T['G'][:,m('G',1)]+=a_m*e
        y=1.0-0.10*e-0.05*C[:,0]-0.03*C[:,2]-0.03*C[:,3]+rng.normal(0,0.12,n)
        for t in 'BNG':
            for c_ in range(2): y=y+b*T[t][:,m(t,c_)]
            T[t][:,slot[(t,'outcome-only',0)]]+=0; y=y-0.04*T[t][:,slot[(t,'outcome-only',0)]]
            for c_ in range(dict(PLAN[t])['lesion marker']): T[t][:,slot[(t,'lesion marker',c_)]]+=a_l*e
            T[t][:,slot[(t,'measured-confounder proxy',0)]]+=0.8*C[:,0]; T[t][:,slot[(t,'unmeasured-confounder proxy',0)]]+=lamU*u
        yz=(y-y.mean())/y.std()
        for t in 'BNG': T[t][:,slot[(t,'collider',0)]]+=0.6*e-0.6*yz
        Cz=(C-C.mean(0))/(C.std(0)+1e-9); return T,e,y,Cz,np.concatenate([role[t] for t in 'BNG'])
    tr=build(idx); te=build(tst) if ntest else None
    if ntest:  # test set must share the training feature roles: rebuild test with same slots by regenerating jointly
        pass
    return tr
def generate_pair(n1,n0,rng,**kw):
    st=rng.bit_generator.state; tr=generate(n1+500,n0+500,rng,**kw)
    T,e,y,C,roles=tr; i1,i0=np.where(e==1)[0],np.where(e==0)[0]; itr=np.r_[i1[:n1],i0[:n0]]; ite=np.r_[i1[n1:],i0[n0:]]
    sub=lambda ix: ({t:v[ix] for t,v in T.items()},e[ix],y[ix],C[ix])
    return sub(itr),sub(ite),roles
FAMS={'B':np.array([str(i) for i in range(45)]),'N':None,'G':None}
def m_cmfs_order(order):
    def f(X,e,T=None,y=None,c=None,cg=None,fams=None,**k):
        sel,_,_,_,_=run(order,(),T,T,e,y,c,cg,fams); off={'B':0,'N':45,'G':61}; return [off[t]+int(j) for t in order for j in sel[t]]
    return f
def one_rep(n1,n0,seed,methods,**kw):
    rng=np.random.default_rng(seed); (Ttr,e,y,C),(Tte,ete,_,_),roles=generate_pair(n1,n0,rng,**kw)
    X=np.column_stack([Ttr[t] for t in 'BNG']); mu,sd=X.mean(0),X.std(0)+1e-9; Xs=(X-mu)/sd; Xt=(np.column_stack([Tte[t] for t in 'BNG'])-mu)/sd
    out=[]
    for name,fn in methods.items():
        try: S=list(fn(Xs,e,T=Ttr,y=y,c=C,cg=C,cc=C,fams=FAMS))
        except Exception as ex: S=[]
        rs=roles[S] if S else np.array([])
        row=dict(method=name,size=len(S),AUC=auc(Xs[:,S],e,Xt[:,S],ete) if S else np.nan)
        nmed=(roles=='mediator').sum(); row['mediator recall']=np.sum(rs=='mediator')/nmed; row['mediator precision']=np.sum(rs=='mediator')/max(len(S),1)
        for r in ROLES[1:]: row[f'selected: {r}']=np.sum(rs==r)/(roles==r).sum()
        out.append(row)
    return out
if __name__=='__main__':
    scen=sys.argv[1]; reps=int(sys.argv[2])
    SCEN={'n16_40':dict(n1=16,n0=40),'n50_50':dict(n1=50,n0=50),'n150_150':dict(n1=150,n0=150),
          'weak_50':dict(n1=50,n0=50,a_m=0.3,b=-0.025),'strongU_50':dict(n1=50,n0=50,gU=1.5,lamU=1.2),'order_50':dict(n1=50,n0=50)}
    kw=dict(SCEN[scen]); n1,n0=kw.pop('n1'),kw.pop('n0')
    meths=dict(METHODS) if scen!='order_50' else {'CMFS, correct tier order':m_cmfs,'CMFS, tier order reversed':m_cmfs_order(('G','N','B')),'CMFS, cognitive and gait swapped':m_cmfs_order(('B','G','N'))}
    res=[]; t0=time.time()
    for r in range(reps):
        for row in one_rep(n1,n0,1000+r,meths,**kw): row.update(scenario=scen,rep=r); res.append(row)
        if r%10==9: print(f'{scen}: {r+1}/{reps} ({time.time()-t0:.0f}s)',flush=True); pickle.dump(res,open(f'/home/claude/simstudy/res_{scen}.pkl','wb'))
    pickle.dump(res,open(f'/home/claude/simstudy/res_{scen}.pkl','wb')); print('done',scen,flush=True)
