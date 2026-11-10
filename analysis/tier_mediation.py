import sys, warnings, itertools; warnings.filterwarnings('ignore')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAc
from sklearn.model_selection import StratifiedKFold
ibF=EnvImputer().fit(Xb_all[cohg]); inF=EnvImputer().fit(Xn_all[cohg]); igF=EnvImputer().fit(Gm[cohg]); drawF=np.random.default_rng(5).random(len(lab))<0.5
def cf_index(X,lab_,reps=10):
    out=np.zeros(len(lab_))
    for r in range(reps):
        s=np.zeros(len(lab_))
        for tr,te in StratifiedKFold(4,shuffle=True,random_state=r).split(X,lab_):
            m=LDAc(solver='lsqr',shrinkage='auto').fit(X[tr],lab_[tr]); s[te]=m.decision_function(X[te])
        out+=(s-s.mean())/s.std()
    out/=reps; return (out-out.mean())/out.std()
def ols(y,X): return np.linalg.lstsq(np.column_stack([np.ones(len(y)),X]),y,rcond=None)[0]
PATHS=['blood','neuro','gait','blood>neuro','blood>gait','neuro>gait','blood>neuro>gait']
def decompose(e,B,N,G,y,c,cg,i=None):
    i=np.arange(len(e)) if i is None else i; e,B,N,G,y,c,cg=e[i],B[i],N[i],G[i],y[i],c[i],cg[i]
    a1=ols(B,np.column_stack([e,c]))[1]
    p2=ols(N,np.column_stack([e,B,c])); a2,d21=p2[1],p2[2]
    p3=ols(G,np.column_stack([e,B,N,cg])); a3,d31,d32=p3[1],p3[2],p3[3]
    py=ols(y,np.column_stack([e,B,N,G,cg])); cp,b1,b2,b3=py[1],py[2],py[3],py[4]
    tot=ols(y,np.column_stack([e,cg]))[1]
    P={'blood':a1*b1,'neuro':a2*b2,'gait':a3*b3,'blood>neuro':a1*d21*b2,'blood>gait':a1*d31*b3,'neuro>gait':a2*d32*b3,'blood>neuro>gait':a1*d21*d32*b3}
    P['through blood (all routes)']=P['blood']+P['blood>neuro']+P['blood>gait']+P['blood>neuro>gait']
    P['through cognition (all routes)']=P['neuro']+P['blood>neuro']+P['neuro>gait']+P['blood>neuro>gait']
    P['through gait (all routes)']=P['gait']+P['blood>gait']+P['neuro>gait']+P['blood>neuro>gait']
    P['total indirect']=sum(P[k] for k in PATHS); P['direct (unexplained)']=cp; P['total']=tot; return P
rng=np.random.default_rng(77); OUT={}
for sname,(sv,BC) in SIDES.items():
    k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]; T,X,e,names,fam=side_data(k,sv,BC,ibF.transform(Xb_all[k]),inF.transform(Xn_all[k]),igF.transform(Gm[k]),drawF[k])
    y=Y[k]; c=C5[k]; cg=np.column_stack([c,FT[k]])
    B,N,G=cf_index(T['B'],e),cf_index(T['N'],e),cf_index(T['G'],e)
    est=decompose(e,B,N,G,y,c,cg); i1,i0=np.where(e==1)[0],np.where(e==0)[0]
    boot=[decompose(e,B,N,G,y,c,cg,np.r_[rng.choice(i1,len(i1)),rng.choice(i0,len(i0))]) for _ in range(2000)]
    null=[]
    for p in range(60):
        ep=rng.permutation(e); Bp,Np,Gp=cf_index(T['B'],ep,3),cf_index(T['N'],ep,3),cf_index(T['G'],ep,3); null.append(decompose(e,Bp,Np,Gp,y,c,cg))
    OUT[sname]=dict(est=est,boot=boot,null=null,n=(int(e.sum()),int((1-e).sum())))
    tot_c=ols(y,np.column_stack([e,c]))[1]   # total effect adjusted for confounders only (denominator of the shares in Table 7)
    print(f"\n=== {sname}-hemisphere strokes ({int(e.sum())} vs {int((1-e).sum())} controls): total effect {est['total']:+.3f} m/s (also adjusted for recording type); {tot_c:+.3f} m/s (confounders only, used for the shares in the paper)")
    for key in PATHS+['through blood (all routes)','through cognition (all routes)','through gait (all routes)','total indirect','direct (unexplained)']:
        b=np.array([x[key] for x in boot]); nl=np.array([x[key] for x in null])
        print(f"   {key:32s} {est[key]:+.3f} [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]  share {100*est[key]/est['total']:5.0f}% (paper: {100*est[key]/tot_c:3.0f}%)   shuffled-label null 95% range [{np.percentile(nl,2.5):+.3f}, {np.percentile(nl,97.5):+.3f}]")
pd.to_pickle(OUT,'tier_mediation.pkl')
