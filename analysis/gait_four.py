import sys, time, io, contextlib
from four_designs import *
cl=pd.read_csv('/mnt/user-data/outputs/Gait_Steps_Features_cleaned.csv'); cl['sid']=cl.file.str.extract(r'S(\d{4})')[0]
G=cl.set_index('sid').reindex(d.subject_number.str.extract(r'(\d{4})')[0].values); mass=num(d,'mass/kg').values
base=['max_force[N]','mean_force[N]','max_pressure[N/cm2]','stride','step','stance','stance_in_%_of_stride','swing','swing_in_%_init','double_in_%','single_limb_support_%','terminate_double_%']
GN=['peak force/BW','mean force/BW','peak pressure','stride time','step time','stance time','stance %','swing time','swing %','double support %','single-limb support %','terminal double support %']
gcols=[f'{sd}_{c}' for sd in ['Left','Right'] for c in base]; Gm=G[gcols].values.astype(float).copy(); bad=G['flag'].astype(str).str.contains('force').values
for j,c in enumerate(gcols):
    if 'force' in c: Gm[:,j]=Gm[:,j]/(mass*9.81)
    if 'force' in c or 'pressure' in c: Gm[bad,j]=np.nan
FT=(G.file_type=='S').astype(float).values; iL,iR=list(range(12)),list(range(12,24))
cohg=np.array([i for i in coh if not np.isnan(G['Right_stride'].values.astype(float)[i])])
def paretic(Gi,labs,sv,draw): return np.where(np.where(labs==sv,sv==2,draw)[:,None],Gi[:,iL],Gi[:,iR])
def g_index(kind,M,e,y,scoring): return idx_fit('encoder' if kind=='encoder' else 'LDA',M,e,y,scoring)
def D1(Xb,P,e,y,c,cg,kind,k=5):
    GI,_=g_index(kind,P,e,y,True); a,pa=coef_p(Xb,np.column_stack([e,c]),1); tot=np.zeros(Xb.shape[1])
    for j in range(Xb.shape[1]):
        d_=ols_b(GI,np.column_stack([Xb[:,j],e,cg]))[1]; bb=ols_b(y,np.column_stack([Xb[:,j],GI,e,cg])); tot[j]=a[j]*(bb[1]+d_*bb[2])
    return topk(tot,pa,k)
def D2(Xb,P,e,y,c,cg,kind,k=3):
    BI,_=g_index(kind,Xb,e,y,True); at,pa=coef_p(P,np.column_stack([e,cg]),1); sc=np.zeros(P.shape[1])
    for j in range(P.shape[1]): sc[j]=at[j]*ols_b(y,np.column_stack([P[:,j],BI,e,cg]))[1]
    return topk(sc,pa,k)
def D3(Xb,P,e,y,c,cg,k=3):
    rows=[]
    for i in range(Xb.shape[1]):
        a1,pa1=tp(Xb[:,i],np.column_stack([e,c]),1)
        for j in range(P.shape[1]):
            d_,pd_=tp(P[:,j],np.column_stack([Xb[:,i],e,cg]),1); b2,pb2=tp(y,np.column_stack([P[:,j],Xb[:,i],e,cg]),1); rows.append((i,j,a1*d_*b2,max(pa1,pd_,pb2)))
    R=np.array(rows); R=R[np.lexsort((R[:,2],R[:,3]))]; return [(int(r[0]),int(r[1])) for r in R[:k]]
if __name__=='__main__':
    Xb_all=Xraw[sorted(set(LB+RB))].values.astype(float); bcols=sorted(set(LB+RB)); rng=np.random.default_rng(301)
    print(f'cohort: {int((lab[cohg]==1).sum())} left + {int((lab[cohg]==2).sum())} right strokes + {int((lab[cohg]==0).sum())} controls with clean step data')
    SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(cohg,lab[cohg])); res={}; SEL={}; t0=time.time()
    for f,(trp,tep) in enumerate(SP):
        tr,te=cohg[trp],cohg[tep]; ib=EnvImputer().fit(Xb_all[tr]); Btr_all,Bte_all=ib.transform(Xb_all[tr]),ib.transform(Xb_all[te])
        ig=EnvImputer().fit(Gm[tr]); Gtr,Gte=ig.transform(Gm[tr]),ig.transform(Gm[te])
        for sname,(sv,BC) in SIDES.items():
            bi=[bcols.index(c_) for c_ in BC]; ktr=np.where((lab[tr]==0)|(lab[tr]==sv))[0]; kte=np.where((lab[te]==0)|(lab[te]==sv))[0]
            e=(lab[tr][ktr]==sv).astype(float); ete=(lab[te][kte]==sv).astype(float)
            Btr,Bte=Btr_all[ktr][:,bi],Bte_all[kte][:,bi]; Ptr=paretic(Gtr[ktr],lab[tr][ktr],sv,rng.random(len(ktr))<0.5); Pte=paretic(Gte[kte],lab[te][kte],sv,rng.random(len(kte))<0.5)
            y,c=Y[tr][ktr],C5[tr][ktr]; cg=np.column_stack([c,FT[tr][ktr]])
            for kind in ['encoder','LDA']:
                s1=D1(Btr,Ptr,e,y,c,cg,kind); _,gf=g_index(kind,Ptr,e,y,False); SEL.setdefault((sname,'1',kind),[]).append(tuple(np.array(BC)[s1]))
                res.setdefault((sname,'1',kind,'selected blood'),[]).append(auc(Btr[:,s1],e,Bte[:,s1],ete))
                res.setdefault((sname,'1',kind,'selected blood + paretic gait index'),[]).append(auc(np.column_stack([Btr[:,s1],gf(Ptr)]),e,np.column_stack([Bte[:,s1],gf(Pte)]),ete))
                s2=D2(Btr,Ptr,e,y,c,cg,kind); _,bf=g_index(kind,Btr,e,y,False); SEL.setdefault((sname,'2',kind),[]).append(tuple(np.array(GN)[s2]))
                res.setdefault((sname,'2',kind,'selected paretic gait'),[]).append(auc(Ptr[:,s2],e,Pte[:,s2],ete))
                res.setdefault((sname,'2',kind,'selected paretic gait + blood index'),[]).append(auc(np.column_stack([Ptr[:,s2],bf(Btr)]),e,np.column_stack([Pte[:,s2],bf(Bte)]),ete))
            paths=D3(Btr,Ptr,e,y,c,cg); SEL.setdefault((sname,'3','-'),[]).append(tuple(f'{BC[i]} -> paretic {GN[j]}' for i,j in paths))
            Bs=sorted(set(i for i,_ in paths)); Ps=sorted(set(j for _,j in paths))
            res.setdefault((sname,'3','-','features of top-3 paths'),[]).append(auc(np.column_stack([Btr[:,Bs],Ptr[:,Ps]]),e,np.column_stack([Bte[:,Bs],Pte[:,Ps]]),ete))
            Z=np.column_stack([Btr,Ptr]); Zt=np.column_stack([Bte,Pte]); assoc=np.argsort([mannwhitneyu(Z[e==1,j],Z[e==0,j]).pvalue for j in range(Z.shape[1])])[:5]
            res.setdefault((sname,'ref','-','association top-5 (blood+gait)'),[]).append(auc(Z[:,assoc],e,Zt[:,assoc],ete))
        if f%10==9: print(f'  fold {f+1}/50 ({time.time()-t0:.0f}s)',flush=True)
    pd.to_pickle(dict(res=res,SEL=SEL),'gait_four_nested.pkl')
    print(f"\n{'side':6s} {'design':6s} {'transform':8s} {'classifier input':38s} AUC")
    for (s_,dsg,k_,inp),v in res.items(): print(f'{s_:6s} {dsg:6s} {k_:8s} {inp:38s} {np.nanmean(v):.3f}')
    print()
    for (s_,dsg,k_),v in SEL.items():
        J=np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for i,a in enumerate(v) for b in v[i+1:]])
        print(f'{s_} design {dsg} {k_}: stability {J:.2f}; top: '+'; '.join(f'{x} {c/50:.0%}' for x,c in Counter(y for t in v for y in t).most_common(3)))
