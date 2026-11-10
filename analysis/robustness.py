import sys, os, time, warnings; warnings.filterwarnings('ignore')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
ENR=(lambda v:(v-np.nanmean(v))/np.nanstd(v))(d.subject_number.str.extract(r'(\d+)')[0].astype(float).values)
CI=pd.read_pickle('comparison_internal.pkl'); FULL=CI['FULL']; METH=list(METHODS)
ibF=EnvImputer().fit(Xb_all[cohg]); inF=EnvImputer().fit(Xn_all[cohg]); igF=EnvImputer().fit(Gm[cohg]); drawF=np.random.default_rng(5).random(len(lab))<0.5
def base(sname):
    sv,BC=SIDES[sname]; k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]
    T,X,e,names,fam=side_data(k,sv,BC,ibF.transform(Xb_all[k]),inF.transform(Xn_all[k]),igF.transform(Gm[k]),drawF[k])
    return k,T,e,names,fam
def select_all(T,e,y,c,cg,fam,names):
    X=np.column_stack([T['B'],T['N'],T['G']]); X=(X-X.mean(0))/(X.std(0)+1e-9); out={}
    for mn,fn in METHODS.items(): out[mn]=frozenset(np.array(names)[fn(X,e,T=T,y=y,c=c,cg=cg,cc=cg,fams={'B':fam,'N':None,'G':None})])
    return out
def strat_idx(e,rng,frac=None):
    i1,i0=np.where(e==1)[0],np.where(e==0)[0]
    if frac is None: return np.r_[rng.choice(i1,len(i1)),rng.choice(i0,len(i0))]
    return np.r_[rng.choice(i1,int(round(frac*len(i1))),replace=False),rng.choice(i0,int(round(frac*len(i0))),replace=False)]
jac=lambda a,b: len(a&b)/max(1,len(a|b))
stage=os.environ.get('STAGE','perturb'); rng=np.random.default_rng(404); t0=time.time()
if stage=='perturb':
    R={}
    for sname in SIDES:
        k,T,e,names,fam=base(sname); y=Y[k]; c=C5[k]; cg=np.column_stack([c,FT[k]]); full={m:frozenset(map(str,FULL[(sname,m)])) for m in METH}
        boots=[]
        for b in range(50):
            i=strat_idx(e,rng); boots.append(select_all({t:v[i] for t,v in T.items()},e[i],y[i],c[i],cg[i],fam,names))
        for m in METH:
            sets=[bb[m] for bb in boots]; R[(sname,m,'bootstrap stability')]=np.mean([jac(a,b) for x,a in enumerate(sets) for b in sets[x+1:]])
            R[(sname,m,'bootstrap overlap with full panel')]=np.mean([jac(s,full[m]) for s in sets])
        subs=[]
        for b in range(30):
            i=strat_idx(e,rng,0.5); subs.append(select_all({t:v[i] for t,v in T.items()},e[i],y[i],c[i],cg[i],fam,names))
        for m in METH: R[(sname,m,'half-sample overlap with full panel')]=np.mean([s[m] and jac(s[m],full[m]) for s in subs])
        noi=[]
        for b in range(20):
            Tn={t:v+0.3*v.std(0)*rng.standard_normal(v.shape) for t,v in T.items()}; noi.append(select_all(Tn,e,y,c,cg,fam,names))
        for m in METH: R[(sname,m,'noise overlap with full panel')]=np.mean([s[m] and jac(s[m],full[m]) for s in noi])
        print(f'  {sname} done ({time.time()-t0:.0f}s)',flush=True)
    pd.to_pickle(R,'robust_perturb.pkl')
    P=pd.DataFrame([dict(side=k[0],method=k[1],test=k[2],value=v) for k,v in R.items()]).pivot_table(index='method',columns=['test','side'],values='value')
    pd.set_option('display.width',260); print(P.round(3).to_string())
if stage=='decoy':
    wt=Xraw['WTAR: Raw Score'].values.astype(float); ia=Xraw['IADL'].values.astype(float)
    CLASS={'noise':'Pure noise','conf_age':'Confounder proxy (age, adjusted)','conf_phase':'Confounder proxy (study phase, unadjusted)','collider':'Collider (stroke + slow walking)',
           'WTAR':'Real negative control (WTAR)','IADL':'Real consequence of walking (IADL)','lesion':'Diagnostic-only lesion marker'}
    LAYOUT={'B':['noise','noise','conf_age','lesion'],'N':['noise','conf_phase','collider','WTAR','IADL','lesion'],'G':['noise','conf_age','collider']}
    RES=[]
    for sname in SIDES:
        k,T,e,names,fam=base(sname); y=Y[k]; c=C5[k]; cg=np.column_stack([c,FT[k]]); z=lambda v:(v-np.nanmean(v))/np.nanstd(v)
        ez,yz,az,pz=z(e),z(y),z(C5[k][:,0]),z(ENR[k]); wz=np.nan_to_num(z(wt[k])); iz=np.nan_to_num(z(ia[k]))
        for b in range(40):
            N_=lambda: rng.standard_normal(len(k))
            gen={'noise':lambda: N_(),'conf_age':lambda: 0.8*az+0.6*N_(),'conf_phase':lambda: 0.8*pz+0.6*N_(),'collider':lambda: 0.7*ez-0.7*yz+0.5*N_(),'WTAR':lambda: wz,'IADL':lambda: iz,'lesion':lambda: 1.0*ez+0.8*N_()}
            Ta={t:np.column_stack([T[t]]+[gen[g]() for g in LAYOUT[t]]) for t in T}
            nB=[f'D:{g}:B{j}' for j,g in enumerate(LAYOUT['B'])]; nN=[f'D:{g}:N{j}' for j,g in enumerate(LAYOUT['N'])]; nG=[f'D:{g}:G{j}' for j,g in enumerate(LAYOUT['G'])]
            nb=len(T['B'][0]); names_a=names[:nb]+nB+names[nb:nb+16]+nN+names[nb+16:]+nG; fam_a=np.r_[fam,np.array(nB)]
            i=strat_idx(e,rng); S=select_all({t:v[i] for t,v in Ta.items()},e[i],y[i],c[i],cg[i],fam_a,names_a)
            for m,s in S.items():
                row={'side':sname,'method':m,'rep':b}
                for g in CLASS: row[g]=sum(1 for x in s if x.startswith(f'D:{g}:'))
                RES.append(row)
        print(f'  {sname} decoys done ({time.time()-t0:.0f}s)',flush=True)
    D=pd.DataFrame(RES); D.to_pickle('robust_decoy.pkl')
    ncls={g:sum(v.count(g) for v in LAYOUT.values()) for g in CLASS}
    false=['noise','conf_age','conf_phase','collider','WTAR','IADL']
    D['false_selected']=D[false].sum(1); D['any_false']=(D.false_selected>0).astype(float)
    S=D.groupby('method').agg({**{g:'mean' for g in CLASS},'false_selected':'mean','any_false':'mean'})
    for g in CLASS: S[g]=S[g]/ncls[g]
    S.columns=[CLASS.get(c_,c_) for c_ in S.columns]; pd.set_option('display.width',300)
    print('\nSelection rate of each decoy (share of that decoy type picked per panel, both sides, 40 bootstraps each):'); print(S.round(3).to_string())
if stage=='confound':
    rows=[]
    for sname in SIDES:
        k,T,e,names,fam=base(sname); X=np.column_stack([T['B'],T['N'],T['G']])
        for m in METH:
            for f in map(str,FULL[(sname,m)]):
                j=names.index(f); v=X[:,j]; isg=f.startswith('paretic ')
                p0=sm.OLS(v,sm.add_constant(e)).fit().pvalues[1]
                Z=np.column_stack([e,C5[k],ENR[k]]+([FT[k]] if isg else [])); p1=sm.OLS(v,sm.add_constant(Z)).fit().pvalues[1]
                rows.append(dict(side=sname,method=m,feature=f,p_unadjusted=p0,p_adjusted_confounders_phase=p1))
    C=pd.DataFrame(rows); C['sig_unadj']=C.p_unadjusted<0.05; C['sig_adj']=C.p_adjusted_confounders_phase<0.05
    S=C.groupby('method').apply(lambda g: pd.Series({'panel features':len(g),'stroke effect significant before adjustment (%)':100*g.sig_unadj.mean(),
        'still significant after confounders + study phase (%)':100*g.sig_adj.mean(),'lost significance on adjustment (%)':100*(g.sig_unadj&~g.sig_adj).sum()/max(1,g.sig_unadj.sum())}))
    C.to_pickle('robust_confound.pkl'); pd.set_option('display.width',250); print(S.round(1).to_string())
