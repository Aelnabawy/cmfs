import sys, time, itertools
exec(open('gait_four.py').read().split("if __name__=='__main__':")[0])
TN={'B':'blood','N':'neuro','G':'paretic gait'}; K={'B':5,'N':3,'G':3}
COMBOS=[('single',('B',)),('single',('N',)),('single',('G',)),('double',('B','N')),('double',('B','G')),('double',('N','G')),('triple',('B','N','G'))]
def lda_pair(Xtr,Xte,e,y):
    f=fit_index('lda',Xtr,e,y); return crossfit_index('lda',Xtr,e,y,n_splits=4),f(Xtr),f(Xte)
def run(order,trans,T_tr,T_te,e,y,c,cg,fams):
    idx={t:lda_pair(T_tr[t],T_te[t],e,y) for t in trans}; sel={}; up=[]
    for t in order:
        if t in trans: up.append(idx[t][0][:,None]); continue
        X=T_tr[t]; cc=cg if t=='G' else c
        a,pa=coef_p(X,np.column_stack([e,cc]),1); adj=np.column_stack([e,cc]+up)
        sc=a*np.array([ols_b(y,np.column_stack([X[:,j],adj]))[1] for j in range(X.shape[1])])
        el=set(np.where(pa<0.1)[0]); o=[]; seen=set()
        for j in [j for j in np.argsort(sc) if j in el or len(el)<K[t]]:
            key=fams[t][j] if fams[t] is not None else j
            if key in seen: continue
            o.append(j); seen.add(key)
            if len(o)==K[t]: break
        sel[t]=np.array(o); up.append(crossfit_index('lda',X,e,y,n_splits=4)[:,None])
    F_tr=[T_tr[t][:,sel[t]] for t in order if t in sel]; F_te=[T_te[t][:,sel[t]] for t in order if t in sel]
    I_tr=[idx[t][1][:,None] for t in order if t in idx]; I_te=[idx[t][2][:,None] for t in order if t in idx]
    return sel,F_tr,F_te,I_tr,I_te
bcols=sorted(set(LB+RB)); Xb_all=Xraw[bcols].values.astype(float); Xn_all=Xraw[NEURO].values.astype(float); rng=np.random.default_rng(321)
SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(cohg,lab[cohg])); res={}; SEL={}; t0=time.time()
for f,(trp,tep) in enumerate(SP):
    tr,te=cohg[trp],cohg[tep]; ib=EnvImputer().fit(Xb_all[tr]); inn=EnvImputer().fit(Xn_all[tr]); ig=EnvImputer().fit(Gm[tr])
    B_tr,B_te,N_tr,N_te,G_tr,G_te=ib.transform(Xb_all[tr]),ib.transform(Xb_all[te]),inn.transform(Xn_all[tr]),inn.transform(Xn_all[te]),ig.transform(Gm[tr]),ig.transform(Gm[te])
    for sname,(sv,BC) in SIDES.items():
        bi=[bcols.index(c_) for c_ in BC]; ktr=np.where((lab[tr]==0)|(lab[tr]==sv))[0]; kte=np.where((lab[te]==0)|(lab[te]==sv))[0]
        e=(lab[tr][ktr]==sv).astype(float); ete=(lab[te][kte]==sv).astype(float); y=Y[tr][ktr]; c=C5[tr][ktr]; cg=np.column_stack([c,FT[tr][ktr]])
        T_tr={'B':B_tr[ktr][:,bi],'N':N_tr[ktr],'G':paretic(G_tr[ktr],lab[tr][ktr],sv,rng.random(len(ktr))<0.5)}
        T_te={'B':B_te[kte][:,bi],'N':N_te[kte],'G':paretic(G_te[kte],lab[te][kte],sv,rng.random(len(kte))<0.5)}
        fams={'B':np.array([family(x) for x in BC]),'N':None,'G':None}; NM={'B':np.array(BC),'N':np.array(NEURO),'G':np.array(['paretic '+g for g in GN])}
        for lvl,order in COMBOS:
            for r_ in range(len(order)+1):
                for trans in itertools.combinations(order,r_):
                    sel,F_tr,F_te,I_tr,I_te=run(order,trans,T_tr,T_te,e,y,c,cg,fams)
                    key=(lvl,' -> '.join(TN[t] for t in order),', '.join(TN[t] for t in trans) or 'none')
                    if F_tr: res.setdefault((sname,)+key+('selected features',),[]).append(auc(np.column_stack(F_tr),e,np.column_stack(F_te),ete))
                    if I_tr: res.setdefault((sname,)+key+(('selected features + LDA indices' if F_tr else 'LDA indices only'),),[]).append(auc(np.column_stack(F_tr+I_tr),e,np.column_stack(F_te+I_te),ete))
                    for t,s in sel.items(): SEL.setdefault((sname,)+key+(TN[t],),[]).append(tuple(NM[t][s]))
    if f%10==9: print(f'  fold {f+1}/50 ({time.time()-t0:.0f}s)',flush=True)
pd.to_pickle(dict(res=res,SEL=SEL),'multi_tier_lda.pkl')
rows=[dict(side=k[0],level=k[1],tiers=k[2],LDA_transformed=k[3],classifier_input=k[4],AUC=np.nanmean(v)) for k,v in res.items()]
T=pd.DataFrame(rows); W=T.pivot_table(index=['level','tiers','LDA_transformed','classifier_input'],columns='side',values='AUC',sort=False).reset_index()
pd.set_option('display.width',230); print(W.round(3).to_string(index=False)); T.to_csv('multi_tier_lda.csv',index=False)
