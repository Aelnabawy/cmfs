import sys, time, warnings; warnings.filterwarnings('ignore')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
rng=np.random.default_rng(321); SP=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=987654321).split(cohg,lab[cohg]))
R={}; t0=time.time()
for f,(trp,tep) in enumerate(SP[:0]):
    tr,te=cohg[trp],cohg[tep]; ib=EnvImputer().fit(Xb_all[tr]); inn=EnvImputer().fit(Xn_all[tr]); ig=EnvImputer().fit(Gm[tr])
    blocks={}
    for sname,(sv,BC) in SIDES.items():
        ktr=tr[(lab[tr]==0)|(lab[tr]==sv)]; kte=te[(lab[te]==0)|(lab[te]==sv)]
        Ttr,_,etr,names,fam=side_data(ktr,sv,BC,ib.transform(Xb_all[ktr]),inn.transform(Xn_all[ktr]),ig.transform(Gm[ktr]),rng.random(len(ktr))<0.5)
        bi=[bcols.index(c_) for c_ in BC]; Pte=paretic(ig.transform(Gm[kte]),lab[kte],sv,rng.random(len(kte))<0.5)
        Xtr=np.column_stack([Ttr['B'],Ttr['N'],Ttr['G']]); Xte=np.column_stack([ib.transform(Xb_all[kte])[:,bi],inn.transform(Xn_all[kte]),Pte])
        mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9; blocks[sname]=((Xtr-mu)/sd,etr,(Xte-mu)/sd,(lab[kte]==sv).astype(float))
    # separate per-side model on all aligned features (no pooling)
    for s_,(A,y,B,yt) in blocks.items(): R.setdefault((s_,'Separate per-side ridge, all 73 features'),[]).append(auc(A,y,B,yt))
    # hemisphere-aware multitask model: shared effects + side-specific deviations (partial pooling)
    (AL,yL,BL,ytL),(AR,yR,BR,ytR)=blocks['LEFT'],blocks['RIGHT']; p=AL.shape[1]; Z0=np.zeros
    Ztr=np.r_[np.column_stack([AL,AL,Z0((len(AL),p))]),np.column_stack([AR,Z0((len(AR),p)),AR])]; ytr=np.r_[yL,yR]
    m=RidgeClassifierCV(alphas=[0.1,1,10,100,1000]).fit(Ztr,ytr)
    for s_,B,yt in [('LEFT',np.column_stack([BL,BL,Z0((len(BL),p))]),ytL),('RIGHT',np.column_stack([BR,Z0((len(BR),p)),BR]),ytR)]:
        if len(np.unique(yt))==2: R.setdefault((s_,'Hemisphere-aware hierarchical (shared + side-specific)'),[]).append(roc_auc_score(yt,m.decision_function(B)))
    if f%10==9: print(f'  fold {f+1}/50 ({time.time()-t0:.0f}s)',flush=True)
out=pd.DataFrame()
print('\n(B) per-side AUC, same folds as all CMFS designs:'); print(out.round(3).to_string())
# full-data hierarchical model: largest side-specific deviations
ibF=EnvImputer().fit(Xb_all[cohg]); inF=EnvImputer().fit(Xn_all[cohg]); igF=EnvImputer().fit(Gm[cohg]); dF=np.random.default_rng(5).random(len(lab))<0.5; full={}
for sname,(sv,BC) in SIDES.items():
    k=cohg[(lab[cohg]==0)|(lab[cohg]==sv)]; T,X,e,names,fam=side_data(k,sv,BC,ibF.transform(Xb_all[k]),inF.transform(Xn_all[k]),igF.transform(Gm[k]),dF[k]); full[sname]=(X,e,names)
(XL,eL,nL),(XR,eR,nR)=full['LEFT'],full['RIGHT']; p=XL.shape[1]
m=RidgeClassifierCV(alphas=[0.1,1,10,100,1000]).fit(np.r_[np.column_stack([XL,XL,np.zeros((len(XL),p))]),np.column_stack([XR,np.zeros((len(XR),p)),XR])],np.r_[eL,eR]); w=np.ravel(m.coef_)
gen=[n.replace('MCAL','MCA').replace('MCAR','MCA').replace('MFVL','MFV').replace('MFVR','MFV').replace('RCO2L','RCO2').replace('RCO2R','RCO2').replace('ARL','AR').replace('ARR','AR').replace('(Left)','').replace('(Right)','') for n in nL]
D=pd.DataFrame({'feature (aligned: lesion-side MCA, paretic foot)':gen,'shared':w[:p],'left-specific':w[p:2*p],'right-specific':w[2*p:]}); D['|right-left|']=(D['right-specific']-D['left-specific']).abs()
print('\nLargest hemisphere-specific deviations (hierarchical model, full data):'); print(D.sort_values('|right-left|',ascending=False).head(8).round(3).to_string(index=False))
# (C) replicate literature lateralisation claims in CVES (right minus left strokes, control SD units, adjusted, 95% bootstrap CI)
strokes=coh[lab[coh]>0]; ctrl=coh[lab[coh]==0]; rngb=np.random.default_rng(31)
hrv=[c for c in Xraw.columns if c.strip().upper().startswith('HRV')][:12]
def contrast(v,idx,ctl,cov):
    sdc=np.nanstd(v[ctl]); mc=np.nanmean(v[ctl]); m_=~np.isnan(v[idx]); s=idx[m_]; z=(v[s]-mc)/sdc; r=(lab[s]==2).astype(float); C=cov[s]
    est=lambda i: np.linalg.lstsq(np.column_stack([np.ones(len(i)),r[i],C[i]]),z[i],rcond=None)[0][1]
    i1,i0=np.where(r==1)[0],np.where(r==0)[0]; B=[est(np.r_[rngb.choice(i1,len(i1)),rngb.choice(i0,len(i0))]) for _ in range(2000)]
    return est(np.arange(len(z))),np.percentile(B,2.5),np.percentile(B,97.5),z[r==0].mean(),z[r==1].mean(),len(z)
print('\n(C) Lateralisation claims tested in CVES (right minus left strokes; controls = 0 SD):')
for c in hrv:
    v=Xraw[c].values.astype(float); e,lo,hi,zl,zr,n=contrast(v,strokes,ctrl,C5); print(f'  HRV  {c:30s} left {zl:+.2f}, right {zr:+.2f} | right-left {e:+.2f} [{lo:+.2f}, {hi:+.2f}] n={n}')
XG=EnvImputer().fit(Gm[cohg]).transform(Gm); sg=cohg[lab[cohg]>0]; cg_=cohg[lab[cohg]==0]
pare=np.where((lab==2)[:,None],XG[:,:12],XG[:,12:])   # paretic = left foot for right strokes, right foot for left strokes
for j,g in enumerate(GN):
    if g in ('stride time','step time','stance %','swing %','single-limb support %','double support %','swing time'):
        v=pare[:,j].copy(); ctlv=np.where(dF[:,None],XG[:,:12],XG[:,12:])[:,j]; v[lab==0]=ctlv[lab==0]
        e,lo,hi,zl,zr,n=contrast(v,sg,cg_,np.column_stack([C5,FT])); print(f'  GAIT paretic {g:25s} left {zl:+.2f}, right {zr:+.2f} | right-left {e:+.2f} [{lo:+.2f}, {hi:+.2f}] n={n}')
