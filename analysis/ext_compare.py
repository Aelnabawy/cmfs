import sys, warnings; warnings.filterwarnings('ignore')
exec(open('gait_four.py').read().split("if __name__=='__main__':")[0])
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
CI=pd.read_pickle('comparison_internal.pkl'); FULL=CI['FULL']; rng=np.random.default_rng(23)
METH=[m for (s,m) in FULL if s=='LEFT' and m!='_names']
# ---- CVES direction of every candidate (pooled strokes vs controls, 5 confounders) ----
bcols=sorted(set(LB+RB)); XG=EnvImputer().fit(Gm[cohg]).transform(Gm); draw=np.random.default_rng(5).random(len(lab))<0.5
yy=(lab[coh]>0).astype(float); SIGN={}
for c_ in bcols+NEURO:
    v=Xraw[c_].values.astype(float)[coh]; m=~np.isnan(v)
    if m.sum()>20: SIGN[c_]=float(np.sign(sm.OLS(v[m],sm.add_constant(np.column_stack([yy[m],C5[coh][m]]))).fit().params[1]))
k=cohg; usel=np.where(lab[k]==0,draw[k],lab[k]==2)
for j,g in enumerate(GN):
    v=np.where(usel,XG[k][:,j],XG[k][:,12+j]); SIGN['paretic '+g]=float(np.sign(sm.OLS(v,sm.add_constant(np.column_stack([(lab[k]>0).astype(float),C5[k],FT[k]]))).fit().params[1]))
# ---- external frames with CVES feature names ----
num=lambda v: pd.to_numeric(v,errors='coerce'); E=pd.read_pickle('/home/claude/ext2/ext_data.pkl'); EXT={}
d,_,e=E['NHANES 2011-2014']; EXT['NHANES 2011-2014 (stroke)']=(pd.DataFrame({'y':d[e].values,'VF: # Animals':d.animal_fluency.values,'TM: Part B Time (sec)':-d.DSST.values,'HVLT: Total Recall':d.CERAD_learning.values,'HVLT: Delayed Recall':d.CERAD_delayed_recall.values,'GDS Total Response':d.PHQ9.values}),None)
d,_,e=E['NHANES 1999-2002']; EXT['NHANES 1999-2002 (stroke)']=(pd.DataFrame({'y':d[e].values,'TM: Part B Time (sec)':-d.CFDRIGHT.values}),None)
for nm_,key in [('PhysioNet neurodegenerative','PhysioNet neurodegenerative'),("PhysioNet Parkinson's",'PhysioNet Parkinson')]:
    d,_,e=E[key]; f=pd.DataFrame({'y':d[e].values,'paretic stride time':d.stride.values,'paretic swing time':d.swing_time.values,'paretic swing %':d.swing_pct.values,'paretic stance %':d.stance_pct.values,'paretic double support %':d.double_pct.values})
    if 'peak_force_bw' in d: f['paretic peak force/BW']=d.peak_force_bw.values; f['paretic stance time']=d.stance_time.values
    EXT[nm_]=(f,None)
sk=pd.read_excel('/home/claude/ext2/skv/stroke_gait.xlsx','Общее',header=None).iloc[4:].reset_index(drop=True); n_=lambda j: num(sk[j]).values; rows=[]
for i in range(len(sk)):
    for par,(cy,st,si,db) in [(1.0,(3,6,8,10)),(0.0,(4,7,9,11))]:
        c_,s_=n_(cy)[i],n_(st)[i]
        if not np.isnan(s_): rows.append({'y':par,'person':i,'paretic stance %':s_,'paretic swing %':100-s_,'paretic single-limb support %':n_(si)[i],'paretic double support %':n_(db)[i],'paretic swing time':c_*(100-s_)/100,'paretic stance time':c_*s_/100})
EXT['Skvortsov stroke (paretic leg)']=(pd.DataFrame(rows),'person')
cves_ids=set(pd.read_csv('/home/claude/ext/cves_subjects.csv',low_memory=False).iloc[:,0].astype(str).str.extract(r'(\d{3,4})')[0].dropna().astype(int))
sid=lambda v: v.astype(str).str.extract(r'(\d{2,4})')[0].astype(float); rd=lambda f: pd.read_csv('/home/claude/ext/'+f,low_memory=False,encoding='latin-1')
Dm=rd('GE-79_Summary_Table-Demographics-MRI-Part1.csv'); Cg=rd('GE-79_Summary_Table-Cognitive-Testing.csv')
Dm['sid']=sid(Dm['Patient ID']); Cg['sid']=sid(Cg['patient ID']); Dm['vis']=Dm.Visit.astype(str).str.strip(); Cg['vis']=Cg.Visit.astype(str).str.strip()
g=Dm[['sid','vis','DM, Non-DM, STROKE']].merge(Cg,on=['sid','vis']); g=g[(g.vis=='2')&(~g.sid.isin(cves_ids))]
f=pd.DataFrame({'y':(g['DM, Non-DM, STROKE'].astype(str).str.strip()=='DM').astype(float).values})
for c_ in NEURO:
    if c_ in g: f[c_]=num(g[c_]).values
EXT['GE-79 type 2 diabetes']=(f,None)
g=pd.read_csv('/home/claude/ext2/novak/GE-71_Data_Summary_Table.csv',low_memory=False,encoding='latin-1'); g['sid']=sid(g['SUBJECT NUMBER']); g=g[~g.sid.isin(cves_ids)]
f=pd.DataFrame({'y':(g['group2'].astype(str).str.strip()=='DM').astype(float).values})
for S_,LR,F_ in [('R','MCAR','MFVR'),('L','MCAL','MFVL')]:
    b=num(g[f'mn_{LR}(Baseline MR)']); 
    for cv_,ge in [('BASELINE','Baseline MR'),('TILT','Tilt MN'),('HV','HyperMean'),('SUPINE REBREATHING','Hypo Mean')]:
        f[f'SYST {LR} {cv_}']=num(g[f'sys_{LR}({ge})']).values; f[f'DIAST {LR} {cv_}']=num(g[f'dia_{LR}({ge})']).values; f[f'MEAN {LR} {cv_}']=num(g[f'mn_{LR}({ge})']).values
    f[f'{F_}_base']=b.values; f[f'{F_}_hyper_%']=((num(g[f'mn_{LR}(HyperMean)'])/b-1)*100).values; f[f'{F_}_hypo_%']=((num(g[f'mn_{LR}(Hypo Mean)'])/b-1)*100).values
    f[f'{F_}_tilt_%']=((num(g[f'mn_{LR}(Tilt MN)'])/b-1)*100).values; f[f'DELTA MEAN {LR} TILT-BASELINE']=(num(g[f'mn_{LR}(Tilt MN)'])-b).values
    dco2=(num(g['ETCO2(Tilt MN)'])-num(g['ETCO2(Baseline MR)'])).replace(0,np.nan)
    f[f'RCO2{S_}_base_tilt']=((num(g[f'mn_{LR}(Tilt MN)'])-b)/dco2).values; f[f'RCO2{S_}_base_tilt_%']=((num(g[f'mn_{LR}(Tilt MN)'])/b-1)*100/dco2).values
    f[f'RCO2{S_}_base_hyper_hypo_%']=num(g[f'Relative CO2 Reactivity {LR}']).values
f['CO2_reactivity_MCAR']=num(g['Absolute CO2 Reactivity MCAR']).values
EXT['GE-71 type 2 diabetes (Doppler)']=(f,None)
# ---- evaluation helpers ----
def splits(y,groups,reps=5):
    if groups is None: return list(RepeatedStratifiedKFold(n_splits=5,n_repeats=reps,random_state=3).split(np.zeros(len(y)),y))
    return [s for r in range(reps) for s in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=r).split(np.zeros(len(y)),y,groups)]
def multi_auc(df,cols,groups):
    X=df[cols].values.astype(float); y=df.y.values; gr=None if groups is None else df[groups].values; out=[]
    for tr,te in splits(y,gr):
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med); A=np.where(np.isnan(X[tr]),med,X[tr]); B=np.where(np.isnan(X[te]),med,X[te]); mu,sd=A.mean(0),A.std(0)+1e-9
        m=LogisticRegressionCV(Cs=[0.01,0.1,1,10],class_weight='balanced',scoring='roc_auc',cv=4,max_iter=2000).fit((A-mu)/sd,y[tr])
        if len(np.unique(y[te]))==2: out.append(roc_auc_score(y[te],m.decision_function((B-mu)/sd)))
    return float(np.mean(out))
SING={}
def single(dname,df,c_):
    if (dname,c_) in SING: return SING[(dname,c_)]
    k=df[c_].notna().values; y=df.y.values[k]; x=SIGN[c_]*df[c_].values[k]
    if k.sum()<10 or len(np.unique(y))<2: SING[(dname,c_)]=None; return None
    a=roc_auc_score(y,x); B=[roc_auc_score(y[i],x[i]) for i in [rng.integers(0,len(y),len(y)) for _ in range(400)] if len(np.unique(y[i]))==2]
    SING[(dname,c_)]=(a,np.percentile(B,2.5),np.percentile(B,97.5)); return SING[(dname,c_)]
def evaluate(panel):
    res={}
    for dname,(df,groups) in EXT.items():
        cols=[c_ for c_ in dict.fromkeys(panel) if c_ in df.columns and df[c_].notna().sum()>=10]
        if not cols: continue
        s=[single(dname,df,c_) for c_ in cols]; s=[v for v in s if v]
        res[dname]=dict(features=cols,auc=multi_auc(df,cols,groups),reliable=sum(v[1]>0.5 for v in s),opposite=sum(v[2]<0.5 for v in s),tested=len(s),mean_single=np.mean([v[0] for v in s]))
    return res
OUT={}
for m in METH:
    panel=[str(x) for x in FULL[('LEFT',m)]]+[str(x) for x in FULL[('RIGHT',m)]]; OUT[m]=evaluate(panel); print('done',m,flush=True)
names_L=[str(x) for x in FULL[('LEFT','_names')]]; names_R=[str(x) for x in FULL[('RIGHT','_names')]]; NULL=[]
for r in range(30):
    panel=list(rng.choice(names_L,11,replace=False))+list(rng.choice(names_R,11,replace=False)); NULL.append(evaluate(panel))
pd.to_pickle(dict(OUT=OUT,NULL=NULL,SING=SING,SIGN=SIGN,EXT_N={k:(len(v[0]),int(v[0].y.sum())) for k,v in EXT.items()}),'comparison_external.pkl')
D=list(EXT); rows=[]
for m in METH+['Random panels (null, 30 draws)']:
    R=OUT.get(m)
    if R is None:
        agg={d:[x[d] for x in NULL if d in x] for d in D}
        row=dict(method=m,datasets_covered=np.mean([len(x) for x in NULL]),pairs=np.mean([sum(v['tested'] for v in x.values()) for x in NULL]),
                 pct_reliable=100*np.mean([sum(v['reliable'] for v in x.values())/max(1,sum(v['tested'] for v in x.values())) for x in NULL]),
                 pct_opposite=100*np.mean([sum(v['opposite'] for v in x.values())/max(1,sum(v['tested'] for v in x.values())) for x in NULL]))
        for d in D: row[d]=np.mean([v['auc'] for v in agg[d]]) if agg[d] else np.nan
    else:
        t=sum(v['tested'] for v in R.values()); row=dict(method=m,datasets_covered=len(R),pairs=t,pct_reliable=100*sum(v['reliable'] for v in R.values())/max(1,t),pct_opposite=100*sum(v['opposite'] for v in R.values())/max(1,t))
        for d in D: row[d]=R[d]['auc'] if d in R else np.nan
    rows.append(row)
T=pd.DataFrame(rows); T['mean_external_AUC']=T[D].mean(1); pd.set_option('display.width',300); pd.set_option('display.max_columns',20)
print(T.round(3).to_string(index=False)); T.to_csv('comparison_external_table.csv',index=False)
for m in METH: print(m,{d:v['features'] for d,v in OUT[m].items()})
