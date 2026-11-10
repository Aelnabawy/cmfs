import sys, warnings; warnings.filterwarnings('ignore')
exec(open('gait_four.py').read().split("if __name__=='__main__':")[0])
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
rng=np.random.default_rng(17)
# ---------- concept map: CMFS feature -> (dataset column, CVES direction: +1 = higher in patients) ----------
CON=[('Animal fluency',-1),('Trail Making B',+1),('HVLT total recall',-1),('HVLT delayed recall',-1),('GDS depression',+1),('Supermarket fluency',-1),('Letter fluency (FAS)',-1),('Rey figure total correct',-1),
     ('RDW',+1),('RBC',-1),('Haemoglobin',-1),('Nocturnal heart rate',+1),('R-MCA CO2 reactivity',-1),('R-MCA autoregulation phase',-1),('R-MCA velocity, rebreathing',-1),('L-MCA CO2 response to tilt',-1),('L-MCA CO2 reactivity',-1),('R-MCA systolic velocity, upright',-1),
     ('Paretic stride time',+1),('Paretic swing time',+1),('Paretic swing %',+1),('Paretic stance %',-1),('Paretic single-limb support %',-1),('Paretic double support %',-1),('Paretic peak force / BW',-1)]
SIGN=dict(CON)
cv_cols={'Animal fluency':'VF: # Animals','Trail Making B':'TM: Part B Time (sec)','HVLT total recall':'HVLT: Total Recall','HVLT delayed recall':'HVLT: Delayed Recall','GDS depression':'GDS Total Response',
 'Supermarket fluency':'VF: # supermarket','Letter fluency (FAS)':'VF: FAS Total','Rey figure total correct':'RCFT: Total Correct','RDW':'RDW%','RBC':'RBC m/uL','Haemoglobin':'Hgb g/dL','Nocturnal heart rate':'24Hour-Nightime-HR',
 'R-MCA CO2 reactivity':'RCO2R_base_hyper_hypo_%','R-MCA autoregulation phase':'MMPF_Base_Mean_PhaseShift (Right)','R-MCA velocity, rebreathing':'MEAN MCAR SUPINE REBREATHING','L-MCA CO2 response to tilt':'RCO2L_base_tilt','L-MCA CO2 reactivity':'RCO2L_base_hyper_hypo_%','R-MCA systolic velocity, upright':'Sys MCAR StandEC'}
col=lambda n: [x for x in Xraw.columns if x.strip()==n.strip()][0]
gidx={'Paretic stride time':'stride','Paretic swing time':'swing','Paretic swing %':'swing_in_%_init','Paretic stance %':'stance_in_%_of_stride','Paretic single-limb support %':'single_limb_support_%','Paretic double support %':'double_in_%','Paretic peak force / BW':'max_force[N]'}
XG=EnvImputer().fit(Gm[cohg]).transform(Gm); draw=rng.random(len(lab))<0.5

# ---------- CVES direction of every concept: sign of the pooled stroke effect (all strokes vs controls, 5 confounders) ----------
import statsmodels.api as sm
kk=coh; yy=(lab[kk]>0).astype(float); usel_all=np.where(lab[kk]==0,draw[kk],lab[kk]==2); ing_all=np.isin(kk,cohg)
for c_,cn in cv_cols.items():
    v=Xraw[col(cn)].values.astype(float)[kk]; m_=~np.isnan(v); SIGN[c_]=float(np.sign(sm.OLS(v[m_],sm.add_constant(np.column_stack([yy[m_],C5[kk][m_]]))).fit().params[1]))
for c_,g in gidx.items():
    jl,jr=base.index(g),12+base.index(g); v=np.where(usel_all,XG[kk][:,jl],XG[kk][:,jr]); m_=ing_all
    SIGN[c_]=float(np.sign(sm.OLS(v[m_],sm.add_constant(np.column_stack([yy[m_],C5[kk][m_],FT[kk][m_]]))).fit().params[1]))
CON=[(c,SIGN[c]) for c,_ in CON]; print('CVES directions:',{c:int(v) for c,v in SIGN.items()})
DS={}
for sname,sv in [('CVES left-hemisphere',1),('CVES right-hemisphere',2)]:
    k=coh[(lab[coh]==0)|(lab[coh]==sv)]; df=pd.DataFrame({'y':(lab[k]==sv).astype(float)})
    for c_,cn in cv_cols.items(): df[c_]=Xraw[col(cn)].values.astype(float)[k]
    ing=np.isin(k,cohg); usel=np.where(lab[k]==0,draw[k],sv==2)   # paretic: left foot for right strokes; controls random foot
    for c_,g in gidx.items():
        jl,jr=base.index(g),12+base.index(g); v=np.where(usel,XG[k][:,jl],XG[k][:,jr]); v[~ing]=np.nan; df[c_]=v
    df['age']=C5[k][:,0]; df['male']=C5[k][:,1]; df['bmi']=C5[k][:,2]; df['packyears']=C5[k][:,3]; df['htn']=C5[k][:,4]
    DS[sname]=(df,list(cv_cols)+list(gidx),['age','male','bmi','packyears','htn'],None)
E=pd.read_pickle('/home/claude/ext2/ext_data.pkl'); M={'CFDRIGHT':'Trail Making B','DSST':'Trail Making B','LBXRDW':'RDW','RDW':'RDW','LBXRBCSI':'RBC','RBC':'RBC','LBXHGB':'Haemoglobin','Hgb':'Haemoglobin',
  'BPXPLS':'Nocturnal heart rate','pulse':'Nocturnal heart rate','animal_fluency':'Animal fluency','CERAD_learning':'HVLT total recall','CERAD_delayed_recall':'HVLT delayed recall','PHQ9':'GDS depression',
  'stride':'Paretic stride time','swing_time':'Paretic swing time','swing_pct':'Paretic swing %','stance_pct':'Paretic stance %','double_pct':'Paretic double support %','peak_force_bw':'Paretic peak force / BW'}
# direction of the external measure relative to its CVES concept (DSST: higher = better, opposite to Trail Making time)
FLIP={'CFDRIGHT':-1,'DSST':-1}
for k_,(d,F,e) in E.items():
    df=pd.DataFrame({'y':d[e].values.astype(float)}); feats=[]
    for c0,*_ in F:
        cn=M[c0]; df[cn]=d[c0].values.astype(float)*FLIP.get(c0,1); feats.append(cn)
    conf=[c for c in (['RIDAGEYR','male','BMXBMI','packyears','htn'] if 'NHANES' in k_ else ['age','male','bmi']) if c in d]
    for c in conf: df['conf_'+c]=d[c].values.astype(float)
    DS[k_]=(df,feats,['conf_'+c for c in conf],None)
sk=pd.read_excel('/home/claude/ext2/skv/stroke_gait.xlsx','Общее',header=None).iloc[4:].reset_index(drop=True); nm=lambda j: pd.to_numeric(sk[j],errors='coerce').values
leg=[]
for i in range(len(sk)):
    for par,cols_ in [(1.0,(6,8,10,3)),(0.0,(7,9,11,4))]:
        st,si,db,cy=nm(cols_[0])[i],nm(cols_[1])[i],nm(cols_[2])[i],nm(cols_[3])[i]
        if not np.isnan(st): leg.append({'y':par,'person':i,'Paretic stance %':st,'Paretic swing %':100-st,'Paretic single-limb support %':si,'Paretic double support %':db,'Paretic swing time':cy*(100-st)/100})
sk=pd.DataFrame(leg); DS['Skvortsov stroke (paretic vs intact leg)']=(sk,['Paretic stance %','Paretic swing %','Paretic single-limb support %','Paretic double support %','Paretic swing time'],[],'person')

# ---------- Novak-lab diabetes cohorts (participants shared with CVES removed) ----------
cves_ids=set(pd.read_csv('/home/claude/ext/cves_subjects.csv',low_memory=False).iloc[:,0].astype(str).str.extract(r'(\d{3,4})')[0].dropna().astype(int))
num=lambda v: pd.to_numeric(v,errors='coerce')
def sid(v): return v.astype(str).str.extract(r'(\d{2,4})')[0].astype(float)
rd=lambda f: pd.read_csv('/home/claude/ext/'+f,low_memory=False,encoding='latin-1')
Dm=rd('GE-79_Summary_Table-Demographics-MRI-Part1.csv'); Cg=rd('GE-79_Summary_Table-Cognitive-Testing.csv'); Lb=rd('GE-79_Summary_Table-Labs-BP-Ophthalmogic-Walk.csv')
for t_,idc in [(Dm,'Patient ID'),(Cg,'patient ID'),(Lb,'patient ID')]: t_['sid']=sid(t_[idc]); t_['vis']=t_['Visit'].astype(str).str.strip()
g79=Dm[['sid','vis','DM, Non-DM, STROKE','BMI']].merge(Cg,on=['sid','vis']).merge(Lb,on=['sid','vis'],suffixes=('','_lab'))
g79=g79[(g79.vis=='2')&(~g79.sid.isin(cves_ids))]
H=rd('GE-79_Summary_Table-MRI-Part5-History.csv'); H['sid']=sid(H[[c for c in H.columns if 'id' in c.lower()][0]])
agec=[c for c in H.columns if c.strip().lower() in ('age','age (years)','age(yrs)')]
MAP79={'Animal fluency':'VF: # Animals','Trail Making B':'TM: Part B Time (sec)','HVLT total recall':'HVLT: Total Recall','HVLT delayed recall':'HVLT: Delayed Recall','GDS depression':'GDS Total Response',
       'Supermarket fluency':'VF: # supermarket','Letter fluency (FAS)':'VF: FAS Total','Rey figure total correct':'RCFT: Total Correct','RDW':'RDW%','RBC':'RBC m/uL','Haemoglobin':'Hgb g/dL','Nocturnal heart rate':'24Hour-Nightime-HR'}
df=pd.DataFrame({'y':(g79['DM, Non-DM, STROKE'].astype(str).str.strip()=='DM').astype(float).values})
for k_,c_ in MAP79.items(): df[k_]=num(g79[c_]).values if c_ in g79 else np.nan
df['conf_bmi']=num(g79['BMI']).values
DS['GE-79 diabetes (visit 1)']=(df,[k for k in MAP79 if df[k].notna().sum()>10],['conf_bmi'],None)
g75=pd.read_csv('/home/claude/ext2/novak/GE-75_data_summary_table.csv',low_memory=False,encoding='latin-1'); g75['sid']=sid(g75['patient ID']); g75=g75[~g75.sid.isin(cves_ids)]
MAP75=dict(MAP79); MAP75.pop('Nocturnal heart rate'); MAP75['R-MCA CO2 reactivity']='Global Vasoreactivity'
df=pd.DataFrame({'y':(g75['Group'].astype(str).str.strip()=='DM').astype(float).values})
for k_,c_ in MAP75.items(): df[k_]=num(g75[c_]).values if c_ in g75 else np.nan
for c_,nm_ in [('age','conf_age'),('BMI','conf_bmi')]: df[nm_]=num(g75[c_]).values
df['conf_male']=g75['sex'].astype(str).str.strip().str.upper().str[0].map({'M':1.0,'F':0.0}).values
DS['GE-75 diabetes']=(df,[k for k in MAP75 if df[k].notna().sum()>10],['conf_age','conf_male','conf_bmi'],None)
g71=pd.read_csv('/home/claude/ext2/novak/GE-71_Data_Summary_Table.csv',low_memory=False,encoding='latin-1'); g71['sid']=sid(g71['SUBJECT NUMBER']); g71=g71[~g71.sid.isin(cves_ids)]
dl=(num(g71['mn_MCAL(Tilt MN)'])-num(g71['mn_MCAL(Baseline MR)']))/num(g71['mn_MCAL(Baseline MR)'])*100/(num(g71['ETCO2(Tilt MN)'])-num(g71['ETCO2(Baseline MR)'])).replace(0,np.nan)
df=pd.DataFrame({'y':(g71['group2'].astype(str).str.strip()=='DM').astype(float).values,'R-MCA CO2 reactivity':num(g71['Relative CO2 Reactivity MCAR']).values,'L-MCA CO2 reactivity':num(g71['Relative CO2 Reactivity MCAL']).values,
                 'R-MCA velocity, rebreathing':num(g71['mn_MCAR(Hypo Mean)']).values,'R-MCA systolic velocity, upright':num(g71['sys_MCAR(Tilt MN)']).values,'L-MCA CO2 response to tilt':dl.values,
                 'conf_age':num(g71['Age']).values,'conf_male':(g71['Gender'].astype(str).str.strip().str.upper()=='M').astype(float).values,'conf_bmi':num(g71['BMI']).values})
DS['GE-71 diabetes (TCD)']=(df,['R-MCA CO2 reactivity','L-MCA CO2 reactivity','R-MCA velocity, rebreathing','R-MCA systolic velocity, upright','L-MCA CO2 response to tilt'],['conf_age','conf_male','conf_bmi'],None)
for k_ in ['GE-79 diabetes (visit 1)','GE-75 diabetes','GE-71 diabetes (TCD)']: print(k_,'n',len(DS[k_][0]),'patients',int(DS[k_][0].y.sum()),'features',DS[k_][1])
# ---------- metrics ----------
def splits(y,groups,reps=10):
    if groups is None: return list(RepeatedStratifiedKFold(n_splits=5,n_repeats=reps,random_state=3).split(np.zeros(len(y)),y))
    out=[]
    for r in range(reps): out+=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=r).split(np.zeros(len(y)),y,groups))
    return out
def single(df,f,groups):
    k=df[f].notna().values; y=df.y.values[k]; x=SIGN[f]*df[f].values[k]; g=None if groups is None else df[groups].values[k]
    if len(np.unique(y))<2 or k.sum()<10: return None
    auc=roc_auc_score(y,x); B=[]
    for _ in range(500):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))==2: B.append(roc_auc_score(y[i],x[i]))
    sens=[];spec=[]
    for tr,te in splits(y,g):
        xs=np.unique(x[tr]); best=max(xs,key=lambda t: (x[tr][y[tr]==1]>=t).mean()+(x[tr][y[tr]==0]<t).mean()-1)
        p=x[te]>=best; sens.append(p[y[te]==1].mean()); spec.append((~p)[y[te]==0].mean())
    return dict(n=int(k.sum()),cases=int(y.sum()),prevalence=y.mean(),AUC=auc,AUC_lo=np.percentile(B,2.5),AUC_hi=np.percentile(B,97.5),PR_AUC=average_precision_score(y,x),
                minority_recall=np.mean(sens),specificity=np.mean(spec),balanced_accuracy=(np.mean(sens)+np.mean(spec))/2)
def multi(df,feats,conf,groups,use):
    cols=(feats if 'f' in use else [])+(conf if 'c' in use else [])
    if not cols: return None
    X=df[cols].values.astype(float); y=df.y.values; g=None if groups is None else df[groups].values; P=np.zeros(len(y)); res=[]; shap=[]; coefs=[]
    for r_,(tr,te) in enumerate(splits(y,g,reps=5)):
        med=np.nanmedian(X[tr],0); Xtr=np.where(np.isnan(X[tr]),med,X[tr]); Xte=np.where(np.isnan(X[te]),med,X[te]); mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9
        m=LogisticRegressionCV(Cs=[0.01,0.1,1,10],class_weight='balanced',scoring='roc_auc',cv=4,max_iter=2000).fit((Xtr-mu)/sd,y[tr])
        z=(Xte-mu)/sd; pr=m.predict_proba(z)[:,1]; c=m.coef_[0]; coefs.append(c); shap.append(np.abs(z*c).mean(0))
        if len(np.unique(y[te]))==2: res.append((roc_auc_score(y[te],pr),average_precision_score(y[te],pr),(pr[y[te]==1]>=0.5).mean(),(pr[y[te]==0]<0.5).mean()))
    R=np.array(res); sh=np.mean(shap,0); cf=np.mean(coefs,0)
    out=dict(AUC=R[:,0].mean(),PR_AUC=R[:,1].mean(),minority_recall=R[:,2].mean(),specificity=R[:,3].mean(),balanced_accuracy=(R[:,2].mean()+R[:,3].mean())/2,prevalence=y.mean(),n=len(y))
    expl=pd.DataFrame({'feature':cols,'mean_abs_SHAP':sh,'mean_coef':cf,'sign_matches_CVES':[(np.sign(c)==SIGN.get(f,0)) if f in SIGN else None for f,c in zip(cols,cf)]})
    return out,expl
ROWS=[]; MULT=[]; EXPL={}
for name,(df,feats,conf,groups) in DS.items():
    for f in feats:
        r=single(df,f,groups)
        if r: ROWS.append(dict(dataset=name,feature=f,**r))
    for use,lab_ in [('f','CMFS features'),('c','confounders only'),('fc','CMFS features + confounders')]:
        o=multi(df,feats,conf,groups,use)
        if o: MULT.append(dict(dataset=name,model=lab_,**o[0]));
        if o and use=='f': EXPL[name]=o[1]
    print('done',name,flush=True)
pd.to_pickle(dict(single=pd.DataFrame(ROWS),multi=pd.DataFrame(MULT),expl=EXPL,concepts=CON),'/home/claude/sim/reliability2.pkl')

S1=pd.DataFrame(ROWS); pd.set_option('display.width',260)
new=['GE-79 diabetes (visit 1)','GE-75 diabetes','GE-71 diabetes (TCD)','Skvortsov stroke (paretic vs intact leg)']
print(S1[S1.dataset.isin(new)][['dataset','feature','n','cases','AUC','AUC_lo','AUC_hi','minority_recall','balanced_accuracy']].round(2).to_string(index=False))
print(pd.DataFrame(MULT)[lambda x: x.dataset.isin(new)].round(3).to_string(index=False))
