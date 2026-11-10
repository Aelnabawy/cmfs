import sys, warnings; sys.path.insert(0,'/home/claude/sim'); warnings.filterwarnings('ignore')
from cves_pipe import catmap, norm, UP
import numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
cm=catmap()
s=pd.read_csv(f'{UP}/subjects.csv',low_memory=False); s.columns=[c.strip() for c in s.columns]
num=lambda d,c: pd.to_numeric(d[c],errors='coerce')
s['speed']=np.nanmean(np.column_stack([num(s,'GAIT - Walk 1 speed (m/s)'),num(s,'GAIT - Walk 2 speed (m/s)')]),axis=1)
d=s[(s.completed_visit_status.astype(str).str.upper().str.strip()=='COMPLETED')&s.speed.notna()].reset_index(drop=True)
S=(d.group.str.upper().str.strip()=='STROKE').astype(int).values; Y=d.speed.values; n=len(d)
z=lambda v:(v-np.nanmean(v))/np.nanstd(v); fill=lambda v: np.where(np.isnan(v),np.nanmedian(v),v)
CN=['Age','Sex','BMI','Pack-years','Hypertension']
C=np.column_stack([z(fill(num(d,'age').values)),(d.gender.astype(str).str.upper().str[0]=='M').astype(float),z(fill(num(d,'bmi').values)),z(fill(num(d,'pack_years').values)),(d.htn_patient_medical_history.astype(str).str.upper().str.strip()=='YES').astype(float)])
EXC={'physical traits','behavioral traits','symptoms','family history','stroke information','class'}
MED={norm(m) for m in ['ANTIPLATELETS','ANTICOAGULANTS','ANTIHYPERLIPIDEMIC','ANTIPARKINSONIAN','STATINS','ESTROGEN','ACE INHIBITORS','ARBS','BETA BLOCKERS','DIURETICS','CA ++ BLOCKERS','INSULIN(Yes_or_No)','ORAL HYPOGLYCEMIC','MRI YES OR NO','HRV_SDNNINDX']}
admin={norm(x) for x in ['subject_number','study','completed_visit_status','htn_patient_medical_history','GROUP 2','group','speed']}
NEURO={'rey-osterreith complex figure (rocf)','hopkins verbal learning test','verbal fluency test','digit span test','response measurements','the trail making test (parts a and b)','clock-in-a box','wechsler test of adult reading (wtar)'}
OTHER={'sleep','white matter hyerintensities','none'}
feats=[c for c in d.columns if norm(c) not in admin and norm(c) not in MED and cm.get(norm(c),'none') not in EXC and not cm.get(norm(c),'none').startswith('gait walk') and cm.get(norm(c),'none')!='none']
Xraw=d[feats].apply(pd.to_numeric,errors='coerce'); Xraw=Xraw.loc[:,(Xraw.isna().mean()<=0.5)&(Xraw.nunique()>1)]
names=list(Xraw.columns); cat={c:cm.get(norm(c),'none') for c in names}
domain={c:('Neuro' if cat[c] in NEURO else ('Other' if cat[c] in OTHER else 'Blood')) for c in names}
def impute(Xtr,Xte=None):
    sc=StandardScaler().fit(Xtr); imp=KNNImputer(n_neighbors=5).fit(sc.transform(Xtr)); tr=imp.transform(sc.transform(Xtr))
    return (tr,imp.transform(sc.transform(Xte))) if Xte is not None else tr
Xi=impute(Xraw.values)
def family(f):
    if f.startswith('TM:'): return 'Trail Making'
    if 'animals' in f.lower(): return 'Animal fluency'
    if f.startswith('RCFT: Copy'): return 'RCFT copy'
    if f.startswith('RCFT: I.R.'): return 'RCFT immediate recall'
    if f.startswith('RCFT: D.R.'): return 'RCFT delayed recall'
    return f
