import warnings; warnings.filterwarnings('ignore')
exec(open('comparison.py').read().split("# ---------------- 1. internal nested comparison ----------------")[0])
num=lambda col: pd.to_numeric(d[col],errors='coerce').values.astype(float)
print('years col (strokes vs controls):',np.nanmean(num('years')[lab>0]),np.nanmean(num('years')[lab==0]), '| NIHSS strokes mean',np.nanmean(num('NIHSS')[lab>0]),'| MRS strokes',pd.Series(num('MRS')[lab>0]).value_counts().to_dict())
# ---- participant flow
s=pd.read_csv('/mnt/user-data/uploads/subjects.csv',low_memory=False); s.columns=[c.strip() for c in s.columns]
st=s.completed_visit_status.astype(str).str.upper().str.strip(); g=s.group.astype(str).str.upper().str.strip()
print('\nFLOW: enrolled',len(s),dict(g.value_counts()),'| completed',int((st=='COMPLETED').sum()),dict(g[st=='COMPLETED'].value_counts()))
print('  completed with walking speed (d):',len(d),{k:int(v) for k,v in pd.Series(np.where(d.group.str.upper().str.strip()=='STROKE','STROKE','CONTROL')).value_counts().items()})
sd_=d['Stroke Side'].astype(str).str.upper().str.strip(); strokes=d.group.str.upper().str.strip()=='STROKE'
print('  strokes in d by side:',sd_[strokes].value_counts().to_dict())
print('  full analysis cohort (coh):',len(coh),{k:int((lab[coh]==v).sum()) for k,v in [('left',1),('right',2),('control',0)]})
print('  step-data analysis cohort (cohg):',len(cohg),{k:int((lab[cohg]==v).sum()) for k,v in [('left',1),('right',2),('control',0)]})
# ---- characteristics table for cohg by group (and coh)
def fmt_mean(v): v=v[~np.isnan(v)]; return f'{v.mean():.1f} ± {v.std(ddof=1):.1f}' if len(v) else '—'
def fmt_med(v): v=v[~np.isnan(v)]; return f'{np.median(v):.1f} [{np.percentile(v,25):.1f}, {np.percentile(v,75):.1f}]' if len(v) else '—'
def fmt_n(m,den): return f'{int(m.sum())} ({100*m.sum()/den:.0f}%)'
race=d.race.astype(str).str.upper().str.strip(); white=race.isin(['WHITE']); black=race.isin(['AFRICAN AMERICAN','AA'])
rows=[]
for cname,cset in [('Analysis cohort (step data)',cohg),('Full cohort',coh)]:
    for gname,gv in [('Left-hemisphere strokes',1),('Right-hemisphere strokes',2),('Controls',0)]:
        k=cset[lab[cset]==gv]; n=len(k)
        rows.append({'Cohort':cname,'Group':gname,'n':n,'Age, years':fmt_mean(num('age')[k]),'Female':fmt_n((d.gender.astype(str).str.upper().str[0]=='F').values[k],n),
            'BMI, kg/m²':fmt_mean(num('bmi')[k]),'Hypertension':fmt_n((d.htn_patient_medical_history.astype(str).str.upper().str.strip()=='YES').values[k],n),
            'Pack-years':fmt_med(num('pack_years')[k]),'White / African American':f'{int(white.values[k].sum())} / {int(black.values[k].sum())}',
            'Walking speed, m/s':fmt_mean(Y[k]),'NIHSS':fmt_med(num('NIHSS')[k]) if gv else '—','modified Rankin':fmt_med(num('MRS')[k]) if gv else '—'})
T=pd.DataFrame(rows); pd.set_option('display.width',250); print('\n',T.to_string(index=False)); T.to_csv('/home/claude/tripod/characteristics.csv',index=False)
# ---- missing data per candidate predictor in the analysis cohort
bc=sorted(set(LB+RB)); miss=[]
for c_ in bc+list(NEURO): miss.append(dict(predictor=c_,tier='cerebrovascular' if c_ in bc else 'cognitive',missing_pct=100*np.isnan(pd.to_numeric(d[c_],errors='coerce').values.astype(float)[cohg]).mean()))
for j,gn in enumerate(GN):
    for side,off in [('left foot',0),('right foot',12)]: miss.append(dict(predictor=f'{gn} ({side})',tier='gait',missing_pct=100*np.isnan(Gm[cohg][:,j+off]).mean()))
M=pd.DataFrame(miss); M.to_csv('/home/claude/tripod/missing.csv',index=False)
print('\nMISSING (% of analysis cohort):'); print(M.groupby('tier').missing_pct.describe()[['count','mean','50%','max']].round(1).to_string()); print('  predictors >20% missing:',int((M.missing_pct>20).sum()))
