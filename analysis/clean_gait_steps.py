"""Robust per-foot gait features from CVES step files (Section 4.1).
Input: Gait_Steps_Features.csv (one row per walking-test file) and the per-step files (Steps/<file>).
Steps 3-327 are kept; strides > 2x or < 0.5x the median are removed as gross artefacts; each foot's
feature is the median over its remaining steps. Files are flagged when peak force exceeds 2x body
weight (calibration) or when more than 5% of strides were removed."""
import pandas as pd, numpy as np, os
g=pd.read_csv('/mnt/user-data/uploads/Gait_Steps_Features.csv').rename(columns={'Unnamed: 0':'file'})
m=pd.read_csv('gait_file_checked.csv',dtype={'sid':str})
feat=[c for c in g.columns if c.startswith(('Right','Left'))]; base=sorted({c.split('_',1)[1] for c in feat},key=lambda x:feat.index('Right_'+x))
def path(f):
    for p in [f'/mnt/user-data/uploads/Steps/{f}',f'Steps/{f}',f'/home/claude/ext/steps/{f}']:
        if os.path.exists(p): return p
rows=[]
for f in g.file:
    p=path(f); r={'file':f}
    if p:
        st=pd.read_csv(p); w=st[(st.step_no>=3)&(st.step_no<=327)]; med=w.stride.median(); bad=(w.stride>2*med)|(w.stride<0.5*med); w=w[~bad]
        r['gross_artefact_strides_removed_%']=round(100*bad.mean(),1)
        for side,lab in [('right','Right'),('left','Left')]:
            q=w[w.foot.str.lower()==side]
            for c in base: r[f'{lab}_{c}']=q[c].median() if c in q.columns else np.nan
    rows.append(r)
Rb=pd.DataFrame(rows)
Rb['file_type']=g.file.str.extract(r'S\d{4}([A-Z])')[0]
mm=m.set_index('file')
Rb['peak_force_over_body_weight']=((Rb['Right_max_force[N]']+Rb['Left_max_force[N]'])/2/(mm.loc[Rb.file,'mass'].values*9.81)).round(2)
Rb['flag']=np.where(Rb['peak_force_over_body_weight']>2,'force > 2x body weight (calibration?)','')
Rb.loc[Rb['gross_artefact_strides_removed_%']>5,'flag']=Rb['flag'].where(Rb['flag']=='',Rb['flag']+'; ')+'>5% artefact strides removed'
Rb.loc[Rb['Right_stride'].isna(),'flag']='step file not available - robust values missing'
Rb.to_csv('/mnt/user-data/outputs/Gait_Steps_Features_cleaned.csv',index=False)
print(Rb.shape, Rb.flag.value_counts().to_dict())
