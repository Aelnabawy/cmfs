import pandas as pd, numpy as np, re, glob, warnings
warnings.filterwarnings('ignore')
UP='/mnt/user-data/uploads'
GAITF=['max_force[N]','mean_force[N]','max_pressure[N/cm2]','stride','step','stance','stance_in_%_of_stride','swing_in_%_init','double_in_%','single_limb_support_%','terminate_double_%']
norm=lambda x:re.sub(r'[\s_]','',str(x)).upper()
def catmap():
    d=pd.read_excel(f'{UP}/files-and-variables.xlsx','Data Dictionary',header=None)
    d=d[d[0].notna()]
    m={}
    for v,c in zip(d[0],d[4]):
        c=re.sub(r'\s+',' ',str(c)).strip().lower() if pd.notna(c) else 'none'
        m[norm(v)]=c
    m.update({norm('height/m'):'physical traits',norm('mass/kg'):'physical traits',norm('DM Non-DM STROKE'):'class',norm('completed_visit_status'):'none'})
    return m
def gait():
    rows={}
    for f in sorted(glob.glob('/home/claude/sim/Steps/*-1-steps.csv')):
        sid=re.search(r'S(\d{4})',f).group(1); d=pd.read_csv(f)
        d=d[(d.step_no>=3)&(d.step_no<=327)]; ft=d.foot.str.lower()
        if (ft=='left').sum()==0 or (ft=='right').sum()==0 or pd.read_csv(f).step_no.max()<327: continue
        r={}
        for side,lab in [('right','Right'),('left','Left')]:
            for g in GAITF: r[f'{lab}_{g}']=d.loc[ft==side,g].mean()
        rows[sid]=r
    return pd.DataFrame(rows).T
def load(exclude_cats, keep_admin_misc=True, drop_gaitwalk=True, speed='GAIT - Walk 1 speed (m/s)'):
    s=pd.read_csv(f'{UP}/subjects.csv',low_memory=False); s.columns=[c.strip() for c in s.columns]
    s['sid']=s.subject_number.str.extract(r'(\d{4})')[0]
    G=gait()
    coh=s[(s.completed_visit_status.str.upper().str.strip()=='COMPLETED')&(s.htn_patient_medical_history.astype(str).str.upper().str.strip()=='YES')&s.sid.isin(G.index)].copy()
    cm=catmap()
    admin={norm(x) for x in ['subject_number','study','completed_visit_status','htn_patient_medical_history','GROUP 2','group','sid','note this dataset does not have brain volumes only WMHS']}
    feats=[]
    for c in coh.columns:
        n=norm(c); cat=cm.get(n,'none')
        if n in admin or cat=='class' or cat in exclude_cats: continue
        if drop_gaitwalk and cat.startswith('gait walk'): continue
        if not keep_admin_misc and cat=='none': continue
        feats.append(c)
    y=(coh.group.str.upper().str.strip()=='STROKE').astype(int).values
    X=coh[feats].copy()
    # encode categoricals
    for c in X.columns:
        num=pd.to_numeric(X[c],errors='coerce')
        if num.notna().sum()>=X[c].notna().sum()*0.8 and X[c].notna().sum()>0: X[c]=num
        else: X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan).astype(float)
    X=X.loc[:,X.isna().mean()<=0.5]
    keep=X.isna().mean(axis=1)<=0.5
    X,coh,y=X[keep.values],coh[keep.values],y[keep.values]
    X=X.loc[:,X.nunique()>1]
    from sklearn.impute import KNNImputer
    Xi=pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X),columns=X.columns,index=coh.sid.values)
    return Xi,y,G.loc[coh.sid.values],pd.to_numeric(coh[speed],errors='coerce').values,coh
def significant(Xi,y,thr=0.1):
    from scipy.stats import shapiro,ttest_ind,mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    P=[]
    for c in Xi.columns:
        v=Xi[c].values
        try: normal=shapiro(v).pvalue>0.05
        except: normal=False
        p=(ttest_ind(v[y==1],v[y==0]).pvalue if normal else mannwhitneyu(v[y==1],v[y==0]).pvalue)
        P.append(p)
    P=np.nan_to_num(np.array(P),nan=1.0)
    adj=multipletests(P,method='fdr_bh')[1]
    return pd.DataFrame({'feature':Xi.columns,'p':P,'adj_p':adj}).query('adj_p<@thr').sort_values('adj_p')
