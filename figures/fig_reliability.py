import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
R=pd.read_pickle('reliability2.pkl'); S=R['single']; MU=R['multi']; EX=R['expl']; CON=R['concepts']
DARK,MED,OFF,ARR,RED='#104861','#0B779E','#5E8CA3','#1F3B46','#B4532A'
plt.rcParams.update({'font.family':'Carlito','font.size':12})
cols=['CVES left-hemisphere','CVES right-hemisphere','NHANES 1999-2002','NHANES 2011-2014','Skvortsov stroke (paretic vs intact leg)','GE-79 diabetes (visit 1)','GE-75 diabetes','GE-71 diabetes (TCD)','PhysioNet Parkinson','PhysioNet neurodegenerative']
short=['CVES\nleft strokes','CVES\nright strokes','NHANES\n1999–2002','NHANES\n2011–2014','Skvortsov\nstroke legs','GE-79\ndiabetes','GE-75\ndiabetes','GE-71\ndiabetes (TCD)',"PhysioNet\nParkinson's",'PhysioNet\nneurodegen.']
rows=[c for c,_ in CON]; groups=[('Cognitive',8),('Blood / cerebrovascular',10),('Paretic gait',7)]
A=S.pivot_table(index='feature',columns='dataset',values='AUC').reindex(index=rows,columns=cols)
LO=S.pivot_table(index='feature',columns='dataset',values='AUC_lo').reindex(index=rows,columns=cols); HI=S.pivot_table(index='feature',columns='dataset',values='AUC_hi').reindex(index=rows,columns=cols)
REC=S.pivot_table(index='feature',columns='dataset',values='minority_recall').reindex(index=rows,columns=cols)
cmap=LinearSegmentedColormap.from_list('d',[RED,'#F3D9CC','#FFFFFF','#CFE3EE',DARK])
fig,ax=plt.subplots(figsize=(18.5,14)); norm=TwoSlopeNorm(vmin=0.15,vcenter=0.5,vmax=1.0)
M=A.values.astype(float); ax.imshow(np.where(np.isnan(M),0.5,M),cmap=cmap,norm=norm,aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isnan(M[i,j]): ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,color='#EEEEEE')); ax.text(j,i,'–',ha='center',va='center',fontsize=12,color='#9A9A9A'); continue
        sig=LO.values[i,j]>0.5 or HI.values[i,j]<0.5; col_='white' if (M[i,j]>0.8 or M[i,j]<0.25) else ARR
        ax.text(j,i-0.12,f'{M[i,j]:.2f}{"*" if sig else ""}',ha='center',va='center',fontsize=13,fontweight='bold' if sig else 'normal',color=col_)
        ax.text(j,i+0.25,f'recall {REC.values[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color=col_)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(short,fontsize=12,color=ARR); ax.xaxis.tick_top()
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=12.5,color=ARR)
y=-0.5
for g,n in groups:
    ax.axhline(y,color='white',lw=4); ax.text(-0.245,y+n/2-0.5,g,rotation=90,ha='center',va='center',fontsize=13,fontweight='bold',color=ARR,transform=matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)); y+=n
ax.axvline(1.5,color=ARR,lw=2.5)
prev={}
for c in cols:
    r_=MU[(MU.dataset==c)&(MU.model=='CMFS features')].iloc[0]; prev[c]=(int(round(r_.prevalence*r_.n)),int(r_.n))
for j,c in enumerate(cols): ax.text(j,len(rows)-0.35,f'{prev[c][0]}/{prev[c][1]}',ha='center',va='top',fontsize=9.5,color=ARR)
ax.text(0.5,len(rows)+0.35,'discovery data',ha='center',va='top',fontsize=10.5,style='italic',color=ARR); ax.text(5.5,len(rows)+0.35,'external datasets',ha='center',va='top',fontsize=10.5,style='italic',color=ARR)
ax.set_ylim(len(rows)+0.9,-0.5); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
fig.suptitle('Does each CMFS feature identify the patient (minority) group?  AUC with the CVES direction frozen',fontsize=16,fontweight='bold',color=ARR,x=0.02,ha='left',y=0.995)
fig.text(0.02,0.012,'Blue: identifies patients in the CVES direction · red: shifts the opposite way · * 95% bootstrap CI excludes 0.5 · recall = minority-class sensitivity at a threshold chosen in training folds (5-fold × 10)\n'
         'Bottom row: cases/total. CVES columns are the discovery data (features were selected there, so these AUCs are optimistic). Skvortsov compares paretic with intact legs of the same patients.\n'
         'GE cohorts: type 2 diabetes vs controls from the CVES laboratory, CVES participants removed (GE-75 has only 8 controls with cognitive data).  – = not measured in that dataset.',fontsize=10.5,color=ARR,style='italic')
plt.subplots_adjust(left=0.2,right=0.995,top=0.92,bottom=0.075)
for e in ['png','pdf']: fig.savefig(f'/mnt/user-data/outputs/CMFS_feature_reliability_map.{e}',dpi=250)
# ---------- per-dataset explanation panels ----------
order=['NHANES 1999-2002','NHANES 2011-2014','Skvortsov stroke (paretic vs intact leg)','GE-79 diabetes (visit 1)','GE-75 diabetes','GE-71 diabetes (TCD)','PhysioNet Parkinson','PhysioNet neurodegenerative','CVES left-hemisphere','CVES right-hemisphere']
title={'NHANES 1999-2002':'NHANES 1999–2002 (stroke)','NHANES 2011-2014':'NHANES 2011–2014 (stroke)','Skvortsov stroke (paretic vs intact leg)':'Skvortsov (paretic vs intact leg)','PhysioNet Parkinson':"PhysioNet Parkinson's",
       'PhysioNet neurodegenerative':'PhysioNet neurodegenerative','CVES left-hemisphere':'CVES left strokes (discovery)','CVES right-hemisphere':'CVES right strokes (discovery)','GE-79 diabetes (visit 1)':'GE-79 type 2 diabetes','GE-75 diabetes':'GE-75 type 2 diabetes (8 controls)','GE-71 diabetes (TCD)':'GE-71 type 2 diabetes (Doppler)'}
fig,axs=plt.subplots(3,4,figsize=(23,19),gridspec_kw={'hspace':0.5,'wspace':0.66}); axs=axs.ravel()
for a_,d in zip(axs,order):
    E=EX[d].sort_values('mean_abs_SHAP'); colr=[MED if m is True else (RED if m is False else OFF) for m in E.sign_matches_CVES]
    if len(E)>12: E=E.tail(12); colr=colr[-12:]
    a_.barh(E.feature,E.mean_abs_SHAP,color=colr); a_.tick_params(axis='y',labelsize=11.5,colors=ARR); a_.tick_params(axis='x',labelsize=10)
    m=MU[(MU.dataset==d)&(MU.model=='CMFS features')].iloc[0]; c0=MU[(MU.dataset==d)&(MU.model=='confounders only')]
    ref=f' (confounders {c0.iloc[0].AUC:.2f})' if len(c0) else ''
    a_.set_title(f'{title[d]}\nAUC {m.AUC:.2f}{ref} · PR-AUC {m.PR_AUC:.2f} vs prevalence {m.prevalence:.2f}\nminority recall {m.minority_recall:.2f} · balanced accuracy {m.balanced_accuracy:.2f}',fontsize=12.5,color=ARR,loc='left')
    a_.set_xlabel('mean |SHAP| (contribution to the class-weighted classifier)',fontsize=10.5,color=ARR); a_.spines[['top','right']].set_visible(False)
axs[-2].axis('off'); axs[-1].axis('off')
axs[-1].text(0,0.98,'How to read',fontsize=15,fontweight='bold',color=ARR,transform=axs[-1].transAxes)
axs[-1].text(0,0.86,'Each panel: class-weighted logistic regression on the\nCMFS features available in that dataset (nested\ncross-validation; the minority class is up-weighted).\n\nBars: how much each feature drives the prediction.\nBlue = pushes towards "patient" in the CVES direction\nRed = pushes the opposite way.\n\nCVES panels show the 12 largest of 25 features and\nare optimistic (features chosen on these data).',fontsize=12.5,color=ARR,transform=axs[-1].transAxes,va='top')
fig.suptitle('Explaining the minority-class classifiers: which CMFS features carry the prediction in each dataset',fontsize=17,fontweight='bold',color=ARR,x=0.01,ha='left')
for e in ['png','pdf']: fig.savefig(f'/mnt/user-data/outputs/CMFS_feature_explanations_SHAP.{e}',dpi=200,bbox_inches='tight')
print('ok')
