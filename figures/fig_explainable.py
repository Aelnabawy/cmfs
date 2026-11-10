import math, cairosvg
from PIL import ImageFont
FB='/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf'
DARK,MED,OFF,ARR='#104861','#0B779E','#5E8CA3','#1F3B46'; TS,SS,LH=22,18,25; fb=ImageFont.truetype(FB,TS)
def node(cx,cy,rx,ry,fill,title,sub):
    lines=title.split('\n'); n=len(lines); top=cy-((n-1)*LH+SS+4)/2+TS*0.35
    s=[f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="#0C3A4F" stroke-width="1"/>']
    for i,l in enumerate(lines):
        y=top+i*LH; w=fb.getlength(l.replace('&amp;','&'))
        s.append(f'<text x="{cx}" y="{y:.1f}" text-anchor="middle" font-family="Carlito" font-weight="bold" font-size="{TS}" fill="#FFFFFF">{l}</text>')
        s.append(f'<line x1="{cx-w/2:.1f}" y1="{y+2.6:.1f}" x2="{cx+w/2:.1f}" y2="{y+2.6:.1f}" stroke="#FFFFFF" stroke-width="1.3"/>')
    s.append(f'<text x="{cx}" y="{top+(n-1)*LH+SS+6:.1f}" text-anchor="middle" font-family="Carlito" font-size="{SS}" fill="#FFFFFF">{sub}</text>')
    return '\n'.join(s)
DASH={'solid':'','dash':'stroke-dasharray="8 5"','dot':'stroke-dasharray="2 5" stroke-linecap="round"'}
def pt(c,d,gap=0):
    (cx,cy,rx,ry)=c; dx,dy=d; t=1/math.sqrt((dx/rx)**2+(dy/ry)**2); L=math.hypot(dx,dy); return cx+dx*t+dx/L*gap, cy+dy*t+dy/L*gap
def lab(x,y,t,anchor='middle'): return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Carlito" font-style="italic" font-size="17" fill="{ARR}">{t}</text>'
def edge(A,B,style='solid'):
    dx,dy=B[0]-A[0],B[1]-A[1]; x1,y1=pt(A,(dx,dy),1); x2,y2=pt(B,(-dx,-dy),3)
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ARR}" stroke-width="1.7" {DASH[style]} marker-end="url(#ah)"/>'
def curve(p0,c,p1,style): return f'<path d="M{p0[0]} {p0[1]} Q {c[0]} {c[1]} {p1[0]} {p1[1]}" fill="none" stroke="{ARR}" stroke-width="1.7" {DASH[style]} marker-end="url(#ah)"/>'

import pandas as pd, numpy as np
RES=pd.read_pickle('/home/claude/ext2/ext_all.pkl')
def T(x,y,t,size=17,bold=False,anchor='middle',italic=False,col=None):
    fw=' font-weight="bold"' if bold else ''; fs=' font-style="italic"' if italic else ''
    return f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}" font-family="Carlito" font-size="{size}"{fw}{fs} fill="{col or ARR}">{t}</text>'
def fnode(cx,cy,rx,ry,fill,t,s1,s2,dashed=False):
    st=' stroke-dasharray="6 4" stroke-width="2"' if dashed else ' stroke-width="1"'
    return (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="#0C3A4F"{st}/>'+T(cx,cy-12,t,18,True,col='#FFFFFF')+T(cx,cy+8,s1,14,col='#FFFFFF')+T(cx,cy+26,s2,14,col='#FFFFFF'))
def seg(x1,y1,x2,y2,style='solid'):
    da={'solid':'','dash':' stroke-dasharray="7 5"'}[style]
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ARR}" stroke-width="1.6"{da} marker-end="url(#ah)"/>'
def onell(cx,cy,rx,ry,tx,ty):
    dx,dy=tx-cx,ty-cy; t=1/np.sqrt((dx/rx)**2+(dy/ry)**2); return cx+dx*t,cy+dy*t
def fmt(v,d=2): return f'{v:+.{d}f}'.replace('-','−')
def p_star(p): return '*' if p<0.05 else ''
def draw(key,fname,expo_t,expo_s,out_t,out_s,b_unit):
    R=RES[key]; Tb=R['table']; nf=len(Tb); W=1320; top=240; gap=112; H=top+nf*gap+170; midY=top+(nf-1)*gap/2+10
    EX=(160,midY,128,66); OU=(1165,midY,120,66); CF=(660,108,240,44); FX=660; RX,RY=238,46
    b=T(30,38,f"{key}: {expo_t.replace(chr(10),' ').replace('Neurodegen.','neurodegenerative')} → mapped CMFS features → {out_t.replace(chr(10),' ').lower()}",22,True,'start')
    b+=seg(468,132,EX[0]+40,EX[1]-EX[3]-4)+seg(852,132,OU[0]-40,OU[1]-OU[3]-4)+T(290,170,'adjusted',16,False,'middle',True)+T(1030,170,'adjusted',16,False,'middle',True)
    for i,r in Tb.iterrows():
        fy=top+i*gap+10; sig=(r.ind_lo>0) or (r.ind_hi<0); fill=MED if sig else OFF
        x1,y1=onell(EX[0],EX[1],EX[2],EX[3],FX-RX,fy); x2,y2=onell(FX,fy,RX,RY,EX[0],EX[1]); b+=seg(x1,y1,x2,y2)
        x3,y3=onell(FX,fy,RX,RY,OU[0],OU[1]); x4,y4=onell(OU[0],OU[1],OU[2],OU[3],FX+RX,fy); b+=seg(x3,y3,x4,y4)
        dirtxt='same direction as CVES' if r.same_direction else 'OPPOSITE to CVES'
        b+=fnode(FX,fy,RX,RY,fill,r.feature,f'a {fmt(r.a)}{p_star(r.a_p)} · b {fmt(r.b,3)}{p_star(r.b_p)} · indirect {fmt(r.indirect,3)} [{fmt(r.ind_lo,3)}, {fmt(r.ind_hi,3)}] ({fmt(100*r.share,0).replace('+','')}%)',f'CVES: {r.CVES_counterpart} · {dirtxt}',dashed=not r.same_direction)
    b+=node(*CF,DARK,'Confounders',R.get('conf_label','(age, sex, BMI' + (', pack-years, hypertension)' if 'NHANES' in key else ')')))
    b+=node(*EX,DARK,expo_t,expo_s)+node(*OU,DARK,out_t,out_s)
    y0=top+nf*gap+40
    a1,a2,a3=R['auc'][0],R['auc_ref'][0],R['auc_both'][0]
    b+=T(660,y0,f"n = {R['n']:,} ({R['n_exp']:,} exposed; outcome measured in {R['n_out']:,}) · total effect of exposure on outcome {fmt(R['total'],3)} {b_unit}",16)
    extra=''
    if 'per_disease' in R: extra=' · by disease: ALS '+f"{R['per_disease'].get('als',(np.nan,))[0]:.2f}"+', Huntington '+f"{R['per_disease'].get('hunt',(np.nan,))[0]:.2f}"+', Parkinson '+f"{R['per_disease'].get('park',(np.nan,))[0]:.2f}"
    b+=T(660,y0+26,f'Classification (nested CV, ridge): mapped features AUC {a1[1]:.2f} · confounders only {a2[1]:.2f} · both {a3[1]:.2f}{extra}',16,True)
    b+=T(660,y0+56,'a = effect of exposure on the feature (SD) · b = feature → outcome per SD, exposure and confounders held fixed · * p &lt; 0.05 · indirect = a × b with 95% bootstrap CI (share of total)',13.5,False,'middle',True)
    b+=T(660,y0+76,'filled blue: indirect effect CI excludes 0 · grey: not significant · dashed outline: shifts opposite to its CVES direction',13.5,False,'middle',True)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{ARR}"/></marker></defs><rect width="{W}" height="{H}" fill="#FFFFFF"/>{b}</svg>'
    open(fname+'.svg','w').write(svg); o='/mnt/user-data/outputs/'+fname; cairosvg.svg2png(url=fname+'.svg',write_to=o+'.png',scale=3); cairosvg.svg2pdf(url=fname+'.svg',write_to=o+'.pdf')
draw('NHANES 1999-2002','CMFS_explainable_NHANES_1999_2002','Stroke','(self-reported)','Gait Speed','(measured 20-ft walk)','m/s')
draw('NHANES 2011-2014','CMFS_explainable_NHANES_2011_2014','Stroke','(self-reported)','Walking\nAbility','(reverse of self-reported\ndifficulty)','points')
draw('PhysioNet gaitndd','CMFS_explainable_PhysioNet_neurodegenerative','Neurodegen.\nDisease','(ALS, Huntington,\nParkinson)','Gait Speed','(measured)','m/s')
draw('PhysioNet gaitpdb','CMFS_explainable_PhysioNet_Parkinsons',"Parkinson's\nDisease",'(93 vs 72 controls)','Gait Speed','(measured)','m/s')
print('ok')
