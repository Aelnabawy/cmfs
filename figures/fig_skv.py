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

K=pd.read_pickle('/home/claude/ext2/skv/skv_res.pkl'); Tb=K['table']
nf=len(Tb); W=1320; top=240; gap=112; H=top+nf*gap+175; midY=top+(nf-1)*gap/2+10
EX=(165,midY,142,66); OU=(1165,midY,120,66); CF=(660,108,240,44); FX=660; RX,RY=262,46
b=T(30,38,'Skvortsov 2024 (subacute stroke): paretic vs intact leg → CMFS gait features → walking speed',22,True,'start')
b+=seg(852,132,OU[0]-40,OU[1]-OU[3]-4)+T(1030,170,'adjusted',16,False,'middle',True)
note={'Double support %':'no difference between legs, as in CVES','Terminal double support onset %':'onset timing, not CVES duration: not comparable','Gait cycle time':'shared by both legs: not testable within person'}
for i,r in Tb.iterrows():
    fy=top+i*gap+10; sigb=(r.b_lo>0) or (r.b_hi<0); siga=r.p_paired_t<0.05
    if r.feature in note: dirtxt=note[r.feature]; dashed=False
    else: dirtxt=f"CVES: {r.CVES_counterpart} · {'same direction as CVES' if r.same_direction else 'OPPOSITE to CVES'}"; dashed=not r.same_direction
    fill=MED if (siga and sigb and r.feature not in note) else OFF
    x1,y1=onell(EX[0],EX[1],EX[2],EX[3],FX-RX,fy); x2,y2=onell(FX,fy,RX,RY,EX[0],EX[1]); b+=seg(x1,y1,x2,y2)
    x3,y3=onell(FX,fy,RX,RY,OU[0],OU[1]); x4,y4=onell(OU[0],OU[1],OU[2],OU[3],FX+RX,fy); b+=seg(x3,y3,x4,y4)
    s1=f"paretic {r.paretic_mean:.1f} vs intact {r.intact_mean:.1f} (d_z {fmt(r.dz)}{p_star(r.p_paired_t)}) · b {fmt(r.b,3)}{p_star(r.b_p)} m/s per SD"
    b+=fnode(FX,fy,RX,RY,fill,r.feature,s1,dirtxt if r.feature in note else dirtxt,dashed=dashed)
b+=node(*CF,DARK,'Confounder','(age; the only one recorded)')
b+=node(*EX,DARK,'Paretic Leg','(vs intact leg, same person)')+node(*OU,DARK,'Walking Speed','(m/s, 10-m walk)')
y0=top+nf*gap+40
b+=T(660,y0,f"n = {K['n']} subacute stroke patients (no controls) · walking speed {K['speed'][0]:.2f} ± {K['speed'][1]:.2f} m/s · b uses the paretic-leg value, age-adjusted",16)
b+=T(660,y0+26,f"Identifying the paretic leg from 4 CMFS gait features (leave-one-person-out ridge): AUC {K['auc_loo']:.2f}, correct in {100*K['pair_acc']:.0f}% of patients",16,True)
b+=T(660,y0+56,'d_z = paired standardised difference, paretic minus intact leg · b = walking speed per SD of the paretic-leg feature · * p &lt; 0.05',13.5,False,'middle',True)
b+=T(660,y0+76,'filled blue: legs differ and the feature relates to walking speed, in the CVES direction · grey: not testable or not comparable',13.5,False,'middle',True)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{ARR}"/></marker></defs><rect width="{W}" height="{H}" fill="#FFFFFF"/>{b}</svg>'
fn='CMFS_explainable_Skvortsov_stroke_paretic'; open(fn+'.svg','w').write(svg); o='/mnt/user-data/outputs/'+fn; cairosvg.svg2png(url=fn+'.svg',write_to=o+'.png',scale=3); cairosvg.svg2pdf(url=fn+'.svg',write_to=o+'.pdf'); print('ok')
