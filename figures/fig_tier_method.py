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
W,H=1400,850
AMB='#B7791F'
def badge(x,y,k,r=15,fs=17): return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{AMB}" stroke="#7A4F0E" stroke-width="1"/><text x="{x}" y="{y+6}" text-anchor="middle" font-family="Carlito" font-weight="bold" font-size="{fs}" fill="#FFFFFF">{k}</text>'
def pill(cx,cy,w,h,fill,t,sub=None,tc='#FFFFFF'):
    o=f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}" stroke="#0C3A4F" stroke-width="1"/>'
    if sub: o+=f'<text x="{cx}" y="{cy-4}" text-anchor="middle" font-family="Carlito" font-weight="bold" font-size="19" fill="{tc}">{t}</text><text x="{cx}" y="{cy+18}" text-anchor="middle" font-family="Carlito" font-size="15.5" fill="{tc}">{sub}</text>'
    else: o+=f'<text x="{cx}" y="{cy+6}" text-anchor="middle" font-family="Carlito" font-weight="bold" font-size="18" fill="{tc}">{t}</text>'
    return o
def line(x1,y1,x2,y2,w=1.7): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ARR}" stroke-width="{w}" marker-end="url(#ah)"/>'
b=f'<text x="28" y="40" font-family="Carlito" font-weight="bold" font-size="24" fill="{ARR}">A  Causal structure and tier levels</text>'
CF=(700,95,250,50); ST=(140,290,112,62); T1=(430,290,128,62); T2=(712,290,128,62); T3=(994,290,128,62); SP=(1268,290,112,62)
b+=line(475,120,215,236)+line(925,120,1190,236)+lab(330,170,'adjusted','end')+lab(1080,170,'adjusted','start')
for A_,B_ in [(ST,T1),(T1,T2),(T2,T3),(T3,SP)]: b+=edge(A_,B_)
b+=node(*CF,DARK,'Confounders','(age, sex, BMI, pack-years, hypertension)')
b+=node(*ST,DARK,'Stroke [side]','(exposure: L or R)')
b+=node(*T1,MED,'Tier 1: Blood','(45 side-specific)')+node(*T2,MED,'Tier 2: Neuro','(16 cognitive tests)')+node(*T3,MED,'Tier 3: Gait','(12 paretic-foot steps)')
b+=node(*SP,DARK,'Walking Speed','(outcome)')
b+=f'<text x="700" y="384" text-anchor="middle" font-family="Carlito" font-style="italic" font-size="16" fill="{ARR}">tiers keep their causal order; a level uses one, two or all three tiers (direct routes to the outcome allowed)</text>'
b+=pill(700,425,1100,40,OFF,'Single tier:  blood   |   neuro   |   paretic gait')
b+=pill(700,475,1100,40,OFF,'Double tier:  blood → neuro   |   blood → paretic gait   |   neuro → paretic gait')
b+=pill(700,525,1100,40,OFF,'Triple tier:  blood → neuro → paretic gait')
b+=f'<line x1="28" y1="566" x2="1372" y2="566" stroke="#C9D3D8" stroke-width="1.5"/>'
b+=f'<text x="28" y="604" font-family="Carlito" font-weight="bold" font-size="24" fill="{ARR}">B  Within each level, per hemisphere, inside nested cross-validation</text>'
b+=pill(330,665,430,70,MED,'Keep tier as features','score a × b, select top k (5 / 3 / 3)')
b+=pill(330,760,430,70,MED,'Or transform tier to LDA index','stroke-supervised, cross-fitted')
b+=badge(100,665,'1')+badge(100,760,'2')
b+=f'<text x="330" y="822" text-anchor="middle" font-family="Carlito" font-style="italic" font-size="15" fill="{ARR}">a = stroke → feature;  b = feature → speed, stroke, confounders and upstream tiers held fixed</text>'
b+=line(548,665,700,705)+line(548,760,700,722)
b+=pill(840,712,260,70,DARK,'Ridge classifier','stroke [side] vs controls')
b+=line(972,712,1040,712)
b+=pill(1195,670,300,60,DARK,'Side-specific panel','features + indices, AUC')
b+=pill(1195,770,300,60,DARK,'External validation','NHANES (mappable features)')
b+=line(1195,702,1195,736)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{ARR}"/></marker></defs><rect width="{W}" height="{H}" fill="#FFFFFF"/>{b}</svg>'
open('tier_method.svg','w').write(svg)
o='/mnt/user-data/outputs/CMFS_tier_methodology'; cairosvg.svg2png(url='tier_method.svg',write_to=o+'.png',scale=4); cairosvg.svg2pdf(url='tier_method.svg',write_to=o+'.pdf'); print('ok')
