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

def mini(cx,cy,rx,ry,fill,t,sub):
    o=f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="#0C3A4F" stroke-width="1"/>'
    lines=t.split('\n'); y0=cy-(len(lines)*19+14)/2+15
    for i,l in enumerate(lines): o+=f'<text x="{cx}" y="{y0+i*19}" text-anchor="middle" font-family="Carlito" font-weight="bold" font-size="17" fill="#FFFFFF">{l}</text>'
    o+=f'<text x="{cx}" y="{y0+len(lines)*19+2}" text-anchor="middle" font-family="Carlito" font-size="14" fill="#FFFFFF">{sub}</text>'; return o
def comp(A,B):
    dx,dy=B[0]-A[0],B[1]-A[1]; x1,y1=pt(A,(dx,dy),2); x2,y2=pt(B,(-dx,-dy),3)
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ARR}" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#ah)"/>'
def txt(x,y,t,size=17,bold=False,anchor='middle',italic=False):
    fw=' font-weight="bold"' if bold else ''; fs=' font-style="italic"' if italic else ''
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Carlito" font-size="{size}"{fw}{fs} fill="{ARR}">{t}</text>'
def save(name,W,H,b):
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{ARR}"/></marker></defs><rect width="{W}" height="{H}" fill="#FFFFFF"/>{b}</svg>'
    open(name+'.svg','w').write(svg); o='/mnt/user-data/outputs/'+name; cairosvg.svg2png(url=name+'.svg',write_to=o+'.png',scale=4); cairosvg.svg2pdf(url=name+'.svg',write_to=o+'.pdf')

W,H=1300,800
b=txt(30,40,'Left-hemisphere strokes · best design: single tier (neuro LDA index) · AUC 0.74',22,True,'start')
ST=(150,400,120,64); NI=(650,400,165,64); SP=(1150,400,115,64)
tests=[(230,'GDS (depression)','(weight +0.24 · NHANES)'),(445,'Rey figure\nrecognition','(weight +0.10)'),(660,'Supermarket\nfluency','(weight −0.09)'),(875,'HVLT delayed\nrecall','(weight −0.09 · NHANES)'),(1090,'Letter fluency\n(FAS)','(weight −0.09)')]
for x,t,sub in tests: b+=comp((x,175,100,48),NI)+mini(x,175,100,48,OFF,t,sub)
b+=edge(ST,NI)+edge(NI,SP)+node(*ST,DARK,'Left-Hemisphere\nStroke','(n = 16 vs 40 controls)')+node(*NI,MED,'Neuro LDA Index','(16 cognitive tests, supervised)')+node(*SP,DARK,'Walking\nSpeed','(outcome)')
G1=(820,590,105,46); G2=(1050,590,105,46)
b+=edge(NI,G1,'dot')+edge(NI,G2,'dot')+edge(G2,SP,'dot')+mini(*G1,OFF,'Paretic stride\ntime','(96% of folds)')+mini(*G2,OFF,'Paretic stance\ntime','(90% of folds)')+edge(G1,SP,'dot')
b+=txt(560,560,'double tier: + paretic gait (AUC 0.72)',15,False,'end',True)
b+=txt(650,690,'Tier levels: single 0.74 · double 0.72 · triple 0.70 — adding tiers does not help after left-hemisphere stroke',17,True)
b+=txt(650,722,'External (NHANES, n = 3,104, 234 strokes, side unknown): all mapped cognitive measures shift in the CVES direction;',16)
b+=txt(650,746,'mapped neuro LDA index carries 16% of stroke\'s effect on walking difficulty [CI excludes 0]; AUC 0.68 (0.72 with confounders)',16)
b+=txt(650,780,'dashed arrows: tests combined in the LDA index (top 5 weights shown) · dotted: double-tier extension · "NHANES" = counterpart replicated in NHANES',14,False,'middle',True)
save('CMFS_results_left',W,H,b)

W,H=1450,820
b=txt(30,40,'Right-hemisphere strokes · best design: triple tier (blood LDA index → neuro features → paretic gait LDA index) · AUC 0.85',22,True,'start')
ST=(130,450,112,64); BI=(420,450,135,62); SP=(1340,450,98,62); GI=(1050,450,135,62)
N1=(735,330,115,46); N2=(735,450,115,46); N3=(735,570,115,46)
blood=[(105,'R-MCA autoreg.\nphase shift','(−0.09)'),(300,'R-MCA velocity,\nrebreathing','(−0.06)'),(495,'R autoregulation\nsit → stand','(−0.06)')]
for x,t,sub in blood: b+=comp((x,175,92,46),BI)+mini(x,175,92,46,OFF,t,sub)
gait=[(935,'Paretic\nswing %','(+0.19)'),(1130,'Paretic\nstance %','(−0.19)'),(1325,'Paretic peak\nforce/BW','(−0.13)')]
for x,t,sub in gait: b+=comp((x,175,88,46),GI)+mini(x,175,88,46,OFF,t,sub)
b+=edge(ST,BI)
for N_ in [N1,N2,N3]: b+=edge(BI,N_)+edge(N_,GI)
b+=edge(GI,SP)
b+=node(*ST,DARK,'Right-Hem.\nStroke','(n = 12 vs 40)')+node(*BI,MED,'Blood LDA Index','(45 right-side features)')+node(*GI,MED,'Gait LDA Index','(12 paretic-foot steps)')+node(*SP,DARK,'Walking\nSpeed','(outcome)')
b+=mini(*N1,MED,'Animal fluency','(94% · NHANES)')+mini(*N2,MED,'Supermarket\nfluency','(92% of folds)')+mini(*N3,MED,'Letter fluency\n(FAS)','(42% of folds)')
b+=txt(735,262,'selected neuro features',15,False,'middle',True)
b+=txt(710,700,'Tier levels: single 0.81 (blood LDA index) · double 0.84 (blood index → neuro) · triple 0.85 — adding tiers helps after right-hemisphere stroke',17,True)
b+=txt(710,732,'External (NHANES, side unknown): animal fluency lower after stroke (−0.35 SD), carries 5% of the effect on walking difficulty [CI excludes 0];',16)
b+=txt(710,756,'frozen CVES rule AUC 0.64. Blood (transcranial Doppler) and gait-step components have no NHANES counterpart.',16)
b+=txt(710,792,'dashed arrows: features combined in each LDA index (top weights shown) · solid: causal tiers · % = selection frequency in 50 folds · "NHANES" = replicated in NHANES',14,False,'middle',True)
save('CMFS_results_right',W,H,b)
