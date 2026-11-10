"""Point every script at your own folders (run once after unzipping).
Usage: python configure_paths.py --data DIR --external DIR --out DIR
  --data      folder holding CVES subjects.csv and Gait_Steps_Features.csv (PhysioNet DOI 10.13026/C2DW96)
  --external  folder holding the external datasets (NHANES .xpt files, PhysioNet gait and diabetes cohorts)
  --out       folder where results, figures and intermediate files are written"""
import argparse, os, glob
ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--external',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
here=os.path.dirname(os.path.abspath(__file__)); j=lambda *p: os.path.join(*p).replace('\\','/')
MAP=[('/mnt/user-data/uploads',a.data),('/mnt/user-data/outputs',a.out),('/home/claude/ext2',j(a.external,'ext2')),('/home/claude/ext',j(a.external,'ext')),
     ('/home/claude/simstudy',j(here,'simulation')),('/home/claude/sim',j(here,'analysis')),('/home/claude/fig',j(here,'figures')),('/home/claude/tripod',j(a.out,'tripod')),('/home/claude/cmfs',here)]
n=0
for f in glob.glob(j(here,'**','*.py'),recursive=True):
    if f.endswith('configure_paths.py'): continue
    s=open(f).read(); t=s
    for old,new in MAP: t=t.replace(old,new.replace('\\','/'))
    if t!=s: open(f,'w').write(t); n+=1
os.makedirs(j(a.out,'tripod'),exist_ok=True); print(f'updated {n} scripts')
