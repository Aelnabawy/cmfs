"""Final CMFS run on the CVES cohort (curated features)."""
import sys, time, pickle; sys.path.insert(0, '.')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from cves_prep import d, S, Y, Xraw, names, domain, family, num
from cmfs import CausalSpec, CMFS
stage = sys.argv[1]
NEURO = ['TM: Part A Time (sec)', 'TM: Part B Time (sec)', 'VF: # Animals', 'VF: FAS Total', 'VF: # supermarket', 'HVLT: Total Recall',
         'HVLT: Delayed Recall', 'RCFT: Copy-Raw', 'RCFT: I.R.-Raw', 'RCFT: D.R.-Raw', 'RCFT: Total Correct', 'DS: Total',
         'Clock: Total Score', 'MMSE Total', 'GDS Total Response', 'BADS Total Responses']
BLOOD = [c for c in names if domain[c] == 'Blood' and not c.startswith(('p_', 'R2_')) and 'olumn 40' not in c]
sr = d['Stroke Side'].astype(str).str.upper().str.strip()
side = np.where(S == 1, np.where(sr.str.contains('BILAT'), '', np.where(sr.str.match(r'^(L|LEFT)'), 'L', np.where(sr.str.match(r'^(R|RIGHT)'), 'R', ''))), '')
df = pd.concat([pd.DataFrame({'stroke': S, 'speed': Y, 'age': num(d, 'age'), 'male': (d.gender.astype(str).str.upper().str[0] == 'M').astype(float),
                              'bmi': num(d, 'bmi'), 'pack_years': num(d, 'pack_years'),
                              'htn': (d.htn_patient_medical_history.astype(str).str.upper().str.strip() == 'YES').astype(float),
                              'enrolment': d.subject_number.str.extract(r'(\d+)')[0].astype(float), 'side': side}),
                Xraw[BLOOD + NEURO].reset_index(drop=True)], axis=1)
df = df[(df.stroke == 0) | df.side.isin(['L', 'R'])].reset_index(drop=True)
spec = CausalSpec(exposure='stroke', outcome='speed', confounders=['age', 'male', 'bmi', 'pack_years', 'htn'], environment='enrolment',
                  tiers={'blood': BLOOD, 'neuro': NEURO}, strata='side', family=family)
model = CMFS(spec, k={'blood': 5, 'neuro': 3}, alpha=0.1, transform='pca')
t0 = time.time()
if stage == 'full':
    print(f'cohort n={len(df)}: {int((df.side=="L").sum())} left, {int((df.side=="R").sum())} right strokes, {int((df.stroke==0).sum())} controls; blood {len(BLOOD)}, neuro {len(NEURO)} candidates')
    full = model.fit_full(df, orient_boot=100); pickle.dump({k: v for k, v in full.items() if k != 'arrays'}, open('full.pkl', 'wb'))
    for g, v in full['groups'].items(): print(f'{g}: total effect {v["total"]:+.3f} m/s; panel: {v["panel"]}')
    print(full['roles'][['feature', 'tier', 'p_exposure', 'p_outcome', 'role', 'evidence', 'selected_in']].round(3).to_string(index=False))
    if 'side_p' in full['roles']: print(full['roles'][['feature', 'L_vs_control', 'R_vs_control', 'side_p', 'adjacent', 'toward_outcome', 'from_outcome', 'invariance_p']].round(2).to_string(index=False))
elif stage == 'nested':
    nested = model.evaluate(df); pickle.dump(nested, open('nested.pkl', 'wb'))
    for k, v in nested['auc'].items(): print(f'   {k:42s} AUC {v[0]:.3f}')
    m = nested['mediated']; print(f'honest mediated effect (union panel): {m["estimate"]:+.3f} m/s [{m["lo"]:+.3f}, {m["hi"]:+.3f}] of {m["total"]:+.3f} ({100*m["share"]:.0f}%), E-value {m["evalue"]:.2f} (CI {m["evalue_ci"]:.2f})')
    for g, c in nested['frequency'].items(): print(f'{g} (stability {nested["stability"][g]:.2f}):', ', '.join(f'{x} {v/50:.0%}' for x, v in c.most_common(8)))
print(f'[{time.time()-t0:.0f}s]')
