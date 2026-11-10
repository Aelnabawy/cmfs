# cmfs — Causal Mediation Feature Selection

Selects diagnostic biomarkers by their **route-specific mediated effect** on a functional outcome, inside a
**tiered causal graph**, assigns each selected biomarker an **evidence-graded causal role**, and uses
**cross-fitted transformations** so learned mediators cannot inflate causal estimates.

## Causal structure (two mediator tiers)
    confounders (+ environment / study phase)  ->  exposure  ->  upstream tier  ->  downstream tier  ->  outcome
                                                                 upstream tier  ------------------>  outcome (direct)
Optional `strata` (e.g. lesion hemisphere) run selection separately per exposed stratum against all controls.

## Modules
| module | stage |
|---|---|
| `spec.py` | `CausalSpec`: exposure, outcome, confounders, environment, ordered tiers, strata, measurement families |
| `data.py` | curation (missingness, constant columns); `EnvImputer` (KNN imputation guided by the environment) |
| `transform.py` | tier indices: `pca` (label-free), `lda` (exposure-supervised), `pls` (outcome-supervised); `crossfit_index`; `shuffle_check` (built-in effect under shuffled labels) |
| `scoring.py` | route-specific scores: upstream features (direct, via downstream, total); last tier (a x b, upstream PCs adjusted); family-aware selection |
| `roles.py` | lesion-side natural experiment, invariance across environments, collider-based orientation (bootstrapped PC), role assignment |
| `pipeline.py` | `CMFS.fit_full()` (full-data scores, panels, roles) and `CMFS.evaluate()` (nested CV: diagnostic AUC per stratum, union vs pooled vs association, honest out-of-fold mediated effect, stability, E-value) |
| `report.py` | Excel report |

## Minimal use
```python
from cmfs import CausalSpec, CMFS
spec = CausalSpec(exposure="stroke", outcome="speed", confounders=["age","male","bmi","pack_years","htn"],
                  environment="enrolment", tiers={"blood": blood_cols, "neuro": neuro_cols}, strata="side", family=family_fn)
model = CMFS(spec, k={"blood": 5, "neuro": 3}, transform="pca")      # "lda" / "pls" are cross-fitted automatically
full = model.fit_full(df)          # scores, panels, roles
nested = model.evaluate(df)        # honest evaluation
```
Supervised transformations (`lda`, `pls`) are always cross-fitted when used for causal scoring; report `shuffle_check`.

## Roles
mediator (exposure -> feature and feature -> outcome) · diagnostic marker (exposure -> feature only) ·
possible consequence of outcome (orientation leans outcome -> feature) · environment-sensitive (paths differ by study phase) ·
evidence tags: lesion-confirmed, oriented toward outcome, invariant across environments.

## Reproducing the CVES analysis
`cves_pipe.py` and `cves_prep.py` load the CVES release (set `UP` in `cves_pipe.py` to the folder holding `subjects.csv`
and the data dictionary). Then: `python run_cves.py full`, `python run_cves.py nested`, `python make_report.py`.
