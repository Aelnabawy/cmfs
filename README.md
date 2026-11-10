# CMFS: Causal Mediation Feature Selection (v0.2.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<!-- After the first Zenodo release, add the DOI badge here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Code for *"CMFS: Causal Mediation Biomarker Discovery Based on Tiered Clinical Selection for Hemisphere-Specific Stroke Classification"*.
Every result in the paper can be regenerated from the scripts below.

## Contents

| Folder | What it holds |
|---|---|
| `cmfs/` | The CMFS library (tier specification, mediated-effect scoring, cross-fitted transformations, role grading, reporting) and the command-line runner `run_cves.py` |
| `analysis/` | Scripts behind every reported analysis (mapped to paper sections below) |
| `figures/` | Scripts that draw the figures |
| `simulation/` | The simulation study and its summary |
| `configure_paths.py` | Points all scripts at your own data and output folders |

## Installation

Tested with Python 3.12.3. Create a virtual environment and install the pinned versions:

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Data

No data are included in this repository. Download each dataset from its original source:

| Dataset | Source |
|---|---|
| CVES (Cerebral Vasoregulation in Elderly with Stroke) | PhysioNet, DOI 10.13026/C2DW96 — https://physionet.org/content/cves/1.0.0/ |
| NHANES 1999–2002 and 2011–2014 | US Centers for Disease Control and Prevention — https://wwwn.cdc.gov/nchs/nhanes/ |
| Gait in neurodegenerative disease | PhysioNet — https://physionet.org/content/gaitndd/1.0.0/ |
| Gait in Parkinson's disease | PhysioNet — https://physionet.org/content/gaitpdb/1.0.0/ |
| Subacute stroke gait (Skvortsov, 2024) | Mendeley Data, DOI 10.17632/szmfkhyjpd.1 |
| GE-79, GE-75, GE-71 diabetes cohorts | PhysioNet: "Cerebromicrovascular Disease in Elderly with Diabetes", "Cerebral Perfusion and Cognitive Decline in Type 2 Diabetes", "Cerebral Vasoregulation in Diabetes" |

From CVES you need `subjects.csv`, plus `Gait_Steps_Features.csv` and the per-step files (`Steps/`) derived from the walking recordings. Place them in your data folder. Participants shared between CVES and the diabetes cohorts are removed by the scripts before the external analyses.

Then point every script at your folders (run once after downloading):

    python configure_paths.py --data /path/to/cves --external /path/to/external --out /path/to/results

## Reproducing the paper

Run from inside `analysis/` (scripts load the data preparation in `prep89.py`, `pipe.py` and `gait_four.py` themselves). Random seeds are fixed in every script.

| Script | Paper section | Output |
|---|---|---|
| `clean_gait_steps.py` | 4.1 | Robust per-foot gait features with quality flags |
| `prep89.py`, `pipe.py`, `gait_four.py`, `four_designs.py` | 3.1–3.2, 4.1 | Curated candidates, confounders, side-specific tiers |
| `multi_tier.py` | 5.1 | Single-, double- and triple-tier designs (Table S1) |
| `causal_pvalues.py` | 5.2 | Single-feature mediated effects with 95% intervals |
| `split_confirm_std.py` | 5.2 | Split-half replication (Table S7) |
| `tier_mediation.py` | 5.3 | Mediation through the transformed tiers (Table S6) |
| `comparison.py`, `unbiased_internal.py` | 5.4 | Internal method comparison (Table S3) |
| `robustness.py` | 5.4 | Robustness benchmark with planted decoys (Figure 5, Table S2) |
| `ext_compare.py`, `unbiased_external.py`, `reliability2.py` | 5.5 | External validation and feature reliability (Table S4) |
| `neighbours.py` | 5.6 | Hemisphere-specific approaches and lateralisation claims |
| `clinical.py` | 5.7 | Accuracy at threshold, likelihood ratios, calibration, decision curves (Table S8) |
| `tripod_data.py`, `tripod_model.py` | 4.1, Table S9 | Participant flow and characteristics, missing data, fairness, full model specification |
| `../simulation/sim_study.py <scenario> 200` | 5.8 | One simulation scenario (`n16_40`, `n50_50`, `n150_150`, `weak_50`, `strongU_50`, `order_50`) |
| `../simulation/summarise_simulation.py` | 5.8 | Simulation tables (Table S10) and Figure 8 |

Figures: `figures/fig_tier_method.py` (methodology), `fig_results_lr.py` (left and right results), `fig_reliability.py` (Figure 6), `fig_clinical.py` (Figure 7), `fig_explainable.py` (Figure S1), `fig_skv.py` (Figure S3).

`causal_pvalues.py` also computes p-values; the paper reports intervals only.

## Running CMFS on new data

`cmfs/run_cves.py` shows the full pipeline on CVES; the library functions in `cmfs/` accept any tiered feature matrix with an exposure, an outcome and confounders.

## Citing

If you use CMFS, please cite the article and the software version you used (see `CITATION.cff`; GitHub shows a "Cite this repository" button):

    @article{elnabawy2026cmfs,
      author  = {El-Nabawy, Ala'a and Hu, Shanfeng and Alshabrawy, Ossama and Gon{\c{c}}alves, Teresa and Aslam, Nauman},
      title   = {{CMFS}: Causal Mediation Biomarker Discovery Based on Tiered Clinical Selection for Hemisphere-Specific Stroke Classification},
      year    = {2026},
      note    = {Preprint}
    }

    @software{elnabawy2026cmfs_code,
      author  = {El-Nabawy, Ala'a},
      title   = {{CMFS}: Causal Mediation Feature Selection},
      version = {0.2.0},
      year    = {2026},
      url     = {https://github.com/Aelnabawy/cmfs},
      doi     = {10.5281/zenodo.XXXXXXX}
    }

## Licence

Released under the MIT licence; see `LICENSE`.
