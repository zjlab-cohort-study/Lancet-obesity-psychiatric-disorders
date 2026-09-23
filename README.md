# Analysis code — Preclinical/clinical obesity and psychiatric risk

R and Python scripts for the analyses reported in:

> **Preclinical and clinical obesity are associated with graded psychiatric risk and distinct brain and circulating biological signatures**
>
> Prospective UK Biobank study (N = 287,902) applying the Lancet Commission obesity
> staging (no obesity / preclinical / clinical) to 13 incident psychiatric disorders,
> integrating obesity-related illness burden, circulating biomarkers, brain MRI and
> risk-prediction modelling.

## Scripts

| Script | Analysis |
|---|---|
| `1-1.Cox_regression_analysis.R` | Cox regression of obesity stage on incident psychiatric disorders; three sequential covariate models; forest plot |
| `1-2.Stratified_analysis.R` | Subgroup analyses by covariate strata with multiplicative interaction tests |
| `2.Dose_response_analysis.R` | Joint associations of excess body fat × obesity-related illness burden (0 / 1 / ≥2); category HRs, P for trend, nonlinearity by LRT |
| `3.trajectory_illness_score.R` | Pseudo-trajectory analysis: standardized illness burden in the years before each diagnosis (LOESS curves, heatmaps, unsupervised curve clustering) |
| `4-1.Linear_association_analysis.R` | Screening of circulating biomarkers associated with obesity stage (linear regression, BH/Bonferroni correction) |
| `4-2.Mediation_analysis.R` | Single-mediator bootstrap mediation (B = 2000) for anxiety, depression and SUD |
| `5.SEM_obesity_psychiatric.R` | Parallel structural equation models with latent brain-structure and blood-biochemistry factors (WLSMV) |
| `common.py` | Shared configuration and helpers for the prediction pipeline |
| `6-1.prepare_data.py` | 80/20 train-test split (stratified), MICE-style imputation fitted on train only, design matrices |
| `6-2.fit_models.py` | Cox PH, elastic-net Cox, random survival forest and XGBoost for three nested predictor sets (covariates / + obesity / + biomarkers) |
| `6-3.metrics.py` | Harrell/Uno C, time-dependent AUC, Brier and integrated Brier scores, calibration (O/E, slope, ICI), NRI/IDI, decision curves; paired bootstrap CIs |
| `6-4.figures_report.py` | ROC, calibration, DCA, KM (risk tertiles) and TreeSHAP figures; markdown report |

## Requirements

- **R** (≥ 4.2): `survival`, `dplyr`, `readr`, `broom`, `tibble`, `ggplot2`, `data.table`, `gridExtra`, `lavaan`, `lavaanPlot` (optionally `DiagrammeRsvg` + `rsvg` for PDF export of SEM diagrams)
- **Python** (≥ 3.9): `numpy`, `pandas`, `scikit-learn`, `scikit-survival`, `lifelines`, `xgboost`, `shap`, `matplotlib`, `pyarrow`

## Data

The analyses use the UK Biobank resource, which is available on application at
<https://www.ukbiobank.ac.uk>; **no data are included in this repository**. Each script
reads one analysis-ready CSV: set the input path and variable names in the configuration
block at the top of the script. Expected columns include a participant id, obesity stage
(0 = none, 1 = preclinical, 2 = clinical), sociodemographic and lifestyle covariates
(age, sex, race, TDI, education, employment, income, smoking, alcohol, physical activity,
sleep duration), per-disorder event and follow-up-time columns, `excess_body_fat`,
`illness_score`, circulating biomarkers and (for `5.` and `6-*`) MRI indicators.

## Usage

1. Adjust the configuration block at the top of each script.
2. R scripts (`1` → `5`) run standalone in numbered order.
3. Prediction pipeline, run from this directory:

   ```bash
   python 6-1.prepare_data.py
   python 6-2.fit_models.py <outcome>    # once per outcome, e.g. SUD
   python 6-3.metrics.py <outcome>
   python 6-4.figures_report.py
   ```

   All pipeline artifacts (models, metrics, figures, `REPORT.md`) are written to
   `output_dir/`. `6-2` supports `--smoke N` (quick test on a subsample) and `--reuse`
   (resume from per-model checkpoints).

## Notes

- Mediation and SEM analyses are exploratory; the prediction models are internally
  validated only and require external validation before any clinical application.
- Please cite the paper above when using this code. UK Biobank data are governed by its
  access terms; users of these scripts are responsible for complying with them.
