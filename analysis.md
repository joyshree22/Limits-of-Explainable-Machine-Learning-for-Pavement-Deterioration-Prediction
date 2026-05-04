# Repository Analysis and Fix Plan

## Executive Summary

The repository implements the broad article idea: separate design-stage prediction from monitoring-stage forecasting, evaluate climate transfer with section-wise splits and leave-one-region-out validation, then use SHAP/PDP only after checking model stability. However, the current outputs are not yet aligned with the article outline or the methodology. The latest `resultandd.docx` presents several post-hoc pipeline changes as improvements, but some of those changes weaken the paper's central argument and some reported metrics are not computed in the way the methodology requires.

The most serious problems are:

1. The distress models have direct target leakage from distress sub-indicator columns.
2. The persistence/monitoring common-set comparison does not actually compare trained monitoring model predictions on a single common set.
3. Section bootstrap confidence intervals are implemented incorrectly because duplicate sampled sections are collapsed.
4. `CLIMATE_ZONE_CODE` and climate-distance LOO weighting introduce region identity and withheld-region information into a study that is supposed to test cross-climate generalization.
5. Several results are overfit or over-interpreted, especially distress R2 near 0.999 and monitoring gains driven by lag persistence.
6. The generated result document mixes old and revised result files, producing inconsistent claims.

The safest path is to treat the current outputs as diagnostic, not publication-ready, then rerun a corrected, pre-registered pipeline that favors validity over higher R2.

## Intended Research Argument

From `Article.docx`, the paper should argue one focused point:

- Pavement ML often conflates operational monitoring accuracy with design-stage transferability.
- Monitoring models can look accurate because prior condition is highly predictive.
- Design models should be judged by cross-climate generalization, especially leave-one-region-out validation.
- SHAP is only trustworthy when the underlying model generalizes and cross-model feature rankings are stable.
- Limitations are part of the contribution: 48 sections, repeated measures, regional imbalance, and climate distribution shift.

From `Methodology.docx`, the expected empirical structure is:

- 48 GPS flexible sections across Arizona, Georgia, Ohio, and Ontario.
- IRI aggregated to 650 visit-level observations; rutting 566; distress 378.
- Section-wise train/validation/test split: 34/7/7 sections.
- Imputation, scaling, and collinearity reduction fitted on training sections only.
- Design task excludes prior condition.
- Monitoring task adds lagged prior condition.
- Persistence benchmarks must be used to interpret monitoring R2.
- Leave-one-region-out validation is the central generalization test.
- SHAP consistency threshold is pre-specified at rho > 0.75.

The current code partially follows this, but the revised results drift toward improving R2 instead of stress-testing the claim.

## Critical Issues

### 1. Distress Target Leakage

The distress target is `DIS_HPMS16_CRACKING_PERCENT_AC`, a composite cracking percentage. The methodology explicitly says the model predicts the composite, not its mechanism-specific subcomponents. But the current distress design datasets retain many `DIS_*` columns that are direct distress measurements or components of the same distress family.

Examples retained in `results/design_distress_train.parquet` include:

- `DIS_WP_LENGTH_CRACKED`
- `DIS_GATOR_CRACK_A_L`
- `DIS_BLOCKVERSE_CRACKING`
- `DIS_TRANS_CRACK_NO`
- `DIS_LONG_CRACK_WP_L`
- many other `DIS_*` cracking and patching fields

`DIS_WP_LENGTH_CRACKED` has Spearman correlation about 0.998 with the distress target in the training data. That explains the suspicious distress R2 values around 0.998 to 0.999. This is not model skill; it is target leakage.

Required fix:

- In `scripts/02_aggregate.py`, expand `NON_FEATURE_COLS` to exclude all distress survey fields when predicting distress, not only the final target and a few named alternatives.
- More generally, implement target-family exclusion rules:
  - IRI target: exclude all `IRI_*`
  - Rutting target: exclude all `RUT_*`
  - Distress target: exclude all `DIS_*`
- If some distress-adjacent field is genuinely available at design time, document it explicitly. Otherwise remove it.
- Rerun steps 02 through 17 and regenerate all distress metrics.
- Expect distress R2 to drop sharply. That is a correction, not a failure.

### 2. Common-Set Persistence Comparison Is Wrong

`scripts/09_benchmarks.py` says it computes a common evaluation set for monitoring vs persistence, but it does not use trained monitoring model predictions. For `monitoring_on_common_*`, it uses the lag column itself:

```python
y_mon_common = test_df[mon_lag_col].values[common_mask]
```

That is persistence, not the trained monitoring model. As a result, `results/common_set_comparison.csv` contains rows where "monitoring" and persistence are identical or not comparable.

There is also a subset mismatch:

- IRI design persistence k=1 uses 32 observations.
- IRI monitoring common row uses 25 observations.
- The result document compares these as if they were one common evaluation set.

Required fix:

- Move common-set model comparison out of `09_benchmarks.py` or load trained models after step 10.
- Define one observation index per target/horizon and evaluate every comparator on exactly that index:
  - persistence k=1
  - persistence k=2
  - persistence k=3
  - trained absolute monitoring model
  - delta-target monitoring model, if retained
- Store section keys, observation dates, actuals, predictions, and model labels in a long-format file such as `results/common_set_predictions_iri.csv`.
- Report skill only when model and benchmark have identical rows.

The quick check I ran for IRI monitoring on the 25-row 365-day common subset showed:

- XGBoost monitoring RMSE about 0.163 vs persistence RMSE about 0.208.
- RF monitoring RMSE about 0.208 vs persistence RMSE about 0.208.

This is different from the generated document and should be recomputed cleanly inside the pipeline.

### 3. Delta-IRI Skill Score Uses Inconsistent Denominators

`write_resultandd.py` computes delta-IRI skill using:

- delta model RMSE on 66 monitoring test observations
- persistence RMSE from `persist_k1_monitoring_iri`, which has 25 observations

That makes the reported `+0.260` skill score invalid. A model cannot claim improvement over persistence if the RMSEs are computed on different observations.

Required fix:

- Compute delta model predictions only on the same common-set rows used for persistence.
- Convert predicted delta back to absolute IRI using the lag value.
- Compare absolute-scale RMSE against persistence on the identical rows.
- Report delta-scale R2 separately, but do not mix it with absolute-scale persistence skill.

### 4. Bootstrap Confidence Intervals Collapse Duplicate Sections

`scripts/11_evaluate.py` tries section-level bootstrap resampling, but this line collapses duplicate sampled sections:

```python
mask = np.isin(section_keys, sampled)
```

If a section is sampled twice, it still appears once. This is not bootstrap resampling with replacement; it is closer to random subset sampling. It will distort confidence intervals, usually making them too narrow or otherwise unreliable.

Required fix:

- Build each bootstrap sample by concatenating all rows from each sampled section, preserving duplicate draws.
- Recompute R2 on that concatenated bootstrap dataframe.
- Apply the same fix to any bootstrap logic in benchmark or LOO utilities.

Correct pattern:

```python
sampled_sections = rng.choice(unique_sections, size=len(unique_sections), replace=True)
boot_idx = np.concatenate([
    np.where(section_keys == sec)[0] for sec in sampled_sections
])
r2_boot.append(r2_score(y_true[boot_idx], y_pred[boot_idx]))
```

### 5. `CLIMATE_ZONE_CODE` Undermines the Generalization Test

`scripts/08_tasks.py` injects an ordinal `CLIMATE_ZONE_CODE` into every design and monitoring dataset. This is described as a "regime context" fix, but it is also a region identity proxy.

Problems:

- The article argues for climate-feature transfer, not region-label memorization.
- The code is ordinal even though the regions are not a smooth one-dimensional physical scale. Georgia is wet-subtropical; it is not simply "between" Arizona and Ohio.
- In leave-one-region-out validation, the withheld region's code is still passed to the model at prediction time. That gives the model a synthetic indicator for a region absent from training.
- The latest result says positive Ontario LOO is "not leakage", but the methodology explicitly says non-negative Ontario LOO should trigger investigation.

Required fix:

- Remove `CLIMATE_ZONE_CODE` from the primary analysis.
- If regime indicators are explored, move them to a sensitivity analysis and label them as deployment-context features, not design-transfer features.
- Prefer physical climate variables: freeze index, freeze-thaw cycles, precipitation, humidity, temperature, wet-freeze composites.
- Keep latitude/longitude excluded as geographic proxies.

### 6. Climate-Distance Weighting Uses Withheld-Region Information

`scripts/12_loo.py` reweights training observations by distance to the withheld region's mean freeze index. This uses knowledge about the held-out target domain to change model training.

That may be acceptable for a domain-adaptation experiment where the target climate distribution is known, but it is not a clean leave-one-region-out generalization test. It shifts the question from "Does the model transfer?" to "Can we tune training weights using the target region's climate summary?"

Required fix:

- Remove climate-distance weighting from the primary LOO.
- Keep only inverse-region-frequency weighting if it is pre-specified and does not depend on the withheld region.
- If climate-distance weighting is kept, report it as a separate transductive/domain-adaptation sensitivity analysis.
- Do not use it to overturn pre-specified LOO failure expectations.

### 7. Post-Hoc Revisions Conflict With the Article

`resultandd.docx` opens by saying six structural faults were fixed and that the changes produced substantial R2 improvements. That framing is misaligned with `Article.docx`, which says the paper should be honest, not optimized for impressive accuracy.

Examples:

- The article expects cross-model SHAP consistency around rho = 0.594 and says only top ranks are stable.
- The latest results report rho = 0.732 and describe it as marginal, but still lean into broader interpretability.
- The methodology says Ontario LOO should be negative and positive Ontario should be treated as possible leakage.
- The latest results report positive Ontario XGBoost LOO as an improvement.
- The article says monitoring R2 should be interpreted through persistence.
- The latest results report large monitoring gains before the persistence comparison is correctly computed.

Required fix:

- Freeze the methodology first.
- Label every post-hoc change as exploratory.
- Make the primary result section answer the original research questions, even if the answer is negative.
- Do not rewrite consistency checks after seeing results.

### 8. Mixed Old and New Result Files

There are both old and revised result files:

- `results/design_model_metrics.csv` and `results/monitoring_model_metrics.csv` contain older values.
- `results/test_metrics.csv` contains revised values.
- `write_rnd.py` uses older hard-coded values.
- `write_resultandd.py` uses `test_metrics.csv` plus additional dynamic calculations.

This creates inconsistent documents. For example, design IRI XGBoost is:

- 0.0563 in `design_model_metrics.csv`
- 0.2192 in `test_metrics.csv`
- discussed differently across result documents

Required fix:

- Use one canonical metrics file per pipeline run.
- Add a `run_id` or timestamped `results/runs/<run_id>/` directory.
- Delete or archive stale metrics before regenerating a document.
- Make document generation read only canonical outputs from the same run.

### 9. Overfitting Signals in Current Models

A quick train/validation/test check shows large gaps:

- XGBoost design IRI: train R2 about 0.956, validation about 0.941, test about 0.219.
- RF design IRI: train about 0.956, validation about 0.961, test about 0.100.
- XGBoost design rutting: train about 0.980, validation about 0.970, test about 0.215.
- XGBoost monitoring IRI: train about 0.999, validation about 1.000, test about 0.780.
- Distress models are near-perfect everywhere because of target leakage.

The validation set is section-wise, but it still appears too similar to training or too small to reliably select hyperparameters. The final model is retrained on train+validation, so post-hoc train/validation metrics are not independent, but the gap to test still warns that hyperparameter selection is optimistic.

Required fix:

- Reduce model capacity:
  - XGBoost `max_depth` 2 to 4 for primary runs.
  - Use stronger `min_child_weight`, `gamma`, `reg_lambda`, and `reg_alpha`.
  - Lower `n_estimators` or use early stopping inside grouped CV.
  - RF `min_samples_leaf` at least 5 to 10, shallower max depth.
- Use nested or repeated grouped CV at the section level for model selection stability.
- Report region-wise test errors, not only pooled R2.
- Consider simpler primary models: Ridge/ElasticNet, monotonic GAM, or constrained gradient boosting.

## Methodology Alignment Problems by Script

### `scripts/02_aggregate.py`

What works:

- IRI visit aggregation is consistent with methodology.
- Section key uses region plus SHRP ID, which avoids non-unique SHRP IDs.

What needs fixing:

- Target-family leakage filtering is incomplete.
- Distress subcomponents remain predictors for the distress target.

### `scripts/04_missingness.py`

What works:

- Threshold is applied independently by target.

Concern:

- Applying missingness before splitting is acceptable only because it uses counts, but it still allows feature availability in validation/test to influence feature retention. The methodology permits this, but for a stricter paper, compute missingness threshold on training only after split.

Recommended:

- Keep as-is only if defended.
- Otherwise split first, then fit all preprocessing, including missingness filtering, on training only.

### `scripts/06_impute_scale.py`

What works:

- Group-median imputation and StandardScaler are fitted on training split only.

Concern:

- `AGE_YEARS` is in `META_COLS`, so it is excluded from feature scaling and then reintroduced as metadata. The methodology treats age as a design feature and AASHTO benchmark input. If age is intentionally excluded from ML features because composites include it, state that clearly. Otherwise remove `AGE_YEARS` from `META_COLS` for modeling.

### `scripts/07_collinearity.py`

What works:

- Spearman clustering is training-only.
- Latitude/longitude are excluded.
- `CLIM_FREEZE_INDEX` is force-retained.

Concern:

- `features_selected_*.txt` is saved before `CLIMATE_ZONE_CODE` is injected, so feature manifest and final design datasets disagree.
- Force-retaining `CLIM_FREEZE_INDEX` after clustering can reintroduce high collinearity with selected composites. That may be okay for interpretability, but should be documented as a physics-mandatory exception.

### `scripts/08_tasks.py`

What works:

- Lag feature is date-aware and section-local.
- Delta target is created cleanly as current minus lag.

What needs fixing:

- Remove `CLIMATE_ZONE_CODE` from the primary run.
- If delta targets are saved for every target, either train/evaluate them consistently or save only the intended delta-IRI variant.

### `scripts/09_benchmarks.py`

What needs fixing:

- Common-set comparison currently uses lag values as "monitoring" predictions.
- It creates duplicate common-set rows from design and monitoring tasks.
- It does not compare trained model predictions to persistence.
- It does not output observation identifiers, making audit difficult.

### `scripts/10_train.py`

What works:

- GroupKFold uses section groups.
- Regional inverse-frequency weighting is consistent with the sample imbalance concern.

What needs fixing:

- Model capacity is too high for 34 training sections.
- Some hyperparameter spaces encourage overfit, especially XGBoost depths up to 12.
- Regime-split rutting is post-hoc and should be sensitivity-only.
- `X_final` and `y_final` include rows with missing delta target filled as zero if they ever pass through; keep explicit target masks for final training too.

### `scripts/11_evaluate.py`

What needs fixing:

- Bootstrap resampling collapses duplicate sections.
- It does not evaluate delta models or regime-split models, even though the result document reports them.

### `scripts/12_loo.py`

What works:

- LOO refits imputation and scaling inside each iteration.
- Predictions are saved by region and architecture.

What needs fixing:

- Remove climate-distance weighting from primary LOO.
- Remove `CLIMATE_ZONE_CODE` from primary LOO features.
- Keep the warning that positive Ontario LOO indicates possible leakage or post-hoc contamination.
- Write combined `loo_<region>.csv` files consistently or remove stale combined files.

### `scripts/13_shap.py`, `14_pdp.py`, `15_residuals.py`

Main concern:

- Interpretability should be downstream of trustworthy generalization. If design IRI LOO is near zero or negative, SHAP/PDP should be framed as model behavior within this dataset, not as design guidance.

Specific fixes:

- Recompute SHAP after leakage and common-set fixes.
- Use the same canonical feature set as model training.
- Add SHAP stability by bootstrap over sections, not only XGB-vs-RF rank correlation.
- Restore residual-vs-freeze-index panel using raw unscaled FI merged by section/date if FI is not in the selected model matrix.

## Recommended Corrected Pipeline

### Phase 1: Lock the Primary Analysis

1. Create a configuration flag such as `PRIMARY_ANALYSIS = True`.
2. Primary run should exclude:
   - `CLIMATE_ZONE_CODE`
   - latitude/longitude
   - all target-family predictors
   - climate-distance LOO weighting
   - post-hoc regime routing
3. Keep:
   - section-wise split
   - training-only imputation/scaling/feature selection
   - region inverse-frequency sample weights, if pre-specified
   - force-retained physical FI only if documented
   - persistence benchmarks
   - LOO as the primary generalization test

### Phase 2: Remove Leakage

1. Define target-family exclusion rules in one shared config object.
2. Apply the rules before feature engineering or before thresholding.
3. Add an audit file:

```text
results/feature_audit_<target>.csv
feature,reason,status
DIS_WP_LENGTH_CRACKED,target_family_exclusion,excluded
CLIM_LONGITUDE,geographic_proxy,excluded
CLIM_FREEZE_INDEX,physics_mandatory,retained
```

4. Fail the pipeline if a selected feature starts with the target family prefix.

### Phase 3: Fix Evaluation

1. Repair section bootstrap duplicate sampling.
2. Build explicit common-set prediction files.
3. Compare monitoring and persistence only on identical rows.
4. Compute delta-IRI skill only on identical rows.
5. Add assertions:
   - all compared models have same `section_key` and `OBSERVATION_DATE`
   - no duplicate model labels in common-set table without task/horizon identifiers
   - R2 lies inside its corrected bootstrap interval

### Phase 4: Reduce Overfitting

Use a conservative model search space for the primary paper:

```python
XGBRegressor(
    max_depth=[2, 3, 4],
    learning_rate=[0.01, 0.03, 0.05],
    n_estimators=[50, 100, 200],
    min_child_weight=[5, 10, 20],
    subsample=[0.6, 0.8],
    colsample_bytree=[0.5, 0.7],
    reg_lambda=[1, 10, 100],
    reg_alpha=[0, 0.1, 1],
)
```

For Random Forest:

```python
RandomForestRegressor(
    max_depth=[3, 5, 7],
    min_samples_leaf=[5, 10, 15],
    max_features=["sqrt", 0.3, 0.5],
)
```

Then report:

- pooled test metrics
- per-region test metrics
- LOO metrics
- persistence skill
- section-bootstrap CIs
- train-vs-validation-vs-test gap table

### Phase 5: Separate Primary Findings From Sensitivity Analyses

Primary results:

- No region code.
- No transductive climate-distance weighting.
- No target-family leakage.
- Conservative models.
- LOO and persistence define the interpretation.

Sensitivity analyses:

- Add `CLIMATE_ZONE_CODE`.
- Add climate-distance/domain-adaptation weighting.
- Add warm/freeze rutting regime split.
- Add delta-IRI monitoring.
- Compare against primary run and label as exploratory.

This will preserve the article's honesty while still allowing useful experimental ideas.

## How the Results Section Should Change

The revised result narrative should not lead with "R2 improvements." It should lead with the research questions:

1. Do design models generalize across climate zones?
   - Use corrected LOO.
   - If R2 is near zero or negative, state that directly.

2. Does monitoring outperform persistence?
   - Use corrected common-set comparisons.
   - Report absolute monitoring and delta-IRI separately.

3. Are SHAP findings stable enough to interpret?
   - Report corrected cross-model rho.
   - Add section-bootstrap SHAP rank uncertainty if possible.
   - Restrict claims to stable top features.

4. What do rutting and distress reveal?
   - Rutting likely remains mechanism-mixed.
   - Distress must be rerun after leakage removal before any claim is made.

## Publication-Safe Claims After Fixes

Likely safe:

- Section-wise splits are necessary; row-wise splits would leak longitudinal section identity.
- Monitoring accuracy must be benchmarked against persistence.
- Design-stage cross-climate transfer is weak with 48 sections.
- SHAP explanations are not causal and are unstable beyond top-ranked features.
- Regional imbalance and limited climate coverage are the central limits.

Not safe yet:

- Distress models generalize with R2 near 0.999.
- Delta-IRI improves over persistence by 26%.
- Positive Ontario LOO proves improved climate conditioning.
- `CLIMATE_ZONE_CODE` is a physically meaningful design feature.
- Regime-split rutting is deployment-ready.

## Concrete Implementation Checklist

1. Add target-family exclusion config and enforce it before thresholding.
2. Remove `CLIMATE_ZONE_CODE` from primary `08_tasks.py`.
3. Add optional sensitivity flag for climate zone code.
4. Remove climate-distance weights from primary `12_loo.py`.
5. Fix section bootstrap in `11_evaluate.py`.
6. Rewrite common-set evaluation to use trained model predictions and identical row IDs.
7. Recompute delta-IRI skill on the same rows as persistence.
8. Reduce model hyperparameter ranges.
9. Add leakage assertions:
   - no `DIS_*` feature for distress target
   - no `RUT_*` feature for rutting target
   - no `IRI_*` feature for IRI target
   - no coordinates in selected features
10. Rerun the full pipeline from step 02 onward.
11. Regenerate `resultandd.docx` from one canonical metrics source.
12. Archive old outputs under `results/archive/` or timestamped run directories.

## Bottom Line

The repository is close to the intended study structure, but the latest results are not publication-safe. The distress results are contaminated by target leakage, the persistence comparison is not valid, the bootstrap CIs need repair, and the climate-zone/LOO revisions blur the line between generalization testing and post-hoc adaptation.

The fix is not to chase higher R2. The fix is to restore the paper's original spine: honest separation of monitoring and design tasks, strict leakage control, persistence-calibrated monitoring claims, and climate-transfer validation that is allowed to fail.
