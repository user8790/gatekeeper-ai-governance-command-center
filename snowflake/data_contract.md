# Gatekeeper Snowflake Data Contract

This contract defines the row shape expected by the reusable Python metric and governance engine.
The local prototype uses synthetic CSVs; a Snowflake deployment should expose an equivalent view.

## Grain

One row equals one model prediction paired to one evaluation outcome for one model version, one
evaluation snapshot, and one label window. Rows are retrospective governance rows, not patient-level
clinical workflow messages.

## Required Fields

- `y_true`: binary observed outcome after the label window. Use `1` for observed DKA/target event and `0` otherwise.
- `y_score`: model probability or risk score in `[0, 1]`.
- `y_pred`: binary model classification generated from the approved decision threshold.

## Strongly Recommended Fields

- Identifiers: `record_id`, `patient_id` or privacy-preserving `patient_key`, `encounter_id`.
- Windows: `dataset_period` with values `Reference` and `Current`, `deployment_phase`, `service_month`, `data_snapshot_date`.
- Model metadata: `model_id`, `model_version`, `prediction_datetime`, `decision_threshold`.
- Governance metadata: `run_id`, `governance_policy_version`, `schema_version`.
- Data quality: `missingness_rate_row`, `is_imputed_any`, `measurement_method_glucose`, `data_source_system`.

## Subgroup Dimensions

Supported dimensions are `age_band`, `sex_at_birth`, `health_zone`, `rural_urban`,
`socioeconomic_quintile`, `race_ethnicity_group`, `indigenous_identity`, `primary_language`,
`newcomer_status`, `cgm_use`, `measurement_method_glucose`, `data_source_system`, `diabetes_type`,
`care_setting`, and `intersectional_group_id`.

Use `intersectional_group_id` only when aggregation is safe. The prototype suppresses subgroup
interpretation when `n < 50` and marks caution when `50 <= n <= 100`.

## Prediction and Outcome Pairing

Predictions and outcomes should be joined by privacy-preserving patient key, encounter key where
available, outcome name, and label window. The view should include only rows with enough follow-up
time to determine `y_true`.

## Reference and Current Windows

- `Reference`: historical or baseline window used for drift comparison.
- `Current`: post-deployment or latest monitoring window.
- Audited-window Tier 1 metrics can use all outcome-paired rows in the snapshot.
- Drift metrics compare `Reference` against `Current`.

## Metric Definitions

- `TP`: `y_true = 1 and y_pred = 1`.
- `FP`: `y_true = 0 and y_pred = 1`.
- `TN`: `y_true = 0 and y_pred = 0`.
- `FN`: `y_true = 1 and y_pred = 0`.
- `Recall`: `TP / (TP + FN)`.
- `FNR`: `FN / (TP + FN)`.
- `PPV`: `TP / (TP + FP)`.
- `ECE`: weighted average absolute difference between mean predicted score and observed outcome rate across score bins.
- `PSI`: population stability index comparing reference and current numeric feature distributions.

## Governance Thresholds

- FNR disparity gap: `<= 8` percentage points.
- Recall: `>= 85%`.
- PPV disparity gap: `<= 12` percentage points.
- ECE: `< 0.10`.
- Any reliable subgroup ECE `> 0.15`: `FAIL`.
- PSI `> 0.20`: `NEEDS REVIEW`.
- FNR drift `+5` percentage points: `NEEDS REVIEW` when the count shift is reliable.
- Missing-data drift `+10` percentage points: `NEEDS REVIEW`.
- Calibration drift `+0.05` ECE: `NEEDS REVIEW`.
- One Tier 1 breach under the 20% exceedance rule: `NEEDS REVIEW`.
- One Tier 1 breach at or above 20% exceedance, two simultaneous breaches, or hard subgroup ECE stop: `FAIL`.

## Privacy and RBAC Assumptions

The dashboard should run on aggregated governance views. Direct identifiers should be excluded from
the Streamlit role unless explicitly approved. Sensitive attributes may be used for fairness
monitoring under controlled RBAC, audit logging, and minimum cell-size suppression.

No real patient data, AHS data, or Snowflake credentials are included in this repository.
