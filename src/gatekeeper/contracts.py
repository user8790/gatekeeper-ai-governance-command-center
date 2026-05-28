from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from gatekeeper.models import ValidationResult

REQUIRED_FIELDS = ["y_true", "y_score", "y_pred"]

RECOMMENDED_FIELDS = [
    "record_id",
    "patient_id",
    "encounter_id",
    "dataset_period",
    "deployment_phase",
    "service_month",
    "data_snapshot_date",
    "model_id",
    "model_version",
    "prediction_datetime",
    "decision_threshold",
    "missingness_rate_row",
    "is_imputed_any",
    "governance_policy_version",
]

SUBGROUP_DIMENSIONS = [
    "age_band",
    "sex_at_birth",
    "health_zone",
    "rural_urban",
    "socioeconomic_quintile",
    "race_ethnicity_group",
    "indigenous_identity",
    "primary_language",
    "newcomer_status",
    "cgm_use",
    "measurement_method_glucose",
    "data_source_system",
    "diabetes_type",
    "care_setting",
    "intersectional_group_id",
]

DEFAULT_GOVERNANCE_DIMENSIONS = [
    "age_band",
    "sex_at_birth",
    "health_zone",
    "rural_urban",
    "socioeconomic_quintile",
    "race_ethnicity_group",
    "indigenous_identity",
    "primary_language",
    "newcomer_status",
    "cgm_use",
    "measurement_method_glucose",
    "data_source_system",
    "diabetes_type",
    "care_setting",
]

NUMERIC_FEATURES = [
    "a1c_last_value",
    "mean_glucose_14d",
    "glucose_variability_14d",
    "time_in_range_14d",
    "age_years",
    "years_since_diagnosis",
    "distance_to_clinic_km",
    "wait_time_days_to_endocrinology",
    "missed_appointments_12mo",
    "prior_ed_visits_12mo",
    "prior_hospitalizations_12mo",
    "prior_dka_events_24mo",
    "medication_adherence_proxy",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
    "heart_rate",
    "creatinine",
    "egfr",
    "ldl_cholesterol",
    "triglycerides",
]

CORE_DATA_QUALITY_FIELDS = [
    "a1c_last_value",
    "mean_glucose_14d",
    "glucose_variability_14d",
    "time_in_range_14d",
    "measurement_method_glucose",
    "data_source_system",
]


def present_columns(columns: Iterable[str], desired: Iterable[str]) -> list[str]:
    available = set(columns)
    return [column for column in desired if column in available]


def validate_contract(df: pd.DataFrame) -> ValidationResult:
    missing_required = [field for field in REQUIRED_FIELDS if field not in df.columns]
    missing_recommended = [field for field in RECOMMENDED_FIELDS if field not in df.columns]
    warnings: list[str] = []

    if "dataset_period" in df.columns:
        periods = set(df["dataset_period"].dropna().astype(str).str.lower())
        if not {"reference", "current"}.issubset(periods):
            warnings.append(
                "dataset_period should contain Reference and Current windows for drift checks."
            )
    else:
        warnings.append("Drift checks will be limited because dataset_period is missing.")

    if "patient_id" in df.columns and "prediction_datetime" in df.columns:
        duplicate_predictions = df.duplicated(["patient_id", "prediction_datetime"]).sum()
        if duplicate_predictions:
            warnings.append(f"{duplicate_predictions} duplicate patient/time prediction rows found.")

    return ValidationResult(
        is_valid=not missing_required,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        row_count=len(df),
        column_count=len(df.columns),
        warnings=warnings,
    )
