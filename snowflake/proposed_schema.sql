-- Gatekeeper AI Governance Command Center
-- Proposed Snowflake schema for a future approved Snowflake Streamlit deployment.
-- This is a design asset only. Do not add credentials or live connection details.

create schema if not exists AI_GOVERNANCE;

create table if not exists AI_GOVERNANCE.MODEL_REGISTRY (
    model_id varchar not null,
    model_version varchar not null,
    model_name varchar,
    intended_use varchar,
    clinical_risk_level varchar,
    owner_team varchar,
    deployment_start_date date,
    deployment_end_date date,
    active_flag boolean default true,
    created_at timestamp_ntz default current_timestamp(),
    primary key (model_id, model_version)
);

create table if not exists AI_GOVERNANCE.EVALUATION_SNAPSHOTS (
    snapshot_id varchar not null,
    run_id varchar not null,
    model_id varchar not null,
    model_version varchar not null,
    data_snapshot_date date not null,
    dataset_period varchar not null,
    deployment_phase varchar not null,
    reference_window_start date,
    reference_window_end date,
    current_window_start date,
    current_window_end date,
    schema_version varchar,
    governance_policy_version varchar,
    created_at timestamp_ntz default current_timestamp(),
    primary key (snapshot_id)
);

create table if not exists AI_GOVERNANCE.PREDICTIONS (
    prediction_id varchar not null,
    snapshot_id varchar not null,
    record_id varchar,
    patient_key varchar not null,
    encounter_key varchar,
    facility_id varchar,
    postal_fsa varchar,
    prediction_datetime timestamp_ntz not null,
    model_id varchar not null,
    model_version varchar not null,
    y_score float not null,
    decision_threshold float not null,
    y_pred integer not null,
    outcome_name varchar not null,
    label_window_days integer,
    created_at timestamp_ntz default current_timestamp(),
    primary key (prediction_id)
);

create table if not exists AI_GOVERNANCE.OUTCOMES (
    outcome_id varchar not null,
    patient_key varchar not null,
    encounter_key varchar,
    outcome_name varchar not null,
    outcome_datetime timestamp_ntz,
    y_true integer not null,
    label_definition_version varchar,
    created_at timestamp_ntz default current_timestamp(),
    primary key (outcome_id)
);

create table if not exists AI_GOVERNANCE.EVALUATION_FEATURES (
    prediction_id varchar not null,
    age_years number(5,2),
    age_band varchar,
    sex_at_birth varchar,
    health_zone varchar,
    rural_urban varchar,
    socioeconomic_quintile varchar,
    race_ethnicity_group varchar,
    indigenous_identity varchar,
    primary_language varchar,
    newcomer_status varchar,
    disability_flag varchar,
    diabetes_type varchar,
    care_setting varchar,
    intersectional_group_id varchar,
    cgm_use varchar,
    measurement_method_glucose varchar,
    data_source_system varchar,
    insulin_regimen varchar,
    a1c_last_value float,
    mean_glucose_14d float,
    glucose_variability_14d float,
    time_in_range_14d float,
    age_at_diagnosis float,
    years_since_diagnosis float,
    obesity_flag varchar,
    distance_to_clinic_km float,
    telehealth_available varchar,
    telehealth_used varchar,
    wait_time_days_to_endocrinology integer,
    has_primary_care_provider varchar,
    diabetes_education_completed varchar,
    social_work_support varchar,
    missed_appointments_12mo integer,
    prior_ed_visits_12mo integer,
    prior_hospitalizations_12mo integer,
    prior_dka_events_24mo integer,
    medication_adherence_proxy float,
    bmi float,
    bp_systolic float,
    bp_diastolic float,
    heart_rate float,
    creatinine float,
    egfr float,
    ldl_cholesterol float,
    triglycerides float,
    missingness_rate_row float,
    is_imputed_any varchar,
    primary key (prediction_id)
);

create table if not exists AI_GOVERNANCE.GOVERNANCE_POLICIES (
    governance_policy_version varchar not null,
    effective_start_date date not null,
    effective_end_date date,
    fnr_disparity_gap float not null default 0.08,
    min_recall float not null default 0.85,
    ppv_disparity_gap float not null default 0.12,
    ece_threshold float not null default 0.10,
    subgroup_ece_fail float not null default 0.15,
    subgroup_suppress_n integer not null default 50,
    subgroup_caution_n integer not null default 100,
    psi_review float not null default 0.20,
    fnr_drift_review float not null default 0.05,
    missingness_drift_review float not null default 0.10,
    ece_drift_review float not null default 0.05,
    approved_by varchar,
    approval_notes varchar,
    created_at timestamp_ntz default current_timestamp(),
    primary key (governance_policy_version)
);

create table if not exists AI_GOVERNANCE.GOVERNANCE_DECISIONS (
    decision_id varchar not null,
    snapshot_id varchar not null,
    model_id varchar not null,
    model_version varchar not null,
    governance_policy_version varchar not null,
    final_decision varchar not null,
    decision_reason varchar,
    decision_timestamp timestamp_ntz default current_timestamp(),
    reviewer_group varchar,
    human_override_flag boolean default false,
    human_override_reason varchar,
    report_json variant,
    primary key (decision_id)
);

create table if not exists AI_GOVERNANCE.DECISION_EVIDENCE (
    decision_id varchar not null,
    rule_id varchar not null,
    rule_name varchar not null,
    tier varchar not null,
    owner varchar not null,
    status varchar not null,
    metric_value float,
    threshold_text varchar,
    summary varchar,
    detail varchar,
    decision_weight boolean,
    hard_fail boolean,
    exceedance_ratio float,
    created_at timestamp_ntz default current_timestamp()
);
