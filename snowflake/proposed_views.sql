-- Proposed views that produce the same row contract used by the local Streamlit prototype.

create or replace view AI_GOVERNANCE.V_PAIRED_PREDICTIONS_OUTCOMES as
select
    p.prediction_id,
    p.snapshot_id,
    p.record_id,
    p.patient_key,
    p.encounter_key,
    p.facility_id,
    p.postal_fsa,
    p.prediction_datetime,
    p.model_id,
    p.model_version,
    p.y_score,
    p.decision_threshold,
    p.y_pred,
    p.outcome_name,
    p.label_window_days,
    o.y_true,
    o.outcome_datetime,
    o.label_definition_version
from AI_GOVERNANCE.PREDICTIONS p
left join AI_GOVERNANCE.OUTCOMES o
    on p.patient_key = o.patient_key
    and coalesce(p.encounter_key, '') = coalesce(o.encounter_key, '')
    and p.outcome_name = o.outcome_name
    and (
        o.outcome_datetime is null
        or datediff(day, p.prediction_datetime, o.outcome_datetime) between 0 and p.label_window_days
    );

create or replace view AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS as
select
    p.record_id,
    p.patient_key as patient_id,
    p.encounter_key as encounter_id,
    p.facility_id,
    p.postal_fsa,
    p.prediction_datetime as index_datetime,
    to_char(p.prediction_datetime, 'Mon-YY') as service_month,
    s.dataset_period,
    s.deployment_phase,
    s.data_snapshot_date,
    s.schema_version,
    f.care_setting,
    p.outcome_name,
    p.label_window_days,
    p.y_true,
    p.outcome_datetime,
    p.label_definition_version,
    p.model_id,
    p.model_version,
    p.prediction_datetime,
    p.y_score,
    p.decision_threshold,
    p.y_pred,
    f.insulin_regimen,
    f.cgm_use,
    f.a1c_last_value,
    f.mean_glucose_14d,
    f.glucose_variability_14d,
    f.time_in_range_14d,
    f.age_years,
    f.age_band,
    f.sex_at_birth,
    f.health_zone,
    f.rural_urban,
    f.socioeconomic_quintile,
    f.race_ethnicity_group,
    f.indigenous_identity,
    f.primary_language,
    f.newcomer_status,
    f.disability_flag,
    f.diabetes_type,
    f.age_at_diagnosis,
    f.years_since_diagnosis,
    f.obesity_flag,
    f.distance_to_clinic_km,
    f.telehealth_available,
    f.telehealth_used,
    f.wait_time_days_to_endocrinology,
    f.has_primary_care_provider,
    f.diabetes_education_completed,
    f.social_work_support,
    f.missed_appointments_12mo,
    f.prior_ed_visits_12mo,
    f.prior_hospitalizations_12mo,
    f.prior_dka_events_24mo,
    f.medication_adherence_proxy,
    f.bmi,
    f.bp_systolic,
    f.bp_diastolic,
    f.heart_rate,
    f.creatinine,
    f.egfr,
    f.ldl_cholesterol,
    f.triglycerides,
    s.run_id,
    s.created_at as run_timestamp,
    s.governance_policy_version,
    f.missingness_rate_row,
    f.is_imputed_any,
    f.measurement_method_glucose,
    f.data_source_system,
    f.intersectional_group_id
from AI_GOVERNANCE.V_PAIRED_PREDICTIONS_OUTCOMES p
join AI_GOVERNANCE.EVALUATION_SNAPSHOTS s
    on p.snapshot_id = s.snapshot_id
left join AI_GOVERNANCE.EVALUATION_FEATURES f
    on p.prediction_id = f.prediction_id
where p.y_true is not null
  and p.y_pred in (0, 1)
  and p.y_score between 0 and 1;

create or replace view AI_GOVERNANCE.V_GATEKEEPER_CURRENT_WINDOW as
select *
from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS
where lower(dataset_period) = 'current';

create or replace view AI_GOVERNANCE.V_GATEKEEPER_REFERENCE_WINDOW as
select *
from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS
where lower(dataset_period) = 'reference';

create or replace view AI_GOVERNANCE.V_GATEKEEPER_AUDIT_PAYLOAD as
select
    d.decision_id,
    d.snapshot_id,
    d.model_id,
    d.model_version,
    d.governance_policy_version,
    d.final_decision,
    d.decision_reason,
    d.decision_timestamp,
    array_agg(
        object_construct(
            'rule_id', e.rule_id,
            'rule_name', e.rule_name,
            'tier', e.tier,
            'owner', e.owner,
            'status', e.status,
            'metric_value', e.metric_value,
            'threshold', e.threshold_text,
            'summary', e.summary,
            'detail', e.detail,
            'decision_weight', e.decision_weight,
            'hard_fail', e.hard_fail,
            'exceedance_ratio', e.exceedance_ratio
        )
    ) as evidence
from AI_GOVERNANCE.GOVERNANCE_DECISIONS d
left join AI_GOVERNANCE.DECISION_EVIDENCE e
    on d.decision_id = e.decision_id
group by
    d.decision_id,
    d.snapshot_id,
    d.model_id,
    d.model_version,
    d.governance_policy_version,
    d.final_decision,
    d.decision_reason,
    d.decision_timestamp;
