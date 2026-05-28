-- Example Snowflake validation queries for Gatekeeper governance metrics.
-- These queries are intentionally read-only.

-- 1. Row contract smoke check.
select
    count(*) as row_count,
    count_if(y_true is null) as missing_y_true,
    count_if(y_pred is null) as missing_y_pred,
    count_if(y_score is null) as missing_y_score,
    min(data_snapshot_date) as min_snapshot_date,
    max(data_snapshot_date) as max_snapshot_date
from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS;

-- 2. Overall confusion matrix.
select
    count_if(y_true = 1 and y_pred = 1) as tp,
    count_if(y_true = 0 and y_pred = 1) as fp,
    count_if(y_true = 0 and y_pred = 0) as tn,
    count_if(y_true = 1 and y_pred = 0) as fn,
    count_if(y_true = 1 and y_pred = 1) / nullif(count_if(y_true = 1), 0) as recall,
    count_if(y_true = 1 and y_pred = 0) / nullif(count_if(y_true = 1), 0) as fnr,
    count_if(y_true = 1 and y_pred = 1) / nullif(count_if(y_pred = 1), 0) as ppv
from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS;

-- 3. Subgroup metrics for one dimension.
select
    health_zone,
    count(*) as n,
    count_if(y_true = 1 and y_pred = 1) as tp,
    count_if(y_true = 0 and y_pred = 1) as fp,
    count_if(y_true = 0 and y_pred = 0) as tn,
    count_if(y_true = 1 and y_pred = 0) as fn,
    count_if(y_true = 1 and y_pred = 1) / nullif(count_if(y_true = 1), 0) as recall,
    count_if(y_true = 1 and y_pred = 0) / nullif(count_if(y_true = 1), 0) as fnr,
    count_if(y_true = 1 and y_pred = 1) / nullif(count_if(y_pred = 1), 0) as ppv
from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS
group by health_zone
having count(*) >= 50
order by fnr desc;

-- 4. Expected Calibration Error by score decile.
with scored as (
    select
        width_bucket(y_score, 0, 1, 10) as score_bin,
        y_score,
        y_true
    from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS
),
bins as (
    select
        score_bin,
        count(*) as n,
        avg(y_score) as avg_predicted,
        avg(y_true) as observed_rate,
        abs(avg(y_score) - avg(y_true)) as bin_error
    from scored
    group by score_bin
)
select
    sum((n / sum(n) over ()) * bin_error) as expected_calibration_error
from bins;

-- 5. Reference/current FNR drift.
with window_metrics as (
    select
        dataset_period,
        count_if(y_true = 1 and y_pred = 0) / nullif(count_if(y_true = 1), 0) as fnr,
        count_if(y_true = 1 and y_pred = 0) as false_negatives
    from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS
    group by dataset_period
)
select
    cur.fnr - ref.fnr as fnr_drift,
    cur.false_negatives - ref.false_negatives as false_negative_delta
from window_metrics ref
join window_metrics cur
    on lower(ref.dataset_period) = 'reference'
   and lower(cur.dataset_period) = 'current';

-- 6. Data quality drift.
select
    avg(case when lower(dataset_period) = 'reference' then missingness_rate_row end) as reference_missingness,
    avg(case when lower(dataset_period) = 'current' then missingness_rate_row end) as current_missingness,
    current_missingness - reference_missingness as missingness_drift
from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS;
