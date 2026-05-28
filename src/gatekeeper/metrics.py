from __future__ import annotations

from math import isnan

import numpy as np
import pandas as pd

from gatekeeper.contracts import (
    CORE_DATA_QUALITY_FIELDS,
    DEFAULT_GOVERNANCE_DIMENSIONS,
    NUMERIC_FEATURES,
    present_columns,
    validate_contract,
)
from gatekeeper.models import (
    CalibrationMetrics,
    ClassificationMetrics,
    EvaluationResult,
    EvidenceStatus,
    GovernanceDecision,
    GovernanceThresholds,
    RuleEvidence,
)


def evaluate_model_governance(
    df: pd.DataFrame,
    scenario_name: str = "Synthetic scenario",
    source: str = "synthetic",
    thresholds: GovernanceThresholds | None = None,
    subgroup_dimensions: list[str] | None = None,
    feature_columns: list[str] | None = None,
    policy_version: str = "policy-0.1",
) -> EvaluationResult:
    thresholds = thresholds or GovernanceThresholds()
    validation = validate_contract(df)
    if not validation.is_valid:
        raise ValueError(f"Dataset missing required fields: {validation.missing_required}")

    audited_rows = _valid_prediction_rows(df)
    reference_rows, current_rows = _split_reference_current(audited_rows)
    subgroup_dimensions = subgroup_dimensions or present_columns(
        audited_rows.columns, DEFAULT_GOVERNANCE_DIMENSIONS
    )
    feature_columns = feature_columns or present_columns(audited_rows.columns, NUMERIC_FEATURES)

    overall = classification_metrics(audited_rows)
    reference = classification_metrics(reference_rows) if len(reference_rows) else None
    current = classification_metrics(current_rows) if len(current_rows) else None

    calibration = expected_calibration_error(audited_rows)
    reference_calibration = (
        expected_calibration_error(reference_rows) if len(reference_rows) else None
    )
    current_calibration = expected_calibration_error(current_rows) if len(current_rows) else None

    subgroup_metrics = compute_subgroup_metrics(
        audited_rows, subgroup_dimensions, thresholds=thresholds
    )
    dimension_summary = summarize_dimensions(subgroup_metrics, thresholds=thresholds)
    feature_drift = compute_feature_drift(reference_rows, current_rows, feature_columns)
    temporal_metrics = compute_temporal_metrics(audited_rows)
    data_quality = compute_data_quality(audited_rows, reference_rows, current_rows)
    governance = decide_governance(
        overall=overall,
        reference=reference,
        current=current,
        calibration=calibration,
        reference_calibration=reference_calibration,
        current_calibration=current_calibration,
        subgroup_metrics=subgroup_metrics,
        dimension_summary=dimension_summary,
        feature_drift=feature_drift,
        data_quality=data_quality,
        thresholds=thresholds,
        policy_version=policy_version,
    )

    return EvaluationResult(
        scenario_name=scenario_name,
        source=source,
        validation=validation,
        audited_rows=audited_rows,
        reference_rows=reference_rows,
        current_rows=current_rows,
        overall=overall,
        reference=reference,
        current=current,
        calibration=calibration,
        reference_calibration=reference_calibration,
        current_calibration=current_calibration,
        subgroup_metrics=subgroup_metrics,
        dimension_summary=dimension_summary,
        feature_drift=feature_drift,
        temporal_metrics=temporal_metrics,
        data_quality=data_quality,
        governance=governance,
    )


def classification_metrics(df: pd.DataFrame) -> ClassificationMetrics:
    valid = _valid_prediction_rows(df)
    if valid.empty:
        return ClassificationMetrics(0, 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    y_true = valid["y_true"].astype(int)
    y_pred = valid["y_pred"].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = int(len(valid))

    return ClassificationMetrics(
        n=n,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        recall=_safe_divide(tp, tp + fn),
        ppv=_safe_divide(tp, tp + fp),
        fnr=_safe_divide(fn, tp + fn),
        fpr=_safe_divide(fp, fp + tn),
        accuracy=_safe_divide(tp + tn, n),
        specificity=_safe_divide(tn, tn + fp),
        prevalence=_safe_divide(tp + fn, n),
    )


def expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> CalibrationMetrics:
    valid = df.dropna(subset=["y_true", "y_score"]).copy()
    if valid.empty:
        empty_curve = pd.DataFrame(columns=["bin", "lower", "upper", "predicted", "observed", "count"])
        return CalibrationMetrics(ece=np.nan, brier_score=np.nan, curve=empty_curve)

    y_true = valid["y_true"].astype(float).to_numpy()
    scores = valid["y_score"].astype(float).clip(0, 1).to_numpy()
    bin_indexes = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(n_bins):
        mask = bin_indexes == index
        count = int(mask.sum())
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if count:
            predicted = float(scores[mask].mean())
            observed = float(y_true[mask].mean())
            ece += (count / len(scores)) * abs(predicted - observed)
        else:
            predicted = np.nan
            observed = np.nan
        rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "predicted": predicted,
                "observed": observed,
                "count": count,
            }
        )

    brier = float(np.mean((scores - y_true) ** 2))
    return CalibrationMetrics(ece=float(ece), brier_score=brier, curve=pd.DataFrame(rows))


def compute_subgroup_metrics(
    df: pd.DataFrame,
    dimensions: list[str],
    thresholds: GovernanceThresholds,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        if dimension not in df.columns:
            continue
        grouped = df.assign(_subgroup_value=df[dimension].fillna("Unknown").astype(str)).groupby(
            "_subgroup_value", dropna=False
        )
        for value, group in grouped:
            metrics = classification_metrics(group)
            calibration = expected_calibration_error(group)
            n = metrics.n
            if n < thresholds.subgroup_suppress_n:
                reliability = "suppressed"
            elif n <= thresholds.subgroup_caution_n:
                reliability = "caution"
            else:
                reliability = "ok"

            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "n": n,
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "tn": metrics.tn,
                    "fn": metrics.fn,
                    "outcome_support": metrics.tp + metrics.fn,
                    "predicted_positive_support": metrics.tp + metrics.fp,
                    "recall": metrics.recall,
                    "ppv": metrics.ppv,
                    "fnr": metrics.fnr,
                    "fpr": metrics.fpr,
                    "accuracy": metrics.accuracy,
                    "ece": calibration.ece,
                    "brier_score": calibration.brier_score,
                    "reliability": reliability,
                    "imputed_rate": _imputed_rate(group),
                    "avg_missingness_rate": _row_missingness(group),
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.sort_values(["dimension", "n"], ascending=[True, False]).reset_index(drop=True)


def summarize_dimensions(
    subgroup_metrics: pd.DataFrame, thresholds: GovernanceThresholds
) -> pd.DataFrame:
    if subgroup_metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for dimension, group in subgroup_metrics.groupby("dimension", dropna=False):
        total_n = float(group["n"].sum())
        reliable = group[group["n"] >= thresholds.subgroup_suppress_n].copy()
        fnr_groups = reliable[
            (reliable["outcome_support"] >= thresholds.min_outcome_support)
            & reliable["fnr"].notna()
        ]
        ppv_groups = reliable[
            (reliable["predicted_positive_support"] >= thresholds.min_predicted_positive_support)
            & reliable["ppv"].notna()
        ]

        fnr_gap, fnr_worst, fnr_best = _gap_details(fnr_groups, "fnr", high_is_worse=True)
        ppv_gap, ppv_worst, ppv_best = _gap_details(ppv_groups, "ppv", high_is_worse=False)
        min_recall_row = _min_metric_row(fnr_groups, "recall")
        reliable_coverage = _safe_divide(float(reliable["n"].sum()), total_n)
        rows.append(
            {
                "dimension": dimension,
                "groups": int(len(group)),
                "reliable_groups": int(len(reliable)),
                "suppressed_groups": int((group["reliability"] == "suppressed").sum()),
                "caution_groups": int((group["reliability"] == "caution").sum()),
                "reliable_coverage": reliable_coverage,
                "fnr_gap": fnr_gap,
                "fnr_worst_group": fnr_worst,
                "fnr_best_group": fnr_best,
                "ppv_gap": ppv_gap,
                "ppv_worst_group": ppv_worst,
                "ppv_best_group": ppv_best,
                "min_recall": None if min_recall_row is None else float(min_recall_row["recall"]),
                "min_recall_group": None if min_recall_row is None else str(min_recall_row["value"]),
                "max_ece": float(reliable["ece"].max()) if len(reliable) else np.nan,
                "max_ece_group": _max_metric_value(reliable, "ece"),
                "max_missingness": float(reliable["avg_missingness_rate"].max())
                if len(reliable)
                else np.nan,
                "max_imputed_rate": float(reliable["imputed_rate"].max()) if len(reliable) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("dimension").reset_index(drop=True)


def compute_feature_drift(
    reference_rows: pd.DataFrame,
    current_rows: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if reference_rows.empty or current_rows.empty:
        return pd.DataFrame(columns=["feature", "psi", "reference_mean", "current_mean", "shift"])

    for feature in feature_columns:
        ref = pd.to_numeric(reference_rows[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current_rows[feature], errors="coerce").dropna()
        if len(ref) < 20 or len(cur) < 20:
            continue
        psi = population_stability_index(ref, cur)
        ref_mean = float(ref.mean())
        cur_mean = float(cur.mean())
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "shift": cur_mean - ref_mean,
                "status": "NEEDS REVIEW" if psi > 0.20 else "WATCH" if psi > 0.10 else "PASS",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "psi", "reference_mean", "current_mean", "shift"])
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if ref.empty or cur.empty:
        return np.nan
    combined_min = min(float(ref.min()), float(cur.min()))
    combined_max = max(float(ref.max()), float(cur.max()))
    if combined_min == combined_max:
        return 0.0

    bins = np.linspace(combined_min, combined_max, n_bins + 1)
    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)
    epsilon = 1e-6
    ref_pct = (ref_counts + epsilon) / (len(ref) + epsilon * n_bins)
    cur_pct = (cur_counts + epsilon) / (len(cur) + epsilon * n_bins)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_temporal_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "service_month" not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for period, group in df.groupby("service_month", dropna=False):
        metrics = classification_metrics(group)
        calibration = expected_calibration_error(group)
        rows.append(
            {
                "period": str(period),
                "sort_key": _service_month_sort_key(str(period)),
                "n": metrics.n,
                "recall": metrics.recall,
                "fnr": metrics.fnr,
                "ppv": metrics.ppv,
                "ece": calibration.ece,
                "prevalence": metrics.prevalence,
                "missingness": _row_missingness(group),
            }
        )
    return pd.DataFrame(rows).sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)


def compute_data_quality(
    audited_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    current_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    core_fields = present_columns(audited_rows.columns, CORE_DATA_QUALITY_FIELDS)
    for field in core_fields:
        reference_missing = _column_missingness(reference_rows, field)
        current_missing = _column_missingness(current_rows, field)
        rows.append(
            {
                "metric": f"{field} missingness",
                "domain": "Core data field",
                "reference": reference_missing,
                "current": current_missing,
                "value": current_missing,
                "delta": current_missing - reference_missing,
                "detail": "Share of rows missing this model input in the current window.",
            }
        )

    rows.append(
        {
            "metric": "Row missingness rate",
            "domain": "Data completeness",
            "reference": _row_missingness(reference_rows),
            "current": _row_missingness(current_rows),
            "value": _row_missingness(current_rows),
            "delta": _row_missingness(current_rows) - _row_missingness(reference_rows),
            "detail": "Mean row-level missingness rate where supplied by the source data.",
        }
    )
    rows.append(
        {
            "metric": "Any imputation flag",
            "domain": "Data completeness",
            "reference": _imputed_rate(reference_rows),
            "current": _imputed_rate(current_rows),
            "value": _imputed_rate(current_rows),
            "delta": _imputed_rate(current_rows) - _imputed_rate(reference_rows),
            "detail": "Rows with at least one imputed value. Used as a stewardship flag.",
        }
    )
    return pd.DataFrame(rows)


def decide_governance(
    *,
    overall: ClassificationMetrics,
    reference: ClassificationMetrics | None,
    current: ClassificationMetrics | None,
    calibration: CalibrationMetrics,
    reference_calibration: CalibrationMetrics | None,
    current_calibration: CalibrationMetrics | None,
    subgroup_metrics: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    feature_drift: pd.DataFrame,
    data_quality: pd.DataFrame,
    thresholds: GovernanceThresholds,
    policy_version: str,
) -> GovernanceDecision:
    evidence: list[RuleEvidence] = []

    evidence.append(_fnr_disparity_evidence(dimension_summary, thresholds))
    evidence.append(_recall_evidence(overall, dimension_summary, thresholds))
    evidence.append(_ppv_disparity_evidence(dimension_summary, thresholds))
    evidence.append(_overall_ece_evidence(calibration, thresholds))
    evidence.append(_subgroup_ece_evidence(subgroup_metrics, thresholds))
    evidence.append(_sample_reliability_evidence(dimension_summary, thresholds))
    evidence.append(_feature_psi_evidence(feature_drift, thresholds))
    evidence.append(_performance_drift_evidence(reference, current, thresholds))
    evidence.append(
        _calibration_drift_evidence(reference_calibration, current_calibration, thresholds)
    )
    evidence.append(_missingness_drift_evidence(data_quality, thresholds))
    evidence.append(_subgroup_missingness_evidence(subgroup_metrics, thresholds))

    decision_weighted = [item for item in evidence if item.decision_weight]
    failures = [item for item in decision_weighted if item.status == "FAIL"]
    reviews = [item for item in decision_weighted if item.status == "NEEDS REVIEW"]
    hard_failures = [item for item in failures if item.hard_fail]
    significant_failures = [
        item
        for item in failures
        if item.exceedance_ratio >= thresholds.significant_exceedance_ratio
    ]

    if hard_failures:
        decision = "FAIL"
        reason = f"Hard stop: {hard_failures[0].name}."
    elif len(failures) >= 2:
        decision = "FAIL"
        reason = f"{len(failures)} simultaneous Tier 1 governance breaches."
    elif significant_failures:
        decision = "FAIL"
        reason = f"{significant_failures[0].name} exceeds policy tolerance by 20% or more."
    elif len(failures) == 1:
        decision = "NEEDS REVIEW"
        reason = f"{failures[0].name} is outside threshold but below the 20% fail margin."
    elif reviews:
        decision = "NEEDS REVIEW"
        reason = f"{len(reviews)} monitoring signal(s) require committee review."
    else:
        decision = "PASS"
        reason = "Tier 1 safety, equity, and calibration checks are within policy thresholds."

    return GovernanceDecision(
        decision=decision,
        reason=reason,
        evidence=evidence,
        pass_count=sum(item.status == "PASS" for item in evidence),
        watch_count=sum(item.status == "WATCH" for item in evidence),
        review_count=sum(item.status == "NEEDS REVIEW" for item in evidence),
        fail_count=len(failures),
        hard_fail_count=len(hard_failures),
        policy_version=policy_version,
    )


def _fnr_disparity_evidence(
    dimension_summary: pd.DataFrame, thresholds: GovernanceThresholds
) -> RuleEvidence:
    row = _max_row(dimension_summary, "fnr_gap")
    if row is None or pd.isna(row["fnr_gap"]):
        return _evidence(
            "tier1.fnr_disparity",
            "FNR disparity gap",
            "WATCH",
            None,
            f"<= {thresholds.fnr_disparity_gap:.0%}",
            "Insufficient subgroup outcome support for an FNR disparity estimate.",
            "Suppressed groups remain visible in the explorer but are excluded from the gap.",
            "Tier 1",
            "Quality and Patient Safety",
            decision_weight=False,
        )
    value = float(row["fnr_gap"])
    status: EvidenceStatus = "PASS" if value <= thresholds.fnr_disparity_gap else "FAIL"
    return _evidence(
        "tier1.fnr_disparity",
        "FNR disparity gap",
        status,
        value,
        f"<= {thresholds.fnr_disparity_gap:.0%}",
        f"Worst gap is {value:.1%} across {row['dimension']}.",
        f"{row['fnr_worst_group']} has higher FNR than {row['fnr_best_group']}.",
        "Tier 1",
        "Quality and Patient Safety",
        exceedance_ratio=_upper_exceedance(value, thresholds.fnr_disparity_gap),
    )


def _recall_evidence(
    overall: ClassificationMetrics,
    dimension_summary: pd.DataFrame,
    thresholds: GovernanceThresholds,
) -> RuleEvidence:
    subgroup_row = _min_row(dimension_summary, "min_recall")
    subgroup_value = None if subgroup_row is None else float(subgroup_row["min_recall"])
    candidate_values = [overall.recall]
    if subgroup_value is not None and not isnan(subgroup_value):
        candidate_values.append(subgroup_value)
    value = float(np.nanmin(candidate_values))
    status: EvidenceStatus = "PASS" if value >= thresholds.min_recall else "FAIL"
    if subgroup_row is not None and value == subgroup_value:
        summary = f"Minimum reliable subgroup recall is {value:.1%} in {subgroup_row['dimension']}."
        detail = f"Lowest reliable subgroup: {subgroup_row['min_recall_group']}."
    else:
        summary = f"Overall audited recall is {overall.recall:.1%}."
        detail = "Recall is sensitivity: true positives divided by all observed positives."
    return _evidence(
        "tier1.recall",
        "Recall / sensitivity",
        status,
        value,
        f">= {thresholds.min_recall:.0%}",
        summary,
        detail,
        "Tier 1",
        "Clinical Leadership",
        exceedance_ratio=_lower_exceedance(value, thresholds.min_recall),
    )


def _ppv_disparity_evidence(
    dimension_summary: pd.DataFrame, thresholds: GovernanceThresholds
) -> RuleEvidence:
    row = _max_row(dimension_summary, "ppv_gap")
    if row is None or pd.isna(row["ppv_gap"]):
        return _evidence(
            "tier1.ppv_disparity",
            "PPV disparity gap",
            "WATCH",
            None,
            f"<= {thresholds.ppv_disparity_gap:.0%}",
            "Insufficient positive prediction support for a PPV disparity estimate.",
            "The dashboard keeps this visible as a monitoring limitation.",
            "Tier 1",
            "Data and Analytics",
            decision_weight=False,
        )
    value = float(row["ppv_gap"])
    status: EvidenceStatus = "PASS" if value <= thresholds.ppv_disparity_gap else "FAIL"
    return _evidence(
        "tier1.ppv_disparity",
        "PPV disparity gap",
        status,
        value,
        f"<= {thresholds.ppv_disparity_gap:.0%}",
        f"Worst PPV gap is {value:.1%} across {row['dimension']}.",
        f"{row['ppv_worst_group']} differs from {row['ppv_best_group']} among reliable groups.",
        "Tier 1",
        "Data and Analytics",
        exceedance_ratio=_upper_exceedance(value, thresholds.ppv_disparity_gap),
    )


def _overall_ece_evidence(
    calibration: CalibrationMetrics, thresholds: GovernanceThresholds
) -> RuleEvidence:
    value = calibration.ece
    status: EvidenceStatus = "PASS" if value < thresholds.ece else "FAIL"
    return _evidence(
        "tier1.ece",
        "Expected Calibration Error",
        status,
        value,
        f"< {thresholds.ece:.2f}",
        f"Audited-window ECE is {value:.3f}.",
        "ECE compares average predicted probability with observed outcome frequency by score band.",
        "Tier 1",
        "Data and Analytics",
        exceedance_ratio=_upper_exceedance(value, thresholds.ece),
    )


def _subgroup_ece_evidence(
    subgroup_metrics: pd.DataFrame, thresholds: GovernanceThresholds
) -> RuleEvidence:
    if subgroup_metrics.empty:
        return _evidence(
            "tier1.subgroup_ece",
            "Subgroup calibration hard stop",
            "WATCH",
            None,
            f"<= {thresholds.subgroup_ece_fail:.2f}",
            "No subgroup calibration estimate is available.",
            "Subgroup ECE is only evaluated for reliable subgroups.",
            "Tier 1",
            "Data and Analytics",
            decision_weight=False,
        )
    reliable = subgroup_metrics[subgroup_metrics["n"] >= thresholds.subgroup_suppress_n]
    if reliable.empty:
        status: EvidenceStatus = "WATCH"
        value = None
        summary = "No reliable subgroup has enough rows for calibration interpretation."
        detail = "Suppression avoids overinterpreting sparse strata."
        decision_weight = False
    else:
        row = reliable.loc[reliable["ece"].idxmax()]
        value = float(row["ece"])
        status = "FAIL" if value > thresholds.subgroup_ece_fail else "PASS"
        summary = f"Maximum reliable subgroup ECE is {value:.3f}."
        detail = f"{row['dimension']} = {row['value']} has the highest subgroup ECE."
        decision_weight = True
    return _evidence(
        "tier1.subgroup_ece",
        "Subgroup calibration hard stop",
        status,
        value,
        f"<= {thresholds.subgroup_ece_fail:.2f}",
        summary,
        detail,
        "Tier 1",
        "Data and Analytics",
        decision_weight=decision_weight,
        hard_fail=status == "FAIL",
        exceedance_ratio=_upper_exceedance(value, thresholds.subgroup_ece_fail)
        if value is not None
        else 0.0,
    )


def _sample_reliability_evidence(
    dimension_summary: pd.DataFrame, thresholds: GovernanceThresholds
) -> RuleEvidence:
    if dimension_summary.empty:
        return _evidence(
            "tier2.sample_reliability",
            "Sample reliability coverage",
            "NEEDS REVIEW",
            None,
            f">= {thresholds.reliable_coverage_floor:.0%}",
            "No subgroup reliability summary is available.",
            "The committee should not interpret subgroup equity metrics without reliability metadata.",
            "Tier 2",
            "Data and Analytics",
        )
    min_row = dimension_summary.loc[dimension_summary["reliable_coverage"].idxmin()]
    value = float(min_row["reliable_coverage"])
    status: EvidenceStatus = (
        "PASS" if value >= thresholds.reliable_coverage_floor else "NEEDS REVIEW"
    )
    suppressed_total = int(dimension_summary["suppressed_groups"].sum())
    caution_total = int(dimension_summary["caution_groups"].sum())
    return _evidence(
        "tier2.sample_reliability",
        "Sample reliability coverage",
        status,
        value,
        f">= {thresholds.reliable_coverage_floor:.0%}",
        f"Lowest reliable coverage is {value:.1%} for {min_row['dimension']}.",
        f"{suppressed_total} groups suppressed and {caution_total} groups marked caution across monitored dimensions.",
        "Tier 2",
        "Data and Analytics",
    )


def _feature_psi_evidence(
    feature_drift: pd.DataFrame, thresholds: GovernanceThresholds
) -> RuleEvidence:
    row = _max_row(feature_drift, "psi")
    if row is None or pd.isna(row["psi"]):
        return _evidence(
            "drift.psi",
            "Population Stability Index",
            "WATCH",
            None,
            f"<= {thresholds.psi_review:.1f}",
            "PSI was not computed because a reference/current split or numeric features were unavailable.",
            "Snowflake source views should provide stable reference and current windows.",
            "Drift",
            "Data and Analytics",
            decision_weight=False,
        )
    value = float(row["psi"])
    status: EvidenceStatus = "NEEDS REVIEW" if value > thresholds.psi_review else "PASS"
    return _evidence(
        "drift.psi",
        "Population Stability Index",
        status,
        value,
        f"<= {thresholds.psi_review:.1f}",
        f"Highest PSI is {value:.3f} for {row['feature']}.",
        "PSI above 0.2 indicates a material current-window input distribution shift.",
        "Drift",
        "Data and Analytics",
    )


def _performance_drift_evidence(
    reference: ClassificationMetrics | None,
    current: ClassificationMetrics | None,
    thresholds: GovernanceThresholds,
) -> RuleEvidence:
    if reference is None or current is None or pd.isna(reference.fnr) or pd.isna(current.fnr):
        return _evidence(
            "drift.performance",
            "Performance drift",
            "WATCH",
            None,
            f"< +{thresholds.fnr_drift_review:.0%} FNR",
            "Reference/current FNR drift could not be computed.",
            "Performance drift requires paired outcomes in both windows.",
            "Drift",
            "Quality and Patient Safety",
            decision_weight=False,
        )
    value = current.fnr - reference.fnr
    fn_increase = current.fn - reference.fn
    reliable_count_shift = fn_increase >= 10
    if value >= thresholds.fnr_drift_review and reliable_count_shift:
        status: EvidenceStatus = "NEEDS REVIEW"
        decision_weight = True
    elif value >= thresholds.fnr_drift_review:
        status = "WATCH"
        decision_weight = False
    else:
        status = "PASS"
        decision_weight = True
    return _evidence(
        "drift.performance",
        "Performance drift",
        status,
        value,
        f"< +{thresholds.fnr_drift_review:.0%} FNR",
        f"Current FNR changed by {value:+.1%} from reference.",
        f"Reference FNR {reference.fnr:.1%}; current FNR {current.fnr:.1%}; false negatives changed by {fn_increase:+d}.",
        "Drift",
        "Quality and Patient Safety",
        decision_weight=decision_weight,
    )


def _calibration_drift_evidence(
    reference_calibration: CalibrationMetrics | None,
    current_calibration: CalibrationMetrics | None,
    thresholds: GovernanceThresholds,
) -> RuleEvidence:
    if reference_calibration is None or current_calibration is None:
        return _evidence(
            "drift.calibration",
            "Calibration drift",
            "WATCH",
            None,
            f"< +{thresholds.ece_drift_review:.2f} ECE",
            "Reference/current calibration drift could not be computed.",
            "Calibration drift requires outcome availability in both windows.",
            "Drift",
            "Data and Analytics",
            decision_weight=False,
        )
    value = current_calibration.ece - reference_calibration.ece
    status: EvidenceStatus = "NEEDS REVIEW" if value >= thresholds.ece_drift_review else "PASS"
    return _evidence(
        "drift.calibration",
        "Calibration drift",
        status,
        value,
        f"< +{thresholds.ece_drift_review:.2f} ECE",
        f"Current ECE changed by {value:+.3f} from reference.",
        f"Reference ECE {reference_calibration.ece:.3f}; current ECE {current_calibration.ece:.3f}.",
        "Drift",
        "Data and Analytics",
    )


def _missingness_drift_evidence(
    data_quality: pd.DataFrame, thresholds: GovernanceThresholds
) -> RuleEvidence:
    if data_quality.empty:
        return _evidence(
            "drift.missingness",
            "Missing data drift",
            "WATCH",
            None,
            f"< +{thresholds.missingness_drift_review:.0%}",
            "Missingness drift could not be computed.",
            "The data contract should provide row and core-field completeness signals.",
            "Drift",
            "Privacy, Legal, and Data Stewardship",
            decision_weight=False,
        )
    row = data_quality[data_quality["metric"] == "Row missingness rate"]
    row = data_quality.loc[[data_quality["delta"].idxmax()]] if row.empty else row.iloc[[0]]
    delta = float(row.iloc[0]["delta"])
    status: EvidenceStatus = (
        "NEEDS REVIEW" if delta >= thresholds.missingness_drift_review else "PASS"
    )
    return _evidence(
        "drift.missingness",
        "Missing data drift",
        status,
        delta,
        f"< +{thresholds.missingness_drift_review:.0%}",
        f"Current row missingness changed by {delta:+.1%}.",
        "A +10 percentage point missingness increase requires data stewardship review.",
        "Drift",
        "Privacy, Legal, and Data Stewardship",
    )


def _subgroup_missingness_evidence(
    subgroup_metrics: pd.DataFrame, thresholds: GovernanceThresholds
) -> RuleEvidence:
    if subgroup_metrics.empty:
        return _evidence(
            "tier2.subgroup_missingness",
            "Subgroup missingness flags",
            "WATCH",
            None,
            f"flag > {thresholds.imputed_or_missing_rate:.0%}",
            "No subgroup missingness summary is available.",
            "This is a stewardship flag rather than a standalone deployment block.",
            "Tier 2",
            "Privacy, Legal, and Data Stewardship",
            decision_weight=False,
        )
    reliable = subgroup_metrics[subgroup_metrics["n"] >= thresholds.subgroup_suppress_n]
    flagged = reliable[
        (reliable["avg_missingness_rate"] > thresholds.imputed_or_missing_rate)
        | (reliable["imputed_rate"] > thresholds.imputed_or_missing_rate)
    ]
    if flagged.empty:
        return _evidence(
            "tier2.subgroup_missingness",
            "Subgroup missingness flags",
            "PASS",
            0.0,
            f"flag > {thresholds.imputed_or_missing_rate:.0%}",
            "No reliable subgroup exceeds the missingness/imputation flag threshold.",
            "Core drift checks remain the decision-weighted data quality control.",
            "Tier 2",
            "Privacy, Legal, and Data Stewardship",
            decision_weight=False,
        )
    row = flagged.sort_values(["avg_missingness_rate", "imputed_rate"], ascending=False).iloc[0]
    value = max(float(row["avg_missingness_rate"]), float(row["imputed_rate"]))
    return _evidence(
        "tier2.subgroup_missingness",
        "Subgroup missingness flags",
        "WATCH",
        value,
        f"flag > {thresholds.imputed_or_missing_rate:.0%}",
        f"{len(flagged)} reliable subgroup(s) exceed the stewardship flag threshold.",
        f"Highest flag: {row['dimension']} = {row['value']}.",
        "Tier 2",
        "Privacy, Legal, and Data Stewardship",
        decision_weight=False,
    )


def _valid_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.copy()
    valid["y_true"] = pd.to_numeric(valid["y_true"], errors="coerce")
    valid["y_pred"] = pd.to_numeric(valid["y_pred"], errors="coerce")
    valid["y_score"] = pd.to_numeric(valid["y_score"], errors="coerce")
    valid = valid.dropna(subset=["y_true", "y_pred", "y_score"]).copy()
    valid["y_true"] = valid["y_true"].round().clip(0, 1).astype(int)
    valid["y_pred"] = valid["y_pred"].round().clip(0, 1).astype(int)
    valid["y_score"] = valid["y_score"].clip(0, 1)
    return valid


def _split_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "dataset_period" not in df.columns:
        return pd.DataFrame(columns=df.columns), df
    normalized = df["dataset_period"].astype(str).str.casefold()
    reference = df[normalized.eq("reference")].copy()
    current = df[normalized.eq("current")].copy()
    return reference, current


def _gap_details(
    df: pd.DataFrame,
    metric: str,
    *,
    high_is_worse: bool,
) -> tuple[float, str | None, str | None]:
    if df.empty or len(df) < 2:
        return np.nan, None, None
    max_row = df.loc[df[metric].idxmax()]
    min_row = df.loc[df[metric].idxmin()]
    if high_is_worse:
        return float(max_row[metric] - min_row[metric]), str(max_row["value"]), str(min_row["value"])
    return float(max_row[metric] - min_row[metric]), str(min_row["value"]), str(max_row["value"])


def _min_metric_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    usable = df[df[metric].notna()]
    if usable.empty:
        return None
    return usable.loc[usable[metric].idxmin()]


def _max_metric_value(df: pd.DataFrame, metric: str) -> str | None:
    usable = df[df[metric].notna()]
    if usable.empty:
        return None
    row = usable.loc[usable[metric].idxmax()]
    return f"{row['dimension']} = {row['value']}"


def _max_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    usable = df[df[metric].notna()]
    if usable.empty:
        return None
    return usable.loc[usable[metric].idxmax()]


def _min_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    usable = df[df[metric].notna()]
    if usable.empty:
        return None
    return usable.loc[usable[metric].idxmin()]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _upper_exceedance(value: float | None, threshold: float) -> float:
    if value is None or pd.isna(value) or threshold == 0:
        return 0.0
    return max(0.0, float(value / threshold - 1))


def _lower_exceedance(value: float | None, threshold: float) -> float:
    if value is None or pd.isna(value) or threshold == 0:
        return 0.0
    return max(0.0, float((threshold - value) / threshold))


def _evidence(
    rule_id: str,
    name: str,
    status: EvidenceStatus,
    value: float | None,
    threshold: str,
    summary: str,
    detail: str,
    tier: str,
    owner: str,
    *,
    decision_weight: bool = True,
    hard_fail: bool = False,
    exceedance_ratio: float = 0.0,
) -> RuleEvidence:
    return RuleEvidence(
        rule_id=rule_id,
        name=name,
        status=status,
        value=None if value is None or pd.isna(value) else float(value),
        threshold=threshold,
        summary=summary,
        detail=detail,
        tier=tier,
        owner=owner,
        decision_weight=decision_weight,
        hard_fail=hard_fail,
        exceedance_ratio=exceedance_ratio,
    )


def _row_missingness(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    if "missingness_rate_row" in df.columns:
        return float(pd.to_numeric(df["missingness_rate_row"], errors="coerce").fillna(0).mean())
    return float(df.isna().mean(axis=1).mean())


def _column_missingness(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    return float(df[column].isna().mean())


def _imputed_rate(df: pd.DataFrame) -> float:
    if df.empty or "is_imputed_any" not in df.columns:
        return np.nan
    values = df["is_imputed_any"].fillna("No").astype(str).str.casefold()
    return float(values.isin(["yes", "true", "1", "y"]).mean())


def _service_month_sort_key(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, format="%b-%y", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.max
    return parsed
