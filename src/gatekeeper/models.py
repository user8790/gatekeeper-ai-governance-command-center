from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Decision = Literal["PASS", "NEEDS REVIEW", "FAIL"]
EvidenceStatus = Literal["PASS", "WATCH", "NEEDS REVIEW", "FAIL", "SUPPRESSED"]
ReliabilityStatus = Literal["ok", "caution", "suppressed"]


@dataclass(frozen=True)
class GovernanceThresholds:
    fnr_disparity_gap: float = 0.08
    min_recall: float = 0.85
    ppv_disparity_gap: float = 0.12
    ece: float = 0.10
    subgroup_ece_fail: float = 0.15
    subgroup_suppress_n: int = 50
    subgroup_caution_n: int = 100
    min_outcome_support: int = 10
    min_predicted_positive_support: int = 10
    imputed_or_missing_rate: float = 0.10
    core_missingness_review: float = 0.15
    reliable_coverage_floor: float = 0.95
    psi_review: float = 0.20
    fnr_drift_review: float = 0.05
    missingness_drift_review: float = 0.10
    ece_drift_review: float = 0.05
    significant_exceedance_ratio: float = 0.20


DEFAULT_THRESHOLDS = GovernanceThresholds()


@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    recall: float
    ppv: float
    fnr: float
    fpr: float
    accuracy: float
    specificity: float
    prevalence: float


@dataclass(frozen=True)
class CalibrationMetrics:
    ece: float
    brier_score: float
    curve: pd.DataFrame


@dataclass(frozen=True)
class RuleEvidence:
    rule_id: str
    name: str
    status: EvidenceStatus
    value: float | None
    threshold: str
    summary: str
    detail: str
    tier: str
    owner: str
    decision_weight: bool = True
    hard_fail: bool = False
    exceedance_ratio: float = 0.0


@dataclass(frozen=True)
class GovernanceDecision:
    decision: Decision
    reason: str
    evidence: list[RuleEvidence]
    pass_count: int
    watch_count: int
    review_count: int
    fail_count: int
    hard_fail_count: int
    policy_version: str = "policy-0.1"


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    missing_required: list[str]
    missing_recommended: list[str]
    row_count: int
    column_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    scenario_name: str
    source: str
    validation: ValidationResult
    audited_rows: pd.DataFrame
    reference_rows: pd.DataFrame
    current_rows: pd.DataFrame
    overall: ClassificationMetrics
    reference: ClassificationMetrics | None
    current: ClassificationMetrics | None
    calibration: CalibrationMetrics
    reference_calibration: CalibrationMetrics | None
    current_calibration: CalibrationMetrics | None
    subgroup_metrics: pd.DataFrame
    dimension_summary: pd.DataFrame
    feature_drift: pd.DataFrame
    temporal_metrics: pd.DataFrame
    data_quality: pd.DataFrame
    governance: GovernanceDecision
