from __future__ import annotations

import pandas as pd

from gatekeeper.data_providers import ScenarioProvider
from gatekeeper.metrics import classification_metrics, evaluate_model_governance


def _evidence(result, name: str):
    return next(item for item in result.governance.evidence if item.name == name)


def test_classification_metrics_counts() -> None:
    df = pd.DataFrame(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
            "y_score": [0.9, 0.2, 0.8, 0.1],
        }
    )
    metrics = classification_metrics(df)
    assert metrics.tp == 1
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert metrics.tn == 1
    assert metrics.recall == 0.5
    assert metrics.fnr == 0.5
    assert metrics.ppv == 0.5


def test_canonical_pass_fixture_passes() -> None:
    provider = ScenarioProvider()
    df, definition = provider.load("pass")
    result = evaluate_model_governance(df, definition.label, definition.source_type)
    assert result.governance.decision == "PASS"
    assert _evidence(result, "FNR disparity gap").status == "PASS"
    assert _evidence(result, "Subgroup calibration hard stop").status == "PASS"


def test_canonical_fail_fixture_fails() -> None:
    provider = ScenarioProvider()
    df, definition = provider.load("fail")
    result = evaluate_model_governance(df, definition.label, definition.source_type)
    assert result.governance.decision == "FAIL"
    assert _evidence(result, "FNR disparity gap").status == "FAIL"
    assert _evidence(result, "Recall / sensitivity").status == "FAIL"


def test_calibration_hard_fail_scenario() -> None:
    provider = ScenarioProvider()
    df, definition = provider.load("calibration_failure")
    result = evaluate_model_governance(df, definition.label, definition.source_type)
    evidence = _evidence(result, "Subgroup calibration hard stop")
    assert result.governance.decision == "FAIL"
    assert evidence.status == "FAIL"
    assert evidence.hard_fail


def test_low_sample_suppression_and_caution() -> None:
    provider = ScenarioProvider()
    df, definition = provider.load("low_sample_size")
    result = evaluate_model_governance(df, definition.label, definition.source_type)
    assert result.governance.decision == "NEEDS REVIEW"
    assert (result.subgroup_metrics["reliability"] == "suppressed").any()
    assert _evidence(result, "Sample reliability coverage").status == "NEEDS REVIEW"


def test_missingness_drift_warning() -> None:
    provider = ScenarioProvider()
    df, definition = provider.load("high_missingness")
    result = evaluate_model_governance(df, definition.label, definition.source_type)
    evidence = _evidence(result, "Missing data drift")
    assert result.governance.decision == "NEEDS REVIEW"
    assert evidence.status == "NEEDS REVIEW"


def test_population_drift_warning() -> None:
    provider = ScenarioProvider()
    df, definition = provider.load("needs_review")
    result = evaluate_model_governance(df, definition.label, definition.source_type)
    evidence = _evidence(result, "Population Stability Index")
    assert result.governance.decision == "NEEDS REVIEW"
    assert evidence.status == "NEEDS REVIEW"


def test_performance_drift_needs_review_when_count_reliable() -> None:
    reference = pd.DataFrame(
        {
            "dataset_period": ["Reference"] * 200,
            "service_month": ["Jan-24"] * 200,
            "y_true": [1] * 100 + [0] * 100,
            "y_pred": [0] * 5 + [1] * 95 + [0] * 100,
            "y_score": [0.2] * 5 + [0.8] * 95 + [0.05] * 100,
        }
    )
    current = pd.DataFrame(
        {
            "dataset_period": ["Current"] * 200,
            "service_month": ["Feb-24"] * 200,
            "y_true": [1] * 100 + [0] * 100,
            "y_pred": [0] * 20 + [1] * 80 + [0] * 100,
            "y_score": [0.2] * 20 + [0.8] * 80 + [0.05] * 100,
        }
    )
    result = evaluate_model_governance(pd.concat([reference, current], ignore_index=True))
    assert _evidence(result, "Performance drift").status == "NEEDS REVIEW"
