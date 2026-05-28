from __future__ import annotations

import json
from dataclasses import asdict

from gatekeeper.models import EvaluationResult, RuleEvidence


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{value:.1%}"


def decimal(value: float | None, places: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{value:.{places}f}"


def governance_summary_dict(result: EvaluationResult) -> dict[str, object]:
    decision = result.governance
    return {
        "scenario": result.scenario_name,
        "source": result.source,
        "decision": decision.decision,
        "reason": decision.reason,
        "policy_version": decision.policy_version,
        "audited_rows": len(result.audited_rows),
        "reference_rows": len(result.reference_rows),
        "current_rows": len(result.current_rows),
        "overall": {
            "tp": result.overall.tp,
            "fp": result.overall.fp,
            "tn": result.overall.tn,
            "fn": result.overall.fn,
            "recall": result.overall.recall,
            "fnr": result.overall.fnr,
            "ppv": result.overall.ppv,
            "ece": result.calibration.ece,
            "brier_score": result.calibration.brier_score,
        },
        "evidence": [asdict(item) for item in decision.evidence],
        "validation": asdict(result.validation),
    }


def governance_summary_json(result: EvaluationResult) -> str:
    return json.dumps(governance_summary_dict(result), indent=2, default=str)


def governance_markdown_report(result: EvaluationResult) -> str:
    decision = result.governance
    lines = [
        f"# Gatekeeper AI Governance Briefing: {result.scenario_name}",
        "",
        f"Decision: **{decision.decision}**",
        f"Policy version: `{decision.policy_version}`",
        f"Reason: {decision.reason}",
        "",
        "This report uses synthetic data only. It is a post-model governance summary, not a diagnostic output, treatment recommendation, or patient-level clinical decision support tool.",
        "",
        "## Audited Window",
        "",
        f"- Audited rows: {len(result.audited_rows):,}",
        f"- Reference rows: {len(result.reference_rows):,}",
        f"- Current rows: {len(result.current_rows):,}",
        f"- True positives: {result.overall.tp:,}",
        f"- False positives: {result.overall.fp:,}",
        f"- True negatives: {result.overall.tn:,}",
        f"- False negatives: {result.overall.fn:,}",
        f"- Recall: {percent(result.overall.recall)}",
        f"- False negative rate: {percent(result.overall.fnr)}",
        f"- PPV: {percent(result.overall.ppv)}",
        f"- ECE: {decimal(result.calibration.ece)}",
        "",
        "## Decision Evidence",
        "",
    ]

    for item in decision.evidence:
        lines.extend(_evidence_lines(item))

    lines.extend(
        [
            "",
            "## Committee Next Steps",
            "",
            "- Review decision-weighted FAIL or NEEDS REVIEW evidence first.",
            "- Confirm subgroup suppression and caution notes before interpreting equity gaps.",
            "- Validate reference/current windows and outcome pairing before live use.",
            "- Preserve the evidence payload with the model version, policy version, data snapshot, and analyst sign-off.",
        ]
    )
    return "\n".join(lines)


def _evidence_lines(item: RuleEvidence) -> list[str]:
    value = decimal(item.value) if item.value is not None else "n/a"
    if item.value is not None and abs(item.value) <= 1.0:
        value = percent(item.value) if "ECE" not in item.name and "Index" not in item.name else decimal(item.value)
    return [
        f"### {item.name}",
        "",
        f"- Status: {item.status}",
        f"- Value: {value}",
        f"- Threshold: {item.threshold}",
        f"- Tier: {item.tier}",
        f"- Owner: {item.owner}",
        f"- Summary: {item.summary}",
        f"- Detail: {item.detail}",
        "",
    ]
