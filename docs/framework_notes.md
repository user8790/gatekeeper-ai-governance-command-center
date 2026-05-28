# AI Governance Framework Notes

Source file: `docs/AIGovernance_Framework.pdf`.

The framework positions Gatekeeper as a post-model governance evaluation layer. It is independent
from direct clinical workflows and produces standardized governance outcomes rather than diagnoses,
treatment recommendations, or patient-level clinical actions.

## Product Framing Used in This Prototype

- Synthetic pediatric diabetes / DKA monitoring context.
- System-level model behavior evaluation after predictions and outcomes are paired.
- Designed for clinical leadership, data/analytics teams, privacy/legal advisors, quality and
  patient-safety reviewers, and governance committees.
- Supports quality improvement, audit, research, monitoring, governance, and escalation.

## Framework Concepts Encoded

- Outcome types: TP, FP, TN, FN.
- Tier 1 metrics: FNR disparity, recall, PPV disparity, ECE.
- Tier 2 reliability: sample size suppression/caution and missingness/imputation flags.
- Drift metrics: PSI, FNR drift, missingness drift, calibration drift.
- Decision statuses: PASS, NEEDS REVIEW, FAIL.
- Escalation: minor deviations require review; major breaches, hard calibration stops, or multiple
  simultaneous breaches fail.
- Auditability: decision evidence traces to model version, data snapshot, thresholds, and policy.

## Implementation Interpretation

The canonical PASS fixture has a small FNR drift watch condition caused by a low absolute false
negative count movement. The prototype surfaces that watch signal but does not let it override the
PASS outcome unless the count shift is reliable. This keeps drift visible while respecting sample
reliability.
