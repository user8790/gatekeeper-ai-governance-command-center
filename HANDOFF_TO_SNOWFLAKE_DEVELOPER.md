# Handoff to Snowflake Developer

Gatekeeper is ready to migrate as a Snowflake Streamlit app once an approved Snowflake environment,
source tables, and RBAC model exist.

## Local Prototype Shape

- UI entrypoint: `streamlit_app.py`.
- Reusable metric core: `src/gatekeeper/metrics.py`.
- Governance thresholds and dataclasses: `src/gatekeeper/models.py`.
- Synthetic CSV provider and generated scenarios: `src/gatekeeper/data_providers.py`.
- Report export: `src/gatekeeper/reports.py`.
- Canonical synthetic fixtures: `data/scenarios/demo_pass_scenario.csv` and `data/scenarios/demo_fail_scenario.csv`.
- Framework source copy: `docs/AIGovernance_Framework.pdf`.

## Migration Target

Expose a Snowflake view equivalent to `AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS` in
`snowflake/proposed_views.sql`. The metric engine expects Pandas-compatible columns with the same
names as the local CSVs.

## Files to Replace

Replace or extend only the provider layer first:

- Replace `ScenarioProvider.load()` with a Snowpark query.
- Keep `contracts.py`, `metrics.py`, `models.py`, and `reports.py` stable.
- Keep Streamlit tabs stable unless stakeholders request a workflow change.

## Required Data Decisions

- Confirm the approved prediction/outcome pairing logic.
- Confirm whether the audited Tier 1 window uses all paired rows in a snapshot or only current rows.
- Confirm the reference/current window dates for each model version.
- Confirm how sensitive attributes are authorized for fairness monitoring.
- Confirm cell-size suppression and export restrictions.
- Confirm where governance decisions and evidence payloads are persisted.

## Validation Checklist

- Contract view has `y_true`, `y_score`, and `y_pred`.
- Reference and current windows are both populated.
- Confusion matrix from Snowflake matches Python output on the same snapshot.
- ECE binning agrees within expected rounding tolerance.
- Subgroup suppression applies before equity interpretation.
- `PASS`, `NEEDS REVIEW`, and `FAIL` scenarios can be reproduced in non-production data.
- No secrets, real patient exports, or unapproved identifiers appear in app code.

## Current Assumptions

- This repo uses synthetic data only.
- The dashboard is non-diagnostic and non-prescriptive.
- Sensitive subgroup dimensions are used only for aggregate fairness monitoring.
- Drift can be calculated before all outcomes are mature; performance metrics require outcome pairing.
- A multidisciplinary committee owns the final governance interpretation.
