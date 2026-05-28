# Gatekeeper AI Governance Command Center

Gatekeeper is a working Streamlit prototype for post-model AI governance in a synthetic pediatric
diabetes / DKA risk monitoring context. It evaluates model behavior at the system level across
safety, equity, calibration, drift, data quality, reliability, and audit readiness.

Live app: [gatekeeper-ai-governance.streamlit.app](https://gatekeeper-ai-governance.streamlit.app/)

This is not a diagnostic tool, treatment recommendation tool, or patient-level clinical decision
support tool. It uses synthetic data only.

## What It Shows

- Executive command center with PASS, NEEDS REVIEW, and FAIL decisions.
- Plain-language explanation of why a governance decision fired.
- Traceable rule evidence with policy thresholds, owners, and decision weights.
- Subgroup fairness explorer across age, sex, health zone, geography, socioeconomic, race/ethnicity,
  Indigenous identity, language, newcomer status, CGM use, source system, diabetes type, and care setting.
- Drift sentinel for PSI, FNR drift, missingness drift, and calibration drift.
- Performance, calibration, and data quality drilldowns.
- Side-by-side scenario comparison.
- Oversight cadence, audit payload, glossary, and committee briefing export.
- Snowflake/Streamlit handoff package under `snowflake/`.

## Architecture

```text
data/scenarios/*.csv
        |
        v
src/gatekeeper/data_providers.py  -> swappable synthetic/Snowflake provider
        |
        v
src/gatekeeper/contracts.py       -> data contract validation
        |
        v
src/gatekeeper/metrics.py         -> metrics, drift, suppression, governance engine
        |
        v
src/gatekeeper/reports.py         -> structured evidence and committee briefing export
        |
        v
streamlit_app.py                  -> visualization and stakeholder workflow
```

## Scenarios

- `PASS scenario`: canonical synthetic fixture from `demo_pass_scenario.csv`.
- `FAIL scenario`: canonical synthetic fixture from `demo_fail_scenario.csv`.
- `NEEDS REVIEW - population drift`: generated from the PASS fixture with feature distribution shift.
- `FAIL - calibration sentinel`: generated subgroup ECE hard stop.
- `NEEDS REVIEW - low sample reliability`: generated sparse subgroup reliability issue.
- `NEEDS REVIEW - missing data drift`: generated core missingness drift.
- `FAIL - subgroup disparity`: generated FNR disparity and recall failure.

## Run Locally

Install dependencies:

```powershell
uv sync
```

Run the app:

```powershell
uv run streamlit run streamlit_app.py
```

Run tests and lint:

```powershell
uv run pytest
uv run ruff check .
```

## Data Placement

The repository includes the uploaded synthetic fixtures and framework copy:

- `data/scenarios/demo_pass_scenario.csv`
- `data/scenarios/demo_fail_scenario.csv`
- `docs/AIGovernance_Framework.pdf`

If those files are ever removed, place replacements at the same paths. The app can still generate
additional deterministic synthetic scenarios from the PASS fixture.

## Governance Policy Encoded

- FNR disparity gap: `<= 8` percentage points.
- Recall: `>= 85%`.
- PPV disparity gap: `<= 12` percentage points.
- ECE: `< 0.10`.
- Reliable subgroup ECE `> 0.15`: hard `FAIL`.
- Suppress subgroup interpretation when `n < 50`; mark caution when `50 <= n <= 100`.
- PSI `> 0.20`: `NEEDS REVIEW`.
- FNR drift `+5` percentage points: `NEEDS REVIEW` when the count shift is reliable.
- Missing-data drift `+10` percentage points: `NEEDS REVIEW`.
- Calibration drift `+0.05` ECE: `NEEDS REVIEW`.
- One Tier 1 breach under 20% exceedance: `NEEDS REVIEW`.
- One significant Tier 1 breach, two simultaneous breaches, or subgroup ECE hard stop: `FAIL`.

## Snowflake Handoff

The Snowflake migration package includes:

- `snowflake/proposed_schema.sql`
- `snowflake/proposed_views.sql`
- `snowflake/example_queries.sql`
- `snowflake/data_contract.md`
- `snowflake/streamlit_migration_notes.md`
- `HANDOFF_TO_SNOWFLAKE_DEVELOPER.md`

The intended migration path is to replace the provider layer with a Snowpark-backed provider while
keeping the contract, metric engine, governance engine, and report export logic stable.

## Privacy and Safety

Do not add real patient data, AHS data, Snowflake credentials, or secrets to this repository.
Sensitive attributes are used only for aggregate fairness monitoring with suppression and audit
assumptions documented in the Snowflake handoff.
