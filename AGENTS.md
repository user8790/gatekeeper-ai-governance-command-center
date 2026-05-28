# Agent Notes

## Mission

Build and maintain Gatekeeper as a Snowflake-portable AI governance command center for synthetic
pediatric diabetes / DKA risk monitoring. Keep the product framed as post-model governance, not
clinical decision support.

## Boundaries

- Use synthetic data only.
- Do not add Snowflake credentials, patient data, AHS data, or secrets.
- Do not present dashboard output as diagnosis, treatment advice, or patient-level CDS.
- Preserve minimum cell-size suppression and reliability cautions.

## Architecture Rules

- Keep ingestion/provider logic in `src/gatekeeper/data_providers.py`.
- Keep validation and field lists in `src/gatekeeper/contracts.py`.
- Keep metric computation and governance decisions in `src/gatekeeper/metrics.py`.
- Keep UI-only concerns in `streamlit_app.py`.
- Keep export payloads in `src/gatekeeper/reports.py`.
- Do not hard-code governance calculations in the Streamlit UI.

## Expected Checks

```powershell
uv run pytest
uv run ruff check .
uv run streamlit run streamlit_app.py
```

Use browser inspection after significant UI changes.

## Important Domain Details

- Tier 1 metrics drive governance: FNR disparity, recall, PPV disparity, and ECE.
- Subgroup ECE above `0.15` is a hard fail for reliable subgroups.
- Subgroups below `n < 50` are suppressed from interpretation.
- Drift and missingness are governance warning signals.
- Audit payloads should trace model version, data snapshot, policy version, thresholds, and evidence.

## Future Snowflake Work

Start from `HANDOFF_TO_SNOWFLAKE_DEVELOPER.md`, then apply SQL assets under `snowflake/`. Replace
the synthetic provider with a Snowpark provider that returns the same data contract.
