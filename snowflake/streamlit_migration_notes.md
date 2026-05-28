# Streamlit in Snowflake Migration Notes

## What Stays Stable

- `src/gatekeeper/contracts.py`: expected fields, subgroup dimensions, and feature lists.
- `src/gatekeeper/metrics.py`: metric computation, suppression, drift, and governance decision logic.
- `src/gatekeeper/models.py`: reusable dataclasses and threshold policy structure.
- `src/gatekeeper/reports.py`: structured evidence and committee briefing export format.

## What to Replace

Replace `ScenarioProvider` in `src/gatekeeper/data_providers.py` with a Snowpark-backed provider
that queries `AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS` and returns a Pandas DataFrame with the
same field names.

Suggested interface:

```python
class SnowflakeScenarioProvider:
    def __init__(self, session):
        self.session = session

    def load(self, model_id: str, model_version: str, snapshot_id: str):
        query = """
            select *
            from AI_GOVERNANCE.V_GATEKEEPER_EVALUATION_ROWS
            where model_id = ?
              and model_version = ?
              and snapshot_id = ?
        """
        return self.session.sql(query, params=[model_id, model_version, snapshot_id]).to_pandas()
```

## Environment Variables

In local development, use names only in `.env.example`. In Snowflake Streamlit, prefer Snowflake
roles, grants, and native connection context instead of secrets in code.

Expected names if a local bridge is later approved:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `GATEKEEPER_POLICY_VERSION`

## RBAC Pattern

- Analytics developers: read evaluation views, write validation tables in development schemas.
- Committee users: read aggregated dashboard views and governance decisions.
- Privacy/legal reviewers: read sensitive-attribute monitoring outputs through approved views.
- Production app role: read only the approved evaluation and decision views; write only decision
  evidence when explicitly approved.

## Deployment Steps

1. Create schema objects from `proposed_schema.sql`.
2. Create contract views from `proposed_views.sql`.
3. Backfill synthetic or approved non-production test rows.
4. Validate `example_queries.sql` against expected confusion matrix and drift outputs.
5. Replace `ScenarioProvider` with a Snowpark provider.
6. Run the Streamlit app inside Snowflake with the same metric engine.
7. Persist `governance_summary_json()` output to `GOVERNANCE_DECISIONS.report_json`.

## Privacy Guardrails

Keep patient-level rows out of screenshots, committee exports, and broad roles. The dashboard is a
system-level governance tool. It should show aggregate metrics, suppression states, and evidence
payloads rather than clinical recommendations.
