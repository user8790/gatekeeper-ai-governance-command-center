# Synthetic Data

This folder contains synthetic data only.

- `scenarios/demo_pass_scenario.csv`: canonical PASS fixture.
- `scenarios/demo_fail_scenario.csv`: canonical FAIL fixture.

Additional scenarios are generated deterministically in `src/gatekeeper/data_providers.py` using a
fixed seed. Generated files should not be committed unless there is a specific handoff need.

The CSV fields are mapped directly to the Snowflake contract described in
`snowflake/data_contract.md`.
