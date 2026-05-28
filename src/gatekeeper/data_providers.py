from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gatekeeper.contracts import REQUIRED_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "scenarios"


@dataclass(frozen=True)
class ScenarioDefinition:
    key: str
    label: str
    source_type: str
    description: str
    file_name: str | None = None


SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        key="pass",
        label="PASS scenario",
        source_type="csv",
        file_name="demo_pass_scenario.csv",
        description="Canonical synthetic fixture expected to pass Tier 1 governance checks.",
    ),
    ScenarioDefinition(
        key="fail",
        label="FAIL scenario",
        source_type="csv",
        file_name="demo_fail_scenario.csv",
        description="Canonical synthetic fixture with safety/equity breaches.",
    ),
    ScenarioDefinition(
        key="needs_review",
        label="NEEDS REVIEW - population drift",
        source_type="generated",
        description="PASS-like performance with a deliberate current-window feature distribution shift.",
    ),
    ScenarioDefinition(
        key="calibration_failure",
        label="FAIL - calibration sentinel",
        source_type="generated",
        description="A subgroup has unreliable predicted probabilities, triggering the ECE hard stop.",
    ),
    ScenarioDefinition(
        key="low_sample_size",
        label="NEEDS REVIEW - low sample reliability",
        source_type="generated",
        description="Small audited population with sparse subgroup strata requiring suppression.",
    ),
    ScenarioDefinition(
        key="high_missingness",
        label="NEEDS REVIEW - missing data drift",
        source_type="generated",
        description="Core-feature missingness rises in the current window.",
    ),
    ScenarioDefinition(
        key="subgroup_disparity",
        label="FAIL - subgroup disparity",
        source_type="generated",
        description="A deliberately degraded subgroup creates an FNR disparity breach.",
    ),
]


class ScenarioProvider:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self._definitions = {scenario.key: scenario for scenario in SCENARIOS}

    def list_scenarios(self) -> list[ScenarioDefinition]:
        return SCENARIOS

    def load(self, key: str) -> tuple[pd.DataFrame, ScenarioDefinition]:
        if key not in self._definitions:
            raise KeyError(f"Unknown scenario: {key}")

        definition = self._definitions[key]
        if definition.source_type == "csv" and definition.file_name:
            path = self.data_dir / definition.file_name
            if not path.exists():
                raise FileNotFoundError(
                    f"Scenario file {path} was not found. Place {definition.file_name} in data/scenarios."
                )
            df = pd.read_csv(path, low_memory=False)
        else:
            df = self._generate_scenario(key)

        df = normalize_dataframe(df)
        for field in REQUIRED_FIELDS:
            if field not in df.columns:
                raise ValueError(f"Scenario {key} is missing required field {field}.")
        return df, definition

    def _load_base_pass(self) -> pd.DataFrame:
        path = self.data_dir / "demo_pass_scenario.csv"
        if path.exists():
            return normalize_dataframe(pd.read_csv(path, low_memory=False))
        return _build_base_synthetic_records()

    def _generate_scenario(self, key: str) -> pd.DataFrame:
        base = self._load_base_pass()
        rng = np.random.default_rng(20260527)

        if key == "low_sample_size":
            sampled = (
                base.groupby("dataset_period", group_keys=False)
                .apply(lambda group: group.sample(n=min(90, len(group)), random_state=42))
                .reset_index(drop=True)
            )
            sampled.loc[:, "race_ethnicity_group"] = np.resize(
                ["Indigenous", "Black", "Latino", "Middle Eastern", "Other/Unknown"],
                len(sampled),
            )
            sampled.loc[:, "primary_language"] = np.resize(["EN", "FR", "Other"], len(sampled))
            sampled.loc[:, "shift_scenario"] = "low_sample_size"
            return sampled

        df = base.copy()
        df.loc[:, "shift_scenario"] = key
        df.loc[:, "run_id"] = f"RUN-{key.upper()}"
        current = df["dataset_period"].astype(str).str.lower().eq("current")

        if key == "needs_review":
            df.loc[current, "a1c_last_value"] = pd.to_numeric(
                df.loc[current, "a1c_last_value"], errors="coerce"
            ).add(1.8)
            df.loc[current, "mean_glucose_14d"] = pd.to_numeric(
                df.loc[current, "mean_glucose_14d"], errors="coerce"
            ).add(24)
            df.loc[current, "shift_intensity"] = "medium"

        elif key == "calibration_failure":
            affected = current & df["cgm_use"].astype(str).str.casefold().eq("no")
            scores = pd.to_numeric(df.loc[affected, "y_score"], errors="coerce").fillna(0)
            df.loc[affected, "y_score"] = np.clip(scores + 0.32, 0.0, 0.99)
            df.loc[affected, "shift_intensity"] = "high"

        elif key == "high_missingness":
            affected = current & df["primary_language"].astype(str).isin(["FR", "Other"])
            df.loc[current, "missingness_rate_row"] = pd.to_numeric(
                df.loc[current, "missingness_rate_row"], errors="coerce"
            ).fillna(0) + 0.12
            df.loc[affected, "missingness_rate_row"] = 0.35
            df.loc[affected, "is_imputed_any"] = "Yes"
            for column in ["a1c_last_value", "mean_glucose_14d", "time_in_range_14d"]:
                mask = affected & (rng.random(len(df)) < 0.55)
                df.loc[mask, column] = np.nan
            df.loc[current, "shift_intensity"] = "high"

        elif key == "subgroup_disparity":
            affected = (
                current
                & df["indigenous_identity"].astype(str).str.casefold().eq("yes")
                & pd.to_numeric(df["y_true"], errors="coerce").eq(1)
            )
            df.loc[affected, "y_pred"] = 0
            df.loc[affected, "y_score"] = np.minimum(
                pd.to_numeric(df.loc[affected, "y_score"], errors="coerce").fillna(0.05),
                0.18,
            )
            df.loc[current, "shift_intensity"] = "high"

        else:
            raise KeyError(f"No generator registered for scenario {key}.")

        return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    if "age_band" in normalized.columns:
        normalized["age_band"] = normalized["age_band"].replace(
            {
                "09-May": "5-9",
                "9-May": "5-9",
                "14-Oct": "10-14",
                "Oct-14": "10-14",
            }
        )

    for column in ["dataset_period", "deployment_phase"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].fillna("Unknown").astype(str).str.strip()

    for column in ["y_true", "y_pred"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round().astype("Int64")

    if "y_score" in normalized.columns:
        normalized["y_score"] = pd.to_numeric(normalized["y_score"], errors="coerce").clip(0, 1)

    if "decision_threshold" in normalized.columns and "y_pred" not in normalized.columns:
        threshold = pd.to_numeric(normalized["decision_threshold"], errors="coerce").fillna(0.5)
        normalized["y_pred"] = (normalized["y_score"] >= threshold).astype("Int64")

    return normalized


def _build_base_synthetic_records(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(20260527)
    periods = np.where(np.arange(n) < n / 2, "Reference", "Current")
    age_years = rng.integers(2, 18, size=n)
    age_band = pd.cut(
        age_years,
        bins=[0, 4, 9, 14, 18],
        labels=["0-4", "5-9", "10-14", "15-18"],
        include_lowest=True,
    ).astype(str)
    cgm_use = rng.choice(["Yes", "No"], size=n, p=[0.66, 0.34])
    indigenous = rng.choice(["No", "Yes"], size=n, p=[0.92, 0.08])
    base_risk = 0.025 + (cgm_use == "No") * 0.018 + (indigenous == "Yes") * 0.015
    y_true = rng.binomial(1, np.clip(base_risk, 0.01, 0.14))
    y_score = np.clip(base_risk + y_true * 0.52 + rng.normal(0, 0.05, n), 0.001, 0.99)
    y_pred = (y_score >= 0.35).astype(int)

    return pd.DataFrame(
        {
            "record_id": np.arange(1, n + 1),
            "patient_id": np.arange(1, n + 1),
            "encounter_id": [f"SYN-{i:05d}" for i in range(1, n + 1)],
            "dataset_period": periods,
            "deployment_phase": np.where(periods == "Reference", "pre", "post"),
            "service_month": rng.choice(
                ["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24"],
                size=n,
            ),
            "model_id": "AHS-DKA-GATEKEEPER",
            "model_version": "2024.01",
            "y_true": y_true,
            "y_score": y_score,
            "decision_threshold": 0.35,
            "y_pred": y_pred,
            "age_years": age_years,
            "age_band": age_band,
            "sex_at_birth": rng.choice(["Female", "Male"], size=n),
            "health_zone": rng.choice(["Calgary", "Central", "Edmonton", "North", "South"], size=n),
            "rural_urban": rng.choice(["Urban", "Rural"], size=n, p=[0.8, 0.2]),
            "socioeconomic_quintile": rng.integers(1, 6, size=n).astype(str),
            "race_ethnicity_group": rng.choice(
                ["White", "Indigenous", "Black", "Latino", "South Asian", "Other/Unknown"],
                size=n,
                p=[0.52, 0.08, 0.08, 0.08, 0.12, 0.12],
            ),
            "indigenous_identity": indigenous,
            "primary_language": rng.choice(["EN", "FR", "Other"], size=n, p=[0.86, 0.06, 0.08]),
            "newcomer_status": rng.choice(["No", "Yes"], size=n, p=[0.9, 0.1]),
            "cgm_use": cgm_use,
            "measurement_method_glucose": np.where(cgm_use == "Yes", "CGM-derived", "Lab-derived"),
            "data_source_system": rng.choice(["EHR_A", "EHR_B"], size=n),
            "diabetes_type": rng.choice(["T1D", "T2D", "Other"], size=n, p=[0.96, 0.03, 0.01]),
            "care_setting": rng.choice(
                ["Specialty clinic", "Community clinic", "Hospital-based program"],
                size=n,
                p=[0.48, 0.38, 0.14],
            ),
            "a1c_last_value": rng.normal(8.2, 1.4, n).round(2),
            "mean_glucose_14d": rng.normal(185, 38, n).round(1),
            "glucose_variability_14d": rng.normal(62, 18, n).round(1),
            "time_in_range_14d": rng.normal(62, 12, n).round(1),
            "missingness_rate_row": rng.choice([0, 0.05, 0.1], size=n, p=[0.72, 0.2, 0.08]),
            "is_imputed_any": rng.choice(["No", "Yes"], size=n, p=[0.88, 0.12]),
            "governance_policy_version": "policy-0.1",
            "shift_scenario": "fallback_synthetic",
        }
    )
