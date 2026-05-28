from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gatekeeper.contracts import SUBGROUP_DIMENSIONS
from gatekeeper.data_providers import SCENARIOS, ScenarioProvider
from gatekeeper.metrics import evaluate_model_governance
from gatekeeper.models import EvaluationResult, GovernanceThresholds
from gatekeeper.reports import (
    decimal,
    governance_markdown_report,
    governance_summary_json,
    percent,
)

APP_ROOT = Path(__file__).resolve().parent
STATUS_COLORS = {
    "PASS": "#167a5b",
    "WATCH": "#5c6676",
    "NEEDS REVIEW": "#b26a00",
    "FAIL": "#b42318",
    "SUPPRESSED": "#6b7280",
}
STATUS_BACKGROUNDS = {
    "PASS": "#e8f6f0",
    "WATCH": "#eef1f5",
    "NEEDS REVIEW": "#fff4df",
    "FAIL": "#fdecea",
    "SUPPRESSED": "#f2f4f7",
}


st.set_page_config(
    page_title="Gatekeeper AI Governance Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    inject_css()

    with st.sidebar:
        st.markdown("### Gatekeeper")
        st.caption("Synthetic pediatric diabetes / DKA model governance")
        scenario_key = st.selectbox(
            "Scenario",
            [scenario.key for scenario in SCENARIOS],
            format_func=lambda key: _scenario_label(key),
            index=0,
        )
        comparison_key = st.selectbox(
            "Comparison scenario",
            [scenario.key for scenario in SCENARIOS if scenario.key != scenario_key],
            format_func=lambda key: _scenario_label(key),
            index=0,
        )
        stakeholder_mode = st.radio(
            "View mode",
            [
                "Executive committee",
                "Clinical safety",
                "Data science",
                "Privacy and legal",
                "Snowflake developer",
            ],
        )
        policy_version = st.text_input("Policy version", value="policy-0.1")
        st.divider()
        st.caption(
            "Post-model governance only. This prototype does not diagnose, recommend treatment, "
            "or support patient-level clinical decisions."
        )

    try:
        result = load_result(scenario_key, policy_version)
        comparison = load_result(comparison_key, policy_version)
    except Exception as exc:  # pragma: no cover - Streamlit error surface
        st.error("The scenario could not be loaded.")
        st.code(str(exc))
        st.info(
            "Expected fixture placement: data/scenarios/demo_pass_scenario.csv and "
            "data/scenarios/demo_fail_scenario.csv. The framework PDF is expected at "
            "docs/AIGovernance_Framework.pdf."
        )
        return

    render_header(result, stakeholder_mode)

    tabs = st.tabs(
        [
            "Command Center",
            "Evidence Trail",
            "Equity Explorer",
            "Drift Sentinel",
            "Performance",
            "Calibration",
            "Data Quality",
            "Scenario Compare",
            "Oversight & Audit",
            "Snowflake Readiness",
            "Glossary & Report",
        ]
    )
    with tabs[0]:
        command_center(result)
    with tabs[1]:
        evidence_trail(result)
    with tabs[2]:
        equity_explorer(result)
    with tabs[3]:
        drift_sentinel(result)
    with tabs[4]:
        performance_view(result)
    with tabs[5]:
        calibration_view(result)
    with tabs[6]:
        data_quality_view(result)
    with tabs[7]:
        scenario_compare(result, comparison)
    with tabs[8]:
        oversight_audit(result, stakeholder_mode)
    with tabs[9]:
        snowflake_readiness(result)
    with tabs[10]:
        glossary_report(result)


@st.cache_data(show_spinner=False)
def load_result(scenario_key: str, policy_version: str) -> EvaluationResult:
    provider = ScenarioProvider()
    df, definition = provider.load(scenario_key)
    return evaluate_model_governance(
        df,
        scenario_name=definition.label,
        source=definition.source_type,
        thresholds=GovernanceThresholds(),
        policy_version=policy_version,
    )


def render_header(result: EvaluationResult, stakeholder_mode: str) -> None:
    decision = result.governance.decision
    st.markdown(
        f"""
        <section class="product-header">
          <div>
            <p class="eyebrow">Gatekeeper AI Governance Command Center</p>
            <h1>Pediatric DKA Risk Model Oversight</h1>
            <p class="subhead">
              Synthetic post-deployment monitoring for safety, equity, calibration, drift,
              data quality, and audit readiness. Current mode: {stakeholder_mode}.
            </p>
          </div>
          <div class="decision-pill" style="background:{STATUS_BACKGROUNDS[decision]};color:{STATUS_COLORS[decision]};border-color:{STATUS_COLORS[decision]}">
            {decision}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def command_center(result: EvaluationResult) -> None:
    decision = result.governance.decision
    st.markdown(
        f"""
        <div class="decision-banner" style="border-left-color:{STATUS_COLORS[decision]}">
          <div>
            <p class="eyebrow">Governance outcome</p>
            <h2>{decision}</h2>
            <p>{result.governance.reason}</p>
          </div>
          <div>
            <span class="small-label">Audited rows</span>
            <strong>{len(result.audited_rows):,}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        metric_card("False negative rate", percent(result.overall.fnr), "Safety risk", "lower is safer")
    with cols[1]:
        metric_card("Recall / sensitivity", percent(result.overall.recall), "Threshold >= 85%", "Tier 1")
    with cols[2]:
        metric_card("PPV", percent(result.overall.ppv), "Burden signal", "Tier 1")
    with cols[3]:
        metric_card("ECE", decimal(result.calibration.ece), "Threshold < 0.10", "Calibration")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Why This Decision")
        decision_evidence = evidence_dataframe(result)
        visible = decision_evidence[
            (decision_evidence["decision_weight"]) | (decision_evidence["status"].isin(["WATCH"]))
        ].copy()
        st.dataframe(
            visible[
                [
                    "status",
                    "name",
                    "value_display",
                    "threshold",
                    "tier",
                    "owner",
                    "summary",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Governance Readiness")
        readiness = pd.DataFrame(
            [
                {"Area": "Tier 1 safety/equity", "Status": _rollup_status(result, "Tier 1")},
                {"Area": "Tier 2 reliability", "Status": _rollup_status(result, "Tier 2")},
                {"Area": "Drift sentinel", "Status": _rollup_status(result, "Drift")},
                {"Area": "Snowflake portability", "Status": "PASS"},
                {"Area": "Audit payload", "Status": "PASS"},
            ]
        )
        fig = px.bar(
            readiness,
            x="Status",
            y="Area",
            orientation="h",
            color="Status",
            color_discrete_map=STATUS_COLORS,
            text="Status",
            height=300,
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=8, t=10, b=0), xaxis_title=None)
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Clinical Governance Interpretation")
    st.markdown(
        """
        Gatekeeper evaluates model behavior at the system level after predictions and outcomes are
        paired. The dashboard highlights missed-event risk, subgroup gaps, probability reliability,
        data drift, and whether the evidence is strong enough for committee interpretation.
        It does not produce diagnoses or treatment recommendations.
        """
    )


def evidence_trail(result: EvaluationResult) -> None:
    st.subheader("Traceable Rule Evidence")
    st.caption("Every outcome is tied to a policy rule, threshold, owner, and decision weight.")
    for item in result.governance.evidence:
        evidence_card(item)


def equity_explorer(result: EvaluationResult) -> None:
    st.subheader("Subgroup Fairness Explorer")
    available = [dimension for dimension in SUBGROUP_DIMENSIONS if dimension in result.subgroup_metrics["dimension"].unique()]
    dimension = st.selectbox("Equity dimension", available, index=available.index("health_zone") if "health_zone" in available else 0)
    data = result.subgroup_metrics[result.subgroup_metrics["dimension"] == dimension].copy()
    data["display"] = data["value"] + " (" + data["n"].astype(str) + ")"

    top_cols = st.columns(3)
    summary = result.dimension_summary[result.dimension_summary["dimension"] == dimension]
    if not summary.empty:
        row = summary.iloc[0]
        with top_cols[0]:
            metric_card("FNR gap", percent(row["fnr_gap"]), "Threshold <= 8%", "Reliable groups")
        with top_cols[1]:
            metric_card("PPV gap", percent(row["ppv_gap"]), "Threshold <= 12%", "Reliable groups")
        with top_cols[2]:
            metric_card("Reliable coverage", percent(row["reliable_coverage"]), "Suppression applied", "n < 50")

    plot_data = data.sort_values("fnr", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_data["value"],
            y=plot_data["fnr"],
            name="FNR",
            marker_color="#b42318",
            hovertemplate="%{x}<br>FNR %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_data["value"],
            y=plot_data["recall"],
            name="Recall",
            mode="lines+markers",
            marker_color="#167a5b",
            yaxis="y2",
            hovertemplate="%{x}<br>Recall %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(title="FNR", tickformat=".0%"),
        yaxis2=dict(title="Recall", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        data[
            [
                "value",
                "n",
                "reliability",
                "tp",
                "fp",
                "tn",
                "fn",
                "recall",
                "fnr",
                "ppv",
                "ece",
                "avg_missingness_rate",
                "imputed_rate",
            ]
        ].style.format(
            {
                "recall": "{:.1%}",
                "fnr": "{:.1%}",
                "ppv": "{:.1%}",
                "ece": "{:.3f}",
                "avg_missingness_rate": "{:.1%}",
                "imputed_rate": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Equity Risk Heatmap")
    heatmap = result.dimension_summary.copy()
    if not heatmap.empty:
        heatmap["FNR gap"] = heatmap["fnr_gap"]
        heatmap["PPV gap"] = heatmap["ppv_gap"]
        heatmap["Max ECE"] = heatmap["max_ece"]
        melted = heatmap.melt(
            id_vars="dimension",
            value_vars=["FNR gap", "PPV gap", "Max ECE"],
            var_name="Metric",
            value_name="Value",
        )
        fig = px.density_heatmap(
            melted,
            x="Metric",
            y="dimension",
            z="Value",
            color_continuous_scale=["#e8f6f0", "#fff4df", "#fdecea"],
            height=480,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_colorbar=dict(tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")


def drift_sentinel(result: EvaluationResult) -> None:
    st.subheader("Drift Sentinel")
    left, right = st.columns([1.1, 0.9])
    with left:
        drift = result.feature_drift.head(12)
        if drift.empty:
            st.info("No feature drift can be computed without reference and current windows.")
        else:
            fig = px.bar(
                drift.sort_values("psi"),
                x="psi",
                y="feature",
                orientation="h",
                color="status",
                color_discrete_map=STATUS_COLORS,
                text=drift.sort_values("psi")["psi"].map(lambda value: f"{value:.2f}"),
                height=460,
            )
            fig.add_vline(x=0.2, line_color="#b26a00", line_dash="dash")
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Population Stability Index")
            st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Drift Decision Signals")
        for item in result.governance.evidence:
            if item.tier == "Drift":
                compact_evidence(item)

    temporal = result.temporal_metrics
    if not temporal.empty:
        st.subheader("Model Health Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temporal["period"], y=temporal["fnr"], mode="lines+markers", name="FNR"))
        fig.add_trace(go.Scatter(x=temporal["period"], y=temporal["ece"], mode="lines+markers", name="ECE"))
        fig.add_trace(
            go.Scatter(
                x=temporal["period"],
                y=temporal["missingness"],
                mode="lines+markers",
                name="Missingness",
            )
        )
        fig.update_layout(
            height=380,
            yaxis_tickformat=".0%",
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, width="stretch")


def performance_view(result: EvaluationResult) -> None:
    st.subheader("Model Performance")
    cols = st.columns(4)
    with cols[0]:
        metric_card("True positives", f"{result.overall.tp:,}", "Correct high-risk identifications", "Audited")
    with cols[1]:
        metric_card("False negatives", f"{result.overall.fn:,}", "Missed observed outcomes", "Safety")
    with cols[2]:
        metric_card("False positives", f"{result.overall.fp:,}", "Unnecessary high-risk flags", "Burden")
    with cols[3]:
        metric_card("True negatives", f"{result.overall.tn:,}", "Correct low-risk classifications", "Audited")

    left, right = st.columns([0.9, 1.1])
    with left:
        matrix = pd.DataFrame(
            [
                [result.overall.tp, result.overall.fn],
                [result.overall.fp, result.overall.tn],
            ],
            index=["Observed outcome", "No observed outcome"],
            columns=["Predicted high risk", "Predicted low risk"],
        )
        fig = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale=["#f4f7fb", "#2f6f73"],
            aspect="auto",
            height=360,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        compare_rows = []
        for label, metrics in [("Reference", result.reference), ("Current", result.current), ("Audited", result.overall)]:
            if metrics is None:
                continue
            compare_rows.append(
                {
                    "Window": label,
                    "n": metrics.n,
                    "Recall": metrics.recall,
                    "FNR": metrics.fnr,
                    "PPV": metrics.ppv,
                    "Accuracy": metrics.accuracy,
                    "Prevalence": metrics.prevalence,
                }
            )
        st.dataframe(
            pd.DataFrame(compare_rows).style.format(
                {
                    "Recall": "{:.1%}",
                    "FNR": "{:.1%}",
                    "PPV": "{:.1%}",
                    "Accuracy": "{:.1%}",
                    "Prevalence": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def calibration_view(result: EvaluationResult) -> None:
    st.subheader("Calibration")
    left, right = st.columns([1.15, 0.85])
    with left:
        curve = result.calibration.curve.dropna(subset=["predicted", "observed"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ideal calibration",
                line=dict(color="#8a94a6", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve["predicted"],
                y=curve["observed"],
                mode="lines+markers",
                name="Observed calibration",
                marker=dict(size=curve["count"].clip(lower=4, upper=28), color="#2f6f73"),
            )
        )
        fig.update_layout(
            height=460,
            xaxis_title="Average predicted probability",
            yaxis_title="Observed outcome rate",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        metric_card("ECE", decimal(result.calibration.ece), "Threshold < 0.10", "Audited")
        metric_card("Brier score", decimal(result.calibration.brier_score), "Lower is better", "Probability error")
        top_ece = result.subgroup_metrics.sort_values("ece", ascending=False).head(8)
        st.dataframe(
            top_ece[["dimension", "value", "n", "reliability", "ece"]].style.format({"ece": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )


def data_quality_view(result: EvaluationResult) -> None:
    st.subheader("Data Quality and Missingness")
    quality = result.data_quality.copy()
    if quality.empty:
        st.info("No data quality metrics are available.")
        return
    quality["Status"] = quality["delta"].apply(lambda value: "NEEDS REVIEW" if value >= 0.10 else "PASS")
    fig = px.bar(
        quality.sort_values("current"),
        x="current",
        y="metric",
        orientation="h",
        color="Status",
        color_discrete_map=STATUS_COLORS,
        height=480,
    )
    fig.add_vline(x=0.10, line_color="#b26a00", line_dash="dash")
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        quality[["domain", "metric", "reference", "current", "delta", "detail"]].style.format(
            {"reference": "{:.1%}", "current": "{:.1%}", "delta": "{:+.1%}"}
        ),
        width="stretch",
        hide_index=True,
    )


def scenario_compare(result: EvaluationResult, comparison: EvaluationResult) -> None:
    st.subheader("Scenario Comparison")
    rows = []
    for item in [result, comparison]:
        rows.append(
            {
                "Scenario": item.scenario_name,
                "Decision": item.governance.decision,
                "Reason": item.governance.reason,
                "Recall": item.overall.recall,
                "FNR": item.overall.fnr,
                "PPV": item.overall.ppv,
                "ECE": item.calibration.ece,
                "Max PSI": item.feature_drift["psi"].max() if not item.feature_drift.empty else None,
                "Decision failures": item.governance.fail_count,
            }
        )
    compare = pd.DataFrame(rows)
    st.dataframe(
        compare.style.format(
            {
                "Recall": "{:.1%}",
                "FNR": "{:.1%}",
                "PPV": "{:.1%}",
                "ECE": "{:.3f}",
                "Max PSI": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    evidence = pd.concat(
        [
            evidence_dataframe(result).assign(Scenario=result.scenario_name),
            evidence_dataframe(comparison).assign(Scenario=comparison.scenario_name),
        ],
        ignore_index=True,
    )
    pivot = evidence.pivot_table(
        index="name",
        columns="Scenario",
        values="status",
        aggfunc="first",
    ).reset_index()
    st.dataframe(pivot, width="stretch", hide_index=True)


def oversight_audit(result: EvaluationResult, stakeholder_mode: str) -> None:
    st.subheader("Oversight Cadence and Auditability")
    st.markdown(
        f"""
        The selected view mode is **{stakeholder_mode}**. Governance decisions should be reviewed
        through a multidisciplinary committee with clinical, analytics, privacy/legal, digital
        health, quality/patient-safety, and patient/caregiver representation.
        """
    )
    cadence = pd.DataFrame(
        [
            {"Cadence": "Daily", "Signal": "Data pipeline completeness and schema validation", "Owner": "Data and Analytics"},
            {"Cadence": "Weekly", "Signal": "Drift sentinel, missingness drift, alert queue", "Owner": "Data Stewardship"},
            {"Cadence": "Monthly", "Signal": "Outcome-paired performance and subgroup equity", "Owner": "AI Oversight Committee"},
            {"Cadence": "Quarterly", "Signal": "Policy thresholds, model version review, audit sample", "Owner": "Clinical and Governance Leads"},
            {"Cadence": "Ad hoc", "Signal": "FAIL decision, hard calibration stop, stakeholder concern", "Owner": "Escalation Chair"},
        ]
    )
    st.dataframe(cadence, width="stretch", hide_index=True)

    metadata = {
        "model_id": _first_present(result.audited_rows, "model_id"),
        "model_version": _first_present(result.audited_rows, "model_version"),
        "policy_version": result.governance.policy_version,
        "data_snapshot_date": _first_present(result.audited_rows, "data_snapshot_date"),
        "run_id": _first_present(result.audited_rows, "run_id"),
        "intended_use": _first_present(result.audited_rows, "intended_use"),
        "clinical_risk_level": _first_present(result.audited_rows, "clinical_risk_level"),
    }
    st.json(metadata)


def snowflake_readiness(result: EvaluationResult) -> None:
    st.subheader("Snowflake and Streamlit Readiness")
    st.markdown(
        """
        The prototype separates ingestion, contract validation, metric computation, governance
        decisions, visualization, and report export. To move into Snowflake Streamlit, replace the
        CSV provider with a Snowpark query provider that returns the same contract fields.
        """
    )
    docs = pd.DataFrame(
        [
            {"Asset": "proposed_schema.sql", "Purpose": "Tables for predictions, outcomes, snapshots, decisions, and policy thresholds"},
            {"Asset": "proposed_views.sql", "Purpose": "Evaluation-ready views for paired prediction/outcome rows and monitoring windows"},
            {"Asset": "example_queries.sql", "Purpose": "Developer validation queries for metrics and audit payloads"},
            {"Asset": "data_contract.md", "Purpose": "Required fields, grain, windows, privacy, and RBAC assumptions"},
            {"Asset": "streamlit_migration_notes.md", "Purpose": "What to replace when running inside Snowflake Streamlit"},
        ]
    )
    st.dataframe(docs, width="stretch", hide_index=True)
    st.markdown("#### Current Contract Snapshot")
    st.dataframe(
        pd.DataFrame(
            [
                {"Field": column, "Present": column in result.audited_rows.columns}
                for column in [
                    "record_id",
                    "patient_id",
                    "encounter_id",
                    "dataset_period",
                    "y_true",
                    "y_score",
                    "y_pred",
                    "model_id",
                    "model_version",
                    "service_month",
                    "missingness_rate_row",
                    "is_imputed_any",
                ]
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def glossary_report(result: EvaluationResult) -> None:
    st.subheader("Glossary and Committee Briefing Export")
    glossary = pd.DataFrame(
        [
            {"Term": "True Positive", "Meaning": "The model flags a case that later has the observed outcome."},
            {"Term": "False Negative", "Meaning": "The model misses a case that later has the observed outcome; primary safety concern."},
            {"Term": "False Positive", "Meaning": "The model flags high risk when the outcome is not observed."},
            {"Term": "Recall / sensitivity", "Meaning": "TP / (TP + FN). Measures observed outcomes detected."},
            {"Term": "FNR", "Meaning": "FN / (TP + FN). Measures missed observed outcomes."},
            {"Term": "PPV", "Meaning": "TP / (TP + FP). Measures correctness among positive predictions."},
            {"Term": "ECE", "Meaning": "Expected Calibration Error, comparing predicted probability to observed frequency."},
            {"Term": "PSI", "Meaning": "Population Stability Index, measuring input distribution shift."},
            {"Term": "Suppression", "Meaning": "Subgroup interpretation is hidden or excluded when n < 50."},
        ]
    )
    st.dataframe(glossary, width="stretch", hide_index=True)

    markdown = governance_markdown_report(result)
    json_payload = governance_summary_json(result)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download committee briefing markdown",
            data=markdown,
            file_name="gatekeeper_governance_briefing.md",
            mime="text/markdown",
        )
    with col2:
        st.download_button(
            "Download structured evidence JSON",
            data=json_payload,
            file_name="gatekeeper_governance_evidence.json",
            mime="application/json",
        )
    st.text_area("Briefing preview", markdown, height=360)


def metric_card(label: str, value: str, helper: str, footnote: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <span class="small-label">{label}</span>
          <strong>{value}</strong>
          <p>{helper}</p>
          <span class="footnote">{footnote}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evidence_card(item) -> None:
    color = STATUS_COLORS[item.status]
    background = STATUS_BACKGROUNDS[item.status]
    value = _value_display(item)
    st.markdown(
        f"""
        <div class="evidence-card" style="border-left-color:{color}">
          <div class="evidence-head">
            <div>
              <span class="small-label">{item.tier} / {item.owner}</span>
              <h3>{item.name}</h3>
            </div>
            <span class="status-chip" style="background:{background};color:{color};border-color:{color}">{item.status}</span>
          </div>
          <p>{item.summary}</p>
          <div class="evidence-grid">
            <span><b>Value</b><br>{value}</span>
            <span><b>Threshold</b><br>{item.threshold}</span>
            <span><b>Decision weighted</b><br>{'Yes' if item.decision_weight else 'No'}</span>
            <span><b>Rule ID</b><br>{item.rule_id}</span>
          </div>
          <p class="detail">{item.detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_evidence(item) -> None:
    color = STATUS_COLORS[item.status]
    st.markdown(
        f"""
        <div class="compact-evidence" style="border-left-color:{color}">
          <strong>{item.name}</strong>
          <span style="color:{color}">{item.status}</span>
          <p>{item.summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evidence_dataframe(result: EvaluationResult) -> pd.DataFrame:
    rows = []
    for item in result.governance.evidence:
        payload = asdict(item)
        payload["value_display"] = _value_display(item)
        rows.append(payload)
    return pd.DataFrame(rows)


def _scenario_label(key: str) -> str:
    for scenario in SCENARIOS:
        if scenario.key == key:
            return scenario.label
    return key


def _value_display(item) -> str:
    if item.value is None:
        return "n/a"
    if "ECE" in item.name or "Index" in item.name:
        return decimal(item.value)
    if abs(item.value) <= 1:
        return percent(item.value)
    return decimal(item.value)


def _rollup_status(result: EvaluationResult, tier: str) -> str:
    statuses = [item.status for item in result.governance.evidence if item.tier == tier]
    if "FAIL" in statuses:
        return "FAIL"
    if "NEEDS REVIEW" in statuses:
        return "NEEDS REVIEW"
    if "WATCH" in statuses:
        return "WATCH"
    return "PASS"


def _first_present(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "not supplied"
    value = df[column].dropna()
    return str(value.iloc[0]) if len(value) else "not supplied"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #17202a;
          --muted: #5c6676;
          --line: #d9e0e8;
          --panel: #ffffff;
          --wash: #f5f8fb;
          --teal: #2f6f73;
        }
        .stApp {
          background: linear-gradient(180deg, #f7fafc 0%, #eef4f7 42%, #f8fafc 100%);
          color: var(--ink);
        }
        .block-container {
          padding-top: 1.1rem;
          max-width: 1480px;
        }
        .product-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          padding: 1.1rem 1.25rem;
          margin-bottom: 1rem;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(255,255,255,0.92);
        }
        .product-header h1 {
          margin: 0;
          font-size: 1.7rem;
          letter-spacing: 0;
        }
        .eyebrow, .small-label {
          margin: 0 0 0.2rem 0;
          text-transform: uppercase;
          letter-spacing: .08em;
          color: var(--muted);
          font-size: .72rem;
          font-weight: 700;
        }
        .subhead {
          color: var(--muted);
          margin: .25rem 0 0 0;
          max-width: 920px;
        }
        .decision-pill {
          min-width: 160px;
          text-align: center;
          border: 1px solid;
          border-radius: 8px;
          padding: .75rem 1rem;
          font-weight: 800;
          font-size: 1.05rem;
        }
        .decision-banner {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding: 1.1rem 1.25rem;
          border-left: 7px solid;
          border-radius: 8px;
          background: #ffffff;
          border-top: 1px solid var(--line);
          border-right: 1px solid var(--line);
          border-bottom: 1px solid var(--line);
          margin-bottom: 1rem;
        }
        .decision-banner h2 {
          margin: 0;
          font-size: 2rem;
          letter-spacing: 0;
        }
        .decision-banner p {
          margin: .2rem 0 0 0;
          color: var(--muted);
        }
        .metric-card {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 1rem;
          min-height: 148px;
          margin-bottom: .75rem;
        }
        .metric-card strong {
          display: block;
          font-size: 1.85rem;
          line-height: 1.15;
          margin: .3rem 0;
          letter-spacing: 0;
        }
        .metric-card p {
          margin: 0;
          color: var(--muted);
        }
        .footnote {
          display: inline-block;
          margin-top: .65rem;
          color: #2f6f73;
          font-size: .82rem;
          font-weight: 700;
        }
        .evidence-card, .compact-evidence {
          background: #ffffff;
          border: 1px solid var(--line);
          border-left: 6px solid;
          border-radius: 8px;
          padding: 1rem;
          margin-bottom: .85rem;
        }
        .evidence-head {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 1rem;
        }
        .evidence-head h3 {
          margin: 0;
          font-size: 1.1rem;
        }
        .status-chip {
          border: 1px solid;
          border-radius: 999px;
          padding: .25rem .55rem;
          font-weight: 800;
          font-size: .78rem;
          white-space: nowrap;
        }
        .evidence-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: .75rem;
          margin: .85rem 0;
        }
        .evidence-grid span {
          border-top: 1px solid var(--line);
          padding-top: .55rem;
          color: var(--muted);
          min-width: 0;
          overflow-wrap: anywhere;
        }
        .detail {
          color: var(--muted);
          margin-bottom: 0;
        }
        .compact-evidence strong,
        .compact-evidence span {
          display: block;
        }
        .compact-evidence p {
          color: var(--muted);
          margin: .3rem 0 0 0;
        }
        div[data-testid="stMetricValue"] {
          letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
