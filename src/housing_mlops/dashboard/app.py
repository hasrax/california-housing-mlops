from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select
from sqlalchemy.engine import Engine

from housing_mlops.api.db import (
    DriftReport,
    ModelVersion,
    PredictionLog,
    PromotionDecision,
    check_connection,
    make_engine,
)
from housing_mlops.config import get_settings
from housing_mlops.data.schema import FEATURE_COLUMNS

SERIES_BLUE = "#2a78d6"
STATUS_SERIOUS = "#ec835a"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
FEATURE_COLORS = dict(
    zip(
        FEATURE_COLUMNS,
        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        strict=True,
    )
)


def base_layout(fig: go.Figure, *, y_title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_SECONDARY},
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(ttl=10)
def load_predictions(database_url: str, _engine: Engine) -> pd.DataFrame:
    with _engine.connect() as connection:
        return pd.read_sql(
            select(PredictionLog.created_at, PredictionLog.model_version, PredictionLog.prediction),
            connection,
        )


@st.cache_data(ttl=10)
def load_reports(database_url: str, _engine: Engine) -> pd.DataFrame:
    with _engine.connect() as connection:
        return pd.read_sql(
            select(DriftReport).order_by(DriftReport.created_at, DriftReport.id), connection
        )


@st.cache_data(ttl=10)
def load_versions(database_url: str, _engine: Engine) -> pd.DataFrame:
    with _engine.connect() as connection:
        return pd.read_sql(select(ModelVersion).order_by(ModelVersion.registered_at), connection)


@st.cache_data(ttl=10)
def load_decisions(database_url: str, _engine: Engine) -> pd.DataFrame:
    with _engine.connect() as connection:
        return pd.read_sql(
            select(PromotionDecision).order_by(PromotionDecision.created_at, PromotionDecision.id),
            connection,
        )


def volume_chart(predictions: pd.DataFrame) -> go.Figure:
    series = pd.to_datetime(predictions["created_at"], utc=True)
    span = series.max() - series.min() if len(series) > 1 else timedelta(0)
    freq = "1min" if span < timedelta(hours=2) else "1h"
    counts = series.dt.floor(freq).value_counts().sort_index()
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker={"color": SERIES_BLUE, "line": {"width": 0}},
            name="predictions",
            hovertemplate="%{x}<br>%{y} predictions<extra></extra>",
        )
    )
    return base_layout(fig, y_title=f"predictions per {freq}")


def psi_chart(features: pd.DataFrame, threshold: float) -> go.Figure:
    colors = [STATUS_SERIOUS if d else SERIES_BLUE for d in features["drifted"]]
    fig = go.Figure(
        go.Bar(
            x=features["feature"],
            y=features["psi"],
            marker={"color": colors, "line": {"width": 0}},
            text=["drift" if d else "" for d in features["drifted"]],
            textposition="outside",
            name="PSI",
            hovertemplate="%{x}<br>PSI %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=threshold,
        line={"dash": "dash", "color": TEXT_SECONDARY, "width": 1},
        annotation_text=f"threshold {threshold}",
        annotation_position="top left",
    )
    return base_layout(fig, y_title="population stability index")


def psi_history_chart(reports: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for feature in FEATURE_COLUMNS:
        if feature not in selected:
            continue
        values = [
            next(f["psi"] for f in details["features"] if f["feature"] == feature)
            for details in reports["details"]
        ]
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(reports["created_at"], utc=True),
                y=values,
                mode="lines+markers",
                name=feature,
                line={"color": FEATURE_COLORS[feature], "width": 2},
                marker={"size": 8},
                hovertemplate="%{y:.3f}<extra>" + feature + "</extra>",
            )
        )
    return base_layout(fig, y_title="PSI per drift report")


def rmse_chart(versions: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=versions["version"],
            y=versions["test_rmse"],
            marker={"color": SERIES_BLUE, "line": {"width": 0}},
            name="test RMSE",
            hovertemplate="%{x}<br>test RMSE %{y:.4f}<extra></extra>",
        )
    )
    return base_layout(fig, y_title="test RMSE", height=260)


def main() -> None:
    st.set_page_config(page_title="California Housing MLOps", layout="wide")
    st.title("California Housing MLOps")

    database_url = get_settings().database_url
    engine = make_engine(database_url)
    if not check_connection(engine):
        st.error("Database is not reachable. Start the stack with `docker compose up`.")
        engine.dispose()
        return

    predictions = load_predictions(database_url, engine)
    reports = load_reports(database_url, engine)
    versions = load_versions(database_url, engine)
    decisions = load_decisions(database_url, engine)
    engine.dispose()
    current = versions[versions["is_current"]].tail(1)
    latest = reports.tail(1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predictions logged", f"{len(predictions):,}")
    col2.metric("Current model", current["version"].iat[0] if len(current) else "none")
    col3.metric("Drift reports", len(reports))
    if len(latest):
        state = "drift detected" if bool(latest["drifted"].iat[0]) else "stable"
        col4.metric("Latest drift status", state, f"{int(latest['n_drifted'].iat[0])} features")
    else:
        col4.metric("Latest drift status", "no report yet")

    st.subheader("Prediction volume")
    if predictions.empty:
        st.info("No predictions yet. Run `python scripts/simulate_traffic.py`.")
    else:
        st.plotly_chart(volume_chart(predictions), width="stretch")

    st.subheader("Drift per feature")
    if reports.empty:
        st.info("No drift report yet. Run `python scripts/drift_check.py`.")
    else:
        details = latest["details"].iat[0]
        features = pd.DataFrame(details["features"])
        st.caption(
            f"Latest report at {latest['created_at'].iat[0]} for model "
            f"{latest['model_version'].iat[0]}, {int(latest['recent_count'].iat[0])} requests."
        )
        st.plotly_chart(psi_chart(features, details["thresholds"]["psi"]), width="stretch")
        st.dataframe(
            features[
                [
                    "feature",
                    "psi",
                    "ks_statistic",
                    "ks_p_value",
                    "reference_mean",
                    "recent_mean",
                    "drifted",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        if len(reports) > 1:
            default = list(features.sort_values("psi", ascending=False)["feature"].head(4))
            selected = st.multiselect("Features in history", FEATURE_COLUMNS, default=default)
            st.plotly_chart(psi_history_chart(reports, selected), width="stretch")

    st.subheader("Model version history")
    if versions.empty:
        st.info("No model registered yet. Start the API with a model in `models/current`.")
    else:
        st.dataframe(
            versions[
                [
                    "version",
                    "model_name",
                    "source",
                    "registered_at",
                    "training_rows",
                    "validation_rmse",
                    "test_rmse",
                    "test_r2",
                    "is_current",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        if len(versions) > 1:
            st.plotly_chart(rmse_chart(versions), width="stretch")

    st.subheader("Promotion decisions")
    if decisions.empty:
        st.info("No retraining run yet. Run `python scripts/retrain.py` after a drift report.")
    else:
        st.dataframe(
            decisions[
                [
                    "created_at",
                    "drift_report_id",
                    "candidate_version",
                    "champion_version",
                    "metric",
                    "candidate_value",
                    "champion_value",
                    "promoted",
                    "reason",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
