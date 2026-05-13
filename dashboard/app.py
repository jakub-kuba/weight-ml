"""
Streamlit dashboard — weight prediction UI.

Consumes the FastAPI service for predictions and model metrics.
The dashboard does not touch models, data, or MLflow directly —
it only calls the API. This separation of concerns means the UI
can be deployed independently from the ML backend.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import altair as alt
import pandas as pd
import requests
import streamlit as st

from src.data.loader import get_monthly_data, get_weekly_data

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_predictions(endpoint: str) -> dict | None:
    """Call the FastAPI prediction endpoint. Cache result for 5 minutes."""
    try:
        r = requests.get(f"{API_BASE}/predict/{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_model_metrics() -> dict:
    """Fetch model metrics from FastAPI /metrics/models endpoint."""
    try:
        r = requests.get(f"{API_BASE}/metrics/models", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def metrics_to_df(data: list) -> pd.DataFrame:
    """Convert list of metric dicts to a display DataFrame."""
    if not data:
        return pd.DataFrame()
    rows = []
    for item in data:
        rows.append({
            "Model": item["model"].replace("_", " ").title(),
            "MAE (kg)": item["mae"],
            "RMSE (kg)": item["rmse"],
            "R²": item["r2"],
            "Dir. Acc": f"{item['directional_accuracy']:.0%}",
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical weight data for charts."""
    monthly = get_monthly_data()
    weekly = get_weekly_data()
    return monthly, weekly


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def weight_chart(df: pd.DataFrame, x_col: str, color: str) -> alt.Chart:
    """
    Build an Altair line chart with Y axis zoomed to actual weight range.
    This makes subtle trends visible — a chart starting at 0 hides everything.
    """
    y_min = df["Weight"].min() - 1.5
    y_max = df["Weight"].max() + 1.5

    return (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=2)
        .encode(
            x=alt.X(x_col, title="Date"),
            y=alt.Y(
                "Weight:Q",
                title="Weight (kg)",
                scale=alt.Scale(domain=[y_min, y_max]),
            ),
            tooltip=[x_col, alt.Tooltip("Weight:Q", format=".2f")],
        )
        .properties(height=300)
        .interactive()
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Weight Tracker",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-left: 2rem; padding-right: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Weight Prediction Dashboard")
st.caption("Predictions powered by Linear Regression, Gradient Boosting and XGBoost.")

# --- Section 1: Predictions -------------------------------------------------
st.header("Predictions")

col_weekly, col_monthly = st.columns(2)

with col_weekly:
    st.subheader("Next week")
    weekly_data = fetch_predictions("weekly")
    if weekly_data:
        cols = st.columns(3)
        for col, (model, kg) in zip(cols, weekly_data["predictions_kg"].items()):
            col.metric(label=model.replace("_", " ").title(), value=f"{kg:.2f} kg")
        st.caption(f"Trained at: {weekly_data['trained_at'][:19]}")

with col_monthly:
    st.subheader("Next month")
    monthly_data = fetch_predictions("monthly")
    if monthly_data:
        cols = st.columns(3)
        for col, (model, kg) in zip(cols, monthly_data["predictions_kg"].items()):
            col.metric(label=model.replace("_", " ").title(), value=f"{kg:.2f} kg")
        st.caption(f"Trained at: {monthly_data['trained_at'][:19]}")

st.divider()

# --- Section 2: Historical data ---------------------------------------------
st.header("Weight history")

try:
    df_monthly_hist, df_weekly_hist = fetch_history()

    tab_monthly, tab_weekly = st.tabs(["Monthly averages", "Weekly averages"])

    with tab_monthly:
        chart_m = df_monthly_hist[["Weight"]].copy()
        chart_m["Date"] = pd.to_datetime(
            df_monthly_hist[["Year", "Month"]].assign(Day=1)
        )
        st.altair_chart(
            weight_chart(chart_m, "Date:T", "#4C9BE8"),
            width="stretch",
        )

    with tab_weekly:
        chart_w = df_weekly_hist[["Date", "Weight"]].dropna()
        st.altair_chart(
            weight_chart(chart_w, "Date:T", "#4CE8A0"),
            width="stretch",
        )

except Exception as e:
    st.warning(f"Could not load historical data: {e}")

st.divider()

# --- Section 3: Model metrics -----------------------------------------------
st.header("Model performance")

all_metrics = fetch_model_metrics()

tab_m, tab_w = st.tabs(["Monthly models", "Weekly models"])

with tab_m:
    df_m = metrics_to_df(all_metrics.get("weight-monthly", []))
    if df_m.empty:
        st.info("No metrics found. Run the API first.")
    else:
        st.dataframe(df_m.set_index("Model"), width="stretch")

with tab_w:
    df_w = metrics_to_df(all_metrics.get("weight-weekly", []))
    if df_w.empty:
        st.info("No metrics found. Run the API first.")
    else:
        st.dataframe(df_w.set_index("Model"), width="stretch")

st.divider()
st.caption("Data source: Google Sheets · API: FastAPI · Tracking: MLflow")