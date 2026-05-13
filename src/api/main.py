"""
FastAPI application — exposes ML models as an HTTP service.

On startup the app trains all models once and keeps them in memory.
Subsequent requests to /predict/* are served instantly without retraining.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.data.loader import get_monthly_data, get_weekly_data, get_latest_weekly_input
from src.models.trainer import train_and_log, MONTHLY_FEATURES, WEEKLY_FEATURES
from src.models.predictor import predict_next_month, predict_next_week

# Suppress noisy MLflow warnings that are irrelevant in this context:
# - pickle serialization warning (we trust our own models)
# - pip version resolution failure (uv project, pip not used)
# - uv/environment detection info logs
for _logger in ("mlflow.sklearn", "mlflow.utils.environment", "mlflow.utils.uv_utils"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "weight_api_requests_total",
    "Total number of prediction requests",
    ["endpoint"],
)
REQUEST_LATENCY = Histogram(
    "weight_api_request_duration_seconds",
    "Prediction request latency",
    ["endpoint"],
)

# ---------------------------------------------------------------------------
# App state — models are trained once at startup and reused
# ---------------------------------------------------------------------------
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train models on startup, clean up on shutdown."""
    print("Training models on startup...")

    df_monthly = get_monthly_data()
    monthly_models = train_and_log(
        df=df_monthly,
        features=MONTHLY_FEATURES,
        target_col="Target_Next_Month_Avg",
        experiment_name="weight-monthly",
    )

    df_weekly = get_weekly_data()
    weekly_models = train_and_log(
        df=df_weekly,
        features=WEEKLY_FEATURES,
        target_col="Target_Next_Week_Avg",
        experiment_name="weight-weekly",
    )

    app_state["monthly_models"] = monthly_models
    app_state["weekly_models"] = weekly_models
    app_state["df_monthly"] = df_monthly
    app_state["df_weekly"] = df_weekly
    app_state["trained_at"] = datetime.now().isoformat()

    print("Models ready.")
    yield
    app_state.clear()


app = FastAPI(
    title="Weight Prediction API",
    description="ML-powered weight prediction service",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — used by Docker and Kubernetes liveness probes."""
    return {
        "status": "ok",
        "trained_at": app_state.get("trained_at"),
    }


@app.get("/predict/monthly")
def predict_monthly():
    """
    Return next month's predicted average weight from all three models.
    Uses the last completed month as input features.
    """
    if "monthly_models" not in app_state:
        raise HTTPException(status_code=503, detail="Models not ready")

    REQUEST_COUNT.labels(endpoint="monthly").inc()

    with REQUEST_LATENCY.labels(endpoint="monthly").time():
        predictions = predict_next_month(
            app_state["monthly_models"],
            app_state["df_monthly"],
        )

    return {
        "period": "next_month",
        "predictions_kg": predictions,
        "trained_at": app_state["trained_at"],
    }


@app.get("/predict/weekly")
def predict_weekly():
    """
    Return next week's predicted average weight from all three models.
    Uses the last completed week (ending Sunday) as input features.
    """
    if "weekly_models" not in app_state:
        raise HTTPException(status_code=503, detail="Models not ready")

    REQUEST_COUNT.labels(endpoint="weekly").inc()

    with REQUEST_LATENCY.labels(endpoint="weekly").time():
        latest_input = get_latest_weekly_input(app_state["df_weekly"])
        predictions = predict_next_week(
            app_state["weekly_models"],
            latest_input,
        )

    return {
        "period": "next_week",
        "predictions_kg": predictions,
        "trained_at": app_state["trained_at"],
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint — scraped by Prometheus every 15s."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/models")
def model_metrics():
    """
    Return latest training metrics for all models from MLflow.
    Used by the Streamlit dashboard — avoids direct filesystem access to MLflow.
    """
    import mlflow

    result = {}
    for experiment_name in ("weight-monthly", "weight-weekly"):
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            result[experiment_name] = []
            continue

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=3,
        )

        result[experiment_name] = [
            {
                "model": run.info.run_name,
                "mae": round(run.data.metrics.get("mae", 0), 3),
                "rmse": round(run.data.metrics.get("rmse", 0), 3),
                "r2": round(run.data.metrics.get("r2", 0), 3),
                "directional_accuracy": round(
                    run.data.metrics.get("directional_accuracy", 0), 3
                ),
            }
            for run in runs
        ]

    return result