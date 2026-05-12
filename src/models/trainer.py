"""
Model training with MLflow experiment tracking.

Every training run is logged as a separate MLflow run — giving you
a full history: when you trained, what the metrics were, what parameters
were used. This is the core of MLOps: reproducibility and auditability.
"""

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pandas as pd

from src.data.loader import TEST_SIZE


MONTHLY_FEATURES = ["Last_Day_Weight", "Weight_Change", "Weight", "Rolling_3m"]
WEEKLY_FEATURES = ["Last_Day_Weight", "Weight_Change", "Weight", "Rolling_4w"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_vals: np.ndarray,
) -> dict:
    """
    Compute a set of model quality metrics.

    Directional accuracy measures whether the model correctly predicts
    the DIRECTION of weight change (up/down) — often more meaningful
    to the end user than raw MAE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    pred_direction = np.sign(y_pred - prev_vals)
    actual_direction = np.sign(y_true - prev_vals)
    directional_accuracy = float(np.mean(pred_direction == actual_direction))

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "mape": round(mape, 4),
        "directional_accuracy": round(directional_accuracy, 4),
    }


def _build_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            reg_alpha=0.5,
            random_state=42,
        ),
    }


def train_and_log(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    experiment_name: str,
) -> dict:
    """
    Train all models, log results to MLflow, and return trained objects.

    Test split is controlled globally by TEST_SIZE in loader.py (default 15%).
    This ensures consistent train/test splits across monthly and weekly models.

    Args:
        df: DataFrame with aggregated data (monthly or weekly)
        features: list of input feature column names
        target_col: name of the target column
        experiment_name: MLflow experiment name (e.g. "monthly" / "weekly")

    Returns:
        Dict of {model_name: trained_model} — consumed by predictor.py
    """
    df_model = df.dropna(subset=features + [target_col]).copy()
    X = df_model[features]
    y = df_model[target_col]

    test_count = int(len(X) * TEST_SIZE)
    X_train, y_train = X.iloc[:-test_count], y.iloc[:-test_count]
    X_test, y_test = X.iloc[-test_count:], y.iloc[-test_count:]
    prev_vals = df.loc[y_test.index, "Weight"].values

    mlflow.set_experiment(experiment_name)

    trained = {}
    for name, model in _build_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test.values, y_pred, prev_vals)

            # Log parameters and metrics — visible later in MLflow UI
            mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name=name)

            trained[name] = model
            print(
                f"[{experiment_name}] {name} → "
                f"MAE: {metrics['mae']:.3f} kg | "
                f"Dir.Acc: {metrics['directional_accuracy']:.0%}"
            )

    return trained