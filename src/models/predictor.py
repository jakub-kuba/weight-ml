"""
Prediction interface — separates inference from training.

Pattern: trainer.py creates models, predictor.py uses them.
FastAPI therefore doesn't need to know how models were trained.
"""

import pandas as pd
from src.models.trainer import MONTHLY_FEATURES, WEEKLY_FEATURES


def predict_next_month(models: dict, df_monthly: pd.DataFrame) -> dict:
    """
    Predict the average weight for the next calendar month.

    Uses the last row of monthly data as the input vector.
    Returns predictions from all three models.
    """
    latest = df_monthly.iloc[[-1]][MONTHLY_FEATURES]
    return {
        name: round(float(model.predict(latest)[0]), 2)
        for name, model in models.items()
    }


def predict_next_week(models: dict, latest_input: pd.DataFrame) -> dict:
    """
    Predict the average weight for the upcoming week.

    latest_input comes from loader.get_latest_weekly_input()
    and contains today's weight as Last_Day_Weight.
    """
    row = latest_input[WEEKLY_FEATURES]
    return {
        name: round(float(model.predict(row)[0]), 2)
        for name, model in models.items()
    }