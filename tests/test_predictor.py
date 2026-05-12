"""
Unit tests for predictor.py.

Key concept: we use mock models instead of real sklearn/XGBoost objects.
Why? Because we are NOT testing whether the model predicts weight well —
that is evaluated during training (see trainer.py metrics).
Here we test that the predict_next_month / predict_next_week functions
correctly call the model and return results in the expected format.
MagicMock behaves like a real model but always returns a fixed value,
so tests run in milliseconds and need no internet connection.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock
from src.models.predictor import predict_next_month, predict_next_week


def make_mock_model(return_value: float):
    """Create a model stand-in that always returns the given value."""
    model = MagicMock()
    model.predict.return_value = [return_value]
    return model


class TestPredictNextMonth:
    def test_returns_prediction_for_each_model(self):
        models = {
            "linear_regression": make_mock_model(85.0),
            "gradient_boosting": make_mock_model(84.5),
            "xgboost": make_mock_model(84.8),
        }
        df = pd.DataFrame({
            "Last_Day_Weight": [85.5],
            "Weight_Change": [-0.3],
            "Weight": [85.2],
            "Rolling_3m": [85.4],
            "Target_Next_Month_Avg": [84.9],
        })
        result = predict_next_month(models, df)

        assert set(result.keys()) == {"linear_regression", "gradient_boosting", "xgboost"}
        assert result["linear_regression"] == pytest.approx(85.0)

    def test_predictions_are_rounded_to_2_decimals(self):
        models = {"linear_regression": make_mock_model(85.12345)}
        df = pd.DataFrame({
            "Last_Day_Weight": [85.5],
            "Weight_Change": [-0.3],
            "Weight": [85.2],
            "Rolling_3m": [85.4],
            "Target_Next_Month_Avg": [84.9],
        })
        result = predict_next_month(models, df)
        # verify result has at most 2 decimal places
        assert result["linear_regression"] == round(result["linear_regression"], 2)


class TestPredictNextWeek:
    def test_returns_prediction_for_each_model(self):
        models = {
            "linear_regression": make_mock_model(85.0),
            "gradient_boosting": make_mock_model(84.5),
            "xgboost": make_mock_model(84.8),
        }
        latest_input = pd.DataFrame({
            "Last_Day_Weight": [85.5],
            "Weight_Change": [-0.3],
            "Weight": [85.2],
            "Rolling_4w": [85.4],
        })
        result = predict_next_week(models, latest_input)
        assert len(result) == 3
        assert result["gradient_boosting"] == pytest.approx(84.5)