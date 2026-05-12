"""
Project entry point.

Run with: uv run python main.py
Trains all models and prints predictions, just like the original notebook,
but now the code is split into focused, testable modules.
"""

from src.data.loader import (
    get_monthly_data,
    get_weekly_data,
    get_latest_weekly_input,
)
from src.models.trainer import train_and_log, MONTHLY_FEATURES, WEEKLY_FEATURES
from src.models.predictor import predict_next_month, predict_next_week


def main():
    print("=" * 60)
    print("MONTHLY MODELS")
    print("=" * 60)

    df_monthly = get_monthly_data()
    monthly_models = train_and_log(
        df=df_monthly,
        features=MONTHLY_FEATURES,
        target_col="Target_Next_Month_Avg",
        experiment_name="weight-monthly",
    )

    monthly_preds = predict_next_month(monthly_models, df_monthly)
    print("\nNext month prediction:")
    for name, pred in monthly_preds.items():
        print(f"  {name}: {pred:.2f} kg")

    print("\n" + "=" * 60)
    print("WEEKLY MODELS")
    print("=" * 60)

    df_weekly = get_weekly_data()
    weekly_models = train_and_log(
        df=df_weekly,
        features=WEEKLY_FEATURES,
        target_col="Target_Next_Week_Avg",
        experiment_name="weight-weekly",
    )

    latest_w_input = get_latest_weekly_input(df_weekly)
    weekly_preds = predict_next_week(weekly_models, latest_w_input)
    print("\nNext week prediction:")
    for name, pred in weekly_preds.items():
        print(f"  {name}: {pred:.2f} kg")


if __name__ == "__main__":
    main()