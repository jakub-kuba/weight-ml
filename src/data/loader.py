"""
Data loading and preprocessing for weight tracking data.

Single responsibility: this is the only module that knows where data
comes from and how to clean it. Everything else receives ready DataFrames.
"""

import pandas as pd
from datetime import datetime, timedelta


SHEET_ID = "1VurJODWndk26VfyNd6IDgXsvX7uR1A_IxhTA4xcGRco"
SHEET_GID = "0"
TEST_SIZE = 0.15


def fetch_raw_data() -> pd.DataFrame:
    """Fetch raw data from Google Sheets and return as DataFrame."""
    url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
        f"/export?format=csv&gid={SHEET_GID}"
    )
    return pd.read_csv(url)


def clean_weight_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Weight column: replace commas with dots, cast to float,
    and linearly interpolate missing values.
    """
    df = df.copy()
    df["Weight"] = (
        df["Weight"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Weight"] = df["Weight"].interpolate(method="linear")
    return df


def get_monthly_data() -> pd.DataFrame:
    """
    Return aggregated monthly data ready for model training.

    Excludes the current (incomplete) month to avoid data leakage —
    models should only learn from full months. ffill() after interpolate()
    handles edge cases where NaNs remain at the end of the series.
    """
    df = fetch_raw_data()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = clean_weight_column(df)

    now = datetime.now()
    last_day_of_prev_month = (now.replace(day=1) - timedelta(days=1)).replace(
        hour=23, minute=59
    )

    df = df[df["Date"] <= last_day_of_prev_month]
    df = df[df["Date"] >= "2021-12-01"]
    df = df.sort_values("Date")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    monthly = df.groupby(["Year", "Month"])["Weight"].mean().reset_index()
    monthly["Last_Day_Weight"] = (
        df.groupby(["Year", "Month"])["Weight"].last().values
    )
    monthly["Weight"] = monthly["Weight"].interpolate(method="linear").ffill()
    monthly["Weight_Change"] = monthly["Weight"].diff()
    monthly["Rolling_3m"] = monthly["Weight"].rolling(3).mean()
    monthly["Target_Next_Month_Avg"] = monthly["Weight"].shift(-1)

    return monthly


def get_weekly_data() -> pd.DataFrame:
    """
    Return aggregated weekly data ready for model training.

    Weeks end on Sunday (W-SUN). Excludes the current incomplete week
    to avoid data leakage — Last_Day_Weight is always a Sunday,
    making predictions fully deterministic regardless of run day.
    Interpolation after resample fills weeks with no measurements.
    """
    df = fetch_raw_data()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = clean_weight_column(df)
    df = df.sort_values("Date")

    now = datetime.now()
    last_sunday = (now - timedelta(days=(now.weekday() + 1) % 7)).replace(
        hour=23, minute=59, second=59
    )

    series = df[df["Date"] <= last_sunday].set_index("Date")["Weight"]

    weekly = series.resample("W-SUN").mean().reset_index()
    weekly["Weight"] = weekly["Weight"].interpolate(method="linear")
    weekly["Last_Day_Weight"] = series.resample("W-SUN").last().values
    weekly["Weight_Change"] = weekly["Weight"].diff()
    weekly["Rolling_4w"] = weekly["Weight"].rolling(4).mean()
    weekly["Target_Next_Week_Avg"] = weekly["Weight"].shift(-1)

    return weekly


def get_latest_weekly_input(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Return the last row of completed weekly data as the prediction input.

    Uses the last day of the most recent FULL week (Sunday) as Last_Day_Weight.
    This makes predictions deterministic regardless of when the script is run —
    avoiding data leakage from mid-week measurements.
    """
    return df_weekly.iloc[[-1]]