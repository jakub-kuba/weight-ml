"""
Unit tests for loader.py.

Rule: test the cleaning logic without hitting Google Sheets.
We use small synthetic DataFrames — tests are fast and deterministic.
"""

import pandas as pd
import pytest

from src.data.loader import clean_weight_column


def make_df(weights: list) -> pd.DataFrame:
    """Helper — build a minimal DataFrame with a Weight column."""
    return pd.DataFrame({"Weight": weights})


class TestCleanWeightColumn:
    def test_replaces_comma_with_dot(self):
        df = make_df(["85,4", "86,1"])
        result = clean_weight_column(df)
        assert result["Weight"].iloc[0] == pytest.approx(85.4)

    def test_strips_whitespace(self):
        df = make_df([" 85.4 ", "86.1"])
        result = clean_weight_column(df)
        assert result["Weight"].iloc[0] == pytest.approx(85.4)

    def test_interpolates_missing_values(self):
        df = make_df([84.0, None, 86.0])
        result = clean_weight_column(df)
        # linear interpolation: (84 + 86) / 2 = 85
        assert result["Weight"].iloc[1] == pytest.approx(85.0)

    def test_converts_to_float(self):
        df = make_df(["85.4", "86.1"])
        result = clean_weight_column(df)
        assert result["Weight"].dtype == float

    def test_handles_mixed_formats(self):
        df = make_df(["85,4", " 86.1 ", "87,0"])
        result = clean_weight_column(df)
        expected = [85.4, 86.1, 87.0]
        assert list(result["Weight"]) == pytest.approx(expected)