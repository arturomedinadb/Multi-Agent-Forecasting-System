"""
Tests for feature_functions.py helpers: dropping unusable group_by columns,
the handle_errors decorator's failure behavior, and PromotionFeatureConfig's
date-based features.
"""
import warnings

import pandas as pd
import pytest

from ai_forecasting_agents.demand_forecasting.tools.feature_functions import (
    _usable_group_by,
    handle_errors,
    create_lag_features,
    create_rolling_features,
    create_promotion_features,
)
from ai_forecasting_agents.demand_forecasting.schemas.feature_models import (
    LagFeatureConfig,
    RollingFeatureConfig,
    PromotionFeatureConfig,
)


class TestUsableGroupBy:
    def test_drops_all_null_column(self):
        df = pd.DataFrame({"product_id": ["P1", "P2"], "store_id": [None, None]})
        with pytest.warns(UserWarning, match="all-null"):
            result = _usable_group_by(df, ["product_id", "store_id"])
        assert result == ["product_id"]

    def test_keeps_columns_with_some_data(self):
        df = pd.DataFrame({"product_id": ["P1", "P2"], "store_id": ["S1", None]})
        result = _usable_group_by(df, ["product_id", "store_id"])
        assert result == ["product_id", "store_id"]

    def test_returns_none_when_group_by_is_falsy(self):
        df = pd.DataFrame({"a": [1]})
        assert _usable_group_by(df, None) is None
        assert _usable_group_by(df, []) is None

    def test_returns_none_when_every_key_is_all_null(self):
        df = pd.DataFrame({"store_id": [None, None]})
        with pytest.warns(UserWarning):
            result = _usable_group_by(df, ["store_id"])
        assert result is None


class TestHandleErrorsDecorator:
    def test_returns_input_df_unchanged_on_failure(self):
        """A failing feature function returns its own input df unchanged,
        so a step that fails doesn't break the ones after it in a chain."""
        @handle_errors
        def always_fails(df, config):
            raise ValueError("simulated failure")

        df = pd.DataFrame({"a": [1, 2, 3]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = always_fails(df, None)

        assert result is df
        assert result is not None
        assert any("simulated failure" in str(w.message) for w in caught)

    def test_passes_through_successful_result(self):
        @handle_errors
        def doubles_a_column(df, config):
            df = df.copy()
            df["a"] = df["a"] * 2
            return df

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = doubles_a_column(df, None)
        assert list(result["a"]) == [2, 4, 6]


class TestGroupedFeaturesWithAllNullGroupBy:
    def test_create_lag_features_does_not_crash(self):
        df = pd.DataFrame({
            "product_id": ["P1", "P1", "P2"],
            "store_id": [None, None, None],
            "units_sold": [10, 20, 30],
        })
        config = LagFeatureConfig(target_column="units_sold", lags=[1], group_by=["product_id", "store_id"])
        with pytest.warns(UserWarning, match="all-null"):
            result = create_lag_features(df, config)
        assert result is not None
        assert "units_sold_lag_1" in result.columns

    def test_create_rolling_features_does_not_crash(self):
        """Grouping by a column that's entirely null must not crash the
        rolling-window computation."""
        df = pd.DataFrame({
            "product_id": ["P1", "P1", "P2"],
            "store_id": [None, None, None],
            "units_sold": [10, 20, 30],
        })
        config = RollingFeatureConfig(
            target_column="units_sold", windows=[7], functions=["mean"], group_by=["product_id", "store_id"]
        )
        with pytest.warns(UserWarning, match="all-null"):
            result = create_rolling_features(df, config)
        assert result is not None
        assert "units_sold_rolling_mean_7" in result.columns


class TestPromotionFeatureConfig:
    def test_has_date_col_field(self):
        config = PromotionFeatureConfig()
        assert hasattr(config, "date_col")

    def test_days_since_promo_does_not_crash(self):
        df = pd.DataFrame({
            "date": ["2022-07-10"],
            "promotion_end_date": ["2022-07-01"],
        })
        config = PromotionFeatureConfig(
            promo_end_col="promotion_end_date",
            date_col="date",
            features=["days_since_promo"],
        )
        result = create_promotion_features(df, config)
        assert result is not None
        assert "days_since_promo" in result.columns
        assert result["days_since_promo"].iloc[0] == 9
