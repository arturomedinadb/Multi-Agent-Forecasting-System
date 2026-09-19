"""
Tests for date handling in schema_mapping.functions: reformatting dates to
a consistent ISO representation, detecting month-level vs day-level
granularity, and collapsing duplicate rows on a merge key.
"""

import pandas as pd
import pytest

from schema_mapping.functions import (
    _normalize_date_columns,
    _is_month_level_dates,
    _dedupe_by_key,
)


class TestNormalizeDateColumns:
    def test_iso_dates_are_unchanged(self):
        """dayfirst parsing must never be applied to unambiguous ISO dates -
        doing so swaps month and day whenever both are <=12."""
        df = pd.DataFrame({"date": ["2022-07-10", "2022-01-12", "2022-12-01"]})
        result = _normalize_date_columns(df.copy())
        assert list(result["date"]) == ["2022-07-10", "2022-01-12", "2022-12-01"]

    def test_slash_dates_are_parsed_day_first(self):
        """Slash-separated dates are treated as day-first. 15/04/2022 can
        only be day=15, month=4 (there is no 15th month), confirming this
        convention."""
        df = pd.DataFrame({"date": ["01/01/2022", "15/04/2022", "23/05/2022"]})
        result = _normalize_date_columns(df.copy())
        assert list(result["date"]) == ["2022-01-01", "2022-04-15", "2022-05-23"]

    def test_month_only_dates_get_a_day_component(self):
        """A bare year-month value (as monthly indicator data uses) gets a
        day component filled in when reformatted to ISO."""
        df = pd.DataFrame({"date": ["2015-01", "2015-02"]})
        result = _normalize_date_columns(df.copy())
        assert list(result["date"]) == ["2015-01-01", "2015-02-01"]

    def test_non_date_columns_are_untouched(self):
        df = pd.DataFrame(
            {"product_id": ["P0001", "P0002"], "date": ["2022-07-10"] * 2}
        )
        result = _normalize_date_columns(df.copy())
        assert list(result["product_id"]) == ["P0001", "P0002"]

    def test_promotion_start_and_end_date_columns_are_normalized_too(self):
        """Any column ending in '_date' is treated as a date field, not just
        the column literally named 'date'."""
        df = pd.DataFrame(
            {
                "promotion_start_date": ["01/03/2022"],
                "promotion_end_date": ["15/03/2022"],
            }
        )
        result = _normalize_date_columns(df.copy())
        assert result["promotion_start_date"].iloc[0] == "2022-03-01"
        assert result["promotion_end_date"].iloc[0] == "2022-03-15"

    def test_unparseable_values_are_left_as_is(self):
        """A value that can't be parsed as a date is kept as-is rather than
        silently blanked to None."""
        df = pd.DataFrame({"date": ["not-a-date"]})
        result = _normalize_date_columns(df.copy())
        assert result["date"].iloc[0] == "not-a-date"


class TestIsMonthLevelDates:
    def test_true_for_month_granularity_data(self):
        series = pd.Series(["2015-01-01", "2015-02-01", "2015-03-01"])
        assert _is_month_level_dates(series) is True

    def test_false_for_day_granularity_data(self):
        series = pd.Series(["2022-07-10", "2022-08-01", "2022-05-24"])
        assert _is_month_level_dates(series) is False

    def test_false_for_empty_or_all_null_series(self):
        series = pd.Series([None, None])
        assert _is_month_level_dates(series) is False


class TestDedupeByKey:
    def test_collapses_duplicate_dates_by_averaging(self):
        """Multiple rows sharing the same key value collapse into one,
        averaging numeric columns."""
        df = pd.DataFrame(
            {
                "date": ["2015-01-01", "2015-01-01", "2015-02-01"],
                "cpi_monthly": [100.0, 200.0, 150.0],
            }
        )
        result = _dedupe_by_key(df, keys=["date"])
        assert len(result) == 2
        jan = result.loc[result["date"] == "2015-01-01", "cpi_monthly"].iloc[0]
        assert jan == pytest.approx(150.0)  # mean of 100 and 200

    def test_noop_when_key_already_unique(self):
        df = pd.DataFrame({"date": ["2022-07-10", "2022-07-11"], "units_sold": [5, 7]})
        result = _dedupe_by_key(df, keys=["date"])
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), df.reset_index(drop=True)
        )

    def test_non_numeric_columns_keep_first_value(self):
        df = pd.DataFrame(
            {
                "date": ["2015-01-01", "2015-01-01"],
                "region_note": ["Ontario", "Quebec"],
                "cpi_monthly": [100.0, 200.0],
            }
        )
        result = _dedupe_by_key(df, keys=["date"])
        assert len(result) == 1
        assert result["region_note"].iloc[0] == "Ontario"
