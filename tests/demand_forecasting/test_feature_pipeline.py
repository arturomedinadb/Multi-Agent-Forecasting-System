"""
Tests for process_feature_engineering_pipeline's failure handling: an
unreadable input file must produce a structured result describing the cause,
and a path the caller could not supply correctly is recovered from the paths
the pipeline runner recorded in the environment.
"""

import asyncio
import json

import pandas as pd

from ai_forecasting_agents.demand_forecasting.tools.feature_functions import (
    process_feature_engineering_pipeline,
)

CONFIG = {
    "lag_features": {
        "target_column": "units_sold",
        "lags": [1],
        "group_by": ["store_id"],
    },
    "time_features": {"date_column": "date", "features": ["year", "month"]},
}


def _invoke(**kwargs):
    async def _run():
        result = process_feature_engineering_pipeline.on_invoke_tool(
            None, json.dumps(kwargs)
        )
        if hasattr(result, "__await__"):
            result = await result
        return result

    return asyncio.run(_run())


def _write_input(path):
    pd.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02"],
            "store_id": ["S1", "S1"],
            "units_sold": [5, 7],
        }
    ).to_csv(path, index=False)


class TestFeaturePipelineErrorHandling:
    def test_unreadable_input_returns_structured_error(self, tmp_path, monkeypatch):
        """The error path must not raise while building its own result - it
        reports the cause instead of surfacing an unrelated internal error."""
        monkeypatch.delenv("FEATURE_ENGINEERING_INPUT_FILE", raising=False)

        result = _invoke(
            input_file=str(tmp_path / "missing.csv"),
            output_file=str(tmp_path / "out.csv"),
            config=CONFIG,
        )

        assert result.success is False
        assert result.errors
        assert "missing.csv" in result.errors[0]


class TestFeaturePipelinePathFallback:
    def test_recovers_path_recorded_by_the_runner(self, tmp_path, monkeypatch):
        real_input = tmp_path / "real_input.csv"
        _write_input(real_input)
        real_output = tmp_path / "engineered.csv"

        monkeypatch.setenv("FEATURE_ENGINEERING_INPUT_FILE", str(real_input))
        monkeypatch.setenv("FEATURE_ENGINEERING_OUTPUT_FILE", str(real_output))

        result = _invoke(
            input_file=str(tmp_path / "guessed_wrong.csv"),
            output_file=str(tmp_path / "guessed_wrong_out.csv"),
            config=CONFIG,
        )

        assert result.success is True
        assert result.features_created > 0
        assert real_output.exists()

    def test_correct_path_is_used_as_given(self, tmp_path, monkeypatch):
        given_input = tmp_path / "given.csv"
        _write_input(given_input)
        given_output = tmp_path / "given_out.csv"

        monkeypatch.setenv(
            "FEATURE_ENGINEERING_INPUT_FILE", str(tmp_path / "other.csv")
        )

        result = _invoke(
            input_file=str(given_input),
            output_file=str(given_output),
            config=CONFIG,
        )

        assert result.success is True
        assert given_output.exists()


class TestAnalyzeDataStructureCache:
    """Repeated analysis of the same file replays the first result, so an
    orchestrator that keeps re-analysing cannot spend its whole turn budget
    here instead of reaching the later pipeline steps. The replayed value
    keeps the same shape, because callers validate it against a schema."""

    def _analyse(self, feature_functions, path):
        async def _run():
            result = feature_functions.analyze_data_structure.on_invoke_tool(
                None, json.dumps({"input_file": str(path)})
            )
            if hasattr(result, "__await__"):
                result = await result
            return result

        return asyncio.run(_run())

    def test_repeat_analysis_returns_the_same_result(self, tmp_path):
        from ai_forecasting_agents.demand_forecasting.tools import feature_functions

        input_path = tmp_path / "analyse_me.csv"
        _write_input(input_path)
        feature_functions._analysis_cache.clear()

        first = self._analyse(feature_functions, input_path)
        second = self._analyse(feature_functions, input_path)

        assert "columns" in first
        assert second == first

        feature_functions._analysis_cache.clear()

    def test_same_file_spelled_differently_hits_one_cache_entry(self, tmp_path):
        from ai_forecasting_agents.demand_forecasting.tools import feature_functions

        input_path = tmp_path / "analyse_me.csv"
        _write_input(input_path)
        feature_functions._analysis_cache.clear()

        self._analyse(feature_functions, input_path)
        # Same file, different spelling of the same path.
        self._analyse(feature_functions, str(input_path).replace("\\", "/"))

        assert len(feature_functions._analysis_cache) == 1

        feature_functions._analysis_cache.clear()
