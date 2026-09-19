"""
Tests for merge_mapped_csvs_to_target's null-output guard: it must fail
loudly instead of reporting "Success" when the merge produces a dataframe
with the correct target schema column names but no real data in them (e.g.
because it was handed raw/unmapped source files).
"""

import asyncio
import json

import pandas as pd
import pytest

from schema_mapping.functions import merge_mapped_csvs_to_target

TARGET_SCHEMA = {
    "properties": {
        "date": {"type": "string"},
        "product_id": {"type": "string"},
        "store_id": {"type": "string"},
        "units_sold": {"type": "number"},
        "unit_net_price": {"type": "number"},
    }
}


def _invoke(**kwargs):
    async def _run():
        result = merge_mapped_csvs_to_target.on_invoke_tool(None, json.dumps(kwargs))
        if hasattr(result, "__await__"):
            result = await result
        return json.loads(result)

    return asyncio.run(_run())


class TestMergeNullGuard:
    def test_fails_when_inputs_are_raw_unmapped_files(self, tmp_path):
        """Feeding raw source files (columns that never match any target
        field name) must not silently succeed with an all-null output."""
        raw_path = tmp_path / "transaction_like_synth.csv"
        pd.DataFrame(
            {
                "transaction_date": ["2022-01-01", "2022-01-02"],
                "prod_code": ["P1", "P2"],
            }
        ).to_csv(raw_path, index=False)

        output_path = tmp_path / "merged_output.csv"
        result = _invoke(
            # forward slashes: the tool's JSON pre-cleaning turns backslashes
            # into slashes, which would double up an already-forward-slash
            # path's separators if we passed a native Windows (backslash) path
            mapped_outputs_json=json.dumps(
                {"outputs": [{"output_path": str(raw_path).replace("\\", "/")}]}
            ),
            target_schema_json=json.dumps(TARGET_SCHEMA),
            output_path=str(output_path),
        )

        assert result["status"] == "Failed"
        assert "null" in result["error"].lower()
        assert not output_path.exists()

    def test_succeeds_when_inputs_are_properly_mapped(self, tmp_path):
        """A mapped CSV that already uses target column names merges and
        writes normally."""
        mapped_path = tmp_path / "transaction_mapped.csv"
        pd.DataFrame(
            {
                "date": ["2022-01-01", "2022-01-02"],
                "product_id": ["P1", "P2"],
                "store_id": ["S1", "S1"],
                "units_sold": [10, 20],
            }
        ).to_csv(mapped_path, index=False)

        output_path = tmp_path / "merged_output.csv"
        result = _invoke(
            mapped_outputs_json=json.dumps(
                {"outputs": [{"output_path": str(mapped_path).replace("\\", "/")}]}
            ),
            target_schema_json=json.dumps(TARGET_SCHEMA),
            output_path=str(output_path),
        )

        assert result["status"] == "Success"
        assert output_path.exists()
        assert result["rows"] == 2

    def test_fails_on_zero_rows(self, tmp_path):
        mapped_path = tmp_path / "empty_mapped.csv"
        pd.DataFrame({"date": [], "product_id": []}).to_csv(mapped_path, index=False)

        output_path = tmp_path / "merged_output.csv"
        result = _invoke(
            mapped_outputs_json=json.dumps(
                {"outputs": [{"output_path": str(mapped_path).replace("\\", "/")}]}
            ),
            target_schema_json=json.dumps(TARGET_SCHEMA),
            output_path=str(output_path),
        )

        assert result["status"] == "Failed"
        assert "0 rows" in result["error"]
