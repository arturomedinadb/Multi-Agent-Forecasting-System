"""
Tests for generate_mapped_csvs' repeat-call guard: once it has succeeded for
a session, a redundant repeat call returns the cached result instead of
reprocessing its arguments again - protecting against a later call carrying
a truncated or otherwise malformed copy of the same (large) arguments.
"""
import asyncio
import json

import pandas as pd

from schema_mapping.functions import generate_mapped_csvs


SESSION_ID = "generate-mapped-csvs-repeat-guard-session"


def _invoke(**kwargs):
    async def _run():
        result = generate_mapped_csvs.on_invoke_tool(None, json.dumps(kwargs))
        if hasattr(result, "__await__"):
            result = await result
        return json.loads(result)

    return asyncio.run(_run())


class TestGenerateMappedCsvsRepeatGuard:
    def test_second_call_returns_cached_result_even_with_malformed_arguments(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)

        src_path = tmp_path / "x.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(src_path, index=False)

        source_metadata = json.dumps({
            "metadata": [{"file_path": str(src_path).replace("\\", "/"), "columns": ["a"]}]
        })
        mapping_plan = json.dumps({
            "mappings": [{"source_file": "x.csv", "mappings": [
                {"source_column": "a", "target_column": "date", "confidence": 0.9, "reasoning": "r"}
            ]}]
        })

        first = _invoke(source_metadata_json=source_metadata, mappings_json=mapping_plan, output_dir=str(tmp_path))
        assert len(first["outputs"]) == 1

        # A redundant second call with genuinely broken JSON must still get
        # back the known-good first result, not a fresh (failing) attempt.
        second = _invoke(source_metadata_json="not valid json {{{", mappings_json="also not valid", output_dir=str(tmp_path))
        assert second == first

    def test_failed_first_call_does_not_get_cached(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)

        failed = _invoke(source_metadata_json="not valid json {{{", mappings_json="also not valid", output_dir=str(tmp_path))
        assert failed["outputs"] == []

        src_path = tmp_path / "x.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(src_path, index=False)
        source_metadata = json.dumps({
            "metadata": [{"file_path": str(src_path).replace("\\", "/"), "columns": ["a"]}]
        })
        mapping_plan = json.dumps({
            "mappings": [{"source_file": "x.csv", "mappings": [
                {"source_column": "a", "target_column": "date", "confidence": 0.9, "reasoning": "r"}
            ]}]
        })
        second = _invoke(source_metadata_json=source_metadata, mappings_json=mapping_plan, output_dir=str(tmp_path))
        assert len(second["outputs"]) == 1
