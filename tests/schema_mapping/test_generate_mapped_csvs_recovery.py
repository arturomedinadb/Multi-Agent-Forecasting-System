"""
Tests for run_generate_mapped_csvs recovering from a malformed
mappings_json, rather than failing the whole tool call: a complete document
followed by extra appended content, and a document cut off partway through.
"""

import json

import pandas as pd

from schema_mapping.functions import (
    run_generate_mapped_csvs,
    _salvage_truncated_json,
)


class TestGenerateMappedCsvsRecoversFromExtraData:
    def test_salvages_a_valid_mapping_document_with_trailing_junk(self, tmp_path):
        src_path = tmp_path / "x.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(src_path, index=False)

        source_metadata = json.dumps(
            {
                "metadata": [
                    {"file_path": str(src_path).replace("\\", "/"), "columns": ["a"]}
                ]
            }
        )

        valid_mapping_doc = json.dumps(
            {
                "mappings": [
                    {
                        "source_file": "x.csv",
                        "mappings": [
                            {
                                "source_column": "a",
                                "target_column": "date",
                                "confidence": 0.9,
                                "reasoning": "r",
                            }
                        ],
                    }
                ]
            }
        )
        # A stray extra object appended after a complete, otherwise-valid
        # document.
        malformed_mappings = (
            valid_mapping_doc + ',{"stray": "extra object appended by mistake"}'
        )

        result = json.loads(
            run_generate_mapped_csvs(source_metadata, malformed_mappings, str(tmp_path))
        )

        assert result.get("error") is None
        assert len(result["outputs"]) == 1
        assert result["outputs"][0]["columns"] == ["date"]

        mapped_csv = tmp_path / "x_mapped.csv"
        assert mapped_csv.exists()


class TestSalvageTruncatedJson:
    def test_recovers_a_document_cut_off_near_the_end(self):
        full = json.dumps(
            {
                "mappings": [
                    {"source_file": f"{name}.csv", "mappings": []}
                    for name in ("a", "b", "c")
                ]
            }
        )
        recovered = _salvage_truncated_json(full[:-2])

        assert recovered is not None
        assert [m["source_file"] for m in recovered["mappings"]] == [
            "a.csv",
            "b.csv",
            "c.csv",
        ]

    def test_keeps_the_entries_that_arrived_when_cut_off_mid_document(self):
        full = json.dumps(
            {
                "mappings": [
                    {"source_file": f"{name}.csv", "mappings": []}
                    for name in ("a", "b", "c")
                ]
            }
        )
        recovered = _salvage_truncated_json(full[: len(full) // 2])

        assert recovered is not None
        assert len(recovered["mappings"]) >= 1

    def test_returns_none_when_nothing_can_be_recovered(self):
        assert _salvage_truncated_json('{"mappings": [') is None


class TestGenerateMappedCsvsRecoversFromTruncation:
    def test_truncated_mappings_still_produce_output(self, tmp_path):
        src_path = tmp_path / "x.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(src_path, index=False)

        source_metadata = json.dumps(
            {
                "metadata": [
                    {"file_path": str(src_path).replace("\\", "/"), "columns": ["a"]}
                ]
            }
        )
        full_mappings = json.dumps(
            {
                "mappings": [
                    {
                        "source_file": "x.csv",
                        "mappings": [
                            {
                                "source_column": "a",
                                "target_column": "date",
                                "confidence": 0.9,
                                "reasoning": "r",
                            }
                        ],
                    }
                ]
            }
        )

        result = json.loads(
            run_generate_mapped_csvs(source_metadata, full_mappings[:-2], str(tmp_path))
        )

        assert result.get("error") is None
        assert len(result["outputs"]) == 1
