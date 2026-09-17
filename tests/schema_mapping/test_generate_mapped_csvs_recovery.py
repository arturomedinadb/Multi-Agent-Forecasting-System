"""
Tests for run_generate_mapped_csvs recovering from a malformed mappings_json:
a complete, valid mapping document followed by extra content the LLM
appended afterward (e.g. a stray second object tacked onto the end), rather
than failing the whole tool call.
"""
import json

import pandas as pd

from schema_mapping.functions import run_generate_mapped_csvs


class TestGenerateMappedCsvsRecoversFromExtraData:
    def test_salvages_a_valid_mapping_document_with_trailing_junk(self, tmp_path):
        src_path = tmp_path / "x.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(src_path, index=False)

        source_metadata = json.dumps({
            "metadata": [{"file_path": str(src_path).replace("\\", "/"), "columns": ["a"]}]
        })

        valid_mapping_doc = json.dumps({
            "mappings": [
                {
                    "source_file": "x.csv",
                    "mappings": [
                        {"source_column": "a", "target_column": "date", "confidence": 0.9, "reasoning": "r"}
                    ],
                }
            ]
        })
        # Simulate the LLM appending a stray extra object after a complete,
        # otherwise-valid document - the real failure seen in a live run.
        malformed_mappings = valid_mapping_doc + ',{"stray": "extra object appended by mistake"}'

        result = json.loads(run_generate_mapped_csvs(source_metadata, malformed_mappings, str(tmp_path)))

        assert result.get("error") is None
        assert len(result["outputs"]) == 1
        assert result["outputs"][0]["columns"] == ["date"]

        mapped_csv = tmp_path / "x_mapped.csv"
        assert mapped_csv.exists()
