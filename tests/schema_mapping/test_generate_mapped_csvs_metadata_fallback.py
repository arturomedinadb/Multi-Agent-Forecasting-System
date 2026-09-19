"""
Tests that run_generate_mapped_csvs recovers dataset metadata directly from
the session database when its source_metadata_json argument is empty,
unparseable, or covers fewer files than the session actually loaded.
"""

import json
import sqlite3

import pandas as pd

from schema_mapping.functions import run_generate_mapped_csvs

SESSION_ID = "metadata-fallback-test-session"


def _make_session_db(db_path, file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE agent_messages (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR,
            message_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    for file_path in file_paths:
        call = json.dumps(
            {
                "type": "function_call",
                "name": "load_and_describe_dataset",
                "arguments": json.dumps({"file_path": file_path}),
            }
        )
        output = json.dumps(
            {
                "type": "function_call_output",
                "output": json.dumps({"file_path": file_path, "columns": ["a"]}),
            }
        )
        conn.execute(
            "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
            (SESSION_ID, call),
        )
        conn.execute(
            "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
            (SESSION_ID, output),
        )
    conn.commit()
    conn.close()


class TestGenerateMappedCsvsMetadataFallback:
    def test_recovers_from_session_db_when_source_metadata_json_is_truncated(
        self, tmp_path, monkeypatch
    ):
        src_path = tmp_path / "x.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(src_path, index=False)
        file_path = str(src_path).replace("\\", "/")

        db_path = tmp_path / "workflow_sessions.db"
        _make_session_db(str(db_path), file_path)
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        mapping_plan = json.dumps(
            {
                "mappings": [
                    {
                        "source_file": file_path,
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

        # Simulates a truncated/malformed source_metadata_json argument.
        truncated_metadata_json = '[{"file_path":"' + file_path + '","columns":["a"'

        result = json.loads(
            run_generate_mapped_csvs(
                truncated_metadata_json, mapping_plan, str(tmp_path)
            )
        )

        assert result.get("error") is None
        assert len(result["outputs"]) == 1
        assert result["outputs"][0]["columns"] == ["date"]

    def test_recovers_from_session_db_when_source_metadata_json_is_empty(
        self, tmp_path, monkeypatch
    ):
        src_path = tmp_path / "x.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(src_path, index=False)
        file_path = str(src_path).replace("\\", "/")

        db_path = tmp_path / "workflow_sessions.db"
        _make_session_db(str(db_path), file_path)
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        mapping_plan = json.dumps(
            {
                "mappings": [
                    {
                        "source_file": file_path,
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

        result = json.loads(run_generate_mapped_csvs("", mapping_plan, str(tmp_path)))

        assert len(result["outputs"]) == 1

    def test_uses_session_db_when_argument_only_covers_a_subset_of_files(
        self, tmp_path, monkeypatch
    ):
        """When source_metadata_json parses fine but describes fewer files
        than the session loaded, the database's more complete record is used
        instead of mapping only the subset named in the argument."""
        first_path = str(tmp_path / "first.csv").replace("\\", "/")
        second_path = str(tmp_path / "second.csv").replace("\\", "/")
        pd.DataFrame({"a": [1, 2]}).to_csv(first_path, index=False)
        pd.DataFrame({"b": [3, 4]}).to_csv(second_path, index=False)

        db_path = tmp_path / "workflow_sessions.db"
        _make_session_db(str(db_path), [first_path, second_path])
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        # Names only first_path, omitting second_path.
        partial_metadata_json = json.dumps(
            [{"file_path": first_path, "columns": ["a"]}]
        )
        mapping_plan = json.dumps(
            {
                "mappings": [
                    {
                        "source_file": first_path,
                        "mappings": [
                            {
                                "source_column": "a",
                                "target_column": "date",
                                "confidence": 0.9,
                                "reasoning": "r",
                            }
                        ],
                    },
                    {
                        "source_file": second_path,
                        "mappings": [
                            {
                                "source_column": "b",
                                "target_column": "product_id",
                                "confidence": 0.9,
                                "reasoning": "r",
                            }
                        ],
                    },
                ]
            }
        )

        result = json.loads(
            run_generate_mapped_csvs(partial_metadata_json, mapping_plan, str(tmp_path))
        )

        assert len(result["outputs"]) == 2
