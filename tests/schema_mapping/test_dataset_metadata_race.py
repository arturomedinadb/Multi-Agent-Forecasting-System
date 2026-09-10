"""
Tests for get_all_dataset_metadata()'s self-verifying retry: it compares how
many load_and_describe_dataset calls were issued against how many results
have actually landed in the session database, and briefly retries when a
result hasn't been persisted yet rather than returning a short list as if
it were complete.
"""
import asyncio
import json
import sqlite3
import threading
import time

import pytest

from schema_mapping.functions import get_all_dataset_metadata, _query_dataset_metadata


SESSION_ID = "test-session"


def _make_db(db_path, num_calls, num_results):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE agent_messages (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR,
            message_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    for i in range(num_calls):
        _insert(conn, is_call=True, file_name=f"file{i}.csv")
    for i in range(num_results):
        _insert(conn, is_call=False, file_name=f"file{i}.csv")
    conn.commit()
    conn.close()


def _insert(conn, is_call, file_name):
    if is_call:
        data = json.dumps({
            "type": "function_call",
            "name": "load_and_describe_dataset",
            "arguments": json.dumps({"file_path": file_name}),
        })
    else:
        output = json.dumps({"file_path": file_name, "columns": ["a", "b"]})
        data = json.dumps({"type": "function_call_output", "output": output})
    conn.execute("INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)", (SESSION_ID, data))


def _insert_later(db_path, file_name, delay):
    def _run():
        time.sleep(delay)
        conn = sqlite3.connect(db_path)
        _insert(conn, is_call=False, file_name=file_name)
        conn.commit()
        conn.close()

    threading.Thread(target=_run, daemon=True).start()


def _invoke_get_all_dataset_metadata():
    async def _run():
        raw = await get_all_dataset_metadata.on_invoke_tool(None, "{}")
        return json.loads(raw)

    return asyncio.run(_run())


class TestQueryDatasetMetadata:
    def test_counts_calls_and_results_separately(self, tmp_path):
        db_path = str(tmp_path / "workflow_sessions.db")
        _make_db(db_path, num_calls=10, num_results=9)
        metadata_list, calls_issued = _query_dataset_metadata(db_path, SESSION_ID)
        assert calls_issued == 10
        assert len(metadata_list) == 9


class TestGetAllDatasetMetadataRetry:
    def test_returns_immediately_when_all_results_already_present(self, tmp_path, monkeypatch):
        db_path = tmp_path / "workflow_sessions.db"
        _make_db(str(db_path), num_calls=5, num_results=5)
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        start = time.time()
        result = _invoke_get_all_dataset_metadata()
        elapsed = time.time() - start

        assert result["dataset_count"] == 5
        assert elapsed < 0.5  # no retry needed, should be near-instant

    def test_waits_for_a_result_that_lands_late(self, tmp_path, monkeypatch):
        """Simulates the real race: the last call's result hasn't been
        persisted yet when this is called. It should retry and pick it up
        rather than returning a short list."""
        db_path = tmp_path / "workflow_sessions.db"
        _make_db(str(db_path), num_calls=10, num_results=9)
        _insert_later(str(db_path), file_name="file9.csv", delay=1.2)

        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        result = _invoke_get_all_dataset_metadata()

        assert result["dataset_count"] == 10

    def test_gives_up_gracefully_when_a_result_never_arrives(self, tmp_path, monkeypatch):
        """If a call's result genuinely never lands (e.g. that call itself
        failed with no matching output), this must still return the
        partial list it does have rather than hanging indefinitely."""
        db_path = tmp_path / "workflow_sessions.db"
        _make_db(str(db_path), num_calls=10, num_results=9)

        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        result = _invoke_get_all_dataset_metadata()

        assert result["dataset_count"] == 9
        assert result["status"] == "success"
