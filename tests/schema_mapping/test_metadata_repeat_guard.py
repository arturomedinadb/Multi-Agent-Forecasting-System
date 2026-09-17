"""
Tests for get_all_dataset_metadata()'s repeat-call guard: a session that has
already received the full metadata payload once must get a short pointer on
later calls instead of the full payload again, since re-dumping it repeatedly
is what exhausts the model's context window on longer runs.
"""
import asyncio
import json
import sqlite3

from schema_mapping.functions import get_all_dataset_metadata


SESSION_ID = "repeat-guard-test-session"


def _make_db(db_path, num_files):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE agent_messages (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR,
            message_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    for i in range(num_files):
        call = json.dumps({
            "type": "function_call",
            "name": "load_and_describe_dataset",
            "arguments": json.dumps({"file_path": f"file{i}.csv"}),
        })
        conn.execute("INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)", (SESSION_ID, call))
        output = json.dumps({
            "type": "function_call_output",
            "output": json.dumps({"file_path": f"file{i}.csv", "columns": ["a", "b"]}),
        })
        conn.execute("INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)", (SESSION_ID, output))
    conn.commit()
    conn.close()


def _invoke():
    async def _run():
        raw = await get_all_dataset_metadata.on_invoke_tool(None, "{}")
        return json.loads(raw)

    return asyncio.run(_run())


class TestMetadataRepeatGuard:
    def test_second_call_returns_short_pointer_instead_of_full_payload(self, tmp_path, monkeypatch):
        db_path = tmp_path / "workflow_sessions.db"
        _make_db(str(db_path), num_files=5)
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        first = _invoke()
        assert first["status"] == "success"
        assert isinstance(first["metadata"], list)
        assert len(first["metadata"]) == 5

        second = _invoke()
        assert second["status"] == "success"
        assert second["dataset_count"] == 5
        assert second["metadata"] == "ALREADY_RETRIEVED"
        assert "note" in second

    def test_different_sessions_are_independent(self, tmp_path, monkeypatch):
        db_path = tmp_path / "workflow_sessions.db"
        _make_db(str(db_path), num_files=3)
        monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))

        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        first_session_result = _invoke()
        assert first_session_result["metadata"] != "ALREADY_RETRIEVED"

        other_session = SESSION_ID + "-other"
        monkeypatch.setenv("CURRENT_SESSION_ID", other_session)
        other_session_result = _invoke()
        assert other_session_result["metadata"] != "ALREADY_RETRIEVED"
