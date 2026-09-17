"""
Tests for the evaluator gates: evaluate_column_mapping_agent must refuse to
run if generate_mapped_csvs hasn't actually succeeded yet this session, and
evaluate_data_integration_agent must refuse to run if
merge_mapped_csvs_to_target hasn't. Both guard against an agent skipping its
required tool calls and jumping straight to self-evaluation.
"""
import asyncio
import json

import schema_mapping.functions as functions_module
from schema_mapping.functions import evaluate_column_mapping_agent, evaluate_data_integration_agent


SESSION_ID = "evaluator-gate-test-session"


def _invoke(tool, **kwargs):
    async def _run():
        result = tool.on_invoke_tool(None, json.dumps(kwargs))
        if hasattr(result, "__await__"):
            result = await result
        return json.loads(result)

    return asyncio.run(_run())


class TestColumnMappingEvaluatorGate:
    def test_blocked_when_generate_mapped_csvs_never_succeeded(self, monkeypatch):
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)

        result = _invoke(
            evaluate_column_mapping_agent,
            agent_input="x",
            agent_output="y",
            mapping_plan_json="{}",
            target_schema_json="{}",
        )

        assert result["status"] == "blocked"
        assert "generate_mapped_csvs" in result["error"]

    def test_proceeds_when_generate_mapped_csvs_already_succeeded(self, monkeypatch):
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.delenv("DEEPEVAL_API_KEY", raising=False)
        functions_module._generate_mapped_csvs_cache[SESSION_ID] = json.dumps({"outputs": [{"output_path": "x"}]})

        result = _invoke(
            evaluate_column_mapping_agent,
            agent_input="x",
            agent_output="y",
            mapping_plan_json="{}",
            target_schema_json="{}",
        )

        # No API key configured, so it takes the "skipped" path rather than
        # "blocked" - the point is it got past the gate.
        assert result["status"] != "blocked"


class TestDataIntegrationEvaluatorGate:
    def test_blocked_when_merge_never_succeeded(self, monkeypatch):
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)

        result = _invoke(
            evaluate_data_integration_agent,
            agent_input="x",
            agent_output="y",
            source_row_count=10,
            final_row_count=10,
        )

        assert result["status"] == "blocked"
        assert "merge_mapped_csvs_to_target" in result["error"]

    def test_proceeds_when_merge_already_succeeded(self, monkeypatch):
        monkeypatch.setenv("CURRENT_SESSION_ID", SESSION_ID)
        monkeypatch.delenv("DEEPEVAL_API_KEY", raising=False)
        functions_module._merge_mapped_csvs_cache[SESSION_ID] = True

        result = _invoke(
            evaluate_data_integration_agent,
            agent_input="x",
            agent_output="y",
            source_row_count=10,
            final_row_count=10,
        )

        assert result["status"] != "blocked"
