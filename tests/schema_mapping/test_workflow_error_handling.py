"""
Tests that run_full_workflow reports a model/API failure (e.g. an OpenAI
context_length_exceeded error) as a structured error result, the same way
every other failure path in this workflow does, instead of letting the
exception crash the whole process.
"""

import asyncio

from agents import Runner

import schema_mapping.run_workflow as run_workflow_module


class TestRunFullWorkflowErrorHandling:
    def test_runner_failure_returns_structured_error_instead_of_raising(
        self, tmp_path, monkeypatch
    ):
        async def _raise(*args, **kwargs):
            raise Exception("Error code: 400 - context_length_exceeded: input too long")

        monkeypatch.setattr(Runner, "run", _raise)

        async def _run():
            return await run_workflow_module.run_full_workflow(
                source_files=[],
                row_limit=10,
                output_dir=str(tmp_path),
            )

        result = asyncio.run(_run())

        assert result["status"] == "error"
        assert "context_length_exceeded" in result["error"]
        assert result["output_dir"] == str(tmp_path)
