from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from .agents.definitions import (
    column_mapping_agent,
    data_integration_agent,
    data_prep_agent,
)
from .agents.definitions_eval_agents import (
    column_mapping_evaluation_agent,
    data_integration_evaluation_agent,
    data_prep_evaluation_agent,
)
from .prompts.factory import get_renderer
from .schemas.models import DemandForecastingRecord

try:
    from agents import RawResponsesStreamEvent, Runner, trace
    from openai import OpenAIError
    from openai.types.responses import (
        ResponseContentPartDoneEvent,
        ResponseTextDeltaEvent,
    )

    AGENTS_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    RawResponsesStreamEvent = None  # type: ignore
    Runner = None  # type: ignore
    trace = None  # type: ignore
    ResponseContentPartDoneEvent = None  # type: ignore
    ResponseTextDeltaEvent = None  # type: ignore
    OpenAIError = RuntimeError  # type: ignore
    AGENTS_SDK_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if not AGENTS_SDK_AVAILABLE:

    @contextmanager
    def _noop_trace(_: str):
        yield

    trace = _noop_trace  # type: ignore


def _extract_json_candidates(text: str) -> List[str]:
    """Return JSON snippets from fenced blocks or raw text."""
    import re

    pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text)
    candidates: List[str] = []
    for match in matches:
        cleaned = "\n".join(
            line for line in match.splitlines() if not line.strip().startswith("//")
        )
        if cleaned:
            candidates.append(cleaned.strip())

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    return candidates


def _parse_json_value(
    text: str,
    *,
    expect_type: type | Sequence[type] | None = None,
    error_message: str | None = None,
    predicate: Callable[[Any], bool] | None = None,
) -> Any:
    """Attempt to parse text into JSON, optionally validating the result type."""
    candidates = _extract_json_candidates(text)
    expected = expect_type if expect_type is not None else ()
    expected_tuple: tuple[type, ...]
    if isinstance(expected, type):
        expected_tuple = (expected,)
    else:
        expected_tuple = tuple(expected) if expected else ()

    decoder = json.JSONDecoder()

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
        if expected_tuple and not isinstance(value, expected_tuple):
            continue
        if predicate and not predicate(value):
            continue
        return value

    raise ValueError(error_message or "Unable to parse expected JSON payload.")


@dataclass
class StageArtifacts:
    source_metadata_path: Path | None = None
    mapping_plan_path: Path | None = None
    mapping_manifest_path: Path | None = None
    final_dataset_path: Path | None = None
    data_prep_eval_path: Path | None = None
    column_mapping_eval_path: Path | None = None
    data_integration_eval_path: Path | None = None
    source_total_rows: int = 0  # Total rows from all source CSVs


@dataclass
class RunContext:
    source_files: List[str]
    row_limit: int
    output_root: Path
    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    artifacts: StageArtifacts = field(default_factory=StageArtifacts)

    def run_directory(self) -> Path:
        return self.output_root / "runs" / self.run_id

    def mapped_directory(self) -> Path:
        return self.run_directory() / "mapped"

    def evaluations_directory(self) -> Path:
        return self.output_root / "evaluations"


class AgentExecutor:
    """Thin wrapper around the Agents SDK to stream output and return RunResult."""

    async def run(
        self,
        *,
        agent,
        messages: Iterable[Dict[str, Any]],
        trace_label: str,
    ):
        if not AGENTS_SDK_AVAILABLE:
            raise RuntimeError(
                "OpenAI Agents SDK is required. Install with 'pip install openai-agents>=0.3.0'."
            )

        with trace(trace_label):
            streamed = Runner.run_streamed(agent, input=list(messages))
            async for event in streamed.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                payload = event.data
                if isinstance(payload, ResponseTextDeltaEvent):
                    print(payload.delta, end="", flush=True)
                elif isinstance(payload, ResponseContentPartDoneEvent):
                    print()
            return streamed


def _truncate_for_evaluation(text: str, max_chars: int = 4000) -> str:
    """Truncate long text for evaluation to avoid token limits."""
    if len(text) <= max_chars:
        return text
    
    # Keep first 70% and last 30% of the allowed characters
    head_chars = int(max_chars * 0.7)
    tail_chars = int(max_chars * 0.3)
    
    return (
        text[:head_chars]
        + f"\n\n... [TRUNCATED {len(text) - max_chars} characters] ...\n\n"
        + text[-tail_chars:]
    )


def _extract_output_text(run_result: Any) -> str:
    """Best-effort extraction of final text output from a RunResult."""
    # Try final_output first (this is the correct attribute for the current SDK version)
    final_output = getattr(run_result, "final_output", None)
    if final_output:
        return str(final_output)
    
    # Fallback to legacy attributes
    primary = getattr(run_result, "output", None)
    if primary:
        return str(primary)

    final_response = getattr(run_result, "final_response", None)
    if final_response is None:
        return ""

    output_text = getattr(final_response, "output_text", None)
    if output_text:
        return str(output_text)

    aggregated: List[str] = []
    output_blocks = getattr(final_response, "output", []) or []
    for block in output_blocks:
        content_items = getattr(block, "content", []) or []
        for item in content_items:
            text_part = getattr(item, "text", None)
            if text_part is None:
                continue
            value = getattr(text_part, "value", None)
            if value:
                aggregated.append(str(value))
            else:
                aggregated.append(str(text_part))
    return "\n".join(part for part in aggregated if part)


class SchemaMappingOrchestrator:
    """Coordinates the end-to-end schema mapping workflow using explicit stages."""

    def __init__(
        self,
        *,
        source_files: Sequence[str],
        row_limit: int,
        output_dir: str | Path,
    ) -> None:
        output_path = Path(output_dir).expanduser().resolve()
        self.context = RunContext(
            source_files=[str(Path(p).expanduser().resolve()) for p in source_files],
            row_limit=row_limit,
            output_root=output_path,
        )
        self.executor = AgentExecutor()
        self.renderer = get_renderer()
        self.target_schema_json = json.dumps(
            DemandForecastingRecord.model_json_schema(), indent=2
        )

    async def execute(self) -> Dict[str, Any]:
        """Run all stages and return high-level summary metadata."""
        self._prepare_directories()
        try:
            metadata = await self._run_dataprep()
            mapping_plan, manifest = await self._run_mapping(metadata)
            final_dataset = await self._run_integration(manifest)
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI API error during orchestration: {exc}") from exc

        return {
            "status": "success",
            "run_id": self.context.run_id,
            "output_root": str(self.context.output_root),
            "final_dataset": str(final_dataset),
            "mapping_plan": str(mapping_plan),
            "mapping_manifest": str(manifest),
            "source_metadata": str(metadata),
        }

    def _prepare_directories(self) -> None:
        run_dir = self.context.run_directory()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.context.mapped_directory().mkdir(parents=True, exist_ok=True)
        self.context.evaluations_directory().mkdir(parents=True, exist_ok=True)
        os.environ["AGENT_ROW_LIMIT"] = str(self.context.row_limit)
        os.environ["AGENT_OUTPUT_DIR"] = str(self.context.output_root)

    async def _run_dataprep(self) -> Path:
        prompt = self.renderer.render(
            "DataPrepAgent",
            source_files=self.context.source_files,
            row_limit=self.context.row_limit,
            target_schema_json=self.target_schema_json,
        )
        print("\n" + "=" * 70)
        print("🔍 AGENT 1: DATA PREPARATION")
        print("=" * 70 + "\n")

        run_result = await self.executor.run(
            agent=data_prep_agent,
            messages=[{"role": "user", "content": prompt}],
            trace_label="Agent 1: DataPrep",
        )

        output_text = _extract_output_text(run_result)
        if not output_text:
            raise ValueError("DataPrep agent returned no output.")

        metadata = _parse_json_value(
            output_text,
            expect_type=list,
            error_message="DataPrep agent output did not include metadata JSON.",
        )

        # Calculate total source rows for row preservation tracking
        total_source_rows = 0
        for file_meta in metadata:
            if isinstance(file_meta, dict) and "shape" in file_meta:
                shape = file_meta["shape"]
                if isinstance(shape, list) and len(shape) >= 1:
                    total_source_rows += shape[0]  # shape[0] is row count
        
        self.context.artifacts.source_total_rows = total_source_rows
        print(f"📊 Total source rows across all files: {total_source_rows}")

        metadata_path = self.context.run_directory() / "source_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        self.context.artifacts.source_metadata_path = metadata_path

        print("\n✅ Data preparation artifact stored at", metadata_path)
        
        # Evaluate Data Prep Agent
        await self._evaluate_dataprep(
            agent_input=prompt,
            agent_output=output_text,
            expected_files=self.context.source_files,
        )
        
        print("---\n")
        return metadata_path

    async def _run_mapping(self, metadata_path: Path) -> tuple[Path, Path]:
        metadata_json = metadata_path.read_text()
        prompt = self.renderer.render(
            "ColumnMappingAgent",
            dataset_metadata_json=metadata_json,
            target_schema_json=self.target_schema_json,
            mapped_dir=str(self.context.mapped_directory()),
        )

        print("\n" + "=" * 70)
        print("🗺️  AGENT 2: COLUMN MAPPING")
        print("=" * 70 + "\n")

        run_result = await self.executor.run(
            agent=column_mapping_agent,
            messages=[{"role": "user", "content": prompt}],
            trace_label="Agent 2: ColumnMapping",
        )

        output_text = _extract_output_text(run_result)
        mapping_plan = _parse_json_value(
            output_text,
            expect_type=dict,
            predicate=lambda value: isinstance(value, dict) and "mappings" in value,
            error_message="ColumnMapping agent did not emit a JSON mapping plan.",
        )

        manifest = _parse_json_value(
            output_text,
            expect_type=dict,
            predicate=lambda value: isinstance(value, dict)
            and ("outputs" in value or "output" in value),
            error_message="ColumnMapping agent did not return mapped CSV manifest.",
        )

        mapping_plan_path = self.context.run_directory() / "mapping_plan.json"
        mapping_manifest_path = self.context.run_directory() / "mapping_manifest.json"
        mapping_plan_path.write_text(json.dumps(mapping_plan, indent=2))
        mapping_manifest_path.write_text(json.dumps(manifest, indent=2))

        self.context.artifacts.mapping_plan_path = mapping_plan_path
        self.context.artifacts.mapping_manifest_path = mapping_manifest_path

        print("\n✅ Column mapping artifacts stored at", mapping_plan_path)
        
        # Evaluate Column Mapping Agent
        await self._evaluate_column_mapping(
            agent_input=prompt,
            agent_output=output_text,
            mapping_plan=mapping_plan,
        )
        
        print("---\n")
        return mapping_plan_path, mapping_manifest_path

    async def _run_integration(self, manifest_path: Path) -> Path:
        manifest_json = manifest_path.read_text()
        output_path = self.context.output_root / "final_mapped_dataset.csv"
        prompt = self.renderer.render(
            "DataIntegrationAgent",
            mapped_manifest_json=manifest_json,
            target_schema_json=self.target_schema_json,
            output_path=str(output_path),
        )

        print("\n" + "=" * 70)
        print("🔗 AGENT 3: DATA INTEGRATION")
        print("=" * 70 + "\n")

        run_result = await self.executor.run(
            agent=data_integration_agent,
            messages=[{"role": "user", "content": prompt}],
            trace_label="Agent 3: Integration",
        )

        output_text = _extract_output_text(run_result)
        integration_payload = _parse_json_value(
            output_text,
            expect_type=dict,
            predicate=lambda value: isinstance(value, dict) and "status" in value,
            error_message="DataIntegration agent did not report integration status JSON.",
        )

        status = integration_payload.get("status", "").lower()
        if status != "success":
            raise ValueError(
                f"DataIntegration agent reported failure: {json.dumps(integration_payload, indent=2)}"
            )

        if not output_path.exists():
            raise FileNotFoundError(
                f"Integration tool reported success but file missing: {output_path}"
            )

        self.context.artifacts.final_dataset_path = output_path

        print("\n✅ Final dataset written to", output_path)
        
        # Evaluate Data Integration Agent
        await self._evaluate_data_integration(
            agent_input=prompt,
            agent_output=output_text,
            integration_payload=integration_payload,
        )
        
        print("---\n")
        return output_path


    async def _evaluate_dataprep(
        self,
        *,
        agent_input: str,
        agent_output: str,
        expected_files: List[str],
    ) -> Path | None:
        """Evaluate the Data Prep Agent's performance."""
        eval_prompt = self.renderer.render(
            "DataPrepEvaluationAgent",
            agent_input=_truncate_for_evaluation(agent_input),
            agent_output=_truncate_for_evaluation(agent_output),
            expected_files=json.dumps(expected_files),
        )

        print("\n" + "─" * 70)
        print("📊 EVALUATING: Data Prep Agent")
        print("─" * 70 + "\n")

        try:
            run_result = await self.executor.run(
                agent=data_prep_evaluation_agent,
                messages=[{"role": "user", "content": eval_prompt}],
                trace_label="Eval: DataPrep",
            )

            output_text = _extract_output_text(run_result)
            eval_path = self.context.run_directory() / "data_prep_evaluation.json"
            
            # Try to extract JSON from output
            try:
                eval_data = _parse_json_value(
                    output_text,
                    expect_type=dict,
                    error_message="Evaluation response did not include JSON.",
                )
                eval_path.write_text(json.dumps(eval_data, indent=2))
            except ValueError:
                # Store as text if JSON parsing fails
                eval_path = self.context.run_directory() / "data_prep_evaluation.txt"
                eval_path.write_text(output_text, encoding='utf-8')
            
            self.context.artifacts.data_prep_eval_path = eval_path
            print(f"\n✅ Data Prep evaluation stored at {eval_path}")
            return eval_path
        except Exception as exc:
            print(f"\n⚠️  Data Prep evaluation failed: {exc}")
            return None

    async def _evaluate_column_mapping(
        self,
        *,
        agent_input: str,
        agent_output: str,
        mapping_plan: Dict[str, Any],
    ) -> Path | None:
        """Evaluate the Column Mapping Agent's performance."""
        # Don't truncate for this evaluation - we need complete JSON
        eval_prompt = self.renderer.render(
            "ColumnMappingEvaluationAgent",
            agent_input=agent_input[:2000] + "..." if len(agent_input) > 2000 else agent_input,
            agent_output=agent_output[:2000] + "..." if len(agent_output) > 2000 else agent_output,
            mapping_plan_json=json.dumps(mapping_plan, indent=2),
            target_schema_json=self.target_schema_json,
        )

        print("\n" + "─" * 70)
        print("📊 EVALUATING: Column Mapping Agent")
        print("─" * 70 + "\n")

        try:
            run_result = await self.executor.run(
                agent=column_mapping_evaluation_agent,
                messages=[{"role": "user", "content": eval_prompt}],
                trace_label="Eval: ColumnMapping",
            )

            output_text = _extract_output_text(run_result)
            eval_path = self.context.run_directory() / "column_mapping_evaluation.json"
            
            # Try to extract JSON from output
            try:
                eval_data = _parse_json_value(
                    output_text,
                    expect_type=dict,
                    error_message="Evaluation response did not include JSON.",
                )
                eval_path.write_text(json.dumps(eval_data, indent=2))
            except ValueError:
                # Store as text if JSON parsing fails
                eval_path = self.context.run_directory() / "column_mapping_evaluation.txt"
                eval_path.write_text(output_text, encoding='utf-8')
            
            self.context.artifacts.column_mapping_eval_path = eval_path
            print(f"\n✅ Column Mapping evaluation stored at {eval_path}")
            return eval_path
        except Exception as exc:
            print(f"\n⚠️  Column Mapping evaluation failed: {exc}")
            return None

    async def _evaluate_data_integration(
        self,
        *,
        agent_input: str,
        agent_output: str,
        integration_payload: Dict[str, Any],
    ) -> Path | None:
        """Evaluate the Data Integration Agent's performance and validate final dataset."""
        # Get actual source and final row counts for proper row preservation check
        source_row_count = self.context.artifacts.source_total_rows
        final_row_count = integration_payload.get("rows", 0)
        final_dataset_path = integration_payload.get("output_path", "")
        
        # Get mapping plan for validation
        mapping_plan_path = self.context.artifacts.mapping_plan_path
        mapping_plan_json = mapping_plan_path.read_text() if mapping_plan_path and mapping_plan_path.exists() else "{}"
        
        eval_prompt = self.renderer.render(
            "DataIntegrationEvaluationAgent",
            agent_input=_truncate_for_evaluation(agent_input),
            agent_output=_truncate_for_evaluation(agent_output),
            source_row_count=source_row_count,
            final_row_count=final_row_count,
            final_dataset_path=final_dataset_path,
            target_schema_json=self.target_schema_json,
            mapping_plan_json=mapping_plan_json,
        )

        print("\n" + "─" * 70)
        print("📊 EVALUATING: Data Integration Agent")
        print("─" * 70 + "\n")

        try:
            run_result = await self.executor.run(
                agent=data_integration_evaluation_agent,
                messages=[{"role": "user", "content": eval_prompt}],
                trace_label="Eval: DataIntegration",
            )

            output_text = _extract_output_text(run_result)
            eval_path = self.context.run_directory() / "data_integration_evaluation.json"
            
            # Try to extract JSON from output
            try:
                eval_data = _parse_json_value(
                    output_text,
                    expect_type=dict,
                    error_message="Evaluation response did not include JSON.",
                )
                eval_path.write_text(json.dumps(eval_data, indent=2))
            except ValueError:
                # Store as text if JSON parsing fails
                eval_path = self.context.run_directory() / "data_integration_evaluation.txt"
                eval_path.write_text(output_text, encoding='utf-8')
            
            self.context.artifacts.data_integration_eval_path = eval_path
            print(f"\n✅ Data Integration evaluation stored at {eval_path}")
            return eval_path
        except Exception as exc:
            print(f"\n⚠️  Data Integration evaluation failed: {exc}")
            return None
