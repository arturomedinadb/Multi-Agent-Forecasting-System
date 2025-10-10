"""Execution harness for schema mapping DeepEval checks."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from .metrics import (
    FieldCoverageMetric,
    MetricOutcome,
    SemanticSimilarityMetric,
    TypeCompatibilityMetric,
)
from .test_cases import (
    SchemaMappingEvaluationContext,
    build_context,
    build_schema_mapping_cases,
)

try:
    from deepeval import assert_test  # type: ignore
    from deepeval.metrics import AnswerRelevancyMetric, TaskCompletionMetric  # type: ignore
    from deepeval.test_case import LLMTestCase  # type: ignore
except ImportError:  # pragma: no cover - allow unit tests without deepeval installed
    assert_test = None  # type: ignore
    AnswerRelevancyMetric = None  # type: ignore
    TaskCompletionMetric = None  # type: ignore
    LLMTestCase = None  # type: ignore

DEFAULT_OUTPUT_DIR = Path("output") / "evaluations"
_ENV_PATTERN = re.compile(r"\${([A-Za-z0-9_]+)(:-([^}]+))?}")


@dataclass
class SchemaMappingEvaluationConfig:
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str]) -> "SchemaMappingEvaluationConfig":
        if not config_path:
            return cls()
        data = yaml.safe_load(Path(config_path).read_text()) or {}

        def _expand(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _expand(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_expand(item) for item in value]
            if isinstance(value, str):
                return _expand_env_string(value)
            return value

        metrics = _expand(data.get("metrics", {}))
        thresholds = _expand(data.get("thresholds", {}))
        llm = _expand(data.get("llm", {}))
        runtime = _expand(data.get("runtime", {}))
        paths = _expand(data.get("paths", {}))

        return cls(
            metrics=metrics,
            thresholds=thresholds,
            llm=llm,
            runtime=runtime,
            paths=paths,
        )

    def metric_setting(self, name: str, key: str, default: Any) -> Any:
        return ((self.metrics.get(name, {}) or {}).get(key)) or default

    def threshold(self, name: str, default: float) -> float:
        raw = self.thresholds.get(name, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def runtime_option(self, key: str, default: Any) -> Any:
        return self.runtime.get(key, default)


@dataclass
class EvaluationSummary:
    run_directory: Path
    metrics: List[MetricOutcome]
    llm_metrics: List[Dict[str, Any]]
    context_snapshot: SchemaMappingEvaluationContext
    improvement_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_directory": str(self.run_directory),
            "metrics": [
                {
                    "name": item.name,
                    "score": round(item.score, 4),
                    "threshold": item.threshold,
                    "success": item.success,
                    "details": item.details,
                }
                for item in self.metrics
            ],
            "llm_metrics": self.llm_metrics,
            "improvement_prompt": self.improvement_prompt,
            "context": {
                "final_dataset_path": self.context_snapshot.final_dataset_path,
                "dataset_columns": self.context_snapshot.dataset_columns,
                "required_fields": self.context_snapshot.required_fields,
            },
        }


def _ensure_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    value_str = str(value)
    candidate = Path(value_str)
    if candidate.exists() and candidate.is_file():
        try:
            return json.loads(candidate.read_text())
        except json.JSONDecodeError:
            raise ValueError(f"File at {candidate} did not contain valid JSON")
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        raise ValueError(f"Unable to parse JSON payload: {value_str[:80]}")


def _expand_env_string(value: str) -> str:
    def _replacement(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(3) or ""
        return os.getenv(var_name, default)

    return _ENV_PATTERN.sub(_replacement, value)


def _resolve_output_dir(config: SchemaMappingEvaluationConfig, override: Optional[str]) -> Path:
    if override:
        base = Path(override)
    else:
        configured = config.runtime_option("output_dir", DEFAULT_OUTPUT_DIR)
        base = Path(configured)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"schema_mapping_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _serialize_inputs(run_dir: Path, *, mapping_plan: Dict[str, Any], source_metadata: Sequence[Dict[str, Any]], target_schema: Dict[str, Any]) -> None:
    (run_dir / "mapping_plan.json").write_text(json.dumps(mapping_plan, indent=2))
    (run_dir / "source_metadata.json").write_text(json.dumps(list(source_metadata), indent=2))
    (run_dir / "target_schema.json").write_text(json.dumps(target_schema, indent=2))


def _build_improvement_prompt(metric_outcomes: Sequence[MetricOutcome]) -> str:
    failing = [item for item in metric_outcomes if not item.success]
    if not failing:
        return (
            "All deterministic metrics passed. Maintain current mapping heuristics, "
            "but log future runs to watch for drift in coverage, type compatibility, and semantics."
        )

    bullet_lines = []
    for item in failing:
        detail = item.details
        if item.name == "Field Coverage" and detail.get("missing_fields"):
            missing = ", ".join(detail["missing_fields"][:10])
            bullet_lines.append(f"Increase coverage for: {missing}")
        elif item.name == "Type Compatibility" and detail.get("incompatible_fields"):
            incompatible = [entry["field"] for entry in detail["incompatible_fields"]][:10]
            bullet_lines.append(f"Fix dtype alignment for: {', '.join(incompatible)}")
        elif item.name == "Semantic Similarity":
            low = [
                f"{entry['source']}→{entry['target']} ({entry['score']})"
                for entry in detail.get("mappings", [])
                if entry.get("score", 0) < item.threshold
            ][:5]
            if low:
                bullet_lines.append("Revisit semantic alignment for: " + ", ".join(low))
        else:
            bullet_lines.append(f"Investigate metric '{item.name}'")

    guidance = "\n".join(f"- {line}" for line in bullet_lines)
    return (
        "Deterministic evaluation flagged the following issues:\n"
        f"{guidance}\n"
        "Regenerate mappings focusing on these corrections before the next run."
    )


def _evaluate_deterministic_metrics(test_case: LLMTestCase, config: SchemaMappingEvaluationConfig) -> List[MetricOutcome]:
    additional_metadata = test_case.additional_metadata or {}
    required_fields: List[str] = additional_metadata.get("required_fields") or []
    target_types: Dict[str, str] = additional_metadata.get("target_types") or {}
    if not required_fields:
        required_fields = list(target_types.keys())

    coverage_threshold = config.metric_setting("field_coverage", "minimum_ratio", 0.9)
    coverage_metric = FieldCoverageMetric(required_fields=required_fields, threshold=float(coverage_threshold))

    fail_on_casts = config.metric_setting("type_compatibility", "fail_on_casts", [])
    type_metric = TypeCompatibilityMetric(expected_types=target_types, fail_on_casts=fail_on_casts)

    semantic_threshold = config.metric_setting("semantic_similarity", "minimum_score", 0.5)
    semantic_metric = SemanticSimilarityMetric(minimum_score=float(semantic_threshold))

    deterministic_metrics = [coverage_metric, type_metric, semantic_metric]
    outcomes: List[MetricOutcome] = []

    for metric in deterministic_metrics:
        try:
            metric.measure(test_case)
            metric.is_successful()
            score = float(metric.score or 0.0)
            details = dict(getattr(metric, "score_breakdown", {}))
        except Exception as exc:  # pragma: no cover - defensive
            score = 0.0
            details = {"error": str(exc)}
            metric.success = False
        outcome = MetricOutcome(
            name=metric.__name__,
            score=score,
            threshold=float(getattr(metric, "threshold", 0.0)),
            success=bool(metric.success),
            details=details,
        )
        outcomes.append(outcome)
    return outcomes


def _evaluate_llm_metrics(
    test_case: LLMTestCase,
    config: SchemaMappingEvaluationConfig,
) -> List[Dict[str, Any]]:
    llm_metrics_cfg = config.metrics.get("llm_judge", {}) if config.metrics else {}
    if not llm_metrics_cfg or not llm_metrics_cfg.get("enabled", False):
        return []

    api_key = os.getenv("DEEPEVAL_API_KEY")
    if not api_key:
        return [
            {
                "name": "LLM Metrics",
                "status": "skipped",
                "reason": "DEEPEVAL_API_KEY not set; deterministic metrics still executed.",
            }
        ]

    if AnswerRelevancyMetric is None or TaskCompletionMetric is None or assert_test is None:
        return [
            {
                "name": "LLM Metrics",
                "status": "skipped",
                "reason": "deepeval runtime unavailable; reinstall dependencies to enable LLM checks.",
            }
        ]

    metric_names: Sequence[str] = llm_metrics_cfg.get("metric_names", [])
    instantiated = []
    for name in metric_names:
        key = name.lower()
        if key == "taskcompletion":
            instantiated.append(
                TaskCompletionMetric(
                    threshold=config.threshold("task_completion", 0.6),
                    model=config.llm.get("judge_model"),
                    async_mode=False,
                    verbose_mode=False,
                )
            )
        elif key == "answerrelevancy":
            instantiated.append(
                AnswerRelevancyMetric(
                    threshold=config.threshold("answer_relevancy", 0.6),
                    model=config.llm.get("judge_model"),
                    async_mode=False,
                    verbose_mode=False,
                )
            )

    if not instantiated:
        return []

    results: List[Dict[str, Any]] = []
    for metric in instantiated:
        try:
            metric.measure(test_case)
            metric.is_successful()
            results.append(
                {
                    "name": metric.__name__,
                    "score": float(metric.score or 0.0),
                    "threshold": float(getattr(metric, "threshold", 0.0)),
                    "success": bool(metric.success),
                    "reason": getattr(metric, "reason", None),
                }
            )
        except Exception as exc:  # pragma: no cover - handle API/model failures
            results.append(
                {
                    "name": metric.__name__ if hasattr(metric, "__name__") else metric.__class__.__name__,
                    "status": "error",
                    "reason": str(exc),
                }
            )
    return results


def run_schema_mapping_evaluation(
    *,
    final_dataset_path: str,
    mapping_plan: Any,
    source_metadata: Any,
    target_schema: Any,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if LLMTestCase is None:
        raise RuntimeError("deepeval must be installed to execute the evaluation workflow.")

    config = SchemaMappingEvaluationConfig.load(config_path)

    dataset_path = Path(final_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Final dataset not found at {dataset_path}")

    dataframe = pd.read_csv(dataset_path)

    mapping_plan_payload = _ensure_payload(mapping_plan)
    metadata_payload = _ensure_payload(source_metadata)
    target_schema_payload = _ensure_payload(target_schema)

    if not isinstance(metadata_payload, list):
        raise ValueError("source_metadata must be a JSON array of metadata records")
    if not isinstance(mapping_plan_payload, dict):
        raise ValueError("mapping_plan must be a JSON object")
    if not isinstance(target_schema_payload, dict):
        raise ValueError("target_schema must be a JSON object")

    run_dir = _resolve_output_dir(config, output_dir)
    _serialize_inputs(
        run_dir,
        mapping_plan=mapping_plan_payload,
        source_metadata=metadata_payload,
        target_schema=target_schema_payload,
    )

    context = build_context(
        dataframe=dataframe,
        mapping_plan=mapping_plan_payload,
        source_metadata=metadata_payload,
        target_schema=target_schema_payload,
        final_dataset_path=dataset_path,
    )
    test_cases = build_schema_mapping_cases(context)
    test_case = test_cases[0]

    deterministic = _evaluate_deterministic_metrics(test_case, config)
    llm_results = _evaluate_llm_metrics(test_case, config)
    improvement_prompt = _build_improvement_prompt(deterministic)

    summary = EvaluationSummary(
        run_directory=run_dir,
        metrics=deterministic,
        llm_metrics=llm_results,
        context_snapshot=context,
        improvement_prompt=improvement_prompt,
    )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

    if bool(config.runtime_option("write_markdown_summary", True)):
        markdown_path = run_dir / "summary.md"
        _write_markdown_report(markdown_path, summary)

    return summary.to_dict()


def _write_markdown_report(path: Path, summary: EvaluationSummary) -> None:
    lines = [
        f"# Schema Mapping Evaluation ({summary.run_directory.name})",
        "",
        "## Deterministic Metrics",
    ]
    for metric in summary.metrics:
        status = "PASS" if metric.success else "FAIL"
        lines.append(f"- **{metric.name}** – {status} (score={metric.score:.3f}, threshold={metric.threshold})")
        if metric.details:
            lines.append(f"  - Details: {json.dumps(metric.details, indent=2)}")
    if summary.llm_metrics:
        lines.append("\n## LLM Metrics")
        for item in summary.llm_metrics:
            status = item.get("status")
            if status:
                lines.append(f"- **{item.get('name', 'Metric')}** – {status}: {item.get('reason')}")
            else:
                success = "PASS" if item.get("success") else "FAIL"
                lines.append(
                    f"- **{item.get('name', 'Metric')}** – {success} (score={item.get('score')}, threshold={item.get('threshold')})"
                )
                reason = item.get("reason")
                if reason:
                    lines.append(f"  - Reason: {reason}")
    lines.extend([
        "",
        "## Recommended Next Prompt",
        "",
        summary.improvement_prompt,
    ])
    path.write_text("\n".join(lines))


__all__ = [
    "run_schema_mapping_evaluation",
    "SchemaMappingEvaluationConfig",
]
