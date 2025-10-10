"""Utilities for building DeepEval test cases from agent artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json

try:
    from deepeval.test_case import LLMTestCase
except ImportError:  # pragma: no cover - deepeval optional during unit tests
    LLMTestCase = None  # type: ignore


@dataclass
class SchemaMappingEvaluationContext:
    dataset_columns: List[str]
    dataset_dtypes: Dict[str, str]
    required_fields: List[str]
    target_types: Dict[str, str]
    mapping_plan: List[Dict[str, Any]]
    column_descriptions: Dict[str, str]
    source_metadata: List[Dict[str, Any]]
    target_schema: Dict[str, Any]
    final_dataset_path: str


def _extract_required_fields(target_schema: Dict[str, Any]) -> List[str]:
    required = target_schema.get("required") or []
    if isinstance(required, list):
        return list(dict.fromkeys(str(item) for item in required))
    return []


def _extract_target_types(target_schema: Dict[str, Any]) -> Dict[str, str]:
    properties = target_schema.get("properties", {})
    result: Dict[str, str] = {}
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        if "type" in spec and isinstance(spec["type"], str):
            result[name] = spec["type"]
        elif "anyOf" in spec and isinstance(spec["anyOf"], list):
            # Prefer the first non-null type entry
            type_entry = next((item for item in spec["anyOf"] if item.get("type") != "null"), None)
            if isinstance(type_entry, dict) and isinstance(type_entry.get("type"), str):
                result[name] = type_entry["type"]
    return result


def _collect_column_descriptions(metadata: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    for entry in metadata:
        if not isinstance(entry, dict):
            continue
        columns = entry.get("columns") or []
        column_descriptions = entry.get("column_descriptions") or {}
        for column in columns:
            column = str(column)
            if column in column_descriptions:
                descriptions[column] = str(column_descriptions[column])
    return descriptions


def build_context(
    *,
    dataframe,
    mapping_plan: Dict[str, Any],
    source_metadata: Sequence[Dict[str, Any]],
    target_schema: Dict[str, Any],
    final_dataset_path: Path,
) -> SchemaMappingEvaluationContext:
    mapping_entries = mapping_plan.get("mappings") if isinstance(mapping_plan, dict) else []
    if not isinstance(mapping_entries, list):
        mapping_entries = []

    dataset_columns = dataframe.columns.tolist()
    dataset_dtypes = {col: str(dataframe[col].dtype) for col in dataframe.columns}

    required_fields = _extract_required_fields(target_schema)
    target_types = _extract_target_types(target_schema)
    column_descriptions = _collect_column_descriptions(source_metadata)

    return SchemaMappingEvaluationContext(
        dataset_columns=dataset_columns,
        dataset_dtypes=dataset_dtypes,
        required_fields=required_fields,
        target_types=target_types,
        mapping_plan=[entry for entry in mapping_entries if isinstance(entry, dict)],
        column_descriptions=column_descriptions,
        source_metadata=[dict(item) for item in source_metadata],
        target_schema=target_schema,
        final_dataset_path=str(final_dataset_path),
    )


def build_schema_mapping_cases(context: SchemaMappingEvaluationContext) -> List[LLMTestCase]:
    if LLMTestCase is None:
        raise RuntimeError("DeepEval is not installed; cannot build LLMTestCase objects.")

    metadata = {
        "dataset_columns": context.dataset_columns,
        "dataset_dtypes": context.dataset_dtypes,
        "required_fields": context.required_fields,
        "target_types": context.target_types,
        "mapping_plan": context.mapping_plan,
        "column_descriptions": context.column_descriptions,
        "source_metadata": context.source_metadata,
        "target_schema": context.target_schema,
        "final_dataset_path": context.final_dataset_path,
    }

    description_lines = [
        f"Final dataset columns: {', '.join(context.dataset_columns)}",
        f"Required fields: {', '.join(context.required_fields) if context.required_fields else 'None'}",
        f"Mappings provided: {len(context.mapping_plan)}",
    ]

    case = LLMTestCase(
        input="Evaluate the schema mapping output against the target schema.",
        actual_output="Final mapped dataset generated successfully.",
        context=description_lines,
        additional_metadata=metadata,
        name="schema_mapping_overview",
    )

    return [case]


__all__ = [
    "SchemaMappingEvaluationContext",
    "build_context",
    "build_schema_mapping_cases",
]
