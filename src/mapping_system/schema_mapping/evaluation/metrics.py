"""Custom DeepEval metrics for schema mapping quality checks.

These deterministic metrics run locally without requiring an LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
except ImportError:  # pragma: no cover - defensive fallback for environments without deepeval installed
    BaseMetric = object  # type: ignore
    LLMTestCase = object  # type: ignore


def _canonical_tokens(text: str) -> List[str]:
    """Split a string into lowercase alphanumeric tokens."""
    cleaned = "" if text is None else text.lower()
    tokens: List[str] = []
    current = []
    for char in cleaned:
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _canonical_dtype(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    lower = value.lower()
    if "int" in lower:
        return "integer"
    if any(tok in lower for tok in ("float", "double", "decimal")):
        return "number"
    if any(tok in lower for tok in ("date", "time")):
        return "date"
    if any(tok in lower for tok in ("bool", "flag")):
        return "boolean"
    return "string"


@dataclass
class MetricOutcome:
    name: str
    score: float
    threshold: float
    success: bool
    details: Dict[str, object]


class FieldCoverageMetric(BaseMetric):
    """Measure how many required target fields are present in the mapped dataset."""

    def __init__(self, *, required_fields: Sequence[str], threshold: float = 0.9):
        self.required_fields = list({field for field in required_fields})
        self.threshold = threshold
        self.async_mode = False
        self.score_breakdown: Dict[str, object] = {}
        self.score: Optional[float] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: LLMTestCase, *_: object, **__: object) -> float:  # type: ignore[override]
        metadata: Dict[str, object] = getattr(test_case, "additional_metadata", {}) or {}
        dataset_columns: Iterable[str] = metadata.get("dataset_columns", [])
        covered = set(dataset_columns).intersection(self.required_fields)
        required_total = len(self.required_fields)
        if required_total == 0:
            self.score = 1.0
        else:
            self.score = len(covered) / required_total
        missing = [field for field in self.required_fields if field not in covered]
        self.score_breakdown = {
            "covered_fields": sorted(covered),
            "missing_fields": missing,
            "required_total": required_total,
        }
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *_: object, **__: object) -> float:  # type: ignore[override]
        return self.measure(test_case)

    def is_successful(self) -> bool:  # type: ignore[override]
        score = self.score if self.score is not None else 0.0
        self.success = score >= self.threshold
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Field Coverage"


class TypeCompatibilityMetric(BaseMetric):
    """Validate whether mapped field dtypes are compatible with target schema expectations."""

    def __init__(
        self,
        *,
        expected_types: Dict[str, str],
        fail_on_casts: Optional[Sequence[str]] = None,
    ) -> None:
        self.expected_types = expected_types
        self.fail_on_casts = set(fail_on_casts or [])
        self.threshold = 1.0
        self.async_mode = False
        self.score_breakdown: Dict[str, object] = {}
        self.score: Optional[float] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: LLMTestCase, *_: object, **__: object) -> float:  # type: ignore[override]
        metadata: Dict[str, object] = getattr(test_case, "additional_metadata", {}) or {}
        dataset_types: Dict[str, str] = metadata.get("dataset_dtypes", {}) or {}
        incompatible: List[Tuple[str, str, str]] = []
        inspected = 0
        for field, expected_raw in self.expected_types.items():
            if field not in dataset_types:
                continue
            inspected += 1
            observed_raw = dataset_types[field]
            observed = _canonical_dtype(observed_raw)
            expected = _canonical_dtype(expected_raw)
            if observed == expected:
                continue
            cast_key = f"{observed}->{expected}"
            if cast_key in self.fail_on_casts:
                incompatible.append((field, observed_raw or observed, expected_raw or expected))
                continue
            # allow integer -> number widening
            if observed == "integer" and expected == "number":
                continue
            incompatible.append((field, observed_raw or observed, expected_raw or expected))

        total = max(inspected, 1)
        self.score = 1.0 - (len(incompatible) / total)
        self.score_breakdown = {
            "inspected_fields": inspected,
            "incompatible_fields": [
                {"field": field, "observed": observed, "expected": expected}
                for field, observed, expected in incompatible
            ],
        }
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *_: object, **__: object) -> float:  # type: ignore[override]
        return self.measure(test_case)

    def is_successful(self) -> bool:  # type: ignore[override]
        score = self.score if self.score is not None else 0.0
        self.success = math.isclose(score, 1.0, rel_tol=1e-5) and not self.score_breakdown.get("incompatible_fields")
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Type Compatibility"


class SemanticSimilarityMetric(BaseMetric):
    """Estimate semantic alignment between source fields and assigned target fields."""

    def __init__(
        self,
        *,
        minimum_score: float = 0.5,
    ) -> None:
        self.threshold = minimum_score
        self.async_mode = False
        self.score_breakdown: Dict[str, object] = {}
        self.score: Optional[float] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: LLMTestCase, *_: object, **__: object) -> float:  # type: ignore[override]
        metadata: Dict[str, object] = getattr(test_case, "additional_metadata", {}) or {}
        mapping_plan: List[Dict[str, object]] = metadata.get("mapping_plan", [])
        column_descriptions: Dict[str, str] = metadata.get("column_descriptions", {}) or {}
        if not mapping_plan:
            self.score = 0.0
            self.score_breakdown = {"mappings": [], "note": "Mapping plan empty"}
            return self.score

        per_mapping: List[Dict[str, object]] = []
        total = 0.0
        for entry in mapping_plan:
            source = str(entry.get("source_column", "")).strip()
            target = str(entry.get("target_column", "")).strip()
            if not source or not target:
                continue
            source_text = column_descriptions.get(source, source)
            target_text = target
            tokens_source = set(_canonical_tokens(source_text))
            tokens_target = set(_canonical_tokens(target_text))
            if not tokens_source or not tokens_target:
                score = 0.0
            else:
                score = len(tokens_source & tokens_target) / len(tokens_source | tokens_target)
            total += score
            per_mapping.append({
                "source": source,
                "target": target,
                "score": round(score, 3),
                "reasoning": entry.get("reasoning"),
            })

        evaluated = len(per_mapping)
        self.score = 0.0 if evaluated == 0 else total / evaluated
        self.score_breakdown = {"mappings": per_mapping, "evaluated": evaluated}
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *_: object, **__: object) -> float:  # type: ignore[override]
        return self.measure(test_case)

    def is_successful(self) -> bool:  # type: ignore[override]
        score = self.score if self.score is not None else 0.0
        self.success = score >= self.threshold
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Semantic Similarity"


__all__ = [
    "FieldCoverageMetric",
    "TypeCompatibilityMetric",
    "SemanticSimilarityMetric",
    "MetricOutcome",
]
