import json
from pathlib import Path

import pandas as pd
import pytest

from mapping_system.schema_mapping.evaluation.deepeval_runner import (
    SchemaMappingEvaluationConfig,
    run_schema_mapping_evaluation,
)
from mapping_system.schema_mapping.tools.functions import run_schema_mapping_deepeval


@pytest.mark.unit
def test_run_schema_mapping_evaluation(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPEVAL_API_KEY", raising=False)

    final_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "product_id": ["sku-1", "sku-2"],
            "store_id": ["store-1", "store-2"],
        }
    )
    final_dataset_path = tmp_path / "final.csv"
    final_df.to_csv(final_dataset_path, index=False)

    mapping_plan = {
        "mappings": [
            {"source_column": "date", "target_column": "date", "confidence": 0.95},
            {"source_column": "product", "target_column": "product_id", "confidence": 0.9, "reasoning": "SKU matches"},
            {"source_column": "store", "target_column": "store_id", "confidence": 0.9},
        ]
    }

    source_metadata = [
            {
                "file_path": "data/source.csv",
                "columns": ["date", "product", "store"],
                "dtypes": {"date": "object", "product": "object", "store": "object"},
                "column_descriptions": {
                    "product": "Product identifier",
                    "store": "Store identifier code",
                },
            }
    ]

    target_schema = {
        "properties": {
            "date": {"type": "string"},
            "product_id": {"type": "string"},
            "store_id": {"type": "string"},
        },
        "required": ["date", "product_id", "store_id"],
    }

    summary = run_schema_mapping_evaluation(
        final_dataset_path=str(final_dataset_path),
        mapping_plan=mapping_plan,
        source_metadata=source_metadata,
        target_schema=target_schema,
        config_path=None,
        output_dir=str(tmp_path / "evaluations"),
    )

    metrics = {item["name"]: item for item in summary["metrics"]}
    assert metrics["Field Coverage"]["success"] is True
    assert metrics["Type Compatibility"]["success"] is True
    assert metrics["Semantic Similarity"]["success"] is True

    summary_path = Path(summary["run_directory"]) / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["metrics"]


@pytest.mark.unit
def test_run_schema_mapping_deepeval_inline_json(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPEVAL_API_KEY", raising=False)

    final_df = pd.DataFrame({"date": ["2024-01-01"], "product_id": ["sku-1"], "store_id": ["store-1"]})
    final_dataset_path = tmp_path / "final.csv"
    final_df.to_csv(final_dataset_path, index=False)

    mapping_plan = {
        "mappings": [
            {"source_column": "date", "target_column": "date", "confidence": 0.9},
            {"source_column": "product_id", "target_column": "product_id", "confidence": 0.9},
            {"source_column": "store_id", "target_column": "store_id", "confidence": 0.9},
        ]
    }
    source_metadata = [
        {
            "file_path": str(final_dataset_path),
            "columns": ["date", "product_id", "store_id"],
            "dtypes": {"date": "object", "product_id": "object", "store_id": "object"},
        }
    ]
    target_schema = {
        "properties": {
            "date": {"type": "string"},
            "product_id": {"type": "string"},
            "store_id": {"type": "string"},
        },
        "required": ["date", "product_id", "store_id"],
    }

    summary = run_schema_mapping_deepeval(
        final_dataset_path=str(final_dataset_path),
        mapping_plan_json=json.dumps(mapping_plan, indent=2),
        source_metadata_json=json.dumps(source_metadata, indent=2),
        target_schema_json=json.dumps(target_schema, indent=2),
        config_path=None,
        output_dir=str(tmp_path / "evaluations"),
    )

    assert any(metric["success"] for metric in summary["metrics"])


@pytest.mark.unit
def test_schema_mapping_evaluation_config_env_expansion(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    monkeypatch.setenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")
    config_file.write_text(
        """
llm:
  judge_model: ${DEEPEVAL_JUDGE_MODEL:-gpt-3.5-turbo}
runtime:
  seed: ${DEEPEVAL_SEED:-123}
"""
    )

    config = SchemaMappingEvaluationConfig.load(str(config_file))
    assert config.llm["judge_model"] == "gpt-4o"
    assert config.runtime["seed"] == "123"
