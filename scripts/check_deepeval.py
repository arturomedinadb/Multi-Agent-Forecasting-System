"""
Checks whether deepeval is installed, configured, and able to run an
evaluation end-to-end in the current environment.

Verifies, in order:
1. Whether DEEPEVAL_API_KEY is set.
2. Whether the deepeval and schema_mapping.evaluation.metrics imports used
   by evaluate_column_mapping_agent succeed.
3. Whether evaluate_column_mapping_agent itself runs successfully against a
   small synthetic mapping plan and target schema.

Usage:
    uv run python scripts/check_deepeval.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def check_api_key() -> bool:
    has_key = bool(os.getenv("DEEPEVAL_API_KEY"))
    print("1) DEEPEVAL_API_KEY present:", has_key)
    return has_key


def check_imports() -> bool:
    print("\n2) Checking required imports:")
    all_ok = True

    try:
        from deepeval.test_case import LLMTestCase, ToolCall  # noqa: F401
        print("   deepeval.test_case: OK")
    except ImportError as e:
        print("   deepeval.test_case: FAILED ->", e)
        all_ok = False

    try:
        from deepeval.metrics import TaskCompletionMetric  # noqa: F401
        print("   deepeval.metrics.TaskCompletionMetric: OK")
    except ImportError as e:
        print("   deepeval.metrics.TaskCompletionMetric: FAILED ->", e)
        all_ok = False

    try:
        from schema_mapping.evaluation.metrics import (  # noqa: F401
            FieldCoverageMetric,
            TypeCompatibilityMetric,
            SemanticSimilarityMetric,
        )
        print("   schema_mapping.evaluation.metrics: OK")
    except ImportError as e:
        print("   schema_mapping.evaluation.metrics: FAILED ->", e)
        all_ok = False

    import deepeval
    print("   deepeval version:", deepeval.__version__)

    return all_ok


def run_synthetic_evaluation():
    print("\n3) Running evaluate_column_mapping_agent with synthetic data:")

    from schema_mapping.functions import evaluate_column_mapping_agent
    from schema_mapping.schemas.models import DemandForecastingRecord

    mapping_plan = {
        "mappings": [
            {
                "source_file": "x.csv",
                "mappings": [
                    {"source_column": "transaction_date", "target_column": "date", "confidence": 0.95, "reasoning": "date match"},
                    {"source_column": "prod_code", "target_column": "product_id", "confidence": 0.9, "reasoning": "id match"},
                ],
            }
        ]
    }
    target_schema = DemandForecastingRecord.model_json_schema()

    kwargs = dict(
        agent_input="Create intelligent mappings based on the target schema provided.",
        agent_output="Successfully created mapped CSV files.",
        mapping_plan_json=json.dumps(mapping_plan),
        target_schema_json=json.dumps(target_schema),
    )

    async def run():
        result = evaluate_column_mapping_agent.on_invoke_tool(None, json.dumps(kwargs))
        if hasattr(result, "__await__"):
            result = await result
        return result

    output = asyncio.run(run())
    print(output)


def main():
    check_api_key()
    imports_ok = check_imports()
    if imports_ok:
        run_synthetic_evaluation()
    else:
        print("\nSkipping synthetic evaluation: one or more imports failed.")


if __name__ == "__main__":
    main()
