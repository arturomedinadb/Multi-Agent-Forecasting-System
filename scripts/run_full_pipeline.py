"""
Run the complete 3-stage pipeline from the command line:
Schema Mapping -> Feature Engineering -> Model Training.

Usage:
    uv run python scripts/run_full_pipeline.py
    uv run python scripts/run_full_pipeline.py --row-limit 1000 --output-dir output/full_run
"""

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents import Runner, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from schema_mapping.run_workflow import run_full_workflow
from ai_forecasting_agents.demand_forecasting.agents.feature_engineering_agent import (
    orchestrator_agent,
)
from ai_forecasting_agents.demand_forecasting.agents.demand_forecasting_agent import (
    training_agent,
)

MERGED_OUTPUT_FILE = "merged_output.csv"
FEATURE_ENGINEERED_FILE = "engineered_features.csv"

DEFAULT_SOURCE_FILES = [
    str(PROJECT_ROOT / "data" / "transaction_like_synth.csv"),
    str(PROJECT_ROOT / "data" / "product_like_synth_wBrand.csv"),
    str(PROJECT_ROOT / "data" / "store_like_synth.csv"),
    str(PROJECT_ROOT / "data" / "holidays.csv"),
    str(PROJECT_ROOT / "data" / "promotion_like_synth.csv"),
    str(PROJECT_ROOT / "data" / "weather_monthly.csv"),
    str(PROJECT_ROOT / "data" / "CPI-monthly.csv"),
    str(PROJECT_ROOT / "data" / "employment_data.csv"),
    str(PROJECT_ROOT / "data" / "GDP_monthly.csv"),
    str(PROJECT_ROOT / "data" / "Population.csv"),
]


def find_merged_csv(output_dir: Path) -> Optional[Path]:
    file_path = output_dir / MERGED_OUTPUT_FILE
    return file_path if file_path.exists() else None


async def run_feature_engineering_stage(
    input_file: str, output_file: str, target_column: str
) -> dict:
    conversation_id = str(uuid.uuid4().hex[:16])
    session = SQLAlchemySession.from_url(
        conversation_id,
        url=f"sqlite+aiosqlite:///{PROJECT_ROOT / 'training_database.db'}",
        create_tables=True,
    )

    # Recorded so the pipeline tool can recover them if the orchestrator
    # forwards its config to the execution sub-agent without the paths.
    os.environ["FEATURE_ENGINEERING_INPUT_FILE"] = input_file
    os.environ["FEATURE_ENGINEERING_OUTPUT_FILE"] = output_file

    initial_prompt = f"""
    You are tasked with feature engineering for demand forecasting.

    INPUT FILE: {input_file}
    OUTPUT FILE: {output_file}
    TARGET COLUMN: {target_column}

    Please analyze the data, generate intelligent feature recommendations, and execute the feature engineering pipeline.
    """

    with trace("Feature Engineering", group_id=conversation_id):
        try:
            result = await Runner.run(
                orchestrator_agent,
                input=initial_prompt,
                # This orchestrator calls three sub-agents in sequence and
                # may retry them, which does not fit the SDK's default of 10.
                max_turns=50,
                session=session,
            )
        except Exception as e:
            # An SDK or model failure must be reported as a stage result
            # rather than crashing the whole pipeline.
            print(f"ERROR: feature engineering failed: {e}")
            return {
                "status": "error",
                "success": False,
                "error": str(e),
                "result": None,
                "output_file": output_file,
            }

    # The agent finishing its conversation says nothing about whether it
    # actually wrote the engineered dataset, so verify the file on disk.
    engineered = Path(output_file)
    produced = engineered.exists() and engineered.stat().st_size > 0

    return {
        "status": "completed" if produced else "error",
        "success": produced,
        "error": (
            None
            if produced
            else f"Feature engineering reported completion but did not write {output_file}"
        ),
        "result": (
            result.final_output if hasattr(result, "final_output") else str(result)
        ),
        "output_file": output_file,
    }


async def run_training_stage(
    input_file: str,
    output_dir: str,
    inference_dir: str,
    id_columns: list[str],
    target_column: str,
    model_types: list[str],
) -> dict:
    conversation_id = str(uuid.uuid4().hex[:16])
    session = SQLAlchemySession.from_url(
        conversation_id,
        url=f"sqlite+aiosqlite:///{PROJECT_ROOT / 'training_database.db'}",
        create_tables=True,
    )

    initial_message = f"""
    Train demand forecasting models with these parameters:
    - Input file: {input_file}
    - Output directory: {output_dir}
    - Inference directory: {inference_dir}
    - ID columns: {id_columns}
    - Target column: {target_column}
    - Model types: {', '.join(model_types)}

    Start by creating model configurations and training the models.
    """

    with trace("Demand Forecasting Training", group_id=conversation_id):
        try:
            result = await Runner.run(
                training_agent, input=initial_message, max_turns=100, session=session
            )
        except Exception as e:
            # An SDK or model failure must be reported as a stage result
            # rather than crashing the whole pipeline.
            print(f"ERROR: training failed: {e}")
            return {
                "status": "error",
                "success": False,
                "error": str(e),
                "result": None,
                "conversation_id": conversation_id,
                "model_file": None,
            }

    items = await session.get_items()

    # A finished conversation is not evidence that anything was produced.
    # Training intermediate models is not the deliverable either - the stage
    # has only done its job once a model is saved for inference and test
    # predictions are written, so require both.
    model_files = sorted(Path(inference_dir).glob("**/*.pkl"))
    prediction_files = sorted(Path(inference_dir).glob("**/test_predictions.csv"))
    best_model_file = model_files[-1] if model_files else None

    missing = []
    if not best_model_file:
        missing.append("no model saved for inference")
    if not prediction_files:
        missing.append("no test_predictions.csv")

    return {
        "status": "completed" if not missing else "error",
        "success": not missing,
        "error": (
            None
            if not missing
            else f"Training reported completion but {' and '.join(missing)}"
        ),
        "result": (
            result.final_output if hasattr(result, "final_output") else str(result)
        ),
        "conversation_id": conversation_id,
        "total_turns": len(items),
        "model_file": str(best_model_file) if best_model_file else None,
        "predictions_file": str(prediction_files[-1]) if prediction_files else None,
    }


async def run_full_pipeline(
    source_files: list[str],
    row_limit: int,
    output_dir: str,
    target_column: str,
    id_columns: list[str],
    model_types: list[str],
) -> dict:
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    pipeline_result = {
        "schema_mapping": {"status": "pending"},
        "feature_engineering": {"status": "pending"},
        "training": {"status": "pending"},
    }

    # --- Stage 1: Schema Mapping ---
    print("=" * 70)
    print("STAGE 1: SCHEMA MAPPING")
    print("=" * 70)
    schema_result = await run_full_workflow(
        source_files=source_files,
        row_limit=row_limit,
        output_dir=str(resolved_output),
    )
    pipeline_result["schema_mapping"] = {
        "status": "completed",
        "success": schema_result.get("status") == "success",
        "result": schema_result,
    }

    time.sleep(1)
    merged_csv = find_merged_csv(resolved_output)
    if not merged_csv:
        time.sleep(2)
        merged_csv = find_merged_csv(resolved_output)
    if not merged_csv:
        pipeline_result["schema_mapping"]["status"] = "error"
        pipeline_result["schema_mapping"]["success"] = False
        pipeline_result["schema_mapping"][
            "error"
        ] = f"Could not find {MERGED_OUTPUT_FILE} in {resolved_output}"
        return pipeline_result

    # --- Stage 2: Feature Engineering ---
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 70)
    feature_output_file = str(resolved_output / FEATURE_ENGINEERED_FILE)
    feature_result = await run_feature_engineering_stage(
        input_file=str(merged_csv),
        output_file=feature_output_file,
        target_column=target_column,
    )
    pipeline_result["feature_engineering"] = feature_result
    if not feature_result["success"]:
        return pipeline_result

    # --- Stage 3: Training ---
    print("\n" + "=" * 70)
    print("STAGE 3: MODEL TRAINING")
    print("=" * 70)
    training_output_dir = str(resolved_output / "training_results")
    training_inference_dir = str(resolved_output / "training_results" / "inference")
    os.makedirs(training_output_dir, exist_ok=True)
    os.makedirs(training_inference_dir, exist_ok=True)

    training_result = await run_training_stage(
        input_file=feature_result["output_file"],
        output_dir=training_output_dir,
        inference_dir=training_inference_dir,
        id_columns=id_columns,
        target_column=target_column,
        model_types=model_types,
    )
    pipeline_result["training"] = training_result

    return pipeline_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full schema-mapping -> feature-engineering -> training pipeline."
    )
    parser.add_argument(
        "--source-files",
        nargs="+",
        default=DEFAULT_SOURCE_FILES,
        help="CSV files to map (default: the sample data/ set)",
    )
    parser.add_argument(
        "--row-limit", type=int, default=500, help="Rows to sample per source file"
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "full_pipeline_run"),
        help="Where to write all pipeline outputs",
    )
    parser.add_argument("--target-column", default="units_sold")
    parser.add_argument(
        "--id-columns", nargs="+", default=["date", "product_id", "store_id"]
    )
    parser.add_argument(
        "--model-types", nargs="+", default=["xgboost", "lightgbm", "catboost"]
    )
    return parser.parse_args()


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Warning: OPENAI_API_KEY not set - the agents will not be able to call the OpenAI API."
        )

    args = parse_args()
    result = await run_full_pipeline(
        source_files=args.source_files,
        row_limit=args.row_limit,
        output_dir=args.output_dir,
        target_column=args.target_column,
        id_columns=args.id_columns,
        model_types=args.model_types,
    )

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for stage in ("schema_mapping", "feature_engineering", "training"):
        stage_result = result.get(stage, {})
        print(
            f"  {stage}: {stage_result.get('status')} (success={stage_result.get('success')})"
        )
        if stage_result.get("error"):
            print(f"    error: {stage_result['error']}")

    all_success = all(
        result.get(s, {}).get("success")
        for s in ("schema_mapping", "feature_engineering", "training")
    )
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
