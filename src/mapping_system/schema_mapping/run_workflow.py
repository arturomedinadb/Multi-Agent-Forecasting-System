"""
Main entry point for the schema mapping workflow.

This script runs the complete 4-agent workflow:
1. DataPrepAgent - Analyze source datasets
2. ColumnMappingAgent - Create column mappings
3. DataIntegrationAgent - Merge into final dataset
4. SchemaMappingEvaluationAgent - Evaluate quality

Usage:
    poetry run schema-mapper
    
Or directly:
    python -m mapping_system.schema_mapping.run_workflow
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from .agents.definitions import (
    data_prep_agent,
    column_mapping_agent,
    data_integration_agent,
    schema_mapping_evaluation_agent,
    schema_mapping_handoff_filter,
    integration_handoff_filter,
    evaluation_handoff_filter,
    HandoffInputData,
)
from .schemas.models import DemandForecastingRecord
from .prompts.factory import get_renderer

# Load environment variables
load_dotenv()

# Conditional imports for OpenAI Agents SDK
try:
    from agents import Runner, RawResponsesStreamEvent, trace
    from openai import OpenAIError
    from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    print("WARNING: OpenAI Agents SDK not available. Running in fallback mode.")


# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


async def stream_agent(agent, inputs, trace_label):
    """Helper to stream a single agent turn and return the runner result."""
    if not AGENTS_SDK_AVAILABLE:
        raise RuntimeError("OpenAI Agents SDK is required to run the workflow.")
    
    with trace(trace_label):
        result = Runner.run_streamed(agent, input=inputs)
        async for ev in result.stream_events():
            if not isinstance(ev, RawResponsesStreamEvent):
                continue
            payload = ev.data
            if isinstance(payload, ResponseTextDeltaEvent):
                print(payload.delta, end="", flush=True)
            elif isinstance(payload, ResponseContentPartDoneEvent):
                print()
    return result


async def run_full_workflow(
    source_files: List[str],
    row_limit: int = 10,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Run the complete 4-agent schema mapping workflow.
    
    Args:
        source_files: List of paths to source CSV files
        row_limit: Number of rows to sample from each dataset
        output_dir: Directory for output files (defaults to PROJECT_ROOT/output)
    
    Returns:
        Dictionary with workflow results and paths to generated files
    """
    if not AGENTS_SDK_AVAILABLE:
        raise RuntimeError(
            "OpenAI Agents SDK is required. Install with:\n"
            "  pip install openai-agents>=0.3.0"
        )
    
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output")
    
    print("=" * 70)
    print("🚀 SCHEMA MAPPING AGENT WORKFLOW")
    print("=" * 70)
    print(f"\n📁 Processing {len(source_files)} datasets:")
    for f in source_files:
        print(f"  - {Path(f).name}")
    print(f"\n📊 Sampling {row_limit} rows per dataset")
    print(f"📂 Output directory: {output_dir}\n")
    
    # Prepare target schema
    target_schema_json = json.dumps(
        DemandForecastingRecord.model_json_schema(), 
        indent=2
    )
    
    # Get prompt renderer
    renderer = get_renderer()
    
    # Build initial prompt for DataPrepAgent
    dataprep_prompt = renderer.render(
        "DataPrepAgent",
        source_files=source_files,
        row_limit=row_limit,
        target_schema_json=target_schema_json,
    )
    
    inputs: List[Dict[str, Any]] = [{
        "role": "user",
        "content": dataprep_prompt
    }]
    
    try:
        # ===== TURN 1: Data Preparation =====
        print("\n" + "=" * 70)
        print("🔍 AGENT 1: DATA PREPARATION")
        print("=" * 70 + "\n")
        
        r1 = await stream_agent(data_prep_agent, inputs, "Agent 1: DataPrep")
        prep_history = r1.to_input_list()
        
        print("\n\n✅ Data preparation complete")
        print("---\n")
        
        # ===== TURN 2: Column Mapping =====
        print("\n" + "=" * 70)
        print("🗺️  AGENT 2: COLUMN MAPPING")
        print("=" * 70 + "\n")
        
        mapping_handoff = schema_mapping_handoff_filter(
            HandoffInputData(
                input_history=prep_history, 
                pre_handoff_items=tuple(), 
                new_items=tuple()
            )
        )
        mapping_inputs = mapping_handoff.input_history
        
        r2 = await stream_agent(column_mapping_agent, mapping_inputs, "Agent 2: ColumnMapping")
        mapping_history = r2.to_input_list()
        
        print("\n\n✅ Column mapping complete")
        print("---\n")
        
        # ===== TURN 3: Data Integration =====
        print("\n" + "=" * 70)
        print("🔗 AGENT 3: DATA INTEGRATION")
        print("=" * 70 + "\n")
        
        integration_handoff = integration_handoff_filter(
            HandoffInputData(
                input_history=mapping_history, 
                pre_handoff_items=tuple(), 
                new_items=tuple()
            )
        )
        integration_inputs = integration_handoff.input_history
        
        r3 = await stream_agent(data_integration_agent, integration_inputs, "Agent 3: Integration")
        integration_history = r3.to_input_list()
        
        print("\n\n✅ Data integration complete")
        print("---\n")
        
        # ===== TURN 4: Evaluation =====
        print("\n" + "=" * 70)
        print("📊 AGENT 4: QUALITY EVALUATION")
        print("=" * 70 + "\n")
        
        evaluation_handoff = evaluation_handoff_filter(
            HandoffInputData(
                input_history=integration_history, 
                pre_handoff_items=tuple(), 
                new_items=tuple()
            )
        )
        evaluation_inputs = evaluation_handoff.input_history
        
        await stream_agent(schema_mapping_evaluation_agent, evaluation_inputs, "Agent 4: Evaluation")
        
        print("\n\n✅ Evaluation complete")
        
        # ===== Workflow Summary =====
        print("\n" + "=" * 70)
        print("🎉 WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"\n📁 Output files:")
        print(f"  - Final dataset: {output_dir}/final_mapped_dataset.csv")
        print(f"  - Mapped CSVs: {output_dir}/mapped/")
        print(f"  - Evaluation: {output_dir}/evaluations/")
        print("\n" + "=" * 70 + "\n")
        
        return {
            "status": "success",
            "output_dir": output_dir,
            "final_dataset": f"{output_dir}/final_mapped_dataset.csv",
            "mapped_dir": f"{output_dir}/mapped/",
            "evaluation_dir": f"{output_dir}/evaluations/",
        }
        
    except OpenAIError as err:
        print(f"\n❌ OpenAI API error: {err}")
        return {"status": "error", "error": str(err)}
    except Exception as err:
        print(f"\n❌ Unexpected error: {err}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(err)}


def main():
    """
    Main entry point for CLI execution.
    
    Reads source files from environment or uses defaults.
    """
    # Default source files
    default_files = [
        PROJECT_ROOT / "data" / "transaction_like_synth.csv",
        PROJECT_ROOT / "data" / "product_like_synth_wBrand.csv",
        PROJECT_ROOT / "data" / "store_like_synth.csv",
        PROJECT_ROOT / "data" / "holidays.csv",
        PROJECT_ROOT / "data" / "promotion_like_synth.csv",
        PROJECT_ROOT / "data" / "weather_monthly.csv",
    ]
    
    # Convert to strings and verify existence
    source_files = []
    for f in default_files:
        if f.exists():
            source_files.append(str(f))
        else:
            print(f"⚠️  Warning: File not found: {f}")
    
    if not source_files:
        print("❌ Error: No valid source files found!")
        sys.exit(1)
    
    # Get configuration from environment
    row_limit = int(os.getenv("AGENT_ROW_LIMIT", "10"))
    output_dir = os.getenv("AGENT_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
    
    # Run the workflow
    result = asyncio.run(run_full_workflow(
        source_files=source_files,
        row_limit=row_limit,
        output_dir=output_dir,
    ))
    
    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()

