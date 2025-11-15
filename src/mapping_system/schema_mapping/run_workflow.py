"""
Main entry point for the schema mapping workflow using orchestrator pattern.

Usage:
    poetry run schema-mapper

Or directly:
    python -m mapping_system.schema_mapping.run_workflow
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from agents import Runner
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from .agents.definitions import create_workflow_orchestrator_agent
from .schemas.models import DemandForecastingRecord
from .prompts.factory import get_renderer

# Load environment variables once at module import
load_dotenv()

# Get project root for locating default assets
PROJECT_ROOT = Path(__file__).resolve().parents[3]


async def run_full_workflow(
    source_files: List[str],
    row_limit: int = 10,
    output_dir: str | None = None,
) -> dict:
    """Execute the schema mapping workflow using orchestrator agent with session memory."""
    resolved_output = output_dir or str(PROJECT_ROOT / "output")
    
    # Create output directory for session database
    os.makedirs(resolved_output, exist_ok=True)
    
    # Generate a unique session ID for this workflow run
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # Create SQLAlchemy session for conversation memory
    # This allows the orchestrator to remember all agent interactions
    db_path = f"{resolved_output}/workflow_sessions.db"
    session = SQLAlchemySession.from_url(
        session_id=session_id,
        url=f"sqlite+aiosqlite:///{db_path}",
        create_tables=True,
    )

    print("=" * 70)
    print("SCHEMA MAPPING AGENT WORKFLOW")
    print("=" * 70)
    print(f"\nSession ID: {session_id}")
    print(f"Session DB: {db_path}")
    print(f"\nProcessing {len(source_files)} datasets:")
    for file_path in source_files:
        print(f"  - {Path(file_path).name}")
    print(f"\nSampling {row_limit} rows per dataset")
    print(f"Output directory: {resolved_output}\n")

    # Set environment variables for agent tools
    os.environ["AGENT_ROW_LIMIT"] = str(row_limit)
    os.environ["AGENT_OUTPUT_DIR"] = resolved_output

    # Get target schema
    target_schema_dict = DemandForecastingRecord.model_json_schema()
    target_schema_json = json.dumps(target_schema_dict, indent=2)

    renderer = get_renderer()
    orchestrator_instructions = renderer.render(
        "WorkflowOrchestrator",
        source_files=source_files,
        row_limit=row_limit,
        output_dir=resolved_output,
        target_schema_json=target_schema_json,
    )
    orchestrator_agent = create_workflow_orchestrator_agent(orchestrator_instructions)

    print("Starting workflow with WorkflowOrchestrator...\n")
    print("=" * 70 + "\n")

    source_file_lines = "\n".join(f"- {f}" for f in source_files)
    initial_message = f"""
Run the schema mapping workflow end-to-end using the configuration below.

Source CSV files ({len(source_files)}):
{source_file_lines}

Settings:
- Sample rows per file: {row_limit}
- Output directory: {resolved_output}

Target schema (JSON):
{target_schema_json}

Your first response MUST be a call to transfer_to_data_prep_agent using this request.
After each agent completes, continue delegating in the prescribed order and report the final results.
""".strip()

    # Run the workflow starting with the orchestrator agent.
    result = await Runner.run(
        orchestrator_agent,
        initial_message,
        session=session,
        max_turns=100,  # Allow sufficient turns for multi-agent workflow
    )
    
    # Retrieve conversation history for logging
    all_messages = await session.get_items()
    print(f"\n\nWorkflow completed with {len(all_messages)} conversation turns")
    
    # Show agent interaction summary
    agent_counts = {}
    for msg in all_messages:
        agent_name = msg.get("name", "unknown")
        agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
    
    print("\nAgent Interaction Summary:")
    for agent_name, count in sorted(agent_counts.items()):
        print(f"  {agent_name}: {count} messages")

    return {
        "status": "success",
        "session_id": session_id,
        "output_dir": resolved_output,
        "db_path": db_path,
        "total_turns": len(all_messages),
        "agent_interactions": agent_counts,
        "final_output": result.final_output if hasattr(result, 'final_output') else str(result),
    }


def main() -> None:
    """CLI entry point."""
    default_files = [
        PROJECT_ROOT / "data" / "transaction_like_synth.csv",
        PROJECT_ROOT / "data" / "product_like_synth_wBrand.csv",
        PROJECT_ROOT / "data" / "store_like_synth.csv",
        PROJECT_ROOT / "data" / "holidays.csv",
        PROJECT_ROOT / "data" / "promotion_like_synth.csv",
        PROJECT_ROOT / "data" / "weather_monthly.csv",
        PROJECT_ROOT / "data" / "CPI-monthly.csv",
        PROJECT_ROOT / "data" / "employment_data.csv",
        PROJECT_ROOT / "data" / "GDP_monthly.csv",
        PROJECT_ROOT / "data" / "Population.csv",
    ]

    source_files: List[str] = []
    for file_path in default_files:
        if file_path.exists():
            source_files.append(str(file_path))
        else:
            print(f"⚠️  Warning: File not found: {file_path}")

    if not source_files:
        print("❌ Error: No valid source files found!")
        sys.exit(1)

    row_limit = int(os.getenv("AGENT_ROW_LIMIT", "10"))
    output_dir = os.getenv("AGENT_OUTPUT_DIR", str(PROJECT_ROOT / "output"))

    try:
        result = asyncio.run(
            run_full_workflow(
                source_files=source_files,
                row_limit=row_limit,
                output_dir=output_dir,
            )
        )
    except Exception as exc:
        print(f"\n❌ Workflow failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nSession ID: {result.get('session_id')}")
    print(f"Session DB: {result.get('db_path')}")
    print(f"Total conversation turns: {result.get('total_turns')}")
    print(f"\nOutput directory: {result.get('output_dir')}")
    print(f"Final output: {result.get('final_output')}")
    print("\n" + "=" * 70 + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
