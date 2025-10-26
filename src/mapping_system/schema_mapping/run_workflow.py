"""
Main entry point for the schema mapping workflow using an explicit orchestrator.

Usage:
    poetry run schema-mapper

Or directly:
    python -m mapping_system.schema_mapping.run_workflow
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from .orchestrator import SchemaMappingOrchestrator

# Load environment variables once at module import
load_dotenv()

# Get project root for locating default assets
PROJECT_ROOT = Path(__file__).resolve().parents[3]


async def run_full_workflow(
    source_files: List[str],
    row_limit: int = 10,
    output_dir: str | None = None,
) -> Dict[str, str]:
    """Execute the schema mapping workflow through the orchestrator."""
    resolved_output = output_dir or str(PROJECT_ROOT / "output")

    print("=" * 70)
    print("🚀 SCHEMA MAPPING AGENT WORKFLOW")
    print("=" * 70)
    print(f"\n📁 Processing {len(source_files)} datasets:")
    for file_path in source_files:
        print(f"  - {Path(file_path).name}")
    print(f"\n📊 Sampling {row_limit} rows per dataset")
    print(f"📂 Output directory: {resolved_output}\n")

    orchestrator = SchemaMappingOrchestrator(
        source_files=source_files,
        row_limit=row_limit,
        output_dir=resolved_output,
    )
    return await orchestrator.execute()


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
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🎉 WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\n🆔 Run ID: {result.get('run_id')}")
    print(f"📁 Output root: {result.get('output_root')}")
    print(f"📄 Final dataset: {result.get('final_dataset')}")
    print(f"🗂️  Mapping plan: {result.get('mapping_plan')}")
    print(f"📑 Manifest: {result.get('mapping_manifest')}")
    print(f"🧾 Source metadata: {result.get('source_metadata')}")
    print("\n" + "=" * 70 + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
