import asyncio
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.definitions import (
    data_prep_agent,
    column_mapping_agent,
    data_integration_agent,
    quality_assessor_agent,
    # orchestrator_agent,  # Commented out for direct chain approach
)
from agents import Runner
from src.schemas.models import DemandForecastingRecord

# === Direct Chain Workflow ===

async def main():
    """
    Main function to run the multi-agent schema mapping workflow.
    It now processes the top 5 rows of each specified dataset.
    """
    print("=== Starting Multi-Agent Schema Mapping Workflow (Top 5 Rows) ===")

    # Use all relevant datasets, excluding test samples and target benchmarks
    source_files = [
        "data/transaction_like_synth.csv",
        "data/product_like_synth_wBrand.csv",
        "data/store_like_synth.csv",
        "data/holidays.csv",
        "data/promotion_like_synth.csv"
    ]

    output_file = "data/final_mapped_demand_forecast_5_rows.csv"

    # The target schema is derived directly from the Pydantic model
    target_schema_dict = DemandForecastingRecord.model_json_schema()
    target_schema_json = json.dumps(target_schema_dict, indent=2)

    # A goal-driven prompt for the orchestrator
    initial_prompt = f"""
    GOAL: Produce a validated, schema-mapped dataset and a final quality report.

    INPUTS:
    - Source Data Files (use top 5 rows): {source_files}
    - Output File Path: '{output_file}'
    - Target Schema:
    {target_schema_json}

    Please formulate a plan and achieve this goal using your available specialist agents.
    """
    
    print(f"\nProcessing top 5 rows from {len(source_files)} datasets...")
    print(f"Target Output: {output_file}")
    print("\n" + "="*60)

    # Run the direct chain starting with data_prep_agent
    # The agents will automatically hand off through the chain: DataPrepAgent → ColumnMappingAgent → DataIntegrationAgent → QualityAssessorAgent
    result = await Runner.run(data_prep_agent, [{"role": "user", "content": initial_prompt}])

    print("\n" + "="*60)
    print("=== Workflow Complete ===")
    print(f"\nCheck the output file at: '{output_file}'")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
