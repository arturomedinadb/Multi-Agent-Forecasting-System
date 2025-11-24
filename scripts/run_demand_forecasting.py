"""
Usage of the routing-based demand forecasting workflow.
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents import Agent, Runner, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from ai_forecasting_agents.demand_forecasting.agents.demand_forecasting_agent import training_agent
from ai_forecasting_agents.demand_forecasting.agents.feature_engineering_agent import orchestrator_agent

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("   The agent will work but AI recommendations may not be available")
        print("   Set your API key with: export OPENAI_API_KEY='your-key-here'")
    
    mapped_data_file = Path(__file__).parent.parent / "data" / "mapping_target.csv"
    if not mapped_data_file.exists():
        print(f"Error: Data file not found at {mapped_data_file}")
        print("   Please ensure the mapping_target.csv file exists in the data directory")
        return 1
    
    # Create conversation ID for tracing
    conversation_id = str(uuid.uuid4().hex[:16])

    # Create a session instance with conversation ID.
    session = SQLAlchemySession.from_url(
        conversation_id,
        url="sqlite+aiosqlite:///demand_forecasting_database.db",
        create_tables=True,
    )
    
    # ------ Run Feature Engineering Workflow ------

    print("-" * 60)
    print("FEATURE ENGINEERING WORKFLOW EXECUTION")
    print("-" * 60)

    feature_input_file=str(mapped_data_file),
    feature_output_file=str(Path(__file__).parent.parent / "output" / "feature_engineered_output.csv"),
    target_column="units_sold"

    initial_prompt = f"""
    You are tasked with feature engineering for demand forecasting.
    
    INPUT FILE: {feature_input_file}
    OUTPUT FILE: {feature_output_file}
    TARGET COLUMN: {target_column}
    
    Please analyze the data, generate intelligent feature recommendations, and execute the feature engineering pipeline.
    
    The workflow should:
    1. Analyze the input dataset structure and characteristics
    2. Generate AI-powered feature engineering recommendations
    3. Execute the complete feature engineering pipeline
    4. Save the engineered dataset to the output file
    5. Provide comprehensive results and metrics
    
    Focus on creating features that will improve demand forecasting accuracy for retail/sales data.
    """

    # Run the workflow using routing
    with trace("Feature Engineering", group_id=conversation_id):
        feature_engineering_result = await Runner.run(orchestrator_agent, input=initial_prompt, session=session)
    
    # Display results
    if feature_engineering_result:
        print("\n---------- Workflow Completed ----------")
        print(f"   Status: {'SUCCESS' if feature_engineering_result else 'FAILED'}")
        print(f"   Final Output: \n{feature_engineering_result.final_output}")
    else:
        print("\nFeature engineering failed")

    # ------ Run Training Workflow ------
    
    print("-" * 60)
    print("DEMAND FORECASTING WORKFLOW EXECUTION")
    print("-" * 60)
    
    # Set up paths
    training_input_file = feature_output_file
    training_output_dir = str(Path(__file__).parent.parent / "output" / "training_results")
    inference_dir = str(Path(__file__).parent.parent / "output" / "training_results" / "inference")
    id_columns = ["transaction_date", "product_id", "store_id"]
    target_column = "units_sold"
    
    # Create output directory
    os.makedirs(training_output_dir, exist_ok=True)
    
    # Create initial message
    initial_message = f"""
    Train demand forecasting models with these parameters:
    - Input file: {training_input_file}
    - Output directory: {training_output_dir}
    - Inference directory: {inference_dir}
    - ID columns: {id_columns}
    - Target column: {target_column}
    - Model types: xgboost, lightgbm, catboost
    
    Start by creating model configurations and training the models.
    """
    
    # Run the workflow using routing
    with trace("Demand Forecasting Training", group_id=conversation_id):
        training_result = await Runner.run(training_agent, input=initial_message, max_turns=150, session=session)
    
    print("\n---------- Workflow Completed ----------")
    print(f"Conversation ID: {conversation_id}")
    print(f"Result: {training_result}")

    # ------ Fetch the conversation history from db ------

    print("\n---------- Conversation History ----------")
    items = await session.get_items()
    for i, msg in enumerate(items, 1):
        role = msg.get("role", "role_unknown")
        action_type = msg.get("type", "type_unknown")
        content = msg.get("content", "")
        status = msg.get("status", "status_unknown")
        print(f"  {i}. {role} ({action_type}): {content} ({status})\n")

    print(f"\nFetched {len(items)} conversation history.")
    
    return training_result

if __name__ == "__main__":
    asyncio.run(main())