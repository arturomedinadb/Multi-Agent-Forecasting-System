import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from mapping_system.schema_mapping.prompts.factory import get_renderer

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mapping_system.schema_mapping.agents.definitions import (
    data_prep_agent,
    column_mapping_agent,
    data_integration_agent,
    schema_mapping_evaluation_agent,
    schema_mapping_handoff_filter,
    integration_handoff_filter,
    evaluation_handoff_filter,
    HandoffInputData,
)
from mapping_system.schema_mapping.schemas.models import DemandForecastingRecord
from agents import Runner, RawResponsesStreamEvent, trace
from openai import OpenAIError
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent


async def stream_agent(agent, inputs, trace_label):
    """Helper to stream a single agent turn and return the runner result."""

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


async def main():
    print("=== Running 4-Agent Workflow (Top 10 Rows + DeepEval) ===")

    source_files = [
        os.path.join(PROJECT_ROOT, "data/transaction_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/product_like_synth_wBrand.csv"),
        os.path.join(PROJECT_ROOT, "data/store_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/holidays.csv"),
        os.path.join(PROJECT_ROOT, "data/promotion_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/weather_monthly.csv"),
    ]

    target_schema_json = json.dumps(DemandForecastingRecord.model_json_schema(), indent=2)

    renderer = get_renderer()
    dataprep_prompt = renderer.render(
        "DataPrepAgent",
        source_files=source_files,
        row_limit=10,  # keep same behavior as before
        target_schema_json=target_schema_json,
    )

    inputs: List[Dict[str, Any]] = [{
        "role": "user",
        "content": dataprep_prompt
    }]


    try:
        # Turn 1: Data Preparation
        r1 = await stream_agent(data_prep_agent, inputs, "Agent 1: DataPrep")
        prep_history = r1.to_input_list()
        print("\n--- Handoff complete: DataPrepAgent -> ColumnMappingAgent ---\n")

        # Prepare inputs for Column Mapping
        mapping_handoff = schema_mapping_handoff_filter(
            HandoffInputData(input_history=prep_history, pre_handoff_items=tuple(), new_items=tuple())
        )
        mapping_inputs = mapping_handoff.input_history

        # Turn 2: Column Mapping
        r2 = await stream_agent(column_mapping_agent, mapping_inputs, "Agent 2: ColumnMapping")
        mapping_history = r2.to_input_list()
        print("\n--- Handoff complete: ColumnMappingAgent -> DataIntegrationAgent ---\n")

        # Prepare inputs for Data Integration
        integration_handoff = integration_handoff_filter(
            HandoffInputData(input_history=mapping_history, pre_handoff_items=tuple(), new_items=tuple())
        )
        integration_inputs = integration_handoff.input_history

        # Turn 3: Data Integration
        r3 = await stream_agent(data_integration_agent, integration_inputs, "Agent 3: Integration")
        integration_history = r3.to_input_list()
        print("\n--- Handoff complete: DataIntegrationAgent -> EvaluationAgent ---\n")

        # Prepare inputs for Evaluation
        evaluation_handoff = evaluation_handoff_filter(
            HandoffInputData(input_history=integration_history, pre_handoff_items=tuple(), new_items=tuple())
        )
        evaluation_inputs = evaluation_handoff.input_history

        # Turn 4: Evaluation
        await stream_agent(schema_mapping_evaluation_agent, evaluation_inputs, "Agent 4: Evaluation")

        print("=== 4-Agent run complete ===")
    except NotImplementedError:
        print("Runner is a stub; streaming handoffs are unavailable here. Run in an environment with the Agents SDK.")
    except OpenAIError as err:
        print(f"OpenAI API error: {err}")
    except Exception as err:
        print(f"Unexpected error during agent run: {err}")


if __name__ == "__main__":
    asyncio.run(main())
