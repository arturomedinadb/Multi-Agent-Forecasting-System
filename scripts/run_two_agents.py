import asyncio
import json
import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mapping_system.schema_mapping.agents.definitions import data_prep_agent, column_mapping_agent
from mapping_system.schema_mapping.schemas.models import DemandForecastingRecord
from agents import Runner, RawResponsesStreamEvent, trace
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent


async def main():
    print("=== Running DataPrepAgent -> ColumnMappingAgent (Top 5 Rows) ===")

    source_files = [
        os.path.join(PROJECT_ROOT, "data/transaction_like_synth (1).csv"),
        os.path.join(PROJECT_ROOT, "data/product_like_synth_wBrand.csv"),
        os.path.join(PROJECT_ROOT, "data/store_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/holidays.csv"),
        os.path.join(PROJECT_ROOT, "data/promotion_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/weather_monthly.csv"),
    ]

    # Use streamed execution to allow actual handoff between agents
    target_schema_json = json.dumps(DemandForecastingRecord.model_json_schema(), indent=2)
    conversation_id = "handoff-" + os.path.basename(PROJECT_ROOT)

    inputs: List[Dict[str, Any]] = [{
        "role": "user",
        "content": (
            "Analyze the following datasets and return a JSON array of per-file metadata.\n"
            "Use ONLY the top 5 rows per dataset. For each file, call the tool and compile results.\n\n"
            f"Source Data Files: {source_files}\n\n"
            "Target Schema:\n" + target_schema_json
        )
    }]

    try:
        # Turn 1: DataPrepAgent (will handoff per Agent 1 handoffs config)
        with trace("Agent 1: DataPrep"):
            r1 = Runner.run_streamed(data_prep_agent, input=inputs)
            async for ev in r1.stream_events():
                if not isinstance(ev, RawResponsesStreamEvent):
                    continue
                d = ev.data
                if isinstance(d, ResponseTextDeltaEvent):
                    print(d.delta, end="", flush=True)
                elif isinstance(d, ResponseContentPartDoneEvent):
                    print()

        # Prepare next turn using outputs and current agent from r1
        inputs = r1.to_input_list()
        agent = r1.current_agent

        # Turn 2: ColumnMappingAgent after handoff
        with trace("Agent 2: ColumnMapping"):
            r2 = Runner.run_streamed(agent, input=inputs)
            async for ev in r2.stream_events():
                if not isinstance(ev, RawResponsesStreamEvent):
                    continue
                d = ev.data
                if isinstance(d, ResponseTextDeltaEvent):
                    print(d.delta, end="", flush=True)
                elif isinstance(d, ResponseContentPartDoneEvent):
                    print()

        print("\n=== Handoff test complete ===")
    except NotImplementedError:
        print("Runner is a stub; streaming handoffs are unavailable here. Use run_data_prep_agent.py or run in a full Runner environment.")


if __name__ == "__main__":
    asyncio.run(main())


