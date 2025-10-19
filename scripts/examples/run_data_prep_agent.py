import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Ensure package imports resolve when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mapping_system.schema_mapping.agents.definitions import data_prep_agent
from mapping_system.schema_mapping.tools.functions import load_and_describe_dataset
from agents import Runner
from mapping_system.schema_mapping.prompts.factory import get_renderer



async def main():
    # Get row limit from environment or use default
    row_limit = int(os.getenv("AGENT_ROW_LIMIT", "10"))
    
    print(f"=== Running DataPrepAgent (Top {row_limit} Rows) ===")

    source_files = [
        os.path.join(PROJECT_ROOT, "data/transaction_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/product_like_synth_wBrand.csv"),
        os.path.join(PROJECT_ROOT, "data/store_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/holidays.csv"),
        os.path.join(PROJECT_ROOT, "data/promotion_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/weather_monthly.csv"),
    ]

    renderer = get_renderer()
    prompt = renderer.render(
        "DataPrepAgent",
        source_files=source_files,
        row_limit=row_limit,  
    )

    try:
        result = await Runner.run(data_prep_agent, [{"role": "user", "content": prompt}])
        print("\n=== Agent Output ===")
        print(result.output if hasattr(result, 'output') else result)
        return
    except NotImplementedError:
        print("Runner not available; using fallback execution for Agent 1...")

    compiled = []
    for path in source_files:
        meta_json = load_and_describe_dataset(path)
        meta = json.loads(meta_json)
        compiled.append(meta)

    print("\n=== Agent Output (Fallback) ===")
    print(json.dumps(compiled, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


