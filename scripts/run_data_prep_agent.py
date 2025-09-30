import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Ensure package imports resolve when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mapping_system.schema_mapping.agents.definitions import data_prep_agent
from mapping_system.schema_mapping.tools.functions import load_and_describe_dataset
from agents import Runner


async def main():
    print("=== Running DataPrepAgent (Top 5 Rows) ===")

    source_files = [
        os.path.join(PROJECT_ROOT, "data/transaction_like_synth (1).csv"),
        os.path.join(PROJECT_ROOT, "data/product_like_synth_wBrand.csv"),
        os.path.join(PROJECT_ROOT, "data/store_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/holidays.csv"),
        os.path.join(PROJECT_ROOT, "data/promotion_like_synth.csv"),
        os.path.join(PROJECT_ROOT, "data/weather_monthly.csv"),
    ]

    prompt = f"""
    Analyze the following datasets and return a JSON array of per-file metadata.
    Use ONLY the top 5 rows per dataset. For each file, call the tool and compile results.

    Source Data Files: {source_files}
    """

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


