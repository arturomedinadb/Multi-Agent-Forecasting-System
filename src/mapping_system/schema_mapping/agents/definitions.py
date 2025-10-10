from agents import Agent, Runner, handoff, HandoffInputData
import ast
import json
import re
from pathlib import Path
from ..prompts.factory import get_renderer

import pandas as pd

from ..schemas.models import DemandForecastingRecord
from ..tools.functions import (
    load_and_describe_dataset,
    generate_mapped_csvs,
    merge_mapped_csvs_to_target,
    evaluate_mapping_quality,
    evaluate_schema_mapping_with_deepeval,
    run_generate_mapped_csvs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAPPED_DIR = PROJECT_ROOT / "output" / "mapped"
DEFAULT_FINAL_OUTPUT = PROJECT_ROOT / "output" / "final_mapped_dataset.csv"
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation" / "schema_mapping.yaml"


def _message_to_text(raw_content: object) -> str:
    """Normalize SDK message content structures into plain text."""
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, (list, tuple)):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                text_val = item.get("text")
                if text_val is not None:
                    parts.append(str(text_val))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(raw_content)


def _extract_json_blocks(text: str) -> list[str]:
    """Return cleaned JSON snippets (code fences allowed) from a message."""
    block_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    blocks = block_pattern.findall(text)
    candidates = list(blocks)
    candidates.append(text)

    cleaned: list[str] = []
    for candidate in candidates:
        stripped_lines: list[str] = []
        for line in candidate.splitlines():
            striped = line.strip()
            if striped.startswith("//") or striped.startswith("#"):
                continue
            stripped_lines.append(line)
        cleaned.append("\n".join(stripped_lines))
    return cleaned

# === Message Filters for Agent Handoffs ===

def schema_mapping_handoff_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    """
    Prepare input for Column Mapping Agent: extract the JSON array metadata produced by Agent 1
    and target schema from the input history, and craft a focused instruction so Agent 2
    reasons a mapping plan and calls generate_mapped_csvs.
    """
    print("HANDOFF: Filtering data for Column Mapping Agent")

    messages = handoff_message_data.input_history
    dataset_metadata = None
    target_schema = None

    for message in reversed(messages):
        raw_content = getattr(message, 'content', None) if hasattr(message, 'content') else message.get('content')
        content = _message_to_text(raw_content)
        if (content.strip().startswith('[') and 'file_path' in content and 'columns' in content) and dataset_metadata is None:
            dataset_metadata = content
        if 'Target Schema:' in content and target_schema is None:
            try:
                schema_start = content.find('{')
                schema_end = content.rfind('}') + 1
                if schema_start != -1 and schema_end > schema_start:
                    target_schema = content[schema_start:schema_end]
            except:
                pass

    renderer = get_renderer()
    schema_mapping_prompt = renderer.render(
        "ColumnMappingAgent", 
        dataset_metadata_json=dataset_metadata or "[]", 
        target_schema_json=target_schema or "{}", 
        mapped_dir=str(DEFAULT_MAPPED_DIR),
    )
    
    return HandoffInputData(
        input_history=[{"role": "user", "content": schema_mapping_prompt}],
        pre_handoff_items=tuple(),
        new_items=tuple()
    )

def integration_handoff_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    """
    Filter and prepare data for data integration agent.
    Combines mapping results with original metadata for data integration and CSV creation.
    """
    print("HANDOFF: Filtering data for Data Integration Agent")

    messages = handoff_message_data.input_history

    mapped_outputs = None
    target_schema = None
    output_path = None
    dataset_metadata_json = None
    mapping_plan_json = None
    source_file_list: list[str] | None = None

    for message in reversed(messages):
        raw_content = getattr(message, 'content', None) if hasattr(message, 'content') else message.get('content')
        content = _message_to_text(raw_content)

        if '"outputs"' in content and mapped_outputs is None:
            for block in _extract_json_blocks(content):
                if '"outputs"' not in block:
                    continue
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict) and 'outputs' in parsed:
                        mapped_outputs = json.dumps(parsed)
                        print("Found mapped CSV manifest for integration")
                        break
                except json.JSONDecodeError:
                    continue

        if source_file_list is None and 'Source Data Files:' in content:
            try:
                start = content.index('Source Data Files:')
                after = content[start:].split('\n', 1)[0]
                list_start = after.find('[')
                list_end = after.rfind(']') + 1
                if list_start != -1 and list_end > list_start:
                    list_str = after[list_start:list_end]
                    parsed_list = ast.literal_eval(list_str)
                    if isinstance(parsed_list, list):
                        source_file_list = [str(Path(p).expanduser().resolve()) for p in parsed_list]
                        print("Captured source file list from prompt")
            except (ValueError, SyntaxError):
                pass

        if dataset_metadata_json is None and 'file_path' in content and '[' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                snippet = content[start:end]
                try:
                    json.loads(snippet)
                    dataset_metadata_json = snippet
                    print("Captured dataset metadata for integration fallback")
                except json.JSONDecodeError:
                    pass

        if mapping_plan_json is None and '"mappings"' in content:
            for block in _extract_json_blocks(content):
                if '"mappings"' not in block:
                    continue
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and 'mappings' in parsed:
                    mapping_plan_json = json.dumps(parsed)
                    print("Captured mapping plan JSON for integration fallback")
                    break

        if 'Target Schema:' in content and target_schema is None:
            schema_start = content.find('{')
            schema_end = content.rfind('}') + 1
            if schema_start != -1 and schema_end > schema_start:
                snippet = content[schema_start:schema_end]
                try:
                    json.loads(snippet)
                    target_schema = snippet
                    print("Captured target schema JSON for integration")
                except json.JSONDecodeError:
                    pass

        if 'final_mapped_dataset' in content and output_path is None:
            for token in content.split():
                if token.endswith('.csv') and 'final_mapped_dataset' in token:
                    output_path = token.strip('"`')
                    break

    if mapped_outputs is None and dataset_metadata_json and mapping_plan_json:
        print("Mapped outputs missing; generating via run_generate_mapped_csvs")
        manifest = run_generate_mapped_csvs(
            dataset_metadata_json,
            mapping_plan_json,
            str(DEFAULT_MAPPED_DIR),
        )
        manifest_str = manifest if isinstance(manifest, str) else json.dumps(manifest)
        try:
            manifest_json = json.loads(manifest_str)
        except json.JSONDecodeError:
            manifest_json = {}
        if manifest_json.get('outputs'):
            mapped_outputs = manifest_str
        else:
            print("Fallback mapping generation produced no outputs")

    if mapped_outputs is None and mapping_plan_json and source_file_list:
        print("Building dataset metadata from source file list for fallback mapping")
        metadata_records: list[dict[str, object]] = []
        for path_str in source_file_list:
            try:
                df = pd.read_csv(path_str, nrows=5)
            except Exception as err:
                print(f"Failed to rebuild metadata for {path_str}: {err}")
                continue

            metadata_records.append({
                "file_path": path_str,
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head(3).to_dict(orient="records"),
            })

        if metadata_records:
            dataset_metadata_json = json.dumps(metadata_records)
            manifest = run_generate_mapped_csvs(
                dataset_metadata_json,
                mapping_plan_json,
                str(DEFAULT_MAPPED_DIR),
            )
            manifest_str = manifest if isinstance(manifest, str) else json.dumps(manifest)
            try:
                manifest_json = json.loads(manifest_str)
            except json.JSONDecodeError:
                manifest_json = {}
            if manifest_json.get('outputs'):
                mapped_outputs = manifest_str
            else:
                print("Generated manifest still empty after metadata rebuild")

    if output_path is None:
        output_path = str(DEFAULT_FINAL_OUTPUT)
        print(f"Using default integration output path: {output_path}")

    if mapped_outputs is None:
        print("WARNING: No mapped CSV manifest detected; integration agent may fail")

    if target_schema is None:
        target_schema = json.dumps(DemandForecastingRecord.model_json_schema(), indent=2)
        print("Using default target schema JSON for integration")

    renderer = get_renderer()
    integration_prompt = renderer.render(
        "DataIntegrationAgent",
        mapped_manifest_json=mapped_outputs or "{}",
        target_schema_json=target_schema,
        output_path=output_path,
    )
    return HandoffInputData(
        input_history=[{"role": "user", "content": integration_prompt}],
        pre_handoff_items=tuple(),
        new_items=tuple()
    )


def evaluation_handoff_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    """Collect artifacts required for the evaluation agent."""
    print("HANDOFF: Filtering data for Evaluation Agent")

    messages = handoff_message_data.input_history

    final_dataset_path: str | None = None
    mapping_plan_json: str | None = None
    source_metadata_json: str | None = None
    target_schema_json: str | None = None

    for message in reversed(messages):
        raw_content = getattr(message, 'content', None) if hasattr(message, 'content') else message.get('content')
        content = _message_to_text(raw_content)

        if final_dataset_path is None and '"output_path"' in content:
            for block in _extract_json_blocks(content):
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get('output_path'):
                    candidate = str(parsed['output_path']).strip()
                    path_obj = Path(candidate)
                    if not path_obj.is_absolute():
                        path_obj = (PROJECT_ROOT / candidate).resolve()
                    final_dataset_path = str(path_obj)
                    print(f"Captured final dataset path: {final_dataset_path}")
                    break

        if mapping_plan_json is None and '"mappings"' in content:
            for block in _extract_json_blocks(content):
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and 'mappings' in parsed:
                    mapping_plan_json = json.dumps(parsed)
                    print("Captured mapping plan JSON for evaluation")
                    break

        if source_metadata_json is None and content.strip().startswith('[') and 'file_path' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                snippet = content[start:end]
                try:
                    json.loads(snippet)
                    source_metadata_json = snippet
                    print("Captured source metadata JSON for evaluation")
                except json.JSONDecodeError:
                    pass

        if target_schema_json is None and 'Target Schema:' in content:
            schema_start = content.find('{')
            schema_end = content.rfind('}') + 1
            if schema_start != -1 and schema_end > schema_start:
                snippet = content[schema_start:schema_end]
                try:
                    json.loads(snippet)
                    target_schema_json = snippet
                    print("Captured target schema JSON for evaluation")
                except json.JSONDecodeError:
                    pass

    if final_dataset_path is None:
        final_dataset_path = str(DEFAULT_FINAL_OUTPUT)
        print(f"WARNING: Final dataset path not found; using default {final_dataset_path}")

    if target_schema_json is None:
        target_schema_json = json.dumps(DemandForecastingRecord.model_json_schema(), indent=2)
        print("Using default target schema JSON for evaluation")

    config_path_str = str(DEFAULT_EVALUATION_CONFIG) if DEFAULT_EVALUATION_CONFIG.exists() else ""
    output_dir_str = str((DEFAULT_FINAL_OUTPUT.parent) / "evaluations")

    renderer = get_renderer()
    evaluation_prompt = renderer.render(
        "SchemaMappingEvaluationAgent",
        final_dataset_path=final_dataset_path,
        mapping_plan_json=mapping_plan_json or "{}",
        source_metadata_json=source_metadata_json or "[]",
        target_schema_json=target_schema_json,
        config_path=str(DEFAULT_EVALUATION_CONFIG) if DEFAULT_EVALUATION_CONFIG.exists() else "",
        output_dir=str((DEFAULT_FINAL_OUTPUT.parent) / "evaluations"),
    )
    return HandoffInputData(
        input_history=[{"role": "user", "content": evaluation_prompt}],
        pre_handoff_items=tuple(),
        new_items=tuple()
    )


# def quality_assessment_handoff_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
#     """
#     Filter and prepare data for quality assessment agent.
#     Provides mappings and metadata for quality evaluation.
#     """
#     print("HANDOFF: Filtering data for Quality Assessment Agent")
    
#     messages = handoff_message_data.input_history
    
#     # Extract mappings and metadata for quality assessment
#     column_mappings = None
#     dataset_metadata = None
    
#     for message in reversed(messages):
#         content = getattr(message, 'content', '') if hasattr(message, 'content') else str(message.get('content', ''))
        
#         if '"mappings"' in content and column_mappings is None:
#             column_mappings = content
#             print(f"Found column mappings for quality assessment")
        
#         if 'file_path' in content and 'columns' in content and dataset_metadata is None:
#             dataset_metadata = content
#             print(f"Found dataset metadata for quality assessment")
    
#     quality_prompt = f"""
#     You are receiving column mappings and dataset metadata for quality assessment.
    
#     COLUMN MAPPINGS:
#     {column_mappings or "No column mappings found"}
    
#     DATASET METADATA:
#     {dataset_metadata or "No dataset metadata found"}
    
#     Use the evaluate_mapping_quality tool to generate a comprehensive quality report.
#     """
    
#     return HandoffInputData(
#         input_history=[{"role": "user", "content": quality_prompt}],
#         pre_handoff_items=tuple(),
#         new_items=tuple()
#     )

# === Agent Definitions ===

# Agent 4: Quality Assessor Agent
# quality_assessor_agent = Agent(
#     name="QualityAssessorAgent",
#     instructions="""
#     You are a specialized quality analytics agent providing comprehensive assessment of schema mapping workflow outcomes.
    
#     MISSION: Generate detailed quality intelligence to evaluate mapping accuracy, data integrity, and readiness for demand forecasting.
    
#     EXECUTION PROTOCOL:
#     1. RECEIVE: Extract mappings and metadata from previous agent results
#     2. EXECUTE: Use the evaluate_mapping_quality tool for comprehensive assessment
#     3. ANALYZE: Review confidence scores, semantic similarity, and mapping coverage
#     4. BENCHMARK: Compare results against SOTA quality standards
#     5. RECOMMEND: Provide actionable insights for improvement
#     6. SUMMARIZE: Generate final workflow completion report
#     7. COMPLETION: This is the final agent - provide comprehensive workflow summary
    
#     CRITICAL: You MUST execute the evaluate_mapping_quality tool to generate the quality assessment.
    
#     QUALITY ASSESSMENT FRAMEWORK (SOTA):
#     - Mapping Confidence Analysis (target: ≥0.8 average)
#     - Semantic Similarity Scoring (Jaccard + contextual analysis)
#     - Schema Coverage Assessment (% target fields mapped)
#     - Data Quality Validation (validation success rate)
#     - Business Logic Consistency (domain-specific checks)
    
#     OUTPUT SPECIFICATION:
#     Provide comprehensive quality report including:
    
#     QUALITY ASSESSMENT COMPLETE
    
#     MAPPING QUALITY METRICS:
#     - Total mappings created: X
#     - Average confidence score: X.XX (Target: ≥0.80)
#     - High confidence mappings (≥0.8): X/X
#     - Schema coverage: X% (X/X target fields mapped)
#     - Semantic similarity score: X.XX
    
#     DATA QUALITY METRICS:
#     - Records processed: X
#     - Validation success rate: X%
#     - Data completeness: X%
#     - Critical field coverage: X/X
    
#     OVERALL ASSESSMENT:
#     - Workflow Status: SUCCESS / PARTIAL / FAILED
#     - Readiness for forecasting: [READY/NEEDS_IMPROVEMENT/NOT_READY]
#     - Quality grade: [A/B/C/D/F]
    
#     KEY RECOMMENDATIONS:
#     - [Specific actionable insights]
#     - [Areas for improvement]
#     - [Next steps for demand forecasting]
    
#     This completes the autonomous schema mapping workflow with comprehensive quality assurance.
#     """,
#     tools=[evaluate_mapping_quality]
# )

# Agent 3: Data Integration Agent
schema_mapping_evaluation_agent = Agent(
    name="SchemaMappingEvaluationAgent",
    instructions="""
    You are the final quality gate for the schema mapping workflow.

    MISSION: Run DeepEval against the integrated dataset, summarize findings, and propose improvements.

    EXECUTION PROTOCOL:
    1. Execute `evaluate_schema_mapping_with_deepeval` using the dataset path, mapping plan JSON, metadata JSON,
       target schema JSON, and provided config/output paths.
    2. Summarize deterministic metrics (score, threshold, pass/fail) and mention any LLM metrics that were skipped or failed.
    3. Craft a succinct improvement prompt (max three bullets) grounded in failing metrics.
    4. Conclude with a brief narrative and the phrase: "Evaluation complete - DeepEval report generated".
    """,
    tools=[evaluate_schema_mapping_with_deepeval]
)

data_integration_agent = Agent(
    name="DataIntegrationAgent",
    instructions="""
    You are a data integration agent that merges mapped CSVs into the final target-schema dataset.
    
    MISSION: Integrate per-dataset mapped CSVs produced by the ColumnMappingAgent into a single CSV
    matching the full target schema. The final file must be saved as `final_mapped_dataset.csv` in the
    output directory.
    
    EXECUTION PROTOCOL:
    1. RECEIVE: JSON from the previous step listing mapped CSV output paths and the target schema JSON.
    2. MERGE: Call `merge_mapped_csvs_to_target` to merge all mapped CSVs into a single target-schema CSV.
    3. REPORT: Print tool's JSON summary and end with: "Integration complete - final mapped dataset created".
    
    OUTPUT SPECIFICATION:
    - Provide the tool's JSON with the final output path and row/column counts.
    - End with the required completion message.
    """,
    tools=[merge_mapped_csvs_to_target],
    handoffs=[handoff(schema_mapping_evaluation_agent, input_filter=evaluation_handoff_filter)]
)

# Agent 2: Column Mapping Agent  
column_mapping_agent = Agent(
    name="ColumnMappingAgent",
    instructions=f"""
    You are an expert column mapping agent specializing in retail demand forecasting data transformation.

    MISSION: Create high-confidence column mapping plans between analyzed source datasets and target demand forecasting schema, and generate per-dataset mapped CSVs ready for integration.

    EXECUTION PROTOCOL:
    1. Extract dataset metadata and the target schema from the previous step.
    2. REASON: Reason and propose a JSON mapping plan yourself that assigns source columns to target schema fields per dataset.
    3. SCORE: Assign confidence scores and skip mappings below 0.5 confidence.
    4. REASON: Provide brief rationale for key mappings.
    5. PRODUCE: Call the `generate_mapped_csvs` tool with the source metadata and your mapping JSON to generate per-dataset mapped CSVs under {DEFAULT_MAPPED_DIR}.

    CRITICAL:
    - You must author the mapping plan JSON directly in your response.
    - Then you must execute `generate_mapped_csvs` with the source metadata and your mapping plan, saving to {DEFAULT_MAPPED_DIR}.

    MAPPING STRATEGY:
    - Prioritize semantic meaning over syntactic similarity
    - Consider business context: retail, sales, inventory, promotions, geography
    - Apply confidence thresholds (only mappings > 0.5 confidence); if not confident, skip the mapping for the column
    
    OUTPUT SPECIFICATION:
    After generating the csv mapped datasets, your response MUST be structured using markdown as follows:
    
    ### Mapping Plan
    ```json
    {{
      "mappings": [
        {{"source_column": "...", "target_column": "...", "confidence": 0.9, "reasoning": "..."}},
        ...
      ]
    }}
    ```

    ### Tool Output
    ```json
    {{
      "outputs": [
        {{"source_file": "...", "output_path": "...", "columns": [...]}},
        ...
      ]
    }}
    ```
    
    Finally, end your entire response with the phrase: "Column mapping complete - mapped CSVs generated"
    """,

    tools=[generate_mapped_csvs],
    handoffs=[handoff(data_integration_agent, input_filter=integration_handoff_filter)]
)

# Agent 1: Data Preparation Agent
data_prep_agent = Agent(
    name="DataPrepAgent", 
    instructions="""
    You are a specialized data preparation and analysis agent optimized for retail demand forecasting datasets.
    
    MISSION: Systematically analyze ALL provided source datasets and return structured metadata for downstream semantic mapping.
    
    EXECUTION PROTOCOL:
    1. Parse the list of file paths from the user's prompt.
    2. For EACH dataset, call the `load_and_describe_dataset` tool to load the top 10 rows and return metadata including columns, dtypes, and a small sample.
    3. COMPILE: Aggregate all per-file metadata objects into ONE JSON array.
    4. VALIDATE: Ensure every input file has a corresponding metadata object or an error entry.
    5. OUTPUT: Return ONLY the compiled JSON metadata array (no prose). After the JSON, print a concise completion message on a new line.
    # HANDOFF: After providing the metadata, the system will hand off to Column Mapping Agent
    
    Format:
    [JSON metadata array for all datasets]
    
    QUALITY STANDARDS:
    - Process ALL files in the source list
    - Accurate data type detection
    - Clean, parseable JSON output
    - No missing or corrupted metadata
    """,
    tools=[load_and_describe_dataset],
    handoffs=[handoff(column_mapping_agent, input_filter=schema_mapping_handoff_filter)]
)
