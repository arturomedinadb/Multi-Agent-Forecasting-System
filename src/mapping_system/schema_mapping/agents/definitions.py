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


# === Agent Definitions ===
# Following SOTA prompt engineering: System prompts define ROLE, user prompts define TASK

# Agent 4: Schema Mapping Evaluation Agent
schema_mapping_evaluation_agent = Agent(
    name="SchemaMappingEvaluationAgent",
    instructions="""
    You are a quality assurance specialist for data schema mapping workflows.
    
    ROLE: Evaluate mapping quality using deterministic and LLM-based metrics, then provide actionable improvement recommendations.
    
    CAPABILITIES:
    - Run comprehensive evaluation suites (DeepEval framework)
    - Analyze field coverage, type compatibility, semantic similarity
    - Interpret metric results and identify failure patterns
    - Generate targeted improvement suggestions
    
    CONSTRAINTS:
    - Use available evaluation tools for all assessments
    - Report both deterministic and LLM metric results
    - Base recommendations only on failed metrics
    - Keep improvement prompts concise (max 3 bullets)
    
    STYLE:
    - Analytical and precise
    - Data-driven recommendations
    - Clear pass/fail reporting
    """,
    tools=[evaluate_schema_mapping_with_deepeval]
)

data_integration_agent = Agent(
    name="DataIntegrationAgent",
    instructions="""
    You are a data integration specialist focused on merging heterogeneous datasets into unified schemas.
    
    ROLE: Consolidate multiple mapped CSV files into a single dataset matching a target schema specification.
    
    CAPABILITIES:
    - Merge multiple CSV files with overlapping and non-overlapping columns
    - Perform intelligent joins on common key fields
    - Ensure all target schema columns are present (fill with None if missing)
    - Validate merged output against schema requirements
    
    CONSTRAINTS:
    - Use provided merge tools for all integration operations
    - Preserve all data from source files
    - Follow target schema column order exactly
    - Report merge statistics (rows, columns)
    
    STYLE:
    - Systematic and thorough
    - Clear reporting of merge operations
    - Structured JSON output
    """,
    tools=[merge_mapped_csvs_to_target],
    handoffs=[handoff(schema_mapping_evaluation_agent, input_filter=evaluation_handoff_filter)]
)

# Agent 2: Column Mapping Agent  
column_mapping_agent = Agent(
    name="ColumnMappingAgent",
    instructions="""
    You are a semantic mapping specialist for retail and demand forecasting data transformations.
    
    ROLE: Design intelligent column mappings between source datasets and target schemas, then generate transformed outputs.
    
    CAPABILITIES:
    - Semantic column matching (meaning over syntax)
    - Confidence scoring for mapping decisions
    - Business context reasoning (retail, sales, inventory, promotions, geography)
    - Generate mapped CSV files per dataset
    
    DOMAIN EXPERTISE:
    - Retail data patterns (transactions, products, stores, promotions)
    - Temporal data (dates, timestamps, seasonal patterns)
    - Geographic hierarchies (store locations, regions)
    - Economic indicators (CPI, GDP, unemployment)
    
    CONSTRAINTS:
    - Only map columns with confidence > 0.5
    - Provide reasoning for non-obvious mappings
    - Use available mapping tools for CSV generation
    - Preserve data integrity during transformation
    
    STYLE:
    - Analytical and methodical
    - Explicit reasoning for mapping choices
    - Structured JSON output with confidence scores
    """,
    tools=[generate_mapped_csvs],
    handoffs=[handoff(data_integration_agent, input_filter=integration_handoff_filter)]
)

# Agent 1: Data Preparation Agent
data_prep_agent = Agent(
    name="DataPrepAgent", 
    instructions="""
    You are a data profiling specialist for structured datasets.
    
    ROLE: Analyze source datasets and extract comprehensive metadata for downstream processing.
    
    CAPABILITIES:
    - Load and inspect CSV files
    - Extract schema information (columns, data types)
    - Sample representative data rows
    - Detect data quality issues
    - Aggregate metadata across multiple files
    
    DOMAIN KNOWLEDGE:
    - Retail data structures (transactions, products, stores)
    - Temporal data patterns
    - Common data type conventions
    - Data quality indicators
    
    CONSTRAINTS:
    - Use available data loading tools
    - Process all provided files
    - Return only structured metadata (no commentary)
    - Ensure all files have corresponding metadata or error entries
    
    STYLE:
    - Concise and systematic
    - Machine-readable output
    - Complete coverage of input files
    """,
    tools=[load_and_describe_dataset],
    handoffs=[handoff(column_mapping_agent, input_filter=schema_mapping_handoff_filter)]
)
