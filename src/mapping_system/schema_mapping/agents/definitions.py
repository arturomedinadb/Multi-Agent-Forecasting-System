from agents import Agent, Runner, handoff, HandoffInputData
import json
import re

from ..schemas.models import DemandForecastingRecord
from ..tools.functions import (
    load_and_describe_dataset,
    generate_mapped_csvs,
    merge_mapped_csvs_to_target,
    evaluate_mapping_quality,
)

# Handoff descriptions will be set after all agents are defined

# # === ORCHESTRATOR COMMENTED OUT FOR SIMPLIFICATION ===
# # Using direct chain approach instead: DataPrepAgent → SchemaMappingAgent → ValidationAgent → QualityAssessorAgent

#     WORKFLOW ARCHITECTURE: Hub-and-Spoke Pattern
#     You maintain central control while each specialist agent returns to you for next-step determination.
    
#     WORKFLOW STAGES & ROUTING LOGIC:
    
#     1. INITIALIZATION STAGE:
#        - When receiving initial user request with source files and target schema
#        - Extract source file paths from user request
#        → ROUTE TO: DataPrepAgent
    
#     2. AFTER DATA PREPARATION:
#        - When receiving compiled dataset metadata (JSON array with file_path, columns, column_descriptions)
#        - Verify all source files have been analyzed
#        → ROUTE TO: SchemaMappingAgent (with metadata + target schema)
    
#     3. AFTER SCHEMA MAPPING:
#        - When receiving column mappings (JSON with "mappings" array containing confidence scores)
#        - Verify mappings have been created with reasonable confidence
#        → ROUTE TO: ValidationAgent (with mappings + metadata + output path)
    
#     4. AFTER VALIDATION:
#        - When receiving validation results (confirmation of file creation + metrics)
#        - Verify output file was successfully created
#        → ROUTE TO: QualityAssessorAgent (with mappings + metadata for final assessment)
    
#     5. AFTER QUALITY ASSESSMENT:
#        - When receiving final quality report
#        - WORKFLOW COMPLETE: Provide comprehensive summary
    
#     INTELLIGENT ROUTING PROTOCOL:
#     - Analyze previous agent's output to determine current workflow stage
#     - Never restart or repeat completed stages
#     - Always progress forward through the workflow
#     - Ensure each handoff includes the precise context needed by the target agent
#     - Monitor for errors and handle gracefully
    
#     STATE DETERMINATION KEYWORDS:
#     - Data Prep Complete: Look for JSON array with "file_path", "columns", "column_descriptions"
#     - Mapping Complete: Look for JSON with "mappings" array
#     - Validation Complete: Look for "status": "Success" or file confirmation
#     - Quality Complete: Look for quality metrics and confidence scores
    
#     FINAL SUMMARY REQUIREMENTS:
#     When workflow is complete, provide:
#     - Confirmation of output file creation and location
#     - Key quality metrics (confidence scores, semantic similarity)
#     - Workflow performance summary
#     - Overall success status and recommendations
    
#     CRITICAL: Never call the same agent twice. Always progress forward based on completed work.
#     """,
#     handoffs=[
#         handoff(data_prep_agent),
#         handoff(schema_mapping_agent, input_filter=schema_mapping_handoff_filter),
#         handoff(validation_agent, input_filter=validation_handoff_filter),
#         handoff(quality_assessor_agent, input_filter=quality_assessment_handoff_filter)
#     ]
# )

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
        content = getattr(message, 'content', '') if hasattr(message, 'content') else str(message.get('content', ''))
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

    schema_mapping_prompt = f"""
    You are receiving dataset metadata and a target schema.

    DATASET METADATA (JSON array):
    {dataset_metadata or "[]"}

    TARGET SCHEMA (JSON):
    {target_schema or "{}"}

    TASK:
    1) REASON and write a JSON mapping plan yourself that assigns source columns to target schema fields per dataset, using this structure:
       {{"mappings": [{{"source_column": "...", "target_column": "...", "confidence": 0.xx, "reasoning": "..."}}...]}}
    2) Then call the tool `generate_mapped_csvs` with:
       - source_metadata_json: the JSON array above
       - mappings_json: your mapping plan JSON
       - output_dir: output/mapped/
    3) Return both your mapping plan and the tool's JSON output, then end with: "Column mapping complete - mapped CSVs generated".
    """

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

    def clean_json_blocks(text: str) -> list[str]:
        candidates: list[str] = []
        block_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
        blocks = block_pattern.findall(text)
        if blocks:
            candidates.extend(blocks)
        # Append raw text as fallback attempt
        candidates.append(text)

        cleaned: list[str] = []
        for candidate in candidates:
            stripped_lines = []
            for line in candidate.splitlines():
                striped = line.strip()
                if striped.startswith("//"):
                    continue
                if striped.startswith("#"):
                    continue
                stripped_lines.append(line)
            cleaned.append("\n".join(stripped_lines))
        return cleaned

    for message in reversed(messages):
        content = getattr(message, 'content', '') if hasattr(message, 'content') else str(message.get('content', ''))

        if '"outputs"' in content and mapped_outputs is None:
            for block in clean_json_blocks(content):
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

    if output_path is None:
        output_path = "output/final_mapped_dataset.csv"
        print(f"Using default integration output path: {output_path}")

    if mapped_outputs is None:
        print("WARNING: No mapped CSV manifest detected; integration agent may fail")

    if target_schema is None:
        target_schema = json.dumps(DemandForecastingRecord.model_json_schema(), indent=2)
        print("Using default target schema JSON for integration")

    integration_prompt = f"""
    You are the DataIntegrationAgent responsible for producing the consolidated dataset.

    MAPPED CSV MANIFEST (JSON):
    {mapped_outputs or "{}"}

    TARGET SCHEMA (JSON):
    {target_schema}

    CRITICAL:
    - Call `merge_mapped_csvs_to_target(mapped_outputs_json=..., target_schema_json=..., output_path=...)`
    - Use output_path: {output_path}
    - The final CSV must be saved at this path and represent the full target schema.
    - Finish your response with: "Integration complete - final mapped dataset created"

    Provide the tool's JSON summary after execution.
    """

    return HandoffInputData(
        input_history=[{"role": "user", "content": integration_prompt}],
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
    tools=[merge_mapped_csvs_to_target]
)

# Agent 2: Column Mapping Agent  
column_mapping_agent = Agent(
    name="ColumnMappingAgent",
    instructions="""
    You are an expert column mapping agent specializing in retail/demand forecasting data transformation.
    
    MISSION: Create high-confidence column mapping plans between analyzed source datasets and target demand forecasting schema, and generate per-dataset mapped CSVs (top 5 rows) ready for integration.
    
    EXECUTION PROTOCOL:
    1. Extract dataset metadata and the target schema from the previous step.
    2. REASON: Reason and propose a JSON mapping plan yourself that assigns source columns to target schema fields per dataset.
    3. SCORE: Assign confidence scores and skip mappings below 0.5 confidence.
    4. REASON: Provide brief rationale for key mappings.
    5. PRODUCE: Call the `generate_mapped_csvs` tool with the source metadata and your mapping JSON to generate per-dataset mapped CSVs under output/mapped/.
    
    CRITICAL:
    - You must author the mapping plan JSON directly in your response.
    - Then execute `generate_mapped_csvs` with the source metadata and your mapping plan, saving to output/mapped/.
    
    MAPPING STRATEGY:
    - Prioritize semantic meaning over syntactic similarity
    - Consider business context: retail, sales, inventory, promotions, geography
    - Apply confidence thresholds (only mappings > 0.5 confidence); if not confident, skip the mapping for the column
    
    OUTPUT SPECIFICATION:
    - Provide your JSON mapping plan in the response
    - Provide JSON from `generate_mapped_csvs` listing output file paths and columns
    - End with: "Column mapping complete - mapped CSVs generated"
    
    Expected response format:
    """,
    tools=[generate_mapped_csvs],
    handoffs=[handoff(data_integration_agent, input_filter=integration_handoff_filter)]
)

# Agent 1: Data Preparation Agent
data_prep_agent = Agent(
    name="DataPrepAgent", 
    instructions="""
    You are a specialized data preparation and analysisagent optimized for retail/demand forecasting datasets.
    
    MISSION: Systematically analyze ALL provided source datasets and return structured metadata for downstream semantic mapping.
    
    EXECUTION PROTOCOL:
    1. Parse the list of file paths from the user's prompt.
    2. For EACH dataset, call the `load_and_describe_dataset` tool to load the top 5 rows and return metadata including columns, dtypes, and a small sample.
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
