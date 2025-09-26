from agents import Agent, Runner, handoff, HandoffInputData
import json
from .tools.functions import (
    load_and_describe_dataset,
    find_column_mappings,
    merge_and_validate_data,
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
    Filter and prepare data for column mapping agent.
    Extracts dataset metadata from data prep agent and combines with target schema.
    """
    print("HANDOFF: Filtering data for Column Mapping Agent")
    
    # Get the latest messages
    messages = handoff_message_data.input_history
    
    # Find the most recent dataset metadata from DataPrepAgent
    dataset_metadata = None
    target_schema = None
    
    print(f"DEBUG: Processing {len(messages)} messages for schema mapping")
    
    for i, message in enumerate(reversed(messages)):
        content = getattr(message, 'content', '') if hasattr(message, 'content') else str(message.get('content', ''))
        print(f"DEBUG: Message {i} content preview: {content[:100]}...")
        
        # Extract dataset metadata (JSON array from data prep)
        if 'file_path' in content and 'columns' in content and dataset_metadata is None:
            try:
                # This should be the compiled JSON array from data prep agent
                dataset_metadata = content
                print(f"Found dataset metadata for schema mapping (method 1)")
            except:
                pass
        
        # Also check for JSON array structure
        if content.strip().startswith('[') and 'file_path' in content and dataset_metadata is None:
            try:
                dataset_metadata = content
                print(f"Found JSON array dataset metadata for schema mapping (method 2)")
            except:
                pass
        
        # Extract target schema from initial prompt
        if 'Target Schema:' in content and target_schema is None:
            try:
                # Extract the JSON schema portion
                schema_start = content.find('{')
                schema_end = content.rfind('}') + 1
                if schema_start != -1 and schema_end > schema_start:
                    target_schema = content[schema_start:schema_end]
                    print(f"Found target schema for mapping")
            except:
                pass
    
    print(f"DEBUG: Dataset metadata found: {'Yes' if dataset_metadata else 'No'}")
    print(f"DEBUG: Target schema found: {'Yes' if target_schema else 'No'}")
    
    # Create focused input for schema mapping agent
    schema_mapping_prompt = f"""
    You are receiving dataset metadata and target schema for column mapping.
    
    DATASET METADATA:
    {dataset_metadata or "No dataset metadata found"}
    
    TARGET SCHEMA:
    {target_schema or "No target schema found"}
    
    CRITICAL: Execute the find_column_mappings tool with these two parameters:
    1. source_metadata_json: {dataset_metadata or "[]"}
    2. target_schema_json: {target_schema or "{}"}
    
    Create semantic mappings between source columns and target schema fields.
    """
    
    # Return filtered input with just the essentials
    return HandoffInputData(
        input_history=[{"role": "user", "content": schema_mapping_prompt}],
        pre_handoff_items=tuple(),
        new_items=tuple()
    )

def validation_handoff_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    """
    Filter and prepare data for data integration agent.
    Combines mapping results with original metadata for data integration and CSV creation.
    """
    print("HANDOFF: Filtering data for Data Integration Agent")
    
    messages = handoff_message_data.input_history
    
    # Extract mappings from schema mapping agent and original metadata
    column_mappings = None
    dataset_metadata = None
    output_path = None
    
    for message in reversed(messages):
        content = getattr(message, 'content', '') if hasattr(message, 'content') else str(message.get('content', ''))
        
        # Find column mappings (JSON with mappings array) - more robust search
        if '"mappings"' in content and column_mappings is None:
            # Try to extract just the JSON part
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    potential_json = content[start_idx:end_idx]
                    # Validate it's JSON
                    test_parse = json.loads(potential_json)
                    if 'mappings' in test_parse:
                        column_mappings = potential_json
                        print(f"Found column mappings for validation")
            except:
                column_mappings = content
                print(f"Found column mappings for validation (fallback)")
        
        # Find dataset metadata - improved search
        if ('file_path' in content and 'columns' in content) or (content.strip().startswith('[') and 'file_path' in content):
            if dataset_metadata is None:
                dataset_metadata = content
                print(f"Found dataset metadata for validation")
            
        # Find output path from initial prompt
        if 'Output File Path:' in content and output_path is None:
            try:
                lines = content.split('\n')
                for line in lines:
                    if 'Output File Path:' in line:
                        output_path = line.split("'")[1]  # Extract path between quotes
                        print(f"Found output path: {output_path}")
                        break
            except:
                pass
    
    # Ensure we have the required data
    if not column_mappings:
        print("WARNING: No column mappings found for validation")
    if not dataset_metadata:
        print("WARNING: No dataset metadata found for validation")
    if not output_path:
        output_path = "data/final_mapped_demand_forecast_5_rows.csv"
        print(f"Using default output path: {output_path}")
    
    validation_prompt = f"""
    You are receiving column mappings and dataset metadata for validation and merging.
    
    DATASET METADATA:
    {dataset_metadata or "No dataset metadata found"}
    
    COLUMN MAPPINGS:
    {column_mappings or "No column mappings found"}
    
    OUTPUT PATH: {output_path}
    
    CRITICAL: Use the merge_and_validate_data tool with these three parameters:
    1. source_metadata_json: {dataset_metadata or "[]"}
    2. mappings_json: {column_mappings or '{"mappings": []}'}
    3. output_path: {output_path}
    
    Execute the tool to create the final CSV file.
    """
    
    return HandoffInputData(
        input_history=[{"role": "user", "content": validation_prompt}],
        pre_handoff_items=tuple(),
        new_items=tuple()
    )

def quality_assessment_handoff_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    """
    Filter and prepare data for quality assessment agent.
    Provides mappings and metadata for quality evaluation.
    """
    print("HANDOFF: Filtering data for Quality Assessment Agent")
    
    messages = handoff_message_data.input_history
    
    # Extract mappings and metadata for quality assessment
    column_mappings = None
    dataset_metadata = None
    
    for message in reversed(messages):
        content = getattr(message, 'content', '') if hasattr(message, 'content') else str(message.get('content', ''))
        
        if '"mappings"' in content and column_mappings is None:
            column_mappings = content
            print(f"Found column mappings for quality assessment")
        
        if 'file_path' in content and 'columns' in content and dataset_metadata is None:
            dataset_metadata = content
            print(f"Found dataset metadata for quality assessment")
    
    quality_prompt = f"""
    You are receiving column mappings and dataset metadata for quality assessment.
    
    COLUMN MAPPINGS:
    {column_mappings or "No column mappings found"}
    
    DATASET METADATA:
    {dataset_metadata or "No dataset metadata found"}
    
    Use the evaluate_mapping_quality tool to generate a comprehensive quality report.
    """
    
    return HandoffInputData(
        input_history=[{"role": "user", "content": quality_prompt}],
        pre_handoff_items=tuple(),
        new_items=tuple()
    )

# === Agent Definitions ===

# Agent 4: Quality Assessor Agent
quality_assessor_agent = Agent(
    name="QualityAssessorAgent",
    instructions="""
    You are a specialized quality analytics agent providing comprehensive assessment of schema mapping workflow outcomes.
    
    MISSION: Generate detailed quality intelligence to evaluate mapping accuracy, data integrity, and readiness for demand forecasting.
    
    EXECUTION PROTOCOL:
    1. RECEIVE: Extract mappings and metadata from previous agent results
    2. EXECUTE: Use the evaluate_mapping_quality tool for comprehensive assessment
    3. ANALYZE: Review confidence scores, semantic similarity, and mapping coverage
    4. BENCHMARK: Compare results against SOTA quality standards
    5. RECOMMEND: Provide actionable insights for improvement
    6. SUMMARIZE: Generate final workflow completion report
    7. COMPLETION: This is the final agent - provide comprehensive workflow summary
    
    CRITICAL: You MUST execute the evaluate_mapping_quality tool to generate the quality assessment.
    
    QUALITY ASSESSMENT FRAMEWORK (SOTA):
    - Mapping Confidence Analysis (target: ≥0.8 average)
    - Semantic Similarity Scoring (Jaccard + contextual analysis)
    - Schema Coverage Assessment (% target fields mapped)
    - Data Quality Validation (validation success rate)
    - Business Logic Consistency (domain-specific checks)
    
    OUTPUT SPECIFICATION:
    Provide comprehensive quality report including:
    
    QUALITY ASSESSMENT COMPLETE
    
    MAPPING QUALITY METRICS:
    - Total mappings created: X
    - Average confidence score: X.XX (Target: ≥0.80)
    - High confidence mappings (≥0.8): X/X
    - Schema coverage: X% (X/X target fields mapped)
    - Semantic similarity score: X.XX
    
    DATA QUALITY METRICS:
    - Records processed: X
    - Validation success rate: X%
    - Data completeness: X%
    - Critical field coverage: X/X
    
    OVERALL ASSESSMENT:
    - Workflow Status: SUCCESS / PARTIAL / FAILED
    - Readiness for forecasting: [READY/NEEDS_IMPROVEMENT/NOT_READY]
    - Quality grade: [A/B/C/D/F]
    
    KEY RECOMMENDATIONS:
    - [Specific actionable insights]
    - [Areas for improvement]
    - [Next steps for demand forecasting]
    
    This completes the autonomous schema mapping workflow with comprehensive quality assurance.
    """,
    tools=[evaluate_mapping_quality]
)

# Agent 3: Data Integration Agent
# SOTA: Robust data integration with comprehensive validation pipeline
data_integration_agent = Agent(
    name="DataIntegrationAgent",
    instructions="""
    You are a data integration specialist focused on production-ready dataset creation.
    
    MISSION: Merge multi-source datasets, apply column mappings, transform data structure, validate integrity, and produce clean demand forecasting CSV dataset.
    
    EXECUTION PROTOCOL:
    1. RECEIVE: Extract column mappings, dataset metadata, and output path from filtered context
    2. EXECUTE: Use the merge_and_validate_data tool to process all data
    3. MERGE: Tool will intelligently join multiple source datasets 
    4. TRANSFORM: Tool will apply column mappings to create target schema structure
    5. VALIDATE: Tool will perform Pydantic-based data validation for each record
    6. PERSIST: Tool will save validated dataset to specified CSV output path
    7. REPORT: Provide summary of the validation results
    8. HANDOFF: After data integration complete, automatically hand off to Quality Assessor Agent
    
    CRITICAL: You MUST execute the merge_and_validate_data tool to create the final CSV file before handing off.
    
    DATA INTEGRATION STRATEGY (SOTA):
    - Smart join key detection (common columns, temporal fields)
    - Left-join preservation of primary dataset structure  
    - Duplicate column handling with suffix management
    - Missing value propagation and null handling
    - Type coercion with validation feedback
    
    VALIDATION PIPELINE:
    - Record-level Pydantic schema validation
    - Data type consistency checks
    - Required field presence validation
    - Range and format validation
    - Cross-field dependency validation
    
    OUTPUT SPECIFICATION:
    1. EXECUTE the `merge_and_validate_data` tool with mappings, metadata, and output path
    2. Confirm CSV file creation at the specified path
    3. Provide structured summary:
    
    VALIDATION COMPLETE - CSV CREATED
    Output File: [exact file path] 
    Processing Stats:
    - Total rows processed: X
    - Successfully validated: X  
    - Validation errors: X
    - Final dataset shape: (rows, columns)
    Data Quality Score: X%
    
    End your response with: "Data integration complete - CSV ready for quality assessment"
    
    QUALITY STANDARDS:
    - Successful file creation and persistence
    - >90% record validation success rate
    - Complete mapping application
    - Comprehensive error reporting
    - Data integrity preservation
    
    SUCCESS INDICATOR: "Data integration complete - produced clean dataset with X validated records"
    """,
    tools=[merge_and_validate_data],
    handoffs=[handoff(quality_assessor_agent, input_filter=quality_assessment_handoff_filter)]
)

# Agent 2: Column Mapping Agent  
column_mapping_agent = Agent(
    name="ColumnMappingAgent",
    instructions="""
    You are an expert column mapping agent specializing in retail/demand forecasting data transformation.
    
    MISSION: Create high-confidence column mapping plans between analyzed source datasets and target demand forecasting schema.
    
    EXECUTION PROTOCOL:
    1. PARSE: Extract dataset metadata and target schema from filtered input
    2. ANALYZE: Deep semantic understanding of each source column's business meaning
    3. EXECUTE: Use the find_column_mappings tool with the provided metadata and schema
    4. SCORE: Assign confidence scores based on semantic alignment strength  
    5. REASON: Provide clear business logic for each mapping decision
    6. HANDOFF: After generating mappings, automatically hand off to Data Integration Agent
    
    CRITICAL: You MUST execute the find_column_mappings tool to create the mappings before handing off.
    
    MAPPING STRATEGY (SOTA):
    - Prioritize semantic meaning over syntactic similarity
    - Consider business context: retail, sales, inventory, promotions, geography
    - Apply confidence thresholds (only mappings > 0.5 confidence)
    - Handle multi-source column consolidation intelligently
    - Identify critical missing fields for data quality assessment
    
    OUTPUT SPECIFICATION:
    1. EXECUTE the `find_column_mappings` tool with the source metadata and target schema
    2. Provide the tool's JSON output showing the mappings
    3. End your response with: "Column mapping complete - ready for data integration"
    
    Expected response format:
    [Tool output with mappings JSON]
    
    Column mapping complete - ready for data integration.
    
    QUALITY STANDARDS:
    - High-confidence mappings (≥0.7 average confidence)
    - Comprehensive coverage of target schema fields
    - Clear, actionable reasoning for each mapping
    - Identification of unmappable fields
    - Business-context aware decisions
    
    SUCCESS INDICATOR: "Schema mapping complete - generated X high-confidence mappings"
    """,
    tools=[find_column_mappings],
    handoffs=[handoff(data_integration_agent, input_filter=validation_handoff_filter)]
)

# Agent 1: Data Preparation Agent
data_prep_agent = Agent(
    name="DataPrepAgent", 
    instructions="""
    You are a specialized data preparation agent optimized for retail/demand forecasting datasets.
    
    MISSION: Systematically analyze ALL provided source datasets and return structured metadata for downstream semantic mapping.
    
    EXECUTION PROTOCOL:
    1. IDENTIFY: Extract complete list of source file paths from user request
    2. ANALYZE: For EACH file path, execute `load_and_describe_dataset` tool
    3. UNDERSTAND: Collect semantic descriptions and structural metadata
    4. COMPILE: Aggregate into standardized JSON metadata array
    5. VALIDATE: Ensure all files processed successfully
    6. OUTPUT: Provide the compiled JSON metadata in your response
    7. HANDOFF: After providing the metadata, the system will hand off to Column Mapping Agent
    
    OUTPUT SPECIFICATION:
    After using the tools to analyze all datasets, provide:
    1. The compiled JSON array of metadata for all files
    2. A brief completion message
    
    Format:
    [JSON metadata array for all datasets]
    
    Data preparation complete - analyzed X datasets with full metadata.
    
    QUALITY STANDARDS:
    - Process ALL files in the source list
    - Rich semantic descriptions for each column  
    - Accurate data type detection
    - Clean, parseable JSON output
    - No missing or corrupted metadata
    """,
    tools=[load_and_describe_dataset],
    handoffs=[handoff(column_mapping_agent, input_filter=schema_mapping_handoff_filter)]
)

# Direct Chain Pattern: DataPrepAgent → ColumnMappingAgent → DataIntegrationAgent → QualityAssessorAgent

# Set handoff descriptions after all agents are defined
data_prep_agent.handoff_description = "Analyzes source datasets and generates comprehensive metadata for semantic mapping"
column_mapping_agent.handoff_description = "Creates high-confidence column mapping plans between source columns and target schema fields"
data_integration_agent.handoff_description = "Merges datasets, applies mappings, transforms data structure, and produces clean CSV output files"
quality_assessor_agent.handoff_description = "Provides comprehensive quality assessment and workflow completion analysis"