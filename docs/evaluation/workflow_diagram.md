# Schema Mapping Workflow with Integrated Evaluation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      🎯 SCHEMA MAPPING ORCHESTRATOR                              │
│                                                                                  │
│  ROLE: Coordinates end-to-end schema mapping workflow with explicit stages      │
│  RESPONSIBILITIES:                                                               │
│    • Manages sequential execution of core agents                                │
│    • Triggers evaluation agents after each stage                                │
│    • Handles file I/O and artifact storage                                      │
│    • Renders prompts from Jinja templates                                       │
│    • Maintains run context and directory structure                              │
│    • Error handling and graceful degradation                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WORKFLOW START                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   📥 USER INPUTS             │
                        │   • Source CSV files (N)     │
                        │   • Target schema definition │
                        │   • Row limit (optional)     │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │ 🔧 Orchestrator Setup        │
                        │ • Create output directories  │
                        │ • Render prompts (Jinja2)    │
                        │ • Initialize agent executor  │
                        └──────────────────────────────┘
                                      │
╔═════════════════════════════════════════════════════════════════════════════════╗
║                          AGENT 1: DATA PREPARATION                               ║
║                                                                                  ║
║  🔨 TOOLS:                                                                       ║
║    • load_and_describe_dataset                                                  ║
║                                                                                  ║
║  📥 INPUTS:                                                                      ║
║    • Source CSV files (from data/ directory)                                    ║
║    • Row limit configuration                                                    ║
║                                                                                  ║
║  🎯 ACTIONS:                                                                     ║
║    • Analyzes file structure                                                    ║
║    • Extracts metadata (columns, types, shape)                                  ║
║    • Samples top N rows                                                         ║
║    • Profiles data types                                                        ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │ 📤 OUTPUT FILES:             │
                        │ • source_metadata.json       │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                   📊 EVALUATION AGENT 1: DATA PREP EVALUATOR                     ║
║                                                                                  ║
║  🔨 TOOLS:                                                                       ║
║    • evaluate_data_prep_agent                                                   ║
║    • generate_summary_report ──────────────────────────────────────┐            ║
║                                                                     │            ║
║  📥 INPUTS:                                                         │            ║
║    • Agent input prompt (truncated)                                │            ║
║    • Agent output (source_metadata.json content, truncated)        │            ║
║    • Expected files list                                           │            ║
║                                                                     │            ║
║  📊 METRICS:                                                         │            ║
║    • Task Completion (DeepEval, threshold: 0.6)                    │            ║
║    • Tool Correctness (Deterministic)                              │            ║
║    • Answer Relevancy (DeepEval, threshold: 0.6)                   │            ║
╚═════════════════════════════════════════════════════════════════════│════════════╝
                                      │                               │
                                      ▼                               │
                                                                      │
                                                           ┌──────────▼──────────┐
                                                           │  🤖 SUMMARY REPORT  │
                                                           │       AGENT         │
                                                           │   (OpenAI GPT-4)    │
                                                           │                     │
                                                           │  Analyzes metrics   │
                                                           │  Diagnoses issues   │
                                                           │  Recommends fixes   │
                                                           └─────────────────────┘
                        ┌──────────────────────────────┐
                        │ 📤 OUTPUT FILES:             │
                        │ • data_prep_evaluation.txt   │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                          AGENT 2: COLUMN MAPPING                                 ║
║                                                                                  ║
║  🔨 TOOLS:                                                                       ║
║    • generate_mapped_csvs                                                       ║
║                                                                                  ║
║  📥 INPUTS:                                                                      ║
║    • source_metadata.json                                                       ║
║    • target_schema.json (DemandForecastingRecord)                               ║
║                                                                                  ║
║  🎯 ACTIONS:                                                                     ║
║    • Maps source → target columns                                               ║
║    • Semantic matching with confidence scoring                                  ║
║    • Generates per-dataset mapped CSVs                                          ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │ 📤 OUTPUT FILES:             │
                        │ • mapping_plan.json          │
                        │ • mapping_manifest.json      │
                        │ • mapped/*.csv (N files)     │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                📊 EVALUATION AGENT 2: COLUMN MAPPING EVALUATOR                   ║
║                                                                                  ║
║  🔨 TOOLS:                                                                       ║
║    • evaluate_column_mapping_agent                                              ║
║    • generate_summary_report ──────────────────────────────────────┐            ║
║                                                                     │            ║
║  📥 INPUTS:                                                         │            ║
║    • Agent input prompt (truncated)                                │            ║
║    • Agent output (truncated)                                      │            ║
║    • mapping_plan.json (sanitized)                                 │            ║
║    • target_schema.json (sanitized)                                │            ║
║                                                                     │            ║
║  📊 METRICS:                                                         │            ║
║    • Task Completion (DeepEval, threshold: 0.6)                    │            ║
║    • Tool Correctness (Deterministic)                              │            ║
║    • Field Coverage (Deterministic, threshold: 0.9)                │            ║
║    • Type Compatibility (Deterministic, threshold: 1.0)            │            ║
║    • Semantic Similarity (Token-based, threshold: 0.5)             │            ║
╚═════════════════════════════════════════════════════════════════════│════════════╝
                                      │                               │
                                      ▼                               └──────────┐
                                                                                 │
                                                           ┌─────────────────────▼┐
                                                           │  🤖 SUMMARY REPORT   │
                                                           │       AGENT          │
                                                           │   (Shared, GPT-4)    │
                                                           └──────────────────────┘
                        ┌──────────────────────────────┐
                        │ 📤 OUTPUT FILES:             │
                        │ • column_mapping_eval.txt    │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                       AGENT 3: DATA INTEGRATION                                  ║
║                                                                                  ║
║  🔨 TOOLS:                                                                       ║
║    • merge_mapped_csvs_to_target                                                ║
║                                                                                  ║
║  📥 INPUTS:                                                                      ║
║    • mapping_manifest.json                                                      ║
║    • mapped/*.csv (all mapped CSVs)                                             ║
║    • target_schema.json                                                         ║
║                                                                                  ║
║  🎯 ACTIONS:                                                                     ║
║    • Merges mapped CSVs                                                         ║
║    • Consolidates on key columns                                                ║
║    • Ensures schema compliance                                                  ║
║    • Creates final unified dataset                                              ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │ 📤 OUTPUT FILES:             │
                        │ • final_mapped_dataset.csv   │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║               📊 EVALUATION AGENT 3: DATA INTEGRATION EVALUATOR                  ║
║                                                                                  ║
║  🔨 TOOLS:                                                                       ║
║    • evaluate_data_integration_agent                                            ║
║    • validate_final_dataset                                                     ║
║    • generate_summary_report ──────────────────────────────────────┐            ║
║                                                                     │            ║
║  📥 INPUTS:                                                         │            ║
║    • Agent input prompt (truncated)                                │            ║
║    • Agent output (truncated)                                      │            ║
║    • Source row count (from data prep stage)                      │            ║
║    • Final row count (from merge tool output)                     │            ║
║    • final_mapped_dataset.csv (for validation)                     │            ║
║    • target_schema.json (sanitized)                                │            ║
║    • mapping_plan.json (sanitized)                                 │            ║
║                                                                     │            ║
║  📊 AGENT PERFORMANCE METRICS:                                      │            ║
║    • Task Completion (DeepEval, threshold: 0.6)                    │            ║
║    • Tool Correctness (Deterministic)                              │            ║
║    • Data Quality - Row Preservation (threshold: 0.95)             │            ║
║                                                                     │            ║
║  📊 FINAL DATASET VALIDATION (Deterministic):                       │            ║
║    • Field Coverage (threshold: 0.9)                               │            ║
║    • Type Compatibility (threshold: 0.95)                          │            ║
║    • Null Analysis (flags >50% nulls)                              │            ║
║    • Duplicate Detection (on key columns)                          │            ║
╚═════════════════════════════════════════════════════════════════════│════════════╝
                                      │                               │
                                      ▼                               └──────────┐
                                                                                 │
                                                           ┌─────────────────────▼┐
                                                           │  🤖 SUMMARY REPORT   │
                                                           │       AGENT          │
                                                           │   (Shared, GPT-4)    │
                                                           └──────────────────────┘
                        ┌──────────────────────────────┐
                        │ 📤 OUTPUT FILES:             │
                        │ • data_integration_eval.txt  │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │     ✅ WORKFLOW COMPLETE     │
                        │                              │
                        │  📁 All Outputs:             │
                        │  • final_mapped_dataset.csv  │
                        │  • source_metadata.json      │
                        │  • mapping_plan.json         │
                        │  • mapping_manifest.json     │
                        │  • mapped/*.csv (N files)    │
                        │  • data_prep_evaluation.txt  │
                        │  • column_mapping_eval.txt   │
                        │  • data_integration_eval.txt │
                        └──────────────────────────────┘
```

## Key Components

### Orchestrator
**SchemaMappingOrchestrator** (`orchestrator.py`)
- **Role**: Central coordinator for the entire workflow
- **Responsibilities**:
  - Sequential execution of all agents (core + evaluation)
  - File I/O management and artifact storage
  - Prompt rendering from Jinja2 templates
  - Run context management (timestamps, directories)
  - Error handling and graceful degradation
- **Configuration**: 
  - Source files, target schema, row limits
  - Output directory structure
  - API keys (OPENAI_API_KEY, DEEPEVAL_API_KEY)

### Core Agents
Each core agent has specific tools and produces distinct outputs:

1. **Data Preparation Agent** (`data_prep_agent`)
   - **Tools**: `load_and_describe_dataset`
   - **Inputs**: Source CSV files (N files from `data/`)
   - **Outputs**: `source_metadata.json`
   - **Purpose**: Profiles source data and extracts metadata

2. **Column Mapping Agent** (`column_mapping_agent`)
   - **Tools**: `generate_mapped_csvs`
   - **Inputs**: `source_metadata.json`, `target_schema.json`
   - **Outputs**: `mapping_plan.json`, `mapping_manifest.json`, `mapped/*.csv` (N files)
   - **Purpose**: Creates semantic mappings and generates transformed CSVs

3. **Data Integration Agent** (`data_integration_agent`)
   - **Tools**: `merge_mapped_csvs_to_target`
   - **Inputs**: `mapping_manifest.json`, `mapped/*.csv`, `target_schema.json`
   - **Outputs**: `final_mapped_dataset.csv`
   - **Purpose**: Merges all mapped data into final unified dataset

### Evaluation Agents
Each evaluation agent runs **immediately after** its corresponding core agent:

1. **Data Prep Evaluation Agent** (`data_prep_evaluation_agent`)
   - **Tools**: `evaluate_data_prep_agent`, `generate_summary_report`
   - **Inputs**: Agent I/O (truncated), expected files list
   - **Outputs**: `data_prep_evaluation.txt`
   - **Purpose**: 
     - Validates metadata extraction
     - Checks file coverage
     - Assesses output relevancy

2. **Column Mapping Evaluation Agent** (`column_mapping_evaluation_agent`)
   - **Tools**: `evaluate_column_mapping_agent`, `generate_summary_report`
   - **Inputs**: Agent I/O (truncated), `mapping_plan.json`, `target_schema.json`
   - **Outputs**: `column_mapping_evaluation.txt`
   - **Purpose**:
     - Measures field coverage
     - Validates type compatibility
     - Analyzes semantic similarity
     - Identifies missing/weak mappings

3. **Data Integration Evaluation Agent** (`data_integration_evaluation_agent`)
   - **Tools**: `evaluate_data_integration_agent`, `validate_final_dataset`, `generate_summary_report`
   - **Inputs**: Agent I/O (truncated), source row count (from data prep), final row count (from merge output), `final_mapped_dataset.csv`, `target_schema.json`, `mapping_plan.json`
   - **Outputs**: `data_integration_evaluation.txt`
   - **Purpose**:
     - Verifies merge success
     - Validates row preservation (compares source vs final row counts)
     - Checks final dataset quality
     - Performs comprehensive schema validation
   - **Note**: Tracks total rows from all source files (data prep stage) and compares against final merged dataset to detect data loss during integration

### Summary Report Agent (Shared)
**Summary Report Agent** (`summary_report_agent`)
- **Type**: AI-powered analysis agent (OpenAI GPT-4)
- **Invoked By**: All three evaluation agents via `generate_summary_report` tool
- **Tools**: None (this agent provides analysis, doesn't call tools)
- **Inputs**: 
  - Evaluation results JSON (metrics, scores, pass/fail status)
  - Agent name being evaluated
- **Outputs**: Markdown-formatted analysis report embedded in evaluation .txt files
- **Purpose**:
  - Interprets quantitative metric results
  - Diagnoses root causes of failures
  - Identifies performance patterns
  - Generates specific, actionable recommendations
  - Provides balanced assessment of successes and failures
- **Key Capabilities**:
  - Multi-agent system performance assessment
  - Root cause analysis
  - Prioritized recommendations by impact and feasibility
  - Professional, data-driven communication

### Data Flow Summary

```
┌─────────────────────────┐
│   Source CSV Files      │
│   (data/*.csv)          │
└─────────────────────────┘
           │
           ▼
    [Data Prep Agent]
           │
           ▼
┌─────────────────────────┐
│ source_metadata.json    │
└─────────────────────────┘
           │
           ▼
   [Column Mapping Agent]
           │
           ▼
┌─────────────────────────┐
│ mapping_plan.json       │
│ mapping_manifest.json   │
│ mapped/*.csv (N files)  │
└─────────────────────────┘
           │
           ▼
  [Data Integration Agent]
           │
           ▼
┌─────────────────────────┐
│ final_mapped_dataset.csv│ ← FINAL OUTPUT
└─────────────────────────┘

Parallel Evaluation Outputs:
• data_prep_evaluation.txt
• column_mapping_evaluation.txt
• data_integration_evaluation.txt
```

### Evaluation Features

#### Shared Across All Evaluators:
- **Task Completion Metric**: Did the agent complete its task?
- **Tool Correctness Metric**: Were tools used properly?
- **AI-Powered Summary Report**: OpenAI GPT-4 analyzes results and provides recommendations

#### Evaluation Flow:
```
Core Agent Execution
      ↓
Agent Output Captured (with truncation for token limits)
      ↓
Evaluation Agent Triggered
      ↓
DeepEval Metrics Calculated
      ↓
Summary Report Generated (OpenAI GPT-4)
      ↓
Evaluation Saved (UTF-8 encoded .txt file)
      ↓
Continue to Next Agent
```

#### Tools Used by Agents

**Core Agents:**
1. `load_and_describe_dataset` - Loads CSVs and extracts metadata
2. `generate_mapped_csvs` - Creates transformed CSVs based on mapping plan
3. `merge_mapped_csvs_to_target` - Merges all mapped CSVs into final dataset

**Evaluation Agents:**
1. `evaluate_data_prep_agent` - Calculates metrics for Data Prep performance
2. `evaluate_column_mapping_agent` - Calculates metrics for Column Mapping quality
3. `evaluate_data_integration_agent` - Calculates metrics for Integration success
4. `validate_final_dataset` - Performs deterministic validation on final CSV
5. `generate_summary_report` - AI-powered analysis and recommendations (shared)

### Environment Variables

Required for full evaluation functionality:
```bash
DEEPEVAL_API_KEY=sk-...        # For DeepEval metrics
OPENAI_API_KEY=sk-...          # For AI summary reports
AGENT_ROW_LIMIT=N              # Limit rows for testing
```

**Note**: If API keys are not provided:
- DeepEval metrics return "No Deepeval API provided"
- Deterministic metrics still run
- Workflow continues without interruption

### Output Structure

```
output/
├── final_mapped_dataset.csv
├── runs/
│   └── <timestamp>/
│       ├── source_metadata.json
│       ├── mapping_plan.json
│       ├── mapping_manifest.json
│       ├── data_prep_evaluation.txt           ← NEW
│       ├── column_mapping_evaluation.txt      ← NEW
│       ├── data_integration_evaluation.txt    ← NEW
│       └── mapped/
│           └── [mapped CSV files]
```

### Error Handling

- **Evaluation Failures**: Non-blocking - workflow continues even if evaluation fails
- **Missing API Keys**: Graceful fallback to deterministic metrics only
- **Encoding Issues**: Fixed with UTF-8 encoding for special characters
- **Token Limits**: Input/output text truncated to avoid API limits (JSON preserved)

## Migration Notes

**Previous Approach** (Legacy):
- Single `schema_mapping_evaluation_agent` ran after entire workflow
- Monolithic evaluation at the end
- Less granular feedback

**Current Approach** (New):
- Three specialized evaluation agents
- Evaluation after each core agent
- Immediate feedback per stage
- More detailed, actionable insights
- Parallel evaluation and workflow execution

All legacy evaluation code and dependencies have been removed to eliminate redundancy.

