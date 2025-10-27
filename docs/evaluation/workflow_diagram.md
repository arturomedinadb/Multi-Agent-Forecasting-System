# Schema Mapping Workflow with Integrated Evaluation

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WORKFLOW START                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   Define Source CSV Files    │
                        │   + Target Schema            │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │      Create Prompt           │
                        └──────────────────────────────┘
                                      │
╔═════════════════════════════════════════════════════════════════════════════════╗
║                          AGENT 1: DATA PREPARATION                               ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                        ┌──────────────────────────────┐
                        │  • Analyzes file structure   │
                        │  • Extracts metadata         │
                        │  • Samples top N rows        │
                        │  • Profiles data types       │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   Saves metadata JSON        │
                        │   (source_metadata.json)     │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                   📊 EVALUATION AGENT 1: DATA PREP EVALUATOR                     ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║  Metrics:                                                                        ║
║  • Task Completion (DeepEval, threshold: 0.6)                                   ║
║  • Tool Correctness (Deterministic)                                             ║
║  • Answer Relevancy (DeepEval, threshold: 0.6)                                  ║
║                                                                                  ║
║  Output:                                                                         ║
║  • Evaluation scores and analysis                                               ║
║  • AI-generated Summary Report (OpenAI GPT-4)                                   ║
║  • Recommendations for improvement                                              ║
║  • Saved as: data_prep_evaluation.txt                                           ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                          AGENT 2: COLUMN MAPPING                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                        ┌──────────────────────────────┐
                        │  • Maps source → target      │
                        │  • Semantic matching         │
                        │  • Confidence scoring        │
                        │  • Generates mapped CSVs     │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   Saves:                     │
                        │   • mapping_plan.json        │
                        │   • mapping_manifest.json    │
                        │   • Mapped CSV files (N)     │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                📊 EVALUATION AGENT 2: COLUMN MAPPING EVALUATOR                   ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║  Metrics:                                                                        ║
║  • Task Completion (DeepEval, threshold: 0.6)                                   ║
║  • Tool Correctness (Deterministic)                                             ║
║  • Field Coverage (Deterministic, threshold: 0.9)                               ║
║  • Type Compatibility (Deterministic, threshold: 1.0)                           ║
║  • Semantic Similarity (Token-based, threshold: 0.5)                            ║
║                                                                                  ║
║  Output:                                                                         ║
║  • Mapping quality scores                                                       ║
║  • Missing field analysis                                                       ║
║  • Weak semantic mappings identified                                            ║
║  • AI-generated Summary Report (OpenAI GPT-4)                                   ║
║  • Saved as: column_mapping_evaluation.txt                                      ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                       AGENT 3: DATA INTEGRATION                                  ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                        ┌──────────────────────────────┐
                        │  • Merges mapped CSVs        │
                        │  • Consolidates on keys      │
                        │  • Ensures schema compliance │
                        │  • Creates final dataset     │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   Saves:                     │
                        │   final_mapped_dataset.csv   │
                        └──────────────────────────────┘
                                      │
                                      ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║               📊 EVALUATION AGENT 3: DATA INTEGRATION EVALUATOR                  ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║  Agent Performance Metrics:                                                      ║
║  • Task Completion (DeepEval, threshold: 0.6)                                   ║
║  • Tool Correctness (Deterministic)                                             ║
║  • Data Quality - Row Preservation (threshold: 0.95)                            ║
║                                                                                  ║
║  Final Dataset Validation (Deterministic):                                       ║
║  • Field Coverage (threshold: 0.9)                                              ║
║  • Type Compatibility (threshold: 0.95)                                         ║
║  • Null Analysis (flags >50% nulls)                                             ║
║  • Duplicate Detection (on key columns)                                         ║
║                                                                                  ║
║  Output:                                                                         ║
║  • Agent performance scores                                                     ║
║  • Comprehensive data quality report                                            ║
║  • Schema compliance validation                                                 ║
║  • AI-generated Summary Report (OpenAI GPT-4)                                   ║
║  • Saved as: data_integration_evaluation.txt                                    ║
╚═════════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │     WORKFLOW COMPLETE        │
                        │                              │
                        │  Outputs:                    │
                        │  ✓ Final dataset             │
                        │  ✓ Mapping artifacts         │
                        │  ✓ 3 Evaluation reports      │
                        └──────────────────────────────┘
```

## Key Components

### Core Agents (Yellow boxes in original diagram)
1. **Data Preparation Agent**: Profiles source data and extracts metadata
2. **Column Mapping Agent**: Creates semantic mappings between source and target
3. **Data Integration Agent**: Merges all mapped data into final unified dataset

### Evaluation Agents (Pink boxes - NEW!)
Each evaluation agent runs **immediately after** its corresponding core agent:

1. **Data Prep Evaluation Agent**: 
   - Validates metadata extraction
   - Checks file coverage
   - Assesses output relevancy

2. **Column Mapping Evaluation Agent**:
   - Measures field coverage
   - Validates type compatibility
   - Analyzes semantic similarity
   - Identifies missing/weak mappings

3. **Data Integration Evaluation Agent**:
   - Verifies merge success
   - Validates row preservation
   - Checks final dataset quality
   - Performs comprehensive schema validation

### Evaluation Features

#### Shared Across All Evaluators:
- **Task Completion Metric**: Did the agent complete its task?
- **Tool Correctness Metric**: Were tools used properly?
- **AI-Powered Summary Report**: OpenAI GPT-4 analyzes results and provides recommendations

#### Evaluation Flow:
```
Core Agent Execution
      ↓
Agent Output Captured
      ↓
Evaluation Agent Triggered
      ↓
DeepEval Metrics Calculated
      ↓
Summary Report Generated (OpenAI)
      ↓
Evaluation Saved (UTF-8 encoded)
      ↓
Continue to Next Agent
```

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

