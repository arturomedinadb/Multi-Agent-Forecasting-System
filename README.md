# End-to-End AI Forecasting Agent System

A multi-agent AI system for automated data schema mapping and demand forecasting. This project transforms heterogeneous retail datasets into unified forecasting models through intelligent agent orchestration and specialized AI workflows.

## Project Overview

**Course**: Independent Study: AI Forecasting Agent  
**Instructor**: Fatih Nayebi  
**Team**: Ayda Elzohbi, Arturo Medina, Arial Huang, Katherine Gong

### Current System Status (v0.2.0)

**System Type**: Multi-agent schema mapping orchestration  
**Architecture**: Specialized agent clusters with intelligent orchestration  
**Datasets Supported**: 5 retail data sources (transactions, products, stores, holidays, promotions)  
**Target Schema**: 36 demand forecasting fields  
**Integration**: Successfully merges and maps data from all source datasets  

## Current Architecture (v0.2.0) - Multi-Agent System

### Implemented Multi-Agent Orchestration
```
┌─────────────────────────────────────────────────────────────────┐
│                    SCHEMA ORCHESTRATOR                          │
│              (Workflow Coordination & Performance)              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┬─────────────────────────┐
    │                 │                 │                         │
    ▼                 ▼                 ▼                         ▼
┌─────────┐    ┌─────────────┐   ┌──────────────┐    ┌─────────────────┐
│  DATA   │    │   SCHEMA    │   │ VALIDATION   │    │   UNIFIED       │
│  PREP   │───▶│   MAPPING   │──▶│   AGENT      │───▶│   OUTPUT        │
│ AGENT   │    │    AGENT    │   │              │    │ (36 fields)     │
└─────────┘    └─────────────┘   └──────────────┘    └─────────────────┘

```

### Current Components (Implemented)
- **SchemaOrchestrator**: Multi-agent workflow coordination and performance monitoring
- **DataPrepAgent**: Dataset preparation, analysis, and quality scoring
- **SchemaMappingAgent**: AI-powered semantic column mapping with confidence scoring  
- **ValidationAgentFixed**: Advanced validation with proper dataset merging and record creation
- **Intelligent Merging**: Proper join key identification (prod_code, store_label, date)
- **Target Schema**: 36 fields covering all retail forecasting dimensions

## Future Architecture (v0.3.0) - Advanced Specialized System

### Planned Multi-Agent Data Mapping Orchestration
```
                            ┌─────────────────────────────────┐
                            │       ORCHESTRATOR AGENT       │
                            │   (Mapping Performance         │
                            │    Optimization & Workflow     │
                            │    Coordination)               │
                            └─────────────┬───────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
        ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
        │   DATA PREP AGENT   │ │  SCHEMA MATCH AGENT │ │ VALIDATION & SELF-  │
        │      CLUSTER        │ │      CLUSTER        │ │ IMPROVEMENT CLUSTER │
        │ ┌─────────────────┐ │ │ ┌─────────────────┐ │ │ ┌─────────────────┐ │
        │ │ Data Prefilter  │ │ │ │ Column Analysis │ │ │ │ Constraint      │ │
        │ │     Agent       │ │ │ │     Agent       │ │ │ │ Validation      │ │
        │ └─────────────────┘ │ │ └─────────────────┘ │ │ │     Agent       │ │
        │ ┌─────────────────┐ │ │ ┌─────────────────┐ │ │ └─────────────────┘ │
        │ │ Table Selection │ │ │ │ Semantic Match  │ │ │ ┌─────────────────┐ │
        │ │     Agent       │ │ │ │     Agent       │ │ │ │ Mapping Quality │ │
        │ └─────────────────┘ │ │ └─────────────────┘ │ │ │ Assessment Agent│ │
        │ ┌─────────────────┐ │ │                     │ │ └─────────────────┘ │
        │ │ Preprocessing   │ │ │                     │ │ ┌─────────────────┐ │
        │ │     Agent       │ │ │                     │ │ │ LLM Evaluator/  │ │
        │ └─────────────────┘ │ │                     │ │ │ Optimization*   │ │
        └─────────────────────┘ └─────────────────────┘ │ └─────────────────┘ │
                    │                     │               └─────────────────────┘
                    └─────────────────────┼─────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │     UNIFIED SCHEMA    │
                              │    OUTPUT INTERFACE   │
                              │  (Feeds Downstream    │
                              │   Forecasting System) │
                              └───────────────────────┘
```

### Specialized Agent Hierarchy

#### Data Preparation Agent Cluster
- **Data Prefiltering Agent**: Pre-screening datasets for relevance and compatibility
- **Table Selection Agent**: Intelligent dataset prioritization and relevance scoring
- **Preprocessing Agent**: Adaptive cleaning, transformation, and optimization

#### Schema Matching Agent Cluster
- **Column Analysis Agent**: Deep semantic understanding and pattern recognition
- **Semantic Matching Agent**: Vector-based similarity and context-aware mapping

#### Validation & Self-Improvement Agent Cluster
- **Constraint Validation Agent**: Business rule enforcement and consistency checking
- **Mapping Quality Assessment Agent**: Quality scoring and validation with entropy-based uncertainty detection
- **Human-in-the-Loop Agent***: Interactive review interface for high-uncertainty mappings using confidence entropy metrics
- **LLM Evaluator/Optimization Agent***: Zero-shot self-improvement using Chain-of-Thought evaluation and synthetic in-context examples

*Advanced features marked with * are optional enhancements for future phases

## Directory Structure

### Current Structure (v0.2.0)
```
├── src/
│   ├── schema_mapping/          # Core schema mapping module
│   │   ├── __init__.py          # Multi-agent system exports
│   │   ├── schemas.py           # Pydantic data models (unified schema)
│   │   ├── schema_mapping_agent.py  # Legacy monolithic agent
│   │   └── agents/              # Multi-agent system
│   │       ├── __init__.py      # Agent exports
│   │       ├── base_agent.py    # Common agent functionality
│   │       ├── orchestrator.py  # Multi-agent coordinator
│   │       ├── data_prep_agent.py       # Data preparation
│   │       ├── schema_mapping_agent.py  # Intelligent mapping
│   │       ├── validation_agent.py      # Original validation
│   │       └── validation_agent_fixed.py # Fixed merging logic
│   └── __init__.py
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_agent.py           # Unit and integration tests
├── data/                        # Sample datasets (5 sources)
│   ├── transaction_like_synth.csv    # Main transaction data
│   ├── product_like_synth_wBrand.csv # Product master data
│   ├── store_like_synth.csv          # Store information
│   ├── holidays.csv                  # Holiday calendar
│   └── promotion_like_synth.csv      # Promotion data
├── output/                      # Generated outputs
│   └── unified_mapped_data.csv  # Multi-agent mapping results
├── examples/                    # Usage examples
│   ├── run_agent.py            # Legacy single-agent example
│   ├── run_multi_agent.py      # Multi-agent orchestration
│   ├── test_multi_agent.py     # Quick testing with sample data
│   └── test_unified_mapping.py # Legacy testing utilities
├── docs/                        # Documentation
│   ├── notebooks/               # Analysis notebooks
│   └── documentation/           # API and schema documentation
├── scripts/                     # Utility scripts
├── config/                      # Configuration files
├── pyproject.toml              # Project dependencies and metadata
└── README.md                   # This file
```

## Usage

### Multi-Agent System (Current - v0.2.0)

```python
from schema_mapping.agents import SchemaOrchestrator

# Initialize multi-agent orchestrator
orchestrator = SchemaOrchestrator()

# Check system health
health = await orchestrator.health_check()
print(f"System health: {health['overall_healthy']}")

# Run complete workflow on multiple datasets
result = await orchestrator.orchestrate_mapping(
    dataset_paths=[
        "data/transaction_like_synth.csv",
        "data/product_like_synth_wBrand.csv", 
        "data/store_like_synth.csv",
        "data/holidays.csv",
        "data/promotion_like_synth.csv"
    ],
    output_path="output/unified_mapped_data.csv"
)

# Access comprehensive results
summary = result["workflow_summary"]
print(f"Quality Grade: {summary['quality_grade']}")
print(f"Total Mappings: {summary['total_mappings']}")
print(f"Confidence: {summary['average_confidence']:.1%}")
print(f"Success Rate: {summary['validation_success_rate']:.1%}")
```

### Legacy Single-Agent System (v0.1.0)

```python
from schema_mapping import LegacySchemaMappingAgent

# Initialize legacy agent
agent = LegacySchemaMappingAgent()

# Map single dataset
result = await agent.map_dataset("data/product_like_synth.csv")
```

### Command Line Usage

```bash
# Run multi-agent system (recommended)
python examples/run_multi_agent.py

# Quick testing with sample data
python examples/test_multi_agent.py

# Legacy single-agent system
python examples/run_agent.py
```

### Expected Output (Multi-Agent)

```
[INFO] Initializing orchestrator...
[SUCCESS] All agents healthy

[INFO] Running multi-agent workflow...

--- Phase 1: Data Preparation ---
[DataPrepAgent] Processing 5 dataset(s)
[Orchestrator] Phase 1 completed in 0.03s

--- Phase 2: Schema Mapping ---
[SchemaMappingAgent] Created 36 mappings - 70.4% avg confidence
[Orchestrator] Phase 2 completed in 62.2s

--- Phase 3: Validation & Quality Assessment ---
[ValidationAgentFixed] Final merged dataset: (8, 46)
[ValidationAgentFixed] Created 3 complete records - 37.5% success rate
[Orchestrator] Phase 3 completed in 0.04s

[SUCCESS] WORKFLOW COMPLETED
   Duration: 62.3 seconds
   Quality Grade: B
   Total Mappings: 36
   Field Coverage: 30.6% (11/36 columns populated)
   Key Fields: date, product_id, store_id, unit_price, units_sold, category, store_city

Results saved to: output/unified_mapped_data.csv
```
