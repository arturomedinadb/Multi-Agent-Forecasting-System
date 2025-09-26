# End-to-End AI Forecasting Agent System

A multi-agent AI system for automated data schema mapping and demand forecasting. This project transforms heterogeneous retail datasets into unified forecasting models through intelligent agent orchestration and specialized AI workflows.

## Project Overview

**Course**: Independent Study: AI Forecasting Agent  
**Instructor**: Fatih Nayebi  
**Team**: Ayda Elzohbi, Arturo Medina, Arial Huang, Katherine Gong

### Current System Status (v0.1.0)

**System Type**: Single-agent schema mapping system  
**Architecture**: Monolithic agent with LLM-powered mapping  
**Datasets Supported**: 5 retail data sources (transactions, products, stores, holidays, promotions)  
**Target Schema**: 35+ demand forecasting fields  

### Key Capabilities

- **Intelligent Schema Mapping**: LLM-powered semantic column understanding
- **Multi-Dataset Integration**: Unified mapping across 5+ heterogeneous datasets
- **Robust Validation**: Pydantic-based schema validation with transformation
- **Agent Framework**: Built on openai-agents with function tools
- **Production Pipeline**: End-to-end data processing with comprehensive error handling

## Current Architecture (v0.1.0)

### Single-Agent System
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Multiple CSV  │───▶│  Schema Mapping  │───▶│  Unified Schema │
│   Datasets      │    │     Agent        │    │  Output (CSV)   │
│   (5 sources)   │    │   (Monolithic)   │    │  35+ Fields     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   LLM-Powered    │
                    │   Column         │
                    │   Analysis       │
                    └──────────────────┘
```

### Current Components
- **SchemaMappingAgent**: Monolithic agent handling all tasks
- **Function Tools**: CSV loading, pattern analysis, validation, confidence scoring
- **Target Schema**: 35 fields covering sales, products, stores, promotions, economics
- **Data Sources**: Transactions, products, stores, holidays, promotions

## Target Architecture (v0.2.0) - Specialized Data Mapping System

### Multi-Agent Data Mapping Orchestration
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

### Current Structure (v0.1.0)
```
├── src/
│   ├── schema_mapping/          # Core schema mapping module
│   │   ├── __init__.py
│   │   ├── schemas.py           # Pydantic data models (99 lines)
│   │   └── schema_mapping_agent.py  # Monolithic agent (695 lines)
│   └── __init__.py
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_agent.py           # Unit and integration tests
├── data/                        # Sample datasets
│   ├── transaction_like_synth.csv    # Main transaction data
│   ├── product_like_synth_wBrand.csv # Product master data
│   ├── store_like_synth.csv          # Store information
│   ├── holidays.csv                  # Holiday calendar
│   └── promotion_like_synth.csv      # Promotion data
├── docs/                        # Documentation
│   ├── notebooks/               # Jupyter notebooks
│   └── documentation/           # API documentation
├── examples/                    # Usage examples
│   ├── run_agent.py            # Main example script
│   └── test_unified_mapping.py # Testing utilities
├── scripts/                     # Utility scripts
│   └── run_with_uv.sh          # UV-based runner
├── config/                      # Configuration files
├── pyproject.toml              # Project dependencies and metadata
└── README.md                   # This file
```

## Usage

### Basic Usage

```python
from src.schema_mapping import SchemaMappingAgent

# Initialize agent
agent = SchemaMappingAgent()

# Map single dataset
result = await agent.map_dataset("data/product_like_synth.csv")

# Map multiple datasets
result = await agent.map_dataset(
    "data/product_like_synth.csv", 
    "data/store_like_synth.csv"
)

# Access results
print(f"Mapping confidence: {result.mapping_confidence:.2%}")
print(f"Successfully mapped: {result.successful_records}/{result.total_records}")
```

### Command Line Usage

```bash
# Run with AI-powered mapping
python examples/run_agent.py

# Run basic functionality test
python examples/test_unified_mapping.py

# Run with UV (handles environment issues)
./scripts/run_with_uv.sh
```

### Expected Output

```
Schema Mapping Agent initialized
Target schema loaded with 18 fields
Processing multiple files: product_like_synth.csv, store_like_synth.csv

Mapping Results:
   Total records: 30
   Successful: 28
   Confidence: 87%
   Unmapped columns: ['item_weight']

Results saved to: ai_mapped_output.csv
```
