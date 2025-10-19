# End-to-End AI Schema Mapping & Forecasting Agent System

An intelligent multi-agent system that transforms heterogeneous retail datasets into standardized demand forecasting schemas using OpenAI's Agents SDK.

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export OPENAI_API_KEY="sk-your-key-here"

# Run workflow
python scripts/examples/run_four_agents.py
```

## Documentation

- `docs/HOW_JINJA2_WORKS.md` - Understanding the Jinja2 template system
- `scripts/examples/README.md` - Testing guide

## System Architecture

This system implements a 4-agent sequential workflow:
1. **DataPrepAgent** - Analyzes source datasets
2. **ColumnMappingAgent** - Creates semantic column mappings
3. **DataIntegrationAgent** - Merges mapped CSVs
4. **SchemaMappingEvaluationAgent** - Validates quality

For more details, see the documentation files.

