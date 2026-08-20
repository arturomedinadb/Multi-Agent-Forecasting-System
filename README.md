<h1 align="center">
  🔮 AI Forecasting Agents
</h1>

<p align="center">
  <strong>End-to-end AI-powered demand forecasting with intelligent schema mapping</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/OpenAI-Agents%20SDK-412991" alt="OpenAI Agents">
</p>

---

## Overview

**AI Forecasting Agents** is a multi-agent system that automates the entire demand forecasting pipeline—from raw, heterogeneous retail data to trained, evaluated ML models. Built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), it combines intelligent data transformation with state-of-the-art forecasting.

### 🎯 The Problem

Retail demand forecasting requires:
1. **Data wrangling** — Mapping disparate data sources (transactions, products, stores, weather, economic indicators) into a unified schema
2. **Model training** — Building and tuning ML models with proper evaluation
3. **Iteration** — Continuously improving based on feedback

This traditionally requires significant manual effort and domain expertise.

### 💡 Our Solution

Two intelligent agent systems that work together:

| System | Purpose | Agents |
|--------|---------|--------|
| **Schema Mapping** | Transform any retail data into a standardized forecasting schema | DataPrep → ColumnMapping → DataIntegration (each with evaluators) |
| **Demand Forecasting** | Train, evaluate, and optimize ML models | Training ↔ Evaluation (with automated feedback loops) |

---

## Features

### 🗺️ Schema Mapping System
- **Multi-agent orchestration** — Coordinated workflow with specialized agents
- **Semantic understanding** — AI-powered column mapping that understands data meaning, not just names
- **Chain-based evaluation** — Each agent has a dedicated evaluator for quality assurance
- **Session memory** — SQLAlchemy-based context sharing across agents
- **Jinja2 templating** — Customizable prompts for different use cases

### 📈 Demand Forecasting System
- **Multi-model support** — XGBoost, LightGBM, CatBoost, and ensemble methods
- **Automated optimization** — Hyperparameter tuning via Optuna/Hyperopt
- **Feedback loops** — Evaluation agent provides actionable improvements
- **Feature engineering** — Automated feature creation based on model feedback
- **Model interpretation** — SHAP and LIME for explainability

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI FORECASTING AGENTS                        │
├─────────────────────────────────┬───────────────────────────────────┤
│      SCHEMA MAPPING SYSTEM      │     DEMAND FORECASTING SYSTEM     │
│                                 │                                   │
│  ┌───────────────────────────┐  │  ┌─────────────────────────────┐  │
│  │    Orchestrator Agent     │  │  │      Training Agent         │  │
│  └─────────────┬─────────────┘  │  │  • Model configuration      │  │
│                │                │  │  • Data preprocessing       │  │
│       ┌────────┼────────┐       │  │  • Feature engineering      │  │
│       ▼        ▼        ▼       │  │  • Hyperparameter tuning    │  │
│  ┌─────────┐┌─────────┐┌─────┐  │  └──────────────┬──────────────┘  │
│  │DataPrep ││ Column  ││Data │  │                 │                 │
│  │ Agent   ││ Mapping ││Integ│  │                 ▼                 │
│  └────┬────┘└────┬────┘└──┬──┘  │  ┌─────────────────────────────┐  │
│       │          │        │     │  │     Evaluation Agent        │  │
│       ▼          ▼        ▼     │  │  • Performance metrics      │  │
│  ┌─────────┐┌─────────┐┌─────┐  │  │  • Convergence checking     │  │
│  │Evaluator││Evaluator││Eval │  │  │  • Feedback generation      │  │
│  └─────────┘└─────────┘└─────┘  │  └─────────────────────────────┘  │
│                                 │                                   │
└─────────────────────────────────┴───────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Standardized Schema   │
                    │   + Trained Models      │
                    └─────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- [uv](https://github.com/astral-sh/uv)
- OpenAI API key

### Clone and Install

```bash
# Clone and install
git clone https://github.com/your-org/End-To-End-AI-Forecasting-Agent.git
cd End-To-End-AI-Forecasting-Agent

# Install with uv
uv sync
```

### Environment Setup

```bash
# Copy the environment template
cp .env.example .env

# Edit with your API key
echo "OPENAI_API_KEY=your-key-here" >> .env
```

---

## Quick Start

### 1️⃣ Schema Mapping

Transform your retail data into the standardized demand forecasting schema:

```bash
# Run the schema mapper
uv run schema-mapper
```

Or programmatically:

```python
import asyncio
from schema_mapping.run_workflow import run_full_workflow

async def main():
    result = await run_full_workflow(
        source_files=[
            "data/transaction_like_synth.csv",
            "data/product_like_synth_wBrand.csv",
            "data/store_like_synth.csv",
            "data/weather_monthly.csv",
        ],
        row_limit=1000,
        output_dir="output/mapped_data"
    )
    print(f"Mapping complete: {result}")

asyncio.run(main())
```

### 2️⃣ Demand Forecasting

Train and evaluate forecasting models:

```bash
# Run the forecasting pipeline
python scripts/run_demand_forecasting.py
```

Or programmatically:

```python
from ai_forecasting_agents.demand_forecasting.agents import training_agent, evaluation_agent
from agents import Runner

async def train_models():
    runner = Runner()
    result = await runner.run(
        training_agent,
        input="Train XGBoost and LightGBM models on the mapped data",
        context={
            "input_file": "output/mapped_data/final_dataset.csv",
            "output_dir": "output/training_results"
        }
    )
    return result
```

---

## Project Structure

```
End-To-End-AI-Forecasting-Agent/
├── src/
│   ├── schema_mapping/           # 🗺️ Data transformation system
│   │   ├── agents/               # Agent definitions
│   │   ├── evaluation/           # Evaluation metrics
│   │   ├── prompts/              # Renderer + registry loader code
│   │   ├── schemas/              # Pydantic models
│   │   ├── tools/                # Agent tool functions
│   │   └── run_workflow.py       # Main entry point
│   │
│   ├── ai_forecasting_agents/    # 📈 ML training system
│   │   └── demand_forecasting/
│   │       ├── agents/           # Training & evaluation agents
│   │       ├── prompts/          # Renderer factory (reuses schema_mapping's renderer)
│   │       ├── schemas/          # Model output schemas
│   │       └── tools/            # ML functions
│   │
│   └── ui/                       # 🖥️ Flask web UI for the full pipeline
│       └── templates/            # Jinja2 HTML templates
│
├── data/                         # Sample datasets
├── prompts/                      # LLM prompt templates (Jinja2 + registry.yaml)
│   ├── schema_mapping/           # DataPrepAgent, ColumnMappingAgent, etc.
│   └── demand_forecasting/       # TrainingAgent, EvaluationAgent, etc.
├── tests/                        # pytest suite (mirrors schema_mapping / demand_forecasting)
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
└── output/                       # Generated outputs
```

---

## Target Schema

The system maps data to this standardized demand forecasting schema:

| Category | Key Fields |
|----------|------------|
| **Identifiers** | `date`, `product_id`, `store_id` |
| **Sales** | `units_sold`, `unit_net_price` |
| **Promotions** | `promotion_active`, `promotional_price`, `promotion_start_date` |
| **Product** | `category`, `subcategory`, `is_seasonal_product` |
| **Store** | `store_cluster`, `store_size_sqft`, `store_region` |
| **Weather** | `avg_temperature_c`, `precipitation_mm` |
| **Economic** | `unemployment_rate_monthly`, `cpi_monthly`, `gdp_monthly` |

See [docs/demand_forecasting_schema.md](docs/demand_forecasting_schema.md) for the complete schema.

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | ✅ |
| `OPENAI_MODEL` | Model to use (default: `gpt-4o-mini`) | ❌ |
| `LOG_LEVEL` | Logging level (default: `INFO`) | ❌ |

### Prompt Customization

Customize agent behavior via Jinja2 templates in `prompts/`, split per system:

```
prompts/
├── schema_mapping/
│   ├── registry.yaml
│   └── v1/
│       ├── column_mapping_agent.j2
│       ├── data_prep_agent.j2
│       └── data_integration_agent.j2
└── demand_forecasting/
    ├── registry.yaml
    └── v1/
        ├── training_agent.j2
        └── evaluation_agent.j2
```

Each agent's instructions are rendered once at import time from its template
(via `registry.yaml`), rather than living inline in the agent definitions.

---

## Supported Models

| Model | Type | Best For |
|-------|------|----------|
| **XGBoost** | Gradient Boosting | General purpose, fast |
| **LightGBM** | Gradient Boosting | Large datasets, categorical features |
| **CatBoost** | Gradient Boosting | Categorical-heavy data |
| **Ensemble** | Stacking/Voting | Maximum accuracy |

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
ruff check src/
```

---

## User Interface (Prototype)

A Flask web UI that automates demand forecasting: upload CSVs and let AI agents take care of schema mapping, feature engineering, and model training.

✨ **Features**
- Multi-file CSV upload
- AI-powered schema detection
- Automated feature engineering
- Multi-model training with tuning
- Real-time progress dashboard
- Session tracking + detailed outputs

  
➡️ **Check out the UI prototype and its dedicated README file for more details.**


---

## Authors

- **Ayda Elzohbi** 
- **Katherine Gong** 
- **Arial Huang** 
- **Arturo Medina**

---

<p align="center">
  Built with ❤️ using <a href="https://github.com/openai/openai-agents-python">OpenAI Agents SDK</a>
</p>

