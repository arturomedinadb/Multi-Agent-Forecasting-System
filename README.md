# End-To-End AI Forecasting Agent

This project orchestrates a multi-agent workflow to clean, map, and evaluate retail demand forecasting data.

## Environment Variables

Set the following variables before running any agent or evaluation workflows:

- `DEEPEVAL_API_KEY` – optional; required only when using DeepEval metrics that rely on an external judge model (e.g., `TaskCompletionMetric`, `AnswerRelevancyMetric`). Deterministic metrics and custom checks run without this key.
- `DEEPEVAL_JUDGE_MODEL` – optional override for the language model DeepEval should use when a judge is needed. Defaults to `gpt-4o-mini` if unset.
- `DEEPEVAL_SEED` – optional integer used to seed DeepEval runs for repeatable sampling and metric behavior.

Export these variables with `export VAR_NAME=value` (or add them to your `.env`). When no API key is provided, the evaluation pipeline automatically falls back to deterministic metrics only.

## Quick Start

```bash
poetry install
poetry run schema-mapper
poetry run schema-mapping-eval
```

See `docs/evaluation/schema_mapping.md` for detailed evaluation guidance.
