# Schema Mapping Evaluation Guide

This guide explains how to run the DeepEval-powered checks for the schema mapping workflow and interpret the resulting reports.

## Prerequisites

1. Install dependencies and lock files:
   ```bash
   poetry install --no-root
   ```
2. Export environment variables (optional when using only deterministic metrics):
   ```bash
   export DEEPEVAL_API_KEY=sk-...
   export DEEPEVAL_JUDGE_MODEL=gpt-4o-mini
   export DEEPEVAL_SEED=2024
   ```
   The API key is only required for LLM judge metrics (e.g., Task Completion, Answer Relevancy). Deterministic metrics run without it.
3. Ensure the schema mapping workflow has produced the following artifacts:
   - Final dataset CSV (default: `output/final_mapped_dataset.csv`)
   - Mapping plan JSON
   - Source metadata JSON array
   - Target schema JSON (typically derived from `DemandForecastingRecord`)

## Running the Evaluator

Use the packaged CLI entry point:

```bash
poetry run schema-mapping-eval \
  --final-dataset output/final_mapped_dataset.csv \
  --mapping-plan output/mapped/mapping_plan.json \
  --metadata output/mapped/source_metadata.json \
  --target-schema output/mapped/target_schema.json \
  --print-summary
```

The command writes artifacts to `output/evaluations/<timestamp>/`:

- `summary.json` – machine-readable results (scores, thresholds, pass/fail flags)
- `summary.md` – concise human report suitable for sharing in PRs
- Input snapshots (`mapping_plan.json`, `source_metadata.json`, `target_schema.json`)

Pass `--config` to use a custom YAML configuration or `--output-dir` to override the destination directory.

## Metric Interpretations

Deterministic metrics always run:

- **Field Coverage** – ratio of required target fields present in the final dataset. Fails if coverage drops below `metrics.field_coverage.minimum_ratio`.
- **Type Compatibility** – validates dtype alignment against the target schema. Configurable through `metrics.type_compatibility.fail_on_casts`.
- **Semantic Similarity** – Jaccard-based similarity between source column descriptions/names and their mapped target fields.

Optional LLM metrics (Task Completion, Answer Relevancy, etc.) run when `DEEPEVAL_API_KEY` is present and remain skipped otherwise. Skipped metrics are reported with an explanatory reason.

## Reading the Summary

`summary.json` contains:

- `metrics` – list of deterministic outcomes (`score`, `threshold`, `success`, and diagnostic `details`).
- `llm_metrics` – optional judge outputs or skip reasons.
- `improvement_prompt` – short follow-up instructions auto-generated from failing metrics.
- `context` – high-level dataset characteristics for reproducibility.

Use `summary.md` for a lightweight textual recap (deterministic table + improvement prompt). Attach this file to PRs or incident tickets to document evaluation status.

## Common Troubleshooting

- **Missing artifacts** – ensure the mapping workflow saved `mapping_plan.json` and `source_metadata.json` under `output/mapped/` (the evaluation tool also regenerates them when possible).
- **LLM metrics skipped** – set `DEEPEVAL_API_KEY` and optionally `DEEPEVAL_JUDGE_MODEL`. Without credentials, only deterministic metrics run.
- **Unexpected dtype failures** – adjust `metrics.type_compatibility.fail_on_casts` in `configs/evaluation/schema_mapping.yaml` to tolerate expected coercions (e.g., integer→number).
- **Reproducibility** – pin `DEEPEVAL_SEED` to control randomness and capture evaluation artifacts in version control (`output/evaluations/.gitignore` keeps JSON/Markdown summaries).

## CI Integration

Add the following step to your CI workflow after running the mapping pipeline:

```bash
poetry run schema-mapping-eval --print-summary
cat output/evaluations/latest/summary.md
```

Fail the build when any deterministic metric fails or when LLM metrics return a failing score.

For more details, inspect the configuration file at `configs/evaluation/schema_mapping.yaml`.
