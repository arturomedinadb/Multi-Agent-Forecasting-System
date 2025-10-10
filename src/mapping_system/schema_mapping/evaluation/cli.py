"""Command line interface for running schema mapping evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .deepeval_runner import (
    SchemaMappingEvaluationConfig,
    run_schema_mapping_evaluation,
)

DEFAULT_CONFIG = Path("configs") / "evaluation" / "schema_mapping.yaml"


def _resolve_required_arg(value: str | None, fallback: str | None, label: str) -> str:
    candidate = value or fallback
    if not candidate:
        raise SystemExit(f"Missing required argument: {label}")
    return candidate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval checks for the schema mapping agent workflow.")
    parser.add_argument("--final-dataset", dest="final_dataset", help="Path to the integrated target-schema dataset." )
    parser.add_argument("--mapping-plan", dest="mapping_plan", help="JSON file or string containing the mapping plan." )
    parser.add_argument("--metadata", dest="source_metadata", help="JSON file or string containing the source metadata array.")
    parser.add_argument("--target-schema", dest="target_schema", help="JSON file or string with the target schema definition.")
    parser.add_argument("--config", dest="config_path", default=str(DEFAULT_CONFIG), help="Evaluation configuration file (YAML).")
    parser.add_argument("--output-dir", dest="output_dir", help="Directory for evaluation artifacts (overrides config).")
    parser.add_argument("--print-summary", action="store_true", help="Print the JSON summary to stdout after the run.")

    args = parser.parse_args(argv)

    config_path = args.config_path if args.config_path and Path(args.config_path).exists() else None
    config = SchemaMappingEvaluationConfig.load(config_path)

    paths_cfg = config.paths or {}

    final_dataset = _resolve_required_arg(args.final_dataset, paths_cfg.get("final_dataset"), "--final-dataset")
    mapping_plan = _resolve_required_arg(args.mapping_plan, paths_cfg.get("mapping_plan"), "--mapping-plan")
    source_metadata = _resolve_required_arg(args.source_metadata, paths_cfg.get("source_metadata"), "--metadata")
    target_schema = _resolve_required_arg(args.target_schema, paths_cfg.get("target_schema"), "--target-schema")

    summary = run_schema_mapping_evaluation(
        final_dataset_path=final_dataset,
        mapping_plan=mapping_plan,
        source_metadata=source_metadata,
        target_schema=target_schema,
        config_path=config_path,
        output_dir=args.output_dir,
    )

    if args.print_summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
