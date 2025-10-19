import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from agents import function_tool
from ..schemas.models import DemandForecastingRecord, ColumnMapping, MappingResult
from ..evaluation import run_schema_mapping_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation" / "schema_mapping.yaml"
DEFAULT_EVALUATION_OUTPUT = PROJECT_ROOT / "output" / "evaluations"


# === Helper Functions ===

def _get_row_limit(default: int = 10) -> int:
    """Return the configured row sampling limit for tooling workflows."""
    raw_value = os.getenv("AGENT_ROW_LIMIT", "").strip()
    if not raw_value:
        return default

    try:
        row_limit = int(raw_value)
    except ValueError:
        return default

    return row_limit if row_limit > 0 else default


def _normalize_path(value: str | None) -> str | None:
    """Normalize a file path to absolute form."""
    if not value:
        return None
    try:
        return str(Path(value).expanduser().resolve())
    except Exception:
        return os.path.abspath(value)


def _load_json_payload(raw: str | Dict[str, Any] | List[Dict[str, Any]]) -> Dict[str, Any] | List[Dict[str, Any]]:
    """
    Load JSON from string, dict, list, or file path.
    
    Tries in order:
    1. If already dict/list, return as-is
    2. Parse as inline JSON string
    3. Try as file path
    4. Raise error if all fail
    """
    if isinstance(raw, (dict, list)):
        return raw
    
    text = raw.strip()
    
    # Attempt to parse inline JSON first to avoid filesystem lookups on large payloads
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    
    # Try as file path
    candidate_path = Path(text)
    try:
        if candidate_path.exists():
            return json.loads(candidate_path.read_text())
    except OSError:
        pass
    
    # Last resort: parse as JSON string
    return json.loads(text)

# --- Data Preparation Tools ---

@function_tool
def load_and_describe_dataset(file_path: str) -> str:
    """
    Load the top 10 rows from a CSV and return lightweight metadata including
    columns and dtypes plus a small sample. The AGENT will write the
    column_descriptions based on this context.

    Returns a JSON string with keys: file_path, shape, columns, dtypes, sample.
    """
    row_limit = _get_row_limit()
    print(f"TOOL: Loading top {row_limit} rows of '{file_path}'...")
    try:
        df = pd.read_csv(file_path, nrows=row_limit)
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample": df.head(2).to_dict(orient="records"),
        }
        return json.dumps(metadata)
    except Exception as e:
        print(f"TOOL ERROR in load_and_describe_dataset: {e}")
        return json.dumps({"error": str(e), "file_path": file_path})

# --- Schema Mapping Tool ---

def run_generate_mapped_csvs(
    source_metadata_json: str,
    mappings_json: str,
    output_dir: str,
) -> str:
    """
    Shared implementation for generating mapped CSV manifests.
    
    Takes source metadata and mapping plan, generates per-dataset mapped CSVs.
    Used by both the @function_tool wrapper and handoff filters.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        source_meta = json.loads(source_metadata_json) or []
        mapping_payload = json.loads(mappings_json) or {}
    except Exception as e:
        print(f"TOOL ERROR in generate_mapped_csvs: Invalid JSON input - {e}")
        return json.dumps({"outputs": [], "error": f"Invalid JSON input: {e}"})

    mapping_entries = mapping_payload.get("mappings", [])
    per_file: Dict[str, List[Dict[str, Any]]] = {}
    fallback: List[Dict[str, Any]] = []

    for entry in mapping_entries:
        if not isinstance(entry, dict):
            continue
        if "source_file" in entry:
            normalized_key = _normalize_path(entry.get("source_file"))
            if normalized_key:
                mappings_for_file = entry.get("mappings", [])
                if isinstance(mappings_for_file, list):
                    filtered = [m for m in mappings_for_file if isinstance(m, dict)]
                    per_file[normalized_key] = filtered
                    per_file[Path(normalized_key).name] = filtered
        elif entry.get("source_column") and entry.get("target_column"):
            fallback.append(entry)

    results: List[Dict[str, Any]] = []

    row_limit = _get_row_limit()

    for meta in source_meta:
        path = meta.get("file_path")
        if not path:
            continue

        normalized_meta_path = _normalize_path(path)
        mapping_list = per_file.get(normalized_meta_path)

        if mapping_list is None and normalized_meta_path is not None:
            mapping_list = per_file.get(os.path.basename(normalized_meta_path))

        if mapping_list is None:
            mapping_list = fallback

        if not mapping_list:
            print(f"TOOL WARNING: No mappings matched for {path}; skipping file")
            continue

        try:
            df = pd.read_csv(path, nrows=row_limit)
        except Exception as e:
            print(f"TOOL ERROR: Unable to read '{path}': {e}")
            continue

        df = df.astype(object).where(pd.notnull(df), None)

        mapped_df = pd.DataFrame()
        used_targets: set[str] = set()

        for m in mapping_list:
            src = m.get("source_column")
            tgt = m.get("target_column")
            if not src or not tgt:
                continue
            if src not in df.columns:
                continue
            if tgt in used_targets:
                continue
            mapped_df[tgt] = df[src]
            used_targets.add(tgt)

        if mapped_df.empty:
            print(f"TOOL WARNING: No valid columns mapped for {path}; skipping file")
            continue

        out_name = f"{Path(path).stem}_mapped.csv"
        out_path = Path(output_dir) / out_name
        mapped_df.to_csv(out_path, index=False)

        results.append({
            "source_file": path,
            "output_path": str(out_path),
            "columns": list(mapped_df.columns),
        })

    print(f"TOOL: Wrote {len(results)} mapped CSVs to {output_dir}")
    return json.dumps({"outputs": results})


def run_schema_mapping_deepeval(
    *,
    final_dataset_path: str,
    mapping_plan_json: str,
    source_metadata_json: str,
    target_schema_json: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    mapping_plan = _load_json_payload(mapping_plan_json)
    source_metadata = _load_json_payload(source_metadata_json)
    target_schema = _load_json_payload(target_schema_json)

    if not isinstance(mapping_plan, dict):
        raise ValueError("mapping_plan_json must decode to a JSON object")
    if not isinstance(source_metadata, list):
        raise ValueError("source_metadata_json must decode to a JSON array")
    if not isinstance(target_schema, dict):
        raise ValueError("target_schema_json must decode to a JSON object")

    mapped_dir = PROJECT_ROOT / "output" / "mapped"
    mapped_dir.mkdir(parents=True, exist_ok=True)

    (mapped_dir / "mapping_plan.json").write_text(json.dumps(mapping_plan, indent=2))
    (mapped_dir / "source_metadata.json").write_text(json.dumps(source_metadata, indent=2))
    (mapped_dir / "target_schema.json").write_text(json.dumps(target_schema, indent=2))

    resolved_config = config_path or (str(DEFAULT_EVALUATION_CONFIG) if DEFAULT_EVALUATION_CONFIG.exists() else None)
    resolved_output = output_dir or str(DEFAULT_EVALUATION_OUTPUT)

    summary = run_schema_mapping_evaluation(
        final_dataset_path=final_dataset_path,
        mapping_plan=mapping_plan,
        source_metadata=source_metadata,
        target_schema=target_schema,
        config_path=resolved_config,
        output_dir=resolved_output,
    )

    return summary


@function_tool
def generate_mapped_csvs(source_metadata_json: str, mappings_json: str, output_dir: str) -> str:
    """
    Apply mappings separately per source dataset and write one CSV per dataset
    containing only mapped target columns.

    Returns JSON: {"outputs": [{"source_file": str, "output_path": str, "columns": [..]}]}
    """
    print("TOOL: Generating per-dataset mapped CSVs...")
    try:
        return run_generate_mapped_csvs(source_metadata_json, mappings_json, output_dir)
    except Exception as e:
        print(f"TOOL ERROR in generate_mapped_csvs: {e}")
        return json.dumps({"outputs": [], "error": str(e)})


@function_tool
def merge_mapped_csvs_to_target(mapped_outputs_json: str, target_schema_json: str, output_path: str) -> str:
    """
    Merge multiple mapped CSVs (each already using target columns) into a single
    CSV matching the full target schema. Fills present fields, leaves others blank.

    Inputs:
    - mapped_outputs_json: JSON from generate_mapped_csvs tool: {"outputs": [{"output_path": str, ...}, ...]}
    - target_schema_json: JSON schema from DemandForecastingRecord.model_json_schema()
    - output_path: path to write the integrated CSV

    Behavior:
    - Loads each mapped CSV (top rows were produced earlier, but we merge whatever is present)
    - Iteratively left-joins on available common key columns among ['date','product_id','store_id','store_region']
    - Ensures all target columns exist; missing columns are added with None
    - Writes the final CSV
    """
    print("TOOL: Merging mapped CSVs into target schema CSV...")
    try:
        raw = json.loads(mapped_outputs_json) if isinstance(mapped_outputs_json, str) else mapped_outputs_json

        if isinstance(raw, dict):
            mapped = raw.get("outputs", []) or raw.get("output", [])
        elif isinstance(raw, list):
            mapped = raw
        else:
            mapped = []

        normalized: List[Dict[str, Any]] = []
        for entry in mapped:
            if isinstance(entry, str):
                try:
                    entry = json.loads(entry)
                except json.JSONDecodeError:
                    continue
            if isinstance(entry, dict):
                normalized.append(entry)

        mapped = normalized
        if not mapped:
            return json.dumps({"status": "Failed", "error": "No mapped outputs provided."})

        # Load mapped dataframes
        dataframes: List[pd.DataFrame] = []
        for entry in mapped:
            p = entry.get("output_path")
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                dataframes.append(df)
            else:
                print(f"TOOL WARNING: Missing mapped file: {p}")

        if not dataframes:
            return json.dumps({"status": "Failed", "error": "No mapped CSVs could be loaded."})

        # Start with the widest dataframe
        main_df = max(dataframes, key=lambda d: d.shape[1]).copy()
        others = [d for d in dataframes if d is not main_df]

        candidate_keys = ["date", "product_id", "store_id", "store_region"]

        # Iterative left joins on available common keys
        for df in others:
            common = [k for k in candidate_keys if k in main_df.columns and k in df.columns]
            if common:
                try:
                    main_df = pd.merge(main_df, df, on=common, how="left", suffixes=("", "_dup"))
                    dup_cols = [c for c in main_df.columns if c.endswith("_dup")]
                    if dup_cols:
                        main_df.drop(columns=dup_cols, inplace=True)
                    print(f"TOOL: Merged on keys {common}")
                except Exception as me:
                    print(f"TOOL WARNING: Merge failed on keys {common}: {me}")
            else:
                print("TOOL WARNING: No common keys; skipping a dataframe")

        # Ensure full target schema columns exist
        target_schema = json.loads(target_schema_json)
        target_fields: List[str] = list(target_schema.get("properties", {}).keys())
        for col in target_fields:
            if col not in main_df.columns:
                main_df[col] = None

        # Reorder columns to target schema order
        main_df = main_df[target_fields]

        # Normalize NaNs to None and write CSV
        main_df = main_df.astype(object).where(pd.notnull(main_df), None)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        main_df.to_csv(output_path, index=False)

        return json.dumps({
            "status": "Success",
            "output_path": output_path,
            "rows": len(main_df),
            "columns": list(main_df.columns)
        })
    except Exception as e:
        print(f"TOOL ERROR in merge_mapped_csvs_to_target: {e}")
        import traceback; traceback.print_exc()
        return json.dumps({"status": "Failed", "error": str(e)})


@function_tool
def evaluate_schema_mapping_with_deepeval(
    final_dataset_path: str,
    mapping_plan_json: str,
    source_metadata_json: str,
    target_schema_json: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Run the DeepEval-powered evaluation pipeline for the schema mapping workflow."""

    try:
        summary = run_schema_mapping_deepeval(
            final_dataset_path=final_dataset_path,
            mapping_plan_json=mapping_plan_json,
            source_metadata_json=source_metadata_json,
            target_schema_json=target_schema_json,
            config_path=config_path,
            output_dir=output_dir,
        )
        return json.dumps(summary, indent=2)
    except Exception as exc:
        print(f"TOOL ERROR in evaluate_schema_mapping_with_deepeval: {exc}")
        return json.dumps({"status": "Failed", "error": str(exc)})
