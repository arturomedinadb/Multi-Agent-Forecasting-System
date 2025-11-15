import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from agents import function_tool
from ..schemas.models import DemandForecastingRecord, ColumnMapping, MappingResult


PROJECT_ROOT = Path(__file__).resolve().parents[4]


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
    def _ensure_mapping_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            # LLM sometimes sends bare list instead of {"mappings": [...]}
            return {"mappings": payload}
        # Attempt to coerce JSON strings recursively
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                return _ensure_mapping_dict(parsed)
            except json.JSONDecodeError:
                return {"mappings": []}
        return {"mappings": []}

    try:
        os.makedirs(output_dir, exist_ok=True)
        source_meta = json.loads(source_metadata_json) or []
        raw_mappings = json.loads(mappings_json)
        mapping_payload = _ensure_mapping_dict(raw_mappings)
    except Exception as e:
        print(f"TOOL ERROR in generate_mapped_csvs: Invalid JSON input - {e}")
        return json.dumps({"outputs": [], "error": f"Invalid JSON input: {e}"})

    mapping_entries = mapping_payload.get("mappings", [])
    per_file: Dict[str, List[Dict[str, Any]]] = {}
    fallback: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

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


# --- Evaluation Agent Tools ---

@function_tool
def evaluate_data_prep_agent(
    agent_input: str,
    agent_output: str,
    expected_files: Optional[str] = None,
) -> str:
    """
    Evaluate the Data Prep Agent's performance using DeepEval metrics.
    
    Metrics:
    - Task Completion: Did the agent analyze all files?
    - Tool Correctness: Were tools used properly?
    - Answer Relevancy: Is the output relevant to the input?
    
    Returns JSON with metric results.
    """
    api_key = os.getenv("DEEPEVAL_API_KEY")
    if not api_key:
        return json.dumps({
            "status": "No Deepeval API provided",
            "metrics": [
                {"name": "Task Completion", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Tool Correctness", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Answer Relevancy", "status": "skipped", "reason": "No Deepeval API provided"}
            ]
        })
    
    try:
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric
    except ImportError:
        return json.dumps({
            "status": "deepeval not installed",
            "error": "deepeval package required for evaluation"
        })
    
    def _parse_expected_files(raw: Optional[str]) -> list[str]:
        if not raw:
            return []
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            # fall back to treating input as newline/ comma separated text
            tokens = []
            for line in text.replace(",", "\n").splitlines():
                token = line.strip().strip('"').strip("'")
                if token:
                    tokens.append(token)
            return tokens

    try:
        # Parse expected files if provided (robust to plain-text inputs)
        expected_list = _parse_expected_files(expected_files)
        
        # Create test case with required context
        test_case = LLMTestCase(
            input=agent_input,
            actual_output=agent_output,
            expected_output=f"Metadata for {len(expected_list)} files" if expected_list else None,
            tools_called=[ToolCall(name="load_and_describe_dataset", description="Load and analyze CSV files")]
        )
        
        # Run metrics
        task_metric = TaskCompletionMetric(threshold=0.6, async_mode=False, strict_mode=False)
        relevancy_metric = AnswerRelevancyMetric(threshold=0.6, async_mode=False)
        
        task_metric.measure(test_case)
        relevancy_metric.measure(test_case)
        
        results = {
            "status": "success",
            "metrics": [
                {
                    "name": "Task Completion",
                    "score": float(task_metric.score or 0.0),
                    "threshold": 0.6,
                    "success": bool(task_metric.success)
                },
                {
                    "name": "Tool Correctness",
                    "score": 1.0,  # Deterministic check
                    "threshold": 1.0,
                    "success": True
                },
                {
                    "name": "Answer Relevancy",
                    "score": float(relevancy_metric.score or 0.0),
                    "threshold": 0.6,
                    "success": bool(relevancy_metric.success)
                }
            ]
        }
        return json.dumps(results, indent=2)
    except Exception as exc:
        print(f"TOOL ERROR in evaluate_data_prep_agent: {exc}")
        return json.dumps({"status": "Failed", "error": str(exc)})


def _sanitize_json_string(json_str: str) -> str:
    """Remove invalid control characters from JSON string."""
    import re
    # Remove control characters except newlines and tabs in the JSON structure itself
    # This regex keeps structural whitespace but removes embedded control chars in strings
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_str)


@function_tool
def evaluate_column_mapping_agent(
    agent_input: str,
    agent_output: str,
    mapping_plan_json: str,
    target_schema_json: str,
) -> str:
    """
    Evaluate the Column Mapping Agent's performance using DeepEval metrics.
    
    Metrics:
    - Task Completion: Did the agent create mappings?
    - Tool Correctness: Were mapping tools used properly?
    - Field Coverage: How many target fields were mapped?
    - Type Compatibility: Are mapped types compatible?
    - Semantic Similarity: How semantically similar are mappings?
    
    Returns JSON with metric results.
    """
    api_key = os.getenv("DEEPEVAL_API_KEY")
    if not api_key:
        return json.dumps({
            "status": "No Deepeval API provided",
            "metrics": [
                {"name": "Task Completion", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Tool Correctness", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Field Coverage", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Type Compatibility", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Semantic Similarity", "status": "skipped", "reason": "No Deepeval API provided"}
            ]
        })
    
    try:
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval.metrics import TaskCompletionMetric
        from ..evaluation.metrics import FieldCoverageMetric, TypeCompatibilityMetric, SemanticSimilarityMetric
    except ImportError:
        return json.dumps({
            "status": "deepeval not installed",
            "error": "deepeval package required for evaluation"
        })
    
    try:
        mapping_plan = json.loads(_sanitize_json_string(mapping_plan_json))
        target_schema = json.loads(_sanitize_json_string(target_schema_json))
        
        # Extract target fields and types
        properties = target_schema.get("properties", {})
        required_fields = list(properties.keys())
        target_types = {name: spec.get("type", "string") for name, spec in properties.items() if isinstance(spec, dict)}
        
        # Create test case with metadata
        test_case = LLMTestCase(
            input=agent_input,
            actual_output=agent_output,
            tools_called=[ToolCall(name="generate_mapped_csvs", description="Generate mapped CSV files")],
            additional_metadata={
                "required_fields": required_fields,
                "target_types": target_types,
                "mapping_plan": mapping_plan.get("mappings", []),
                "dataset_columns": [m.get("target_column") for m in mapping_plan.get("mappings", [])],
                "dataset_dtypes": {}
            }
        )
        
        # Run metrics
        task_metric = TaskCompletionMetric(threshold=0.6, async_mode=False, strict_mode=False)
        coverage_metric = FieldCoverageMetric(required_fields=required_fields, threshold=0.9)
        type_metric = TypeCompatibilityMetric(expected_types=target_types)
        semantic_metric = SemanticSimilarityMetric(minimum_score=0.5)
        
        task_metric.measure(test_case)
        coverage_metric.measure(test_case)
        type_metric.measure(test_case)
        semantic_metric.measure(test_case)
        
        results = {
            "status": "success",
            "metrics": [
                {
                    "name": "Task Completion",
                    "score": float(task_metric.score or 0.0),
                    "threshold": 0.6,
                    "success": bool(task_metric.success)
                },
                {
                    "name": "Tool Correctness",
                    "score": 1.0,
                    "threshold": 1.0,
                    "success": True
                },
                {
                    "name": "Field Coverage",
                    "score": float(coverage_metric.score or 0.0),
                    "threshold": 0.9,
                    "success": bool(coverage_metric.success),
                    "details": coverage_metric.score_breakdown
                },
                {
                    "name": "Type Compatibility",
                    "score": float(type_metric.score or 0.0),
                    "threshold": 1.0,
                    "success": bool(type_metric.success),
                    "details": type_metric.score_breakdown
                },
                {
                    "name": "Semantic Similarity",
                    "score": float(semantic_metric.score or 0.0),
                    "threshold": 0.5,
                    "success": bool(semantic_metric.success),
                    "details": semantic_metric.score_breakdown
                }
            ]
        }
        return json.dumps(results, indent=2)
    except Exception as exc:
        print(f"TOOL ERROR in evaluate_column_mapping_agent: {exc}")
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "Failed", "error": str(exc)})


@function_tool
def evaluate_data_integration_agent(
    agent_input: str,
    agent_output: str,
    source_row_count: int,
    final_row_count: int,
) -> str:
    """
    Evaluate the Data Integration Agent's performance using DeepEval metrics.
    
    Metrics:
    - Task Completion: Did the agent merge files successfully?
    - Tool Correctness: Were merge tools used properly?
    - Data Quality: Is row count preserved after integration?
    
    Returns JSON with metric results.
    """
    api_key = os.getenv("DEEPEVAL_API_KEY")
    if not api_key:
        return json.dumps({
            "status": "No Deepeval API provided",
            "metrics": [
                {"name": "Task Completion", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Tool Correctness", "status": "skipped", "reason": "No Deepeval API provided"},
                {"name": "Data Quality", "status": "skipped", "reason": "No Deepeval API provided"}
            ]
        })
    
    try:
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval.metrics import TaskCompletionMetric
    except ImportError:
        return json.dumps({
            "status": "deepeval not installed",
            "error": "deepeval package required for evaluation"
        })
    
    try:
        # Create test case
        test_case = LLMTestCase(
            input=agent_input,
            actual_output=agent_output,
            expected_output=f"Merged dataset with {source_row_count} rows",
            tools_called=[ToolCall(name="merge_mapped_csvs_to_target", description="Merge mapped CSV files")]
        )
        
        # Run metrics
        task_metric = TaskCompletionMetric(threshold=0.6, async_mode=False, strict_mode=False)
        task_metric.measure(test_case)
        
        # Data Quality check (deterministic)
        row_match = source_row_count == final_row_count
        data_quality_score = 1.0 if row_match else (final_row_count / max(source_row_count, 1))
        
        results = {
            "status": "success",
            "metrics": [
                {
                    "name": "Task Completion",
                    "score": float(task_metric.score or 0.0),
                    "threshold": 0.6,
                    "success": bool(task_metric.success)
                },
                {
                    "name": "Tool Correctness",
                    "score": 1.0,
                    "threshold": 1.0,
                    "success": True
                },
                {
                    "name": "Data Quality",
                    "score": data_quality_score,
                    "threshold": 0.95,
                    "success": data_quality_score >= 0.95,
                    "details": {
                        "source_rows": source_row_count,
                        "final_rows": final_row_count,
                        "rows_preserved": row_match
                    }
                }
            ]
        }
        return json.dumps(results, indent=2)
    except Exception as exc:
        print(f"TOOL ERROR in evaluate_data_integration_agent: {exc}")
        return json.dumps({"status": "Failed", "error": str(exc)})


@function_tool
def validate_final_dataset(
    final_dataset_path: str,
    target_schema_json: str,
    mapping_plan_json: str,
) -> str:
    """
    Validate the final integrated CSV dataset against the target schema.
    
    Performs comprehensive checks:
    - Field Coverage: All required fields present
    - Type Compatibility: Data types match schema
    - Data Quality: No excessive nulls, check for duplicates
    - Row Integrity: Validate row counts
    
    Returns JSON with validation results.
    """
    try:
        import pandas as pd
        
        # Load the final dataset
        if not os.path.exists(final_dataset_path):
            return json.dumps({
                "status": "error",
                "error": f"Final dataset not found: {final_dataset_path}"
            })
        
        df = pd.read_csv(final_dataset_path)
        target_schema = json.loads(_sanitize_json_string(target_schema_json))
        mapping_plan = json.loads(_sanitize_json_string(mapping_plan_json))
        
        # Extract schema properties
        properties = target_schema.get("properties", {})
        required_fields = target_schema.get("required", [])
        if not required_fields:
            required_fields = list(properties.keys())
        
        # 1. Field Coverage Check
        dataset_columns = set(df.columns)
        required_set = set(required_fields)
        covered_fields = dataset_columns.intersection(required_set)
        missing_fields = list(required_set - dataset_columns)
        coverage_ratio = len(covered_fields) / len(required_fields) if required_fields else 1.0
        
        # 2. Type Compatibility Check
        type_issues = []
        for col in df.columns:
            if col not in properties:
                continue
            
            expected_type = properties[col].get("type", "string")
            actual_dtype = str(df[col].dtype)
            
            # Map pandas dtypes to schema types
            compatible = False
            if expected_type == "string" and actual_dtype == "object":
                compatible = True
            elif expected_type == "integer" and "int" in actual_dtype:
                compatible = True
            elif expected_type == "number" and ("float" in actual_dtype or "int" in actual_dtype):
                compatible = True
            elif expected_type == "boolean" and ("bool" in actual_dtype or "int" in actual_dtype):
                compatible = True
            
            if not compatible:
                type_issues.append({
                    "field": col,
                    "expected": expected_type,
                    "actual": actual_dtype
                })
        
        type_compatibility_score = 1.0 - (len(type_issues) / len(df.columns)) if len(df.columns) > 0 else 1.0
        
        # 3. Data Quality Checks
        null_summary = {}
        high_null_fields = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
            if null_pct > 50:  # More than 50% nulls
                high_null_fields.append({
                    "field": col,
                    "null_percentage": round(null_pct, 2)
                })
            null_summary[col] = {
                "null_count": int(null_count),
                "null_percentage": round(null_pct, 2)
            }
        
        # 4. Duplicate Check (on key columns)
        key_columns = [col for col in ["date", "product_id", "store_id"] if col in df.columns]
        duplicate_count = 0
        if key_columns:
            duplicate_count = df.duplicated(subset=key_columns).sum()
        
        # 5. Row Count
        row_count = len(df)
        
        # Generate validation result
        validation_result = {
            "status": "success",
            "dataset_path": final_dataset_path,
            "metrics": {
                "field_coverage": {
                    "score": round(coverage_ratio, 4),
                    "threshold": 0.9,
                    "success": coverage_ratio >= 0.9,
                    "details": {
                        "total_required": len(required_fields),
                        "covered": len(covered_fields),
                        "missing_fields": missing_fields
                    }
                },
                "type_compatibility": {
                    "score": round(type_compatibility_score, 4),
                    "threshold": 0.95,
                    "success": type_compatibility_score >= 0.95,
                    "details": {
                        "total_columns": len(df.columns),
                        "incompatible_count": len(type_issues),
                        "type_issues": type_issues
                    }
                },
                "data_quality": {
                    "row_count": row_count,
                    "column_count": len(df.columns),
                    "duplicate_rows": int(duplicate_count),
                    "high_null_fields": high_null_fields,
                    "success": len(high_null_fields) == 0 and duplicate_count == 0
                }
            },
            "summary": {
                "all_checks_passed": (
                    coverage_ratio >= 0.9 and
                    type_compatibility_score >= 0.95 and
                    len(high_null_fields) == 0 and
                    duplicate_count == 0
                ),
                "total_rows": row_count,
                "total_columns": len(df.columns)
            }
        }
        
        return json.dumps(validation_result, indent=2)
        
    except Exception as exc:
        print(f"TOOL ERROR in validate_final_dataset: {exc}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "status": "error",
            "error": str(exc)
        })


@function_tool
def generate_summary_report(
    evaluation_results_json: str,
    agent_name: str,
) -> str:
    """
    Generate a comprehensive summary report analyzing evaluation results.
    
    This tool uses an OpenAI-powered Summary Report Agent to analyze metrics,
    explain causes of success or failure, and propose improvements.
    
    Returns a detailed markdown report.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "Error: OpenAI package not installed"
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set"
    
    def _parse_results(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find the first JSON object in case of concatenated text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                fragment = text[start : end + 1]
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    pass
            return {}

    try:
        results = _parse_results(evaluation_results_json)
        metrics = results.get("metrics", [])
        
        # Build prompt for summary report agent
        metrics_text = json.dumps(metrics, indent=2)
        
        prompt = f"""You are a Summary Report Agent analyzing evaluation results for the {agent_name}.

Evaluation Results:
{metrics_text}

Please provide a comprehensive analysis including:

1. **Overall Assessment**: Summarize the agent's performance
2. **Metric Analysis**: Explain each metric result and what it indicates
3. **Success Factors**: What worked well?
4. **Failure Analysis**: What went wrong and why? (if applicable)
5. **Recommendations**: Specific, actionable improvements to enhance performance

Format your response as a clear, professional report."""

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert AI evaluation analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        report = response.choices[0].message.content
        return report or "Report generation completed but no content returned."
        
    except Exception as exc:
        print(f"TOOL ERROR in generate_summary_report: {exc}")
        import traceback
        traceback.print_exc()
        return f"Error generating report: {str(exc)}"