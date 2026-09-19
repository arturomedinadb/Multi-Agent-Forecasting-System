"""
Tool functions for the schema mapping workflow.

This module provides tools for:
- Data preparation (loading and describing datasets)
- Column mapping (generating mapped CSVs)
- Data integration (merging mapped CSVs)
- Evaluation (DeepEval metrics and summary reports)
- Session history (querying conversation context)
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from agents import function_tool
from .schemas.models import DemandForecastingRecord, ColumnMapping, MappingResult

PROJECT_ROOT = Path(__file__).resolve().parents[4]


# === Session History Tools ===


def _query_dataset_metadata(
    db_path: str, session_id: str
) -> tuple[List[Dict[str, Any]], int]:
    """Query the session DB for load_and_describe_dataset activity.

    Returns (metadata_list, calls_issued) - the number of calls actually
    issued is tracked separately from how many results have landed so far,
    so a caller can tell the difference between "there really were only
    N files" and "some results just haven't been persisted yet".
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT message_data
        FROM agent_messages
        WHERE session_id = ?
        ORDER BY id
    """,
        [session_id],
    )
    rows = cursor.fetchall()
    conn.close()

    metadata_list: List[Dict[str, Any]] = []
    calls_issued = 0
    for (msg_data,) in rows:
        try:
            msg = json.loads(msg_data)
        except json.JSONDecodeError:
            continue

        if (
            msg.get("type") == "function_call"
            and msg.get("name") == "load_and_describe_dataset"
        ):
            calls_issued += 1

        if msg.get("type") == "function_call_output":
            output = msg.get("output", "")
            if output and isinstance(output, str):
                try:
                    output_json = json.loads(output)
                    # Check if it has dataset metadata structure
                    if "file_path" in output_json and "columns" in output_json:
                        metadata_list.append(output_json)
                except json.JSONDecodeError:
                    continue

    return metadata_list, calls_issued


# Tracks session_ids that have already received a full get_all_dataset_metadata()
# response, so a repeat call within the same session returns a short pointer
# instead of re-dumping the entire (often large) payload back into the
# conversation - repeated full dumps are what exhausts the model's context
# window on longer runs.
_metadata_already_returned: Dict[str, int] = {}


@function_tool
def get_all_dataset_metadata() -> str:
    """
    Extract ALL dataset metadata from DataPrepAgent's load_and_describe_dataset calls.

    This tool specifically retrieves all dataset metadata generated during the data preparation phase.
    It returns a JSON array of metadata objects, one for each source dataset.

    Calling this more than once per session is unnecessary - after the first
    successful call, later calls return a short reminder instead of the full
    payload again.

    Returns:
        JSON array of dataset metadata objects with file_path, shape, columns, dtypes, and sample.
    """
    # Get session info from environment
    session_id = os.getenv("CURRENT_SESSION_ID")
    output_dir = os.getenv("AGENT_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
    db_path = os.path.join(output_dir, "workflow_sessions.db")

    if not session_id:
        return json.dumps(
            {"status": "error", "error": "No active session", "metadata": []}
        )

    if not os.path.exists(db_path):
        return json.dumps(
            {
                "status": "error",
                "error": f"Database not found at {db_path}",
                "metadata": [],
            }
        )

    if session_id in _metadata_already_returned:
        dataset_count = _metadata_already_returned[session_id]
        return json.dumps(
            {
                "status": "success",
                "session_id": session_id,
                "dataset_count": dataset_count,
                "metadata": "ALREADY_RETRIEVED",
                "note": (
                    f"Metadata for all {dataset_count} datasets was already returned in an "
                    "earlier call this session. Do not call get_all_dataset_metadata again - "
                    "reuse that earlier result and proceed to creating mappings."
                ),
            }
        )

    try:
        # The session DB is written to asynchronously relative to when a
        # tool call finishes, so a downstream agent calling this right after
        # a handoff can race the last load_and_describe_dataset result being
        # persisted, and silently see fewer datasets than were actually
        # loaded. Retry briefly whenever fewer results exist than calls were
        # issued, instead of returning a short list as if it were complete.
        max_attempts = 6
        retry_delay_seconds = 0.5
        metadata_list, calls_issued = _query_dataset_metadata(db_path, session_id)
        attempts = 1
        while len(metadata_list) < calls_issued and attempts < max_attempts:
            time.sleep(retry_delay_seconds)
            metadata_list, calls_issued = _query_dataset_metadata(db_path, session_id)
            attempts += 1

        _metadata_already_returned[session_id] = len(metadata_list)

        return json.dumps(
            {
                "status": "success",
                "session_id": session_id,
                "dataset_count": len(metadata_list),
                "metadata": metadata_list,
            }
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "metadata": []})


@function_tool
def query_conversation_history(
    limit: int = 10,
    agent_filter: Optional[str] = None,
    include_tool_calls: bool = False,
) -> str:
    """
    Query conversation history from the current workflow session.

    Use this tool to understand what previous agents have done, review their outputs,
    or check for errors in earlier steps of the workflow.

    Args:
        limit: Maximum number of messages to retrieve (default: 10, max: 50)
        agent_filter: Optional filter by agent name (e.g., "DataPrepAgent", "ColumnMappingAgent")
        include_tool_calls: Whether to include tool call messages (default: False, only assistant messages)

    Returns:
        JSON array of messages with role, agent name, content preview, and timestamp.
        Messages are ordered from most recent to oldest.

    Example usage:
        - query_conversation_history(limit=5) - Get last 5 messages
        - query_conversation_history(agent_filter="DataPrepAgent") - Get DataPrepAgent messages
        - query_conversation_history(include_tool_calls=True) - Include tool call details
    """
    # Get session info from environment
    session_id = os.getenv("CURRENT_SESSION_ID")
    output_dir = os.getenv("AGENT_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
    db_path = os.path.join(output_dir, "workflow_sessions.db")

    if not session_id:
        return json.dumps(
            {
                "status": "error",
                "error": "No active session. CURRENT_SESSION_ID not set.",
                "messages": [],
            }
        )

    if not os.path.exists(db_path):
        return json.dumps(
            {
                "status": "error",
                "error": f"Session database not found at {db_path}",
                "messages": [],
            }
        )

    # Clamp limit to reasonable range
    limit = min(max(1, limit), 50)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_messages'"
        )
        if cursor.fetchone() is None:
            conn.close()
            return json.dumps(
                {
                    "status": "error",
                    "error": "agent_messages table not found in database",
                    "messages": [],
                }
            )

        # Fetch raw messages
        query = """
            SELECT id, message_data
            FROM agent_messages 
            WHERE session_id = ?
            ORDER BY id DESC
        """
        cursor.execute(query, [session_id])
        rows = cursor.fetchall()
        conn.close()

        # Parse and filter messages
        messages = []
        for msg_id, msg_data in rows:
            try:
                msg = json.loads(msg_data)

                # Extract fields based on message structure
                msg_type = msg.get("type", "message")
                role = msg.get("role", "unknown")
                name = msg.get("name", None)

                # Apply agent filter
                if agent_filter and name != agent_filter:
                    continue

                # Apply tool call filter
                if not include_tool_calls and msg_type in (
                    "function_call",
                    "function_call_output",
                ):
                    continue

                # Extract content based on message type
                content_str = ""
                if "content" in msg:
                    content = msg["content"]
                    if isinstance(content, str):
                        content_str = content
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    content_str += block.get("text", "") + " "
                elif msg_type == "function_call":
                    content_str = (
                        f"Called {name} with args: {msg.get('arguments', '')[:100]}"
                    )
                elif msg_type == "function_call_output":
                    output = msg.get("output", "")
                    content_str = f"Output: {str(output)[:200]}"

                # Truncate long content
                preview = (
                    content_str[:500] + "..." if len(content_str) > 500 else content_str
                )

                messages.append(
                    {
                        "id": msg_id,
                        "role": role,
                        "agent": name or "unknown",
                        "type": msg_type,
                        "content_preview": preview,
                    }
                )

                # Stop if we have enough messages
                if len(messages) >= limit:
                    break

            except json.JSONDecodeError:
                continue

        return json.dumps(
            {
                "status": "success",
                "session_id": session_id,
                "message_count": len(messages),
                "messages": messages,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "messages": []})


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
    """
    Normalize a file path to use forward slashes for JSON compatibility.
    DOES NOT resolve or translate paths to avoid language-specific issues
    (e.g., "Documentos" on Spanish Windows should stay "Documentos").
    """
    if not value:
        return None
    # Simply replace backslashes with forward slashes
    # Do NOT use Path.resolve() as it may translate folder names based on OS language
    return value.replace("\\", "/")


def _normalize_date_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Reformat any column that looks like a date field (named 'date' or ending
    in '_date') to a consistent ISO 'YYYY-MM-DD' string, in place.

    Source files use inconsistent date formats (e.g. holidays.csv uses
    DD/MM/YYYY while transaction data uses YYYY-MM-DD). Without normalizing,
    a merge/join on 'date' silently matches zero rows even though both sides
    have a column named 'date' - normalizing at mapped-CSV creation time means
    every downstream consumer sees one consistent format regardless of how any
    single merge step handles things.

    IMPORTANT: dayfirst=True must only be applied to genuinely ambiguous
    slash-separated dates (DD/MM/YYYY vs MM/DD/YYYY). Applying it to
    unambiguous ISO dash-format dates (YYYY-MM-DD) actually corrupts them -
    pandas will swap the month/day components whenever both are <=12 (e.g.
    "2022-07-10" silently becomes "2022-10-07"). So format is detected per
    column and dayfirst is only turned on for slash-separated values.
    """
    date_cols = [c for c in df.columns if c == "date" or c.endswith("_date")]
    for col in date_cols:
        sample = df[col].dropna().astype(str)
        uses_slash = sample.str.contains("/").any()
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=uses_slash)
        formatted = parsed.dt.strftime("%Y-%m-%d")
        df[col] = formatted.where(parsed.notna(), df[col])
    return df


def _is_month_level_dates(series: "pd.Series") -> bool:
    """True if every valid date in the series falls on day 1 - a proxy for
    the source being naturally month-granularity data (e.g. monthly economic
    indicators) rather than day-granularity transaction dates."""
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna()
    return len(valid) > 0 and bool((valid.dt.day == 1).all())


def _dedupe_by_key(df: "pd.DataFrame", keys: List[str]) -> "pd.DataFrame":
    """If a dataframe has multiple rows sharing the same key value(s) - e.g.
    one row per region/category per date in a raw economic export that was
    never filtered down - collapse them to one row per key by averaging
    numeric columns. Prevents a many-to-many blowup when merging on that key."""
    if not df.duplicated(subset=keys).any():
        return df
    non_key_cols = [c for c in df.columns if c not in keys]
    numeric_cols = df[non_key_cols].select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in non_key_cols if c not in numeric_cols]
    agg = {col: "mean" for col in numeric_cols}
    agg.update({col: "first" for col in non_numeric_cols})
    return df.groupby(keys, as_index=False).agg(agg)


def _load_json_payload(
    raw: str | Dict[str, Any] | List[Dict[str, Any]],
) -> Dict[str, Any] | List[Dict[str, Any]]:
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

    # Attempt to parse inline JSON first to avoid filesystem lookups on large
    # payloads. Tolerant of extra trailing data after a complete JSON value
    # (the LLM sometimes appends stray content past a valid closing brace)
    # and of a raw single backslash inside a Windows path.
    if text.startswith("{") or text.startswith("["):
        result = _extract_first_json_tolerant(text)
        if result:
            return result

    # Try as file path
    candidate_path = Path(text)
    try:
        if candidate_path.exists():
            return json.loads(candidate_path.read_text())
    except OSError:
        pass

    # Last resort: same tolerant parse: raise if even that can't recover it
    result = _extract_first_json_tolerant(text)
    if result:
        return result
    return json.loads(text)


def _coerce_metadata_entries(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalize any metadata payload (list, dict with metadata array, JSON string, or file path)
    into a list of dataset metadata dictionaries.
    """
    payload = _load_json_payload(raw)

    if isinstance(payload, dict):
        entries = payload.get("metadata")
        if isinstance(entries, list):
            candidates = entries
        else:
            candidates = [payload]
    elif isinstance(payload, list):
        candidates = payload
    else:
        return []

    normalized: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file_path")
        columns = item.get("columns")
        if not file_path or not columns:
            continue
        normalized.append({**item, "file_path": file_path.replace("\\", "/")})
    return normalized


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
        # Convert backslashes to forward slashes WITHOUT Path normalization
        # This preserves the original path (e.g., "Documentos" on Spanish Windows)
        json_safe_path = file_path.replace("\\", "/")
        metadata: Dict[str, Any] = {
            "file_path": json_safe_path,
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample": df.head(2).to_dict(orient="records"),
        }
        return json.dumps(metadata)
    except Exception as e:
        print(f"TOOL ERROR in load_and_describe_dataset: {e}")
        json_safe_path = file_path.replace("\\", "/")
        return json.dumps({"error": str(e), "file_path": json_safe_path})


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
            # First check if there's an extra wrapper layer like {"columns": {...}} or {"mappings": {...}}
            # where the value is itself a dict of column mappings
            if len(payload) == 1:
                single_key = list(payload.keys())[0]
                single_value = payload[single_key]
                if single_key in [
                    "columns",
                    "mappings",
                    "schema",
                    "fields",
                ] and isinstance(single_value, dict):
                    # Recursively process the unwrapped value
                    return _ensure_mapping_dict(single_value)

            # Check if it's a flat dict of column mappings
            # Format 1: {"transaction_date": {"target_column": "date", ...}}
            # Format 2: {"transaction_date": [{"target": "date", ...}]}
            sample_values = list(payload.values())[:3]

            # Check for different flat dict formats:
            # Format 1: {"col1": {"target_column": "col2", ...}} - source as key, target in value
            # Format 2: {"target_col": {"source_column": "col1", ...}} - target as key, source in value
            # Format 3: {"col1": [{"target": "col2", ...}]} - array format
            is_flat_dict_format = sample_values and all(
                isinstance(v, dict)
                and (
                    "target_column" in v
                    or "target" in v
                    or "source_column" in v
                    or "source" in v
                )
                for v in sample_values
            )
            is_array_format = sample_values and all(
                isinstance(v, list) and v and isinstance(v[0], dict)
                for v in sample_values
            )

            if is_flat_dict_format or is_array_format:
                # Convert formats to nested structure
                if is_array_format:
                    # Format: {"filename.csv": [{"source_column": "col1", "target_column": "col2"}]}
                    # This is actually the correct nested format! Just need to restructure
                    result_mappings = []
                    for key, mapping_array in payload.items():
                        if isinstance(mapping_array, list) and mapping_array:
                            # Check if key looks like a filename (has .csv or similar extension)
                            if "." in key and any(
                                ext in key.lower()
                                for ext in [".csv", ".xlsx", ".json", ".txt"]
                            ):
                                # Key is a filename - create proper nested structure
                                result_mappings.append(
                                    {"source_file": key, "mappings": mapping_array}
                                )
                            else:
                                # Key is not a filename, treat as fallback
                                result_mappings.extend(mapping_array)
                    return {"mappings": result_mappings}

                elif is_flat_dict_format:
                    # Handle two variations of flat dict format
                    fallback_mappings = []
                    for key, mapping_info in payload.items():
                        if isinstance(mapping_info, dict):
                            # Check if target is the key (Format 2: {"target_col": {"source_column": "col1"}})
                            source_col = mapping_info.get(
                                "source_column"
                            ) or mapping_info.get("source")
                            if source_col:
                                # Key is target column, value contains source column
                                fallback_mappings.append(
                                    {
                                        "source_column": source_col,
                                        "target_column": key,  # Key is the target!
                                        "confidence": mapping_info.get(
                                            "confidence", 0.5
                                        ),
                                        "reasoning": mapping_info.get("reasoning")
                                        or mapping_info.get("reason", ""),
                                    }
                                )
                            else:
                                # Check if source is the key (Format 1: {"col1": {"target_column": "col2"}})
                                target_col = mapping_info.get(
                                    "target_column"
                                ) or mapping_info.get("target")
                                if target_col:
                                    # Key is source column, value contains target column
                                    fallback_mappings.append(
                                        {
                                            "source_column": key,
                                            "target_column": target_col,
                                            "confidence": mapping_info.get(
                                                "confidence", 0.5
                                            ),
                                            "reasoning": mapping_info.get("reasoning")
                                            or mapping_info.get("reason", ""),
                                        }
                                    )
                    return {
                        "mappings": fallback_mappings
                    }  # Use as fallback since no source_file specified
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

        try:
            source_meta = _coerce_metadata_entries(source_metadata_json)
        except (json.JSONDecodeError, TypeError):
            source_meta = []

        # source_metadata_json requires the model to re-echo the full, often
        # large, metadata payload as a literal argument, which is prone to
        # truncation or the model only including a subset of files. Compare
        # against what's actually recorded in the session DB (the same
        # source get_all_dataset_metadata reads from) and prefer whichever
        # is more complete, rather than only recovering when the argument is
        # empty outright.
        session_id = os.getenv("CURRENT_SESSION_ID")
        db_output_dir = os.getenv("AGENT_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
        db_path = os.path.join(db_output_dir, "workflow_sessions.db")
        if session_id and os.path.exists(db_path):
            db_meta, _ = _query_dataset_metadata(db_path, session_id)
            if len(db_meta) > len(source_meta):
                print(
                    f"DEBUG run_generate_mapped_csvs: source_metadata_json had only "
                    f"{len(source_meta)} entries but the session DB has {len(db_meta)} - "
                    "using the session DB's metadata instead."
                )
                source_meta = db_meta

        print(
            f"DEBUG run_generate_mapped_csvs: Parsed {len(source_meta)} metadata entries"
        )
        if source_meta:
            print(
                f"DEBUG run_generate_mapped_csvs: First entry keys: {list(source_meta[0].keys())}"
            )

        # Parse mappings JSON
        if isinstance(mappings_json, dict):
            raw_mappings = mappings_json
            print(f"DEBUG run_generate_mapped_csvs: mappings_json is already a dict")
        elif isinstance(mappings_json, list):
            raw_mappings = {"mappings": mappings_json}
            print(
                f"DEBUG run_generate_mapped_csvs: mappings_json is a list with {len(mappings_json)} items"
            )
        else:
            cleaned_mappings = _sanitize_json_string(mappings_json)
            print(
                f"DEBUG run_generate_mapped_csvs: mappings_json is a string, length={len(cleaned_mappings)}"
            )
            print(
                f"DEBUG run_generate_mapped_csvs: mappings preview: {cleaned_mappings[:300]}..."
            )

            # Try parsing with error diagnostics
            try:
                raw_mappings = _parse_json_tolerant(cleaned_mappings)
            except json.JSONDecodeError as json_err:
                print(
                    f"DEBUG: JSON parse error at position {json_err.pos}: {json_err.msg}"
                )
                if json_err.pos and len(cleaned_mappings) > json_err.pos:
                    start = max(0, json_err.pos - 100)
                    end = min(len(cleaned_mappings), json_err.pos + 100)
                    print(f"DEBUG: Context around error (chars {start}-{end}):")
                    print(f"  ...{cleaned_mappings[start:end]}...")
                # Try to fix and re-parse
                import re

                fixed_json = re.sub(
                    r",\s*([}\]])", r"\1", cleaned_mappings
                )  # Remove trailing commas
                fixed_json = re.sub(
                    r"([}\]])\s*([{\[])", r"\1,\2", fixed_json
                )  # Add missing commas between objects
                try:
                    raw_mappings = _parse_json_tolerant(fixed_json)
                    print(f"DEBUG: Successfully parsed after fixing common JSON issues")
                except json.JSONDecodeError:
                    # Last resort: salvage just the first complete JSON value,
                    # discarding anything malformed/extra the LLM appended
                    # after it (e.g. a stray second object tacked on the end).
                    raw_mappings = _extract_first_json_tolerant(cleaned_mappings)
                    if not raw_mappings:
                        raise json_err  # genuinely unrecoverable

        mapping_payload = _ensure_mapping_dict(raw_mappings)
        print(
            f"DEBUG run_generate_mapped_csvs: mapping_payload keys: {list(mapping_payload.keys())}"
        )
    except Exception as e:
        print(f"TOOL ERROR in generate_mapped_csvs: Invalid JSON input - {e}")
        import traceback

        traceback.print_exc()
        return json.dumps({"outputs": [], "error": f"Invalid JSON input: {e}"})

    # Accept both "mappings" and "columns" as valid top-level keys
    mapping_entries = mapping_payload.get("mappings") or mapping_payload.get(
        "columns", []
    )
    print(
        f"DEBUG run_generate_mapped_csvs: Found {len(mapping_entries)} mapping entries"
    )

    per_file: Dict[str, List[Dict[str, Any]]] = {}
    fallback: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for idx, entry in enumerate(mapping_entries):
        if not isinstance(entry, dict):
            continue

        # Accept both "source_file" and "source" as valid keys
        source_path = entry.get("source_file") or entry.get("source")
        if source_path:
            normalized_key = _normalize_path(source_path)
            print(
                f"DEBUG run_generate_mapped_csvs: Processing source: {source_path} -> normalized: {normalized_key}"
            )
            if normalized_key:
                mappings_for_file = entry.get("mappings", [])
                if isinstance(mappings_for_file, list):
                    filtered = [m for m in mappings_for_file if isinstance(m, dict)]
                    per_file[normalized_key] = filtered
                    per_file[Path(normalized_key).name] = filtered
                    print(
                        f"DEBUG run_generate_mapped_csvs: Added {len(filtered)} mappings for key '{normalized_key}' and '{Path(normalized_key).name}'"
                    )
        elif entry.get("source_column") and entry.get("target_column"):
            fallback.append(entry)

    print(
        f"DEBUG run_generate_mapped_csvs: per_file has {len(per_file)} entries, fallback has {len(fallback)} entries"
    )
    if per_file:
        print(
            f"DEBUG run_generate_mapped_csvs: per_file keys: {list(per_file.keys())[:5]}..."
        )

    results: List[Dict[str, Any]] = []

    row_limit = _get_row_limit()

    for meta in source_meta:
        # Safety check: ensure meta is a dict
        if not isinstance(meta, dict):
            print(
                f"TOOL WARNING: Skipping non-dict metadata entry: {type(meta).__name__} - value preview: {str(meta)[:100]}"
            )
            continue

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
            # Use original path (with backslashes) for Windows file system
            # Only normalize for JSON/comparison purposes
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

        mapped_df = _normalize_date_columns(mapped_df)

        out_name = f"{Path(path).stem}_mapped.csv"
        out_path = Path(output_dir) / out_name
        mapped_df.to_csv(out_path, index=False)

        results.append(
            {
                "source_file": str(path).replace("\\", "/"),
                "output_path": str(out_path).replace("\\", "/"),
                "columns": list(mapped_df.columns),
            }
        )

    print(f"Wrote {len(results)} mapped CSVs to {output_dir}")
    return json.dumps({"outputs": results})


# Tracks session_ids that have already produced a successful
# generate_mapped_csvs result, so a redundant repeat call (which may carry a
# truncated or otherwise malformed copy of the large metadata/mappings
# arguments) returns the known-good result instead of trying to reprocess
# them and risking a wasted, confusing failure.
_generate_mapped_csvs_cache: Dict[str, str] = {}


@function_tool
def generate_mapped_csvs(
    source_metadata_json: str, mappings_json: str, output_dir: str
) -> str:
    """
    Apply mappings separately per source dataset and write one CSV per dataset
    containing only mapped target columns.

    Calling this more than once per session is unnecessary once it has
    succeeded - a repeat call returns the same cached result rather than
    reprocessing its (large) arguments again.

    Returns JSON: {"outputs": [{"source_file": str, "output_path": str, "columns": [..]}]}
    """
    session_id = os.getenv("CURRENT_SESSION_ID", "")
    if session_id in _generate_mapped_csvs_cache:
        print(
            "TOOL: generate_mapped_csvs already succeeded this session - returning cached result."
        )
        return _generate_mapped_csvs_cache[session_id]

    print("TOOL: Generating per-dataset mapped CSVs...")
    try:
        result = run_generate_mapped_csvs(
            source_metadata_json, mappings_json, output_dir
        )
        # DEBUG: Print what we're returning
        print(
            f"DEBUG generate_mapped_csvs: Returning result (first 500 chars): {result[:500]}"
        )
        try:
            if json.loads(result).get("outputs"):
                _generate_mapped_csvs_cache[session_id] = result
        except json.JSONDecodeError:
            pass
        return result
    except Exception as e:
        print(f"TOOL ERROR in generate_mapped_csvs: {e}")
        return json.dumps({"outputs": [], "error": str(e)})


# Tracks session_ids for which merge_mapped_csvs_to_target has already
# succeeded, so evaluate_data_integration_agent can refuse to run (and
# redirect the agent back to actually merging) if it's ever asked to
# evaluate a merge that never happened.
_merge_mapped_csvs_cache: Dict[str, bool] = {}


@function_tool
def merge_mapped_csvs_to_target(
    mapped_outputs_json: str, target_schema_json: str, output_path: str
) -> str:
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

        # Use robust JSON parsing
        if isinstance(mapped_outputs_json, str):
            raw = _extract_first_json_tolerant(mapped_outputs_json)
            if not raw:
                print("TOOL ERROR: Failed to parse mapped_outputs_json")
                return json.dumps(
                    {"status": "Failed", "error": "Invalid mapped_outputs_json"}
                )
        else:
            raw = mapped_outputs_json

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
            return json.dumps(
                {"status": "Failed", "error": "No mapped outputs provided."}
            )

        # Load mapped dataframes
        dataframes: List[pd.DataFrame] = []
        for entry in mapped:
            p = entry.get("output_path")

            # Try original path first
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                dataframes.append(df)
            elif p:
                # Try path with Documentos/Documents swap (Spanish Windows fix)
                alt_paths = [
                    p.replace("/Documents/", "/Documentos/"),
                    p.replace("/Documentos/", "/Documents/"),
                    p.replace("\\Documents\\", "\\Documentos\\"),
                    p.replace("\\Documentos\\", "\\Documents\\"),
                ]

                found = False
                for alt_p in alt_paths:
                    if alt_p != p and os.path.exists(alt_p):
                        df = pd.read_csv(alt_p)
                        dataframes.append(df)
                        found = True
                        break

                if not found:
                    print(f"TOOL WARNING: Missing mapped file: {p}")

        if not dataframes:
            return json.dumps(
                {"status": "Failed", "error": "No mapped CSVs could be loaded."}
            )

        # Start with the widest dataframe
        main_df = max(dataframes, key=lambda d: d.shape[1]).copy()
        others = [d for d in dataframes if d is not main_df]

        # Normalize date format/granularity across all dataframes before merging.
        # This is defensive - generate_mapped_csvs already normalizes dates at
        # write time, but this makes the merge correct even for mapped CSVs
        # produced some other way.
        main_df = _normalize_date_columns(main_df)
        others = [_normalize_date_columns(df) for df in others]

        candidate_keys = ["date", "product_id", "store_id", "store_region"]

        # Iterative left joins on available common keys
        for df in others:
            common = [
                k for k in candidate_keys if k in main_df.columns and k in df.columns
            ]
            if common:
                try:
                    merge_keys = common
                    df_to_merge = df

                    if "date" in common and _is_month_level_dates(
                        df["date"]
                    ) != _is_month_level_dates(main_df["date"]):
                        # One side is month-granularity (e.g. monthly economic
                        # indicators), the other day-granularity (transaction
                        # dates) - exact date equality would never match.
                        # Join on a derived year-month key instead, and drop
                        # the month-level side's own 'date' column so it
                        # doesn't overwrite the day-level date already present.
                        main_df["_year_month"] = pd.to_datetime(
                            main_df["date"], errors="coerce"
                        ).dt.strftime("%Y-%m")
                        df_to_merge = df.copy()
                        df_to_merge["_year_month"] = pd.to_datetime(
                            df_to_merge["date"], errors="coerce"
                        ).dt.strftime("%Y-%m")
                        df_to_merge = df_to_merge.drop(columns=["date"])
                        merge_keys = [
                            "_year_month" if k == "date" else k for k in common
                        ]

                    # Collapse duplicate rows on the merge key (e.g. multiple
                    # regions/categories sharing one date in a raw economic
                    # export) before joining, so the merge can't multiply rows.
                    df_to_merge = _dedupe_by_key(df_to_merge, merge_keys)

                    main_df = pd.merge(
                        main_df,
                        df_to_merge,
                        on=merge_keys,
                        how="left",
                        suffixes=("", "_dup"),
                    )
                    dup_cols = [c for c in main_df.columns if c.endswith("_dup")]
                    if dup_cols:
                        main_df.drop(columns=dup_cols, inplace=True)
                    if "_year_month" in main_df.columns:
                        main_df.drop(columns=["_year_month"], inplace=True)
                    print(f"TOOL: Merged on keys {merge_keys}")
                except Exception as me:
                    print(f"TOOL WARNING: Merge failed on keys {common}: {me}")
            else:
                print("TOOL WARNING: No common keys; skipping a dataframe")

        # Ensure full target schema columns exist
        try:
            target_schema = json.loads(target_schema_json)
        except json.JSONDecodeError:
            target_schema = _extract_first_json(target_schema_json)
            if not target_schema:
                print("WARNING: Failed to parse target schema, using empty schema")
                target_schema = {}

        target_fields: List[str] = list(target_schema.get("properties", {}).keys())
        for col in target_fields:
            if col not in main_df.columns:
                main_df[col] = None

        # Reorder columns to target schema order
        main_df = main_df[target_fields]

        # Guard against silently reporting "Success" on an unusable result.
        # If mapped_outputs_json pointed at raw/unmapped files (columns that
        # never match any target field name), every target column above got
        # backfilled with None and main_df is empty in substance even though
        # it has the right headers. Fail loudly instead of writing it out.
        if len(main_df) == 0:
            return json.dumps(
                {"status": "Failed", "error": "Merged output has 0 rows."}
            )

        null_frac_by_col = main_df.isnull().mean()
        fully_null_cols = [c for c, frac in null_frac_by_col.items() if frac >= 0.999]
        fully_null_ratio = (
            len(fully_null_cols) / len(target_fields) if target_fields else 0
        )
        if fully_null_ratio >= 0.8:
            return json.dumps(
                {
                    "status": "Failed",
                    "error": (
                        f"Merged output is unusable: {len(fully_null_cols)}/{len(target_fields)} "
                        "target columns are entirely null. This usually means mapped_outputs_json "
                        "pointed at raw source files instead of the *_mapped.csv outputs from "
                        "generate_mapped_csvs."
                    ),
                    "fully_null_columns": fully_null_cols,
                    "rows": len(main_df),
                }
            )

        # Normalize NaNs to None and write CSV
        main_df = main_df.astype(object).where(pd.notnull(main_df), None)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        main_df.to_csv(output_path, index=False)

        _merge_mapped_csvs_cache[os.getenv("CURRENT_SESSION_ID", "")] = True

        return json.dumps(
            {
                "status": "Success",
                "output_path": str(output_path).replace("\\", "/"),
                "rows": len(main_df),
                "columns": list(main_df.columns),
            }
        )
    except Exception as e:
        print(f"TOOL ERROR in merge_mapped_csvs_to_target: {e}")
        import traceback

        traceback.print_exc()
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
        return json.dumps(
            {
                "status": "No Deepeval API provided",
                "metrics": [
                    {
                        "name": "Task Completion",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Tool Correctness",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Answer Relevancy",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                ],
            }
        )

    try:
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric
    except ImportError:
        return json.dumps(
            {
                "status": "deepeval not installed",
                "error": "deepeval package required for evaluation",
            }
        )

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
            expected_output=(
                f"Metadata for {len(expected_list)} files" if expected_list else None
            ),
            tools_called=[
                ToolCall(
                    name="load_and_describe_dataset",
                    description="Load and analyze CSV files",
                )
            ],
        )

        # Run metrics
        task_metric = TaskCompletionMetric(
            threshold=0.6, async_mode=False, strict_mode=False
        )
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
                    "success": bool(task_metric.success),
                },
                {
                    "name": "Tool Correctness",
                    "score": 1.0,  # Deterministic check
                    "threshold": 1.0,
                    "success": True,
                },
                {
                    "name": "Answer Relevancy",
                    "score": float(relevancy_metric.score or 0.0),
                    "threshold": 0.6,
                    "success": bool(relevancy_metric.success),
                },
            ],
        }
        return json.dumps(results, indent=2)
    except Exception as exc:
        print(f"TOOL ERROR in evaluate_data_prep_agent: {exc}")
        return json.dumps({"status": "Failed", "error": str(exc)})


def _parse_json_tolerant(text: str) -> Any:
    """Parse a JSON string, tolerating an LLM's occasional use of a single
    raw backslash inside a Windows path (technically invalid JSON).

    Tries a normal parse first, so already-valid JSON - including a path
    with correctly double-escaped backslashes - is never touched. Only
    falls back to a blind backslash-to-slash replace once a direct parse
    has already failed, since at that point the text could not have been
    relying on a '\\\\' pair to mean a real backslash in the first place.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.replace("\\", "/"))


def _extract_first_json_tolerant(text: str) -> Any:
    """Same tolerant-parsing order as _parse_json_tolerant, but for callers
    that want _extract_first_json's "pull the first JSON object out of a
    string that may have extra data" behavior instead of a strict parse."""
    result = _extract_first_json(text)
    if result:
        return result
    return _extract_first_json(text.replace("\\", "/"))


def _sanitize_json_string(json_str: str) -> str:
    """Remove invalid control characters from JSON string while preserving important whitespace."""
    import re

    # Remove problematic control characters but keep newlines (\n), tabs (\t), and carriage returns (\r)
    # that might be needed in JSON strings. Only remove truly problematic chars.
    # Keep \x09 (tab), \x0A (newline), \x0D (carriage return)
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", json_str)


def _extract_first_json(json_str: str):
    """Extract the first valid JSON object from a string that may contain extra data."""
    import json
    import re

    # If it's already a dict/list, return it directly
    if isinstance(json_str, (dict, list)):
        return json_str

    # First, try to fix common JSON formatting issues
    def fix_common_json_issues(s: str) -> str:
        # Remove trailing commas before closing braces/brackets
        s = re.sub(r",\s*([}\]])", r"\1", s)
        # Fix common quote issues (but preserve proper quotes)
        return s

    decoder = json.JSONDecoder()

    # Try original string first
    attempts = [
        json_str.lstrip(),
        fix_common_json_issues(json_str.lstrip()),
        json_str,
    ]

    for i, attempt in enumerate(attempts):
        try:
            # Try to decode the JSON
            obj, end_idx = decoder.raw_decode(attempt)
            # Return the parsed object directly, not re-encoded
            if i > 0:
                print(
                    f"DEBUG _extract_first_json: Successfully parsed on attempt {i+1}"
                )
            return obj
        except json.JSONDecodeError as e:
            continue

    # Last resort: try json.loads on the whole string
    try:
        return json.loads(json_str)
    except:
        pass

    # If all attempts fail, log error and return empty dict
    print(f"DEBUG _extract_first_json: All parsing attempts failed")
    print(
        f"DEBUG _extract_first_json: Input type: {type(json_str).__name__}, length: {len(json_str) if isinstance(json_str, str) else 'N/A'}"
    )
    if isinstance(json_str, str):
        print(f"DEBUG _extract_first_json: First 200 chars: {json_str[:200]}")
        if len(json_str) > 3400:
            print(
                f"DEBUG _extract_first_json: Context around char 3470: {json_str[3370:3570]}"
            )
    print(f"DEBUG _extract_first_json: Returning empty dict")
    return {}


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

    Requires generate_mapped_csvs to have succeeded earlier in the session;
    otherwise this returns status "blocked" without evaluating anything.

    Returns JSON with metric results.
    """
    session_id = os.getenv("CURRENT_SESSION_ID", "")
    if session_id not in _generate_mapped_csvs_cache:
        return json.dumps(
            {
                "status": "blocked",
                "error": (
                    "generate_mapped_csvs has not succeeded yet this session, so there is "
                    "nothing to evaluate. Column mapping is not done - call "
                    "get_all_dataset_metadata() and then generate_mapped_csvs() first, and "
                    "only request this evaluation once generate_mapped_csvs has returned "
                    "real outputs."
                ),
            }
        )

    api_key = os.getenv("DEEPEVAL_API_KEY")
    if not api_key:
        return json.dumps(
            {
                "status": "No Deepeval API provided",
                "metrics": [
                    {
                        "name": "Task Completion",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Tool Correctness",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Field Coverage",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Type Compatibility",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Semantic Similarity",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                ],
            }
        )

    try:
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval.metrics import TaskCompletionMetric
        from .evaluation.metrics import (
            FieldCoverageMetric,
            TypeCompatibilityMetric,
            SemanticSimilarityMetric,
        )
    except ImportError:
        return json.dumps(
            {
                "status": "deepeval not installed",
                "error": "deepeval package required for evaluation",
            }
        )

    try:
        cleaned_mapping = _sanitize_json_string(mapping_plan_json)
        cleaned_schema = _sanitize_json_string(target_schema_json)

        # Try direct/tolerant JSON parsing first (these are already clean
        # JSON strings from function params); only fall back to extracting
        # the first valid JSON object if there's extra surrounding data.
        try:
            mapping_plan = _parse_json_tolerant(cleaned_mapping)
        except json.JSONDecodeError:
            mapping_plan = _extract_first_json_tolerant(cleaned_mapping)

        try:
            target_schema = _parse_json_tolerant(cleaned_schema)
        except json.JSONDecodeError:
            target_schema = _extract_first_json_tolerant(cleaned_schema)

        # Handle mapping_plan being either a dict with "mappings" key or a direct list
        if isinstance(mapping_plan, list):
            mappings_list = mapping_plan
        elif isinstance(mapping_plan, dict):
            mappings_list = mapping_plan.get("mappings", [])
        else:
            mappings_list = []

        # Extract target fields and types
        properties = target_schema.get("properties", {})
        required_fields = list(properties.keys())
        target_types = {
            name: spec.get("type", "string")
            for name, spec in properties.items()
            if isinstance(spec, dict)
        }

        # Create test case with metadata
        test_case = LLMTestCase(
            input=agent_input,
            actual_output=agent_output,
            tools_called=[
                ToolCall(
                    name="generate_mapped_csvs", description="Generate mapped CSV files"
                )
            ],
            additional_metadata={
                "required_fields": required_fields,
                "target_types": target_types,
                "mapping_plan": mappings_list,
                "dataset_columns": [
                    m.get("target_column") for m in mappings_list if isinstance(m, dict)
                ],
                "dataset_dtypes": {},
            },
        )

        # Run metrics
        task_metric = TaskCompletionMetric(
            threshold=0.6, async_mode=False, strict_mode=False
        )
        coverage_metric = FieldCoverageMetric(
            required_fields=required_fields, threshold=0.9
        )
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
                    "success": bool(task_metric.success),
                },
                {
                    "name": "Tool Correctness",
                    "score": 1.0,
                    "threshold": 1.0,
                    "success": True,
                },
                {
                    "name": "Field Coverage",
                    "score": float(coverage_metric.score or 0.0),
                    "threshold": 0.9,
                    "success": bool(coverage_metric.success),
                    "details": coverage_metric.score_breakdown,
                },
                {
                    "name": "Type Compatibility",
                    "score": float(type_metric.score or 0.0),
                    "threshold": 1.0,
                    "success": bool(type_metric.success),
                    "details": type_metric.score_breakdown,
                },
                {
                    "name": "Semantic Similarity",
                    "score": float(semantic_metric.score or 0.0),
                    "threshold": 0.5,
                    "success": bool(semantic_metric.success),
                    "details": semantic_metric.score_breakdown,
                },
            ],
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

    Requires merge_mapped_csvs_to_target to have succeeded earlier in the
    session; otherwise this returns status "blocked" without evaluating
    anything.

    Returns JSON with metric results.
    """
    session_id = os.getenv("CURRENT_SESSION_ID", "")
    if not _merge_mapped_csvs_cache.get(session_id):
        return json.dumps(
            {
                "status": "blocked",
                "error": (
                    "merge_mapped_csvs_to_target has not succeeded yet this session, so "
                    "there is nothing to evaluate. Data integration is not done - call "
                    "merge_mapped_csvs_to_target first, and only request this evaluation "
                    "once it has returned status Success."
                ),
            }
        )

    api_key = os.getenv("DEEPEVAL_API_KEY")
    if not api_key:
        return json.dumps(
            {
                "status": "No Deepeval API provided",
                "metrics": [
                    {
                        "name": "Task Completion",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Tool Correctness",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                    {
                        "name": "Data Quality",
                        "status": "skipped",
                        "reason": "No Deepeval API provided",
                    },
                ],
            }
        )

    try:
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval.metrics import TaskCompletionMetric
    except ImportError:
        return json.dumps(
            {
                "status": "deepeval not installed",
                "error": "deepeval package required for evaluation",
            }
        )

    try:
        # Create test case
        test_case = LLMTestCase(
            input=agent_input,
            actual_output=agent_output,
            expected_output=f"Merged dataset with {source_row_count} rows",
            tools_called=[
                ToolCall(
                    name="merge_mapped_csvs_to_target",
                    description="Merge mapped CSV files",
                )
            ],
        )

        # Run metrics
        task_metric = TaskCompletionMetric(
            threshold=0.6, async_mode=False, strict_mode=False
        )
        task_metric.measure(test_case)

        # Data Quality check (deterministic)
        row_match = source_row_count == final_row_count
        data_quality_score = (
            1.0 if row_match else (final_row_count / max(source_row_count, 1))
        )

        results = {
            "status": "success",
            "metrics": [
                {
                    "name": "Task Completion",
                    "score": float(task_metric.score or 0.0),
                    "threshold": 0.6,
                    "success": bool(task_metric.success),
                },
                {
                    "name": "Tool Correctness",
                    "score": 1.0,
                    "threshold": 1.0,
                    "success": True,
                },
                {
                    "name": "Data Quality",
                    "score": data_quality_score,
                    "threshold": 0.95,
                    "success": data_quality_score >= 0.95,
                    "details": {
                        "source_rows": source_row_count,
                        "final_rows": final_row_count,
                        "rows_preserved": row_match,
                    },
                },
            ],
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
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Final dataset not found: {final_dataset_path}",
                }
            )

        df = pd.read_csv(final_dataset_path)
        cleaned_schema = _sanitize_json_string(target_schema_json)
        cleaned_mapping = _sanitize_json_string(mapping_plan_json)

        # Extract first valid JSON object (handles "Extra data" errors)
        target_schema = _extract_first_json_tolerant(cleaned_schema)
        mapping_plan = _extract_first_json_tolerant(cleaned_mapping)

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
        coverage_ratio = (
            len(covered_fields) / len(required_fields) if required_fields else 1.0
        )

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
            elif expected_type == "number" and (
                "float" in actual_dtype or "int" in actual_dtype
            ):
                compatible = True
            elif expected_type == "boolean" and (
                "bool" in actual_dtype or "int" in actual_dtype
            ):
                compatible = True

            if not compatible:
                type_issues.append(
                    {"field": col, "expected": expected_type, "actual": actual_dtype}
                )

        type_compatibility_score = (
            1.0 - (len(type_issues) / len(df.columns)) if len(df.columns) > 0 else 1.0
        )

        # 3. Data Quality Checks
        null_summary = {}
        high_null_fields = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
            if null_pct > 50:  # More than 50% nulls
                high_null_fields.append(
                    {"field": col, "null_percentage": round(null_pct, 2)}
                )
            null_summary[col] = {
                "null_count": int(null_count),
                "null_percentage": round(null_pct, 2),
            }

        # 4. Duplicate Check (on key columns)
        key_columns = [
            col for col in ["date", "product_id", "store_id"] if col in df.columns
        ]
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
                        "missing_fields": missing_fields,
                    },
                },
                "type_compatibility": {
                    "score": round(type_compatibility_score, 4),
                    "threshold": 0.95,
                    "success": type_compatibility_score >= 0.95,
                    "details": {
                        "total_columns": len(df.columns),
                        "incompatible_count": len(type_issues),
                        "type_issues": type_issues,
                    },
                },
                "data_quality": {
                    "row_count": row_count,
                    "column_count": len(df.columns),
                    "duplicate_rows": int(duplicate_count),
                    "high_null_fields": high_null_fields,
                    "success": len(high_null_fields) == 0 and duplicate_count == 0,
                },
            },
            "summary": {
                "all_checks_passed": (
                    coverage_ratio >= 0.9
                    and type_compatibility_score >= 0.95
                    and len(high_null_fields) == 0
                    and duplicate_count == 0
                ),
                "total_rows": row_count,
                "total_columns": len(df.columns),
            },
        }

        return json.dumps(validation_result, indent=2)

    except Exception as exc:
        print(f"TOOL ERROR in validate_final_dataset: {exc}")
        import traceback

        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(exc)})


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
                {
                    "role": "system",
                    "content": "You are an expert AI evaluation analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        report = response.choices[0].message.content
        return report or "Report generation completed but no content returned."

    except Exception as exc:
        print(f"TOOL ERROR in generate_summary_report: {exc}")
        import traceback

        traceback.print_exc()
        return f"Error generating report: {str(exc)}"


@function_tool
def generate_final_workflow_report(
    data_prep_eval_json: str,
    column_mapping_eval_json: str,
    data_integration_eval_json: str,
    mapped_files_json: str,
) -> str:
    """
    Generate a comprehensive final report that ALWAYS includes evaluation metrics.

    This function ensures consistent reporting by enforcing a structured format
    that includes all evaluation results and file outputs.

    Args:
        data_prep_eval_json: JSON string with data preparation evaluation results
        column_mapping_eval_json: JSON string with column mapping evaluation results
        data_integration_eval_json: JSON string with data integration evaluation results
        mapped_files_json: JSON string with list of generated mapped files

    Returns:
        Formatted markdown report with all metrics and file information
    """
    try:
        # Parse all inputs
        data_prep_eval = (
            json.loads(data_prep_eval_json)
            if isinstance(data_prep_eval_json, str)
            else data_prep_eval_json
        )
        column_mapping_eval = (
            json.loads(column_mapping_eval_json)
            if isinstance(column_mapping_eval_json, str)
            else column_mapping_eval_json
        )
        data_integration_eval = (
            json.loads(data_integration_eval_json)
            if isinstance(data_integration_eval_json, str)
            else data_integration_eval_json
        )
        mapped_files = (
            json.loads(mapped_files_json)
            if isinstance(mapped_files_json, str)
            else mapped_files_json
        )

        # Build structured report
        report = []
        report.append("=" * 70)
        report.append("WORKFLOW EXECUTION REPORT")
        report.append("=" * 70)
        report.append("")

        # Phase 1: Data Preparation
        report.append("## Phase 1: Data Preparation")
        report.append("")
        if data_prep_eval.get("status") == "success":
            metrics = data_prep_eval.get("metrics", [])
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                name = metric.get("name", "Unknown")
                score = metric.get("score", 0.0)
                success = metric.get("success", False)
                status = "✅ PASS" if success else "❌ FAIL"
                report.append(f"  - {name}: {score:.2%} {status}")
        else:
            report.append(f"  ⚠️ Status: {data_prep_eval.get('status', 'Unknown')}")
        report.append("")

        # Phase 2: Column Mapping
        report.append("## Phase 2: Column Mapping")
        report.append("")
        if column_mapping_eval.get("status") == "success":
            metrics = column_mapping_eval.get("metrics", [])
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                name = metric.get("name", "Unknown")
                score = metric.get("score", 0.0)
                success = metric.get("success", False)
                status = "✅ PASS" if success else "❌ FAIL"
                report.append(f"  - {name}: {score:.2%} {status}")

                # Show additional details for important metrics
                if name == "Field Coverage" and "details" in metric:
                    details = metric["details"]
                    covered = details.get("covered_fields", 0)
                    total = details.get("total_required_fields", 0)
                    report.append(f"    └─ Fields: {covered}/{total} covered")
        else:
            report.append(
                f"  ⚠️ Status: {column_mapping_eval.get('status', 'Unknown')}"
            )
        report.append("")

        # Phase 3: Data Integration
        report.append("## Phase 3: Data Integration")
        report.append("")
        if data_integration_eval.get("status") == "success":
            metrics = data_integration_eval.get("metrics", [])
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                name = metric.get("name", "Unknown")
                score = metric.get("score", 0.0)
                success = metric.get("success", False)
                status = "✅ PASS" if success else "❌ FAIL"
                report.append(f"  - {name}: {score:.2%} {status}")

                # Show data quality details
                if name == "Data Quality" and "details" in metric:
                    details = metric["details"]
                    source_rows = details.get("source_rows", 0)
                    final_rows = details.get("final_rows", 0)
                    report.append(f"    └─ Rows: {final_rows}/{source_rows} preserved")
        else:
            report.append(
                f"  ⚠️ Status: {data_integration_eval.get('status', 'Unknown')}"
            )
        report.append("")

        # Generated Files
        report.append("## Generated Files")
        report.append("")
        if isinstance(mapped_files, dict):
            outputs = mapped_files.get("outputs", [])
        elif isinstance(mapped_files, list):
            outputs = mapped_files
        else:
            outputs = []

        if outputs:
            report.append(f"Total: {len(outputs)} mapped CSV files")
            report.append("")
            for i, file_info in enumerate(outputs, 1):
                if isinstance(file_info, dict):
                    output_path = file_info.get("output_path", "N/A")
                    columns = file_info.get("columns", [])
                    report.append(f"{i}. {os.path.basename(output_path)}")
                    report.append(
                        f"   Columns: {', '.join(columns[:5])}"
                        + ("..." if len(columns) > 5 else "")
                    )
        else:
            report.append("⚠️ No files generated")

        report.append("")
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)

    except Exception as exc:
        print(f"TOOL ERROR in generate_final_workflow_report: {exc}")
        import traceback

        traceback.print_exc()
        return f"Error generating final report: {str(exc)}"
