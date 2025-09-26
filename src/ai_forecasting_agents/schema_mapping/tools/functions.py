import json
import pandas as pd
from typing import Dict, Any, List
from openai import OpenAI
import os

from agents import function_tool 
from src.schemas.models import DemandForecastingRecord, ColumnMapping, MappingResult

# Initialize the OpenAI client for direct use in specific tools
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Data Preparation Tools ---

@function_tool
def load_and_describe_dataset(file_path: str) -> str:
    """
    Loads a CSV dataset, analyzes its basic structure, and generates semantic descriptions
    for each column using an LLM. This provides essential context for mapping.

    :param file_path: The path to the source CSV file.
    :return: A JSON string containing the dataset's metadata and column descriptions.
    """
    print(f"TOOL: Loading and describing top 5 rows of '{file_path}'...")
    try:
        df = pd.read_csv(file_path, nrows=5)
        metadata = {
            "file_path": file_path,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Use LLM to generate semantic descriptions
        system_prompt = """You are a data analysis expert. Analyze the DataFrame head and provide 
        a concise, one-sentence JSON description of each column's likely business meaning in a retail context.
        Return ONLY a JSON object with column names as keys and descriptions as values."""
        
        user_prompt = f"Analyze this DataFrame head and describe each column:\n{df.head().to_string()}"

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        descriptions = json.loads(response.choices[0].message.content)
        metadata["column_descriptions"] = descriptions

        print(f"TOOL: Successfully described '{file_path}'.")
        return json.dumps(metadata, indent=2)

    except Exception as e:
        print(f"TOOL ERROR in load_and_describe_dataset: {e}")
        return json.dumps({"error": str(e)})

# --- Schema Mapping Tool ---

@function_tool
def find_column_mappings(source_metadata_json: str, target_schema_json: str) -> str:
    """
    Uses an LLM to find the best semantic mappings between the described source columns
    and the target schema fields.

    :param source_metadata_json: JSON string of metadata from one or more source datasets,
                                 including column descriptions.
    :param target_schema_json: JSON string representing the target schema with descriptions.
    :return: A JSON string containing a list of proposed mappings with confidence scores and reasoning.
    """
    print("TOOL: Finding column mappings...")
    system_prompt = """You are an expert in retail data schema mapping. Your task is to find the best 
    matches between the provided source columns and the target schema.

    Rules:
    - Match based on semantic meaning and business context, not just names.
    - Only suggest mappings with a confidence score greater than 0.5.
    - Provide a brief, clear reasoning for each mapping decision.
    - Return a JSON object with a single key "mappings" containing an array of mapping objects.
    
    Each mapping object in the array must have:
    - source_column: The name of the column from the source dataset.
    - target_column: The corresponding field name from the target schema.
    - confidence: A float from 0.0 to 1.0.
    - reasoning: A brief explanation for the mapping.
    """

    user_prompt = f"""
    Source dataset metadata:
    {source_metadata_json}

    Target schema fields:
    {target_schema_json}

    Find the best mappings.
    """
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        mappings = json.loads(response.choices[0].message.content)
        print(f"TOOL: Found {len(mappings.get('mappings', []))} potential mappings.")
        return json.dumps(mappings, indent=2)
    except Exception as e:
        print(f"TOOL ERROR in find_column_mappings: {e}")
        return json.dumps({"error": str(e), "mappings": []})

# --- Validation and Merging Tools ---

@function_tool
def merge_and_validate_data(
    source_metadata_json: str, 
    mappings_json: str, 
    output_path: str
) -> str:
    """
    Merges multiple source datasets, applies the schema mappings, validates each resulting record,
    and saves the final, clean dataset to a CSV file.

    :param source_metadata_json: JSON string of metadata for all source datasets.
    :param mappings_json: JSON string of the final column mappings to apply.
    :param output_path: The file path to save the final validated CSV.
    :return: A JSON string summarizing the validation and save operation results.
    """
    print("TOOL: Merging, validating, and saving data...")
    try:
        all_metadata = json.loads(source_metadata_json)
        mappings = json.loads(mappings_json)["mappings"]
        
        # Load all dataframes (top 5 rows only)
        dataframes = {meta['file_path']: pd.read_csv(meta['file_path'], nrows=5) for meta in all_metadata}
        
        # Identify main dataset (most columns, or one with key fields)
        main_df_path = max(dataframes, key=lambda k: dataframes[k].shape[1])
        main_df = dataframes.pop(main_df_path).copy()
        
        print(f"TOOL: Identified '{main_df_path}' as the main dataset.")

        # Heuristic-based merging (can be improved with more sophisticated key detection)
        for path, df in dataframes.items():
            # Simple join key detection
            common_cols = list(set(main_df.columns) & set(df.columns))
            if "date" in str(path): common_cols = ["date"] # Special case for holidays
            
            if common_cols:
                try:
                    main_df = pd.merge(main_df, df, on=common_cols, how="left", suffixes=("", "_dup"))
                    main_df.drop([col for col in main_df.columns if '_dup' in col], axis=1, inplace=True)
                    print(f"TOOL: Merged '{path}' on {common_cols}.")
                except Exception as merge_error:
                    print(f"TOOL WARNING: Could not merge '{path}'. Error: {merge_error}")
            else:
                 print(f"TOOL WARNING: No common columns to merge '{path}'. Skipping merge.")

        # Apply mappings
        final_df = pd.DataFrame()
        for mapping in mappings:
            source_col = mapping["source_column"]
            target_col = mapping["target_column"]
            if source_col in main_df.columns:
                final_df[target_col] = main_df[source_col]

        # Validate records using Pydantic
        validated_records = []
        validation_errors = []
        for index, row in final_df.iterrows():
            try:
                # Pydantic will coerce types and validate
                record = DemandForecastingRecord(**row.to_dict())
                validated_records.append(record.model_dump())
            except Exception as e:
                validation_errors.append(f"Row {index}: {e}")

        if not validated_records:
            return json.dumps({
                "error": "No valid records could be produced after validation.",
                "total_rows_processed": len(final_df),
                "validation_error_samples": validation_errors[:5]
            })

        # Save validated data
        final_validated_df = pd.DataFrame(validated_records)
        final_validated_df.to_csv(output_path, index=False)
        
        summary = {
            "status": "Success",
            "output_path": output_path,
            "total_rows_processed": len(main_df),
            "validated_records": len(validated_records),
            "validation_errors": len(validation_errors),
            "final_shape": final_validated_df.shape
        }
        print("TOOL: Merge and validation complete.")
        return json.dumps(summary, indent=2)

    except Exception as e:
        print(f"TOOL ERROR in merge_and_validate_data: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "Failed", "error": str(e)})

@function_tool
def evaluate_mapping_quality(
    mappings_json: str,
    source_metadata_json: str,
) -> str:
    """
    Simplified mapping quality evaluation focusing on confidence and semantic similarity.
    
    :param mappings_json: JSON string of column mappings with confidence scores.
    :param source_metadata_json: JSON string of metadata from source datasets.
    :return: JSON string with a simplified quality assessment.
    """
    print("TOOL: Evaluating mapping quality (simplified)...")
    try:
        mappings = json.loads(mappings_json).get("mappings", [])
        all_metadata = json.loads(source_metadata_json)
        
        if not mappings:
            return json.dumps({"warning": "No mappings provided to evaluate."})

        # --- SOTA Simplified Metrics ---

        # 1. Confidence Score Analysis
        confidences = [m["confidence"] for m in mappings]
        avg_confidence = sum(confidences) / len(confidences)
        high_conf_count = sum(1 for c in confidences if c >= 0.8)
        medium_conf_count = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_conf_count = sum(1 for c in confidences if c < 0.5)

        # 2. Semantic Similarity (Jaccard Similarity - SOTA yet simple)
        # Create a lookup for column descriptions
        descriptions = {}
        for meta in all_metadata:
            descriptions.update(meta.get("column_descriptions", {}))
        
        total_similarity = 0.0
        for mapping in mappings:
            source_col = mapping["source_column"]
            target_col = mapping["target_column"]
            
            # Use descriptions for better semantic comparison
            source_text = descriptions.get(source_col, source_col)
            target_text = target_col # Target schema has no descriptions here
            
            set1 = set(source_text.lower().replace("_", " ").split())
            set2 = set(target_text.lower().replace("_", " ").split())
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            similarity = intersection / union if union > 0 else 0
            total_similarity += similarity
            
        avg_similarity = total_similarity / len(mappings)

        # 3. Simplified Quality Report
        quality_assessment = {
            "overall_summary": {
                "total_mappings": len(mappings),
                "average_confidence": f"{avg_confidence:.2f}",
                "average_semantic_similarity": f"{avg_similarity:.2f} (Jaccard)"
            },
            "confidence_distribution": {
                "high (>= 0.8)": high_conf_count,
                "medium (0.5 to 0.8)": medium_conf_count,
                "low (< 0.5)": low_conf_count,
            },
            "sota_notes": "Using Jaccard similarity for dependency-free semantic analysis. For higher accuracy, consider embedding-based models."
        }
        
        print(f"TOOL: Simplified quality evaluation complete. Avg Confidence: {avg_confidence:.2f}")
        return json.dumps(quality_assessment, indent=2)

    except Exception as e:
        print(f"TOOL ERROR in evaluate_mapping_quality: {e}")
        return json.dumps({"error": str(e)})