"""
Flask web application for running the end-to-end demand forecasting workflow via web UI.
Workflow stages: Schema Mapping -> Feature Engineering -> Model Training
"""
import asyncio
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Add src to path for schema_mapping / ai_forecasting_agents imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schema_mapping.run_workflow import run_full_workflow
from mapping_system.demand_forecasting import FeatureEngineeringAgent  # TODO: FeatureEngineeringAgent doesn't exist yet; see agents/feature_engineering_agent.py
from ai_forecasting_agents.demand_forecasting.agents.demand_forecasting_agent import training_agent
from agents import Runner, trace
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

# Flask app configuration
template_dir = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(template_dir))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max file size
app.config["UPLOAD_FOLDER"] = PROJECT_ROOT / "uploads"
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# Create upload directory if it doesn't exist
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

# In-memory session storage (in production, use Redis or database)
sessions: Dict[str, Dict] = {}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


# Hardcoded output file names
MERGED_OUTPUT_FILE = "merged_output.csv"
FEATURE_ENGINEERED_FILE = "engineered_features.csv"

def find_merged_csv(output_dir: Path) -> Optional[Path]:
    """Find the merged CSV file in the output directory using hardcoded filename."""
    file_path = output_dir / MERGED_OUTPUT_FILE
    if file_path.exists():
        return file_path
    return None


async def run_feature_engineering_stage(
    input_file: str,
    output_file: str,
    target_column: str = "units_sold",
    api_key: Optional[str] = None,
) -> Dict:
    """Run feature engineering stage."""
    try:
        agent = FeatureEngineeringAgent(openai_api_key=api_key)
        result = await agent.engineer_features(
            input_file=input_file,
            output_file=output_file,
            target_column=target_column,
        )
        # Extract result information
        result_summary = None
        if result:
            if hasattr(result, 'final_output'):
                result_summary = result.final_output
            elif hasattr(result, 'success'):
                # FeatureEngineeringResult object
                result_summary = {
                    "success": result.success,
                    "input_shape": result.input_shape,
                    "output_shape": result.output_shape,
                    "features_created": result.features_created,
                    "processing_time": result.processing_time,
                    "feature_ratio": result.feature_ratio,
                }
            else:
                result_summary = str(result)
        
        return {
            "status": "completed",
            "success": result is not None,
            "result": result_summary,
            "output_file": output_file,
            "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "success": False,
            "result": None,
            "output_file": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


async def run_training_stage(
    input_file: str,
    output_dir: str,
    inference_dir: str,
    id_columns: list[str],
    target_column: str,
    model_types: list[str],
    api_key: Optional[str] = None,
) -> Dict:
    """Run model training stage."""
    try:
        # Create conversation ID for tracing
        conversation_id = str(uuid.uuid4().hex[:16])

        # Create a session instance with conversation ID
        session = SQLAlchemySession.from_url(
            conversation_id,
            url=f"sqlite+aiosqlite:///{PROJECT_ROOT / 'training_database.db'}",
            create_tables=True,
        )

        # Create initial message
        initial_message = f"""
        Train demand forecasting models with these parameters:
        - Input file: {input_file}
        - Output directory: {output_dir}
        - Inference directory: {inference_dir}
        - ID columns: {id_columns}
        - Target column: {target_column}
        - Model types: {', '.join(model_types)}
        
        Start by creating model configurations and training the models.
        """

        # Run the workflow using routing
        with trace("Demand Forecasting Training", group_id=conversation_id):
            result = await Runner.run(
                training_agent,
                input=initial_message,
                max_turns=100,
                session=session,
            )

        # Get conversation history
        items = await session.get_items()

        # Find the best model file in inference directory
        inference_path = Path(inference_dir)
        model_files = list(inference_path.glob("**/*.pkl"))
        best_model_file = model_files[-1] if model_files else None

        return {
            "status": "completed",
            "success": result is not None,
            "result": result.final_output if hasattr(result, 'final_output') else str(result),
            "conversation_id": conversation_id,
            "total_turns": len(items),
            "output_dir": output_dir,
            "inference_dir": inference_dir,
            "model_file": str(best_model_file) if best_model_file else None,
            "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "success": False,
            "result": None,
            "model_file": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


async def run_full_pipeline(
    source_files: list[str],
    row_limit: int,
    output_dir: Optional[str],
    target_column: str = "units_sold",
    id_columns: list[str] = None,
    model_types: list[str] = None,
) -> Dict:
    """Run the complete 3-stage pipeline: Schema Mapping -> Feature Engineering -> Training."""
    if id_columns is None:
        id_columns = ["transaction_date", "product_id", "store_id"]
    if model_types is None:
        model_types = ["xgboost", "lightgbm", "catboost"]

    resolved_output = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    resolved_output.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")

    pipeline_result = {
        "schema_mapping": {"status": "pending"},
        "feature_engineering": {"status": "pending"},
        "training": {"status": "pending"},
    }

    # Stage 1: Schema Mapping
    try:
        pipeline_result["schema_mapping"]["status"] = "running"
        schema_result = await run_full_workflow(
            source_files=source_files,
            row_limit=row_limit,
            output_dir=str(resolved_output),
        )
        pipeline_result["schema_mapping"] = {
            "status": "completed",
            "success": schema_result.get("status") == "success",
            "result": schema_result,
            "output_dir": str(resolved_output),
        }

        # Wait a bit for file system to sync
        import time
        time.sleep(1)
        
        # Find merged CSV file using hardcoded filename
        merged_csv = find_merged_csv(resolved_output)
        
        if not merged_csv:
            # Wait a bit more and try again
            time.sleep(2)
            merged_csv = find_merged_csv(resolved_output)
        
        if not merged_csv:
            expected_path = resolved_output / MERGED_OUTPUT_FILE
            # List all files in the output directory for debugging
            all_files = []
            if resolved_output.exists():
                all_files = [str(f.relative_to(resolved_output)) for f in resolved_output.rglob("*") if f.is_file()]
            
            pipeline_result["schema_mapping"]["error"] = f"Could not find merged CSV output file at expected path: {expected_path}"
            pipeline_result["schema_mapping"]["status"] = "error"
            pipeline_result["schema_mapping"]["success"] = False
            pipeline_result["schema_mapping"]["debug_info"] = {
                "expected_path": str(expected_path),
                "output_dir": str(resolved_output),
                "output_dir_exists": resolved_output.exists(),
                "all_files_in_output_dir": all_files,
                "schema_result": schema_result
            }
            return pipeline_result

        merged_csv_path = str(merged_csv)

    except Exception as e:
        import traceback
        pipeline_result["schema_mapping"] = {
            "status": "error",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        return pipeline_result

    # Stage 2: Feature Engineering
    try:
        pipeline_result["feature_engineering"]["status"] = "running"
        feature_output_file = str(resolved_output / FEATURE_ENGINEERED_FILE)
        feature_result = await run_feature_engineering_stage(
            input_file=merged_csv_path,
            output_file=feature_output_file,
            target_column=target_column,
            api_key=api_key,
        )
        pipeline_result["feature_engineering"] = feature_result

        if not feature_result["success"]:
            return pipeline_result

        feature_input_file = feature_result["output_file"]

    except Exception as e:
        import traceback
        pipeline_result["feature_engineering"] = {
            "status": "error",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        return pipeline_result

    # Stage 3: Training
    try:
        pipeline_result["training"]["status"] = "running"
        training_output_dir = str(resolved_output / "training_results")
        training_inference_dir = str(resolved_output / "training_results" / "inference")
        os.makedirs(training_output_dir, exist_ok=True)
        os.makedirs(training_inference_dir, exist_ok=True)

        training_result = await run_training_stage(
            input_file=feature_input_file,
            output_dir=training_output_dir,
            inference_dir=training_inference_dir,
            id_columns=id_columns,
            target_column=target_column,
            model_types=model_types,
            api_key=api_key,
        )
        pipeline_result["training"] = training_result

    except Exception as e:
        import traceback
        pipeline_result["training"] = {
            "status": "error",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    return pipeline_result


def run_workflow_async(
    session_id: str,
    source_files: list[str],
    row_limit: int,
    output_dir: Optional[str],
) -> None:
    """Run complete 3-stage workflow in background thread."""
    sessions[session_id]["status"] = "running"
    sessions[session_id]["start_time"] = datetime.now(timezone.utc).isoformat()

    try:
        # Run the complete async pipeline
        result = asyncio.run(
            run_full_pipeline(
                source_files=source_files,
                row_limit=row_limit,
                output_dir=output_dir,
            )
        )

        # Determine overall status
        all_completed = all(
            stage.get("status") == "completed" and stage.get("success", False)
            for stage in [result.get("schema_mapping", {}), result.get("feature_engineering", {}), result.get("training", {})]
        )
        any_error = any(
            stage.get("status") == "error" for stage in [result.get("schema_mapping", {}), result.get("feature_engineering", {}), result.get("training", {})]
        )

        if all_completed:
            sessions[session_id]["status"] = "completed"
        elif any_error:
            sessions[session_id]["status"] = "error"
        else:
            sessions[session_id]["status"] = "partial"  # Some stages completed, some pending

        sessions[session_id]["result"] = result
        sessions[session_id]["end_time"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)
        sessions[session_id]["end_time"] = datetime.now(timezone.utc).isoformat()
        import traceback

        sessions[session_id]["traceback"] = traceback.format_exc()


@app.route("/")
def index():
    """Display the file upload form."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    """Handle file uploads and start workflow."""
    if "files" not in request.files:
        return render_template("index.html", error="No files selected"), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return render_template("index.html", error="No files selected"), 400

    # Get configuration
    row_limit = int(request.form.get("row_limit", 10))
    output_dir = request.form.get("output_dir", "").strip() or None

    # Validate row limit
    if row_limit < 1:
        row_limit = 10

    # Save uploaded files
    uploaded_paths = []
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = app.config["UPLOAD_FOLDER"] / filename
            file.save(filepath)
            uploaded_paths.append(str(filepath))
        else:
            return (
                render_template("index.html", error=f"Invalid file: {file.filename}"),
                400,
            )

    if not uploaded_paths:
        return render_template("index.html", error="No valid CSV files uploaded"), 400

    # Generate session ID
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Initialize session
    sessions[session_id] = {
        "status": "pending",
        "source_files": uploaded_paths,
        "row_limit": row_limit,
        "output_dir": output_dir or str(PROJECT_ROOT / "output"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Start workflow in background thread
    thread = threading.Thread(
        target=run_workflow_async,
        args=(session_id, uploaded_paths, row_limit, output_dir),
        daemon=True,
    )
    thread.start()

    return redirect(url_for("status", session_id=session_id))


@app.route("/status/<session_id>")
def status(session_id: str):
    """Display workflow status and results."""
    if session_id not in sessions:
        return render_template("status.html", error="Session not found"), 404

    session_data = sessions[session_id]
    return render_template("status.html", session_id=session_id, session=session_data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002)

