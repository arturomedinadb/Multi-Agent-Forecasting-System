"""
Evaluation functions for demand forecasting models.
"""

import json
import os
import pandas as pd
import numpy as np
import uuid
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import catboost as cb

from agents import function_tool
from ..schemas.forecasting_models import TrainedModel, TrainingResultsOutput, EvaluationResults, ModelPerformance

def calculate_overall_score(rmse: float, r2: float, mae: float, mape: float) -> float:
    """Calculate overall performance score (0-100)."""
    # Normalize scores (higher is better for r2, lower is better for errors)
    # Weighted combination where better performance = higher score
    if rmse == 0:
        return 100.0
    
    # Normalize RMSE (lower is better, so invert)
    rmse_score = max(0, 100 * (1 - min(rmse / 100, 1)))  # Assuming max reasonable RMSE is 100
    
    # R² is already between 0-1, scale to 0-100
    r2_score = max(0, r2 * 50)  # Scale R² to 0-50
    
    # Normalize MAE (lower is better)
    mae_score = max(0, 100 * (1 - min(mae / 50, 1))) if mae > 0 else 100  # Assuming max reasonable MAE is 50
    
    # Normalize MAPE (lower is better)
    mape_score = max(0, 100 * (1 - min(mape / 100, 1))) if mape > 0 else 100  # Assuming max reasonable MAPE is 100%
    
    # Weighted average
    overall = (0.3 * rmse_score + 0.3 * r2_score + 0.2 * mae_score + 0.2 * mape_score)
    
    return min(100, max(0, overall))

async def evaluate_model_performance(
    model_info: TrainedModel,
    data_dir: str,
    iteration: int
) -> Dict[str, Any]:
    """Evaluate a single model's performance on validation set."""
    try:
        # Load validation data (not test data for iterative evaluation)
        X_val = pd.read_csv(f"{data_dir}/X_val.csv")
        y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.ravel()

        model_name = model_info.model_name
        model_type = model_info.model_type
        model_path = model_info.model_path
        
        # Load model from the saved path
        if not os.path.exists(model_path):
            return {
                "success": False,
                "error": f"Model file not found: {model_path}",
                "model_name": model_name,
                "iteration": iteration,
                "message": f"Error loading model {model_name}: file not found"
            }
        
        model = joblib.load(model_path)
        
        # Extract timestamp from filename for tracking
        model_timestamp = model_info.model_uuid
        
        # Make predictions on validation set
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        
        # Calculate overall score
        overall_score = calculate_overall_score(rmse, r2, mae, mape)
        
        # Save feature importance and SHAP plots (best-effort)
        try:
            model_dir = os.path.dirname(model_path)
            feature_names = list(X_val.columns)

            # Feature importance (tree-based models)
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = getattr(model, "feature_importances_")
            else:
                # CatBoost specific
                try:
                    from catboost import Pool  # type: ignore
                    if hasattr(model, "get_feature_importance"):
                        pool = Pool(X_val, y_val, feature_names=feature_names)
                        importances = model.get_feature_importance(pool)
                except Exception as e:
                    print(f"{datetime.now()} - Error getting feature importance for {model_name}: {str(e)}")

            if importances is not None and len(importances) == len(feature_names):
                try:
                    fi_path = f"{model_dir}/{model_name}_feature_importance.png"
                    # Plot top 20
                    importances_arr = np.asarray(importances)
                    top_idx = np.argsort(importances_arr)[-20:]
                    top_features = [feature_names[i] for i in top_idx]
                    top_values = importances_arr[top_idx]
                    order = np.argsort(top_values)
                    plt.figure(figsize=(15, 8))
                    plt.barh(np.array(top_features)[order], top_values[order])
                    plt.title(f"Feature Importance - {model_name}")
                    plt.xlabel("Importance")
                    plt.tight_layout()
                    plt.savefig(fi_path)
                    plt.close()
                except Exception as e:
                    print(f"{datetime.now()} - Error saving feature importance plot for {model_name}: {str(e)}")

        except Exception as e:
            print(f"{datetime.now()} - Error saving feature importance plot for {model_name}: {str(e)}")

        print(f"{datetime.now()} - Evaluation completed for {model_name} on validation set")
        print(f"Evaluation results: RMSE: {rmse}, R2: {r2}, MAE: {mae}, MAPE: {mape}, Overall Score: {overall_score}")

        return {
            "success": True,
            "model_name": model_name,
            "model_type": model_type,
            "model_uuid": model_timestamp,
            "hyperparameters": model_info.hyperparameters,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "overall_score": overall_score,
            "iteration": iteration,
            "model_path": model_path,
            "message": f"Evaluation completed for {model_name} on validation set"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name,
            "iteration": iteration,
            "message": f"Error evaluating {model_name}: {str(e)}"
        }

@function_tool(strict_mode=False)  # TrainingResultsOutput nests TrainedModel.hyperparameters: Dict[str, Any]
async def evaluate_all_models(
    training_results: TrainingResultsOutput,
    iteration: int
) -> EvaluationResults:
    """
    Evaluate all models from training results.
    
    Args:
        training_results: TrainingResultsOutput Pydantic model containing:
            - models_directory: str - directory containing model files
            - data_dir: str - directory containing data splits
            - models: List[TrainedModel] - list of trained models
        iteration: Current iteration number
        
    Returns:
        EvaluationResults Pydantic model containing:
            - success: bool
            - model_evaluations: List[ModelPerformance] - list of model evaluations
            - best_model: str - name of best performing model
            - best_score: float - score of best model
            - models_directory: str - directory containing all models
    """
    try:
        print(f"\n{datetime.now()} - Evaluating models at iteration {iteration}")
        
        # Extract data directory from training results (Pydantic model)
        data_dir = training_results.data_dir
        models_directory = training_results.models_directory
        
        print(f"Data directory: {data_dir}")
        print(f"Models directory: {models_directory}")
        
        model_evaluations = []
        best_model = None
        best_score = float('inf')
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            return EvaluationResults(
                success=False,
                model_evaluations=[],
                best_model="none",
                best_score=float('inf'),
                total_models=0,
                iteration=iteration,
                models_directory=models_directory
            )
        
        # Iterate through models from the Pydantic model
        for model_info in training_results.models:
            if model_info.success:
                # Get model path from the Pydantic model
                model_path = model_info.model_path
                if not model_path:
                    print(f"\n{datetime.now()} - Warning: No model path for {model_info.model_name}")
                    continue
                
                print(f"\n{datetime.now()} - Evaluating model: {model_info.model_name} at {model_path}")
                
                evaluation = await evaluate_model_performance(
                    model_info=model_info,
                    data_dir=data_dir,
                    iteration=iteration
                )
                
                if evaluation["success"]:
                    model_evaluations.append(evaluation)
                    
                    # Track best model
                    if evaluation["rmse"] < best_score:
                        best_score = evaluation["rmse"]
                        best_model = evaluation["model_name"]
                else:
                    print(f"\n{datetime.now()} - Evaluation failed for {model_info.model_name}: {evaluation.get('error', 'Unknown error')}")
        
        print(f"\n{datetime.now()} - Successfully evaluated {len(model_evaluations)} models")
        
        # Convert dict evaluations to ModelPerformance objects
        model_performances = []
        for eval_dict in model_evaluations:
            try:
                model_perf = ModelPerformance(
                    model_name=eval_dict["model_name"],
                    model_type=eval_dict.get("model_type", "unknown"),
                    model_path=eval_dict.get("model_path", ""),
                    model_uuid=eval_dict.get("model_uuid", ""),
                    hyperparameters=eval_dict.get("hyperparameters", {}),
                    rmse=eval_dict["rmse"],
                    r2=eval_dict["r2"],
                    mae=eval_dict["mae"],
                    mape=eval_dict["mape"],
                    overall_score=eval_dict["overall_score"]
                )
                model_performances.append(model_perf)
            except Exception as e:
                print(f"Error converting evaluation to ModelPerformance: {str(e)}")
        
        # Return EvaluationResults Pydantic object
        return EvaluationResults(
            success=True,
            model_evaluations=model_performances,
            best_model=best_model or "none",
            best_score=best_score,
            total_models=len(model_performances),
            iteration=iteration,
            models_directory=models_directory
        )
        
    except Exception as e:
        print(f"Error in evaluate_all_models: {str(e)}")
        # Return a minimal EvaluationResults on error
        return EvaluationResults(
            success=False,
            model_evaluations=[],
            best_model="none",
            best_score=float('inf'),
            total_models=0,
            iteration=iteration,
            models_directory=""
        )

@function_tool(strict_mode=False)  # EvaluationResults nests ModelPerformance.hyperparameters: Dict[str, Any]
async def check_convergence(
    evaluation_results: EvaluationResults,
    previous_performance_json: Optional[str] = None,
    max_iterations: int = 5,
    convergence_threshold: float = 0.01
) -> Dict[str, Any]:
    """Check if the training has converged.

    previous_performance_json: optional JSON object string of the previous iteration's
    performance (e.g. '{"rmse": 12.3}'), used to measure improvement.
    """
    previous_performance = json.loads(previous_performance_json) if previous_performance_json else None
    try:
        iteration = evaluation_results.iteration
        print(f"\n{datetime.now()} - Checking convergence at iteration {iteration}")
        # Check if max iterations reached
        if iteration >= max_iterations:
            print(f"\n{datetime.now()} - Maximum iterations ({max_iterations}) reached")
            return {
                "success": True,
                "convergence_achieved": False,
                "reason": "max_iterations_reached",
                "iteration": iteration,
                "should_continue": False,
                "message": f"Maximum iterations ({max_iterations}) reached"
            }
        
        # Get current performance
        best_model = evaluation_results.best_model
        best_score = evaluation_results.best_score
        
        # Check performance improvement
        if previous_performance and best_score < float('inf'):
            previous_score = previous_performance.get("rmse", float('inf'))
            if previous_score < float('inf'):
                improvement = (previous_score - best_score) / previous_score
                if abs(improvement) < convergence_threshold:
                    print(f"\n{datetime.now()} - Model has converged: performance_stable")
                    return {
                        "success": True,
                        "convergence_achieved": True,
                        "reason": "performance_stable",
                        "iteration": iteration,
                        "improvement": improvement,
                        "should_continue": False,
                        "message": f"Performance converged (improvement: {improvement:.4f})"
                    }
        
        # Check if performance is excellent
        if best_score < 0.1:  # Very low RMSE
            print(f"\n{datetime.now()} - Model has converged: excellent_performance")
            return {
                "success": True,
                "convergence_achieved": True,
                "reason": "excellent_performance",
                "iteration": iteration,
                "should_continue": False,
                "message": f"Excellent performance achieved (RMSE: {best_score:.4f})"
            }
        
        print(f"\n{datetime.now()} - Model has not converged")
        # Continue training
        return {
            "success": True,
            "convergence_achieved": False,
            "reason": "continue_training",
            "iteration": iteration,
            "should_continue": True,
            "message": "Training should continue"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "iteration": iteration,
            "message": f"Error checking convergence: {str(e)}"
        }

@function_tool(strict_mode=False)  # EvaluationResults nests ModelPerformance.hyperparameters: Dict[str, Any]
async def save_best_model_for_inference(
    evaluation_results: EvaluationResults,
    inference_dir: str = "output/training_results/inference"
) -> Dict[str, Any]:
    """Save the best performing model for inference by retraining on combined train+val data."""
    try:
        # Find the best model from evaluation results
        model_evaluations = evaluation_results.model_evaluations
        models_directory = evaluation_results.models_directory
        
        if not model_evaluations:
            return {
                "success": False,
                "error": "No model evaluations found",
                "message": "Cannot save best model - no evaluations available"
            }
        
        # Find the best model (lowest RMSE)
        best_model_eval = min(model_evaluations, key=lambda x: x.rmse)
        
        # Get the best model details
        best_model_name = best_model_eval.model_name
        best_model_type = best_model_eval.model_type
        best_model_timestamp = best_model_eval.model_uuid
        hyperparameters = best_model_eval.hyperparameters
        
        # Infer data_dir from models_directory
        # models_directory format: output/training_results/models/{session_uuid}
        # data_dir format: output/training_results/data
        # Need to go up two levels: models/{session_uuid} -> models -> training_results
        training_results_dir = os.path.dirname(os.path.dirname(models_directory))
        data_dir = os.path.join(training_results_dir, "data")
        
        if not os.path.exists(data_dir):
            return {
                "success": False,
                "error": f"Data directory not found: {data_dir}",
                "message": f"Cannot retrain model - data directory not found: {data_dir}"
            }
        
        print(f"{datetime.now()} - Retraining best model for inference on combined train+val data")
        print(f"Data directory: {data_dir}")
        print(f"Model type: {best_model_type}")
        print(f"Hyperparameters: {hyperparameters}")
        
        # Load training and validation data
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_val = pd.read_csv(f"{data_dir}/X_val.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
        y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.ravel()
        
        # Combine training and validation sets
        X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_train_val = np.concatenate([y_train, y_val])
        
        print(f"Combined dataset size: {len(X_train_val)} (train: {len(X_train)}, val: {len(X_val)})")
        
        # Get model class based on model type
        model_type_str = best_model_type.value if hasattr(best_model_type, 'value') else str(best_model_type)
        
        # Map model types to model classes
        SUPPORTED_MODELS = {
            "xgboost": xgb.XGBRegressor,
            "random_forest": RandomForestRegressor,
            "lightgbm": lgb.LGBMRegressor,
            "catboost": cb.CatBoostRegressor
        }
        
        model_class = SUPPORTED_MODELS.get(model_type_str)
        if not model_class:
            return {
                "success": False,
                "error": f"Unsupported model type: {model_type_str}",
                "message": f"Model type {model_type_str} is not supported for retraining"
            }
        
        # Create new model with saved hyperparameters
        print(f"{datetime.now()} - Creating model with hyperparameters: {hyperparameters}")
        inference_model = model_class(**hyperparameters)
        
        # Train on combined train+val data
        print(f"{datetime.now()} - Training model on combined train+val data...")
        start_time = datetime.now()
        inference_model.fit(X_train_val, y_train_val)
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"{datetime.now()} - Model training completed in {training_time:.2f} seconds")
        
        # Create inference directory
        inference_dir = f"{inference_dir}/{best_model_timestamp}"
        os.makedirs(inference_dir, exist_ok=True)
        
        # Save retrained model
        inference_model_path = f"{inference_dir}/best_{best_model_name}.pkl"
        joblib.dump(inference_model, inference_model_path)
        print(f"{datetime.now()} - Retrained model saved to: {inference_model_path}")
        
        # Generate and save feature importance graph
        feature_importance_path = None
        try:
            feature_names = list(X_train_val.columns)
            importances = None
            
            # Feature importance (tree-based models)
            if hasattr(inference_model, "feature_importances_"):
                importances = getattr(inference_model, "feature_importances_")
            else:
                # CatBoost specific
                try:
                    from catboost import Pool  # type: ignore
                    if hasattr(inference_model, "get_feature_importance"):
                        pool = Pool(X_train_val, y_train_val, feature_names=feature_names)
                        importances = inference_model.get_feature_importance(pool)
                except Exception as e:
                    print(f"{datetime.now()} - Error getting feature importance for inference model: {str(e)}")
            
            if importances is not None and len(importances) == len(feature_names):
                try:
                    feature_importance_path = f"{inference_dir}/best_{best_model_name}_feature_importance.png"
                    # Plot top 20
                    importances_arr = np.asarray(importances)
                    top_idx = np.argsort(importances_arr)[-20:]
                    top_features = [feature_names[i] for i in top_idx]
                    top_values = importances_arr[top_idx]
                    order = np.argsort(top_values)
                    plt.figure(figsize=(15, 8))
                    plt.barh(np.array(top_features)[order], top_values[order])
                    plt.title(f"Feature Importance - {best_model_name} (Inference Model)")
                    plt.xlabel("Importance")
                    plt.tight_layout()
                    plt.savefig(feature_importance_path)
                    plt.close()
                    print(f"{datetime.now()} - Feature importance graph saved to: {feature_importance_path}")
                except Exception as e:
                    print(f"{datetime.now()} - Error saving feature importance plot for inference model: {str(e)}")
        except Exception as e:
            print(f"{datetime.now()} - Error generating feature importance for inference model: {str(e)}")
        
        # Calculate metrics on combined data for reference
        y_pred_train_val = inference_model.predict(X_train_val)
        train_val_rmse = np.sqrt(mean_squared_error(y_train_val, y_pred_train_val))
        train_val_r2 = r2_score(y_train_val, y_pred_train_val)
        
        # Make predictions on test set and save to CSV
        test_predictions_path = None
        test_rmse = None
        test_r2 = None
        test_mae = None
        test_mape = None
        
        try:
            # Load test set
            X_test = pd.read_csv(f"{data_dir}/X_test.csv")
            y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
            
            print(f"{datetime.now()} - Making predictions on test set (size: {len(X_test)})")
            
            # Load ID column names if available
            id_columns = []
            try:
                with open(f"{data_dir}/id_columns.json", "r") as f:
                    id_columns = json.load(f)
                print(f"{datetime.now()} - Found ID columns: {id_columns}")
            except Exception as e:
                print(f"{datetime.now()} - Could not load ID columns, will try to identify them: {str(e)}")
                # Try to identify ID columns by common patterns
                id_patterns = ['_id', 'id', 'date', 'transaction_date']
                id_columns = [col for col in X_test.columns if any(pattern in col.lower() for pattern in id_patterns)]
                if id_columns:
                    print(f"{datetime.now()} - Identified potential ID columns: {id_columns}")
            
            # Make predictions
            y_pred_test = inference_model.predict(X_test)
            
            # Calculate test metrics
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            
            print(f"{datetime.now()} - Test set metrics - RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}%")
            
            # Load original ID column values (before encoding) from saved CSV files
            id_data = {}
            for col in id_columns:
                try:
                    # Try to load original ID values from saved CSV
                    id_file = f"{data_dir}/id_test_{col}.csv"
                    if os.path.exists(id_file):
                        id_df = pd.read_csv(id_file)
                        id_data[col] = id_df[col].values
                        print(f"{datetime.now()} - Loaded original values for ID column: {col}")
                    else:
                        # Fallback: use encoded values from X_test if original not found
                        if col in X_test.columns:
                            id_data[col] = X_test[col].values
                            print(f"{datetime.now()} - Using encoded values for ID column: {col} (original not found)")
                except Exception as e:
                    print(f"{datetime.now()} - Error loading ID column {col}: {str(e)}")
                    # Fallback: use encoded values from X_test
                    if col in X_test.columns:
                        id_data[col] = X_test[col].values
            
            # Create predictions DataFrame
            predictions_dict = {
                'actual': y_test,
                'predicted': y_pred_test
            }
            
            # Add ID columns at the beginning
            predictions_dict = {**id_data, **predictions_dict}
            predictions_df = pd.DataFrame(predictions_dict)
            
            # Save predictions to CSV
            test_predictions_path = f"{inference_dir}/test_predictions.csv"
            predictions_df.to_csv(test_predictions_path, index=False)
            print(f"{datetime.now()} - Test predictions saved to: {test_predictions_path}")
            
        except Exception as e:
            print(f"{datetime.now()} - Error making predictions on test set: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Save model metadata
        model_metadata = {
            "model_name": best_model_name,
            "model_type": best_model_type.value if hasattr(best_model_type, 'value') else str(best_model_type),
            "model_uuid": best_model_timestamp,
            "model_filename": f"best_{best_model_name}.pkl",
            "hyperparameters": hyperparameters,
            "training_data": "combined_train_val",
            "train_size": len(X_train),
            "val_size": len(X_val),
            "combined_size": len(X_train_val),
            "training_time_seconds": training_time,
            # Validation metrics from original evaluation (for reference)
            "validation_rmse": best_model_eval.rmse,
            "validation_r2": best_model_eval.r2,
            "validation_mae": best_model_eval.mae,
            "validation_mape": best_model_eval.mape,
            "validation_overall_score": best_model_eval.overall_score,
            # Metrics on combined data (for reference)
            "combined_data_rmse": train_val_rmse,
            "combined_data_r2": train_val_r2,
            # Test set metrics
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_mape": test_mape,
            "test_predictions_path": test_predictions_path,
            "iteration": evaluation_results.iteration,
            "saved_at": datetime.now().isoformat(),
            "inference_path": inference_model_path,
            "feature_importance_path": feature_importance_path,
            "models_directory": models_directory,
            "retrained": True,
            "note": "Model retrained on combined train+val data for inference"
        }
        
        # Save metadata
        with open(f"{inference_dir}/model_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"{datetime.now()} - Best model retrained and saved for inference: {inference_model_path}")
        
        return {
            "success": True,
            "best_model_name": best_model_name,
            "best_model_type": best_model_type,
            "best_model_uuid": best_model_timestamp,
            "best_model_filename": f"best_{best_model_name}.pkl",
            "inference_path": inference_model_path,
            "feature_importance_path": feature_importance_path,
            "test_predictions_path": test_predictions_path,
            "metadata_path": f"{inference_dir}/model_metadata.json",
            "models_directory": models_directory,
            "training_data_size": len(X_train_val),
            "training_time_seconds": training_time,
            "retrained": True,
            "performance": {
                "validation_rmse": best_model_eval.rmse,
                "validation_r2": best_model_eval.r2,
                "validation_mae": best_model_eval.mae,
                "validation_mape": best_model_eval.mape,
                "validation_overall_score": best_model_eval.overall_score,
                "combined_data_rmse": train_val_rmse,
                "combined_data_r2": train_val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "test_mae": test_mae,
                "test_mape": test_mape
            },
            "message": f"Best model retrained on combined train+val data and saved for inference: {inference_model_path}"
        }
        
    except Exception as e:
        print(f"{datetime.now()} - Error saving best model for inference: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error saving best model for inference: {str(e)}"
        }

@function_tool
async def analyze_data_structure_for_feedback(
    input_file: str
) -> Dict[str, Any]:
    """
    Analyze data structure to provide insights for model improvement feedback.
    Analyzes the training data to understand feature types, distributions, and patterns.
    """
    print(f"\n{datetime.now()} - Analyzing data structure for feedback generation")
    
    df = pd.read_csv(input_file)

    # Perform analysis
    analysis = {
        "input_file_path": input_file,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime64"]).columns.tolist(),
        "boolean_columns": df.select_dtypes(include=["bool"]).columns.tolist(),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "sample_data": df.head(1).to_dict("records"),
    }
    print(f"{datetime.now()} - Completed data structure analysis: \n{analysis}")
    
    return analysis

@function_tool(strict_mode=False)  # EvaluationResults nests ModelPerformance.hyperparameters: Dict[str, Any]
async def categorize_feedback(
    evaluation_results: EvaluationResults,
    agent_feedback: str
) -> Dict[str, Any]:
    """
    Categorize agent-generated feedback into specific improvement actions.
    
    Args:
        evaluation_results: EvaluationResults from evaluate_all_models
        agent_feedback: Free-form feedback text from the evaluation agent
    
    Returns:
        Dictionary with categorized feedback including:
        - action_type: "feature_engineering", "hyperparameter_tuning", or "ensemble"
        - parameters: Specific parameters for the action
        - target_model: Model to improve
        - reasoning: Why this action was selected
    """
    try:
        print(f"\n{datetime.now()} - Categorizing feedback: {agent_feedback}")
        
        model_evaluations = evaluation_results.model_evaluations
        # Get best model
        best_model_perf = min(model_evaluations, key=lambda x: x.rmse)
        if not best_model_perf:
            return {
                "success": False,
                "error": "No models to improve",
                "message": "No models available for improvement"
            }
        
        target_model = best_model_perf.model_type.value if hasattr(best_model_perf.model_type, 'value') else str(best_model_perf.model_type)
        best_score = best_model_perf.overall_score
        iteration = evaluation_results.iteration
        
        # Parse agent_feedback to determine action type
        feedback_lower = agent_feedback.lower()
        
        # Determine action type based on keywords and performance
        if any(keyword in feedback_lower for keyword in ["feature", "engineer", "create", "add", "transform", "polynomial", "interaction", "log"]):
            # Feature engineering suggested
            action_type = "feature_engineering"
            
            # Extract suggestions from feedback
            suggestions = []
            if "polynomial" in feedback_lower or "square" in feedback_lower:
                suggestions.append("polynomial_features")
            if "interaction" in feedback_lower or "multipl" in feedback_lower:
                suggestions.append("interaction_features")
            if "log" in feedback_lower:
                suggestions.append("log_transformation")
            
            # Default suggestions if none found
            if not suggestions:
                suggestions = ["polynomial_features", "interaction_features"]
            
            parameters = {
                "suggestions": suggestions,
                "target_model": target_model
            }
            
        elif any(keyword in feedback_lower for keyword in ["hyperparameter", "parameter", "tune", "optimize", "grid", "search"]):
            # Hyperparameter tuning suggested
            action_type = "hyperparameter_tuning"
            
            model_type = best_model_perf.model_type.value if hasattr(best_model_perf.model_type, 'value') else str(best_model_perf.model_type)
            
            # Extract parameter suggestions from feedback
            params = {}
            if "max_depth" in feedback_lower:
                params["max_depth"] = [3, 5, 7]
            if "learning_rate" in feedback_lower or "eta" in feedback_lower:
                params["learning_rate"] = [0.01, 0.1, 0.3]
            if "n_estimators" in feedback_lower:
                params["n_estimators"] = [100, 200]
            
            parameters = {
                "model_type": model_type,
                "target_model": target_model,
                "optimization_method": "grid_search",
                "hyperparameters": params,
                "n_trials": 50
            }
            
        elif any(keyword in feedback_lower for keyword in ["ensemble", "combine", "vote", "stack"]):
            # Ensemble suggested
            action_type = "ensemble"
            
            # Get top models for ensemble
            top_models = evaluation_results.model_evaluations[:3]
            
            parameters = {
                "ensemble_type": "voting",
                "base_models": [m.model_name for m in top_models],
                "target_model": target_model
            }
            
        else:
            # Default: based on performance
            if best_score < 70:
                action_type = "hyperparameter_tuning"
                model_type = best_model_perf.model_type.value if hasattr(best_model_perf.model_type, 'value') else str(best_model_perf.model_type)
                parameters = {
                    "model_type": model_type,
                    "target_model": target_model,
                    "optimization_method": "grid_search",
                    "n_trials": 50
                }
            else:
                action_type = "feature_engineering"
                parameters = {
                    "suggestions": ["polynomial_features", "interaction_features"],
                    "target_model": target_model
                }
        
        print(f"\n{datetime.now()} - Categorized feedback as {action_type}")
        print(f"Parameters: {parameters}")
        print(f"Target model: {target_model}")
        print(f"Next iteration: {iteration+1}")

        return {
            "success": True,
            "action_type": action_type,
            "parameters": parameters,
            "target_model": target_model,
            "agent_feedback": agent_feedback,
            "current_performance": {
                "rmse": best_model_perf.rmse,
                "r2": best_model_perf.r2,
                "overall_score": best_score
            },
            "next_iteration": iteration+1,
            "message": f"Categorized feedback as {action_type}"
        }
        
    except Exception as e:
        print(f"\n{datetime.now()} - Error categorizing feedback: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error categorizing feedback: {str(e)}"
        }