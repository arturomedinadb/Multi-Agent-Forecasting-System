"""
Pydantic models for the routing-based demand forecasting system.
Contains only the essential models needed for the workflow.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class ModelType(str, Enum):
    """Types of forecasting models."""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    model_type: ModelType = Field(..., description="Type of model to train")
    model_name: str = Field(..., description="Unique name for this model instance")
    model_uuid: str = Field(..., description="Timestamp-based unique identifier for the model (YYYYMMDDHHMMSS)")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")


class AllModelConfigs(BaseModel):
    configs: List[ModelConfig] = Field(..., description="List of model configurations")
    total_configs: int = Field(..., description="Total number of model configurations")
    iteration: int = Field(..., description="Current iteration number")
    session_uuid: str = Field(..., description="Timestamp-based identifier for the training session (YYYYMMDDHHMMSS)")


class TrainedModel(BaseModel):
    """Information about a single trained model."""
    model_name: str = Field(..., description="Name of the trained model")
    model_type: str = Field(..., description="Type of model (as string)")
    model_path: str = Field(..., description="Path to saved model file")
    model_uuid: str = Field(..., description="Timestamp-based unique identifier for the model (YYYYMMDDHHMMSS)")
    model_filename: str = Field(..., description="Filename of the saved model")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters used for training")
    success: bool = Field(..., description="Whether training was successful")
    validation_rmse: Optional[float] = Field(default=None, description="Validation RMSE if available")
    error: Optional[str] = Field(default=None, description="Error message if training failed")


class TrainingResultsOutput(BaseModel):
    """Output from training agent - contains all information needed by evaluation agent."""
    success: bool = Field(..., description="Whether training completed successfully")
    models_directory: str = Field(..., description="Directory containing all trained model files")
    data_dir: str = Field(..., description="Directory containing preprocessed data splits")
    
    # Model information
    models: List[TrainedModel] = Field(..., description="List of all trained models")
    best_model: str = Field(default=None, description="Best performing model name")
    best_score: float = Field(default=None, description="Best model score")
    
    # Training metadata
    training_time: float = Field(..., description="Total time taken for training")
    session_uuid: str = Field(default=None, description="Timestamp-based identifier for training session (YYYYMMDD_HHMMSS)")
    training_timestamp: Optional[str] = Field(default=None, description="Timestamp when training completed")
    
    # Statistics
    total_models: int = Field(..., description="Total number of models trained")
    
    # Messages
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if training failed")


class ModelPerformance(BaseModel):
    """Performance metrics for a trained model."""
    model_name: str = Field(..., description="Name of the model")
    model_type: ModelType = Field(..., description="Type of model")
    model_path: str = Field(..., description="Path to saved model file")
    model_uuid: str = Field(..., description="Timestamp-based unique identifier for the model (YYYYMMDDHHMMSS)")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    
    # Core metrics
    rmse: float = Field(..., description="Root Mean Square Error")
    r2: float = Field(..., description="R-squared score")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    overall_score: float = Field(..., description="Overall performance score (0-100)")


class EvaluationResults(BaseModel):
    """Results of evaluating all models."""
    success: bool = Field(..., description="Whether training was successful")
    model_evaluations: List[ModelPerformance] = Field(..., description="List of all model evaluations")
    best_model: str = Field(..., description="Name of the best performing model")
    best_score: float = Field(..., description="Score of the best model")
    total_models: int = Field(..., description="Total number of models evaluated")
    iteration: int = Field(..., description="Current iteration number")
    models_directory: str = Field(..., description="Directory containing all model files")


class HyperparameterTuningParameters(BaseModel):
    """Parameters for hyperparameter tuning from categorize_feedback."""
    hyperparameters: Dict[str, List[Union[int, float, str]]] = Field(
        ..., 
        description="Hyperparameter grid to search - dictionary mapping parameter names to lists of values to try"
    )
    optimization_method: str = Field(
        default="grid_search", 
        description="Optimization method to use (default: grid_search). Options: grid_search, random_search"
    )
    n_trials: Optional[int] = Field(
        default=None, 
        description="Number of trials for random search optimization method"
    )


class EvaluationFeedback(BaseModel):
    """Structured feedback from evaluation agent."""
    model_name: str = Field(..., description="Name of the evaluated model")
    overall_score: float = Field(..., description="Overall performance score (0-100)")
    
    # Analysis
    strengths: List[str] = Field(..., description="Model strengths")
    weaknesses: List[str] = Field(..., description="Model weaknesses")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    
    # Improvement action
    action_type: str = Field(..., description="Type of improvement action")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the improvement action")
    target_model: str = Field(..., description="Model to improve")
    iteration: int = Field(..., description="Current iteration number")
    