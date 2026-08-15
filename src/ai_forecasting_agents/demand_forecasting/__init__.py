"""
Demand forecasting module with AI agents for feature engineering, training, and forecasting.
"""

from .schemas.feature_models import FeatureEngineeringConfig, FeatureSet
from .tools.feature_functions import (
    create_lag_features,
    create_rolling_features,
    create_time_features,
    create_promotion_features,
    create_holiday_features,
    create_weather_features,
    create_economic_features,
    validate_feature_set
)
# Note: Training and evaluation functions are available through agents
# Direct function imports removed to avoid @function_tool decorator issues

__all__ = [
    # Feature Engineering
    "FeatureEngineeringConfig", 
    "FeatureSet",
    "create_lag_features",
    "create_rolling_features", 
    "create_time_features",
    "create_promotion_features",
    "create_holiday_features",
    "create_weather_features",
    "create_economic_features",
    "validate_feature_set",
    
    # Training
    # "TrainingConfig",
    # "ModelConfig",
    # "PortfolioResult",
    # "EvaluationFeedback",
    # "ImprovementAction",
    # "IterationResult",
    # "TrainingSession",
    # "ModelType"
]
