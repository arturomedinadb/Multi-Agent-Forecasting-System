"""
Pydantic models for feature engineering configuration and validation.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum

class DataAnalysisResult(BaseModel):
    """Result of automated dataset structure analysis."""
    input_file_path: str = Field(..., description="Path to the input file used for analysis")
    shape: List[int] = Field(..., description="Dataset shape as [rows, columns]")
    columns: List[str] = Field(..., description="List of all column names")
    dtypes: Dict[str, str] = Field(..., description="Column data types as strings")
    missing_values: Dict[str, int] = Field(..., description="Count of missing values per column")
    numeric_columns: List[str] = Field(default_factory=list, description="List of numeric columns")
    categorical_columns: List[str] = Field(default_factory=list, description="List of categorical columns")
    datetime_columns: List[str] = Field( default_factory=list, description="List of datetime columns")
    boolean_columns: List[str] = Field(default_factory=list, description="List of boolean columns")
    memory_usage_bytes: int = Field(..., description="Total memory usage of the dataset in bytes")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list, description="Sample rows from the dataset")

    model_config = ConfigDict(extra="forbid")

class FeatureType(str, Enum):
    """Types of features that can be engineered."""
    LAG = "lag"
    ROLLING = "rolling"
    TIME = "time"
    PROMOTION = "promotion"
    HOLIDAY = "holiday"
    WEATHER = "weather"
    ECONOMIC = "economic"
    INTERACTION = "interaction"
    POLYNOMIAL = "polynomial"

class LagFeatureConfig(BaseModel):
    """Configuration for lag features."""
    target_column: str = Field(..., description="Column to create lag features for")
    lags: List[int] = Field(default=[1, 7, 14, 30], description="List of lag periods")
    group_by: Optional[List[str]] = Field(default=None, description="Columns to group by for lag calculation")
    
    @validator('lags')
    def validate_lags(cls, v):
        if not v or any(lag <= 0 for lag in v):
            raise ValueError("All lag values must be positive integers")
        return sorted(set(v))  # Remove duplicates and sort

class RollingFeatureConfig(BaseModel):
    """Configuration for rolling window features."""
    target_column: str = Field(..., description="Column to create rolling features for")
    windows: List[int] = Field(default=[7, 14, 30, 90], description="Rolling window sizes")
    functions: List[str] = Field(default=["mean", "std", "min", "max"], description="Aggregation functions")
    group_by: Optional[List[str]] = Field(default=None, description="Columns to group by for rolling calculation")
    
    @validator('functions')
    def validate_functions(cls, v):
        valid_functions = ["mean", "std", "min", "max", "median", "sum", "count"]
        for func in v:
            if func not in valid_functions:
                raise ValueError(f"Function '{func}' not supported. Valid functions: {valid_functions}")
        return v

class TimeFeatureConfig(BaseModel):
    """Configuration for time-based features."""
    date_column: str = Field(default="transaction_date", description="Date column to extract features from")
    features: List[str] = Field(
        default=["year", "month", "day", "dayofweek", "quarter", "is_weekend"],
        description="Time features to extract"
    )
    
    @validator('features')
    def validate_features(cls, v):
        valid_features = [
            "year", "month", "day", "dayofweek", "dayofyear", "week", "quarter",
            "is_weekend", "is_month_start", "is_month_end", "is_quarter_start", 
            "is_quarter_end", "is_year_start", "is_year_end"
        ]
        for feature in v:
            if feature not in valid_features:
                raise ValueError(f"Feature '{feature}' not supported. Valid features: {valid_features}")
        return v

class PromotionFeatureConfig(BaseModel):
    """Configuration for promotion-related features."""
    promo_active_col: str = Field(default="promo_active", description="Promotion active column")
    promo_price_col: str = Field(default="promo_price", description="Promotion price column")
    orig_price_col: str = Field(default="unit_orig_price", description="Original price column")
    net_price_col: str = Field(default="unit_net_price", description="Net price column")
    promo_start_col: str = Field(default="promo_start_date", description="Promotion start date column")
    promo_end_col: str = Field(default="promo_end_date", description="Promotion end date column")
    date_col: str = Field(default="transaction_date", description="Date column, used for days_since_promo")
    
    features: List[str] = Field(
        default=["discount_pct", "promo_duration", "days_since_promo", "promo_intensity"],
        description="Promotion features to create"
    )

class HolidayFeatureConfig(BaseModel):
    """Configuration for holiday-related features."""
    holiday_name_col: str = Field(default="holiday_name", description="Holiday name column")
    is_holiday_col: str = Field(default="is_holiday", description="Is holiday boolean column")
    date_col: str = Field(default="transaction_date", description="Date column")
    
    features: List[str] = Field(
        default=["days_to_holiday", "days_from_holiday", "holiday_type", "is_pre_holiday", "is_post_holiday"],
        description="Holiday features to create"
    )

class WeatherFeatureConfig(BaseModel):
    """Configuration for weather-related features."""
    temp_col: str = Field(default="avg_temperature_c", description="Temperature column")
    precip_col: str = Field(default="precipitation_mm", description="Precipitation column")
    
    features: List[str] = Field(
        default=["temp_category", "precip_category", "weather_severity", "comfort_index"],
        description="Weather features to create"
    )

class EconomicFeatureConfig(BaseModel):
    """Configuration for economic features."""
    cpi_col: str = Field(default="cpi_monthly", description="CPI column")
    gdp_col: str = Field(default="gdp_monthly", description="GDP column")
    unemployment_col: str = Field(default="unemployment_rate_monthly", description="Unemployment rate column")
    population_col: str = Field(default="population_monthly", description="Population column")
    
    features: List[str] = Field(
        default=["cpi_change", "gdp_change", "unemployment_trend", "economic_health_index"],
        description="Economic features to create"
    )

class FeatureEngineeringConfig(BaseModel):
    """Main configuration for feature engineering pipeline."""
    # Feature configurations
    lag_features: Optional[LagFeatureConfig] = None
    rolling_features: Optional[RollingFeatureConfig] = None
    time_features: Optional[TimeFeatureConfig] = None
    promotion_features: Optional[PromotionFeatureConfig] = None
    holiday_features: Optional[HolidayFeatureConfig] = None
    weather_features: Optional[WeatherFeatureConfig] = None
    economic_features: Optional[EconomicFeatureConfig] = None
    
    # Advanced features
    interaction_features: bool = Field(default=True, description="Create interaction features between key variables")
    polynomial_features: bool = Field(default=False, description="Create polynomial features for numeric variables")
    polynomial_degree: int = Field(default=2, description="Degree for polynomial features")
    
    # Data processing options
    handle_missing: str = Field(default="forward_fill", description="Strategy for handling missing values")
    normalize_features: bool = Field(default=False, description="Whether to normalize numeric features")
    
    @validator('handle_missing')
    def validate_handle_missing(cls, v):
        valid_strategies = ["forward_fill", "backward_fill", "interpolate", "drop", "zero_fill"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid missing value strategy. Valid options: {valid_strategies}")
        return v

class FeatureSet(BaseModel):
    """Represents a set of engineered features."""
    feature_names: List[str] = Field(..., description="List of feature names")
    feature_types: Dict[str, FeatureType] = Field(..., description="Mapping of feature names to types")
    feature_descriptions: Dict[str, str] = Field(default_factory=dict, description="Descriptions of each feature")
    created_at: datetime = Field(default_factory=datetime.now, description="When features were created")
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[str]:
        """Get all features of a specific type."""
        return [name for name, ftype in self.feature_types.items() if ftype == feature_type]
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)

class FeatureEngineeringResult(BaseModel):
    """Result of feature engineering process."""
    success: bool = Field(..., description="Whether feature engineering was successful")
    input_shape: tuple = Field(..., description="Shape of input data (rows, columns)")
    output_shape: tuple = Field(..., description="Shape of output data (rows, columns)")
    features_created: int = Field(..., description="Number of features created")
    feature_set: FeatureSet = Field(..., description="Details of created features")
    processing_time: float = Field(..., description="Time taken for processing in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated")
    
    @property
    def feature_ratio(self) -> float:
        """Ratio of new features to original features."""
        original_features = self.input_shape[1]
        return self.features_created / original_features if original_features > 0 else 0.0
