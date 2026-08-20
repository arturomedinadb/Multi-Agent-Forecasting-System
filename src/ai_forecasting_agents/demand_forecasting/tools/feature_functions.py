"""
Feature engineering functions for demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from functools import wraps
from agents import function_tool

from ..schemas.feature_models import (
    FeatureEngineeringConfig,
    FeatureSet,
    FeatureType,
    LagFeatureConfig,
    RollingFeatureConfig,
    TimeFeatureConfig,
    PromotionFeatureConfig,
    HolidayFeatureConfig,
    WeatherFeatureConfig,
    EconomicFeatureConfig,
    FeatureEngineeringResult
)

def handle_errors(func):
    """Decorator to handle errors in feature engineering functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper


def _usable_group_by(df: pd.DataFrame, group_by: Optional[List[str]]) -> Optional[List[str]]:
    """Drop group_by columns that are entirely null (grouping on an all-NaN
    column drops every row from the groupby, which then breaks downstream
    rolling/concat operations). Returns None if nothing usable remains."""
    if not group_by:
        return None
    usable = [col for col in group_by if col in df.columns and df[col].notna().any()]
    if usable != group_by:
        dropped = [col for col in group_by if col not in usable]
        warnings.warn(f"Dropping all-null group_by column(s) {dropped}; grouping on {usable or 'none (ungrouped)'}")
    return usable or None

@handle_errors
def create_lag_features(df: pd.DataFrame, config: LagFeatureConfig) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        df: Input DataFrame
        config: Lag feature configuration
        
    Returns:
        DataFrame with lag features added
    """
    df_result = df.copy()
    group_by = _usable_group_by(df_result, config.group_by)

    if group_by:
        # Group by specified columns and create lag features
        for lag in config.lags:
            lag_col = f"{config.target_column}_lag_{lag}"
            df_result[lag_col] = df_result.groupby(group_by)[config.target_column].shift(lag)
    else:
        # Create lag features without grouping
        for lag in config.lags:
            lag_col = f"{config.target_column}_lag_{lag}"
            df_result[lag_col] = df_result[config.target_column].shift(lag)

    return df_result

@handle_errors
def create_rolling_features(df: pd.DataFrame, config: RollingFeatureConfig) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.
    
    Args:
        df: Input DataFrame
        config: Rolling feature configuration
        
    Returns:
        DataFrame with rolling features added
    """
    df_result = df.copy()
    group_by = _usable_group_by(df_result, config.group_by)

    if group_by:
        # Group by specified columns and create rolling features
        grouped = df_result.groupby(group_by)[config.target_column]
    else:
        grouped = df_result[config.target_column]

    for window in config.windows:
        for func in config.functions:
            feature_name = f"{config.target_column}_rolling_{func}_{window}"

            if group_by:
                df_result[feature_name] = grouped.rolling(window=window, min_periods=1).agg(func).reset_index(level=list(range(len(group_by))), drop=True)
            else:
                df_result[feature_name] = grouped.rolling(window=window, min_periods=1).agg(func)

    return df_result

@handle_errors
def create_time_features(df: pd.DataFrame, config: TimeFeatureConfig) -> pd.DataFrame:
    """
    Create time-based features from date column.
    
    Args:
        df: Input DataFrame
        config: Time feature configuration
        
    Returns:
        DataFrame with time features added
    """
    df_result = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_result[config.date_column]):
        df_result[config.date_column] = pd.to_datetime(df_result[config.date_column])
    
    for feature in config.features:
        if feature == "year":
            df_result["year"] = df_result[config.date_column].dt.year
        elif feature == "month":
            df_result["month"] = df_result[config.date_column].dt.month
        elif feature == "day":
            df_result["day"] = df_result[config.date_column].dt.day
        elif feature == "dayofweek":
            df_result["dayofweek"] = df_result[config.date_column].dt.dayofweek
        elif feature == "dayofyear":
            df_result["dayofyear"] = df_result[config.date_column].dt.dayofyear
        elif feature == "week":
            df_result["week"] = df_result[config.date_column].dt.isocalendar().week
        elif feature == "quarter":
            df_result["quarter"] = df_result[config.date_column].dt.quarter
        elif feature == "is_weekend":
            df_result["is_weekend"] = df_result[config.date_column].dt.dayofweek.isin([5, 6])
        elif feature == "is_month_start":
            df_result["is_month_start"] = df_result[config.date_column].dt.is_month_start
        elif feature == "is_month_end":
            df_result["is_month_end"] = df_result[config.date_column].dt.is_month_end
        elif feature == "is_quarter_start":
            df_result["is_quarter_start"] = df_result[config.date_column].dt.is_quarter_start
        elif feature == "is_quarter_end":
            df_result["is_quarter_end"] = df_result[config.date_column].dt.is_quarter_end
        elif feature == "is_year_start":
            df_result["is_year_start"] = df_result[config.date_column].dt.is_year_start
        elif feature == "is_year_end":
            df_result["is_year_end"] = df_result[config.date_column].dt.is_year_end
    
    return df_result

@handle_errors
def create_promotion_features(df: pd.DataFrame, config: PromotionFeatureConfig) -> pd.DataFrame:
    """
    Create promotion-related features.
    
    Args:
        df: Input DataFrame
        config: Promotion feature configuration
        
    Returns:
        DataFrame with promotion features added
    """
    df_result = df.copy()
    
    for feature in config.features:
        if feature == "discount_pct":
            # Calculate discount percentage
            if config.orig_price_col in df_result.columns and config.net_price_col in df_result.columns:
                df_result["discount_pct"] = ((df_result[config.orig_price_col] - df_result[config.net_price_col]) / df_result[config.orig_price_col] * 100).fillna(0)
        
        elif feature == "promo_duration":
            # Calculate promotion duration
            if config.promo_start_col in df_result.columns and config.promo_end_col in df_result.columns:
                start_dates = pd.to_datetime(df_result[config.promo_start_col], errors='coerce')
                end_dates = pd.to_datetime(df_result[config.promo_end_col], errors='coerce')
                df_result["promo_duration"] = (end_dates - start_dates).dt.days.fillna(0)
        
        elif feature == "days_since_promo":
            # Days since last promotion ended
            if config.promo_end_col in df_result.columns and config.date_col in df_result.columns:
                promo_end_dates = pd.to_datetime(df_result[config.promo_end_col], errors='coerce')
                current_dates = pd.to_datetime(df_result[config.date_col], errors='coerce')
                df_result["days_since_promo"] = (current_dates - promo_end_dates).dt.days.fillna(999)
        
        elif feature == "promo_intensity":
            # Promotion intensity based on discount percentage and duration
            if "discount_pct" in df_result.columns and "promo_duration" in df_result.columns:
                df_result["promo_intensity"] = df_result["discount_pct"] * df_result["promo_duration"] / 100
    
    return df_result

@handle_errors
def create_holiday_features(df: pd.DataFrame, config: HolidayFeatureConfig) -> pd.DataFrame:
    """
    Create holiday-related features.
    
    Args:
        df: Input DataFrame
        config: Holiday feature configuration
        
    Returns:
        DataFrame with holiday features added
    """
    df_result = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_result[config.date_col]):
        df_result[config.date_col] = pd.to_datetime(df_result[config.date_col])
    
    for feature in config.features:
        if feature == "days_to_holiday":
            # Days to next holiday
            df_result["days_to_holiday"] = 0  # Placeholder - would need holiday calendar
            warnings.warn("days_to_holiday feature requires holiday calendar implementation")
        
        elif feature == "days_from_holiday":
            # Days from last holiday
            df_result["days_from_holiday"] = 0  # Placeholder - would need holiday calendar
            warnings.warn("days_from_holiday feature requires holiday calendar implementation")
        
        elif feature == "holiday_type":
            # Categorize holiday types
            if config.holiday_name_col in df_result.columns:
                df_result["holiday_type"] = df_result[config.holiday_name_col].fillna("No Holiday")
        
        elif feature == "is_pre_holiday":
            # Is day before holiday
            if config.is_holiday_col in df_result.columns:
                df_result["is_pre_holiday"] = df_result[config.is_holiday_col].shift(-1).fillna(False)
        
        elif feature == "is_post_holiday":
            # Is day after holiday
            if config.is_holiday_col in df_result.columns:
                df_result["is_post_holiday"] = df_result[config.is_holiday_col].shift(1).fillna(False)
    
    return df_result

@handle_errors
def create_weather_features(df: pd.DataFrame, config: WeatherFeatureConfig) -> pd.DataFrame:
    """
    Create weather-related features.
    
    Args:
        df: Input DataFrame
        config: Weather feature configuration
        
    Returns:
        DataFrame with weather features added
    """
    df_result = df.copy()
    
    for feature in config.features:
        if feature == "temp_category":
            # Categorize temperature
            if config.temp_col in df_result.columns:
                df_result["temp_category"] = pd.cut(
                    df_result[config.temp_col], 
                    bins=[-np.inf, 0, 10, 20, 30, np.inf],
                    labels=["Very Cold", "Cold", "Mild", "Warm", "Hot"]
                )
        
        elif feature == "precip_category":
            # Categorize precipitation
            if config.precip_col in df_result.columns:
                df_result["precip_category"] = pd.cut(
                    df_result[config.precip_col],
                    bins=[-np.inf, 0, 5, 15, 30, np.inf],
                    labels=["No Rain", "Light", "Moderate", "Heavy", "Very Heavy"]
                )
        
        elif feature == "weather_severity":
            # Combined weather severity index
            if config.temp_col in df_result.columns and config.precip_col in df_result.columns:
                # Normalize temperature and precipitation to 0-1 scale
                temp_norm = (df_result[config.temp_col] - df_result[config.temp_col].min()) / (df_result[config.temp_col].max() - df_result[config.temp_col].min())
                precip_norm = df_result[config.precip_col] / df_result[config.precip_col].max()
                df_result["weather_severity"] = (temp_norm + precip_norm) / 2
        
        elif feature == "comfort_index":
            # Weather comfort index (higher is more comfortable)
            if config.temp_col in df_result.columns and config.precip_col in df_result.columns:
                # Optimal temperature around 20-25°C, no precipitation
                temp_comfort = 1 - abs(df_result[config.temp_col] - 22.5) / 50  # Normalize around 22.5°C
                precip_comfort = 1 - (df_result[config.precip_col] / df_result[config.precip_col].max())
                df_result["comfort_index"] = (temp_comfort + precip_comfort) / 2
    
    return df_result

@handle_errors
def create_economic_features(df: pd.DataFrame, config: EconomicFeatureConfig) -> pd.DataFrame:
    """
    Create economic indicator features.
    
    Args:
        df: Input DataFrame
        config: Economic feature configuration
        
    Returns:
        DataFrame with economic features added
    """
    df_result = df.copy()
    
    for feature in config.features:
        if feature == "cpi_change":
            # Month-over-month CPI change
            if config.cpi_col in df_result.columns:
                df_result["cpi_change"] = df_result[config.cpi_col].pct_change().fillna(0)
        
        elif feature == "gdp_change":
            # Month-over-month GDP change
            if config.gdp_col in df_result.columns:
                df_result["gdp_change"] = df_result[config.gdp_col].pct_change().fillna(0)
        
        elif feature == "unemployment_trend":
            # Unemployment trend (3-month moving average)
            if config.unemployment_col in df_result.columns:
                df_result["unemployment_trend"] = df_result[config.unemployment_col].rolling(window=3, min_periods=1).mean()
        
        elif feature == "economic_health_index":
            # Combined economic health index
            health_components = []
            if config.cpi_col in df_result.columns:
                cpi_norm = 1 - (df_result[config.cpi_col] - df_result[config.cpi_col].min()) / (df_result[config.cpi_col].max() - df_result[config.cpi_col].min())
                health_components.append(cpi_norm)
            if config.gdp_col in df_result.columns:
                gdp_norm = (df_result[config.gdp_col] - df_result[config.gdp_col].min()) / (df_result[config.gdp_col].max() - df_result[config.gdp_col].min())
                health_components.append(gdp_norm)
            if config.unemployment_col in df_result.columns:
                unemp_norm = 1 - (df_result[config.unemployment_col] - df_result[config.unemployment_col].min()) / (df_result[config.unemployment_col].max() - df_result[config.unemployment_col].min())
                health_components.append(unemp_norm)
            
            if health_components:
                df_result["economic_health_index"] = pd.concat(health_components, axis=1).mean(axis=1)
    
    return df_result

@handle_errors
def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between specified column pairs.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of (col1, col2) tuples to create interactions for
        
    Returns:
        DataFrame with interaction features added
    """
    df_result = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 in df_result.columns and col2 in df_result.columns:
            # Create multiplicative interaction
            interaction_name = f"{col1}_x_{col2}"
            df_result[interaction_name] = df_result[col1] * df_result[col2]
    
    return df_result

@handle_errors
def create_polynomial_features(df: pd.DataFrame, numeric_columns: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to create polynomial features for
        degree: Degree of polynomial features
        
    Returns:
        DataFrame with polynomial features added
    """
    df_result = df.copy()
    
    for col in numeric_columns:
        if col in df_result.columns and pd.api.types.is_numeric_dtype(df_result[col]):
            for d in range(2, degree + 1):
                poly_name = f"{col}_poly_{d}"
                df_result[poly_name] = df_result[col] ** d
    
    return df_result

def validate_feature_set(df: pd.DataFrame, feature_set: FeatureSet) -> Dict[str, Union[str, int, float, bool]]:
    """
    Validate a feature set against a DataFrame.
    
    Args:
        df: DataFrame to validate against
        feature_set: Feature set to validate
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        "valid": True,
        "missing_features": [],
        "extra_features": [],
        "feature_types": {},
        "data_quality": {}
    }
    
    # Check for missing features
    for feature in feature_set.feature_names:
        if feature not in df.columns:
            validation_results["missing_features"].append(feature)
            validation_results["valid"] = False
    
    # Check for extra features in DataFrame
    for col in df.columns:
        if col not in feature_set.feature_names:
            validation_results["extra_features"].append(col)
    
    # Check feature types
    for feature in feature_set.feature_names:
        if feature in df.columns:
            validation_results["feature_types"][feature] = str(df[feature].dtype)
    
    # Data quality checks
    validation_results["data_quality"] = {
        "missing_values": df.isnull().sum().to_dict(),
        "infinite_values": np.isinf(df.select_dtypes(include=[np.number])).sum().to_dict(),
        "duplicate_rows": df.duplicated().sum()
    }
    
    return validation_results

@function_tool
def analyze_data_structure(input_file: str) -> Dict[str, Union[str, int, float, bool]]:
    """
    Analyze the structure and characteristics of the input data.
    """
    print("-" * 60)
    print("Analyzing data structure...")

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
    print("-" * 60)
    print(f"ANALYSIS: \n{analysis}")
    print("-" * 60)
    
    return analysis

@function_tool
def process_feature_engineering_pipeline(input_file: str, output_file: str, config: FeatureEngineeringConfig) -> FeatureEngineeringResult:
    """
    Process the complete feature engineering pipeline.
    """
    print("Executing feature engineering pipeline...")

    start_time = datetime.now()
    errors = []
    warnings_list = []
    feature_set = FeatureSet(feature_names=[], feature_types={})
    
    try:
        df = pd.read_csv(input_file)
        df_result = df.copy()
        original_shape = df.shape
        
        # Apply feature engineering based on configuration
        if config.lag_features:
            df_result = create_lag_features(df_result, config.lag_features)
            if df_result is None:
                errors.append("Failed to create lag features")
                return FeatureEngineeringResult(
                    success=False,
                    input_shape=original_shape,
                    output_shape=original_shape,
                    features_created=0,
                    feature_set=feature_set,
                    processing_time=0,
                    errors=errors
                )
        
        if config.rolling_features:
            df_result = create_rolling_features(df_result, config.rolling_features)
            if df_result is None:
                errors.append("Failed to create rolling features")
        
        if config.time_features:
            df_result = create_time_features(df_result, config.time_features)
            if df_result is None:
                errors.append("Failed to create time features")
        
        if config.promotion_features:
            df_result = create_promotion_features(df_result, config.promotion_features)
            if df_result is None:
                errors.append("Failed to create promotion features")
        
        if config.holiday_features:
            df_result = create_holiday_features(df_result, config.holiday_features)
            if df_result is None:
                errors.append("Failed to create holiday features")
        
        if config.weather_features:
            df_result = create_weather_features(df_result, config.weather_features)
            if df_result is None:
                errors.append("Failed to create weather features")
        
        if config.economic_features:
            df_result = create_economic_features(df_result, config.economic_features)
            if df_result is None:
                errors.append("Failed to create economic features")
        
        if config.interaction_features:
            # Create interaction features between key variables
            key_vars = ["units_sold", "unit_net_price", "inventory"]
            interaction_pairs = [(var1, var2) for var1 in key_vars for var2 in key_vars if var1 != var2]
            df_result = create_interaction_features(df_result, interaction_pairs)
        
        if config.polynomial_features:
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
            df_result = create_polynomial_features(df_result, numeric_cols, config.polynomial_degree)
        
        # Handle missing values
        if config.handle_missing == "forward_fill":
            df_result = df_result.fillna(method='ffill')
        elif config.handle_missing == "backward_fill":
            df_result = df_result.fillna(method='bfill')
        elif config.handle_missing == "interpolate":
            df_result = df_result.interpolate()
        elif config.handle_missing == "zero_fill":
            df_result = df_result.fillna(0)
        elif config.handle_missing == "drop":
            df_result = df_result.dropna()
        
        # Normalize features if requested
        if config.normalize_features:
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns
            df_result[numeric_cols] = (df_result[numeric_cols] - df_result[numeric_cols].mean()) / df_result[numeric_cols].std()
        
        # Create feature set
        new_features = [col for col in df_result.columns if col not in df.columns]
        feature_set = FeatureSet(
            feature_names=new_features,
            feature_types={feature: FeatureType.LAG for feature in new_features},  # Simplified
            feature_descriptions={feature: f"Engineered feature: {feature}" for feature in new_features}
        )
        print(f"NEW FEATURE CREATED: {new_features}")

        df_result.to_csv(output_file, index=False)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FeatureEngineeringResult(
            success=len(errors) == 0,
            input_shape=original_shape,
            output_shape=df_result.shape,
            features_created=len(new_features),
            feature_set=feature_set,
            processing_time=processing_time,
            errors=errors,
            warnings=warnings_list
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        errors.append(f"Pipeline error: {str(e)}")
        
        return FeatureEngineeringResult(
            success=False,
            input_shape=df.shape,
            output_shape=df.shape,
            features_created=0,
            feature_set=feature_set,
            processing_time=processing_time,
            errors=errors
        )
