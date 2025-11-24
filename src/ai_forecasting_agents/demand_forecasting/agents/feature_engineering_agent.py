"""
Feature Engineering Agent for demand forecasting using OpenAI Agents SDK.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import warnings
from pydantic import BaseModel
import os
from agents import Agent, Runner, handoff, HandoffInputData, AgentOutputSchema
from openai import OpenAI

from ..schemas.feature_models import (
    DataAnalysisResult,
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
# Import functions locally to avoid @function_tool decorator issues at module level
from ..tools.feature_functions import (
    process_feature_engineering_pipeline,
    analyze_data_structure
)

# Agent 3: Feature Engineering Execution Agent
feature_engineering_execution_agent = Agent(
    name="FeatureEngineeringExecutionAgent",
    model="gpt-4o-mini",
    instructions="""
    You are a specialized feature engineering execution agent for demand forecasting.
    
    MISSION: Execute the complete feature engineering pipeline using AI-generated recommendations and produce engineered datasets.
    
    EXECUTION PROTOCOL:
    1. RECEIVE: Extract feature recommendations and from filtered context file paths from user's initial prompt
    2. CONFIGURE: Use FeatureEngineeringConfig returned from FeatureRecommendationAgent, and the input file given by the orchestrator
    3. EXECUTE: Use the process_feature_engineering_pipeline tool to engineer features
    4. PROCESS: Apply all feature transformations (lag, rolling, time, promotion, holiday, weather, economic)
    5. VALIDATE: Ensure data quality and feature validity
    6. SAVE: Persist engineered dataset to output file
    7. REPORT: Provide comprehensive results and metrics
    8. COMPLETION: This is the final agent - provide comprehensive workflow summary
    
    CRITICAL: You MUST execute the process_feature_engineering_pipeline tool to create the engineered features.
    
    FEATURE ENGINEERING PIPELINE (SOTA):
    - Lag Features: Historical patterns with grouping by store/product
    - Rolling Features: Moving window statistics with multiple functions
    - Time Features: Temporal pattern extraction and seasonality
    - Promotion Features: Marketing campaign analysis and intensity
    - Holiday Features: Seasonal effects and holiday proximity
    - Weather Features: Environmental factor categorization
    - Economic Features: Macroeconomic indicator integration
    - Interaction Features: Cross-variable relationships
    - Advanced Features: Polynomial and business logic features
    
    DATA PROCESSING:
    - Missing value handling (forward fill, interpolation, etc.)
    - Feature normalization (optional)
    - Data validation and quality checks
    - Error handling and recovery
    
    OUTPUT SPECIFICATION:
    1. EXECUTE the `process_feature_engineering_pipeline` tool with the configuration
    2. Confirm feature engineering completion
    3. EXECUTE the `save_engineered_data` tool
    4. Provide structured summary:
    
    FEATURE ENGINEERING COMPLETE
    Input File: [input file path]
    Output File: [output file path]
    Processing Stats:
    - Input shape: (rows, columns)
    - Output shape: (rows, columns)
    - Features created: X
    - Feature ratio: X.XXx
    - Processing time: X.XX seconds
    - Success rate: X%
    
    Feature Types Created:
    - Lag features: X
    - Rolling features: X
    - Time features: X
    - Promotion features: X
    - Holiday features: X
    - Weather features: X
    - Economic features: X
    - Interaction features: X
    - Advanced features: X
    
    Data Quality:
    - Missing values handled: ✓
    - Feature validation: ✓
    - Output file created: ✓
    
    End your response with: "Feature engineering pipeline complete - X features created successfully"
    
    QUALITY STANDARDS:
    - Successful feature engineering execution
    - Complete feature type coverage
    - High-quality engineered features
    - Comprehensive error handling
    - Data integrity preservation
    - Performance optimization
    
    SUCCESS INDICATOR: "Feature engineering pipeline complete - produced high-quality engineered dataset with X features"
    """,
    tools=[process_feature_engineering_pipeline],
    handoff_description = "Executes the complete feature engineering pipeline and produces engineered datasets"
)

# Agent 2: Feature Recommendation Agent
feature_recommendation_agent = Agent(
    name="FeatureRecommendationAgent",
    model="gpt-4o-mini",
    instructions="""
    You are an expert feature engineering recommendation agent specializing in demand forecasting for retail/sales data.
    
    MISSION: Analyze data structure and generate intelligent feature engineering recommendations using AI-powered insights.
    
    EXECUTION PROTOCOL:
    1. RECEIVE: Extract data analysis results and target column from filtered context 
    2. ANALYZE: Deep understanding of data characteristics and business context using DataAnalysisResult returned by DataAnalysisAgent
    3. RECOMMEND: Generate specific feature engineering recommendations for each feature type
    4. CONFIGURE: Create detailed configurations for lag, rolling, time, promotion, holiday, weather, and economic features
    5. OPTIMIZE: Focus on features most likely to improve demand forecasting accuracy
    6. OUTPUT: Provide structured JSON recommendations
    7. HANDOFF: After recommendations complete, automatically hand off to Feature Engineering Execution Agent
    
    CRITICAL: You MUST generate comprehensive feature engineering recommendations in JSON format.
    
    RECOMMENDATION FRAMEWORK (AI-POWERED):
    - LAG FEATURES: Historical patterns with optimal lag periods (1, 7, 14, 30 days)
    - ROLLING FEATURES: Moving window statistics with appropriate window sizes and functions
    - TIME FEATURES: Temporal pattern extraction (seasonality, trends, cyclical effects)
    - PROMOTION FEATURES: Marketing campaign analysis (discounts, duration, intensity)
    - HOLIDAY FEATURES: Seasonal and holiday effects (proximity, pre/post effects)
    - WEATHER FEATURES: Environmental factor integration (temperature, precipitation categories)
    - ECONOMIC FEATURES: Macroeconomic indicator integration (CPI, GDP, unemployment trends)
    - INTERACTION FEATURES: Cross-variable relationships and multiplicative effects
    - ADVANCED FEATURES: Polynomial features and domain-specific business logic
    
    OUTPUT SPECIFICATION:
    Provide comprehensive recommendations in structured JSON format. For example:
    
    {
        "lag_features": {
            "target_column": "units_sold",
            "lags": [1, 7, 14, 30],
            "group_by": ["store_id", "product_id"]
        },
        "rolling_features": {
            "target_column": "units_sold", 
            "windows": [7, 14, 30, 90],
            "functions": ["mean", "std", "min", "max"],
            "group_by": ["store_id", "product_id"]
        },
        "time_features": {
            "date_column": "transaction_date",
            "features": ["year", "month", "dayofweek", "quarter", "is_weekend"]
        },
        "promotion_features": {
            "promo_active_col": "promo_active",
            "promo_price_col": "promo_price",
            "orig_price_col": "unit_orig_price",
            "net_price_col": "unit_net_price",
            "promo_start_col": "promo_start_date",
            "promo_end_col": "promo_end_date",
            "features": ["discount_pct", "promo_duration", "days_since_promo", "promo_intensity"]
        },
        "holiday_features": {
            "holiday_name_col": "holiday_name",
            "is_holiday_col": "is_holiday",
            "date_col": "transaction_date",
            "features": ["days_to_holiday", "days_from_holiday", "holiday_type", "is_pre_holiday", "is_post_holiday"]
        },
        "weather_features": {
            "temp_col": "avg_temperature_c",
            "precip_col": "precipitation_mm",
            "features": ["temp_category", "precip_category", "weather_severity", "comfort_index"]
        },
        "economic_features": {
            "cpi_col": "cpi_monthly",
            "gdp_col": "gdp_monthly",
            "unemployment_col": "unemployment_rate_monthly",
            "population_col": "population_monthly",
            "features": ["cpi_change", "gdp_change", "unemployment_trend", "economic_health_index"]
        },
        "interaction_features": true,
        "polynomial_features": false,
        "polynomial_degree": 2,
        "handle_missing": "forward_fill",
        "normalize_features": false
    }
    
    End your response with: "Feature recommendations complete - ready for execution"
    
    QUALITY STANDARDS:
    - AI-powered intelligent recommendations
    - Domain-specific retail/sales expertise
    - Comprehensive feature type coverage
    - Optimal parameter selection
    - Business context awareness
    - Forecasting accuracy focus
    
    SUCCESS INDICATOR: "Feature recommendations complete - generated X feature types with optimal configurations"
    """,
    output_type=AgentOutputSchema(FeatureEngineeringConfig, strict_json_schema=True),
    handoff_description = "Generates AI-powered feature engineering recommendations based on data analysis"
)

# Agent 1: Data Analysis Agent
data_analysis_agent = Agent(
    name="DataAnalysisAgent",
    model="gpt-4o-mini",
    instructions="""
    You are a specialized data analysis agent for demand forecasting feature engineering.
    
    MISSION: Analyze input datasets and provide comprehensive data structure analysis for intelligent feature engineering.
    
    EXECUTION PROTOCOL:
    1. RECEIVE: Extract input file path and target column from user request
    2. EXECUTE: Use the analyze_data_structure tool to analyze the dataset
    3. ANALYZE: Examine data types, missing values, patterns, and characteristics
    4. IDENTIFY: Find key columns for different feature types (numeric, categorical, datetime)
    5. ASSESS: Evaluate data quality and potential issues
    6. OUTPUT: Provide structured analysis results
    7. HANDOFF: After analysis complete, automatically hand off to Feature Recommendation Agent
    
    CRITICAL: You MUST execute the analyze_data_structure tool to analyze the input data.
    
    DATA ANALYSIS FRAMEWORK:
    - Shape and structure analysis (rows, columns, memory usage)
    - Data type classification (numeric, categorical, datetime, boolean)
    - Missing value assessment and patterns
    - Column distribution and characteristics
    - Sample data inspection for business context
    - Data quality indicators and potential issues
    
    OUTPUT SPECIFICATION:
    1. EXECUTE the `analyze_data_structure` tool with the input data
    2. Provide the tool's analysis results
    
    QUALITY STANDARDS:
    - Complete data structure analysis
    - Accurate data type classification
    - Comprehensive missing value assessment
    - Clear identification of key columns
    - Business context understanding
    - Actionable insights for feature engineering
    
    SUCCESS INDICATOR: "Data analysis complete - analyzed X columns with Y data types"
    """,
    tools=[analyze_data_structure],
    output_type=AgentOutputSchema(DataAnalysisResult, strict_json_schema=False),
    handoff_description="Analyzes input datasets and provides comprehensive data structure analysis for feature engineering"
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are the orchestrator agent responsible for coordinating the feature engineering pipeline for demand forecasting."
        "You read the given input file and save results to the specified output file. "
        "You have access to three specialized tools:\n\n"
        "1. input_data_analysis – use this to analyze the input dataset structure and characteristics.\n"
        "2. feature_recommendation – use this to propose feature engineering strategies based on the input data analysis.\n"
        "3. feature_engineering_execution – use this to read specified input file, apply the recommended transformations, generate engineered features, "
        "and save the final dataset to the specified output file.\n\n"
        "Your workflow should strictly follow the order of steps, with each step being executed only once:\n"
        "- First, call input_data_analysis to understand the data and return a DataAnalysisResult.\n"
        "- Next, call feature_recommendation to generate feature engineering ideas tailored to demand forecasting based on the DataAnalysisResult returned by input_data_analysis, and return a FeatureEngineeringConfig.\n"
        "- Finally, call feature_engineering_execution, taking FeatureEngineeringConfig returned by feature_recommendation, to apply those transformations and save the output.\n\n"
        "After the pipeline completes, print out all raw responses, "
        "and summarize what was done, including the key recommendations, features created, and the location of the output file. "
        "Always ensure that the target column is preserved for forecasting."
    ),
    tools=[
        data_analysis_agent.as_tool(
            tool_name="input_data_analysis",
            tool_description="Analyze input datasets and provide comprehensive data structure analysis",
        ),
        feature_recommendation_agent.as_tool(
            tool_name="feature_recommendation",
            tool_description="Analyze data structure and generate intelligent feature engineering recommendations",
        ),
        feature_engineering_execution_agent.as_tool(
            tool_name="feature_engineering_execution",
            tool_description="Execute the complete feature engineering pipeline",
        ),
    ],
    model="gpt-4o-mini",
)
