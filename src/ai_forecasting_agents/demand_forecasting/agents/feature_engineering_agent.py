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
from ..prompts.factory import get_renderer

# Static role instructions for the agents below are rendered once at import
# time from prompts/demand_forecasting/v1/*.j2 (via registry.yaml).
_renderer = get_renderer()

# Agent 3: Feature Engineering Execution Agent
feature_engineering_execution_agent = Agent(
    name="FeatureEngineeringExecutionAgent",
    model="gpt-4o-mini",
    instructions=_renderer.render("FeatureEngineeringExecutionAgent"),
    tools=[process_feature_engineering_pipeline],
    handoff_description = "Executes the complete feature engineering pipeline and produces engineered datasets"
)

# Agent 2: Feature Recommendation Agent
feature_recommendation_agent = Agent(
    name="FeatureRecommendationAgent",
    model="gpt-4o-mini",
    instructions=_renderer.render("FeatureRecommendationAgent"),
    output_type=AgentOutputSchema(FeatureEngineeringConfig, strict_json_schema=True),
    handoff_description = "Generates AI-powered feature engineering recommendations based on data analysis"
)

# Agent 1: Data Analysis Agent
data_analysis_agent = Agent(
    name="DataAnalysisAgent",
    model="gpt-4o-mini",
    instructions=_renderer.render("DataAnalysisAgent"),
    tools=[analyze_data_structure],
    output_type=AgentOutputSchema(DataAnalysisResult, strict_json_schema=False),
    handoff_description="Analyzes input datasets and provides comprehensive data structure analysis for feature engineering"
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=_renderer.render("orchestrator_agent"),
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
