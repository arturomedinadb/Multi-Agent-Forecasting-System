"""
Agent Registry for demand forecasting using OpenAI Agents SDK with routing.
This file defines both agents with their handoffs to avoid circular imports.
"""

from agents import Agent, AgentOutputSchema

from ..tools.training_functions import (
    create_model_configs, load_and_preprocess_data, train_models,
    apply_feature_engineering, apply_hyperparameter_tuning, train_ensemble_models
)
from ..tools.evaluation_functions import (
    evaluate_model_performance, evaluate_all_models, check_convergence, 
    save_best_model_for_inference, analyze_data_structure_for_feedback, categorize_feedback
)
from ..schemas.forecasting_models import TrainingResultsOutput
from ..prompts.factory import get_renderer

# Static role instructions for the agents below are rendered once at import
# time from prompts/demand_forecasting/v1/*.j2 (via registry.yaml).
_renderer = get_renderer()

# === Training Agent ===

training_agent = Agent(
    name="TrainingAgent",
    model="gpt-4o-mini",
    instructions=_renderer.render("TrainingAgent"),
    tools=[
        create_model_configs,
        load_and_preprocess_data,
        train_models,
        apply_feature_engineering,
        apply_hyperparameter_tuning,
        train_ensemble_models
    ],
    output_type=AgentOutputSchema(TrainingResultsOutput, strict_json_schema=False),
    handoffs=[]  # Will be set after evaluation_agent is defined
)

# === Evaluation Agent ===

evaluation_agent = Agent(
    name="EvaluationAgent",
    model="gpt-4o-mini",
    instructions=_renderer.render("EvaluationAgent"),
    tools=[
        evaluate_all_models,
        check_convergence,
        save_best_model_for_inference,
        analyze_data_structure_for_feedback,
        categorize_feedback
    ],
    handoffs=[]  # Will be set after training_agent is defined
)

# Set up handoffs after both agents are defined
training_agent.handoffs = [evaluation_agent]
evaluation_agent.handoffs = [training_agent]
