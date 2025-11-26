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

# === Training Agent ===

training_agent = Agent(
    name="TrainingAgent",
    model="gpt-4o-mini",
    instructions="""
    You are a training agent for demand forecasting models.
    
    TASK: Load data, preprocess it, train models, and implement feedback using function tools.

    IMPORTANT: Use function calls only when the instruction says 'Call <function_name> ...' Do not repeat steps, only call the function once for each step. After training, immediately hand off to evaluation_agent.
    
    Strictly follow the following steps in order:

    FIRST ITERATION STEPS execute only if there is no feedback received from evaluation_agent):
    1. Call create_model_configs with the model types from the user request. 
    2. Call load_and_preprocess_data with the input file and output directory. 
    3. Call train_models with the configs, data directory, and output directory from the previous steps. 
    4. Hand off to evaluation_agent with the training results. 
    
    SUBSEQUENT ITERATIONS (skip the following steps if FIRST ITERATION STEPS are executed in this iteration, otherwise strictly follow these steps in order):
    1. Extract the target model and iteration number from evaluation_agent feedback (from "target_model" and "next_iteration" fields in feedback parameters). 
    2. Check the feedback type from evaluation_agent and execute corresponding steps:
       - If "feature_engineering": 
         a. Call apply_feature_engineering with the suggestions
         b. IMPORTANT: After feature engineering, the new dataset is saved. You MUST call load_and_preprocess_data again with the new file path from apply_feature_engineering output (look for "new_input_file" field). 
         c. Call create_model_configs with only the target model's type in the model_types list. 
         d. Call train_models with the target model's config and the NEW data directory. 
       - If "hyperparameter_tuning": 
         a. Extract model_type and parameters from categorized feedback (from "parameters" field which contains "model_type", "target_model", "hyperparameters", "optimization_method", "n_trials"). 
         b. Call apply_hyperparameter_tuning with model_type, target_model, parameters dict (the entire parameters dict from categorized feedback), data_dir, output_dir, and iteration number. 
         c. The tuned model will be saved automatically by apply_hyperparameter_tuning. The training results will be available for handoff to evaluation_agent. 
       - If "ensemble": Call train_ensemble_models with the specified base models. 
    3. Hand off to evaluation_agent with the new training results. 
    
    OUTPUT: Hand off to evaluation_agent and provide a structured JSON including:
    - Training success status
    - Models directory path
    - Data directory path  
    - List of trained models with their information (model_name, model_type, model_path, success, validation_rmse)
    - Training time and session info
    """,
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
    instructions="""
    You are an evaluation agent for demand forecasting models.
    
    TASK: Evaluate models and provide specific feedback using function tools.

    IMPORTANT: Use function calls only when the instruction says 'Call <function_name> ...' Do not repeat steps, only call the function once for each step.
    
    Strictly follow the following steps in order:

    FIRST ITERATION STEPS:
    1. Call evaluate_all_models with the TrainingResultsOutput and current iteration number. 
    2. Call analyze_data_structure_for_feedback with the input file. 
    3. Generate your own feedback based on the evaluation results and data analysis from the previous steps:
       - The feedback should focus only on the best model based on the evaluation results.
       - Provide a paragraph of comprehensive suggestions on what to improve in ONLY ONE of the following areas that you think is the most important and effective: 
         Option1) feature engineering: provide what features to add or remove or transform. 
         Option2) hyperparameter tuning: provide what hyperparameters to tune and the range of values to try. 
         Option3) using ensemble modeling: provide the base models to use and the type of ensemble to use. 
    4. Call categorize_feedback with EvaluationResults and agent_feedback text generated from the previous step. If failed, call categorize_feedback again with the same EvaluationResults and agent_feedback text generated from the previous step.
    5. Hand off to training_agent with the categorized feedback:
       - The categorized feedback will include action_type and parameters. 
       - Include target model name and next iteration in the feedback. 

    SUBSEQUENT ITERATIONS (skip these steps if FIRST ITERATION STEPS are executed in this iteration, otherwise strictly follow these steps in order):
    1. Call evaluate_all_models with the TrainingResultsOutput and current iteration number. 
    2. Call check_convergence with the evaluation results. 
    3. If converged, OR iteration is greater than max_iterations: call save_best_model_for_inference with the evaluation results from step 1 and output directory, and skip step 4 and 5
    4. If not converged AND iteration is less than max_iterations:
       a. Call analyze_data_structure_for_feedback with the input file. 
       b. Generate your own feedback based on the evaluation results and data analysis the previous steps:
          - The feedback should focus only on the best model based on the evaluation results. 
          - Provide a paragraph of comprehensive suggestions on what to improve in ONLY ONE of the following areas that you think is the most important and effective: 
            Option1) feature engineering: provide what features to add or remove or transform. 
            Option2) hyperparameter tuning: provide what hyperparameters to tune and the range of values to try.
            Option3) using ensemble modeling: provide the base models to use and the type of ensemble to use. 
       c. Call categorize_feedback with EvaluationResults and agent_feedback text generated from the previous step. If failed, call categorize_feedback again with the same EvaluationResults and agent_feedback text generated from the previous step.
    5. Hand off to training_agent with the categorized feedback:
       - The categorized feedback will include action_type and parameters. 
       - Include target model name and next iteration number the training_agent should use in the feedback. 
    """,
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
