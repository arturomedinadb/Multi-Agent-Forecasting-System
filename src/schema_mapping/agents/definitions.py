"""Agent instructions and tool bindings for the schema mapping workflow.

This module implements a chain-based handoff pattern:
- Orchestrator → Work Agents (DataPrep, ColumnMapping, DataIntegration)
- Work Agents → Their Evaluation Agents
- Evaluation Agents → Back to Orchestrator

This ensures deterministic flow: Agent → Evaluator → Orchestrator
"""
from agents import Agent

from ..functions import (
    generate_mapped_csvs,
    load_and_describe_dataset,
    merge_mapped_csvs_to_target,
    evaluate_data_prep_agent,
    evaluate_column_mapping_agent,
    evaluate_data_integration_agent,
    validate_final_dataset,
    generate_summary_report,
    generate_final_workflow_report,
    query_conversation_history,
    get_all_dataset_metadata,
)
from ..prompts.factory import get_renderer


# Forward declarations for handoffs
workflow_orchestrator_agent = None

# Static role instructions for the agents below are rendered once at import
# time from prompts/v1/*.j2 (via registry.yaml), rather than living inline
# here, so the same prompt content is used regardless of how each agent is
# constructed.
_renderer = get_renderer()


# Agent 1: Data Preparation Agent
MODEL_DEFAULT = "gpt-4o-mini"


data_prep_agent = Agent(
    name="DataPrepAgent",
    model=MODEL_DEFAULT,
    instructions=_renderer.render("DataPrepAgent"),
    tools=[load_and_describe_dataset],  # Will add save_metadata_to_file tool
    handoffs=[],  # Set after all agents are defined
)


# Agent 2: Column Mapping Agent
column_mapping_agent = Agent(
    name="ColumnMappingAgent",
    model=MODEL_DEFAULT,
    instructions=_renderer.render("ColumnMappingAgent"),
    tools=[get_all_dataset_metadata, generate_mapped_csvs, load_and_describe_dataset, query_conversation_history],
    handoffs=[],  # Set after all agents are defined
)


# Agent 3: Data Integration Agent
data_integration_agent = Agent(
    name="DataIntegrationAgent",
    model=MODEL_DEFAULT,
    instructions=_renderer.render("DataIntegrationAgent"),
    tools=[merge_mapped_csvs_to_target, query_conversation_history],
    handoffs=[],  # Set after all agents are defined
)


# Evaluation Agent 1: Data Prep Evaluator
data_prep_evaluation_agent = Agent(
    name="DataPrepEvaluationAgent",
    model=MODEL_DEFAULT,
    instructions=_renderer.render("DataPrepEvaluationAgent"),
    tools=[evaluate_data_prep_agent, generate_summary_report, query_conversation_history],
    handoffs=[],  # Set after orchestrator is built
)


# Evaluation Agent 2: Column Mapping Evaluator
column_mapping_evaluation_agent = Agent(
    name="ColumnMappingEvaluationAgent",
    model=MODEL_DEFAULT,
    instructions=_renderer.render("ColumnMappingEvaluationAgent"),
    tools=[evaluate_column_mapping_agent, generate_summary_report, query_conversation_history],
    handoffs=[],  # Set after orchestrator is built
)


# Evaluation Agent 3: Data Integration Evaluator
data_integration_evaluation_agent = Agent(
    name="DataIntegrationEvaluationAgent",
    model=MODEL_DEFAULT,
    instructions=_renderer.render("DataIntegrationEvaluationAgent"),
    tools=[evaluate_data_integration_agent, validate_final_dataset, generate_summary_report, query_conversation_history],
    handoffs=[],  # Set after orchestrator is built
)


def create_workflow_orchestrator_agent(instructions: str) -> Agent:
    """
    Instantiate the Workflow Orchestrator agent with runtime instructions and
    configure chain-based handoffs:
    
    Chain Pattern:
    - Orchestrator → Work Agents (DataPrep, ColumnMapping, DataIntegration)
    - Work Agents → Their Evaluation Agents
    - Evaluation Agents → Back to Orchestrator
    
    This ensures deterministic flow where each work agent automatically
    routes to its evaluator, which then returns results to the orchestrator.
    """
    global workflow_orchestrator_agent

    orchestrator = Agent(
        name="WorkflowOrchestrator",
        model=MODEL_DEFAULT,
        instructions=instructions,
        tools=[generate_final_workflow_report, query_conversation_history],
        handoffs=[
            # Orchestrator only hands off to WORK agents, not evaluators
            data_prep_agent,
            column_mapping_agent,
            data_integration_agent,
        ],
    )

    # Chain Pattern: Work Agent → Evaluator → Orchestrator
    # Each work agent hands off to its evaluator (not orchestrator)
    data_prep_agent.handoffs = [data_prep_evaluation_agent]
    column_mapping_agent.handoffs = [column_mapping_evaluation_agent]
    data_integration_agent.handoffs = [data_integration_evaluation_agent]
    
    # Each evaluator hands back to orchestrator
    data_prep_evaluation_agent.handoffs = [orchestrator]
    column_mapping_evaluation_agent.handoffs = [orchestrator]
    data_integration_evaluation_agent.handoffs = [orchestrator]

    workflow_orchestrator_agent = orchestrator
    return orchestrator


