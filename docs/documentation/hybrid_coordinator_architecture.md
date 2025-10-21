# Hybrid Coordinator Architecture Plan

This document outlines how to evolve the schema-mapping workflow into a hybrid system that combines deterministic stage guarantees with an autonomous coordinator agent. It captures the agent hierarchy, control loop, evaluation flows, and integration points with existing code.

## High-Level Goals
- Preserve the current ordered stages (DataPrep → Mapping → Integration → Evaluation) so mandatory work always completes with the same guardrails.
- Introduce a **Coordinator Agent** that reasons about run state, decides when to advance or retry stages, and ensures evaluations pass before continuing.
- Pair each stage with an **Evaluator Subagent** that validates its output and reports structured pass/fail signals.
- Conclude runs with a **Final Judge Agent** that executes DeepEval metrics and synthesizes improvement guidance.

## Agent Stack

| Layer | Responsibility | Tools Invoked | Notes |
|-------|----------------|---------------|-------|
| Coordinator Agent | Orchestrates stage execution, tracks artifacts, polices retry budgets | Stage wrapper tools (`run_dataprep`, `run_mapping`, …) and evaluator tools | Emits JSON directives (`{"action": "RUN_MAPPING"}`) so Python control loop can dispatch deterministically |
| Stage Worker Agents | Produce artifacts for their respective stage (existing four agents) | `load_and_describe_dataset`, `generate_mapped_csvs`, `merge_mapped_csvs_to_target`, `evaluate_schema_mapping_with_deepeval` | Continue to use Jinja templates with deterministic prompts |
| Stage Evaluator Subagents | Validate outputs immediately after each stage, return structured verdicts | New tools per stage (e.g., `evaluate_dataprep_metadata`, `audit_mapping_confidence`) | Keep evaluations modular and reusable; coordinator must call them before advancing |
| Final Judge Agent | Runs DeepEval suite and optional LLM critique | Existing evaluation harness via a tool wrapper | Produces final metrics, improvement prompt, and signals success/failure of the run |

## Control Flow Diagram

```mermaid
flowchart TD
    Start([Start Run]) --> CoordinatorLoop{Coordinator decides action}

    CoordinatorLoop -->|RUN_DATAPREP| DPTool[Call deterministic\nDataPrep wrapper]
    DPTool --> DPAgent[DataPrep Agent]
    DPAgent --> DPArtifacts[(Metadata JSON)]
    DPArtifacts --> DPEvalTool[Call DataPrep Evaluator tool]
    DPEvalTool --> DPEvalAgent[DataPrep Evaluator Agent]
    DPEvalAgent --> DPResult{{Stage pass?}}
    DPResult -->|no & retries left| CoordinatorLoop
    DPResult -->|yes| CoordinatorLoop

    CoordinatorLoop -->|RUN_MAPPING| MapTool[Call deterministic\nMapping wrapper]
    MapTool --> MapAgent[Column Mapping Agent]
    MapAgent --> MapArtifacts[(Plan & Manifest)]
    MapArtifacts --> MapEvalTool[Call Mapping Evaluator tool]
    MapEvalTool --> MapEvalAgent[Mapping Evaluator Agent]
    MapEvalAgent --> MapResult{{Stage pass?}}
    MapResult -->|no & retries left| CoordinatorLoop
    MapResult -->|yes| CoordinatorLoop

    CoordinatorLoop -->|RUN_INTEGRATION| IntTool[Call deterministic\nIntegration wrapper]
    IntTool --> IntAgent[Integration Agent]
    IntAgent --> IntArtifact[(Final CSV)]
    IntArtifact --> IntEvalTool[Call Integration Evaluator tool]
    IntEvalTool --> IntEvalAgent[Integration Evaluator Agent]
    IntEvalAgent --> IntResult{{Stage pass?}}
    IntResult -->|no & retries left| CoordinatorLoop
    IntResult -->|yes| CoordinatorLoop

    CoordinatorLoop -->|RUN_EVALUATION| EvalTool[Call deterministic\nWorkflow evaluation wrapper]
    EvalTool --> EvalAgent[Evaluation Agent]
    EvalAgent --> EvalArtifact[(Deterministic metrics)]
    EvalArtifact --> FinalJudge[Final Judge Agent\n(DeepEval + LLM)]
    FinalJudge --> FinalResult{{All metrics pass?}}
    FinalResult -->|no & retries left| CoordinatorLoop
    FinalResult -->|yes| Finish([Finish Run])

    CoordinatorLoop -->|STOP| Finish
```

## Implementation Phases

1. **Extract deterministic stage wrappers**  
   - Lift `_run_dataprep`, `_run_mapping`, `_run_integration`, and `_run_evaluation` from the current orchestrator into standalone async functions so they can be exposed as tool endpoints without losing validation (`orchestrator.py:259`, `294`, `341`, `386`).
   - Standardize return payloads: `{"status": "success", "artifacts": {...}}`.

2. **Build evaluator tool layer**  
   - For each stage, add lightweight checks (e.g., validate metadata coverage, enforce mapping confidence thresholds, ensure integrated CSV schema alignment) and wrap them with `@function_tool`.
   - Provide evaluator agents with prompts that focus on deterministic pass/fail decisions and concise remediation suggestions.

3. **Define the Coordinator Agent prompt**  
   - Document the canonical order, permitted actions (`RUN_DATAPREP`, `RUN_MAPPING`, `RUN_INTEGRATION`, `RUN_EVALUATION`, `RETRY_STAGE`, `STOP`, `FINISH`), retry policies, and required evaluation calls after each stage.
   - Require responses in a two-part structure: natural-language reasoning followed by a JSON directive describing the next action and its inputs.

4. **Create the coordinator loop**  
   - Implement an async loop in Python that sends the coordinator the current state (completed stages, last evaluation result, retry counts).
   - Parse the agent’s directive JSON; call the matching tool; automatically trigger the paired evaluator tool; update state; enforce retry ceilings; and continue until the agent emits `FINISH` or `STOP`.

5. **Integrate the Final Judge Agent**  
   - Wrap `run_schema_mapping_evaluation` (`deepeval_runner.py:206`) in a tool that returns deterministic metrics and markdown paths.
   - Prompt the judge agent to interpret metrics, call optional LLM judge metrics, and feed improvement prompts back to the coordinator for possible retries.

6. **Testing & validation**  
   - Unit-test each tool wrapper and evaluator to ensure they reject malformed data and respect schema constraints.
   - Add orchestrator loop tests that stub coordinator directives to verify required order is enforced and retries are bounded.
   - Create integration scenarios where evaluator failures trigger retries, ensuring the loop eventually converges or stops with clear error states.

7. **Documentation & observability**  
   - Update user-facing docs (README, run guides) to explain the agent hierarchy, coordinator behavior, and evaluation checkpoints.
   - Log coordinator directives, tool invocations, and evaluation summaries so debugging multi-step runs remains practical.

## Key Design Principles
- **Deterministic Guardrails First**: All file I/O and schema validation remain in Python wrappers to prevent the agent from bypassing required checks.
- **Structured Communication**: Agents exchange JSON-formatted directives and results, eliminating fragile text parsing and enabling strict validation.
- **Bounded Autonomy**: Retry counts, mandatory evaluation steps, and action whitelists keep the coordinator from deviating from the required pipeline.
- **Modular Evaluators**: Stage-specific evaluators allow targeted quality checks and reusable feedback loops without coupling everything to the final judge.
- **Transparent Reasoning**: Coordinator outputs include explicit rationales so operators can audit why a run advanced, retried, or stopped.

With this architecture, you retain the reliability of the existing orchestrator while empowering an agentic coordinator to manage retries, reason about evaluation outcomes, and integrate stage-specific and end-to-end quality gates.
