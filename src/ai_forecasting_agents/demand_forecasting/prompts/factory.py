"""Factory for creating a configured PromptRenderer for demand_forecasting agents.

Reuses the generic PromptRenderer from schema_mapping.prompts, pointed at
demand_forecasting's own templates_root/registry_path.
"""
from pathlib import Path
from schema_mapping.prompts.renderer import PromptRenderer

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def get_renderer() -> PromptRenderer:
    """Return a configured PromptRenderer using demand_forecasting's prompt templates."""
    return PromptRenderer(
        templates_root=PROJECT_ROOT / "prompts" / "demand_forecasting",
        registry_path=PROJECT_ROOT / "prompts" / "demand_forecasting" / "registry.yaml",
    )
