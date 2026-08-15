"""Factory for creating configured PromptRenderer instances."""
from pathlib import Path
from .renderer import PromptRenderer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_renderer() -> PromptRenderer:
    """Return a configured PromptRenderer using project defaults."""
    return PromptRenderer(
        templates_root=PROJECT_ROOT / "prompts" / "schema_mapping",
        registry_path=PROJECT_ROOT / "prompts" / "schema_mapping" / "registry.yaml",
    )

