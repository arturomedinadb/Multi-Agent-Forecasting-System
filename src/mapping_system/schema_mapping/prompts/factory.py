import os
from .renderer import PromptRenderer

def get_renderer() -> PromptRenderer:
    # In a later batch we can read PROMPTS_VERSION to switch folders (v1, v1_1, etc.)
    return PromptRenderer(
        templates_root=os.getenv("PROMPTS_TEMPLATES_ROOT", "templates"),
        registry_path=os.getenv("PROMPTS_REGISTRY_PATH", "templates/prompts/registry.yaml"),
    )
