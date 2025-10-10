from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound
import yaml


# ---- Jinja environment & custom filters ----------------------------------------------------------

def _tojson_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

def _tojson_pretty(value: Any, indent: int = 2) -> str:
    return json.dumps(value, indent=indent, ensure_ascii=False)

def _ensure_triple_backticks(value: str, lang: str = "json") -> str:
    v = value.strip()
    if v.startswith("```"):
        return v
    return f"```{lang}\n{v}\n```"

def _reject_none(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"Missing required variable '{name}' (got None)")
    return value

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- Registry loading ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemplateSpec:
    path: str
    required_vars: Iterable[str]


class Registry:
    def __init__(self, registry_path: str | Path):
        self._path = Path(registry_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Prompt registry not found: {self._path}")
        self._data = yaml.safe_load(self._path.read_text()) or {}

    def get(self, key: str) -> TemplateSpec:
        entry = self._data.get(key)
        if not entry:
            raise KeyError(f"Prompt '{key}' not found in registry {self._path}")
        path = entry.get("template")
        required = entry.get("required_vars", [])
        if not isinstance(required, (list, tuple)):
            raise ValueError(f"Invalid required_vars for '{key}' in {self._path}")
        return TemplateSpec(path=path, required_vars=tuple(required))


# ---- Renderer ------------------------------------------------------------------------------------

class PromptRenderer:
    """
    Central entrypoint to render prompts by logical name.
    - StrictUndefined: any missing var raises an error
    - Whitespace controlled to keep diffs clean
    - Adds custom filters usable in templates
    """

    def __init__(
        self,
        *,
        templates_root: str | Path = "templates",
        registry_path: str | Path = "prompts/registry.yaml",
    ) -> None:
        self._templates_root = Path(templates_root)
        self._env = Environment(
            loader=FileSystemLoader(str(self._templates_root)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,  # prompts are plain text; be explicit if you escape
        )
        # filters
        self._env.filters["tojson_compact"] = _tojson_compact
        self._env.filters["tojson_pretty"] = _tojson_pretty
        self._env.filters["ensure_triple_backticks"] = _ensure_triple_backticks
        self._env.filters["reject_none"] = _reject_none

        self._registry = Registry(registry_path)

    def render(self, prompt_key: str, **kwargs: Any) -> str:
        spec = self._registry.get(prompt_key)
        # validate required vars presence (exist in kwargs, non-None)
        for name in spec.required_vars:
            if name not in kwargs:
                raise ValueError(f"Missing required variable '{name}' for prompt '{prompt_key}'")
            _reject_none(kwargs[name], name)

        try:
            template = self._env.get_template(Path(spec.path).relative_to(self._templates_root).as_posix())
        except (TemplateNotFound, ValueError):
            # Allow absolute or relative (from templates root) paths in registry
            template = self._env.get_template(spec.path)

        rendered = template.render(**kwargs)
        # useful for logging/handoffs
        sha = _sha256(rendered)
        return rendered  # caller can log the hash if desired

