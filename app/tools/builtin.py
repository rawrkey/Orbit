"""Deterministic construction of the default tool registry.

Registration order is fixed so the registry is reproducible. Tools are
constructed from configuration (workspace, size limits, search provider)
- never from LLM output.
"""

from __future__ import annotations

from pathlib import Path

from app.config.settings import Settings
from app.tools.calculator import CalculatorTool
from app.tools.file_reader import FileReaderTool
from app.tools.file_writer import FileWriterTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import SearchProvider, WebSearchTool

__all__ = ["BUILTIN_TOOL_NAMES", "create_default_registry", "create_search_provider"]

BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "calculator",
    "file_reader",
    "file_writer",
    "web_search",
)


def create_search_provider(settings: Settings) -> SearchProvider | None:
    """Map ``WEB_SEARCH_PROVIDER`` to a real provider.

    Phase 2 supports only ``none`` (no provider). Unknown values fail
    explicitly instead of silently degrading to fake results.
    """
    name = (settings.web_search_provider or "none").strip().lower()
    if name in ("", "none"):
        return None
    raise ValueError(
        f"unknown WEB_SEARCH_PROVIDER {name!r}; Phase 2 supports: none "
        "(real providers arrive in a later phase)"
    )


def create_default_registry(settings: Settings) -> ToolRegistry:
    """Register the builtin tools according to ``settings``."""
    workspace = Path(settings.workspace_dir).expanduser()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileReaderTool(workspace, max_bytes=settings.max_file_bytes))
    registry.register(FileWriterTool(workspace, max_bytes=settings.max_file_bytes))
    registry.register(WebSearchTool(provider=create_search_provider(settings)))
    return registry
