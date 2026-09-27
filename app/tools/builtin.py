"""Deterministic construction of the default tool registry.

Registration order is fixed so the registry is reproducible. Tools are
constructed from configuration (workspace, size limits, search provider)
- never from LLM output.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from app.config.settings import Settings
from app.tools.calculator import CalculatorTool
from app.tools.file_reader import FileReaderTool
from app.tools.file_writer import FileWriterTool
from app.tools.registry import ToolRegistry
from app.tools.search_providers import LangSearchProvider
from app.tools.web_search import SearchProvider, WebSearchTool

__all__ = [
    "BUILTIN_TOOL_NAMES",
    "SUPPORTED_SEARCH_PROVIDERS",
    "create_default_registry",
    "create_search_provider",
]

BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "calculator",
    "file_reader",
    "file_writer",
    "web_search",
)

SUPPORTED_SEARCH_PROVIDERS: tuple[str, ...] = ("none", "langsearch")


def create_search_provider(
    settings: Settings, *, client: httpx.Client | None = None
) -> SearchProvider | None:
    """Map ``WEB_SEARCH_PROVIDER`` to a real provider (or None, cleanly).

    Routing policy:
    - ``none`` / unset            -> None (offline behavior preserved)
    - a supported provider without credentials -> None (disabled, never faked)
    - a supported provider with credentials    -> real HTTP provider
    - anything else               -> explicit ``ValueError`` (never a silent
      downgrade to fake results)

    Credentials come only from configuration; they are never echoed back.
    ``client`` lets tests inject a mocked HTTP transport - the production
    path always uses a real ``httpx.Client``.
    """
    name = (settings.web_search_provider or "none").strip().lower()
    if name in ("", "none"):
        return None
    if name == "langsearch":
        api_key = settings.web_search_api_key
        if not api_key:
            # Missing credentials: the real provider stays disabled and the
            # tool keeps its explicit "not configured" failure. No faking.
            return None
        return LangSearchProvider(
            api_key=api_key,
            endpoint=settings.web_search_endpoint or "",
            client=client,
        )
    raise ValueError(
        f"unknown WEB_SEARCH_PROVIDER {name!r}; supported providers: "
        + ", ".join(SUPPORTED_SEARCH_PROVIDERS)
    )


def create_default_registry(
    settings: Settings, *, search_client: httpx.Client | None = None
) -> ToolRegistry:
    """Register the builtin tools according to ``settings``."""
    workspace = Path(settings.workspace_dir).expanduser()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileReaderTool(workspace, max_bytes=settings.max_file_bytes))
    registry.register(FileWriterTool(workspace, max_bytes=settings.max_file_bytes))
    registry.register(
        WebSearchTool(provider=create_search_provider(settings, client=search_client))
    )
    return registry
