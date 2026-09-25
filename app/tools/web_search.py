"""Web search tool: real interface, no fake results.

Phase 2 ships the contract and configuration only. When no search provider
is configured, the tool fails with ``WebSearchUnavailableError`` - it NEVER
fabricates results. Real providers (Brave, Serper, ...) plug into the
``SearchProvider`` protocol in a later phase.

Dependency note: free/public search libraries were deliberately not added to
keep the dependency budget at pydantic + httpx; a provider built on httpx
can be added later without new dependencies.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.tools.base import Tool, WebSearchUnavailableError

__all__ = [
    "SearchProvider",
    "SearchResult",
    "WebSearchInput",
    "WebSearchOutput",
    "WebSearchTool",
]


class WebSearchInput(BaseModel):
    """Arguments for the web search tool."""

    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    """One search result. Only produced by a real provider."""

    title: str
    url: str
    snippet: str = ""


class WebSearchOutput(BaseModel):
    """Structured result of a successful search."""

    query: str
    provider: str
    results: list[SearchResult] = Field(default_factory=list)


@runtime_checkable
class SearchProvider(Protocol):
    """Real search backend. Implementations must return genuine results."""

    @property
    def name(self) -> str:
        """Provider identifier reported in the tool output."""
        ...

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Run a real search or raise a typed tool error."""
        ...


class WebSearchTool(Tool):
    """Web search via a configured provider; unavailable when unconfigured."""

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web for a query and return real results with titles, "
        "URLs, and snippets. Requires a configured search provider; fails "
        "explicitly when none is configured."
    )
    input_schema: ClassVar[type[BaseModel]] = WebSearchInput
    output_schema: ClassVar[type[BaseModel]] = WebSearchOutput
    timeout_s: ClassVar[float] = 15.0
    max_retries: ClassVar[int] = 1
    retry_backoff_s: ClassVar[float] = 1.0

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> SearchProvider | None:
        return self._provider

    @property
    def available(self) -> bool:
        return self._provider is not None

    def execute(self, arguments: BaseModel) -> BaseModel:
        assert isinstance(arguments, WebSearchInput)
        if self._provider is None:
            raise WebSearchUnavailableError(
                "web search is not configured. Set WEB_SEARCH_PROVIDER to "
                "enable a real search provider (Phase 2 supports 'none'; "
                "real providers arrive in a later phase).",
                tool_name=self.name,
            )
        results = self._provider.search(arguments.query, arguments.max_results)
        return WebSearchOutput(
            query=arguments.query,
            provider=self._provider.name,
            results=results,
        )
