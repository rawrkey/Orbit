"""Web search tool: real interface, no fake results.

The tool delegates to a configured ``SearchProvider``. Without a provider it
fails with ``WebSearchUnavailableError`` - it NEVER fabricates results.
Real providers (see ``app.tools.search_providers``) plug into the
``SearchProvider`` protocol and are routed from configuration in
``app.tools.builtin.create_search_provider``.

Dependency note: free/public search libraries were deliberately not added to
keep the dependency budget at pydantic + httpx; real providers are built on
httpx with no extra dependencies.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.tools.base import Tool, WebSearchUnavailableError

__all__ = [
    "MAX_RESULT_SNIPPET_CHARS",
    "MAX_RESULT_SOURCE_CHARS",
    "MAX_RESULT_TITLE_CHARS",
    "MAX_RESULT_URL_CHARS",
    "SearchProvider",
    "SearchResult",
    "WebSearchInput",
    "WebSearchOutput",
    "WebSearchTool",
]

MAX_RESULT_TITLE_CHARS = 200
MAX_RESULT_URL_CHARS = 500
MAX_RESULT_SNIPPET_CHARS = 300
MAX_RESULT_SOURCE_CHARS = 100
"""Hard bounds on one normalized result (defense in depth: the producing
provider truncates first, the schema guarantees it)."""


class WebSearchInput(BaseModel):
    """Arguments for the web search tool."""

    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    """One search result. Only produced by a real provider.

    Deliberately narrow: title/url/snippet/source are the ONLY fields that
    ever enter workflow state, so arbitrary provider JSON can never flow
    through the agent. Lengths are bounded by the producing provider.
    """

    title: str = Field(max_length=MAX_RESULT_TITLE_CHARS)
    url: str = Field(max_length=MAX_RESULT_URL_CHARS)
    snippet: str = Field(default="", max_length=MAX_RESULT_SNIPPET_CHARS)
    source: str = Field(default="", max_length=MAX_RESULT_SOURCE_CHARS)


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
                "web search is not configured. Set WEB_SEARCH_PROVIDER to a "
                "supported provider (e.g. 'langsearch') and "
                "LANGSEARCH_API_KEY to enable real search.",
                tool_name=self.name,
            )
        results = self._provider.search(arguments.query, arguments.max_results)
        return WebSearchOutput(
            query=arguments.query,
            provider=self._provider.name,
            results=results,
        )
