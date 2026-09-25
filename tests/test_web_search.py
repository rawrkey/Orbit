"""Web search tool tests: real interface, explicit unavailability, no fakes."""

from __future__ import annotations

import pytest

from app.config.settings import load_settings
from app.tools.base import WebSearchUnavailableError
from app.tools.builtin import create_search_provider
from app.tools.web_search import SearchResult, WebSearchTool


class FakeProvider:
    """Test double standing in for a REAL provider implementation.

    Note: production code paths never fabricate results - an unconfigured
    tool raises instead (asserted below).
    """

    @property
    def name(self) -> str:
        return "test-provider"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"Result for {query}",
                url="https://example.com/1",
                snippet="snippet",
            )
        ][:max_results]


def test_unconfigured_tool_fails_explicitly() -> None:
    tool = WebSearchTool()
    assert tool.available is False
    with pytest.raises(WebSearchUnavailableError) as exc_info:
        tool.run({"query": "mumbai hostels"})
    assert "WEB_SEARCH_PROVIDER" in str(exc_info.value)
    assert "not configured" in str(exc_info.value)


def test_unconfigured_tool_never_returns_fake_results() -> None:
    tool = WebSearchTool()
    with pytest.raises(WebSearchUnavailableError):
        tool.run({"query": "anything"})
    # The only way to get results is a real provider.


def test_configured_tool_returns_provider_results() -> None:
    tool = WebSearchTool(provider=FakeProvider())
    assert tool.available is True
    output = tool.run({"query": "hostels in mumbai", "max_results": 3})
    assert output.provider == "test-provider"
    assert len(output.results) == 1
    assert output.results[0].url == "https://example.com/1"


def test_result_counts_are_bounded_by_schema() -> None:
    tool = WebSearchTool(provider=FakeProvider())
    with pytest.raises(Exception):
        tool.run({"query": "x", "max_results": 50})  # > le=10
    with pytest.raises(Exception):
        tool.run({"query": "x", "max_results": 0})  # < ge=1


def test_search_provider_configuration_maps_to_none_by_default() -> None:
    settings = load_settings({})
    assert settings.web_search_provider == "none"
    assert create_search_provider(settings) is None


def test_unknown_search_provider_fails_explicitly() -> None:
    settings = load_settings({"WEB_SEARCH_PROVIDER": "some-paid-api"})
    with pytest.raises(ValueError, match="WEB_SEARCH_PROVIDER"):
        create_search_provider(settings)
