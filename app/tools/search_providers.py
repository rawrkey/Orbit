"""Real HTTP search provider: LangSearch Web Search API behind ``SearchProvider``.

Architecture (unchanged, only the leaf is different):

    LLM -> ToolDecision -> deterministic validation -> WebSearchTool
        -> SearchProvider -> [THIS MODULE] -> LangSearch HTTP API

This module is the ONLY place in ORBIT that performs an external search
request. The response is treated as UNTRUSTED DATA: it is parsed, bounded,
and normalized into ``SearchResult`` values (title/url/snippet/source) that
flow through the workflow as data. Nothing in a response can execute code,
alter state, invoke tools, or change configuration.

Provider choice: LangSearch exposes a documented JSON REST API (POST with a
``Authorization: Bearer`` header), works with the already-present httpx
dependency, offers a free plan suitable for this project, and needs no SDK,
browser automation, or HTML scraping. The endpoint is overridable for
testing via ``WEB_SEARCH_ENDPOINT``.

Request/response contract (only what ORBIT needs):

    POST {"query": "...", "count": N}
    -> {"data": {"webPages": {"value": [{"name", "url", "snippet", ...}]}}}

Optional API features (``contents.text``, ``freshness``, ``includeDomains``,
``excludeDomains``, full page ``text``) are deliberately NOT requested:
snippets are sufficient and page text must never bulk up workflow state.

Failure behavior is typed and bounded (no unbounded retries, no credentials
in messages): ``WebSearchTimeoutError``, ``WebSearchProviderError``,
``WebSearchAuthError``, ``WebSearchResponseError``. An empty but well-formed
result set is DATA, not an error - validation/replanning decides what to do
with missing evidence.
"""

from __future__ import annotations

import time

import httpx

from app.tools.base import (
    WebSearchAuthError,
    WebSearchProviderError,
    WebSearchResponseError,
    WebSearchTimeoutError,
)
from app.tools.web_search import (
    MAX_RESULT_SNIPPET_CHARS,
    MAX_RESULT_SOURCE_CHARS,
    MAX_RESULT_TITLE_CHARS,
    SearchResult,
)

__all__ = [
    "DEFAULT_BACKOFF_S",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "MAX_RESPONSE_BYTES",
    "MAX_RESULTS",
    "MAX_URL_CHARS",
    "LangSearchProvider",
]

DEFAULT_ENDPOINT = "https://api.langsearch.com/v1/web-search"
DEFAULT_TIMEOUT_S = 6.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_BACKOFF_S = 0.25
MAX_RESULTS = 10
MAX_URL_CHARS = 300
"""Hard bound on one result URL - stricter than the schema bound so long
LangSearch links never enter workflow state."""
MAX_RESPONSE_BYTES = 512_000
"""Hard cap on a provider response body; anything larger is rejected."""

_REDACTED = "***REDACTED***"


def _collapse(text: str, limit: int) -> str:
    """Whitespace-collapsed, control-character-free, length-bounded text."""
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    cleaned = " ".join(printable.split())
    return cleaned[:limit]


def _domain(url: str) -> str:
    """Host of ``url`` without userinfo, port, path, query, or fragment."""
    host = url.split("://", 1)[-1]
    host = host.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    return host.lower()[:MAX_RESULT_SOURCE_CHARS]


def _is_http_url(url: str) -> bool:
    """Only http(s) URLs without whitespace ever reach workflow state."""
    if not url.startswith(("http://", "https://")):
        return False
    return not any(character.isspace() for character in url)


class LangSearchProvider:
    """Synchronous ``SearchProvider`` over the LangSearch Web Search API.

    Constructed from configuration (never from LLM output) and executed by
    the deterministic tool layer inside its timeout budget.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        client: httpx.Client | None = None,
    ) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError("search provider API key must not be empty")
        url = (endpoint or "").strip() or DEFAULT_ENDPOINT
        if not url.startswith(("http://", "https://")):
            raise ValueError(
                "search endpoint must start with http:// or https://, "
                f"got {url!r}"
            )
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if backoff_s < 0:
            raise ValueError(f"backoff_s must be >= 0, got {backoff_s}")

        self._api_key = key
        self._endpoint = url
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._client = client if client is not None else httpx.Client()
        self._owns_client = client is None

    # ------------------------------------------------------------------
    # Identity / lifecycle
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "langsearch"

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def __repr__(self) -> str:
        # Never expose the API key.
        return (
            f"LangSearchProvider(endpoint={self._endpoint!r}, "
            f"api_key='{_REDACTED}')"
        )

    def close(self) -> None:
        """Close the underlying client when this provider owns it."""
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------
    # SearchProvider protocol
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Run one real search and return bounded, normalized results.

        Raises a typed ``ToolError`` on failure; never returns fabricated
        data. An empty list means the provider answered with no results.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            raise WebSearchResponseError("search query must not be empty")
        count = max(1, min(int(max_results), MAX_RESULTS))

        response = self._post(cleaned, count)
        payload = self._parse(response)
        return self._normalize(payload, count)

    # ------------------------------------------------------------------
    # HTTP (bounded attempts, sanitized errors)
    # ------------------------------------------------------------------

    def _post(self, query: str, count: int) -> httpx.Response:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "ORBIT/6.0 (+web-search-tool)",
        }
        body = {"query": query, "count": count}
        attempts = self._max_retries + 1
        last_error: WebSearchProviderError | None = None

        for attempt in range(attempts):
            try:
                response = self._client.post(
                    self._endpoint,
                    json=body,
                    headers=headers,
                    timeout=self._timeout_s,
                )
            except httpx.TimeoutException:
                last_error = WebSearchTimeoutError(
                    f"search provider timed out after {self._timeout_s}s"
                )
            except httpx.HTTPError as exc:
                last_error = WebSearchProviderError(
                    f"search provider transport failure: {type(exc).__name__}"
                )
            else:
                status = response.status_code
                if status in (401, 403):
                    # Credentials are wrong: retrying cannot help.
                    body_text = response.text[:200]
                    raise WebSearchAuthError(
                        self._sanitize(
                            f"search provider authentication failed "
                            f"(HTTP {status}): {body_text}"
                        )
                    )
                if status == 429 or status >= 500:
                    last_error = WebSearchProviderError(
                        self._sanitize(f"search provider error (HTTP {status})")
                    )
                elif status >= 400:
                    # Other 4xx: deterministic rejection, not transient.
                    body_text = response.text[:200]
                    raise WebSearchResponseError(
                        self._sanitize(
                            f"search provider rejected the request "
                            f"(HTTP {status}): {body_text}"
                        )
                    )
                else:
                    return response

            if attempt < attempts - 1 and self._backoff_s > 0:
                time.sleep(self._backoff_s * (2**attempt))

        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Parsing and normalization (untrusted data -> bounded results)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(response: httpx.Response) -> dict:
        body = response.content
        if len(body) == 0:
            raise WebSearchResponseError("search provider returned an empty body")
        if len(body) > MAX_RESPONSE_BYTES:
            raise WebSearchResponseError(
                f"search provider response exceeded {MAX_RESPONSE_BYTES} bytes"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise WebSearchResponseError(
                "search provider returned malformed JSON"
            ) from exc
        if not isinstance(data, dict):
            raise WebSearchResponseError(
                "search provider returned a non-object JSON payload"
            )
        if data.get("error"):
            raise WebSearchResponseError("search provider reported an error payload")
        return data

    def _sanitize(self, text: str) -> str:
        """Strip the API key from any text that may reach logs or errors."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, _REDACTED)
        return text

    def _normalize(self, payload: dict, count: int) -> list[SearchResult]:
        # LangSearch: results live at data.webPages.value. Optional fields
        # may be absent - a well-formed response without them is DATA.
        data = payload.get("data")
        if data is None:
            return []
        if not isinstance(data, dict):
            raise WebSearchResponseError("search provider payload has no data object")
        pages = data.get("webPages")
        if pages is None:
            return []
        if not isinstance(pages, dict):
            raise WebSearchResponseError(
                "search provider payload has no webPages object"
            )
        raw_results = pages.get("value")
        if raw_results is None:
            return []
        if not isinstance(raw_results, list):
            raise WebSearchResponseError(
                "search provider payload has a malformed results list"
            )

        results: list[SearchResult] = []
        for entry in raw_results:
            if len(results) >= count:
                break
            if not isinstance(entry, dict):
                continue
            # Sanitize BEFORE bounding so a redaction can never re-expand a
            # field past its limit. Credentials never reach workflow state.
            url = self._sanitize(str(entry.get("url") or "")).strip()
            if len(url) > MAX_URL_CHARS or not _is_http_url(url):
                continue
            source = _domain(url)
            title = _collapse(
                self._sanitize(str(entry.get("name") or "")),
                MAX_RESULT_TITLE_CHARS,
            )
            snippet = _collapse(
                self._sanitize(
                    str(entry.get("snippet") or entry.get("text") or "")
                ),
                MAX_RESULT_SNIPPET_CHARS,
            )
            if not title:
                title = source[:MAX_RESULT_TITLE_CHARS]
            if not title and not snippet:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=source,
                )
            )
        return results
