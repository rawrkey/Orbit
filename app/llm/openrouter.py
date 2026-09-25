"""OpenRouter LLM provider (Nemotron 3 Ultra by default).

Implements the ``LLMProvider`` protocol over the OpenRouter chat-completions
API using httpx. Design rules:

- Structured output: requests ``response_format`` json_schema when supported,
  parses JSON, and validates against the caller's Pydantic schema.
- At most ONE repair attempt for malformed output, then
  ``LLMOutputInvalidError``. Never guesses missing fields.
- Bounded timeout, retries, and backoff - never infinite.
- The API key is never logged, never placed in events, and never allowed to
  appear in exception messages (all outbound text is sanitized).

Per-call ``temperature`` / ``timeout_s`` are supplied by the caller (the
future executor passes ``settings.llm_temperature`` / ``settings.llm_timeout_s``);
construction-time configuration comes from ``from_settings``.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.config.settings import Settings
from app.llm.base import LLMError, LLMOutputInvalidError, LLMTimeoutError, T

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "OpenRouterCallRecord",
    "OpenRouterProvider",
]

DEFAULT_MODEL = "nvidia/nemotron-3-ultra"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are ORBIT, an adaptive personal workflow agent. "
    "Respond with a single JSON object that validates against the required "
    "schema. Do not include markdown, code fences, or explanations."
)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
_REDACTED = "***REDACTED***"


@dataclass(frozen=True, slots=True)
class OpenRouterCallRecord:
    """Record of one ``complete`` call. Contains no credentials."""

    schema_name: str
    prompt: str
    system: str | None
    temperature: float
    timeout_s: float
    requests_made: int
    repaired: bool


class OpenRouterProvider:
    """Async OpenRouter client satisfying the ``LLMProvider`` protocol."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError("OPENROUTER_API_KEY must not be empty")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if backoff_s < 0:
            raise ValueError(f"backoff_s must be >= 0, got {backoff_s}")

        self._api_key = key
        self._model = model.strip() or DEFAULT_MODEL
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._send_json_schema = True
        self._client = client if client is not None else httpx.AsyncClient()
        self._owns_client = client is None
        self.calls: list[OpenRouterCallRecord] = []

    # ------------------------------------------------------------------
    # Identity / lifecycle
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model

    def __repr__(self) -> str:
        # Never expose the API key.
        return f"OpenRouterProvider(model={self._model!r}, api_key='{_REDACTED}')"

    async def aclose(self) -> None:
        """Close the underlying client when this provider owns it."""
        if self._owns_client:
            await self._client.aclose()

    @classmethod
    def from_settings(
        cls, settings: Settings, *, client: httpx.AsyncClient | None = None
    ) -> OpenRouterProvider:
        """Build a provider from configuration; fails explicitly if the key
        is missing (the key itself is never echoed back)."""
        key = settings.openrouter_api_key
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not configured")
        return cls(
            api_key=key,
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            max_retries=settings.llm_max_retries,
            backoff_s=settings.llm_backoff_s,
            client=client,
        )

    # ------------------------------------------------------------------
    # LLMProvider protocol
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
    ) -> T:
        """Return a validated instance of ``schema``.

        Failure behavior:
        - timeout / transport failure -> ``LLMTimeoutError`` / ``LLMError``
          (bounded retries with backoff)
        - malformed or mismatched structured output -> ONE repair attempt,
          then ``LLMOutputInvalidError``
        """
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")

        stats = {"requests": 0}
        base_system = system or DEFAULT_SYSTEM_PROMPT
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{base_system}\n\n"
                    f"Required JSON schema:\n{schema_json}"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        repaired = False
        try:
            raw: str | None = None
            try:
                raw = await self._fetch(messages, schema, temperature, timeout_s, stats)
                return self._validate(raw, schema)
            except LLMOutputInvalidError as first_error:
                # Exactly one repair attempt.
                repaired = True
                repair_messages = list(messages)
                if raw is not None:
                    repair_messages.append({"role": "assistant", "content": raw})
                repair_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be validated "
                            f"against schema {schema.__name__}: "
                            f"{self._sanitize(str(first_error))}. "
                            "Respond again with ONLY a corrected JSON object."
                        ),
                    }
                )
                try:
                    raw2 = await self._fetch(
                        repair_messages, schema, temperature, timeout_s, stats
                    )
                    return self._validate(raw2, schema)
                except LLMOutputInvalidError as second_error:
                    raise LLMOutputInvalidError(
                        f"{schema.__name__}: structured output remained invalid "
                        f"after one repair attempt: {self._sanitize(str(second_error))}"
                    ) from first_error
        finally:
            self.calls.append(
                OpenRouterCallRecord(
                    schema_name=schema.__name__,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    requests_made=stats["requests"],
                    repaired=repaired,
                )
            )

    # ------------------------------------------------------------------
    # Request pipeline
    # ------------------------------------------------------------------

    async def _fetch(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        temperature: float,
        timeout_s: float,
        stats: dict[str, int],
    ) -> str:
        payload = self._build_payload(messages, schema, temperature)
        response = await self._post(payload, timeout_s, stats)
        return self._content_from(response)

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if self._send_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": False,
                    "schema": schema.model_json_schema(),
                },
            }
        return payload

    async def _post(
        self,
        payload: dict[str, Any],
        timeout_s: float,
        stats: dict[str, int],
    ) -> httpx.Response:
        """POST with bounded retries/backoff. Raises typed LLM errors."""
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "ORBIT",
        }
        attempts = self._max_retries + 1
        last_error: LLMError | None = None

        for attempt in range(attempts):
            stats["requests"] += 1
            try:
                response = await self._client.post(
                    url, json=payload, headers=headers, timeout=timeout_s
                )
            except httpx.TimeoutException:
                last_error = LLMTimeoutError(
                    f"{self._model} request timed out after {timeout_s}s"
                )
            except httpx.HTTPError as exc:
                last_error = LLMError(
                    f"transport failure talking to OpenRouter: {type(exc).__name__}"
                )
            else:
                status = response.status_code
                if status == 400 and self._send_json_schema:
                    # Provider may not accept response_format: retry once
                    # without it (JSON is still validated against the schema).
                    self._send_json_schema = False
                    payload.pop("response_format", None)
                    continue
                if status in (401, 403):
                    raise LLMError(
                        self._sanitize(f"OpenRouter authentication failed (HTTP {status})")
                    )
                if status == 429 or status >= 500:
                    last_error = LLMError(
                        self._sanitize(f"OpenRouter API error (HTTP {status})")
                    )
                elif status >= 400:
                    body = response.text[:200]
                    raise LLMError(
                        self._sanitize(f"OpenRouter API error (HTTP {status}): {body}")
                    )
                else:
                    return response

            if attempt < attempts - 1 and self._backoff_s > 0:
                await asyncio.sleep(self._backoff_s * (2**attempt))

        if last_error is None:
            last_error = LLMError("OpenRouter request failed")
        raise last_error

    def _content_from(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMOutputInvalidError("model response was not valid JSON") from exc

        if not isinstance(data, dict):
            raise LLMOutputInvalidError("model response was not a JSON object")

        if data.get("error"):
            raise LLMOutputInvalidError(
                self._sanitize(f"model returned an error payload: {str(data['error'])[:200]}")
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMOutputInvalidError("model response contained no choices")

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") or {}
        parsed = message.get("parsed")
        if isinstance(parsed, dict):
            return json.dumps(parsed)
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        raise LLMOutputInvalidError("model response content was empty")

    # ------------------------------------------------------------------
    # Parsing and validation (strict - never guesses)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(content: str) -> Any:
        """Parse model content STRICTLY.

        Tolerated: a markdown code fence wrapping an otherwise complete JSON
        document. NOT tolerated: arbitrary prose surrounding a JSON-looking
        fragment - that would silently accept malformed output as a valid
        decision. Anything that is not a single parseable JSON value after
        fence-stripping raises ``LLMOutputInvalidError`` (one repair attempt
        then a typed failure; never a silent best-effort object).
        """
        text = content.strip()
        fence = _FENCE_RE.match(text)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMOutputInvalidError(
                "model returned malformed JSON"
            ) from exc

    @classmethod
    def _validate(cls, content: str, schema: type[T]) -> T:
        data = cls._parse_json(content)
        if not isinstance(data, dict):
            raise LLMOutputInvalidError(
                f"expected a JSON object for {schema.__name__}, "
                f"got {type(data).__name__}"
            )
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            parts = []
            for error in exc.errors()[:3]:
                location = ".".join(str(part) for part in error["loc"]) or "(root)"
                parts.append(f"{location}: {error['msg']}")
            raise LLMOutputInvalidError(
                f"{schema.__name__} validation failed - " + "; ".join(parts)
            ) from exc

    # ------------------------------------------------------------------
    # Secret hygiene
    # ------------------------------------------------------------------

    def _sanitize(self, text: str) -> str:
        """Strip the API key from any text that may reach logs or errors."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, _REDACTED)
        return text
