"""OpenRouter provider tests - fully offline via httpx.MockTransport.

No API key, no internet, no real network requests are ever made.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from app.config.settings import load_settings
from app.llm.base import LLMError, LLMOutputInvalidError, LLMTimeoutError
from app.llm.openrouter import OpenRouterProvider
from app.llm.schemas import ToolDecision

TEST_KEY = "sk-orbit-secret-test-key-do-not-leak"
BASE_URL = "https://openrouter.ai/api/v1"

Handler = Callable[[httpx.Request], httpx.Response]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def chat(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "id": "gen-1",
            "choices": [{"message": {"role": "assistant", "content": content}}],
        },
    )


def empty_choices() -> httpx.Response:
    return httpx.Response(200, json={"id": "gen-1", "choices": []})


def scripted_handler(
    responses: list[httpx.Response | Exception], captured: list[httpx.Request]
) -> Handler:
    """Return a handler replaying ``responses`` (last one repeats)."""
    state = {"index": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        idx = min(state["index"], len(responses) - 1)
        state["index"] += 1
        result = responses[idx]
        if isinstance(result, Exception):
            raise result
        return result

    return handler


@asynccontextmanager
async def provider_scope(
    handler: Handler,
    *,
    max_retries: int = 2,
    backoff_s: float = 0.0,
    api_key: str = TEST_KEY,
    base_url: str = BASE_URL,
) -> Iterator[OpenRouterProvider]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        yield OpenRouterProvider(
            api_key=api_key,
            model="nvidia/nemotron-3-ultra",
            base_url=base_url,
            max_retries=max_retries,
            backoff_s=backoff_s,
            client=client,
        )
    finally:
        await client.aclose()


def valid_decision_content() -> str:
    return json.dumps(
        {
            "tool_name": "calculator",
            "arguments": {"expression": "2+2"},
            "reason_summary": "Recompute the estimated total.",
        }
    )


def body_of(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


async def test_request_construction() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler([chat(valid_decision_content())], captured)

    async with provider_scope(handler) as provider:
        result = await provider.complete(
            "Choose a tool for the task.",
            schema=ToolDecision,
            system="Be terse.",
            temperature=0.1,
            timeout_s=5.0,
        )

    assert isinstance(result, ToolDecision)
    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert request.headers["content-type"].startswith("application/json")

    payload = body_of(request)
    assert payload["model"] == "nvidia/nemotron-3-ultra"
    assert payload["temperature"] == 0.1

    # Structured output requested with the exact caller schema.
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "ToolDecision"
    assert response_format["json_schema"]["schema"] == ToolDecision.model_json_schema()

    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("Be terse.")
    assert "Required JSON schema" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Choose a tool for the task."}


# ---------------------------------------------------------------------------
# Valid / malformed structured output
# ---------------------------------------------------------------------------


async def test_valid_structured_response_is_validated() -> None:
    handler = scripted_handler([chat(valid_decision_content())], [])
    async with provider_scope(handler) as provider:
        result = await provider.complete("p", schema=ToolDecision)
    assert isinstance(result, ToolDecision)
    assert result.tool_name == "calculator"
    assert result.arguments == {"expression": "2+2"}
    assert provider.calls[0].schema_name == "ToolDecision"
    assert provider.calls[0].repaired is False
    assert provider.calls[0].requests_made == 1


async def test_fenced_json_is_accepted() -> None:
    fenced = f"```json\n{valid_decision_content()}\n```"
    handler = scripted_handler([chat(fenced)], [])
    async with provider_scope(handler) as provider:
        result = await provider.complete("p", schema=ToolDecision)
    assert isinstance(result, ToolDecision)


async def test_malformed_output_triggers_exactly_one_repair() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [chat("this is definitely not json {"), chat(valid_decision_content())],
        captured,
    )
    async with provider_scope(handler) as provider:
        result = await provider.complete("p", schema=ToolDecision)

    assert isinstance(result, ToolDecision)
    assert len(captured) == 2  # original + exactly one repair
    repair_payload = body_of(captured[1])
    roles = [m["role"] for m in repair_payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "could not be validated" in repair_payload["messages"][3]["content"]
    assert provider.calls[0].repaired is True
    assert provider.calls[0].requests_made == 2


async def test_malformed_twice_raises_after_single_repair() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [chat("garbage {"), chat("still garbage {")], captured
    )
    async with provider_scope(handler) as provider:
        with pytest.raises(LLMOutputInvalidError) as exc_info:
            await provider.complete("p", schema=ToolDecision)

    assert len(captured) == 2  # never a second repair
    assert "repair attempt" in str(exc_info.value)


async def test_schema_mismatch_repairs_then_succeeds() -> None:
    missing_field = json.dumps(
        {"arguments": {}, "reason_summary": "no tool name present"}
    )
    handler = scripted_handler([chat(missing_field), chat(valid_decision_content())], [])
    async with provider_scope(handler) as provider:
        result = await provider.complete("p", schema=ToolDecision)
    assert isinstance(result, ToolDecision)


async def test_schema_mismatch_is_never_guessed_into_an_object() -> None:
    missing_field = json.dumps({"arguments": {}, "reason_summary": "no tool name"})
    handler = scripted_handler([chat(missing_field), chat(missing_field)], [])
    async with provider_scope(handler) as provider:
        with pytest.raises(LLMOutputInvalidError) as exc_info:
            await provider.complete("p", schema=ToolDecision)

    message = str(exc_info.value)
    assert "tool_name" in message  # names the missing field instead of inventing it


async def test_valid_json_wrong_shape_is_rejected() -> None:
    handler = scripted_handler([chat("42"), chat("[1, 2]")], [])
    async with provider_scope(handler) as provider:
        with pytest.raises(LLMOutputInvalidError):
            await provider.complete("p", schema=ToolDecision)


async def test_empty_choices_fail_with_one_repair() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler([empty_choices(), empty_choices()], captured)
    async with provider_scope(handler) as provider:
        with pytest.raises(LLMOutputInvalidError):
            await provider.complete("p", schema=ToolDecision)
    assert len(captured) == 2


# ---------------------------------------------------------------------------
# Hardening: strict parsing (markdown fences tolerated, prose never accepted)
# ---------------------------------------------------------------------------


def prose_wrapped_decision() -> str:
    """Valid JSON fragment surrounded by prose - must NOT be silently mined."""
    return f"Sure! Here is the JSON you asked for: {valid_decision_content()} Enjoy!"


async def test_prose_wrapped_json_is_repaired_rather_than_mined() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [chat(prose_wrapped_decision()), chat(valid_decision_content())],
        captured,
    )
    async with provider_scope(handler) as provider:
        result = await provider.complete("p", schema=ToolDecision)

    assert isinstance(result, ToolDecision)
    assert len(captured) == 2  # strict failure -> exactly one repair
    assert provider.calls[0].repaired is True


async def test_prose_wrapped_json_is_never_accepted_silently() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [chat(prose_wrapped_decision()), chat(prose_wrapped_decision())],
        captured,
    )
    async with provider_scope(handler) as provider:
        with pytest.raises(LLMOutputInvalidError):
            await provider.complete("p", schema=ToolDecision)

    assert len(captured) == 2  # one repair, then typed failure - no mining


async def test_fenced_json_must_contain_a_complete_document() -> None:
    incomplete = f"```json\n{valid_decision_content()[:-2]}\n```"
    captured: list[httpx.Request] = []
    handler = scripted_handler([chat(incomplete), chat(incomplete)], captured)
    async with provider_scope(handler) as provider:
        with pytest.raises(LLMOutputInvalidError):
            await provider.complete("p", schema=ToolDecision)
    assert len(captured) == 2


async def test_400_fallback_output_is_still_strictly_schema_validated() -> None:
    """After the response_format fallback, output must still pass Pydantic."""
    missing_field = json.dumps(
        {"arguments": {}, "reason_summary": "no tool name present"}
    )
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [
            httpx.Response(400, text="response_format not supported"),
            chat(missing_field),
            chat(missing_field),
        ],
        captured,
    )
    async with provider_scope(handler, max_retries=1) as provider:
        with pytest.raises(LLMOutputInvalidError) as exc_info:
            await provider.complete("p", schema=ToolDecision)

    assert len(captured) == 3  # 400 + fallback + exactly one repair
    assert "tool_name" in str(exc_info.value)  # names the missing field
    assert "response_format" not in body_of(captured[1])  # fallback payload
    assert "response_format" not in body_of(captured[2])  # repair payload


async def test_400_fallback_still_recovers_on_valid_output() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [
            httpx.Response(400, text="response_format not supported"),
            chat(f"```json\n{valid_decision_content()}\n```"),
        ],
        captured,
    )
    async with provider_scope(handler, max_retries=1) as provider:
        result = await provider.complete("p", schema=ToolDecision)

    assert isinstance(result, ToolDecision)
    assert len(captured) == 2
    assert provider.calls[0].repaired is False
    assert provider.calls[0].requests_made == 2


# ---------------------------------------------------------------------------
# Timeouts, API errors, retries
# ---------------------------------------------------------------------------


async def test_timeout_retries_are_bounded() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"),
         httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")],
        captured,
    )
    async with provider_scope(handler, max_retries=2) as provider:
        with pytest.raises(LLMTimeoutError):
            await provider.complete("p", schema=ToolDecision, timeout_s=1.0)
    assert len(captured) == 3  # 1 + max_retries, never infinite


async def test_server_error_retries_then_succeeds() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [httpx.Response(500, text="server exploded"), chat(valid_decision_content())],
        captured,
    )
    async with provider_scope(handler, max_retries=2) as provider:
        result = await provider.complete("p", schema=ToolDecision)
    assert isinstance(result, ToolDecision)
    assert len(captured) == 2


async def test_server_error_exhausts_retry_limit() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler([httpx.Response(503, text="unavailable")], captured)
    async with provider_scope(handler, max_retries=1) as provider:
        with pytest.raises(LLMError) as exc_info:
            await provider.complete("p", schema=ToolDecision)
    assert len(captured) == 2
    assert "503" in str(exc_info.value)


async def test_auth_error_fails_fast_without_retry() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler([httpx.Response(401, text="unauthorized")], captured)
    async with provider_scope(handler, max_retries=3) as provider:
        with pytest.raises(LLMError) as exc_info:
            await provider.complete("p", schema=ToolDecision)
    assert len(captured) == 1
    assert "401" in str(exc_info.value)


async def test_api_error_body_never_leaks_the_api_key() -> None:
    handler = scripted_handler(
        [httpx.Response(404, text=f"bad credentials for {TEST_KEY}")], []
    )
    async with provider_scope(handler) as provider:
        with pytest.raises(LLMError) as exc_info:
            await provider.complete("p", schema=ToolDecision)
    assert TEST_KEY not in str(exc_info.value)
    assert "REDACTED" in str(exc_info.value)


async def test_json_schema_fallback_on_400() -> None:
    captured: list[httpx.Request] = []
    handler = scripted_handler(
        [httpx.Response(400, text="response_format not supported"),
         chat(valid_decision_content())],
        captured,
    )
    async with provider_scope(handler, max_retries=1) as provider:
        result = await provider.complete("p", schema=ToolDecision)

    assert isinstance(result, ToolDecision)
    assert len(captured) == 2
    first = body_of(captured[0])
    second = body_of(captured[1])
    assert "response_format" in first
    assert "response_format" not in second  # fell back to plain JSON


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------


def test_api_key_is_never_exposed_in_repr() -> None:
    provider = OpenRouterProvider(api_key=TEST_KEY)
    assert TEST_KEY not in repr(provider)
    assert "REDACTED" in repr(provider)


def test_missing_api_key_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(api_key="   ")


def test_from_settings_requires_a_key_but_does_not_echo_it() -> None:
    with pytest.raises(ValueError) as exc_info:
        OpenRouterProvider.from_settings(load_settings({}))
    assert "not configured" in str(exc_info.value)


def test_from_settings_uses_configuration() -> None:
    settings = load_settings(
        {
            "OPENROUTER_API_KEY": TEST_KEY,
            "OPENROUTER_MODEL": "meta-llama/llama-3.1-70b-instruct",
            "LLM_MAX_RETRIES": "4",
            "LLM_BACKOFF_S": "0.01",
        }
    )
    provider = OpenRouterProvider.from_settings(settings)
    assert provider.model_id == "meta-llama/llama-3.1-70b-instruct"
    assert TEST_KEY not in repr(provider)


async def test_recorded_calls_never_contain_the_api_key() -> None:
    handler = scripted_handler([chat(valid_decision_content())], [])
    async with provider_scope(handler) as provider:
        await provider.complete("p", schema=ToolDecision, system="sys")
    record = provider.calls[0]
    assert record.prompt == "p"
    assert record.system == "sys"
    assert record.temperature == 0.2
    assert record.timeout_s == 30.0
    assert TEST_KEY not in repr(record)


async def test_owned_client_lifecycle() -> None:
    provider = OpenRouterProvider(api_key=TEST_KEY)  # owns its own client
    await provider.aclose()  # must not raise


# ---------------------------------------------------------------------------
# Default model expectation
# ---------------------------------------------------------------------------


def test_default_model_is_nemotron_ultra() -> None:
    provider = OpenRouterProvider(api_key=TEST_KEY)
    assert provider.model_id == "nvidia/nemotron-3-ultra"
