"""Deterministic mock LLM provider for offline tests.

Scripted responses let tests exercise the structured-output contract
(valid output, invalid output, mismatched schema) with no network access
and no API key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.base import LLMError, LLMOutputInvalidError, T

__all__ = ["LLMCallRecord", "MockProvider", "RoutedMockProvider"]


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """Record of one ``complete`` call, kept for assertions in tests."""

    prompt: str
    schema_name: str
    system: str | None
    temperature: float
    timeout_s: float


class MockProvider:
    """Scripted ``LLMProvider`` implementation.

    Each response is consumed in order. A response is either:
    - an instance of the requested schema (returned directly), or
    - a raw mapping (validated against the requested schema).

    Anything else raises ``LLMOutputInvalidError``; an exhausted script
    raises ``LLMError``.
    """

    def __init__(
        self,
        responses: Sequence[BaseModel | Mapping[str, Any]] = (),
        *,
        model_id: str = "mock/model",
    ) -> None:
        self._responses: list[BaseModel | Mapping[str, Any]] = list(responses)
        self._model_id = model_id
        self.calls: list[LLMCallRecord] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    async def complete(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
    ) -> T:
        self.calls.append(
            LLMCallRecord(
                prompt=prompt,
                schema_name=schema.__name__,
                system=system,
                temperature=temperature,
                timeout_s=timeout_s,
            )
        )
        if not self._responses:
            raise LLMError("MockProvider has no scripted responses left")

        response = self._responses.pop(0)
        return _resolve(response, schema)


class RoutedMockProvider:
    """Scripted provider that dispatches by the requested schema NAME.

    Stage call order can vary (resume, clarification, synthesis), so each
    schema keeps its own independent queue. An exhausted queue raises
    ``LLMError``; responses resolve exactly like ``MockProvider``.
    """

    def __init__(
        self,
        **queues: Sequence[BaseModel | Mapping[str, Any]],
    ) -> None:
        self._queues: dict[str, list[BaseModel | Mapping[str, Any]]] = {
            name: list(items) for name, items in queues.items()
        }
        self.calls: list[LLMCallRecord] = []

    @property
    def model_id(self) -> str:
        return "mock/routed"

    async def complete(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
    ) -> T:
        self.calls.append(
            LLMCallRecord(
                prompt=prompt,
                schema_name=schema.__name__,
                system=system,
                temperature=temperature,
                timeout_s=timeout_s,
            )
        )
        queue = self._queues.get(schema.__name__)
        if not queue:
            raise LLMError(f"no scripted {schema.__name__} responses left")
        response = queue.pop(0)
        return _resolve(response, schema)


def _resolve(response: BaseModel | Mapping[str, Any], schema: type[T]) -> T:
    """Resolve a scripted response against the requested schema."""
    if isinstance(response, schema):
        return response

    if isinstance(response, BaseModel):
        raise LLMOutputInvalidError(
            f"scripted response {type(response).__name__} does not match "
            f"requested schema {schema.__name__}"
        )

    if isinstance(response, Mapping):
        try:
            return schema.model_validate(dict(response))
        except ValidationError as exc:
            raise LLMOutputInvalidError(
                f"response failed validation for {schema.__name__}: "
                f"{len(exc.errors())} error(s)"
            ) from exc

    raise LLMOutputInvalidError(
        f"unsupported response type {type(response).__name__} for "
        f"schema {schema.__name__}"
    )
