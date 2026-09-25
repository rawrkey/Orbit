"""LLM provider abstraction.

Agent code depends only on the ``LLMProvider`` protocol. Phase 1 provides the
contract plus a deterministic ``MockProvider`` for tests. The real OpenRouter
implementation (Nemotron 3 Ultra) arrives in a later phase - other OpenRouter
models or local models can be swapped in without changing agent code.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

__all__ = [
    "LLMError",
    "LLMOutputInvalidError",
    "LLMProvider",
    "LLMTimeoutError",
]

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for LLM provider failures."""


class LLMTimeoutError(LLMError):
    """The provider did not return a valid result within the timeout."""


class LLMOutputInvalidError(LLMError):
    """The provider returned output that failed structured-output validation."""


@runtime_checkable
class LLMProvider(Protocol):
    """Structured-output LLM interface.

    Every call declares the Pydantic schema it must produce. Implementations
    must raise ``LLMOutputInvalidError`` rather than returning free-form text,
    so orchestration never depends on parsing prose.
    """

    @property
    def model_id(self) -> str:
        """Identifier of the backing model (for events and diagnostics)."""
        ...

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
        - timeout -> ``LLMTimeoutError``
        - invalid/mismatched output -> ``LLMOutputInvalidError``
        """
        ...
