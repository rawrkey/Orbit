"""Tool contract.

Tools are deterministic and have no knowledge of the LLM. The LLM only
*proposes* a tool (via a ``ToolDecision``); the executor runs an
already-approved call; the observer reads the result.

Phase 1 defines the interface only - concrete tools arrive later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

__all__ = [
    "DestinationExistsError",
    "FileMissingError",
    "FileTooLargeError",
    "MalformedToolInputError",
    "PathTraversalError",
    "Tool",
    "ToolError",
    "ToolExecutionError",
    "ToolSpec",
    "ToolTimeoutError",
    "WebSearchAuthError",
    "WebSearchProviderError",
    "WebSearchResponseError",
    "WebSearchTimeoutError",
    "WebSearchUnavailableError",
]


class ToolError(Exception):
    """Base class for tool-layer failures."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        self.tool_name = tool_name
        self.code = type(self).__name__
        super().__init__(message)


class ToolTimeoutError(ToolError):
    """The tool did not finish within its timeout."""


class ToolExecutionError(ToolError):
    """The tool failed while running, or violated its output contract."""


class MalformedToolInputError(ToolError):
    """The provided arguments failed the tool's input schema."""


class PathTraversalError(ToolError):
    """A file path tried to escape the configured workspace."""


class FileMissingError(ToolError):
    """The requested file does not exist inside the workspace."""


class FileTooLargeError(ToolError):
    """The file exceeds the configured size limit."""


class DestinationExistsError(ToolError):
    """A write would overwrite an existing file without permission."""


class WebSearchUnavailableError(ToolError):
    """Web search was requested but no search provider is configured."""


class WebSearchProviderError(ToolError):
    """The configured search provider failed (transport, rate limit, HTTP 5xx).

    Base class of the provider-specific failures below. Messages are always
    sanitized by the provider: they never contain credentials.
    """


class WebSearchTimeoutError(WebSearchProviderError):
    """The provider did not answer within the bounded HTTP timeout."""


class WebSearchAuthError(WebSearchProviderError):
    """The provider rejected the configured credentials (HTTP 401/403)."""


class WebSearchResponseError(WebSearchProviderError):
    """The provider returned a malformed, oversized, or rejected response."""


class ToolSpec(BaseModel):
    """Serializable description of a tool (for listing and diagnostics)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_s: float
    max_retries: int


class Tool(ABC):
    """Base class every tool must subclass.

    Declarations are class-level contract data:
    - ``name``            unique identifier used in tool decisions
    - ``description``     purpose shown to the tool-selection step
    - ``input_schema``    Pydantic model validating arguments
    - ``output_schema``   Pydantic model validating results
    - ``timeout_s``       execution budget enforced by the executor
    - ``max_retries``     retry attempts on transient failure
    - ``retry_backoff_s`` delay between retries
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    timeout_s: ClassVar[float] = 5.0
    max_retries: ClassVar[int] = 0
    retry_backoff_s: ClassVar[float] = 0.5

    def validate_input(self, raw_arguments: Mapping[str, Any]) -> BaseModel:
        """Validate arguments against ``input_schema`` or fail explicitly."""
        try:
            return self.input_schema.model_validate(dict(raw_arguments))
        except ValidationError as exc:
            raise MalformedToolInputError(
                f"invalid arguments for tool {self.name!r}: "
                f"{len(exc.errors())} validation error(s)",
                tool_name=self.name,
            ) from exc

    @abstractmethod
    def execute(self, arguments: BaseModel) -> BaseModel:
        """Run the tool with already-validated arguments.

        Implementations should raise ``ToolError`` subclasses on failure.
        """

    def run(self, raw_arguments: Mapping[str, Any]) -> BaseModel:
        """Validate input, execute, then enforce the output contract."""
        arguments = self.validate_input(raw_arguments)
        result = self.execute(arguments)

        if isinstance(result, self.output_schema):
            return result
        if isinstance(result, BaseModel):
            data: Any = result.model_dump()
        elif isinstance(result, Mapping):
            data = dict(result)
        else:
            raise ToolExecutionError(
                f"tool {self.name!r} returned {type(result).__name__}, "
                f"expected {self.output_schema.__name__}",
                tool_name=self.name,
            )
        try:
            return self.output_schema.model_validate(data)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"tool {self.name!r} violated its output contract: "
                f"{len(exc.errors())} validation error(s)",
                tool_name=self.name,
            ) from exc

    def spec(self) -> ToolSpec:
        """Serializable spec of this tool's contract."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema.model_json_schema(),
            output_schema=self.output_schema.model_json_schema(),
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
