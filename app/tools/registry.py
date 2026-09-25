"""Deterministic tool registry.

Registers tools, resolves them by name, and reports availability. Unknown or
duplicate tools raise typed errors. The registry never calls an LLM.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.tools.base import Tool, ToolSpec

__all__ = [
    "DuplicateToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolRegistry",
]


class ToolRegistrationError(Exception):
    """A tool was submitted with an invalid contract (e.g. empty name)."""


class DuplicateToolError(Exception):
    """A tool with the same name is already registered."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"tool already registered: {name}")


class ToolNotFoundError(Exception):
    """No tool is registered under the requested name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unknown tool: {name}")


class ToolRegistry:
    """Name -> tool lookup with explicit failure modes."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register ``tool`` after checking its declared contract."""
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ToolRegistrationError(
                f"tool {type(tool).__name__} must declare a non-empty string 'name'"
            )
        if not isinstance(getattr(tool, "description", None), str) or not tool.description.strip():
            raise ToolRegistrationError(
                f"tool {name!r} must declare a non-empty string 'description'"
            )
        input_schema = getattr(tool, "input_schema", None)
        output_schema = getattr(tool, "output_schema", None)
        if not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
            raise ToolRegistrationError(
                f"tool {name!r} must declare an input_schema Pydantic model"
            )
        if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise ToolRegistrationError(
                f"tool {name!r} must declare an output_schema Pydantic model"
            )
        if name in self._tools:
            raise DuplicateToolError(name)
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name`` or raise."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[ToolSpec]:
        """Serializable specs of all registered tools, sorted by name."""
        return [self._tools[name].spec() for name in sorted(self._tools)]

    def __len__(self) -> int:
        return len(self._tools)
