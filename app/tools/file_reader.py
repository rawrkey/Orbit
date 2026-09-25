"""Workspace-scoped file reader.

Reads UTF-8 text files that live strictly inside the configured workspace.
Path traversal, absolute paths, and symlink escapes are rejected before any
filesystem access. Files larger than ``max_bytes`` are rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from app.tools.base import (
    FileMissingError,
    FileTooLargeError,
    Tool,
    ToolExecutionError,
)
from app.tools.path_safety import resolve_workspace_path, workspace_relative

__all__ = ["FileReaderInput", "FileReaderOutput", "FileReaderTool"]

_DEFAULT_MAX_BYTES = 1_000_000


class FileReaderInput(BaseModel):
    """Arguments for the file reader tool."""

    path: str = Field(min_length=1, max_length=1024)

    @field_validator("path")
    @classmethod
    def path_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be blank")
        return value


class FileReaderOutput(BaseModel):
    """Structured result of a successful read."""

    path: str
    content: str
    size_bytes: int = Field(ge=0)


class FileReaderTool(Tool):
    """Read a UTF-8 text file inside the workspace only."""

    name: ClassVar[str] = "file_reader"
    description: ClassVar[str] = (
        "Read a UTF-8 text file located inside the workspace. "
        "The path is relative to the workspace root; '..' and absolute "
        "paths are rejected."
    )
    input_schema: ClassVar[type[BaseModel]] = FileReaderInput
    output_schema: ClassVar[type[BaseModel]] = FileReaderOutput
    timeout_s: ClassVar[float] = 10.0
    max_retries: ClassVar[int] = 0
    retry_backoff_s: ClassVar[float] = 0.0

    def __init__(
        self, workspace: str | Path, *, max_bytes: int = _DEFAULT_MAX_BYTES
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
        self._workspace = Path(workspace).expanduser().resolve()
        self._max_bytes = max_bytes

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def execute(self, arguments: BaseModel) -> BaseModel:
        assert isinstance(arguments, FileReaderInput)
        target = resolve_workspace_path(self._workspace, arguments.path)

        if not target.exists():
            raise FileMissingError(
                f"file not found in workspace: {arguments.path!r}",
                tool_name=self.name,
            )
        if target.is_dir():
            raise ToolExecutionError(
                f"path is a directory, not a file: {arguments.path!r}",
                tool_name=self.name,
            )

        size = target.stat().st_size
        if size > self._max_bytes:
            raise FileTooLargeError(
                f"file is {size} bytes which exceeds the "
                f"{self._max_bytes}-byte limit",
                tool_name=self.name,
            )

        try:
            data = target.read_bytes()
        except OSError as exc:
            raise ToolExecutionError(
                f"could not read file: {exc.strerror or type(exc).__name__}",
                tool_name=self.name,
            ) from exc

        if len(data) > self._max_bytes:  # race guard
            raise FileTooLargeError(
                f"file grew beyond the {self._max_bytes}-byte limit",
                tool_name=self.name,
            )

        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                "file is not valid UTF-8 text",
                tool_name=self.name,
            ) from exc

        return FileReaderOutput(
            path=workspace_relative(self._workspace, target),
            content=content,
            size_bytes=len(data),
        )
