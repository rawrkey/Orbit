"""Workspace-scoped file writer.

Writes UTF-8 text files strictly inside the configured workspace. Path
traversal, absolute paths, and symlink escapes are rejected before any
filesystem access. Parent directories are created only within the
workspace. Overwriting an existing file requires ``overwrite=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from app.tools.base import (
    DestinationExistsError,
    FileTooLargeError,
    Tool,
    ToolExecutionError,
)
from app.tools.path_safety import resolve_workspace_path, workspace_relative

__all__ = ["FileWriterInput", "FileWriterOutput", "FileWriterTool"]

_DEFAULT_MAX_BYTES = 1_000_000


class FileWriterInput(BaseModel):
    """Arguments for the file writer tool."""

    path: str = Field(min_length=1, max_length=1024)
    content: str
    overwrite: bool = False

    @field_validator("path")
    @classmethod
    def path_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be blank")
        return value


class FileWriterOutput(BaseModel):
    """Structured result of a successful write."""

    path: str
    bytes_written: int = Field(ge=0)
    overwritten: bool


class FileWriterTool(Tool):
    """Write a UTF-8 text file inside the workspace only."""

    name: ClassVar[str] = "file_writer"
    description: ClassVar[str] = (
        "Write a UTF-8 text file inside the workspace. The path is relative "
        "to the workspace root; '..' and absolute paths are rejected. Set "
        "overwrite=true to replace an existing file."
    )
    input_schema: ClassVar[type[BaseModel]] = FileWriterInput
    output_schema: ClassVar[type[BaseModel]] = FileWriterOutput
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
        assert isinstance(arguments, FileWriterInput)
        target = resolve_workspace_path(self._workspace, arguments.path)

        if target == self._workspace:
            raise ToolExecutionError(
                "cannot write to the workspace root directory itself",
                tool_name=self.name,
            )

        data = arguments.content.encode("utf-8")
        if len(data) > self._max_bytes:
            raise FileTooLargeError(
                f"content is {len(data)} bytes which exceeds the "
                f"{self._max_bytes}-byte limit",
                tool_name=self.name,
            )

        overwritten = False
        if target.exists():
            if target.is_dir():
                raise ToolExecutionError(
                    f"target is a directory: {arguments.path!r}",
                    tool_name=self.name,
                )
            if not arguments.overwrite:
                raise DestinationExistsError(
                    f"file already exists (set overwrite=true to replace): "
                    f"{arguments.path!r}",
                    tool_name=self.name,
                )
            overwritten = True

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not create parent directory: "
                f"{exc.strerror or type(exc).__name__}",
                tool_name=self.name,
            ) from exc

        try:
            target.write_bytes(data)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not write file: {exc.strerror or type(exc).__name__}",
                tool_name=self.name,
            ) from exc

        return FileWriterOutput(
            path=workspace_relative(self._workspace, target),
            bytes_written=len(data),
            overwritten=overwritten,
        )
