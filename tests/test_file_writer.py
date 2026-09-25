"""File writer tests: workspace containment, traversal, size limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.base import (
    DestinationExistsError,
    FileTooLargeError,
    MalformedToolInputError,
    PathTraversalError,
    ToolExecutionError,
)
from app.tools.file_writer import FileWriterOutput, FileWriterTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def writer(workspace: Path, max_bytes: int = 1_000_000) -> FileWriterTool:
    return FileWriterTool(workspace, max_bytes=max_bytes)


# ---------------------------------------------------------------------------
# Valid writes
# ---------------------------------------------------------------------------


def test_valid_write(workspace: Path) -> None:
    output = writer(workspace).run({"path": "out.txt", "content": "result: 42"})
    assert isinstance(output, FileWriterOutput)
    assert output.path == "out.txt"
    assert output.bytes_written == len("result: 42")
    assert output.overwritten is False
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "result: 42"


def test_nested_workspace_path_creates_parents(workspace: Path) -> None:
    output = writer(workspace).run(
        {"path": "plans/day1/note.md", "content": "fly to mumbai"}
    )
    assert output.path == "plans/day1/note.md"
    assert (workspace / "plans" / "day1" / "note.md").read_text(
        encoding="utf-8"
    ) == "fly to mumbai"


def test_overwrite_requires_explicit_permission(workspace: Path) -> None:
    target = workspace / "report.txt"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(DestinationExistsError):
        writer(workspace).run({"path": "report.txt", "content": "new"})

    output = writer(workspace).run(
        {"path": "report.txt", "content": "new", "overwrite": True}
    )
    assert output.overwritten is True
    assert target.read_text(encoding="utf-8") == "new"


def test_empty_content_writes_empty_file(workspace: Path) -> None:
    output = writer(workspace).run({"path": "empty.txt", "content": ""})
    assert output.bytes_written == 0
    assert (workspace / "empty.txt").read_text(encoding="utf-8") == ""


def test_unicode_content_round_trip(workspace: Path) -> None:
    writer(workspace).run({"path": "uni.txt", "content": "预算 8000 INR"})
    assert (workspace / "uni.txt").read_text(encoding="utf-8") == "预算 8000 INR"


# ---------------------------------------------------------------------------
# Path traversal / containment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["../evil.txt", "../../evil.txt", "sub/../../evil.txt", ".."],
)
def test_path_traversal_rejected_and_nothing_written(
    workspace: Path, tmp_path: Path, path: str
) -> None:
    with pytest.raises(PathTraversalError):
        writer(workspace).run({"path": path, "content": "x"})
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()


@pytest.mark.parametrize(
    "path",
    ["/tmp/evil.txt", "C:\\Windows\\evil.txt", "\\\\server\\share\\evil"],
)
def test_absolute_paths_rejected(workspace: Path, path: str) -> None:
    with pytest.raises(PathTraversalError):
        writer(workspace).run({"path": path, "content": "x"})


def test_writing_workspace_root_rejected(workspace: Path) -> None:
    with pytest.raises(ToolExecutionError):
        writer(workspace).run({"path": ".", "content": "x"})


def test_null_byte_rejected(workspace: Path) -> None:
    with pytest.raises(PathTraversalError, match="null"):
        writer(workspace).run({"path": "a\x00b", "content": "x"})


def test_write_to_directory_rejected(workspace: Path) -> None:
    (workspace / "adir").mkdir()
    with pytest.raises(ToolExecutionError, match="directory"):
        writer(workspace).run({"path": "adir", "content": "x"})


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------


def test_size_limit_enforced_and_file_not_created(workspace: Path) -> None:
    with pytest.raises(FileTooLargeError, match="exceeds"):
        writer(workspace, max_bytes=10).run(
            {"path": "big.txt", "content": "x" * 100}
        )
    assert not (workspace / "big.txt").exists()


def test_unicode_byte_size_counts_bytes(workspace: Path) -> None:
    # 4 Chinese chars = 12 UTF-8 bytes > 10-byte limit
    with pytest.raises(FileTooLargeError):
        writer(workspace, max_bytes=10).run(
            {"path": "cn.txt", "content": "预算预算"}
        )


def test_invalid_constructor_and_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FileWriterTool(tmp_path, max_bytes=-1)
    with pytest.raises(MalformedToolInputError):
        FileWriterTool(tmp_path).run({"path": "", "content": "x"})
