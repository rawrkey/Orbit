"""File reader tests: workspace containment, traversal, size limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.base import (
    FileMissingError,
    FileTooLargeError,
    MalformedToolInputError,
    PathTraversalError,
    ToolExecutionError,
)
from app.tools.file_reader import FileReaderOutput, FileReaderTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def reader(workspace: Path, max_bytes: int = 1_000_000) -> FileReaderTool:
    return FileReaderTool(workspace, max_bytes=max_bytes)


# ---------------------------------------------------------------------------
# Valid reads
# ---------------------------------------------------------------------------


def test_valid_read(workspace: Path) -> None:
    (workspace / "notes.txt").write_text("hello orbit", encoding="utf-8")
    output = reader(workspace).run({"path": "notes.txt"})
    assert isinstance(output, FileReaderOutput)
    assert output.content == "hello orbit"
    assert output.path == "notes.txt"
    assert output.size_bytes == len("hello orbit")


def test_nested_path_read(workspace: Path) -> None:
    nested = workspace / "docs" / "deep"
    nested.mkdir(parents=True)
    (nested / "plan.md").write_text("# plan", encoding="utf-8")
    output = reader(workspace).run({"path": "docs/deep/plan.md"})
    assert output.content == "# plan"
    assert output.path == "docs/deep/plan.md"


def test_missing_file_raises_typed_error(workspace: Path) -> None:
    with pytest.raises(FileMissingError):
        reader(workspace).run({"path": "nope.txt"})


def test_directory_path_rejected(workspace: Path) -> None:
    (workspace / "sub").mkdir()
    with pytest.raises(ToolExecutionError, match="directory"):
        reader(workspace).run({"path": "sub"})


def test_non_utf8_file_rejected(workspace: Path) -> None:
    (workspace / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ToolExecutionError, match="UTF-8"):
        reader(workspace).run({"path": "binary.bin"})


# ---------------------------------------------------------------------------
# Path traversal / containment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "../../etc/passwd",
        "docs/../../outside.txt",
        "..",
        "./../x",
    ],
)
def test_path_traversal_rejected(workspace: Path, path: str) -> None:
    with pytest.raises(PathTraversalError):
        reader(workspace).run({"path": path})


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "\\\\server\\share\\file",
        "/",
    ],
)
def test_absolute_paths_rejected(workspace: Path, path: str) -> None:
    with pytest.raises(PathTraversalError):
        reader(workspace).run({"path": path})


def test_outside_file_is_not_read(workspace: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    # Even a syntactically relative path that resolves outside is blocked.
    with pytest.raises(PathTraversalError):
        reader(workspace).run({"path": "../secret.txt"})
    assert secret.exists()  # untouched


def test_null_byte_rejected(workspace: Path) -> None:
    with pytest.raises(PathTraversalError, match="null"):
        reader(workspace).run({"path": "a\x00b"})


def test_symlink_escape_rejected(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("escaped", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(PathTraversalError):
        reader(workspace).run({"path": "link.txt"})


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------


def test_size_limit_enforced(workspace: Path) -> None:
    (workspace / "big.txt").write_text("x" * 100, encoding="utf-8")
    with pytest.raises(FileTooLargeError, match="exceeds"):
        reader(workspace, max_bytes=50).run({"path": "big.txt"})


def test_size_limit_at_boundary_passes(workspace: Path) -> None:
    (workspace / "exact.txt").write_text("12345", encoding="utf-8")
    output = reader(workspace, max_bytes=5).run({"path": "exact.txt"})
    assert output.size_bytes == 5


def test_invalid_constructor_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FileReaderTool(tmp_path, max_bytes=0)
    with pytest.raises(MalformedToolInputError):
        FileReaderTool(tmp_path).run({"path": "   "})
