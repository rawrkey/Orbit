"""Workspace path safety shared by the file reader and file writer.

Enforces, in order:
1. no null bytes
2. no absolute paths (drive letters, UNC, leading / or \\)
3. no ``..`` components (path traversal)
4. after resolution, the target must stay inside the workspace
   (this also catches symlinks that point outside)

Every violation raises ``PathTraversalError`` - the tools never touch the
filesystem for a rejected path.
"""

from __future__ import annotations

from pathlib import Path

from app.tools.base import PathTraversalError

__all__ = ["resolve_workspace_path", "workspace_relative"]


def resolve_workspace_path(workspace: Path, raw_path: str) -> Path:
    """Resolve ``raw_path`` strictly inside ``workspace`` or raise."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PathTraversalError(
            "path must be a non-empty string", tool_name=None
        )
    if "\x00" in raw_path:
        raise PathTraversalError("null bytes are not allowed in paths")

    workspace = workspace.resolve()
    candidate = Path(raw_path)

    if (
        candidate.is_absolute()
        or raw_path[0] in "/\\"
        or (len(raw_path) >= 2 and raw_path[1] == ":")
    ):
        raise PathTraversalError(
            f"absolute paths are not allowed: {raw_path!r}"
        )

    if any(part == ".." for part in candidate.parts):
        raise PathTraversalError(
            f"path traversal ('..') is not allowed: {raw_path!r}"
        )

    resolved = (workspace / candidate).resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise PathTraversalError(
            f"path resolves outside the workspace: {raw_path!r}"
        )
    return resolved


def workspace_relative(workspace: Path, resolved: Path) -> str:
    """POSIX-style path relative to the workspace root (for output)."""
    try:
        relative = resolved.relative_to(workspace.resolve())
    except ValueError as exc:  # pragma: no cover - guarded by resolve step
        raise PathTraversalError(
            f"path outside workspace: {resolved}"
        ) from exc
    return relative.as_posix() or "."
