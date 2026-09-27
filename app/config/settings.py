"""Environment-driven configuration.

Values are read from environment variables with sensible defaults.
When the real environment is read (``load_settings()`` with no argument),
a project-root ``.env`` file is honored as documented in ``.env.example``:
the real environment always wins, ``.env`` only fills in what is missing.
No API key is required to run Phase 1/2 code or the test suite.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_REPLANS = 3
DEFAULT_LLM_TIMEOUT_S = 30.0
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_BACKOFF_S = 0.5
DEFAULT_TOOL_TIMEOUT_S = 10.0
DEFAULT_MAX_PLAN_TASKS = 7
DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_WEB_SEARCH_PROVIDER = "none"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings."""

    # LLM / OpenRouter
    openrouter_api_key: str | None = None
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    llm_backoff_s: float = DEFAULT_LLM_BACKOFF_S

    # Agent safety limits
    max_replans: int = DEFAULT_MAX_REPLANS
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    max_plan_tasks: int = DEFAULT_MAX_PLAN_TASKS

    # Workspace file tools
    workspace_dir: Path = Path("workspace")
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    # Web search (LangSearch provider; disabled cleanly when no key is set)
    web_search_provider: str = DEFAULT_WEB_SEARCH_PROVIDER
    web_search_api_key: str | None = None  # LANGSEARCH_API_KEY (or legacy alias)
    web_search_endpoint: str | None = None

    log_level: str = "INFO"


def _read_str(env: Mapping[str, str], key: str, default: str) -> str:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _read_optional_str(env: Mapping[str, str], key: str) -> str | None:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc


def _project_env_file() -> Path:
    """Project-root ``.env`` (the file README tells you to copy)."""
    return Path(__file__).resolve().parents[2] / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Minimal ``.env`` reader (stdlib only - no added dependency).

    Supports ``KEY=VALUE`` lines, an optional ``export`` prefix, surrounding
    quotes, ``#`` comments, and blank lines. Malformed lines are skipped;
    value validity is enforced later by the explicit typed readers.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load settings from ``env`` (defaults to the real environment).

    When ``env`` is omitted, the project-root ``.env`` file is merged in
    first and the real environment wins on conflicts - so configuration
    written to ``.env`` is actually honored. Passing an explicit mapping
    (as tests do) reads only that mapping and never touches ``.env``.

    Fails explicitly on malformed values instead of silently guessing.
    An API key is optional: it is never required to run tests.
    """
    if env is None:
        source: Mapping[str, str] = {
            **_read_env_file(_project_env_file()),
            **os.environ,
        }
    else:
        source = env

    settings = Settings(
        openrouter_api_key=_read_optional_str(source, "OPENROUTER_API_KEY"),
        openrouter_model=_read_str(source, "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        openrouter_base_url=_read_str(
            source, "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
        ),
        max_replans=_read_int(source, "MAX_REPLANS", DEFAULT_MAX_REPLANS),
        llm_timeout_s=_read_float(source, "LLM_TIMEOUT_S", DEFAULT_LLM_TIMEOUT_S),
        llm_temperature=_read_float(source, "LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE),
        llm_max_retries=_read_int(source, "LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES),
        llm_backoff_s=_read_float(source, "LLM_BACKOFF_S", DEFAULT_LLM_BACKOFF_S),
        tool_timeout_s=_read_float(source, "TOOL_TIMEOUT_S", DEFAULT_TOOL_TIMEOUT_S),
        max_plan_tasks=_read_int(source, "MAX_PLAN_TASKS", DEFAULT_MAX_PLAN_TASKS),
        workspace_dir=Path(_read_str(source, "ORBIT_WORKSPACE", "workspace")),
        max_file_bytes=_read_int(source, "MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES),
        web_search_provider=_read_str(
            source, "WEB_SEARCH_PROVIDER", DEFAULT_WEB_SEARCH_PROVIDER
        ).lower(),
        web_search_api_key=_read_optional_str(source, "LANGSEARCH_API_KEY")
        or _read_optional_str(source, "WEB_SEARCH_API_KEY"),
        web_search_endpoint=_read_optional_str(source, "WEB_SEARCH_ENDPOINT"),
        log_level=_read_str(source, "LOG_LEVEL", "INFO").upper(),
    )

    if settings.max_replans < 0:
        raise ValueError(f"MAX_REPLANS must be >= 0, got {settings.max_replans}")
    if settings.llm_timeout_s <= 0:
        raise ValueError(f"LLM_TIMEOUT_S must be > 0, got {settings.llm_timeout_s}")
    if settings.llm_max_retries < 0:
        raise ValueError(f"LLM_MAX_RETRIES must be >= 0, got {settings.llm_max_retries}")
    if settings.llm_backoff_s < 0:
        raise ValueError(f"LLM_BACKOFF_S must be >= 0, got {settings.llm_backoff_s}")
    if settings.tool_timeout_s <= 0:
        raise ValueError(f"TOOL_TIMEOUT_S must be > 0, got {settings.tool_timeout_s}")
    if settings.max_plan_tasks < 1:
        raise ValueError(
            f"MAX_PLAN_TASKS must be >= 1, got {settings.max_plan_tasks}"
        )
    if settings.max_file_bytes <= 0:
        raise ValueError(f"MAX_FILE_BYTES must be > 0, got {settings.max_file_bytes}")
    if not settings.openrouter_base_url.startswith(("http://", "https://")):
        raise ValueError(
            "OPENROUTER_BASE_URL must start with http:// or https://, "
            f"got {settings.openrouter_base_url!r}"
        )

    return settings
