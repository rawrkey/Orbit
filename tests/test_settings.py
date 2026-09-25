"""Configuration tests: defaults, environment parsing, explicit failures."""

from __future__ import annotations

import pytest

from app.config.settings import (
    DEFAULT_OPENROUTER_MODEL,
    Settings,
    load_settings,
)


def test_defaults_require_no_environment() -> None:
    settings = load_settings({})
    assert isinstance(settings, Settings)
    assert settings.openrouter_api_key is None  # no key needed
    assert settings.openrouter_model == DEFAULT_OPENROUTER_MODEL
    assert settings.max_replans == 3
    assert settings.llm_timeout_s == 30.0
    assert settings.llm_temperature == 0.2
    assert settings.tool_timeout_s == 10.0
    assert settings.log_level == "INFO"


def test_reads_values_from_environment() -> None:
    settings = load_settings(
        {
            "OPENROUTER_API_KEY": "sk-test-123",
            "OPENROUTER_MODEL": "nvidia/nemotron-3-ultra",
            "MAX_REPLANS": "5",
            "LLM_TIMEOUT_S": "12.5",
            "LLM_TEMPERATURE": "0.0",
            "TOOL_TIMEOUT_S": "3",
            "LOG_LEVEL": "debug",
        }
    )
    assert settings.openrouter_api_key == "sk-test-123"
    assert settings.openrouter_model == "nvidia/nemotron-3-ultra"
    assert settings.max_replans == 5
    assert settings.llm_timeout_s == 12.5
    assert settings.llm_temperature == 0.0
    assert settings.tool_timeout_s == 3.0
    assert settings.log_level == "DEBUG"


def test_blank_api_key_treated_as_missing() -> None:
    settings = load_settings({"OPENROUTER_API_KEY": "   "})
    assert settings.openrouter_api_key is None


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"MAX_REPLANS": "many"}, "MAX_REPLANS"),
        ({"LLM_TIMEOUT_S": "soon"}, "LLM_TIMEOUT_S"),
        ({"MAX_REPLANS": "-1"}, "MAX_REPLANS"),
        ({"LLM_TIMEOUT_S": "0"}, "LLM_TIMEOUT_S"),
        ({"TOOL_TIMEOUT_S": "-2"}, "TOOL_TIMEOUT_S"),
        ({"LLM_MAX_RETRIES": "-1"}, "LLM_MAX_RETRIES"),
        ({"LLM_BACKOFF_S": "-0.5"}, "LLM_BACKOFF_S"),
        ({"MAX_FILE_BYTES": "0"}, "MAX_FILE_BYTES"),
        ({"OPENROUTER_BASE_URL": "ftp://example.com"}, "OPENROUTER_BASE_URL"),
    ],
)
def test_invalid_values_fail_explicitly(env: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_settings(env)


# ---------------------------------------------------------------------------
# Phase 2 settings
# ---------------------------------------------------------------------------


def test_phase2_defaults() -> None:
    settings = load_settings({})
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_max_retries == 2
    assert settings.llm_backoff_s == 0.5
    assert str(settings.workspace_dir) == "workspace"
    assert settings.max_file_bytes == 1_000_000
    assert settings.web_search_provider == "none"
    assert settings.web_search_api_key is None
    assert settings.web_search_endpoint is None


def test_phase2_values_from_environment() -> None:
    settings = load_settings(
        {
            "OPENROUTER_BASE_URL": "https://proxy.example/v1",
            "LLM_MAX_RETRIES": "5",
            "LLM_BACKOFF_S": "0.25",
            "ORBIT_WORKSPACE": "my_ws",
            "MAX_FILE_BYTES": "2048",
            "WEB_SEARCH_PROVIDER": "Brave",
            "WEB_SEARCH_API_KEY": "brave-key",
            "WEB_SEARCH_ENDPOINT": "https://api.search.example",
        }
    )
    assert settings.openrouter_base_url == "https://proxy.example/v1"
    assert settings.llm_max_retries == 5
    assert settings.llm_backoff_s == 0.25
    assert str(settings.workspace_dir) == "my_ws"
    assert settings.max_file_bytes == 2048
    assert settings.web_search_provider == "brave"  # normalized to lowercase
    assert settings.web_search_api_key == "brave-key"
    assert settings.web_search_endpoint == "https://api.search.example"


# ---------------------------------------------------------------------------
# Phase 3 settings
# ---------------------------------------------------------------------------


def test_phase3_defaults() -> None:
    settings = load_settings({})
    assert settings.max_plan_tasks == 7


def test_phase3_max_plan_tasks_from_environment() -> None:
    settings = load_settings({"MAX_PLAN_TASKS": "4"})
    assert settings.max_plan_tasks == 4


def test_phase3_invalid_max_plan_tasks_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="MAX_PLAN_TASKS"):
        load_settings({"MAX_PLAN_TASKS": "0"})
    with pytest.raises(ValueError, match="MAX_PLAN_TASKS"):
        load_settings({"MAX_PLAN_TASKS": "many"})
