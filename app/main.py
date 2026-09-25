"""ORBIT entry point.

Phase 1: verifies configuration and imports. FastAPI and workflow endpoints
arrive in a later phase.
"""

from __future__ import annotations

from app.config.settings import load_settings


def main() -> None:
    settings = load_settings()
    print("ORBIT - Adaptive Personal Workflow Agent (Phase 1 skeleton)")
    print(f"  runtime model : {settings.openrouter_model}")
    print(f"  max replans   : {settings.max_replans}")
    print(f"  llm timeout   : {settings.llm_timeout_s}s")
    print(f"  api key set   : {settings.openrouter_api_key is not None}")


if __name__ == "__main__":
    main()
